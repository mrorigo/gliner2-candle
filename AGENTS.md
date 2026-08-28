# AGENTS.md - Guide for AI Agents Working on GLiNER2 Rust

## 🎯 Project Overview

This is a pure Rust implementation of the [GLiNER2](https://huggingface.co/collections/fastino/gliner2-family) and [GLiNER2.5](https://huggingface.co/collections/fastino/gliner25-models) information extraction models. The entire PyTorch/Python codebase has been ported to Rust using HuggingFace's `candle` ML framework.

**Key Achievement**: Both GLiNER2 (span-enumeration) and GLiNER2.5 (boundary-prediction) pipelines work end-to-end with real model weights downloaded from HuggingFace Hub.

**Current Status (Phases 1-5 + Phase D complete)**:
- GLiNER2: Fully functional entity extraction.
- GLiNER2.5: Full boundary pipeline with numeric parity vs the Python
  reference. All four task types match Python outputs:
  - Entities: full matrix parity (global=0.0000, relevant=0.0000)
  - Classifications: exact output match (positive)
  - Relations: exact format match (bare pairs, no flags)
  - Attributes (single-label): softmax logits matched to 8 decimals (0.9966161847)
  - Attributes (multi-label): sigmoid logits matched to 7 decimals (0.5735875)
- Phase D task parity fixes: endpoint difference vector layout (concat d/|d| not interleaved),
  relation scorer receives H-dim word states, content pooler weight ownership, classification [C] markers.
- Long documents (>384 words) are auto-chunked in `batch_extract` and merged.
- Performance (release, CPU): short inputs at Python parity (~71ms/call).
  Encoder rel-bias optimized: ~15ms → ~7ms/layer. Boundary scorer tensorized:
  score_sample ~72ms → ~27ms/chunk. 2400-word doc: ~3.1s end-to-end.

## 🏗️ Architecture Summary

### Pipeline Flow
```
GLiNER2:   Text + Schema → Tokenizer → Collator → DeBERTa V3 → Span Rep → Classifier → Output
GLiNER2.5: Text + Schema → Tokenizer → Collator → DeBERTa V3 (custom deberta_v3.rs) → gather word/marker states → BoundaryEncoder → shared-pool scoring → decode
           (>384 words: split into overlapping chunks, extract per chunk, merge spans)
```

### Key Components
| Component | File | Purpose |
|-----------|------|---------|
| **DeBERTa V3 Encoder** | `src/model/deberta_v3.rs` | Custom DeBERTa V3 implementation (no token_type_embeddings) |
| **Boundary Encoder** | `src/model/boundary.rs` | Boundary projection, attention, SwiGLU refinement + score_sample |
| **Span Representation** | `src/model/span_rep.rs` | markerV0: project_start/end/out_project (Linear+GELU+Linear) |
| **Classifier** | `src/model/classifier.rs` | 2-layer MLP: 768→1536→1 with ReLU |
| **Count Prediction** | `src/model/count_pred.rs` | 2-layer MLP: 768→1536→20 with ReLU |
| **Candle Encoder** | `src/model/candle_encoder.rs` | Wrapper supporting BERT/DeBERTa V2/V3 |
| **Collator** | `src/batch/collator.rs` | Tokenization + schema encoding + batching |
| **Inference Engine** | `src/inference/engine.rs` | Main GLiNER2/2.5 API + entity extraction logic |
| **Boundary Decode** | `src/inference/boundary.rs` | GLiNER2.5 boundary-path query building + entity decoding |

### Model Architecture (GLiNER2 base-v1)
- **Encoder**: DeBERTa-v3-base (128011 vocab, 768 hidden, 12 layers, 12 heads)
- **Attention**: Standard multi-head (query_proj/key_proj/value_proj), NOT disentangled
- **Relative Position Embeddings**: `encoder.encoder.rel_embeddings.weight` (512, 768)
- **Span Rep**: markerV0 with Linear+GELU+Linear projectors (no LayerNorm)
- **Classifier**: `create_mlp(768, [1536], 1, activation='relu')`
- **Count Pred**: `create_mlp(768, [1536], 20, activation='relu')`

## 🔧 Non-Obvious Technical Details

### 1. DeBERTa V3 vs V2 Differences
- **V3 has NO token_type_embeddings** - Only word_embeddings + LayerNorm
- **V3 uses standard multi-head attention** - query_proj/key_proj/value_proj (not disentangled)
- **V3 has relative position embeddings** - `encoder.encoder.rel_embeddings.weight` (512, 768)
- **Weight naming**: `attention.self.query_proj` not `attention.self.query`

### 2. Weight Name Mapping
GLiNER2 safetensors uses these exact paths:
```
encoder.embeddings.word_embeddings.weight: [128011, 768]
encoder.embeddings.LayerNorm.weight/bias: [768]
encoder.encoder.rel_embeddings.weight: [512, 768]
encoder.encoder.layer.X.attention.self.query_proj.weight: [768, 768]
encoder.encoder.layer.X.attention.self.key_proj.weight: [768, 768]
encoder.encoder.layer.X.attention.self.value_proj.weight: [768, 768]
encoder.encoder.layer.X.attention.output.dense.weight: [768, 768]
encoder.encoder.layer.X.attention.output.LayerNorm.weight/bias: [768]
encoder.encoder.layer.X.intermediate.dense.weight: [3072, 768]
encoder.encoder.layer.X.output.dense.weight: [768, 3072]
encoder.encoder.layer.X.output.LayerNorm.weight/bias: [768]
encoder.encoder.LayerNorm.weight/bias: [768]
span_rep.span_rep_layer.project_start.0.weight: [3072, 768]
span_rep.span_rep_layer.project_start.3.weight: [768, 3072]
span_rep.span_rep_layer.project_end.0.weight: [3072, 768]
span_rep.span_rep_layer.project_end.3.weight: [768, 3072]
span_rep.span_rep_layer.out_project.0.weight: [3072, 1536]
span_rep.span_rep_layer.out_project.3.weight: [768, 3072]
classifier.0.weight: [1536, 768]
classifier.2.weight: [1, 1536]
count_pred.0.weight: [1536, 768]
count_pred.2.weight: [20, 1536]
```

### 3. Schema Token Format
Entity schema tokens are structured as:
```
["(", "[P]", "entities", "(", "[E]", "person", "[E]", "organization", "[E]", "location", ")", ")"]
```
- `[P]` = Prompt token
- `[E]` = Entity type marker
- `(` and `)` = Structural tokens

### 4. HuggingFace Tokenizer Integration
- The collator uses `tokenizers` crate for proper subword tokenization
- **Critical**: `schema_special_indices` must track SUBWORD positions, not schema token positions
- **Critical**: `text_word_indices` must map whitespace tokens to their first subword position
- The HF tokenizer splits tokens like "Cupertino" → ["Cupertino"] or "Apple" → ["▁Apple"]

### 5. Attention Mask Broadcasting
- Input mask shape: `(batch, seq_len)` with 1s for valid, 0s for padding
- Must be expanded to `(batch, num_heads, seq_len, seq_len)` for multi-head attention
- Inverted: 0 for valid positions, `-inf` for padding
- Applied to attention scores before softmax

### 6. Span Representation Computation
For each token position `i` and span width `w`:
```
start_rep = project_start(token_embs[i])
end_rep = project_end(token_embs[i + w])
span_rep[i, w] = out_project(concat(start_rep, end_rep))
```
All projectors are Linear+GELU+Linear (no LayerNorm).

### 7. Entity Extraction Logic
1. Get span representations: `(seq_len, max_width, hidden_size)`
2. Get schema embeddings for each entity type from encoder output
3. Compute dot product between each span rep and schema embedding
4. Apply sigmoid to get probability
5. Filter by threshold (default 0.5)
6. Extract text spans using character position mappings

## 🐛 Debugging Notes (resolved — keep for reference)

### Fixed: Wrong/merged entity extraction (GLiNER2.5)
Root causes, in the order they were found and fixed:
1. **Relative-position sign flip**: candle-transformers' `debertav2` computes
   `rel_pos = k - q`; HF uses `q - k`. This corrupts c2p/p2c attention gathers.
   Fix: GLiNER2.5 uses the custom `src/model/deberta_v3.rs`.
2. **Missing rel-embedding LayerNorm**: HF applies `norm_rel_ebd=layer_norm`
   (`encoder.LayerNorm`) to rel_embeddings before attention. Do not also apply
   it as an output LayerNorm — DeBERTa has no output LN.
3. **Boundary scorer omissions** (`src/model/boundary.rs`): candidate_norm
   must be applied to pool candidates; content LayerNorm applies to the pooled
   span mean (not per-token); FiLM GELU is exact erf, not tanh.

Parity fixtures live in `/tmp/g25diag/*.json` (regenerate via Python dumps);
tests: `test_encoder_parity`, `test_staged_parity`, `test_numeric_parity`
(ignored; require the fixture files).

### Perf pitfalls
- Never benchmark unoptimized builds: candle dispatch overhead in debug is
  ~30-100x. `[profile.dev.package."*"] opt-level = 3` handles this for tests.
- Profile with `GLINER2_PROFILE=1` (stage timings: encoder / boundary head /
  per-layer attn+ffn / rel-bias / softmax).

### Fixed: two distinct span-content poolers (critical parity bug)
The checkpoint has BOTH `boundary_head.shared_pool_scorer.content_pooler.*`
AND `boundary_head.pair_scorer.content_pooler.*` (same shapes, DIFFERENT
weights). The shared-pool scorer must use its own; borrowing the pair
scorer's inflated score errors up to 8 logits on non-top candidates while
top-1 still looked fine — easy to miss if you only compare top spans.
Always validate the FULL candidate matrix vs Python (`pooled_indices.json`
+ `pair_logits.json` fixtures), not just the top hits. Residual agreement
after the fix: decision-relevant logits within ~0.25 of Python; extraction
outputs identical.

### Fixed: endpoint-difference vector layout (Phase D critical bug)
`torch.cat((d, |d|), dim=-1)` concatenates two halves: first all d values,
then all abs values. The Rust code used `.flat_map(|k| [d[k], |d[k]|])`
which interleaves `(d,|d|)` per-dim — identical first few values but
wrong overall, causing a constant logit shift (~0.27–0.38) on explicit
scoring calls. Fixed with: push all diffs first, then all abs diffs.

### Span attributes do NOT need dedicated weights (Phase 3e unblocked)
Verified against all three checkpoints (334 tensors each, zero attribute
keys) AND the Python runtime:
- `Schema.entity_attributes()` registers attribute labels as HIDDEN entity
  queries in the prompt (excluded from public entity order)
- After decoding, retained spans are re-scored against those queries via
  `score_explicit_spans(text_states, text_mask, query_states, query_mask,
  indices[B,Q,C,2])` — bypasses proposal top-k but reuses compat prior +
  pair reranker (`models/boundary/model.py`)
- `_attach_entity_attributes` then applies per-group sigmoid (multi_label)
  or softmax and attaches results to each span dict
- Rust port needs: schema AttributeGroup + hidden queries, an
  explicit-spans scoring path in boundary.rs, post-decode attachment

### Checkpoint facts (verified from Hub configs)
- All three 2.5 checkpoints declare `max_len: 4096`;
  `gliner2.5-multi-v1` vocab is 250112 (others 128011).
- Encoder `max_position_embeddings: 512` is NOT an input cap — DeBERTa uses
  relative positions (`position_buckets: 256` → rel table 512 rows).

## 🧪 Testing

### Run All Tests
```bash
cargo test --lib
```

### Run Integration Tests (Real Hub Downloads)
```bash
cargo test --test real_inference_test
```

### Run Single Test with Debug Output
```bash
cargo test --test real_inference_test test_real_gliner2_model_loading -- --nocapture
```

### Test Files
- `tests/real_inference_test.rs` - Integration tests with real model downloads
- `src/model/extractor.rs` - Unit tests for extractor
- `src/model/span_rep.rs` - Unit tests for span representation
- `src/model/classifier.rs` - Unit tests for classifier
- `src/model/count_pred.rs` - Unit tests for count prediction

## 📦 Dependencies

### Core ML
- `candle-core` - Tensor operations
- `candle-nn` - Neural network layers
- `candle-transformers` - Pre-built models (BERT, DeBERTa V2)

### Tokenization
- `tokenizers` - HuggingFace tokenizer library

### Hub Integration
- `hf-hub` - HuggingFace Hub downloads

### Other
- `serde` / `serde_json` - JSON serialization
- `regex` - Regex validators
- `tracing` - Logging

## 🚀 Build Commands

### Check Compilation
```bash
cargo check --lib
```

### Run Tests
```bash
cargo test --lib
cargo test --test real_inference_test
```

### Build Release
```bash
cargo build --release
```

## 📁 Key File Locations

```
src/
├── model/
│   ├── deberta_v3.rs      # Custom DeBERTa V3 encoder
│   ├── candle_encoder.rs  # Encoder wrapper (BERT/DeBERTa V2/V3)
│   ├── span_rep.rs        # Span representation layer
│   ├── classifier.rs      # Classification head
│   ├── count_pred.rs      # Count prediction layer
│   ├── extractor.rs       # Main Extractor model
│   └── loading.rs         # Weight loading
├── batch/
│   └── collator.rs        # Tokenization + batching
├── inference/
│   └── engine.rs          # GLiNER2 API + entity extraction
├── schema/
│   ├── builder.rs         # Schema builders
│   └── types.rs           # Schema types
└── tokenizer.rs           # Whitespace tokenizer

tests/
└── real_inference_test.rs # Integration tests
```

## 💡 Tips for Agents

1. **Always run tests after changes** - The test suite catches regressions
2. **Use debug output** - `eprintln!` statements are already in place for debugging
3. **Check weight shapes** - Mismatches usually indicate wrong architecture
4. **Verify token positions** - Schema/text indices must match tokenized output
5. **Compare with Python** - The Python implementation in `GLiNER2/` is the reference
6. **Model downloads are cached** - First run downloads ~440MB, subsequent runs use cache
7. **Tests are slow** - ~60-120s each due to model initialization
8. **No PyTorch runtime** - Everything is pure Rust via candle

## 🔗 References

- [GLiNER2 Python Implementation](./GLiNER2/) - Reference implementation
- [GLiNER2.5 Python Implementation](https://github.com/urchade/GLiNER2.5) - Reference implementation
- [README.md](./README.md) - Current status, usage, parity table
- [CHANGELOG.md](./CHANGELOG.md) - Full development history
- [docs/PLAN.md](./docs/PLAN.md) - Historical Phase 1 plan (tch era, maintenance mode)
- [docs/PLAN2.md](./docs/PLAN2.md) - Phase 2 (candle migration) plan
- [docs/PLAN_2.5.md](./docs/PLAN_2.5.md) - GLiNER2.5 boundary pipeline plan (COMPLETE)
- [docs/index.html](./docs/index.html) - HTML status/parity overview page
- [Candle Documentation](https://github.com/huggingface/candle)
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)
