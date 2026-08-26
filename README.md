# GLiNER2 Rust

A high-performance, pure Rust implementation of the [GLiNER2](https://github.com/urchade/GLiNER2) information extraction model. This library provides efficient CPU-based inference for entity extraction, text classification, structured data extraction, and relation extraction — with zero external dependencies beyond Cargo.

## 🎯 Current Status: Full Inference Surface Working, Parity Tuning Ongoing

**The full GLiNER2 inference pipeline is implemented and running end-to-end in Rust.** Key capabilities currently verified:
- ✅ Load real GLiNER2 model weights from HuggingFace Hub
- ✅ Run DeBERTa-based encoder path with GLiNER2-compatible routing
- ✅ count_embed layer with GRU + Transformer architecture
- ✅ HF tokenizer integration with correct subword tracking
- ✅ Schema/text indices tracking aligned with Python behavior
- ✅ Entity order preservation aligned with Python behavior
- ✅ Non-empty entity extraction with confidence + span outputs in integration tests
- ✅ Classification/relation/structure extraction paths implemented in runtime API
- ✅ Offline Python-vs-Rust parity harness running against local model snapshots
- ✅ Strict long-context `max_len` parity fixtures passing for entities/relations/structures

**Current focus:** continued numerical parity tuning and broader regression coverage across more texts/schemas/devices.

## 🚀 GLiNER2.5 Support (boundary architecture)

This fork adds the [GLiNER2.5](https://github.com/urchade/GLiNER2.5)
boundary-prediction pipeline, validated numerically against the Python
reference:

- ✅ All three checkpoints (`fastino/gliner2.5-{small,base,multi}-v1`)
- ✅ Boundary encoder + shared-pool scoring with numeric parity vs Python
  (Tim Cook 0.998 / Apple 0.951 on the parity fixture)
- ✅ Entities, classifications, records, and relation decoding
- ✅ Long-document auto-chunking (>384 words) with span remapping and merge
  policies (`src/chunking.rs`)

```rust
use gliner2_rs::inference::engine::GLiNER2;
use gliner2_rs::schema::types::Schema;

let engine = GLiNER2::from_pretrained("fastino/gliner2.5-small-v1")?;
let schema = Schema::from_dict(&serde_json::json!({
    "entities": ["person", "organization", "location"]
}))?;
// Long documents are chunked automatically.
let result = engine.extract(text, &schema, 0.5, true, true, None)?;
```

Performance (CPU, release): ~74ms per short call — at parity with the Python
implementation; batched throughput ~54ms/sample.

## ✨ What's Working

### Complete Architecture Port
- ✅ **Candle ML Framework** — All PyTorch/tch dependencies replaced with `candle-core`, `candle-nn`, and `candle-transformers`
- ✅ **DeBERTa Encoder Backend** — Candle-based DeBERTa path configured for GLiNER2-compatible behavior:
  - Relative-attention-enabled encoder configuration and compatible masking/bias routing
  - Relative position handling and attention flow aligned with GLiNER2 inference needs
  - DeBERTa-v3-style compatibility settings applied during model loading/inference
  - Ongoing parity validation against Python across broader scenarios
- ✅ **GLiNER2 Heads** — All components match the actual Python architecture:
  - **Span Rep**: markerV0 with project_start/end/out_project (Linear+ReLU+Linear, with ReLU on concatenated start/end reps before out_project)
  - **Classifier**: 2-layer MLP (768→1536→1) with ReLU
  - **Count Pred**: 2-layer MLP (768→1536→20) with ReLU
  - **Count Embed**: Full GRU + Transformer architecture (pos_embedding → GRU → in_projector → 2x Transformer → out_projector)
- ✅ **Weight Loading** — `VarBuilder::from_mmaped_safetensors()` with correct name mapping
- ✅ **HuggingFace Tokenizer** — Automatic Hub download with local fallback

### Real Inference Proven Working
- ✅ **Real Model Downloads** — Downloads from HuggingFace Hub automatically
- ✅ **Weight Loading** — Successfully loads all model components including count_embed
- ✅ **Full Pipeline Execution** — Tokenization → Collation → Encoding → count_embed + einsum scoring → Extraction
- ✅ **Batch Processing** — Parallel batch inference
- ✅ **Entity Extraction API** — Full API with confidence scores and span positions
- ✅ **Text Classification** — Single and multi-label classification
- ✅ **Relation & Structure Extraction** — Full API support
- ⚠️ **Numerical Parity Tuning** — Inference is working; ongoing work focuses on tightening intermediate-value parity with Python across broader scenarios

### Test Coverage
- ✅ **81 Unit Tests** — Included across model, batching, schema, and inference modules
- ✅ **18 Integration Tests** — Included across focused API coverage, real Hub inference, and Python parity fixture validation
- ✅ **Zero tch Dependencies** — Pure Rust, no PyTorch runtime required

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GLiNER2 Inference Pipeline               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐ │
│  │ Tokenizer   │───▶│   Encoder    │───▶│ GLiNER2 Heads  │ │
│  │ (tokenizers)│    │  (candle)    │    │   (candle)     │ │
│  └─────────────┘    └──────────────┘    └────────────────┘ │
│         │                   │                      │       │
│         │              Embeddings              Results      │
│         │                   │                      │       │
│         └───────────────────┼──────────────────────┘       │
│                             │                              │
│                    Float tensor data                       │
│                 (Vec<Vec<f32>> embeddings)                 │
└─────────────────────────────────────────────────────────────┘
```

### Component Architecture (Matches Python GLiNER2)

| Component | Architecture | Status |
|-----------|-------------|--------|
| **Encoder** | DeBERTa encoder backend with GLiNER2-compatible configuration and relative-attention routing | ✅ Complete |
| **Span Rep** | markerV0: project_start/end/out_project (768→3072→768) | ✅ Complete |
| **Classifier** | MLP: 768 → 1536 (ReLU) → 1 | ✅ Complete |
| **Count Pred** | MLP: 768 → 1536 (ReLU) → 20 | ✅ Complete |
| **Count Embed** | GRU + Transformer (pos_embed → GRU → 2x Transformer → projectors) | ✅ Complete |
| **Weight Loading** | VarBuilder with weight name mapping | ✅ Complete |
| **Entity Extraction** | count_embed + einsum scoring | ✅ Working (parity tuning ongoing) |

## ⚖️ Comparison: `gliner2-rs` vs `brainless/gliner2-candle`

Both projects target GLiNER2 with Candle, but they optimize for different priorities.

### At a glance

| Dimension | `gliner2-rs` (this repo) | `brainless/gliner2-candle` |
|---|---|---|
| Scope | Broader GLiNER2 pipeline (entities + broader schema/task plumbing) | Minimal, entity-focused implementation |
| Codebase size | Larger (~multi-module, production-oriented) | Very small (~1 KLOC, easy to audit quickly) |
| Preprocessing/collation | Richer collation and tokenizer/index tracking for parity work | Simpler, purpose-built preprocessing |
| Extensibility | Higher (more architecture and task surface area) | Lower (optimized for a narrow use case) |
| Maintenance burden | Higher complexity, more moving parts | Lower complexity, fewer moving parts |
| Best fit | Teams building a fuller GLiNER2 toolchain | Users who want fast, minimal entity extraction |

### Pros and cons

**Choose `gliner2-rs` if you want:**
- A more complete GLiNER2-style system and API surface
- Better long-term flexibility for schema/task expansion
- A foundation suitable for productization and deeper parity/debug work

**Choose `brainless/gliner2-candle` if you want:**
- The smallest possible implementation to read and modify quickly
- Lower operational complexity
- A focused entity extraction tool without broader pipeline overhead

### Practical guidance

- If your priority is **minimalism and speed of understanding**, start with `brainless/gliner2-candle`.
- If your priority is **capability, extensibility, and a fuller GLiNER2 stack**, use `gliner2-rs`.
- A pragmatic path is to prototype quickly with the minimal repo, then migrate to `gliner2-rs` when you need richer schema/task behavior and long-term maintainability.

## 📦 Installation

```toml
[dependencies]
gliner2-rs = { git = "https://github.com/mrorigo/gliner2-rust" }
```

### Dependencies
- `candle-core`, `candle-nn`, `candle-transformers` — HuggingFace's pure Rust ML framework
- `tokenizers` — HuggingFace tokenizer library
- `hf-hub` — HuggingFace Hub integration for automatic downloads
- `serde` / `serde_json` — JSON serialization
- `regex` — Regex validators for post-processing

## 🚀 Usage

### Basic Entity Extraction

```rust
use gliner2_rs::{GLiNER2, ExtractorConfig, SchemaBuilder};

// Create config with GLiNER2 model
let config = ExtractorConfig::builder()
    .model_name("fastino/gliner2-base-v1")
    .hidden_size(768)
    .vocab_size(128011)
    .num_hidden_layers(12)
    .num_attention_heads(12)
    .intermediate_size(3072)
    .build()?;

// Create engine (tokenizer downloads automatically from Hub)
let engine = GLiNER2::new(&config)?;

// Create schema
let schema = SchemaBuilder::new()
    .entities(vec!["person".to_string(), "organization".to_string()])
    .build()?;
let labels: Vec<&str> = schema.entities.iter().map(|e| e.name.as_str()).collect();

// Extract entities
let result = engine.extract_entities(
    "Apple CEO Tim Cook visited Cupertino.",
    &labels,
    Some(0.5),  // threshold
    true,       // include_confidence
    true,       // include_spans
    None,       // max_len
)?;

println!("{:#?}", result);
```

### Batch Processing

```rust
let texts = vec![
    "Apple CEO Tim Cook".to_string(),
    "Google founder Larry Page".to_string(),
];

let results = engine.batch_extract_entities(
    &texts,
    &["person", "organization"],
    2,      // batch_size
    None,   // threshold
    1,      // num_workers
    true,   // include_confidence
    true,   // include_spans
    None,   // max_len
)?;
```

## 🧪 Testing

### Run All Tests
```bash
cargo test --lib
```

### Run Integration Tests (Real Hub Downloads)
```bash
cargo test --test real_inference_test
```

### Run Focused Parity Fixture Test
```bash
cargo test --release --test python_parity_test
```

### Test Results
- ✅ The project currently includes **81 unit tests** and **18 integration tests**
- ✅ Integration coverage includes real HuggingFace Hub model/tokenizer downloads
- ✅ Integration coverage includes an offline Python-vs-Rust fixture parity harness
- ⏱️ Full integration runs can take ~60–120s due to model initialization and Hub/cache behavior

## 🔍 Parity Status and Remaining Work

### What is now verified
- ✅ End-to-end extraction is functioning in real integration tests
- ✅ Entity extraction returns meaningful outputs with confidence and character spans
- ✅ Collator/tokenizer routing (including subword position handling) is aligned for GLiNER2 usage
- ✅ count_embed and span scoring paths are wired and active in inference
- ✅ Classification, relation, and structure extraction paths are implemented and tested end-to-end
- ✅ Metadata-aware decoding is wired for relation thresholds and structure field metadata (dtype/threshold/validators)
- ✅ Offline Python parity fixture comparisons are in place for reproducible Rust-vs-Python output checks
- ✅ Parity harness coverage now includes threshold-edge fixtures (`0.49/0.50/0.51`) and multi-label classification fixtures
- ✅ Strict long-context truncation parity is validated via `max_len` fixtures for entities/relations/structures

### What remains
- ⚠️ Continue improving numerical parity against Python for intermediate tensors/layer outputs
- ⚠️ Expand parity fixtures to additional long-context and adversarial edge cases across more schema compositions
- ⚠️ Expand regression coverage across diverse schemas, longer texts, and additional model variants
- ⚠️ Validate behavior on additional hardware backends/devices as part of performance hardening

### Next steps
1. Add targeted parity snapshots for intermediate tensors in selected layers
2. Add more deterministic fixture pairs to the offline Python-vs-Rust parity harness (including multi-task mixed schemas)
3. Benchmark and tune CPU/GPU execution paths with parity checks enabled
4. Keep tightening confidence calibration consistency across edge cases

## 🏆 Credits

### Coding Work
This project was developed collaboratively across multiple AI coding sessions.

**Qwen 3.6 Plus Preview (free)** contributed major foundational implementation work, including:
- Core architecture port from PyTorch/tch to candle
- Initial DeBERTa/GLiNER2 component implementations
- Weight loading infrastructure and early end-to-end wiring
- Early debugging and project scaffolding/tests

**GPT-5.3-Codex** contributed substantial follow-up implementation and debugging work, including:
- Collator parity fixes for HuggingFace tokenizer IDs and subword index tracking
- DeBERTa encoder parity improvements (masking, relative position/bias handling, backend alignment)
- Count-aware scoring pipeline fixes in `count_embed` (GRU/math/shape/transformer behavior)
- Span representation parity fixes (activation path and projection behavior)
- Entity extraction/scoring parity fixes (count-slot handling, overlap suppression, calibration debugging)
- Classification/relation/structure inference path implementation and metadata-aware decoding
- Runtime `max_len` handling fixes and schema embedding gather corrections
- Runtime API parity updates for `max_len` in classification/relation convenience methods
- Strict long-context `max_len` fixture parity restoration for entities/relations/structures
- Focused `--release` integration coverage for relation/structure metadata behavior
- Offline Python parity harness (`debug_comparison/python_parity_reference.py`) and fixture-based Rust/Python comparisons
- Integration validation with real model tests and cleanup of debug output

### Guidance & Support
**The Operator** — Cheering on, guiding decisions, and providing crucial feedback throughout the development process. Your encouragement and direction made this achievement possible.

## 📄 License

Apache-2.0

## 🔗 Links

- [GLiNER2 Python Implementation](https://github.com/fastino-ai/GLiNER2)
- [Candle Documentation](https://github.com/huggingface/candle)
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)
