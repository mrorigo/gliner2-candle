# GLiNER2 Rust

A high-performance, pure Rust implementation of [GLiNER2](https://github.com/urchade/GLiNER2)
(span-enumeration) and [GLiNER2.5](https://github.com/urchade/GLiNER2.5)
(boundary-prediction) information extraction models, built on HuggingFace's
[`candle`](https://github.com/huggingface/candle) ML framework — no PyTorch
runtime required. Designed for efficient CPU inference with real model weights
downloaded from the HuggingFace Hub.

> **The `-candle` suffix** sets this crate apart from the existing
> `gliner2` / `gliner2-rs` crates on [crates.io](https://crates.io), which are
> built on the **ONNX Runtime** backend. `gliner2-candle` is the pure-Rust,
> candle-native implementation — no ONNX Runtime, no libtorch, just `cargo
> build`.

## 🎯 Current Status: Full Numeric Parity vs Python

Both architectures run **end-to-end** with a documented numerical parity
report against the Python reference across all four task types:

| Task type | Parity vs Python |
|-----------|------------------|
| **Entities** (2.5) | Full matrix parity — `global = 0.0000`, `relevant = 0.0000` |
| **Classifications** | Exact output match (`positive`) |
| **Relations** | Exact format match (bare pairs, no flags) |
| **Attributes (single-label)** | Softmax logits matched to 8 decimals (`0.9966161847`) |
| **Attributes (multi-label)** | Sigmoid logits matched to 7 decimals (`0.5735875`) |

The GLiNER2 (span-enumeration) pipeline also produces entity/classification/
relation/structure outputs that match the Python reference.

## 🚀 GLiNER2.5 Support (boundary architecture)

The boundary-prediction pipeline is validated numerically against the Python
reference:

- ✅ All three checkpoints load and validate: `fastino/gliner2.5-{small,base,multi}-v1`
- ✅ Custom DeBERTa V3 encoder with exact-erf GELU and relative-position
  embeddings (sign-corrected vs candle-transformers' `debertav2`)
- ✅ Boundary encoder + shared-pool / pair scoring with full-matrix parity
- ✅ Entities, classifications, relations, records, and span attributes
- ✅ Span attributes (**Phase 3e**): no dedicated weights — attribute labels
  become hidden entity queries re-scored via an explicit-spans path
- ✅ Long-document auto-chunking (>384 words) with span remapping and merge
  policies (`src/chunking.rs`)

```rust
use gliner2_candle::GLiNER2;
use gliner2_candle::schema::types::Schema;

fn main() -> gliner2_candle::Result<()> {
    let engine = GLiNER2::from_pretrained("fastino/gliner2.5-small-v1")?;

    // Entities + span attributes (hidden queries, no dedicated weights).
    let mut groups = std::collections::HashMap::new();
    groups.insert(
        "sentiment".to_string(),
        gliner2_candle::schema::types::AttributeGroup {
            labels: vec!["positive".to_string(), "negative".to_string()],
            multi_label: true,
            threshold: 0.5,
            applies_to: Some(vec!["person".to_string()]),
            qualify_labels: false,
        },
    );
    let schema = Schema::new()
        .entities(vec![gliner2_candle::schema::types::EntityDef::new("person")])
        .entity_attributes(groups)?;

    // Long documents are chunked automatically.
    let result = engine.extract(
        "Apple CEO Tim Cook announced great results in Cupertino.",
        &schema,
        0.5,
        true,
        true,
        None,
    )?;
    println!("{result:#}");
    Ok(())
}
```

## ✨ Feature Surface

### GLiNER2 (span-enumeration)
- ✅ Entity extraction with confidence + character spans
- ✅ Single / multi-label text classification
- ✅ Structured (JSON) data extraction
- ✅ Relation extraction between entities
- ✅ `count_embed` (GRU + Transformer) layer

### GLiNER2.5 (boundary-prediction)
- ✅ Boundary encoder + shared-pool / pair reranking scorers
- ✅ Constrained classification (`src/constraints.rs`, Kleene-3 logic)
- ✅ Joint IE (relations) over the candidate pool
- ✅ Span attributes (single- and multi-label groups)
- ✅ Records / JSON structures

### Both
- ✅ Auto-chunking for long documents with deterministic merge policies
- ✅ Batch inference (`batch_size`, parallel preprocessing)
- ✅ Real model weights via `hf-hub`
- ✅ Pure Rust / candle — no PyTorch, no `tch`

## 🏗️ Architecture

```
GLiNER2:   Text + Schema → Collator → DeBERTa V3 → Span Rep → Classifier → Output
GLiNER2.5: Text + Schema → Collator → DeBERTa V3 → boundary encoder → shared-pool/pair scoring → decode
           (>384 words: split into overlapping chunks, extract per chunk, merge spans)
```

| Component | File | Purpose |
|-----------|------|---------|
| **DeBERTa V3 Encoder** | `src/model/deberta_v3.rs` | Custom DeBERTa V3 (no token_type_embeddings, exact-erf GELU, rel-bias) |
| **Boundary Encoder** | `src/model/boundary.rs` | Boundary projection, attention, SwiGLU refinement, scorers |
| **Candle Encoder Wrapper** | `src/model/candle_encoder.rs` | BERT / DeBERTa V2 / V3 routing |
| **Span Representation** | `src/model/span_rep.rs` | markerV0 (GLiNER2 path) |
| **Classifier / Count Pred** | `src/model/classifier.rs`, `count_pred.rs` |
| **Collator** | `src/batch/collator.rs` | Tokenization + schema encoding + batching |
| **Inference Engine** | `src/inference/engine.rs` | Main API + extraction |
| **Boundary Decode** | `src/inference/boundary.rs` | Query building, attribute attachment, span resolution |
| **Chunking** | `src/chunking.rs` | Long-document split/merge policies |

## 📦 Installation

```toml
[dependencies]
gliner2-candle = { git = "https://github.com/mrorigo/gliner2-candle" }
```

### Dependencies
- `candle-core`, `candle-nn`, `candle-transformers` — HuggingFace's pure Rust ML framework
- `tokenizers` — HuggingFace tokenizer library
- `hf-hub` — HuggingFace Hub downloads
- `serde` / `serde_json` — JSON serialization
- `regex` — Regex validators

## 🚀 Usage

### Basic GLiNER2 Entity Extraction

```rust
use gliner2_candle::{GLiNER2, ExtractorConfig, SchemaBuilder};

fn main() -> gliner2_candle::Result<()> {
    let config = ExtractorConfig::builder()
        .model_name("fastino/gliner2-base-v1")
        .hidden_size(768)
        .vocab_size(128011)
        .num_hidden_layers(12)
        .num_attention_heads(12)
        .intermediate_size(3072)
        .build()?;

    // Tokenizer + weights download automatically from the Hub.
    let engine = GLiNER2::new(&config)?;

    let schema = SchemaBuilder::new()
        .entities(vec!["person".to_string(), "organization".to_string()])
        .build()?;

    let result = engine.extract(
        "Apple CEO Tim Cook visited Cupertino.",
        &schema,
        0.5,   // threshold
        true,  // include_confidence
        true,  // include_spans
        None,  // max_len
    )?;
    println!("{result:#}");
    Ok(())
}
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

## ⚖️ vs `brainless/gliner2-candle`

Both target GLiNER2 with candle. `gliner2-candle` is a fuller, production-oriented
stack covering entities **and** broader schema/task plumbing (classifications,
structures, relations, GLiNER2.5 boundary attributes), at the cost of more
complexity. `brainless/gliner2-candle` is a minimal, entity-focused
implementation (~1 KLOC) that is quick to audit and modify. Choose the former
for a complete task surface and long-term extensibility; the latter for the
smallest possible entity-extraction footprint.

## 🧪 Testing

```bash
cargo test --lib                      # 148 unit tests
cargo test --test real_inference_test # GLiNER2 real-hub inference
cargo test --release --test real_inference_test_25 -- --ignored # GLiNER2.5 checkpoints
cargo test --release --test gliner25_boundary_test full_matrix_parity -- --ignored
```

The `test_task_output_parity` test (in `real_inference_test_25.rs`) runs
classifications, relations, and attributes through both pipelines and asserts
the numeric matches documented above.

## 🔍 Parity & Remaining Work

### Validated
- ✅ Encoder final diff `6.68e-6` vs Python; pair logits global / relevant = `0.0000`
- ✅ All four task types match Python (see table at top)
- ✅ Full-matrix candidate validation (not just top-1 spans)

### Remaining / non-goals
- 📦 Release / CI packaging and crates.io publish (semver stays 0.x; additive)
- 🧹 Broader cross-device (GPU) performance hardening
- Pre-1.0 API stabilization

## 📄 License

Apache-2.0

## 🔗 Links

- [GLiNER2 Python Implementation](https://github.com/fastino-ai/GLiNER2)
- [GLiNER2.5 Python Implementation](https://github.com/urchade/GLiNER2.5)
- [Candle Documentation](https://github.com/huggingface/candle)
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)
