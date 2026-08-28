# GLiNER2 Rust Port - Phase 2: All-Candle Implementation Plan

> **STATUS: COMPLETE — HISTORICAL RECORD.** The tch→candle migration described
> in this document is finished. See the top-level `README.md` and `AGENTS.md`
> for current status; `PLAN_2.5.md` covers the GLiNER2.5 boundary pipeline.

## Overview

This document details the plan for migrating the entire GLiNER2 Rust port from `tch` (PyTorch bindings) to `candle` (HuggingFace's pure Rust ML framework). This eliminates the dual-backend complexity, removes the ~2GB libtorch runtime dependency, and creates a clean, maintainable, truly portable Rust library.

**Goal**: Replace all `tch`-based neural network code with `candle`, enabling real inference with a single, consistent tensor backend.

**Design Principle**: One tensor backend, zero data copying, pure Rust distribution.

## Current Status Assessment

### What's Already Working (but tch-dependent)
- ✅ Span representation layer (`span_rep.rs` - ~586 lines)
- ✅ Count prediction layer (`count_pred.rs` - ~670 lines)
- ✅ Classifier head (`classifier.rs` - ~562 lines)
- ✅ Batch collation and preprocessing
- ✅ Schema encoding and tokenization
- ✅ Inference pipeline and result formatting
- ✅ Weight loading infrastructure (safetensors parsing)
- ✅ 115 unit tests passing

### What's Missing
- ❌ Real BERT/DeBERTa encoder (currently returns random tensors)
- ❌ Actual weight application to model parameters
- ❌ Real inference (all outputs are meaningless due to random embeddings)

### Evidence from Codebase

In `src/model/extractor.rs`, the encoder methods return random tensors:
```rust
// Line 329: extract_token_embeddings()
let embeddings = Tensor::randn(
    &[word_count as i64, self.hidden_size as i64],
    (Kind::Float, self.device),
);

// Line 547-561: run_encoder()
pub fn run_encoder(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    // For now, return a placeholder tensor
    Ok(Tensor::randn(...))
}
```

In `src/model/loading.rs`, weight application is mostly debug logging with no actual parameter assignment.

## Why All-Candle Instead of Dual Backend?

| Factor | Dual Backend (tch + candle) | All Candle |
|--------|----------------------------|------------|
| **Dependencies** | libtorch (~2GB) + candle | Just candle |
| **Build complexity** | High (two tensor systems, linking issues) | Low (pure Rust, `cargo build`) |
| **Distribution** | Hard (libtorch runtime required) | Easy (single binary, no env vars) |
| **Data transfer** | Copy tensors between backends | Zero-copy, same memory |
| **Maintenance** | Two APIs, two error models | One consistent API |
| **Code to rewrite** | 0 lines (but permanent complexity) | ~2,500 lines (one-time cost) |
| **CI/CD** | Complex (libtorch setup) | Simple (standard Rust CI) |
| **User experience** | Set env vars, install PyTorch | `cargo add gliner2-candle` |

**Decision**: All-candle is the right choice. The ~2,500 lines to rewrite is a one-time investment that eliminates permanent complexity and enables a truly portable Rust library.

## Python Reference Architecture

From `GLiNER2/gliner2/model.py`:
```python
class Extractor(PreTrainedModel):
    def __init__(self, config, encoder_config=None, tokenizer=None):
        # Load encoder (BERT/DeBERTa via transformers)
        self.encoder = self._load_encoder(config.model_name, encoder_config)
        self.encoder.resize_token_embeddings(len(self.processor.tokenizer))
        self.hidden_size = self.encoder.config.hidden_size
        
        # GLiNER2-specific heads
        self.span_rep = SpanRepLayer(span_mode="markerV0", ...)
        self.classifier = create_mlp(input_dim=self.hidden_size, ...)
        self.count_pred = create_mlp(input_dim=self.hidden_size, ...)
        self.count_embed = CountLSTM(self.hidden_size)
```

The forward pass:
1. Encode batch through transformer: `outputs = self.encoder(input_ids, attention_mask)`
2. Extract embeddings: `token_embeddings = outputs.last_hidden_state`
3. Process through GLiNER2 heads: span_rep, classifier, count_pred

## Candle Capabilities Assessment

Candle has everything we need:

| Component | Candle Support | Notes |
|-----------|---------------|-------|
| **BERT/DeBERTa Encoder** | ✅ `candle_transformers::models::bert::BertModel` | Pre-built, tested |
| **Linear Layers** | ✅ `candle_nn::Linear` | MLP heads |
| **LayerNorm** | ✅ `candle_nn::LayerNorm` | Normalization |
| **GRU/LSTM** | ✅ `candle_nn::GRU` | Count embedding |
| **Tensor Ops** | ✅ gather, stack, matmul, etc. | All needed operations |
| **Safetensors** | ✅ Native loading | Direct from files |
| **Device Support** | ✅ CPU, CUDA | Hardware acceleration |
| **Autograd** | ✅ (not needed for inference) | Training support available |

## Implementation Phases

### Phase 1: Dependency Migration

**Tasks**:
- [ ] Remove `tch` dependency from `Cargo.toml`
- [ ] Add `candle-core` dependency
- [ ] Add `candle-transformers` dependency  
- [ ] Add `candle-nn` dependency
- [ ] Add `hf-hub` dependency (for model downloads)
- [ ] Remove `LIBTORCH_*` environment variable requirements
- [ ] Update `src/lib.rs` to remove tch re-exports
- [ ] Verify clean compilation with candle only

**Dependencies**:
```toml
[dependencies]
# Candle ML framework (HuggingFace)
candle-core = "0.8"
candle-transformers = "0.8" 
candle-nn = "0.8"

# HuggingFace Hub integration
hf-hub = { version = "0.4", features = ["tokio"] }

# Keep existing dependencies
tokenizers = { version = "0.20", features = ["onig"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
regex = "1.10"
safetensors = "0.4"
rayon = "1.10"
tracing = "0.1"
thiserror = "1.0"
```

**Files modified**: `Cargo.toml`, `src/lib.rs`

### Phase 2: Candle Encoder Module

**Tasks**:
- [ ] Create `src/model/candle_encoder.rs`
- [ ] Implement `CandleEncoder` struct wrapping candle's BERT model
- [ ] Implement token ID → embedding forward pass
- [ ] Handle attention mask properly
- [ ] Support both BERT and DeBERTa architectures
- [ ] Implement safetensors weight loading via candle
- [ ] Add device support (CPU, CUDA)
- [ ] Write unit tests for encoder forward pass

**Key struct**:
```rust
use candle_core::{Device, Tensor, DType};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use candle_nn::VarBuilder;

pub struct CandleEncoder {
    model: BertModel,
    device: Device,
    hidden_size: usize,
    is_loaded: bool,
}

impl CandleEncoder {
    pub fn new(config: &ExtractorConfig, device: Device) -> Result<Self> {
        let bert_config = Self::build_bert_config(config)?;
        let vb = VarBuilder::from_tensors(...); // Or from safetensors
        let model = BertModel::new(&bert_config, vb)?;
        
        Ok(Self { model, device, hidden_size: config.hidden_size, is_loaded: false })
    }
    
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let token_type_ids = Tensor::zeros_like(input_ids)?;
        let output = self.model.forward(input_ids, &token_type_ids, attention_mask)?;
        Ok(output)
    }
    
    pub fn load_weights(&mut self, path: &Path) -> Result<()> {
        let vb = VarBuilder::from_safetensors(..., &self.device)?;
        self.model = BertModel::new(&self.config, vb)?;
        self.is_loaded = true;
        Ok(())
    }
    
    pub fn hidden_size(&self) -> usize { self.hidden_size }
}
```

**Files created**: `src/model/candle_encoder.rs`

### Phase 3: Rewrite GLiNER2 Heads in Candle

**Tasks**:
- [ ] Rewrite `span_rep.rs` using candle tensors
- [ ] Rewrite `count_pred.rs` using candle tensors
- [ ] Rewrite `classifier.rs` using candle tensors
- [ ] Update all tensor operations to candle API
- [ ] Maintain same interfaces for Extractor integration
- [ ] Write unit tests for each head

**Key changes per file**:

`span_rep.rs`:
```rust
use candle_core::{Tensor, Device};
use candle_nn::{Embedding, VarBuilder};

pub struct SpanRepresentationLayer {
    max_width: usize,
    hidden_size: usize,
    width_embedding: Embedding,
    span_linear: candle_nn::Linear,
    layer_norm: candle_nn::LayerNorm,
    device: Device,
}

impl SpanRepresentationLayer {
    pub fn forward(&self, token_embs: &Tensor) -> Result<SpanRepOutput> {
        let seq_len = token_embs.dims()[0];
        let mut span_reps = Vec::new();
        
        for width in 0..self.max_width {
            let end_idx = (width + 1).min(seq_len);
            let start_embs = token_embs.narrow(0, 0, seq_len - width)?;
            let end_embs = token_embs.narrow(0, width, seq_len - width)?;
            
            // Combine start + end + width embedding
            let combined = Tensor::cat(&[&start_embs, &end_embs], 1)?;
            let span_rep = self.span_linear.forward(&combined)?;
            span_reps.push(span_rep);
        }
        
        let span_rep = Tensor::stack(&span_reps, 1)?;
        Ok(SpanRepOutput::new(span_rep, ...))
    }
}
```

`classifier.rs`:
```rust
use candle_nn::{Linear, LayerNorm, VarBuilder};

pub struct ClassifierHead {
    layers: Vec<Linear>,
    layer_norm: Option<LayerNorm>,
    activation: candle_nn::Activation,
}

impl ClassifierHead {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out)?;
            out = out.gelu()?; // Or other activation
        }
        Ok(out)
    }
}
```

`count_pred.rs`:
```rust
use candle_nn::{Linear, GRU, VarBuilder};

pub struct CountPredictionLayer {
    gru: GRU,
    linear: Linear,
    embedding: candle_nn::Embedding,
}

impl CountPredictionLayer {
    pub fn predict_count(&self, schema_emb: &Tensor) -> Result<CountOutput> {
        // GRU forward + linear projection
        let hidden = self.gru.step(schema_emb, &self.initial_state)?;
        let logits = self.linear.forward(&hidden)?;
        let count = logits.argmax(1)?;
        Ok(CountOutput { count, logits })
    }
}
```

**Files modified**: `src/model/span_rep.rs`, `src/model/count_pred.rs`, `src/model/classifier.rs`

### Phase 4: Extractor Integration

**Tasks**:
- [ ] Rewrite `Extractor` struct to use candle throughout
- [ ] Update `forward()` method to use candle encoder + heads
- [ ] Replace all tch tensor operations with candle equivalents
- [ ] Ensure embedding shapes match expectations
- [ ] Update device handling to candle's Device enum
- [ ] Update tests to use candle tensors

**Key struct**:
```rust
use candle_core::{Device, Tensor};

pub struct Extractor {
    pub config: ExtractorConfig,
    pub hidden_size: usize,
    pub max_width: usize,
    pub device: Device,
    pub is_training: bool,
    pub is_loaded: bool,
    
    // All candle-based components
    pub encoder: CandleEncoder,
    pub span_rep: SpanRepresentationLayer,
    pub count_pred: CountPredictionLayer,
    pub classifier: ClassifierHead,
}

impl Extractor {
    pub fn new(config: &ExtractorConfig) -> Result<Self> {
        config.validate()?;
        let device = Self::parse_device(&config.device)?;
        
        let encoder = CandleEncoder::new(config, device)?;
        let span_rep = SpanRepresentationLayer::from_config(config, device)?;
        let count_pred = CountPredictionLayer::from_config(config, device)?;
        let classifier = ClassifierHead::from_config(config, device)?;
        
        Ok(Self {
            config: config.clone(),
            hidden_size: config.hidden_size,
            max_width: config.max_width,
            device,
            is_training: false,
            is_loaded: false,
            encoder,
            span_rep,
            count_pred,
            classifier,
        })
    }
    
    pub fn forward(&self, batch: &PreprocessedBatch) -> Result<ExtractorOutput> {
        if batch.is_empty() {
            return Ok(ExtractorOutput::empty(self.device));
        }
        
        // Step 1: Run real encoder
        let input_ids = batch.input_ids.to_device(&self.device)?;
        let attention_mask = batch.attention_mask.to_device(&self.device)?;
        let encoder_output = self.encoder.forward(&input_ids, &attention_mask)?;
        
        // Step 2: Extract embeddings using indices
        let token_embeddings = self.extract_token_embeddings(&encoder_output, batch)?;
        let schema_embeddings = self.extract_schema_embeddings(&encoder_output, batch)?;
        
        // Step 3-5: Process through heads
        let span_representations = self.compute_span_representations(&token_embeddings, batch)?;
        let count_predictions = self.compute_count_predictions(&schema_embeddings)?;
        let classification_logits = self.compute_classification_logits(&schema_embeddings, batch)?;
        
        Ok(ExtractorOutput::new(
            token_embeddings,
            schema_embeddings,
            span_representations,
            classification_logits,
            count_predictions,
            self.device,
        ))
    }
}
```

**Files modified**: `src/model/extractor.rs`

### Phase 5: Weight Loading Integration

**Tasks**:
- [ ] Rewrite `ModelLoader` to use candle's VarBuilder
- [ ] Implement proper weight application via candle's safetensors loading
- [ ] Handle sharded model loading
- [ ] Support both local paths and HuggingFace Hub downloads
- [ ] Implement caching for downloaded models
- [ ] Test weight loading with real model files

**Key approach**:
```rust
use candle_nn::VarBuilder;
use safetensors::SafeTensors;

pub struct ModelLoader {
    config: ExtractorConfig,
    device: Device,
}

impl ModelLoader {
    pub fn load_safetensors(&self, path: &Path, model: &mut Extractor) -> Result<()> {
        // Load all weights via candle's VarBuilder
        let vb = VarBuilder::from_safetensors(vec![path], DType::F32, &self.device)?;
        
        // Rebuild model with loaded weights
        model.encoder = CandleEncoder::from_var_builder(&vb, &self.config)?;
        model.span_rep = SpanRepresentationLayer::from_var_builder(&vb, &self.config)?;
        model.count_pred = CountPredictionLayer::from_var_builder(&vb, &self.config)?;
        model.classifier = ClassifierHead::from_var_builder(&vb, &self.config)?;
        
        model.is_loaded = true;
        Ok(())
    }
}
```

**Files modified**: `src/model/loading.rs`

### Phase 6: Tokenizer & Collator Updates ✅ COMPLETE

**Tasks**:
- [x] Update collator to produce candle-compatible tensors
- [x] Ensure input_ids and attention_mask are candle tensors
- [x] Keep whitespace tokenizer for span boundaries
- [x] Add HuggingFace tokenizer for encoder input (now default)
- [x] Test token alignment with span boundaries
- [x] Make HuggingFace tokenizer the default in GLiNER2 engine
- [x] Add tokenizer loading helpers (load_hf_tokenizer, load_hf_tokenizer_from_path)

**Files modified**: `src/batch/collator.rs`, `src/inference/engine.rs`

**Key Changes**:
- Added `hf_tokenizer: Option<Arc<HfTokenizer>>` to `ExtractorCollator`
- Added `with_hf_tokenizer()` constructor for collator with real tokenizer
- Updated `token_to_id()` to use HF tokenizer when available
- Added `encode_text_with_subwords()` for proper subword tokenization
- GLiNER2 engine now loads HF tokenizer by default from config paths
- Graceful fallback to whitespace tokenizer if no HF tokenizer found

### Phase 7: Testing & Validation ✅ COMPLETE

**Tasks**:
- [x] Update all 106 existing tests to use candle tensors
- [x] Unit tests for candle encoder forward pass
- [ ] Integration tests comparing Rust vs Python output
- [ ] Numerical correctness tests (within 1e-5 tolerance)
- [ ] Batch processing tests
- [ ] Performance benchmarks
- [x] Edge case tests (empty input, long text, special characters)

**Test Results**:
- ✅ **106 unit tests passing** (all phases 1-7)
- ✅ All candle-based head tests (span_rep, classifier, count_pred)
- ✅ Extractor tests with real encoder integration
- ✅ Tokenizer and collator tests
- ✅ Config and error handling tests
- ⏳ Integration tests vs Python (pending model weights)
- ⏳ Performance benchmarks (pending real inference)

**Test strategy**:
1. Load same model weights in Python and Rust
2. Run same input through both
3. Compare embedding outputs (should match within tolerance)
4. Compare final entity extraction results
5. Verify batch processing produces identical results

## File Structure Changes

```
gliner2-candle/
├── Cargo.toml                      # ✅ Remove tch, add candle deps
├── PLAN.md                         # ✅ Original plan (Phase 1 complete)
├── PLAN2.md                        # ✅ This file (Phase 2 plan)
├── src/
│   ├── lib.rs                      # ✅ Update exports
│   ├── config.rs                   # ✅ Unchanged
│   ├── error.rs                    # ✅ Add Candle error variant
│   ├── tokenizer.rs                # ✅ Unchanged
│   ├── model/
│   │   ├── mod.rs                  # ✅ Update exports
│   │   ├── extractor.rs            # ✅ Rewrite for candle
│   │   ├── candle_encoder.rs       # 🆕 NEW: BERT encoder
│   │   ├── span_rep.rs             # ✅ Rewrite for candle
│   │   ├── count_pred.rs           # ✅ Rewrite for candle
│   │   ├── classifier.rs           # ✅ Rewrite for candle
│   │   └── loading.rs              # ✅ Rewrite for candle
│   ├── batch/
│   │   ├── mod.rs                  # ✅ Unchanged
│   │   ├── preprocessed.rs         # ✅ Update tensor types
│   │   └── collator.rs             # ✅ Update tensor creation
│   ├── schema/
│   │   ├── mod.rs                  # ✅ Unchanged
│   │   ├── builder.rs              # ✅ Unchanged
│   │   └── types.rs                # ✅ Unchanged
│   └── inference/
│       ├── mod.rs                  # ✅ Unchanged
│       └── engine.rs               # ✅ Unchanged (uses Extractor)
```

## Key Technical Decisions

### 1. Single Backend (Candle)
**Decision**: Use only `candle` for all neural network operations.
**Rationale**: 
- Eliminates libtorch dependency (~2GB)
- Pure Rust distribution
- Zero data copying between backends
- Consistent API throughout
- Easier builds and CI/CD

### 2. Weight Loading via VarBuilder
**Decision**: Use candle's `VarBuilder::from_safetensors()` for all weight loading.
**Rationale**:
- Native safetensors support
- No custom parsing needed
- Direct compatibility with HuggingFace format
- Handles sharded models automatically

### 3. Tokenizer Strategy
**Decision**: Keep whitespace tokenizer for spans, use HuggingFace tokenizer for encoder input.
**Rationale**:
- Whitespace tokenizer needed for word-level span boundaries
- HuggingFace tokenizer needed for proper token IDs
- Both serve different purposes

### 4. Device Management
**Decision**: Use candle's Device enum throughout.
**Rationale**:
- Single device abstraction
- No coordination between backends
- Clean API

### 5. Error Handling
**Decision**: Add `GlinerError::Candle(candle_core::Error)` variant.
**Rationale**:
- Preserves candle error context
- Consistent error handling
- Allows callers to handle candle-specific errors

## Implementation Order & Status

1. **Dependencies** ✅ (1-2 hours)
   - [x] Remove tch, add candle crates
   - [x] Verify compilation
   - [x] Update lib.rs

2. **Candle Encoder** ✅ (4-6 hours)
   - [x] Create `candle_encoder.rs`
   - [x] Implement BERT/DeBERTa wrapper
   - [x] Test forward pass

3. **Rewrite Heads** ✅ (6-8 hours)
   - [x] Rewrite `span_rep.rs` (~586 lines)
   - [x] Rewrite `count_pred.rs` (~670 lines)
   - [x] Rewrite `classifier.rs` (~562 lines)
   - [x] Test each component

4. **Extractor Integration** ✅ (3-4 hours)
   - [x] Rewrite `extractor.rs` (~858 lines)
   - [x] Update forward pass with real encoder
   - [x] Test end-to-end

5. **Weight Loading** ✅ (2-3 hours)
   - [x] Rewrite `loading.rs` (~660 lines)
   - [x] Use VarBuilder for safetensors loading
   - [x] Rebuild model components with loaded weights

6. **Tokenizer & Collator** ✅ (2-3 hours)
   - [x] Add HuggingFace tokenizer support
   - [x] Make HF tokenizer the default
   - [x] Update collator for proper token IDs
   - [x] Test token alignment with spans

7. **Testing & Validation** ✅ (4-6 hours)
   - [x] Update 106 existing tests to candle
   - [x] Fix dtype mismatches (f32 vs f64)
   - [x] Fix softmax dim for 1D tensors
   - [x] All 106 tests passing
   - ⏳ Integration tests vs Python (next step)
   - ⏳ Performance benchmarks (next step)

**Total actual time**: ~20-25 hours of focused development
**Status**: All phases 1-7 complete. Ready for integration testing with real model weights.

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Candle API differences | Medium | Medium | Study candle examples, test thoroughly |
| Weight name mismatches | Medium | Medium | Compare with Python, add validation |
| Performance regression | Low | Low | Candle is optimized, should be fast |
| Tokenizer alignment issues | High | Medium | Extensive testing with various inputs |
| Build complexity | Low | Low | Pure Rust, standard cargo build |

## Success Criteria

1. **Functional**: ✅ Can load a real GLiNER2 model and run meaningful inference
2. **Correct**: ✅ Output matches Python within numerical tolerance — full parity
   achieved (see README parity table; encoder diff 6.68e-6, pair logits 0.0000)
3. **Performant**: ✅ Inference speed at parity with Python CPU (~71ms/call)
4. **Maintainable**: ✅ Clean code, good tests, no placeholder code
5. **Portable**: ✅ No external dependencies beyond cargo, pure Rust binary (candle only)
6. **Tested**: ✅ 148 unit tests pass plus integration/checkpoint suites

## Next Steps After Completion

1. **Examples**: ✅ README shows GLiNER2 + GLiNER2.5 usage
2. **Documentation**: ✅ Updated (README, AGENTS.md, CHANGELOG.md)
3. **Performance**: ✅ Profiled and optimized (rel-bias, tensorized boundary scorer)
4. **CI/CD**: ⏳ Automated pipeline (not yet set up)
5. **Release**: ⏳ crates.io packaging (semver stays 0.x; additive)
6. **GPU Support**: ⏳ Verify CUDA acceleration with candle (not yet validated)

## References

- [Candle Documentation](https://github.com/huggingface/candle)
- [Candle Transformers Examples](https://github.com/huggingface/candle/tree/main/candle-examples/examples)
- [Candle BERT Implementation](https://github.com/huggingface/candle/tree/main/candle-transformers/src/models/bert)
- [GLiNER2 Python Implementation](./GLiNER2/gliner2/model.py)
- [Current Rust Code](./src/model/)
- [Original Implementation Plan](./PLAN.md)