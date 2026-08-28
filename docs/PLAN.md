# GLiNER2 Rust Port - Implementation Plan & Status

> **STATUS: SUPERSEDED — HISTORICAL RECORD.** This was the original Phase 1
> plan against the **tch/PyTorch** backend. The project was subsequently
> migrated to **candle** (see [PLAN2.md](./PLAN2.md)) and extended with the
> GLiNER2.5 boundary pipeline (see [PLAN_2.5.md](./PLAN_2.5.md)). The GLiNER2
> track is now in **maintenance mode**; new feature work targets GLiNER2.5.
> The current canonical status lives in the top-level `README.md` and `AGENTS.md`.

## Overview

This document outlines the plan and current status for porting GLiNER2 inference from Python to Rust. The goal is to enable GLiNER2 inference in Rust while maintaining compatibility with the Python model's behavior and output format. Training, LoRA, and other non-inference features are out of scope.

**Historical Status (tch era)**: ~9,500 lines of foundational code with 115 unit tests, building against tch 0.24 / PyTorch 2.11. **The tch backend has since been fully replaced by candle** ([PLAN2.md](./PLAN2.md)); the current test suite is 148 unit tests plus integration/checkpoint suites.

## Core Architecture

GLiNER2 inference consists of three main stages:

1. **Text Tokenization & Schema Encoding** - Convert text and schema into token IDs
2. **Transformer Forward Pass** - Run the encoder (BERT-like) to get token embeddings
3. **Span/Classification Extraction** - Compute span representations and extract entities, relations, structures, and classifications

## Implementation Status

### Phase 1: Foundation & Dependencies ✅ COMPLETE

**Status**: All foundational code implemented.

**Dependencies**:
- ✅ `tch` (PyTorch bindings) - For neural network inference
- ✅ `tokenizers` - HuggingFace tokenizer library (Rust)
- ✅ `serde` / `serde_json` - JSON serialization/deserialization
- ✅ `regex` - Regex validators
- ✅ `ndarray` - N-dimensional arrays
- ✅ `half` - FP16 support
- ✅ `safetensors` - Model weight loading
- ✅ `ureq` - HTTP client for model downloads (optional)
- ✅ `rayon` - Parallel processing
- ✅ `tracing` / `tracing-subscriber` - Logging

**Tasks**:
- [x] Configure `Cargo.toml` with dependencies
- [x] Set up project structure
- [x] Create error types (`GlinerError`) and Result aliases
- [x] Implement configuration loading (`ExtractorConfig` with builder pattern)

**Files**:
- `Cargo.toml` - Project configuration
- `src/lib.rs` - Public API and re-exports
- `src/error.rs` - Comprehensive error types (395 lines)
- `src/config.rs` - ExtractorConfig with builder, presets, validation (724 lines)

### Phase 2: Tokenizer & Schema Encoding ✅ COMPLETE

**Status**: All tokenization and schema encoding implemented.

**Components**:

#### 2.1 Whitespace Tokenizer ✅
- [x] Port `WhitespaceTokenSplitter` regex-based tokenizer
- [x] Handle URLs, emails, mentions, and word tokenization
- [x] Map character positions to token indices
- [x] `TokenizedText` helper struct with span extraction

**Files**:
- `src/tokenizer.rs` - WhitespaceTokenizer, Token, TokenizedText, mapping utilities (524 lines)

#### 2.2 Schema Types & Builder ✅
- [x] Port `Schema` class with all task types
- [x] Implement special token handling (`[P]`, `[E]`, `[C]`, `[R]`, `[DESC]`, `[EXAMPLE]`, `[OUTPUT]`)
- [x] Schema-to-tokens conversion for entities, classifications, structures, relations
- [x] Fluent API builders for all schema components
- [x] Schema validation and dict/JSON conversion
- [x] `RegexValidator` for post-processing

**Files**:
- `src/schema/types.rs` - Schema, EntityDef, ClassificationDef, StructureDef, RelationDef, FieldDef, RegexValidator, TaskType, FieldDtype, MatchMode (1003 lines)
- `src/schema/builder.rs` - SchemaBuilder, EntityBuilder, ClassificationBuilder, StructureBuilder, FieldBuilder, RelationBuilder (885 lines)

#### 2.3 Batch Collation ✅
- [x] Port `PreprocessedBatch` data structure
- [x] Implement `ExtractorCollator` for inference mode
- [x] Handle padding, attention masks, and index mappings
- [x] Implement `max_len` truncation support
- [x] Special token constants and schema encoding

**Files**:
- `src/batch/preprocessed.rs` - PreprocessedBatch with builder, field access, batch operations (806 lines)
- `src/batch/collator.rs` - ExtractorCollator, schema encoding, token building (777 lines)

### Phase 3: Model Architecture ✅ IMPLEMENTED & COMPILES

**Status**: All neural network components implemented with unit tests. Compiles successfully with tch 0.24.

**Components**:

#### 3.1 Extractor Model ✅
- [x] Port the `Extractor` class with all submodules
- [x] Encoder integration (via tch)
- [x] Span Representation Layer (`span_rep`)
- [x] Count Prediction (`count_embed`, `count_pred`)
- [x] Classifier (`classifier`)
- [x] Forward pass implementation
- [x] Embedding extraction (fast-path and loop-based)
- [x] Builder pattern for Extractor

**Files**:
- `src/model/extractor.rs` - Main Extractor model, forward pass, embedding extraction (858 lines)
- `src/model/span_rep.rs` - SpanRepresentationLayer, SpanRepOutput, width embeddings (586 lines)
- `src/model/count_pred.rs` - CountPredictionLayer, CountEmbedding, count prediction (670 lines)
- `src/model/classifier.rs` - ClassifierHead, activation functions, classification output (562 lines)

#### 3.2 Model Loading ✅
- [x] Load weights from safetensors format
- [x] Support sharded models
- [x] Weight mapping and validation
- [x] HuggingFace directory loading utilities
- [x] Device placement support

**Files**:
- `src/model/loading.rs` - ModelLoader, weight mapping, safetensors loading, utils (660 lines)

#### 3.3 Configuration ✅
- [x] Port `ExtractorConfig` with all fields
- [x] Builder pattern for configuration
- [x] Preset configurations (base, large, fp16, cpu-optimized)
- [x] Serialization/deserialization
- [x] Validation

**Files**:
- `src/config.rs` - (See Phase 1)

### Phase 4: Inference Pipeline ✅ IMPLEMENTED & COMPILES

**Status**: Main inference engine implemented with unit tests. Compiles successfully with tch 0.24.

**Components**:

#### 4.1 Main GLiNER2 API ✅
- [x] `from_pretrained(model_name_or_path)` - Load model
- [x] `create_schema()` - Create schema builder
- [x] `extract_entities(text, schema)` - Entity extraction
- [x] `batch_extract_entities(texts, schema)` - Batch entity extraction
- [x] `classify_text(text, tasks)` - Text classification
- [x] `batch_classify_text(texts, tasks)` - Batch classification
- [x] `extract_relations(text, relations)` - Relation extraction
- [x] `batch_extract_relations(texts, relations)` - Batch relation extraction
- [x] `extract(text, schema)` - General extraction with custom schema
- [x] `batch_extract(texts, schema)` - Batch general extraction
- [x] `extract_json(text, structures)` - Structured data extraction
- [x] `batch_extract_json(texts, structures)` - Batch structured extraction

**Files**:
- `src/inference/engine.rs` - GLiNER2 struct, all extraction methods, batch processing (915 lines)

#### 4.2 Extraction Logic ✅
- [x] Entity extraction with confidence and span support
- [x] Relation extraction with head/tail pairs
- [x] Structure extraction with field values
- [x] Classification with single/multi-label support
- [x] Threshold filtering and overlap removal
- [x] Result formatting

### Phase 5: Result Formatting & Post-Processing ✅ IMPLEMENTED

**Status**: Result formatting integrated into inference engine.

**Components**:
- [x] Entity dict formatting
- [x] Structure formatting
- [x] Relation formatting
- [x] Classification formatting
- [x] Deduplicate and sort results
- [x] Apply regex validators

### Phase 6: Testing & Validation ✅ ALL 115 TESTS PASSING

**Status**: 115 unit tests + 11 doctests all passing. Full test suite executes successfully with PyTorch runtime.

**Test Coverage**:
- ✅ Tokenizer tests (12 tests in `tokenizer.rs`)
- ✅ Schema builder tests (8 tests in `builder.rs`)
- ✅ Schema types tests (9 tests in `types.rs`)
- ✅ Batch collation tests (6 tests in `collator.rs`)
- ✅ Preprocessed batch tests (12 tests in `preprocessed.rs`)
- ✅ Model classifier tests (11 tests in `classifier.rs`)
- ✅ Model count prediction tests (15 tests in `count_pred.rs`)
- ✅ Model extractor tests (8 tests in `extractor.rs`)
- ✅ Model loading tests (4 tests in `loading.rs`)
- ✅ Model span representation tests (10 tests in `span_rep.rs`)
- ✅ Inference engine tests (5 tests in `engine.rs`)
- ✅ Config tests (7 tests in `config.rs`)
- ✅ Error tests (2 tests in `error.rs`)
- ✅ Doctests (11 passing, 13 ignored)

**Fixes Applied**:
- Fixed arithmetic underflow in `span_rep.rs` forward method (`saturating_sub`)
- Fixed tensor stacking for empty width entries (pad to `seq_len` instead of `[0, hidden_size]`)
- Fixed 2D tensor to vector conversion in span mask test (`flatten` before `try_into`)
- Fixed config builder `fp16`/`bf16` mutual exclusion to allow validation to catch errors
- Fixed extractor builder test to use `hidden_size` divisible by `num_attention_heads`
- Updated tokenizer tests to match actual regex behavior (punctuation as separate tokens)
- Fixed doc tests for `max_len(Some(n))`, `Vec<String>` types, and token counts

**Remaining Tasks**:
- [ ] Integration tests comparing Rust vs Python output
- [ ] Batch correctness tests (batch vs single-sample)
- [ ] Performance benchmarks
- [ ] Edge case tests (empty input, long text, special characters)

### Phase 7: Build & Deployment Setup ✅ BUILD SUCCESSFUL

**Status**: Project compiles successfully with tch 0.24 and PyTorch 2.11 via `LIBTORCH_USE_PYTORCH=1`.

**Tasks**:
- [x] Fix Cargo.toml dependency issues (ureq optional, bench commented)
- [x] Install PyTorch in virtual environment
- [x] Upgrade tch from 0.17 to 0.24 for PyTorch 2.11 compatibility
- [x] Fix 56+ compilation errors from tch API changes
- [x] Get project to compile successfully (`cargo check` passes)
- [x] Get tests to compile successfully (`cargo test --no-run` passes)
- [ ] Run test suite and fix any runtime failures
- [ ] Set up CI/CD pipeline
- [ ] Create examples and documentation

## File Structure

> Historical (tch-era) layout. The current structure is candle-based and is
> documented in `AGENTS.md` and `PLAN2.md`.

```
gliner2-candle/
├── Cargo.toml                      # ✅ Project configuration
├── PLAN.md                         # ✅ This file
├── src/
│   ├── lib.rs                      # ✅ Public API and re-exports (57 lines)
│   ├── config.rs                   # ✅ ExtractorConfig with builder (724 lines)
│   ├── error.rs                    # ✅ Error types (395 lines)
│   ├── tokenizer.rs                # ✅ Whitespace tokenizer (524 lines)
│   ├── BACK_STORY.md               # ❌ Removed — see CHANGELOG.md for history
│   ├── schema/
│   │   ├── mod.rs                  # ✅ Schema module (35 lines)
│   │   ├── builder.rs              # ✅ Schema builders (885 lines)
│   │   └── types.rs                # ✅ Schema types (1003 lines)
│   ├── model/
│   │   ├── mod.rs                  # ✅ Model module (36 lines)
│   │   ├── extractor.rs            # ✅ Main Extractor model (858 lines)
│   │   ├── span_rep.rs             # ✅ Span representation layer (586 lines)
│   │   ├── count_pred.rs           # ✅ Count prediction layers (670 lines)
│   │   ├── classifier.rs           # ✅ Classification head (562 lines)
│   │   └── loading.rs              # ✅ Model weight loading (660 lines)
│   ├── batch/
│   │   ├── mod.rs                  # ✅ Batch module (22 lines)
│   │   ├── preprocessed.rs         # ✅ PreprocessedBatch (806 lines)
│   │   └── collator.rs             # ✅ ExtractorCollator (777 lines)
│   └── inference/
│       ├── mod.rs                  # ✅ Inference module (19 lines)
│       └── engine.rs               # ✅ Main GLiNER2 struct (915 lines)
├── tests/                          # ❌ Empty - needs tests
├── examples/                       # ❌ Empty - needs examples
└── benches/                        # ❌ Empty - needs benchmarks
```

**Total Lines of Code**: ~9,534 lines of Rust (excluding tests)

## Key Technical Decisions

### 1. Tensor Backend
**Decision**: Use `tch` (PyTorch bindings) for model compatibility.

**Status**: ✅ Implemented, but requires libtorch installation to build.

**Rationale**: 
- Direct compatibility with PyTorch model weights
- Easier debugging (can compare with Python output)
- Mature ecosystem

### 2. Tokenizer
**Decision**: Custom regex-based whitespace tokenizer matching Python implementation.

**Status**: ✅ Implemented and ready for testing.

### 3. Batch Processing
**Decision**: DataLoader-style batching with configurable workers.

**Status**: ✅ Implemented with PreprocessedBatch and ExtractorCollator.

### 4. Output Format
**Decision**: Match Python output format exactly for drop-in compatibility.

**Status**: ✅ Implemented in inference engine.

## API Compatibility Matrix

| Python Feature | Rust Status | Notes |
|---------------|-------------|-------|
| `from_pretrained()` | ✅ Implemented | Load from HF hub or local path |
| `from_api()` | ❌ Not implemented | Optional, can be added later |
| `create_schema()` | ✅ Implemented | Fluent API |
| `extract_entities()` | ✅ Implemented | Single text |
| `batch_extract_entities()` | ✅ Implemented | Batch with workers |
| `classify_text()` | ✅ Implemented | Single text |
| `batch_classify_text()` | ✅ Implemented | Batch |
| `extract_json()` | ✅ Implemented | Structured data |
| `batch_extract_json()` | ✅ Implemented | Batch |
| `extract_relations()` | ✅ Implemented | Relations |
| `batch_extract_relations()` | ✅ Implemented | Batch |
| `extract()` | ✅ Implemented | Custom schema |
| `batch_extract()` | ✅ Implemented | Batch custom |
| `include_confidence` | ✅ Implemented | All extraction types |
| `include_spans` | ✅ Implemented | All extraction types |
| `max_len` | ✅ Implemented | Truncation |
| `threshold` | ✅ Implemented | Confidence threshold |
| `RegexValidator` | ✅ Implemented | Post-processing |
| `quantize()` | ✅ Implemented | FP16 support |
| `compile()` | ⚠️ Stub | tch compile support |
| Training | ❌ Out of scope | Not implemented |
| LoRA | ❌ Out of scope | Not implemented |

## Current Build Status

### ✅ Build & Tests Successful

The project compiles and all tests pass with:
- **tch**: 0.24 (upgraded from 0.17)
- **PyTorch**: 2.11 (via Python virtual environment)
- **Environment**: `LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1`
- **Runtime**: `DYLD_LIBRARY_PATH` set to PyTorch lib directory on macOS

**Commands**:
```bash
# Check compilation
LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 PATH=".venv/bin:$PATH" cargo check

# Run all tests (115 unit tests + 11 doctests)
LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 PATH=".venv/bin:$PATH" \
  DYLD_LIBRARY_PATH=".venv/lib/python3.14/site-packages/torch/lib:$DYLD_LIBRARY_PATH" \
  cargo test
```

### ⚠️ Remaining Issues

1. **No examples**: No example code demonstrating API usage
2. **Unused code warnings**: ~60 warnings about unused imports/variables (cosmetic, can be cleaned up)
3. **CI/CD**: No automated testing pipeline configured

## Next Steps

### Immediate (Completed ✅)

- [x] Fixed runtime linking with `DYLD_LIBRARY_PATH`
- [x] All 115 unit tests passing
- [x] All 11 doctests passing
- [x] Fixed arithmetic underflow, tensor stacking, and type conversion issues

### Short-term (Testing & Polish)

1. **Integration tests**: Compare Rust output against Python GLiNER2 reference implementation
2. **Batch tests**: Verify batch processing produces same results as single-sample processing
3. **Performance benchmarks**: Measure inference speed vs Python implementation
4. **Clean up warnings**: Remove unused imports/variables (~60 warnings)
4. **Run existing test suite**:
   - 115 unit tests are already written across all modules
   - Run `cargo test` once libtorch is installed
   - Fix any failing tests that arise from tch API changes or implementation issues

5. **Integration tests**:
   - Compare Rust output vs Python output
   - Test with actual model weights
   - Batch processing tests

### Medium-term (Polish)

6. **Add examples**:
   - Basic entity extraction
   - Batch processing
   - Custom schemas
   - Classification
   - Relation extraction
   - Structured data extraction

7. **Documentation**:
   - API documentation
   - Usage guide
   - Performance benchmarks
   - Migration guide from Python

8. **Performance optimization**:
   - Profile critical paths
   - Optimize tensor operations
   - Consider SIMD optimizations
   - Test with various batch sizes

### Long-term (Future)

9. **Additional features**:
   - API client (`from_api()`)
   - Model compilation optimization
   - GPU support verification
   - LoRA adapter support (if needed)

## Performance Targets

- **Single text inference**: < 100ms on CPU for short texts
- **Batch inference**: Linear scaling with batch size, efficient GPU utilization
- **Memory**: < 2GB for base model on CPU
- **Accuracy**: Bit-identical or near-identical (< 1e-5 difference) with Python output

## Risks & Mitigations

| Risk | Status | Mitigation |
|------|--------|-----------|
| Model weight loading complexity | ⚠️ Untested | Use safetensors format, test with multiple models |
| Tensor operation differences | ⚠️ Untested | Extensive testing against Python output |
| Performance on CPU | ⚠️ Untested | Optimize critical paths, consider SIMD |
| Tokenizer edge cases | ⚠️ Untested | Match Python tokenizer exactly, test edge cases |
| Batch size variations | ⚠️ Untested | Test with various batch sizes, ensure correctness |
| libtorch dependency | ❌ Blocking | Provide clear installation instructions, consider candle fallback |

## Milestones

> As a historical document, the M7–M10 items below were completed in the
> candle migration ([PLAN2.md](./PLAN2.md)) and GLiNER2.5 work
> ([PLAN_2.5.md](./PLAN_2.5.md)).

1. **M1**: ~~Project setup, dependencies, basic structure~~ ✅ COMPLETE
2. **M2**: ~~Tokenizer and schema encoding~~ ✅ COMPLETE
3. **M3**: ~~Model architecture~~ ✅ IMPLEMENTED & COMPILES
4. **M4**: ~~Inference pipeline~~ ✅ IMPLEMENTED & COMPILES
5. **M5**: ~~Get project building~~ ✅ BUILD SUCCESSFUL (tch 0.24, PyTorch 2.11)
6. **M6**: ~~Write test suite~~ ✅ 115 TESTS COMPILE
7. **M7**: ~~Run tests and fix runtime failures~~ ✅ DONE (in candle migration)
8. **M8**: ~~Integration testing vs Python~~ ✅ DONE (full numeric parity; see README)
9. **M9**: ~~Examples and documentation~~ ✅ DONE (README, AGENTS.md, CHANGELOG.md)
10. **M10**: ~~Performance optimization and release prep~~ ✅ DONE (release CPU ~71ms/call)

## Development History

See `CHANGELOG.md` at the repo root for the full development and release
history across the tch era, the candle migration, and GLiNER2.5 support.

## Notes

- All code was generated using Qwen3.6 Plus Preview via OpenRouter
- Context grew past 128K tokens during initial generation
- Code includes 115 unit tests across all modules
- **tch upgraded from 0.17 to 0.24** for PyTorch 2.11 compatibility
- **56+ compilation errors fixed** from tch API changes (Init, Tensor methods, Clone traits, etc.)
- **Project compiles successfully** - `cargo check` and `cargo test --no-run` both pass
- Next priority: Run tests and fix any runtime failures, then integration testing vs Python