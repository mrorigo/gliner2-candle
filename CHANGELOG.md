# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/) guidance;
versions remain `0.x` (pre-1.0, additive-only semver policy).

## [Unreleased]

### Changed

- **Rebrand: crate renamed `gliner2-rs` → `gliner2-candle`** (crate namespace
  `gliner2_rs` → `gliner2_candle`), and the GitHub repository renamed to
  `gliner2-candle` (history retained). `gliner2` / `gliner2-rs` on crates.io are
  already taken by ONNX-Runtime-based crates, so `-candle` signals the
  pure-Rust backend. Version stays `0.1.0`.
- Docs: full refresh — rewritten `README.md`, `CHANGELOG.md`,
  `docs/index.html`, and status banners on the historical plans
  (`docs/PLAN.md`, `docs/PLAN2.md`, `docs/PLAN_2.5.md`).
- `Cargo.toml` and `src/lib.rs` now describe GLiNER2.5 support; the `lib.rs`
  quick-start example is corrected to the synchronous API.
- `docs/index.html` rewritten with a refreshed "candle flame" design, task
  parity matrix, animated pipeline, and a backend-comparison table.

## 2026-08-27 — Phase D task parity + hygiene

### Added

- **Phase D: full task parity across all four output types** (commit `9bde59c`):
  - Entities: full matrix parity vs Python (global `0.0000`, relevant `0.0000`)
  - Classifications: exact output match (positive)
  - Relations: exact format match (bare pairs, no flags)
  - Attributions / span attributes (single-label): softmax logits to 8 decimals
    (`0.9966161847`)
  - Attributions / span attributes (multi-label): sigmoid logits to 7 decimals
    (`0.5735875`)
- Constrained classification support (`src/constraints.rs`): Kleene-3,
  implies / excludes / exactly-one-of, and per-label constraints.

### Fixed

- **Endpoint-difference vector layout (Phase D critical bug)**: `torch.cat((d,
  |d|), dim=-1)` concatenates two halves (all diffs, then all abs diffs); the
  Rust code interleaved `(d,|d|)` per-dimension. Fixed to push all diffs first,
  then all absolute diffs, removing a constant ~0.27–0.38 logit shift.
- **Relation scorer input**: now receives H-dimension (768) word states, not
  endpoint-D (128) states.
- **erf() GELU**: implemented with `exp(-x^2)` (exact erf), not the tanh
  approximation.
- **Content pooler ownership**: the shared-pool scorer now uses its own
  `content_pooler` weights rather than the (identically-shaped, differently-
  valued) pair-scorer copy, preventing up to 8 logits of error on non-top
  candidates.
- **Content projection bias** and **classification `[C]` marker** handling.
- Removed all debug `eprintln!` output from `src/model/boundary.rs` and
  `src/inference/boundary.rs` (including a stray `}` that broke compilation).
- DeBERTa V3 encoder parity and boundary scorer correctness (commit `a0cc4c1`).

### Dependency / security

- `cargo update` to address `crossbeam-epoch`, `h2`, and `rustls-webpki`
  advisories; `cargo audit` reports 0 vulnerabilities. Audit ignore policy has
  a review date (`74e34a5`).
- `cargo audit` clean; `cargo fmt` clean; `cargo clippy --all-targets` 0
  warnings.

## 2026-08 — GLiNER2.5 boundary pipeline

### Added

- **GLiNER2.5 full boundary pipeline** (Phases 0–5): boundary encoder,
  shared-pool scorer with relation reranker, record decoder, relation
  scoring + decoding (Phase 3d), span attributes with flat overlap resolution
  and the content-pooler parity fix (Phase 3e), profile-guided performance
  work, long-document auto-chunking + merge (>384 words) in `batch_extract`
  (Phase 4), and all-checkpoint integration tests (Phase 5).
- Custom DeBERTa V3 encoder (`src/model/deberta_v3.rs`) with sign-flipped
  `rel_pos` matching HF (`q - k`), `norm_rel_ebd` applied to rel embeddings,
  and no output LayerNorm.
- Test suite: `tests/real_inference_test_25.rs` with `test_task_output_parity`
  (real parity assertions) and attribute schema examples.

### Performance

- Encoder relative-bias kernels (~15ms → ~7ms/layer) and a tensorized boundary
  scorer (`score_sample` ~72ms → ~27ms/chunk). Release, CPU: short inputs at
  Python parity (~71ms/call); ~3.1s end-to-end for a 2400-word document.
- Added benchmark harness and optimized dev dependencies (`[profile.dev.package."*"]
  opt-level = 3` for fast tests).

## 2026 — Candle migration

### Changed

- Migrated the entire backend from `tch` (PyTorch bindings, ~2GB libtorch
  runtime) to **candle** (HuggingFace's pure-Rust ML framework). One tensor
  backend, zero data copying, no `LIBTORCH_*` environment requirements.

### Added

- `src/model/candle_encoder.rs` wrapping BERT/DeBERTa V2/V3 via candle.
- `hf-hub` integration for direct model weight downloads from HuggingFace Hub.
- Real end-to-end inference with concrete model weights (no placeholder/random
  embeddings).

## 2026 — Initial tch implementation

### Added

- Initial GLiNER2 inference port against the `tch` / PyTorch backend:
  tokenizer + schema encoding, model architecture, span-representation layer,
  classifier / count heads, and the inference engine. ~9,500 lines with 115
  unit tests.
- Superseded by the candle migration (see above). Historical plan:
  `docs/PLAN.md` (tch era), `docs/PLAN2.md` (candle migration).
