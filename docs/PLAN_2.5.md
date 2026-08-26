# GLiNER2.5 Support Plan

## Overview

GLiNER2.5 (released 2026-08-24 by Fastino) replaces GLiNER2's span-enumeration
architecture with **boundary prediction**, and adds five capabilities:

1. **Unlimited span length** — no `max_width` cap; boundaries are predicted, not enumerated.
2. **Long-context extraction** — up to 4,096 words in one pass, plus library-native chunking with offset remapping and cross-chunk merge policies.
3. **Joint entity + relation extraction** — relations decoded from the same candidate pool as entities, via constrained beam search so output graphs are valid by construction (`unique_head`, `no_self_loops`, cardinality).
4. **Constrained classification** — implication/exclusion constraints enforced during decoding; invalid combinations are never admitted; infeasible schemas raise errors instead of returning contradictions.
5. **Span attributes** — attribute groups (e.g. sentiment) scored per extracted span in the same forward pass.

Released checkpoints (Apache 2.0):

| Model | Params | Notes |
|---|---|---|
| `fastino/gliner2.5-small-v1` | 73.9M | |
| `fastino/gliner2.5-base-v1` | 0.2B | |
| `fastino/gliner2.5-multi-v1` | 0.3B | multilingual, best overall F1 |

**Strategy**: support both architectures behind a runtime-detected
`Architecture` enum. GLiNER2 is kept working but is no longer the priority;
all new feature work targets 2.5.

---

## Architectural delta: span enumeration → boundary prediction

GLiNER2 (current code):

```
encoder → token_embs → span_rep[i, w] = out(proj_start(e_i), proj_end(e_{i+w}))
        → classifier over (seq_len, max_width) grid → sigmoid → threshold
```

Computation scales as `O(seq_len × max_width)`, and entities longer than
`max_width` (default 12) are structurally invisible.

GLiNER2.5:

```
encoder → (text tokens + schema queries in one pass)
  → per-query start scores, end scores, inside scores over token positions
  → sparse proposal stage: top-k start/end boundaries per query, paired
    (no distance restriction)
  → reranking head: scores each candidate from boundary evidence + span content
  → relation candidates drawn from the same pool
  → constrained decoding (beam search for joint IE, constraint solver for
    classification)
```

Computation is linear in sequence length for a fixed schema/candidate budget.
No width axis exists anywhere in the computation.

**Open item**: exact weight names/shapes of the new heads must be confirmed by
inspecting the safetensors of each checkpoint on the Hub. Everything below that
depends on head names should be treated as provisional until Phase 2 step 1.

---

## Current coupling inventory (what blocks 2.5)

`max_width` / fixed-span-matrix assumptions appear in:

- `src/config.rs` — `max_width` field (line 240), validation (>0, line 396),
  builder method (481), presets.
- `src/model/extractor.rs` — `max_width` field (150), empty-tensor shapes at
  578–591, accessor/builder at 801–866.
- `src/model/span_rep.rs` — entire module is the GLiNER2-only path
  (`SpanRepOutput`: `(seq_len, max_width, hidden)` etc.). Stays for legacy
  track only; not used by 2.5.
- `src/inference/engine.rs` — decode loops iterate `mask_max_width` /
  `max_width` grids (~75 references, lines 980–1834). These loops are replaced
  wholesale by boundary decoding for 2.5.
- `src/model/count_pred.rs`, `count_embed.rs` — count prediction tied to the
  span-grid path; verify whether 2.5 retains count heads (check weights).

Well-positioned foundations we can build on:

- Schema layer (`src/schema/types.rs`, `builder.rs`) already has `Schema`,
  `EntityDef`, `ClassificationDef`, `RelationDef`, structures, and task types.
- Encoder stack (`deberta_v3.rs`, `candle_encoder.rs`) is reusable as-is,
  modulo possibly larger position-embedding tables for long context.
- Collator pipeline is shared; needs the known bug fixes plus schema-token
  changes if 2.5's serialization format differs (verify against Python repo).

Known gaps with no existing mechanism:

- No versioning/architecture detection anywhere. `EncoderType::from_model_name`
  string-matches `"gliner2"` (candle_encoder.rs:60).
- No relation decoding beyond independent triple thresholding.
- No constraint system, no beam search, no attribute heads, no chunking.

---

## Phases

### Phase 0 — Architecture detection & dual-track scaffolding

Goal: a single `from_pretrained` entry point that loads either architecture.

Tasks:

- [ ] Download/inspect `config.json` + safetensors headers for all three 2.5
      checkpoints. Record: encoder type & size, vocab, max positions, head
      names/shapes, presence/absence of count_pred, any new embeddings.
- [ ] Add `Architecture { Gliner2, Gliner25 }` to config; detect from HF
      config fields first, fall back to weight-name sniffing (presence of
      boundary heads vs `span_rep.span_rep_layer`).
- [ ] Fix `EncoderType::from_model_name` to stop hijacking all "gliner2*"
      names; route 2.5 checkpoints correctly.
- [ ] Presets: `gliner25_small()`, `gliner25_base()`, `glicer25_multi()` in
      `config::presets`. Deprecate nothing yet.
- [ ] `Extractor` dispatches forward-pass on architecture; 2.5 initially
      returns `Error::Unsupported("boundary decoder pending")`.

Acceptance: both a GLiNER2 and a GLiNER2.5 checkpoint load fully into their
respective module sets with zero shape mismatches.

### Phase 1 — GLiNER2 correctness baseline (prerequisite)

The shared pipeline (collator subword index tracking, schema embedding
extraction, engine decode) currently returns empty entities. 2.5 rides on the
same plumbing; fix it first.

- [ ] Fix `schema_special_indices` to track **subword** positions, not schema
      token positions (`src/batch/collator.rs` ~300–400).
- [ ] Fix empty `text_word_indices` mapping (whitespace token → first subword
      position).
- [ ] Add tensor-level golden tests: dump intermediate tensors (encoder output,
      span reps, logits) from the Python reference implementation and compare
      within tolerance in `cargo test`.
- [ ] Verify end-to-end extraction against Python outputs on a fixture set
      (~20 text/schema pairs covering entities, classification, relations,
      structures).

Acceptance: Rust matches Python outputs on fixtures; integration test green.
This unblocks trusting the encoder/collator for 2.5 work.

### Phase 2 — Boundary prediction core

New file: `src/model/boundary.rs`.

- [ ] Implement per-query heads:
      - `start_head`, `end_head`, `inside_head` → `(num_queries, seq_len)`
        score vectors each
      - reranking head consuming concatenated boundary evidence + span content
        representation (exact input composition TBD from weights)
- [ ] Sparse proposal stage:
      - top-k starts and ends per query (k configurable, default TBD)
      - pair candidates; enforce inside-score consistency filter
      - no distance restriction between start/end
- [ ] Rerank proposals → final candidate list per query with confidence.
- [ ] Entity decoding: threshold/filter candidates, map token spans → char
      offsets via existing word-index mappings.
- [ ] Wire into engine behind `Architecture::Gliner25`; delete no GLiNER2 code.
- [ ] Golden tests vs Python boundary scores and final extractions.

Acceptance: unlimited-length entities extractable; parity with Python on
fixtures including >12-word spans.

### Phase 3 — New decoding features

#### 3a. Constrained classification

New file: `src/constraints.rs` (pure logic, no ML — exhaustively testable).

- [ ] Constraint IR: `implies((task,label), (task,label))`,
      `excludes(...)`, per-task min/max label counts, single/multi selection.
- [ ] Decoder: enumerate feasible assignments over per-task score tables
      (label counts are small; exhaustive search with pruning is fine) or
      weighted-solver fallback for large label spaces. Invalid combos never
      admitted; raise `InfeasibleConstraint` error when no valid assignment
      exists.
- [ ] Builder API mirroring Python: `.single(...)`, `.multi(min,max)`,
      `.constrain(C.implies(..), ..)` on `ClassificationBuilder`.
- [ ] Applies to both architectures where sensible, but only required for 2.5.

#### 3b. Joint IE (relations)

- [ ] Extend `RelationDef` with typed `head_entity` / `tail_entity` names and
      structural flags: `unique_head`, `no_self_loops` (builder methods to
      match Python `joint.create_schema()` style).
- [ ] Relation scoring head over the Phase-2 candidate pool (head/tail span
      pairs + relation-type query). Weight layout TBD from inspection.
- [ ] Beam search assembler (`src/inference/joint.rs`): incrementally builds
      graph from ranked candidates, checking constraints during construction;
      returns guaranteed-valid graph with per-edge confidence.
- [ ] Config knobs: `optimizer ∈ {greedy, beam}`, `beam_size`.
- [ ] Output type: entities + relations referencing them (by index/text),
      matching Python result shape.

#### 3c. Span attributes

- [ ] Extend `EntityDef`/schema with `AttributeGroup { labels, applies_to,
      multi, qualify_labels, threshold }`; builder method
      `.entity_attributes({...})`.
- [ ] Attribute heads scored per selected span in the same forward pass
      (weight names TBD).
- [ ] Output JSON: each span carries `{ "<attr>": {label, confidence} }`;
      respect `include_spans` / `include_confidence` flags.

### Phase 4 — Long context & public API (complete)

- [x] Chunking utility (`src/chunking.rs`):
      - word-window splitting with overlap, respecting encoder max positions
      - run extraction per chunk, batch chunks where possible
      - remap every span to original-document character offsets
      - deterministic merge policies for duplicate spans across overlaps
        (highest-confidence, longest-span, first-seen — user selectable)
- [x] Long-context auto-chunking in `batch_extract` (entities, classify,
      JSON schema, relations) with automatic chunking above a length
      threshold; opt-out flag.
- [x] Confirmed: all three checkpoints declare `max_len: 4096`; encoder
      `max_position_embeddings` is not an input cap (relative positions).
- [ ] Public API: `GLiNER2::from_pretrained` transparently dispatches;
      `max_width()` builder methods emit deprecation warnings when a 2.5 model
      is loaded (no-op there).
- [ ] Update output types so 2.5 results carry attributes/graphs without
      breaking GLiNER2 result shapes (additive serde fields).

### Phase 5 — Validation & release

- [ ] Parity tests vs Python `gliner2` repo (fastino-ai/GLiNER2) for all five
      capabilities on shared fixtures.
- [ ] Integration tests downloading all three checkpoints
      (`tests/real_inference_test_25.rs`), gated like existing hub tests.
- [ ] Performance smoke: linear-scaling check on documents of 512 / 2048 /
      4096 words; memory profiling of proposal stage.
- [ ] Docs: update AGENTS.md architecture summary, README examples, mark
      GLiNER2 track as maintenance-mode in PLAN.md.
- [ ] Semver: this is additive; keep 0.x.

---

## Risks / unknowns

| Risk | Mitigation |
|---|---|
| Unknown head weight names/shapes | Phase 0 step 1 inspects real checkpoints before any head code is written |
| 2.5 schema serialization may differ from GLiNER2's `[P]`/`[E]` format | Compare collator behavior against Python 2.5 repo in Phase 1 golden tests |
| Count-prediction path may be removed or reshaped in 2.5 | Determine from weights in Phase 0; keep legacy path untouched |
| Beam search performance in pure Rust | Candidate budgets are small (top-k per query); profile before optimizing |
| Constrained assignment NP-hard in pathological schemas | Label spaces per task are tiny; cap exhaustive enumeration, document limits |
| Long context may exceed DeBERTa rel-embedding table (512) | Check checkpoint's rel_embeddings size in Phase 0 |

## Sequencing summary

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 (3a ∥ 3b ∥ 3c) ──► Phase 4 ──► Phase 5
detect      fix base    boundary    features                   long ctx    validate
```

Phase 1 precedes Phase 2 deliberately: encoder, collator, tokenizer, and schema
embedding extraction are shared and currently broken; everything in 2.5
depends on them being numerically trustworthy.
