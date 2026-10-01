# Response to upstream findings — `gliner2-candle`

Thanks for the detailed reproductions. All three findings were investigated
against the Python reference, and the outcome is: **two fixed, one confirmed
not-a-bug** — plus four further bugs your reproductions uncovered on the way,
one of which changed a decision you have recorded.

Everything below is measured against `fastino-ai/GLiNER2` (git main, the
version that `AutoExtractor` can load). Where a number is quoted it is the
actual output on both sides, not an estimate.

---

## Summary

| ID | Your status | Our status | Outcome |
|---|---|---|---|
| [1](#1-classification-label_descriptions-are-silently-dropped) | Open | **Fixed** | Fixed, and the entity path was broken differently than reported — see below |
| [2](#2-head-composition-changes-confidence-in-every-other-head) | Open, arguably by design | **Confirmed not-a-bug** | Inherent, and we match the reference to float32 in every configuration we tested |
| [3](#3-decide-checkpoints-silently-return-no-entity-spans) | Documented, enhancement requested | **Fixed, better than requested** | `CountLSTM` is implemented. This invalidates `decisions/0004` — please re-read it |

Along the way we found four more defects that your reproductions would have hit
regardless, described in [Additional findings](#additional-findings). Two of
them were severe enough that they would have made your evaluation numbers
meaningless even after finding 1 was fixed.

Per your reporting convention, all three entries are ready to be marked:

- **1** → Fixed in `0.3.1` (PR [#6](https://github.com/mrorigo/gliner2-candle/pull/6))
- **2** → Confirmed not-a-bug; see the numbers below for your citation
- **3** → Fixed in `0.3.1`; supersedes the enhancement request

---

## 1. Classification `label_descriptions` are silently dropped

**Status: Fixed.** Your diagnosis was correct and your reproduction is now a
regression test.

### Your diagnosis was right about the classification path

`Schema::to_dict` hand-built the classification object field by field and
emitted only `task`, `labels`, `multi_label` and `cls_threshold`. The collator
reads `label_descriptions`, `prompt` and `examples`, so all three were dropped
before the model saw them. Fixed by serialising the classification definition
directly, so the emitted keys are the ones the collator reads by construction
rather than by a hand-maintained list.

### The entity path was broken too, but differently

Your report noted the asymmetry — that the entity path emits
`entity_descriptions` while classification does not — and read that as "the
entity path works". It does not, for two independent reasons:

1. **The key is emitted but never read.** `to_dict` emits
   `entity_descriptions` as a sibling of `entities`, but the collator looked up
   `entities[name]`. Since `to_dict` emits `entities` as an *array*, that
   lookup was always `None`. The emitted descriptions were dead on arrival.
2. **Even when supplied, the shape was wrong.** Descriptions were emitted as
   separate top-level `[DESCRIPTION]` tokens rather than folded into the prompt
   string at token index 2. That changes the token sequence and inflates the
   structural-marker count, which is why the classification path had already
   been corrected and the entity path had not.

We also confirmed your reading of the Python source is the right one: upstream
reads the sibling `entity_descriptions` key and **ignores** inline
`entities: {name: description}` values — those are training labels only. We
verified this directly: passing descriptions inline produces output
bit-identical to passing none.

### One more ordering bug you did not report

`entities_with_descriptions` accepted a `HashMap` and iterated it, so the prompt
string — and therefore the `[E]` marker order — varied between processes. If you
were comparing outputs across runs and saw unexplained movement, this was a
contributor. The API now takes any `(name, description)` iterable, so order is
yours to choose.

### Current behaviour, on your exact input

`fastino/GLiNER2.5-Decide`, `"Charged twice, 49 EUR both times"`, labels
`duplicate_charge` / `refund_request` / `other`:

| descriptions | Rust | Python |
|---|---|---|
| all `"billing issue"` | `duplicate_charge` @ 0.6112574 | @ 0.6112577 |
| all `"weather report"` | `duplicate_charge` @ 0.6072796 | @ 0.6072798 |
| none | `duplicate_charge` @ 0.7903281 | @ 0.7903284 |

Descriptions now demonstrably reach the model, and we agree with the reference
to float32 precision (~3e-7). Your original repro now returns three different
confidences instead of three identical ones.

### One caveat on your expected impact

You wrote that you expect "a material accuracy improvement and a lower
abstention rate once descriptions work." **The mechanism works, but that
expectation is not supported by the evidence, and it may go the other way.**

In the table above, adding descriptions *lowered* confidence from 0.790 to
0.611 on an input the model already classified correctly. Descriptions are
real conditioning signal, and Python shows the same direction — so this is
model behaviour, not a port artifact. But it means:

- Descriptions are a lever for **separating adjacent classes**, not a
  monotonic accuracy improvement.
- Your `duplicate_charge` / `refund_request` confusion may well improve. We
  cannot tell you that it will, and we would not want you to size the
  `support-routing` decision on an unmeasured improvement.
- `decisions/0010` ("descriptions not yet consumed") is now obsolete as a
  *technical* constraint. It is still a valid description of your pack's
  current behaviour, since you deliberately do not pass them. Suggest you
  re-open it and, if you adopt descriptions, **re-measure accuracy and
  re-calibrate thresholds together** — the two interact.

If you want to size this before committing, the honest experiment is to A/B
your existing pack with descriptions on the same holdout you used for the 0.882
figure, rather than inferring the effect from confidence movement.

---

## 2. Head composition changes confidence in every other head

**Status: Confirmed not-a-bug.** Inherent to the architecture, and the reference
does the same thing. Your mitigation advice is the right call.

### The mechanism

All schema groups — every entity type, every classification head — are
concatenated into a single sequence joined by `[SEP_STRUCT]`, with the text
after one `[SEP_TEXT]`, and run through **one** encoder pass with full
bidirectional attention. A second head therefore lengthens the prompt and moves
the contextual state that the first head reads. Both implementations do this
identically.

### Measured, on `gliner2-base-v1`

`"The deploy left resource tags inconsistent across staging."`, `severity`
head with `info`/`low`/`medium`/`high`/`critical`:

| schema | Python | Rust |
|---|---|---|
| `severity` alone | `high` @ 0.5517789 | `high` @ 0.5517772 |
| `severity` + a 4-label `other` head | `high` @ 0.7668723 | `high` @ 0.7668726 |
| `severity` + a 5-label `other` head | `high` @ 0.7772620 | `high` @ 0.7772608 |

Adding one unrelated head moved `severity` by **+0.215** — larger than the
0.11 you measured, and in the same direction — and we reproduce all three
configurations to float32 precision (~1e-6). Adding a *fifth* label to the
unrelated head moved it further, by a smaller amount. The effect scales with how
much prompt the other heads contribute, which is the signature of shared
contextual state rather than of a normalisation bug.

This is a much stronger result than "both implementations agree": it rules out
your hypothesis (2), a parity gap in per-head normalisation. There is no
per-head normalisation to get wrong.

### Your reported numbers are no longer reproducible, and that is expected

You measured `0.7861 → 0.6786`. On the current build that input saturates at
`1.0 → 1.0` — the shift is still there, just hidden by saturation. That is not
the fix making your problem go away; it is [finding 4](#4-text-was-not-normalised-before-collation)
below changing the encoder input. If you were planning to treat your
single-head measurement as a baseline, please re-measure it.

### What we changed

Nothing in the code. We documented the behaviour in the README, because a pack
author has no way to anticipate it:

> Adding a classification head shifts the confidence of the heads already
> present. Heads share one encoder pass… This matches the Python reference to
> float32 precision, so it is inherent rather than a port defect — but it does
> mean thresholds are only valid for the head set they were calibrated against.

Your `Pack::all_tasks` guidance stands, and the table above is a citation you
can use with your pack authors.

---

## 3. Decide checkpoints silently return no entity spans

**Status: Fixed — and this one is better news than you asked for. Please
re-read `decisions/0004`.**

You asked for a signal, and noted that implementing `CountLSTM` would be the
fuller answer. We implemented it. Your reproduction now returns a span:

```text
"Charged twice, 49 EUR both times", schema: entities ["amount"], threshold 0.5

  Rust:   {"amount": [{"text": "49 EUR", "confidence": 0.9222937, ...}]}
  Python: {"amount": [{"text": "49 EUR", "confidence": 0.922312,  ...}]}
```

### What was actually wrong

Two things, and the second is the reason your repro was silent.

**The layer was missing, as you deduced.** Decide and `gliner2-large-v1` ship
`count_embed.gru` + `count_embed.projector.{0,2}` (9 tensors, the
`layers.py::CountLSTM` layout). Only the `CountLSTMv2` layout
(GRU + `DownscaledTransformer`, 37 tensors) was implemented, so those
checkpoints had no count-guided scorer.

**But the loader was also broken on the checkpoints that *did* have the
supported layout.** The probe that decided whether to load `count_embed` asked
for `transformer.in_projector.weight` with shape `(hidden, 128)`. Candle stores
Linear weight as `(out, in)`, and the checkpoint has `(128, 768)`. The probe
failed, `count_embed` became `None`, and **span, relation and structure
decoding returned empty results on every checkpoint in the family** —
`gliner2-base-v1` included. The existing test passed throughout because it only
asserted the output string contains `"entities"`: the key was present, every
value absent.

So the scope of the silent-empty-result problem was much larger than the Decide
family. If you had pointed finding 3 at `gliner2-base-v1` instead, it would have
applied there too.

### What this means for your architecture

`decisions/0004` chose deterministic regex extraction on the grounds that Decide
cannot extract entities. **That premise no longer holds.** Decide now extracts
entities, relations and structures, matching the reference to float32
precision. You do not have to act on this — regex extraction may still be the
right call for support routing on cost, latency and predictability grounds —
but "the model cannot" is no longer among the reasons, and we would rather you
re-decide on the merits than leave a constraint in place that no longer exists.

The 340M Decide model at full forward-pass cost per request is a legitimate
reason to keep regex. The checkpoint's own benchmark (60.2% exact-match on
`fastino/fast-decisions`, from the model card) is a legitimate reason to be
sceptical about swapping it in. Neither of those is "it does not work."

### Your requested signal, delivered

Independently of the port, a checkpoint with no count-aware projection at all now
raises a typed error naming the checkpoint and noting that classification still
works. "The model found nothing" is never again ambiguous with "the model cannot
do this" — which is the failure mode that cost you the afternoon you described.

---

## Additional findings

These came out of building the gate that verified the above, and two of them
would have corrupted your evaluation numbers. Neither is a response to
anything you reported; we are flagging them because they affect measurements you
have already taken.

### 4. Text was not normalized before collation

**Severity: high for your evaluation.** Python's collator appends sentence-final
punctuation when a text does not already end in `.`, `!` or `?`. That punctuation
is a word, so it tokenizes to an extra subword in the encoder input. We never
did this, so for **any text without terminal punctuation** our encoder saw a
sequence one token shorter than the reference.

Every number in this response is from a text ending in a period. Yours
(`"Charged twice, 49 EUR both times"`, `"Charged 49 EUR twice"`) does not. So
your reproductions were run against a different encoder input than the one that
produced the reference values you compared them to. Any confidence you recorded
from text lacking terminal punctuation is not comparable to Python's, and will
have moved when you upgrade.

This is also the direct cause of the `severity` label flipping in finding 2
during our investigation.

### 5. Object keys in schemas were sorted alphabetically

**Severity: high, and it fails silently.** `serde_json` orders object keys
alphabetically unless its `preserve_order` feature is on. Python dictionaries
keep insertion order. For a structure schema this is load-bearing, not
cosmetic: it decides which `[C]` marker precedes which field name, which
fixes query order, which fixes which candidate score belongs to which field.

`{"product_info": {"name": "", "company": ""}}` was being emitted as
`[C] company [C] name`. Each field still received its own marker's embedding,
so nothing looked broken — each field simply carried the other field's scores.
The symptom was a confidence drift of a few percent that reads as numeric noise:

| field / span | Python | Rust (before) |
|---|---|---|
| `name` / `iPhone` | 0.984857 | 0.975979 |
| `company` / `Apple` | 0.984133 | 0.982428 |

If you have a structure or record schema with fields whose alphabetical order
differs from your intended order, your confidences were wrong and are now right.

### 6. Empty results had the wrong shape, and multi-label could return nothing

Two output-shape mismatches with the reference, both visible only when a model
finds nothing:

- Declared entity and relation types were omitted entirely instead of emitted as
  empty lists, so `{"entities": {}}` where Python gives
  `{"entities": {"person": [], "organization": [], "location": []}}`.
- Multi-label classification returned an empty array when every label sat below
  `cls_threshold`. The reference falls back to the argmax so the head still
  answers the question. With five aspects at threshold 0.5, the top label scores
  0.4647 — Python returns it, we previously returned nothing.

If you treat "no label applies" and "the model declined" as different states,
the second one was silently collapsing them.

### 7. Multi-group structure schemas lose late candidates (known, unfixed)

Recorded rather than fixed, so you can judge whether it affects you.

With one structure group and six words, `gliner2.5-multi-v1` returns `iPhone` at
0.994267, matching Python exactly. With **two** structure groups and nine words,
Python returns `iPhone` at 0.983998 and we return no span for that field, with
candidates past roughly word four missing. Single-group structure schemas are
unaffected and are gated. The signature points at shared-candidate-pool
construction rather than scoring, and it is tracked in
`KNOWN_DIVERGENCES` in the parity test.

This is the one place we know we are behind the reference. It has been in
`KNOWN_DIVERGENCES` since the day it was found rather than being fixed quietly,
so it is visible in our CI output on every run.

---

## On our own reporting during this work

You saw interim notes from us. Two were wrong, and both would have cost you time
if you had acted on them, so they are worth stating plainly:

- We reported a **~10% encoder divergence at hidden=1024** affecting
  `gliner2-large-v1` and `GLiNER2.5-Decide`. There is no such divergence. It was
  a fault in our own comparison harness: we used object-form entity schemas,
  which Rust sorted and Python did not (finding 5). The encoder reproduces the
  committed Python fixture to **3.9e-6**.
- We reported that **`GLiNER2.5-multi-Decide` was failing to extract entities**.
  It was not. Python returns empty entity lists for the same inputs; that model
  is classification-first, like Decide, and its scores top out around 0.38. The
  only genuine defect was output shape (finding 6).

Both were caught by the parity gate, which is the main argument for having built
it before drawing conclusions.

---

## What changed, and how to check it

One PR: [#6](https://github.com/mrorigo/gliner2-candle/pull/6). 9 commits, each
one self-contained.

- **New gate:** `tests/parity_test.rs` + `tests/parity/cases.json` +
  `debug_comparison/parity_reference.py`. Both sides read the same case file, so
  the battery cannot drift between implementations. 19 cases per checkpoint
  across entities, relations, structures and classification; one test per
  checkpoint. All seven in-scope checkpoints pass — 133 case-runs, zero
  failures. `GLiNER2.5-Decide-1B` is out of scope (ModernBERT encoder, which
  this port does not have).

  The suite is `#[ignore]`d because it loads real weights:

  ```sh
  cargo test --test parity_test -- --ignored --nocapture
  ```

- **API change, source-compatible:** `entities_with_descriptions` now takes any
  `(name, description)` iterable. Existing callers passing a `HashMap` still
  compile; you should switch to a `Vec` so the order is deterministic.

- **Everything else** is behavioural parity. No schema fields, no method
  signatures, no output fields changed.

- **Doc corrections:** the README previously stated that entity extraction was
  unavailable on Decide checkpoints. That became false when `CountLSTM` landed,
  and the note was removed in the same PR rather than left to mislead the next
  reader.

---

## What we would like from you

1. **Re-measure your baseline.** Findings 4 and 5 changed encoder inputs and
   schema ordering. The 0.882 on `support-routing` and any confidence you
   recorded from text without terminal punctuation should be re-taken.
2. **Re-read `decisions/0004`.** Decide can extract. The reason for the regex
   path is now a cost and quality trade-off, not a capability limit.
3. **Re-open `decisions/0010` on the merits.** Descriptions now work. We would
   rather you A/B them on your holdout than adopt or skip them on our word.
4. **File findings 2 and 3 as you see fit** — we suggest confirmed-not-a-bug for
   2 (the table above is the citation) and fixed for 3. We are happy to be
   quoted either way.
5. **Tell us if finding 7 affects you.** Multi-group structure schemas are the
   one place we are knowingly behind. If it is not in your decision path, we
   will fix it on our schedule; if it is, we would like to know.

On the relationship, to repeat what you wrote: we are a port, you are the
consumer, and the only thing that helps either of us is the other one noticing
something. Findings 4 and 5 would have quietly degraded your results for as long
as you trusted them, and we only found them because you told us to go and get
parity rather than patch what was reported. Thank you for that.
