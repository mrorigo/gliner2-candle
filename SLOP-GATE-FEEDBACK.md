# Slop Gate — feedback from an adopter

Notes from wiring Slop Gate into [`mrorigo/gliner2-candle`](https://github.com/mrorigo/gliner2-candle)
(a pure-Rust GLiNER2/GLiNER2.5 port, ~7k lines of `src/`, no Python runtime) and
then actually using it to drive a refactor. Repro commands are at the bottom so
every claim is checkable.

Short version: **the gate works well for the regression it was built for, but its
near-clone signal has inverted precision.** On this repo it reported 34 clones,
essentially all of them unavoidable and harmless, while the one piece of real
duplication — byte-identical, cross-function, production — was not reported at
all. That's a recall problem, not a tuning problem, and it points at function
granularity rather than thresholds.

---

## What worked, and should not change

- **`check` attributing findings only to high-complexity functions intersecting added head lines.** This is the right design. It flagged my own new code immediately and never nagged about the 64 pre-existing high-CC functions. Without it, a first PR in a legacy repo would be unreadable.
- **`history --top-contributors`.** The single most actionable output. It named the three worst functions with CC and mass, which is what made a refactor plan concrete instead of vibes:

  ```
  BoundaryModel::score_spans          src/model/boundary.rs:1755   CC 101  mass 1717
  extract_sample                      src/inference/boundary.rs:122 CC 61  mass 1118
  GLiNER2::extract_entities_from_output src/inference/engine.rs:1107 CC 59  mass 948
  ```
- **`function-mass` as a rule.** It caught a real regression in my own diff (`rebuild_model` +43.56 mass, limit 20.0) that I would otherwise have shipped. This is the rule with the best signal-to-noise here, and it's underused in my own workflow.
- **Status code semantics.** `0`/`1`/`2` with `2` explicitly meaning operational failure let an agent branch correctly. The skill text is clear that warnings are not failures and that `2` must not be reclassified.

---

## Finding 1 — near-clone precision is poor, and all 34 findings are noise

Every one of the 34 findings is code that *should* look alike. I triaged all 15 clone families by hand:

| Family | Mass | What it actually is |
|---|---|---|
| `deberta_v3.rs:849` | 155.98 | c2p and p2c gather kernels — the duplication **is** the ~2x rel-bias speedup |
| `constraints.rs:20` | 34.64 | `k_and` / `k_or` — Kleene-3 conjunction and disjunction. Parallel structure is the specification |
| `schema/builder.rs:120` | 26.22 | `entities` / `relations` loops — 6 lines; unifying costs more than it saves |
| `error.rs:239` | 12.80 | struct-variant constructors |
| `types.rs:431`, `classifier.rs:45`, `count_embed.rs:249`, `candle_encoder.rs:574`, `boundary.rs:2745` | ~6–10 ea | `new()` / `from_config()` / config-mapping boilerplate |
| `preprocessed.rs:417`, `:486` | 6.93 ea | one-line accessors, reported at 100% similarity |
| `constraints.rs:695` | 6.32 | `k_and_basic` / `k_or_basic` — parallel table tests, correct as written |
| `collator.rs:838`, `gliner25_test.rs:12`, `gliner25_boundary_test.rs:9` | ~7 ea | test duplication |

There is no plausible refactor here that isn't a regression in readability. The only mechanical "fix" for most of it is a macro, which is worse code in every case — and would arguably be *gaming the metric* rather than satisfying it.

**Observation on the signal:** the minimum reported similarity is `0.868` against a `0.85` threshold. Every finding is barely over the line, which is what you'd expect if the threshold is below the natural similarity floor of "two idiomatic Rust functions of similar shape."

## Finding 2 — the real duplication was missed, and the cause is granularity

`extract_relations_from_output` and `extract_structures_from_output` each contained a **byte-identical 30-line block**: take `count_embed`, run it, then flatten `struct_proj` / `span_rep` / `span_mask` / `spans_idx` into contiguous buffers with an early return on failure. `extract_entities_from_output` inlined a near-copy with a different empty value.

Not reported. Nothing near the threshold.

The cause is that similarity is computed **function-to-function**, and these functions are large:

```
extract_relations_from_output    ~187 lines
extract_structures_from_output   ~186 lines
extract_entities_from_output    ~259 lines
shared block                       30 lines
```

A 30-line island in a 187-line function puts whole-function similarity at roughly 0.16 by line count — nowhere near `0.85`. So:

> **Precision and recall are inversely correlated with function size.** Small
> functions can't dilute anything, so their trivial similarities surface and
> dominate the report. Large functions hide real duplication, so it never
> appears.

That's why the report reads as 34 false positives and zero true positives. The
one-line accessors are reported precisely *because* they're one line.

This is the case the rule most needs to catch, and the risk profile is asymmetric: a byte-identical pair silently diverges when someone fixes one copy. The relations and structures paths have only stayed in sync because nobody has had cause to touch either.

**Suggestion:** add block-level detection independent of function boundaries — a sliding window over statement/token sequences within a function, compared across functions. A 30-line window inside two 187-line functions is a 1.0 match and unambiguous. `minimum_sloc`/`minimum_tokens` already express the right grain for this; the enclosing function is what's wrong.

**Suggestion:** weight or tag findings by copy relationship, since the risk gradient is steep:
- cross-function, same file, production → highest (the case above)
- cross-file, production → high
- test → test → low
- single small function vs a family of similar constructors → lowest

A `kind` hint (`accessor` / `constructor` / `test-table`) would also let an agent triage without reading 34 sites. Right now the only discrimination is `finding_kind: family-summary` vs pairwise.

## Finding 3 — erosion is a ratio, and the denominator is easy to move without improving anything

Deleting 437 lines of Rust test code moved erosion from 75.89% → 76.06% (**+0.17%**). The deleted code contained no high-complexity functions, so `total_mass` shrank while the CC≥10 mass stayed put and the ratio rose. No new complexity was introduced; the number went up because I removed code.

Meanwhile the real dedup moved it the right way: extracting the shared plumbing took it 76.06% → 75.48% (**−0.58%**).

Both effects are correct arithmetic and both are uninformative on their own. A +0.17% from a deletion and a −0.58% from a refactor aren't comparable quantities.

**Suggestion:** report the ratio *and* absolute high-complexity mass, and prefer absolute deltas for the `delta_limit` comparison. A ratio makes "delete tests" and "add complexity" look like the same class of change.

## Finding 4 — `complexity_cutoff = 10` saturates the metric

At cutoff 10 this repo has **69 high-complexity functions** and erosion 75.48%. For tensor/numerical code — long dispatch chains, per-task-type decoding, shape validation — CC 10 is routine, so "most mass sits in functions above 10" is the steady state and carries no signal.

A metric that reads 76% in steady state can't distinguish a well-maintained repo from a deteriorating one; it only responds to large moves, which is why the real dedup only bought 0.58%.

**Suggestion:** consider a percentile-relative cutoff, or report the CC distribution rather than a single ratio, so the number is interpretable per-repo. Alternatively raise the default and let `erosion_limit` be tuned per project — a low default that's always tripped trains agents to ignore the rule, which is the same failure mode as a lint everyone disables.

## Finding 5 — artifact/version mismatch presents as a consumer config error

This cost real debugging time and is worth hardening.

`config.json` was fine, `.slop-gate.toml` was fine, the workflow was verbatim from the reference. The failure was:

```
slop-gate check: invalid artifact summary: does not match the artifact function facts
```

Root cause: artifacts are version-specific. The workflow downloads from
`releases/latest`, which was `v0.4.0`, whose artifact validation is broken.
`0.4.1` was on crates.io but had no GitHub Release, so CI kept getting the
broken binary. A `0.4.0`-built index also fails under `0.4.1` and vice versa.

Three asks:

1. **Say what mismatched and which versions.** "does not match the artifact function facts" is unactionable. The tool knows both versions — something like `artifact built by 0.4.0, this is 0.4.1; rebuild the index` would have identified the cause immediately.
2. **Index artifacts should carry an explicit `tool_version` + policy fingerprint field**, and `check` should name the mismatch rather than infer it.
3. **The reference workflow should print the resolved version to the log**, e.g. `slop-gate 0.4.0` before the run. When `latest` drifts, the first line of diagnosis should not require an API call to the releases endpoint.

I did not set `SLOP_GATE_RELEASE: latest` on my own initiative — it is the shipped default, which is correct for a template. But given that `latest` silently served a broken binary, the template's pin comment ("Pin SLOP_GATE_RELEASE to a tag when reproducible tool selection matters") may deserve to be a stronger default or at least a louder note in the install step. I pinned after hitting this.

---

## Ranked asks

1. **Block-level (function-independent) duplicate detection.** Highest value: this is the entire recall gap, and the failure mode is silent divergence.
2. **Better status-2 diagnostics for artifact/version mismatch.** Cheap, and converts a confusing failure into a one-line fix.
3. **Absolute mass alongside the erosion ratio**, so deletions and complexity additions aren't compared as if equivalent.
4. **Risk-weighted or `kind`-tagged near-clone findings**, so an agent can triage 34 findings without opening 34 files.
5. **Reconsider the `complexity_cutoff` default** (or add a distribution view) so the metric is not saturated at rest.

Not asking for: better thresholds. Tuning `similarity_threshold` or `minimum_sloc` would change which findings appear without touching the precision/recall inversion in Finding 2, and would likely just shuffle noise.

---

## Repro

On `mrorigo/gliner2-candle` @ `chore/tier-a-dedup`, against `main` as base:

```sh
cargo install slop-gate --locked
BASE="$(git rev-parse origin/main)"
mkdir -p .slop-gate

slop-gate index --ref "$BASE" --output .slop-gate/main.json
slop-gate check --base "$BASE" --head HEAD --index .slop-gate/main.json --format human

# the 34 findings, all near-clone, all hand-triaged in Finding 1
slop-gate scan --ref HEAD --format human

# 69 high-CC functions, erosion 75.48%, top contributors from Finding "what worked"
slop-gate history --ref HEAD --count 1 --format json
```

The missed duplication, for comparison:

```sh
diff <(sed -n '1607,1636p' src/inference/engine.rs) \
     <(sed -n '1804,1833p' src/inference/engine.rs) && echo IDENTICAL
# IDENTICAL   (30 lines; functions are ~187 and ~186 lines)
```

Versions in use: `slop-gate 0.4.1` (locally and in CI after the `v0.4.1` release). Reproduced the Finding 5 failure with the `v0.4.0` binary.
