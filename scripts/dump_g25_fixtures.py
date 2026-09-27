#!/usr/bin/env python
"""Regenerate the committed GLiNER2.5 numeric parity fixtures.

Writes `tests/fixtures/g25/{encoder,pool}.json` from the upstream Python
reference implementation (fastino-ai/GLiNER2, git main). These fixtures are the
permanent parity gate asserted by `test_full_matrix_parity`; they are committed
so CI can check numeric parity without a Python toolchain.

Usage:
    uv venv --python 3.12 .venv
    VIRTUAL_ENV=.venv uv pip install -r scripts/requirements-reference.txt
    .venv/bin/python scripts/dump_g25_fixtures.py

See scripts/requirements-reference.txt for why the version pins matter.
"""

import json
import pathlib
import sys

import torch

# Must match tests/gliner25_boundary_test.rs::test_full_matrix_parity exactly.
MODEL_ID = "fastino/gliner2.5-small-v1"
TEXT = "Apple CEO Tim Cook announced the new iPhone 15 in Cupertino."
# NOTE: the reference processor wants entity names as dict KEYS; the Rust
# collator also accepts a bare list. Values are gold-span placeholders and are
# irrelevant here because we build no targets.
SCHEMA = {
    "entities": {
        "person": [0, []],
        "organization": [0, []],
        "location": [0, []],
    }
}

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "tests" / "fixtures" / "g25"

# Mirrors MASK_LOGIT in gliner2/models/boundary/constants.py. The Rust test
# treats |logit| > 100 as a padding slot and skips it.
MASK_LOGIT = -10000.0


def nested(t):
    """Serialize a tensor as a plain nested list of float32.

    The committed fixtures are bare nested lists, not {shape, data} envelopes;
    test_full_matrix_parity flattens them recursively.
    """
    return t.detach().to(torch.float32).cpu().contiguous().tolist()


def main():
    from gliner2 import AutoExtractor

    print(f"loading {MODEL_ID} ...", file=sys.stderr)
    model = AutoExtractor.from_pretrained(MODEL_ID)
    model.eval()
    inner = getattr(model, "model", model)
    proc = inner.processor
    head = inner.boundary_head

    # --- preprocessing, matching the Rust collator's schema shape -----------
    batch = proc.collate_fn_inference(
        [(TEXT, SCHEMA)], architecture="boundary", build_targets=False
    )
    with torch.no_grad():
        # --- encoder stages -------------------------------------------------
        # hidden_states[0] is the embedding output (input to layer 0), matching
        # the Rust forward_debug() stages[0]; last_hidden_state matches stages[N].
        enc_out = inner.encoder(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            output_hidden_states=True,
        )
        l0_in = enc_out.hidden_states[0]
        final = enc_out.last_hidden_state

        # --- word / query states -------------------------------------------
        encoded = inner.encode(batch)
        text_states = encoded.text_states
        text_mask = encoded.text_mask
        query_states = encoded.query_states
        query_mask = encoded.query_mask

        # --- boundary head: pool candidates + pair logits --------------------
        # Mirrors BoundaryHead.forward() for candidate_pool == "shared".
        enc = head.boundary_encoder(text_states, text_mask)
        marg = head.boundary_query_head(
            enc.states,
            enc.mask,
            text_states,
            text_mask,
            query_states,
            query_mask,
        )
        pooled = head.shared_pool_builder(
            enc.states,
            enc.mask,
            query_mask,
            marg.start_logits,
            marg.end_logits,
            return_stats=head.collect_diagnostics,
        )
        inside_prefix = marg.inside_prefix if head.use_inside_evidence else None
        pooled_logits, _ = head.shared_pool_scorer(
            enc.states,
            query_states,
            query_mask,
            pooled,
            marg.start_logits,
            marg.end_logits,
            inside_prefix,
            encoded.text_lengths,
            text_states,
            text_mask,
            inside_prefix_mean=marg.inside_prefix_mean,
        )

    # Batch dim is 1 throughout; candidates are shared across queries, so the
    # index matrix is broadcast to one row per query (matching
    # pooled.indices.unsqueeze(1).expand(b, q, c, 2)).
    c = pooled.indices.shape[1]
    q = query_states.shape[1]
    indices = pooled.indices[0].tolist()  # (c, 2)
    valid = pooled.mask[0]  # (c,)
    # shared_pool_scorer returns (b, c, q); the model transposes to (b, q, c) as
    # `pair_logits` (see BoundaryHead.forward). The fixture is (q, c).
    logits = pooled_logits[0].transpose(0, 1).to(torch.float32).contiguous()
    logits = torch.where(
        valid.unsqueeze(0).expand_as(logits),
        logits,
        torch.full_like(logits, MASK_LOGIT),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    encoder_json = {
        "l0_in": nested(l0_in[0]),
        "final": nested(final[0]),
    }
    pool_json = {
        "indices": [indices for _ in range(q)],
        "logits": logits.tolist(),
    }

    (OUT_DIR / "encoder.json").write_text(json.dumps(encoder_json))
    (OUT_DIR / "pool.json").write_text(json.dumps(pool_json))

    print(f"wrote {OUT_DIR/'encoder.json'}", file=sys.stderr)
    print(
        f"  l0_in {tuple(l0_in.shape[1:])}  final {tuple(final.shape[1:])}",
        file=sys.stderr,
    )
    print(f"wrote {OUT_DIR/'pool.json'}", file=sys.stderr)
    print(f"  indices (q={q}, c={c})  logits (q={q}, c={c})", file=sys.stderr)


if __name__ == "__main__":
    main()
