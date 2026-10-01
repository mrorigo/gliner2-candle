#!/usr/bin/env python3
"""Emit normalized GLiNER2 outputs for the shared parity battery.

Both sides of `tests/parity_test.rs` read `tests/parity/cases.json`, so the
case list cannot drift between the Rust and Python implementations.

Usage:
    .venv/bin/python debug_comparison/parity_reference.py <model_id> [cases.json]

Prints a single JSON object between the __PARITY_JSON_* markers, keyed by case
id. Each case normalizes to a stable, order-independent shape:

    entities        {label: [{text, confidence}, ...]}
    classifications {task: {labels: [{label, confidence}, ...]}}
    relations       {name:  [{head, tail, confidence}, ...]}
    structure       [ {field: [chunk, ...]}, ... ]

`confidence` is rounded to 6 decimals: enough to catch real divergence, not
enough to trip over float32 noise.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PRECISION = 6


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, PRECISION)
    return value


def _text_of(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("text", ""))
    return ""


def _conf_of(item: Any) -> Any:
    if isinstance(item, dict):
        return _round(item.get("confidence"))
    return None


def _norm_entities(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, items in (raw.get("entities") or {}).items():
        rows = [
            {"text": _text_of(i).strip().lower(), "confidence": _conf_of(i)}
            for i in (items or [])
        ]
        rows.sort(key=lambda r: (r["text"], -(r["confidence"] or 0.0)))
        out[label] = rows
    return out


def _norm_classifications(raw: dict[str, Any], tasks: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task in tasks:
        value = raw.get(task)
        rows: list[dict[str, Any]] = []
        if isinstance(value, dict):
            rows.append({"label": value.get("label"), "confidence": _round(value.get("confidence"))})
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    rows.append(
                        {"label": item.get("label"), "confidence": _round(item.get("confidence"))}
                    )
                else:
                    rows.append({"label": item, "confidence": None})
        elif isinstance(value, str):
            rows.append({"label": value, "confidence": None})
        rows.sort(key=lambda r: str(r["label"]))
        out[task] = {"labels": rows}
    return out


def _norm_relations(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rels = raw.get("relation_extraction") or {}
    for name, items in rels.items():
        rows = []
        for item in items or []:
            head = tail = None
            head_conf = tail_conf = None
            if isinstance(item, dict):
                head, tail = item.get("head"), item.get("tail")
                head_conf = _round(item.get("confidence"))
                tail_conf = _round(item.get("confidence"))
                if isinstance(head, dict):
                    head_conf = _round(head.get("confidence"))
                    head = head.get("text")
                if isinstance(tail, dict):
                    tail_conf = _round(tail.get("confidence"))
                    tail = tail.get("text")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                head, tail = item[0], item[1]
            rows.append(
                {
                    "head": str(head).strip().lower() if head is not None else None,
                    "tail": str(tail).strip().lower() if tail is not None else None,
                    "head_confidence": head_conf,
                    "tail_confidence": tail_conf,
                }
            )
        rows.sort(key=lambda r: (r["head"] or "", r["tail"] or ""))
        out[name] = rows
    return out


# Structures return a long tail of near-zero candidate spans; keep the
# decision-relevant head of each field so the comparison stays stable.
TOP_SPANS_PER_FIELD = 3


def _norm_structure(raw: dict[str, Any], key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for inst in raw.get(key) or []:
        if not isinstance(inst, dict):
            continue
        for field, value in inst.items():
            items = value if isinstance(value, list) else [value]
            rows = [
                {"text": _text_of(i).strip().lower(), "confidence": _conf_of(i)}
                for i in items
                if _text_of(i).strip()
            ]
            rows.sort(key=lambda r: -(r["confidence"] or 0.0))
            out[field] = rows[:TOP_SPANS_PER_FIELD]
    return out


def run_case(model: Any, case: dict[str, Any]) -> Any:
    kind = case["kind"]
    text = case["text"]

    if kind == "entities":
        return _norm_entities(
            model.extract_entities(
                text,
                case["entities"],
                threshold=case.get("threshold", 0.5),
                include_confidence=True,
            )
        )

    if kind == "entities_described":
        schema = {
            "entities": {name: "" for name in case["entities"]},
            "entity_descriptions": case.get("entity_descriptions", {}),
        }
        return _norm_entities(
            model.extract(
                text,
                schema,
                threshold=case.get("threshold", 0.5),
                include_confidence=True,
            )
        )

    if kind == "classifications":
        return _norm_classifications(
            model.classify_text(
                text,
                case["classifications"],
                include_confidence=True,
            ),
            list(case["classifications"].keys()),
        )

    if kind == "relations":
        return _norm_relations(
            model.extract_relations(
                text,
                case["relations"],
                threshold=case.get("threshold", 0.5),
                include_confidence=True,
                include_spans=True,
            )
        )

    if kind == "structure":
        key = next(iter(case["structure"]))
        schema = {"json_structures": [{key: {f: "" for f in case["structure"][key]}}]}
        return _norm_structure(
            model.extract(
                text,
                schema,
                threshold=case.get("threshold", 0.0),
                include_confidence=True,
                include_spans=True,
            ),
            key,
        )

    raise ValueError(f"unknown case kind: {kind}")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: parity_reference.py <model_id> [cases.json]", file=sys.stderr)
        return 2

    model_id = sys.argv[1]
    cases_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("tests/parity/cases.json")
    cases = json.loads(cases_path.read_text())["cases"]

    # AutoExtractor resolves the architecture from the checkpoint's config, so
    # the same harness covers the span checkpoints (gliner2*, Decide) and the
    # boundary ones (gliner2.5*). Plain GLiNER2.from_pretrained cannot load the
    # 2.5 configs at all.
    from gliner2.auto import AutoExtractor

    model = AutoExtractor.from_pretrained(model_id)

    results: dict[str, Any] = {}
    for case in cases:
        try:
            results[case["id"]] = {"ok": True, "value": run_case(model, case)}
        except Exception as exc:  # noqa: BLE001 - report, do not abort the battery
            results[case["id"]] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(f"  {case['id']}: done", file=sys.stderr, flush=True)

    print("__PARITY_JSON_START__")
    print(json.dumps(results, sort_keys=True))
    print("__PARITY_JSON_END__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())