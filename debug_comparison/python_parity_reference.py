#!/usr/bin/env python3
"""
Offline Python reference parity output.

Usage:
  .venv/bin/python debug_comparison/python_parity_reference.py /abs/path/to/snapshot
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

from gliner2 import GLiNER2


def _load_model_offline(snapshot: str):
    """Load GLiNER2 from a local snapshot directory, working around
    transformers 5.x incompatibility with file-path config loading."""
    from pathlib import Path
    from safetensors.torch import load_file
    from gliner2.model import ExtractorConfig
    from transformers import AutoConfig, AutoTokenizer

    snap = Path(snapshot)

    # If config.json exists, use from_pretrained path.
    if (snap / "config.json").exists():
        return GLiNER2.from_pretrained(str(snap))

    # Otherwise construct the model from known config + local weights.
    config = ExtractorConfig(
        hidden_size=768,
        vocab_size=128011,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    )
    encoder_config = AutoConfig.from_pretrained(
        "microsoft/deberta-v3-base"
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    model = GLiNER2(config, encoder_config=encoder_config, tokenizer=tokenizer)

    model_path = snap / "model.safetensors"
    if model_path.exists():
        state_dict = load_file(str(model_path))
    else:
        import torch
        state_dict = torch.load(str(snap / "pytorch_model.bin"), map_location="cpu")

    model.load_state_dict(state_dict, strict=False)
    return model


def _norm_text(v: Any) -> str:
    return str(v).strip().lower()

def _split_structure_chunks(v: Any) -> list[str]:
    text = _norm_text(v)
    if not text:
        return []
    return re.findall(r"[a-z0-9]+", text)


def normalize_entities(result: dict[str, Any]) -> dict[str, list[str]]:
    entities = result.get("entities", {})
    out: dict[str, list[str]] = {}
    if not isinstance(entities, dict):
        return out
    for key, values in entities.items():
        if isinstance(values, list):
            out[key] = sorted({_norm_text(v.get("text") if isinstance(v, dict) else v) for v in values if v})
        else:
            out[key] = []
    return out


def normalize_relations(result: dict[str, Any]) -> dict[str, list[str]]:
    rel = result.get("relation_extraction", {})
    out: dict[str, list[str]] = {}
    if not isinstance(rel, dict):
        return out
    for name, values in rel.items():
        pairs: list[str] = []
        if isinstance(values, list):
            for item in values:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    pairs.append(f"{_norm_text(item[0])}|||{_norm_text(item[1])}")
                elif isinstance(item, dict):
                    head = item.get("head")
                    tail = item.get("tail")
                    if isinstance(head, dict):
                        head = head.get("text")
                    if isinstance(tail, dict):
                        tail = tail.get("text")
                    if head and tail:
                        pairs.append(f"{_norm_text(head)}|||{_norm_text(tail)}")
        out[name] = sorted(set(pairs))
    return out


def normalize_structures(result: dict[str, Any], key: str) -> list[dict[str, Any]]:
    vals = result.get(key, [])
    out: list[dict[str, Any]] = []
    if not isinstance(vals, list):
        return out
    for inst in vals:
        if not isinstance(inst, dict):
            continue
        norm_inst: dict[str, Any] = {}
        for field, value in inst.items():
            if isinstance(value, list):
                texts = []
                for v in value:
                    if isinstance(v, dict):
                        v = v.get("text")
                    if v is not None:
                        texts.extend(_split_structure_chunks(v))
                norm_inst[field] = sorted(set(texts))
            elif isinstance(value, dict):
                text = value.get("text")
                norm_inst[field] = _norm_text(text) if text is not None else None
            elif value is None:
                norm_inst[field] = None
            else:
                norm_inst[field] = _norm_text(value)
        out.append(norm_inst)
    out.sort(key=lambda x: json.dumps(x, sort_keys=True))
    return out


def normalize_classification(result: dict[str, Any], task: str) -> Any:
    value = result.get(task)
    if isinstance(value, str):
        return [_norm_text(value)]
    if isinstance(value, dict):
        label = value.get("label")
        return [_norm_text(label)] if label is not None else []
    if isinstance(value, list):
        labels = []
        for item in value:
            if isinstance(item, dict):
                item = item.get("label")
            if item is not None:
                labels.append(_norm_text(item))
        return sorted(set(labels))
    return []


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python_parity_reference.py <snapshot_path>", file=sys.stderr)
        return 2

    snapshot = sys.argv[1]
    model = _load_model_offline(snapshot)

    fixtures = [
        {
            "id": "basic_apple",
            "entities_text": "Apple CEO Tim Cook announced iPhone in Cupertino, California.",
            "class_text": "I love this product.",
            "relations_text": "Apple CEO Tim Cook announced iPhone in Cupertino, California.",
            "structure_text": "Apple announced iPhone in Cupertino, California.",
            "structure_key": "product_info",
            "structure_threshold": 0.0,
            "entity_threshold": 0.5,
            "relation_threshold": 0.5,
            "class_multi_label": False,
        },
        {
            "id": "microsoft_founder",
            "entities_text": "Microsoft was founded by Bill Gates in Albuquerque.",
            "class_text": "Microsoft was founded by Bill Gates in Albuquerque.",
            "relations_text": "Microsoft was founded by Bill Gates in Albuquerque.",
            "structure_text": "Microsoft was founded by Bill Gates in Albuquerque.",
            "structure_key": "profile",
            "structure_threshold": 0.0,
            "entity_threshold": 0.5,
            "relation_threshold": 0.5,
            "class_multi_label": False,
        },
        {
            "id": "tesla_engineer",
            "entities_text": "Tesla engineer Elon Musk works in Austin, Texas.",
            "class_text": "Tesla engineer Elon Musk works in Austin, Texas.",
            "relations_text": "Tesla engineer Elon Musk works in Austin, Texas.",
            "structure_text": "Tesla engineer Elon Musk works in Austin, Texas.",
            "structure_key": "team_profile",
            "structure_threshold": 0.0,
            "entity_threshold": 0.5,
            "relation_threshold": 0.5,
            "class_multi_label": False,
        },
        {
            "id": "threshold_edge_049",
            "entities_text": "Apple CEO Tim Cook announced iPhone in Cupertino, California.",
            "class_text": "I love this product.",
            "relations_text": "Apple CEO Tim Cook announced iPhone in Cupertino, California.",
            "structure_text": "Apple announced iPhone in Cupertino, California.",
            "structure_key": "threshold_case",
            "structure_threshold": 0.49,
            "entity_threshold": 0.49,
            "relation_threshold": 0.49,
            "class_multi_label": False,
        },
        {
            "id": "threshold_edge_050",
            "entities_text": "Apple CEO Tim Cook announced iPhone in Cupertino, California.",
            "class_text": "I love this product.",
            "relations_text": "Apple CEO Tim Cook announced iPhone in Cupertino, California.",
            "structure_text": "Apple announced iPhone in Cupertino, California.",
            "structure_key": "threshold_case",
            "structure_threshold": 0.50,
            "entity_threshold": 0.50,
            "relation_threshold": 0.50,
            "class_multi_label": False,
        },
        {
            "id": "threshold_edge_051",
            "entities_text": "Apple CEO Tim Cook announced iPhone in Cupertino, California.",
            "class_text": "I love this product.",
            "relations_text": "Apple CEO Tim Cook announced iPhone in Cupertino, California.",
            "structure_text": "Apple announced iPhone in Cupertino, California.",
            "structure_key": "threshold_case",
            "structure_threshold": 0.51,
            "entity_threshold": 0.51,
            "relation_threshold": 0.51,
            "class_multi_label": False,
        },
        {
            "id": "multilabel_mixed_signal",
            "entities_text": "The app is polished but expensive and occasionally crashes.",
            "class_text": "The app is polished but expensive and occasionally crashes.",
            "relations_text": "The app is polished but expensive and occasionally crashes.",
            "structure_text": "The app is polished but expensive and occasionally crashes.",
            "structure_key": "review_profile",
            "structure_threshold": 0.5,
            "entity_threshold": 0.5,
            "relation_threshold": 0.5,
            "class_multi_label": True,
            "classification_only": True,
        },
        {
            "id": "long_context_maxlen_12",
            "entities_text": (
                "Apple CEO Tim Cook works for Apple while teams discuss roadmap "
                "and planning details filler filler filler filler filler filler filler filler filler filler "
                "filler filler filler filler filler filler filler filler filler filler."
            ),
            "class_text": (
                "Apple CEO Tim Cook works for Apple while teams discuss roadmap "
                "and planning details filler filler filler filler filler filler filler filler filler filler "
                "filler filler filler filler filler filler filler filler filler filler."
            ),
            "relations_text": (
                "Apple CEO Tim Cook works for Apple while teams discuss roadmap "
                "and planning details filler filler filler filler filler filler filler filler filler filler "
                "filler filler filler filler filler filler filler filler filler filler."
            ),
            "structure_text": (
                "Apple CEO Tim Cook works for Apple while teams discuss roadmap "
                "and planning details filler filler filler filler filler filler filler filler filler filler "
                "filler filler filler filler filler filler filler filler filler filler."
            ),
            "structure_key": "long_context",
            "structure_threshold": 0.0,
            "entity_threshold": 0.5,
            "relation_threshold": 0.5,
            "class_multi_label": False,
            "max_len": 12,
            "classification_only": False,
        },
        {
            "id": "long_context_maxlen_24",
            "entities_text": (
                "Apple CEO Tim Cook works for Apple while teams discuss roadmap "
                "and planning details filler filler filler filler filler filler filler filler filler filler "
                "filler filler filler filler filler filler filler filler filler filler."
            ),
            "class_text": (
                "Apple CEO Tim Cook works for Apple while teams discuss roadmap "
                "and planning details filler filler filler filler filler filler filler filler filler filler "
                "filler filler filler filler filler filler filler filler filler filler."
            ),
            "relations_text": (
                "Apple CEO Tim Cook works for Apple while teams discuss roadmap "
                "and planning details filler filler filler filler filler filler filler filler filler filler "
                "filler filler filler filler filler filler filler filler filler filler."
            ),
            "structure_text": (
                "Apple CEO Tim Cook works for Apple while teams discuss roadmap "
                "and planning details filler filler filler filler filler filler filler filler filler filler "
                "filler filler filler filler filler filler filler filler filler filler."
            ),
            "structure_key": "long_context",
            "structure_threshold": 0.0,
            "entity_threshold": 0.5,
            "relation_threshold": 0.5,
            "class_multi_label": False,
            "max_len": 24,
            "classification_only": False,
        },
    ]

    norm_fixtures = []
    for fixture in fixtures:
        entities_raw = model.extract_entities(
            fixture["entities_text"],
            ["person", "organization", "location"],
            threshold=fixture.get("entity_threshold", 0.5),
            include_confidence=False,
            include_spans=False,
            max_len=fixture.get("max_len"),
        )
        class_raw = model.classify_text(
            fixture["class_text"],
            {
                "sentiment": {
                    "labels": ["positive", "negative", "neutral"],
                    "multi_label": fixture.get("class_multi_label", False),
                }
            },
            include_confidence=False,
            max_len=fixture.get("max_len"),
        )
        rel_raw = model.extract_relations(
            fixture["relations_text"],
            ["works_for"],
            threshold=fixture.get("relation_threshold", 0.5),
            include_confidence=False,
            include_spans=False,
            max_len=fixture.get("max_len"),
        )
        struct_schema = {
            "entities": [],
            "classifications": [],
            "relations": [],
            "json_structures": [{fixture["structure_key"]: {"name": "", "company": ""}}],
        }
        struct_raw = model.extract(
            fixture["structure_text"],
            struct_schema,
            threshold=fixture.get("structure_threshold", 0.5),
            include_confidence=False,
            include_spans=False,
            max_len=fixture.get("max_len"),
        )

        if fixture.get("classification_only", False):
            entities_norm = {"person": [], "organization": [], "location": []}
            relations_norm = {"works_for": []}
            structures_norm = []
        else:
            entities_norm = normalize_entities(entities_raw)
            relations_norm = normalize_relations(rel_raw)
            structures_norm = normalize_structures(struct_raw, fixture["structure_key"])

        norm_fixtures.append(
            {
                "id": fixture["id"],
                "entities": entities_norm,
                "classification": normalize_classification(class_raw, "sentiment"),
                "relations": relations_norm,
                "structures": structures_norm,
            }
        )

    norm_fixtures.sort(key=lambda x: x.get("id", ""))
    result = {"fixtures": norm_fixtures}

    print("__PARITY_JSON_START__")
    print(json.dumps(result, sort_keys=True))
    print("__PARITY_JSON_END__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
