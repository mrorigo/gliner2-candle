use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use gliner2_rs::schema::builder::SchemaBuilder;
use gliner2_rs::{ExtractorConfig, GLiNER2};
use serde_json::Value as JsonValue;

fn find_snapshot_dir() -> PathBuf {
    let root = if let Some(explicit) = std::env::var_os("GLINER2_SNAPSHOT_DIR") {
        PathBuf::from(explicit)
    } else if let Some(hf_home) = std::env::var_os("HF_HOME") {
        PathBuf::from(hf_home)
            .join("hub")
            .join("models--fastino--gliner2-base-v1")
            .join("snapshots")
    } else {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".cache")
            .join("huggingface")
            .join("hub")
            .join("models--fastino--gliner2-base-v1")
            .join("snapshots")
    };

    assert!(
        root.exists(),
        "snapshot directory not found: {}. set GLINER2_SNAPSHOT_DIR to override",
        root.display()
    );

    let mut candidates = Vec::new();
    for entry in fs::read_dir(root).expect("Failed to read snapshots dir") {
        let entry = entry.expect("Failed to read snapshot entry");
        let path = entry.path();
        let model = path.join("model.safetensors");
        let tok = path.join("tokenizer.json");
        if model.exists() && tok.exists() {
            candidates.push(path);
        }
    }
    candidates.sort();
    candidates.pop().expect("No valid snapshot found")
}

fn norm_text(v: &str) -> String {
    v.trim().to_lowercase()
}

fn split_structure_chunks(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            cur.push(ch.to_ascii_lowercase());
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn normalize_entities(result: &JsonValue) -> JsonValue {
    let mut out = serde_json::Map::new();
    let entities = result
        .get("entities")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    for (k, vals) in entities {
        let mut texts = Vec::<String>::new();
        if let Some(arr) = vals.as_array() {
            for v in arr {
                if let Some(s) = v.as_str() {
                    texts.push(norm_text(s));
                } else if let Some(s) = v.get("text").and_then(|x| x.as_str()) {
                    texts.push(norm_text(s));
                }
            }
        }
        texts.sort();
        texts.dedup();
        out.insert(
            k,
            JsonValue::Array(texts.into_iter().map(JsonValue::String).collect()),
        );
    }
    JsonValue::Object(out)
}

fn normalize_relations(result: &JsonValue) -> JsonValue {
    let mut out = serde_json::Map::new();
    let rels = result
        .get("relation_extraction")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    for (name, vals) in rels {
        let mut pairs = Vec::<String>::new();
        if let Some(arr) = vals.as_array() {
            for item in arr {
                if let Some(obj) = item.as_object() {
                    let mut head = obj.get("head");
                    let mut tail = obj.get("tail");
                    if let Some(h) = head.and_then(|v| v.get("text")) {
                        head = Some(h);
                    }
                    if let Some(t) = tail.and_then(|v| v.get("text")) {
                        tail = Some(t);
                    }
                    if let (Some(hs), Some(ts)) =
                        (head.and_then(|v| v.as_str()), tail.and_then(|v| v.as_str()))
                    {
                        pairs.push(format!("{}|||{}", norm_text(hs), norm_text(ts)));
                    }
                } else if let Some(arr2) = item.as_array()
                    && arr2.len() >= 2
                    && let (Some(h), Some(t)) = (arr2[0].as_str(), arr2[1].as_str())
                {
                    pairs.push(format!("{}|||{}", norm_text(h), norm_text(t)));
                }
            }
        }
        pairs.sort();
        pairs.dedup();
        out.insert(
            name,
            JsonValue::Array(pairs.into_iter().map(JsonValue::String).collect()),
        );
    }
    JsonValue::Object(out)
}

fn normalize_structures(result: &JsonValue, key: &str) -> JsonValue {
    let mut insts = Vec::<JsonValue>::new();
    if let Some(arr) = result.get(key).and_then(|v| v.as_array()) {
        for inst in arr {
            if let Some(obj) = inst.as_object() {
                let mut norm = serde_json::Map::new();
                for (field, value) in obj {
                    let nv = if let Some(a) = value.as_array() {
                        let mut texts = Vec::<String>::new();
                        for v in a {
                            if let Some(s) = v.as_str() {
                                texts.extend(split_structure_chunks(s));
                            } else if let Some(s) = v.get("text").and_then(|x| x.as_str()) {
                                texts.extend(split_structure_chunks(s));
                            }
                        }
                        texts.sort();
                        texts.dedup();
                        JsonValue::Array(texts.into_iter().map(JsonValue::String).collect())
                    } else if let Some(s) = value.as_str() {
                        JsonValue::String(norm_text(s))
                    } else if let Some(s) = value.get("text").and_then(|x| x.as_str()) {
                        JsonValue::String(norm_text(s))
                    } else {
                        JsonValue::Null
                    };
                    norm.insert(field.clone(), nv);
                }
                insts.push(JsonValue::Object(norm));
            }
        }
    }
    insts.sort_by_key(|v| v.to_string());
    JsonValue::Array(insts)
}

fn normalize_classification(result: &JsonValue, task: &str) -> JsonValue {
    let v = result.get(task).cloned().unwrap_or(JsonValue::Null);
    if let Some(s) = v.as_str() {
        return JsonValue::Array(vec![JsonValue::String(norm_text(s))]);
    }
    if let Some(obj) = v.as_object()
        && let Some(lbl) = obj.get("label").and_then(|x| x.as_str())
    {
        return JsonValue::Array(vec![JsonValue::String(norm_text(lbl))]);
    }
    if let Some(arr) = v.as_array() {
        let mut labels = Vec::<String>::new();
        for item in arr {
            if let Some(s) = item.as_str() {
                labels.push(norm_text(s));
            } else if let Some(s) = item.get("label").and_then(|x| x.as_str()) {
                labels.push(norm_text(s));
            }
        }
        labels.sort();
        labels.dedup();
        return JsonValue::Array(labels.into_iter().map(JsonValue::String).collect());
    }
    JsonValue::Array(Vec::new())
}

#[test]
#[ignore = "requires specific Python gliner2 version; use test_task_output_parity for in-process parity"]
fn test_python_reference_parity_fixtures() {
    let snapshot = find_snapshot_dir();
    let script = Path::new("debug_comparison/python_parity_reference.py");
    let py = Path::new(".venv/bin/python");
    assert!(script.exists(), "missing parity script");
    assert!(py.exists(), "missing python venv");

    let output = Command::new(py)
        .arg(script)
        .arg(snapshot.to_string_lossy().to_string())
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("failed to run python parity reference");
    assert!(
        output.status.success(),
        "python reference failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let start = stdout
        .find("__PARITY_JSON_START__")
        .expect("missing start marker");
    let end = stdout
        .find("__PARITY_JSON_END__")
        .expect("missing end marker");
    let json_text = stdout[start + "__PARITY_JSON_START__".len()..end].trim();
    let py_norm: JsonValue = serde_json::from_str(json_text).expect("invalid python json");

    let tokenizer_path = snapshot.join("tokenizer.json");
    let model_path = snapshot.join("model.safetensors");
    let config = ExtractorConfig::builder()
        .model_name("fastino/gliner2-base-v1")
        .tokenizer_path(tokenizer_path.clone())
        .hidden_size(768)
        .vocab_size(128011)
        .num_hidden_layers(12)
        .num_attention_heads(12)
        .intermediate_size(3072)
        .build()
        .expect("failed to build config");

    let mut engine = GLiNER2::new(&config).expect("failed to init rust engine");
    engine
        .load_weights(&model_path)
        .expect("failed to load rust weights");

    let fixtures = vec![
        (
            "basic_apple",
            "Apple CEO Tim Cook announced iPhone in Cupertino, California.",
            "I love this product.",
            "Apple CEO Tim Cook announced iPhone in Cupertino, California.",
            "Apple announced iPhone in Cupertino, California.",
            "product_info",
            false,
            false,
            None,
            0.5f32,
            0.5f32,
            0.0f32,
        ),
        (
            "microsoft_founder",
            "Microsoft was founded by Bill Gates in Albuquerque.",
            "Microsoft was founded by Bill Gates in Albuquerque.",
            "Microsoft was founded by Bill Gates in Albuquerque.",
            "Microsoft was founded by Bill Gates in Albuquerque.",
            "profile",
            false,
            false,
            None,
            0.5f32,
            0.5f32,
            0.0f32,
        ),
        (
            "tesla_engineer",
            "Tesla engineer Elon Musk works in Austin, Texas.",
            "Tesla engineer Elon Musk works in Austin, Texas.",
            "Tesla engineer Elon Musk works in Austin, Texas.",
            "Tesla engineer Elon Musk works in Austin, Texas.",
            "team_profile",
            false,
            false,
            None,
            0.5f32,
            0.5f32,
            0.0f32,
        ),
        (
            "threshold_edge_049",
            "Apple CEO Tim Cook announced iPhone in Cupertino, California.",
            "I love this product.",
            "Apple CEO Tim Cook announced iPhone in Cupertino, California.",
            "Apple announced iPhone in Cupertino, California.",
            "threshold_case",
            false,
            false,
            None,
            0.49f32,
            0.49f32,
            0.49f32,
        ),
        (
            "threshold_edge_050",
            "Apple CEO Tim Cook announced iPhone in Cupertino, California.",
            "I love this product.",
            "Apple CEO Tim Cook announced iPhone in Cupertino, California.",
            "Apple announced iPhone in Cupertino, California.",
            "threshold_case",
            false,
            false,
            None,
            0.50f32,
            0.50f32,
            0.50f32,
        ),
        (
            "threshold_edge_051",
            "Apple CEO Tim Cook announced iPhone in Cupertino, California.",
            "I love this product.",
            "Apple CEO Tim Cook announced iPhone in Cupertino, California.",
            "Apple announced iPhone in Cupertino, California.",
            "threshold_case",
            false,
            false,
            None,
            0.51f32,
            0.51f32,
            0.51f32,
        ),
        (
            "multilabel_mixed_signal",
            "The app is polished but expensive and occasionally crashes.",
            "The app is polished but expensive and occasionally crashes.",
            "The app is polished but expensive and occasionally crashes.",
            "The app is polished but expensive and occasionally crashes.",
            "review_profile",
            true,
            true,
            None,
            0.5f32,
            0.5f32,
            0.5f32,
        ),
        (
            "long_context_maxlen_12",
            "Apple CEO Tim Cook works for Apple while teams discuss roadmap and planning details filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler.",
            "Apple CEO Tim Cook works for Apple while teams discuss roadmap and planning details filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler.",
            "Apple CEO Tim Cook works for Apple while teams discuss roadmap and planning details filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler.",
            "Apple CEO Tim Cook works for Apple while teams discuss roadmap and planning details filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler.",
            "long_context",
            false,
            false,
            Some(12usize),
            0.5f32,
            0.5f32,
            0.0f32,
        ),
        (
            "long_context_maxlen_24",
            "Apple CEO Tim Cook works for Apple while teams discuss roadmap and planning details filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler.",
            "Apple CEO Tim Cook works for Apple while teams discuss roadmap and planning details filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler.",
            "Apple CEO Tim Cook works for Apple while teams discuss roadmap and planning details filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler.",
            "Apple CEO Tim Cook works for Apple while teams discuss roadmap and planning details filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler.",
            "long_context",
            false,
            false,
            Some(24usize),
            0.5f32,
            0.5f32,
            0.0f32,
        ),
    ];

    let mut rust_fixtures = Vec::new();
    for (
        id,
        entities_text,
        class_text,
        relations_text,
        structure_text,
        structure_key,
        class_multi_label,
        classification_only,
        max_len,
        entity_threshold,
        relation_threshold,
        structure_threshold,
    ) in fixtures
    {
        let entities_raw = engine
            .extract_entities(
                entities_text,
                &["person", "organization", "location"],
                Some(entity_threshold),
                false,
                false,
                max_len,
            )
            .expect("rust entities failed");
        let class_raw = if max_len.is_none() {
            engine
                .classify_text(
                    class_text,
                    &[(
                        "sentiment".to_string(),
                        vec![
                            "positive".to_string(),
                            "negative".to_string(),
                            "neutral".to_string(),
                        ],
                    )],
                    Some(0.5),
                    class_multi_label,
                    None,
                )
                .expect("rust classification failed")
        } else {
            let mut class_schema_builder = SchemaBuilder::new();
            class_schema_builder = class_schema_builder
                .classification(
                    "sentiment".to_string(),
                    vec![
                        "positive".to_string(),
                        "negative".to_string(),
                        "neutral".to_string(),
                    ],
                )
                .multi_label(class_multi_label)
                .threshold(0.5)
                .done();
            let class_schema = class_schema_builder
                .build()
                .expect("failed to build rust classification schema");
            engine
                .extract(class_text, &class_schema, 0.5, false, false, max_len)
                .expect("rust classification failed")
        };
        let rel_raw = engine
            .extract_relations(
                relations_text,
                &["works_for"],
                Some(relation_threshold),
                false,
                false,
                max_len,
            )
            .expect("rust relation failed");
        let struct_schema = SchemaBuilder::new()
            .structure(structure_key)
            .field("name")
            .done_field()
            .field("company")
            .done_field()
            .done_structure()
            .build()
            .expect("failed to build rust structure schema");
        let struct_raw = engine
            .extract(
                structure_text,
                &struct_schema,
                structure_threshold,
                false,
                false,
                max_len,
            )
            .expect("rust structure failed");

        let entities_norm = if classification_only {
            serde_json::json!({"person": [], "organization": [], "location": []})
        } else {
            normalize_entities(&entities_raw)
        };
        let relations_norm = if classification_only {
            serde_json::json!({"works_for": []})
        } else {
            normalize_relations(&rel_raw)
        };
        let structures_norm = if classification_only {
            JsonValue::Array(Vec::new())
        } else {
            normalize_structures(&struct_raw, structure_key)
        };

        rust_fixtures.push(serde_json::json!({
            "id": id,
            "entities": entities_norm,
            "classification": normalize_classification(&class_raw, "sentiment"),
            "relations": relations_norm,
            "structures": structures_norm,
        }));
    }

    rust_fixtures.sort_by_key(|v| {
        v.get("id")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string()
    });
    let rust_norm = serde_json::json!({
        "fixtures": rust_fixtures
    });

    assert_eq!(
        rust_norm,
        py_norm,
        "Rust/Python parity mismatch\nrust={}\npython={}",
        serde_json::to_string_pretty(&rust_norm).unwrap_or_default(),
        serde_json::to_string_pretty(&py_norm).unwrap_or_default()
    );
}
