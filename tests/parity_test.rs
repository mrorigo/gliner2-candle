//! Cross-language parity gate against the Python reference.
//!
//! Both sides read `tests/parity/cases.json`, so the case battery cannot drift
//! between implementations. The Python side is `debug_comparison/parity_reference.py`,
//! which emits the same normalized shape this test builds on the Rust side.
//!
//! Each in-scope checkpoint gets its own test so a failure names the checkpoint.
//! All of them are `#[ignore]`d because they load multi-hundred-MB weights from
//! the Hub cache; run them with `--ignored`:
//!
//! ```sh
//! cargo test --test parity_test -- --ignored --nocapture
//! ```
//!
//! `GLiNER2.5-Decide-1B` is deliberately absent: it is a ModernBERT encoder and
//! this port has no ModernBERT implementation.

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use gliner2_candle::schema::builder::SchemaBuilder;
use gliner2_candle::schema::types::Schema;
use gliner2_candle::GLiNER2;
use serde_json::{json, Value as JsonValue};

/// Decision spans and label sets must match exactly; probabilities only have to
/// agree closely. Observed Python/Rust drift on float32 CPU is ~1e-6, so this is
/// three orders of magnitude of headroom while still catching a real divergence.
const CONFIDENCE_TOLERANCE: f64 = 2e-3;

const CASES: &str = "tests/parity/cases.json";

// ---------------------------------------------------------------------------
// Case battery
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Case {
    id: String,
    kind: String,
    text: String,
    entities: Vec<String>,
    entity_descriptions: HashMap<String, String>,
    classifications: JsonValue,
    relations: Vec<String>,
    structure_key: Option<String>,
    structure_fields: Vec<String>,
    threshold: f64,
}

fn load_cases() -> Vec<Case> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(CASES);
    let raw: JsonValue =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read cases.json")).unwrap();

    raw["cases"]
        .as_array()
        .expect("cases array")
        .iter()
        .map(|c| {
            let strings = |key: &str| -> Vec<String> {
                c.get(key)
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default()
            };
            let structure = c.get("structure");
            Case {
                id: c["id"].as_str().unwrap().to_string(),
                kind: c["kind"].as_str().unwrap().to_string(),
                text: c["text"].as_str().unwrap().to_string(),
                entities: strings("entities"),
                entity_descriptions: c
                    .get("entity_descriptions")
                    .and_then(|v| v.as_object())
                    .map(|o| {
                        o.iter()
                            .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
                            .collect()
                    })
                    .unwrap_or_default(),
                classifications: c.get("classifications").cloned().unwrap_or(json!({})),
                relations: strings("relations"),
                structure_key: structure
                    .and_then(|s| s.as_object())
                    .and_then(|o| o.keys().next())
                    .cloned(),
                structure_fields: structure
                    .and_then(|s| s.as_object())
                    .and_then(|o| o.values().next())
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default(),
                threshold: c
                    .get("threshold")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Python reference
// ---------------------------------------------------------------------------

fn python_reference(model_id: &str) -> HashMap<String, JsonValue> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let script = root.join("debug_comparison/parity_reference.py");
    let py = root.join(".venv/bin/python");
    assert!(py.exists(), "missing python venv at {}", py.display());

    let output = Command::new(&py)
        .arg(&script)
        .arg(model_id)
        .arg(root.join(CASES))
        .current_dir(root)
        .output()
        .expect("failed to spawn python parity reference");
    assert!(
        output.status.success(),
        "python reference failed for {model_id}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let start = stdout
        .find("__PARITY_JSON_START__")
        .expect("missing start marker");
    let end = stdout.find("__PARITY_JSON_END__").expect("missing end");
    let parsed: JsonValue =
        serde_json::from_str(stdout[start + "__PARITY_JSON_START__".len()..end].trim())
            .expect("invalid python json");

    parsed
        .as_object()
        .expect("object")
        .iter()
        .map(|(k, v)| (k.clone(), v["value"].clone()))
        .collect()
}

// ---------------------------------------------------------------------------
// Normalization (mirrors debug_comparison/parity_reference.py)
// ---------------------------------------------------------------------------

const TOP_SPANS_PER_FIELD: usize = 3;

fn norm_text(s: &str) -> String {
    s.trim().to_lowercase()
}

fn round6(v: f64) -> f64 {
    (v * 1e6).round() / 1e6
}

fn norm_entities(raw: &JsonValue) -> JsonValue {
    let mut out = serde_json::Map::new();
    let Some(groups) = raw.get("entities").and_then(|v| v.as_object()) else {
        return json!({});
    };
    for (label, items) in groups {
        let mut rows: Vec<JsonValue> = items
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|i| {
                        let text = i.get("text").and_then(|v| v.as_str())?;
                        Some(json!({
                            "text": norm_text(text),
                            "confidence": round6(i.get("confidence")?.as_f64()?),
                        }))
                    })
                    .collect()
            })
            .unwrap_or_default();
        rows.sort_by(|a, b| {
            let (ta, ca) = (a["text"].as_str().unwrap_or(""), a["confidence"].as_f64().unwrap_or(0.0));
            let (tb, cb) = (b["text"].as_str().unwrap_or(""), b["confidence"].as_f64().unwrap_or(0.0));
            ta.cmp(tb).then(cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal).reverse())
        });
        out.insert(label.clone(), JsonValue::Array(rows));
    }
    JsonValue::Object(out)
}

fn norm_classifications(raw: &JsonValue, tasks: &[String]) -> JsonValue {
    let mut out = serde_json::Map::new();
    for task in tasks {
        let mut rows: Vec<JsonValue> = Vec::new();
        match raw.get(task) {
            Some(JsonValue::Object(o)) => rows.push(json!({
                "label": o.get("label"),
                "confidence": o.get("confidence").and_then(|c| c.as_f64()).map(round6),
            })),
            Some(JsonValue::Array(a)) => {
                for i in a {
                    let label = i.get("label").or(Some(i));
                    rows.push(json!({
                        "label": label,
                        "confidence": i.get("confidence").and_then(|c| c.as_f64()).map(round6),
                    }));
                }
            }
            _ => {}
        }
        rows.sort_by_key(|r| r["label"].to_string());
        rows.dedup_by(|a, b| a["label"] == b["label"]);
        out.insert(task.clone(), json!({ "labels": rows }));
    }
    JsonValue::Object(out)
}

fn norm_relations(raw: &JsonValue) -> JsonValue {
    let mut out = serde_json::Map::new();
    let Some(rels) = raw.get("relation_extraction").and_then(|v| v.as_object()) else {
        return json!({});
    };
    for (name, items) in rels {
        let mut rows: Vec<JsonValue> = Vec::new();
        for item in items.as_array().cloned().unwrap_or_default() {
            let side = |k: &str| -> (String, Option<f64>) {
                let v = item.get(k);
                let text = v
                    .and_then(|x| x.get("text"))
                    .and_then(|x| x.as_str())
                    .or_else(|| v.and_then(|x| x.as_str()))
                    .unwrap_or("");
                let conf = v
                    .and_then(|x| x.get("confidence"))
                    .and_then(|x| x.as_f64())
                    .map(round6);
                (norm_text(text), conf)
            };
            let (head, head_conf) = side("head");
            let (tail, tail_conf) = side("tail");
            rows.push(json!({
                "head": head,
                "tail": tail,
                "head_confidence": head_conf,
                "tail_confidence": tail_conf,
            }));
        }
        rows.sort_by_key(|r| {
            (
                r["head"].as_str().unwrap_or("").to_string(),
                r["tail"].as_str().unwrap_or("").to_string(),
            )
        });
        out.insert(name.clone(), JsonValue::Array(rows));
    }
    JsonValue::Object(out)
}

fn norm_structure(raw: &JsonValue, key: &str) -> JsonValue {
    let mut fields = serde_json::Map::new();
    for inst in raw.get(key).and_then(|v| v.as_array()).cloned().unwrap_or_default() {
        let Some(obj) = inst.as_object() else { continue };
        for (field, value) in obj {
            let items = match value {
                JsonValue::Array(a) => a.clone(),
                other => vec![other.clone()],
            };
            let mut rows: Vec<JsonValue> = items
                .iter()
                .filter_map(|i| {
                    let text = i.get("text").and_then(|v| v.as_str())?;
                    if text.trim().is_empty() {
                        return None;
                    }
                    Some(json!({
                        "text": norm_text(text),
                        "confidence": round6(i.get("confidence")?.as_f64()?),
                    }))
                })
                .collect();
            rows.sort_by(|a, b| {
                b["confidence"]
                    .as_f64()
                    .unwrap_or(0.0)
                    .partial_cmp(&a["confidence"].as_f64().unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            fields
                .entry(field.clone())
                .or_insert_with(|| JsonValue::Array(Vec::new()));
            if let JsonValue::Array(existing) = fields.get_mut(field).unwrap() {
                existing.extend(rows);
            }
        }
    }
    for value in fields.values_mut() {
        if let JsonValue::Array(rows) = value {
            rows.sort_by(|a, b| {
                b["confidence"]
                    .as_f64()
                    .unwrap_or(0.0)
                    .partial_cmp(&a["confidence"].as_f64().unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            rows.truncate(TOP_SPANS_PER_FIELD);
        }
    }
    JsonValue::Object(fields)
}

// ---------------------------------------------------------------------------
// Rust side
// ---------------------------------------------------------------------------

fn classification_schema(spec: &JsonValue) -> Schema {
    let mut builder = SchemaBuilder::new();
    if let Some(tasks) = spec.as_object() {
        for (task, head) in tasks {
            let labels: Vec<String> = head["labels"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            let mut cls = builder.classification(task.clone(), labels);
            cls = cls.multi_label(head["multi_label"].as_bool().unwrap_or(false));
            if let Some(descs) = head.get("label_descriptions").and_then(|v| v.as_object()) {
                let map: HashMap<String, String> = descs
                    .iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                    .collect();
                cls = cls.label_descriptions(map);
            }
            if let Some(p) = head.get("prompt").and_then(|v| v.as_str()) {
                cls = cls.prompt(p);
            }
            if let Some(ex) = head.get("examples").and_then(|v| v.as_array()) {
                for e in ex {
                    if let Some(a) = e.as_array()
                        && let (Some(i), Some(o)) = (a.first().and_then(|v| v.as_str()), a.get(1).and_then(|v| v.as_str()))
                    {
                        cls = cls.example(i, o);
                    }
                }
            }
            builder = cls.done();
        }
    }
    builder.build().expect("classification schema")
}

fn structure_schema(key: &str, fields: &[String]) -> Schema {
    let mut builder = SchemaBuilder::new().structure(key.to_string());
    for f in fields {
        builder = builder.field(f.clone()).done_field();
    }
    builder.done_structure().build().expect("structure schema")
}

fn run_case(engine: &GLiNER2, case: &Case) -> JsonValue {
    let threshold = case.threshold as f32;
    match case.kind.as_str() {
        "entities" => norm_entities(
            &engine
                .extract_entities(
                    &case.text,
                    &case.entities.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                    Some(threshold),
                    true,
                    true,
                    None,
                )
                .expect("extract_entities"),
        ),
        "entities_described" => {
            // Pass ordered pairs: the order is the prompt's entity order, and
            // cases.json lists descriptions in the same order as `entities`.
            let ordered: Vec<(String, String)> = case
                .entities
                .iter()
                .filter_map(|name| {
                    case.entity_descriptions
                        .get(name)
                        .map(|d| (name.clone(), d.clone()))
                })
                .collect();
            let schema = SchemaBuilder::new()
                .entities_with_descriptions(ordered)
                .build()
                .expect("entity schema");
            norm_entities(&engine.extract(&case.text, &schema, threshold, true, true, None).unwrap())
        }
        "classifications" => {
            let schema = classification_schema(&case.classifications);
            let tasks: Vec<String> = case
                .classifications
                .as_object()
                .map(|o| o.keys().cloned().collect())
                .unwrap_or_default();
            norm_classifications(
                &engine
                    .extract(&case.text, &schema, threshold, true, true, None)
                    .unwrap(),
                &tasks,
            )
        }
        "relations" => norm_relations(
            &engine
                .extract_relations(
                    &case.text,
                    &case.relations.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                    Some(threshold),
                    true,
                    true,
                    None,
                )
                .expect("extract_relations"),
        ),
        "structure" => {
            let key = case.structure_key.clone().expect("structure key");
            let schema = structure_schema(&key, &case.structure_fields);
            norm_structure(
                &engine.extract(&case.text, &schema, threshold, true, true, None).unwrap(),
                &key,
            )
        }
        other => panic!("unknown case kind: {other}"),
    }
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Walk both normalized trees. Structural shape must match exactly; leaf
/// strings must match exactly; leaves under a `confidence` key are compared
/// with a tolerance. Returns one human-readable line per divergence.
fn diff(path: &str, rust: &JsonValue, python: &JsonValue, out: &mut Vec<String>) {
    match (rust, python) {
        (JsonValue::Object(r), JsonValue::Object(p)) => {
            let mut keys: Vec<&String> = r.keys().chain(p.keys()).collect();
            keys.sort();
            keys.dedup();
            for k in keys {
                match (r.get(k), p.get(k)) {
                    (Some(rv), Some(pv)) => diff(&format!("{path}.{k}"), rv, pv, out),
                    (Some(_), None) => out.push(format!("{path}.{k}: only in rust")),
                    (None, Some(_)) => out.push(format!("{path}.{k}: only in python")),
                    _ => {}
                }
            }
        }
        (JsonValue::Array(r), JsonValue::Array(p)) => {
            if r.len() != p.len() {
                out.push(format!(
                    "{path}: rust has {} items, python has {}",
                    r.len(),
                    p.len()
                ));
                return;
            }
            for (i, (rv, pv)) in r.iter().zip(p.iter()).enumerate() {
                diff(&format!("{path}[{i}]"), rv, pv, out);
            }
        }
        _ => {
            let leaf = path.rsplit('.').next().unwrap_or(path);
            if leaf == "confidence" || leaf.ends_with("_confidence") {
                let rv = rust.as_f64().unwrap_or(f64::NAN);
                let pv = python.as_f64().unwrap_or(f64::NAN);
                if (rv - pv).abs() > CONFIDENCE_TOLERANCE {
                    out.push(format!("{path}: rust {rv} vs python {pv}"));
                }
            } else if rust != python {
                out.push(format!("{path}: rust {rust} vs python {python}"));
            }
        }
    }
}

fn assert_parity(model_id: &str) {
    let cases = load_cases();
    let engine = GLiNER2::from_pretrained(model_id)
        .unwrap_or_else(|e| panic!("failed to load {model_id}: {e}"));
    let python = python_reference(model_id);

    let mut failures: Vec<String> = Vec::new();
    for case in &cases {
        let Some(expected) = python.get(&case.id) else {
            failures.push(format!("{}: python reference produced no output", case.id));
            continue;
        };
        let actual = run_case(&engine, case);
        let mut diffs = Vec::new();
        diff("", &actual, expected, &mut diffs);
        if !diffs.is_empty() {
            failures.push(format!("  {}:\n    {}", case.id, diffs.join("\n    ")));
        }
    }

    assert!(
        failures.is_empty(),
        "parity mismatch for {model_id} ({} case(s)):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Tests, one per in-scope checkpoint
// ---------------------------------------------------------------------------

fn span_parity(model_id: &str) {
    if std::env::var_os("GLINER2_SKIP_HUB").is_some() {
        eprintln!("skipping {model_id}: GLINER2_SKIP_HUB set");
        return;
    }
    assert_parity(model_id);
}

macro_rules! parity_test {
    ($name:ident, $model:expr, $note:expr) => {
        #[test]
        #[ignore = $note]
        fn $name() {
            span_parity($model);
        }
    };
}

parity_test!(gliner2_base_v1_parity, "fastino/gliner2-base-v1", "loads real weights from the Hub cache");
parity_test!(gliner2_large_v1_parity, "fastino/gliner2-large-v1", "loads real weights from the Hub cache");
parity_test!(gliner25_decide_parity, "fastino/GLiNER2.5-Decide", "loads real weights from the Hub cache");
parity_test!(gliner25_base_v1_parity, "fastino/gliner2.5-base-v1", "loads real weights from the Hub cache");
parity_test!(gliner25_small_v1_parity, "fastino/gliner2.5-small-v1", "loads real weights from the Hub cache");
parity_test!(gliner25_multi_v1_parity, "fastino/gliner2.5-multi-v1", "loads real weights from the Hub cache");
parity_test!(gliner25_multi_decide_parity, "fastino/GLiNER2.5-multi-Decide", "loads real weights from the Hub cache");
