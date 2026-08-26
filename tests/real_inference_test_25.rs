//! GLiNER2.5 checkpoint validation across all released models.
//!
//! Downloads each 2.5 checkpoint (~440MB-1.7GB) and verifies loading plus
//! end-to-end entity extraction quality on the shared parity fixture.
//! Gated behind `--ignored` like the other hub tests.

use gliner2_rs::inference::engine::GLiNER2;
use std::path::PathBuf;

fn download_from_hub(repo_id: &str, filename: &str) -> PathBuf {
    use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
    let repo = Repo::with_revision(repo_id.to_string(), RepoType::Model, "main".to_string());
    let api = ApiBuilder::new().with_progress(true).build().expect("HF API");
    let repo_api = api.repo(repo);
    repo_api.get(filename).expect(&format!("download {filename}"))
}

/// Shared fixture text + expected primary extractions (from the Python
/// reference implementation at threshold 0.5).
const FIXTURE_TEXT: &str = "Apple CEO Tim Cook announced the new iPhone 15 in Cupertino.";
const EXPECTED_PERSON: &str = "Tim Cook";
const EXPECTED_ORG: &str = "Apple";

fn check_checkpoint(model_id: &str) {
    let engine = GLiNER2::from_pretrained(model_id).expect("from_pretrained");

    let schema_json = serde_json::json!({
        "entities": ["person", "organization", "location"]
    });
    let schema = gliner2_rs::schema::types::Schema::from_dict(&schema_json).expect("schema");
    let result = engine
        .extract(FIXTURE_TEXT, &schema, 0.5, true, true, None)
        .expect("extract");

    println!("--- {model_id} ---");
    println!("{result:#}");

    let entities = result.get("entities").expect("entities key");
    assert!(entities.is_object(), "entities should be an object");

    // Primary extractions must appear with confidence >= 0.5.
    let has_span = |label: &str, needle: &str| -> bool {
        entities[label]
            .as_array()
            .map(|arr| {
                arr.iter().any(|e| {
                    e["text"].as_str().map(|t| t.contains(needle)).unwrap_or(false)
                        && e["confidence"].as_f64().map(|c| c >= 0.5).unwrap_or(true)
                })
            })
            .unwrap_or(false)
    };
    assert!(
        has_span("person", EXPECTED_PERSON),
        "{model_id}: expected person span containing '{EXPECTED_PERSON}'"
    );
    assert!(
        has_span("organization", EXPECTED_ORG),
        "{model_id}: expected organization span containing '{EXPECTED_ORG}'"
    );
}

#[test]
#[ignore = "downloads ~500MB; run explicitly with --ignored"]
fn test_gliner25_small_checkpoint() {
    check_checkpoint("fastino/gliner2.5-small-v1");
}

#[test]
#[ignore = "downloads ~900MB; run explicitly with --ignored"]
fn test_gliner25_base_checkpoint() {
    check_checkpoint("fastino/gliner2.5-base-v1");
}

#[test]
#[ignore = "downloads ~1GB; run explicitly with --ignored"]
fn test_gliner25_multi_checkpoint() {
    check_checkpoint("fastino/gliner2.5-multi-v1");
}

/// Linear-scaling smoke: chunked extraction time should grow roughly linearly
/// beyond the chunk size (384 words), not quadratically.
#[test]
#[ignore = "perf smoke; downloads ~500MB"]
fn test_perf_linear_scaling() {
    let engine = GLiNER2::from_pretrained("fastino/gliner2.5-small-v1").expect("load");
    let schema = gliner2_rs::schema::types::Schema::from_dict(&serde_json::json!({
        "entities": ["person", "organization"]
    }))
    .expect("schema");

    let para = "Apple CEO Tim Cook announced the new iPhone 15 in Cupertino on Tuesday. \
        Google and Microsoft also sent representatives to the event in California.";

    let mut prev = std::time::Duration::ZERO;
    let mut prev_words = 0usize;
    for take in [5usize, 20, 40] {
        let text: String = std::iter::repeat(para).take(take).collect::<Vec<_>>().join(" ");
        let words = text.split_whitespace().count();
        let _ = engine.extract(&text, &schema, 0.5, false, false, None).unwrap(); // warmup
        let t0 = std::time::Instant::now();
        let _ = engine.extract(&text, &schema, 0.5, false, false, None).unwrap();
        let elapsed = t0.elapsed();
        if prev_words > 0 && words > 384 {
            let ratio = elapsed.as_secs_f64() / prev.as_secs_f64().max(1e-6);
            let word_ratio = words as f64 / prev_words as f64;
            println!(
                "words={words} elapsed={elapsed:?} ratio_vs_prev={ratio:.2} word_ratio={word_ratio:.2}"
            );
            // Allow generous slack but catch quadratic blowups.
            assert!(
                ratio < word_ratio * 2.0,
                "superlinear scaling detected: {ratio:.2}x time for {word_ratio:.2}x words"
            );
        } else {
            println!("words={words} elapsed={elapsed:?}");
        }
        prev = elapsed;
        prev_words = words;
    }
}

/// Span attributes (Phase 3e): attribute labels ride along as hidden entity
/// queries; retained spans are re-scored via score_explicit_spans.
#[test]
#[ignore = "downloads ~500MB; run explicitly with --ignored"]
fn test_gliner25_entity_attributes() {
    use gliner2_rs::schema::types::{AttributeGroup, EntityDef, Schema};
    use std::collections::HashMap;

    let engine = GLiNER2::from_pretrained("fastino/gliner2.5-small-v1").expect("load");

    let mut groups = HashMap::new();
    groups.insert(
        "sentiment".to_string(),
        AttributeGroup {
            labels: vec!["positive".to_string(), "negative".to_string()],
            multi_label: true,
            threshold: 0.5,
            applies_to: Some(vec!["person".to_string()]),
            qualify_labels: false,
        },
    );
    let schema = Schema::new()
        .entities(vec![EntityDef::new("person"), EntityDef::new("organization")])
        .entity_attributes(groups)
        .expect("schema");

    let result = engine
        .extract(
            "Apple CEO Tim Cook announced great results in Cupertino.",
            &schema,
            0.5,
            false,
            false,
            None,
        )
        .expect("extract");
    println!("{result:#}");

    let entities = result.get("entities").expect("entities");
    // Attributed format forces object entries.
    let persons = entities["person"].as_array().expect("person array");
    assert!(!persons.is_empty(), "expected a person span");
    for p in persons {
        assert!(p.get("text").is_some(), "attributed entry must be an object");
        assert!(
            p.get("sentiment").is_some(),
            "person entries carry the sentiment group: {p}"
        );
        let sent = p["sentiment"].as_array().unwrap();
        for v in sent {
            assert!(v.get("label").is_some() && v.get("confidence").is_some());
        }
    }
    if let Some(orgs) = entities["organization"].as_array() {
        for o in orgs {
            assert!(
                o.get("sentiment").is_none(),
                "sentiment must not attach to organization: {o}"
            );
        }
    }
}

/// Attribute prompt labels must not leak into the public entity output.
#[test]
fn test_attribute_schema_expansion() {
    use gliner2_rs::schema::types::{AttributeGroup, EntityDef, Schema};
    use std::collections::HashMap;

    let mut groups = HashMap::new();
    groups.insert(
        "lang".to_string(),
        AttributeGroup {
            labels: vec!["english".to_string()],
            ..Default::default()
        },
    );
    let schema = Schema::new()
        .entities(vec![EntityDef::new("person")])
        .entity_attributes(groups)
        .expect("schema");

    let (expanded, prompts) = schema.expanded_with_attributes().expect("expand");
    assert_eq!(prompts.get("english").map(String::as_str), Some("english"));
    // Hidden query appended after content entities; public order untouched.
    assert_eq!(schema.entities.len(), 1);
    assert_eq!(expanded.entities.len(), 2);
    assert_eq!(expanded.entities[1].name, "english");

    // Validation: unknown applies_to entity rejected.
    let mut bad = HashMap::new();
    bad.insert(
        "g".to_string(),
        AttributeGroup {
            labels: vec!["x".to_string()],
            applies_to: Some(vec!["nonexistent".to_string()]),
            ..Default::default()
        },
    );
    assert!(Schema::new()
        .entities(vec![EntityDef::new("person")])
        .entity_attributes(bad)
        .is_err());

    // Validation: requires entities first.
    let mut g2 = HashMap::new();
    g2.insert(
        "g".to_string(),
        AttributeGroup {
            labels: vec!["x".to_string()],
            ..Default::default()
        },
    );
    assert!(Schema::new().entity_attributes(g2).is_err());
}
