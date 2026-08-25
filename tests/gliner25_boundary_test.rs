//! GLiNER2.5 boundary-path end-to-end tests.
//!
//! Downloads the small checkpoint (~440MB) and runs entity extraction through
//! the boundary scoring path to verify non-empty results.

use gliner2_rs::config::{BoundaryConfig, presets::gliner25_small};
use std::path::PathBuf;

fn download_from_hub(repo_id: &str, filename: &str) -> PathBuf {
    use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
    let repo = Repo::with_revision(repo_id.to_string(), RepoType::Model, "main".to_string());
    let api = ApiBuilder::new().with_progress(true).build().expect("HF API");
    let repo_api = api.repo(repo);
    repo_api.get(filename).expect(&format!("download {filename}"))
}

/// Verify that score_sample runs end-to-end on real checkpoint and produces
/// valid candidates with non-trivial scores.
#[test]
#[ignore = "downloads ~440MB; run explicitly with --ignored"]
fn test_gliner25_small_boundary_scoring() {
    let model_id = "fastino/gliner2.5-small-v1";

    let hf_config_path = download_from_hub(model_id, "config.json");
    let hf_config = std::fs::read_to_string(&hf_config_path).expect("read config.json");

    let mut config = gliner25_small();
    config.boundary = BoundaryConfig::from_hf_config_json(&hf_config);

    let mut model = gliner2_rs::model::Extractor::new(&config).expect("construct extractor");
    let weights_path = download_from_hub(model_id, "model.safetensors");
    model.load_weights(&weights_path).expect("load weights");
    assert!(model.boundary.is_some());

    let ws_tok = gliner2_rs::tokenizer::WhitespaceTokenizer::new();
    let collator = gliner2_rs::batch::ExtractorCollator::with_max_len(ws_tok, false, config.max_len);

    let schema_json = serde_json::json!({
        "entities": ["person", "organization"]
    });
    let schema = gliner2_rs::schema::types::Schema::from_dict(&schema_json).expect("build schema");

    let text = "Apple CEO Tim Cook announced the new iPhone 15 in Cupertino.";
    let schema_dict = schema.to_dict();
    let samples = vec![(text.to_string(), schema_dict)];
    let batch = collator.collate(&samples).expect("collate");
    let batch = batch.to(candle_core::Device::Cpu, None).expect("to device");

    let token_embs = model.encoder.forward(&batch.input_ids, &batch.attention_mask).expect("encoder forward");
    let token_embs = token_embs.narrow(0, 0, 1).unwrap().squeeze(0).unwrap();

    let boundary = model.boundary.as_ref().expect("boundary present");

    // Build text states from word positions
    let h = token_embs.dims()[1];
    let rows = token_embs.to_vec2::<f32>().expect("to_vec2");
    let count = batch.text_word_counts[0];
    let twi = batch.text_word_indices.as_ref().unwrap().to_vec2::<i64>().unwrap();
    let word_positions: Vec<usize> = twi[0][..count].iter().map(|&v| v as usize).collect();
    let text_len = word_positions.len();
    let mut flat_text = Vec::with_capacity(text_len * h);
    for &w in &word_positions {
        flat_text.extend_from_slice(&rows[w]);
    }
    let text_states = candle_core::Tensor::from_slice(&flat_text, (text_len, h), &candle_core::Device::Cpu).unwrap();

    // Build query states from [E] marker positions
    let specials = &batch.schema_special_indices[0][0];
    let schema_tokens = &batch.schema_tokens_list[0][0];
    let mut sp = 0;
    let mut pending: Option<usize> = None;
    let mut queries: Vec<(String, usize)> = Vec::new();
    for token in schema_tokens {
        if token.starts_with('[') && token.ends_with(']') {
            let pos = specials[sp];
            sp += 1;
            pending = if token == "[E]" { Some(pos) } else { None };
            continue;
        }
        if let Some(marker_pos) = pending.take() {
            queries.push((token.clone(), marker_pos));
        }
    }
    let q = queries.len();
    let mut flat_queries = Vec::with_capacity(q * h);
    for (_, pos) in &queries {
        flat_queries.extend_from_slice(&rows[*pos]);
    }
    let query_states = candle_core::Tensor::from_slice(&flat_queries, (q, h), &candle_core::Device::Cpu).unwrap();

    let scored = boundary.score_sample(&text_states, text_len, &query_states).expect("score_sample");

    // Verify we have candidates
    assert!(scored.valid.len() > 0, "should have candidates");
    let num_valid = scored.valid.iter().filter(|&&v| v).count();
    assert!(num_valid > 0, "should have valid candidates");

    // Verify at least one candidate has a positive score (above random)
    let max_score = scored.scores.iter()
        .zip(&scored.valid)
        .filter(|(_, v)| **v)
        .flat_map(|(s, _)| s.iter())
        .cloned()
        .fold(f32::MIN, f32::max);
    assert!(max_score > 0.0, "max score should be positive for a trained model, got {max_score}");

    println!("OK: {num_valid} valid candidates, max_score={max_score:.4}");
    println!("Top candidates (score > 0):");
    for (ci, (((&s, &e), &valid), scores)) in scored.starts.iter()
        .zip(&scored.ends)
        .zip(&scored.valid)
        .zip(&scored.scores)
        .enumerate()
    {
        if !valid { continue; }
        let best = scores.iter().cloned().fold(f32::MIN, f32::max);
        if best > 0.0 {
            let ws = s;
            let we = e;
            let tw = &batch.text_tokens[0];
            let span_text: String = tw.get(ws..we).map(|t| t.join(" ")).unwrap_or_default();
            for (qi, (name, _)) in queries.iter().enumerate() {
                if scores[qi] > 0.0 {
                    println!("  [{name}] ({ws}..{we}) \"{span_text}\" score={:.4}", scores[qi]);
                }
            }
        }
    }
}

/// Full engine path: GLiNER2::from_pretrained → extract_entities for a
/// GLiNER2.5 checkpoint.
#[test]
#[ignore = "downloads ~440MB; run explicitly with --ignored"]
fn test_gliner25_engine_extract_entities() {
    let model_id = "fastino/gliner2.5-small-v1";

    let engine = gliner2_rs::inference::engine::GLiNER2::from_pretrained(model_id)
        .expect("from_pretrained failed");

    let result = engine.extract_entities(
        "Apple CEO Tim Cook announced the new iPhone 15 in Cupertino.",
        &["person", "organization"],
        Some(0.1),
        true,
        true,
        None,
    );

    match result {
        Ok(val) => {
            println!("Engine extraction result: {val:#}");
            let entities = val.get("entities").expect("missing entities key");
            assert!(entities.is_object(), "entities should be a JSON object");
        }
        Err(e) => {
            panic!("Engine extraction failed: {e}");
        }
    }
}
