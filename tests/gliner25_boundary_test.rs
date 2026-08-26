//! GLiNER2.5 boundary-path end-to-end tests.
//!
//! Downloads the small checkpoint (~440MB) and runs entity extraction through
//! the boundary scoring path to verify non-empty results.

use gliner2_rs::config::{BoundaryConfig, presets::gliner25_small};
use std::path::PathBuf;

fn download_from_hub(repo_id: &str, filename: &str) -> PathBuf {
    use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
    let repo = Repo::with_revision(repo_id.to_string(), RepoType::Model, "main".to_string());
    let api = ApiBuilder::new()
        .with_progress(true)
        .build()
        .expect("HF API");
    let repo_api = api.repo(repo);
    repo_api
        .get(filename)
        .unwrap_or_else(|_| panic!("download {filename}"))
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
    let tok_path = download_from_hub(model_id, "tokenizer.json");
    let hf_tok = tokenizers::Tokenizer::from_file(&tok_path).expect("load tokenizer");
    let collator = gliner2_rs::batch::ExtractorCollator::with_hf_tokenizer(
        ws_tok,
        hf_tok,
        false,
        config.max_len,
    );

    let schema_json = serde_json::json!({
        "entities": ["person", "organization"]
    });

    let text = "Apple CEO Tim Cook announced the new iPhone 15 in Cupertino.";
    let samples = vec![(text.to_string(), schema_json)];
    let batch = collator.collate(&samples).expect("collate");
    let batch = batch.to(candle_core::Device::Cpu, None).expect("to device");

    let token_embs = model
        .encoder
        .forward(&batch.input_ids, &batch.attention_mask)
        .expect("encoder forward");
    let token_embs = token_embs.narrow(0, 0, 1).unwrap().squeeze(0).unwrap();

    let boundary = model.boundary.as_ref().expect("boundary present");

    // Build text states from word positions
    let h = token_embs.dims()[1];
    let rows = token_embs.to_vec2::<f32>().expect("to_vec2");
    let count = batch.text_word_counts[0];
    let twi = batch
        .text_word_indices
        .as_ref()
        .unwrap()
        .to_vec2::<i64>()
        .unwrap();
    let word_positions: Vec<usize> = twi[0][..count].iter().map(|&v| v as usize).collect();
    let text_len = word_positions.len();
    let mut flat_text = Vec::with_capacity(text_len * h);
    for &w in &word_positions {
        flat_text.extend_from_slice(&rows[w]);
    }
    let text_states =
        candle_core::Tensor::from_slice(&flat_text, (text_len, h), &candle_core::Device::Cpu)
            .unwrap();

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
    let query_states =
        candle_core::Tensor::from_slice(&flat_queries, (q, h), &candle_core::Device::Cpu).unwrap();

    let scored = boundary
        .score_sample(&text_states, text_len, &query_states)
        .expect("score_sample");

    // Verify we have candidates
    assert!(!scored.valid.is_empty(), "should have candidates");
    let num_valid = scored.valid.iter().filter(|&&v| v).count();
    assert!(num_valid > 0, "should have valid candidates");

    // Verify at least one candidate has a positive score (above random)
    let max_score = scored
        .scores
        .iter()
        .zip(&scored.valid)
        .filter(|(_, v)| **v)
        .flat_map(|(s, _)| s.iter())
        .cloned()
        .fold(f32::MIN, f32::max);
    assert!(
        max_score > 0.0,
        "max score should be positive for a trained model, got {max_score}"
    );

    println!("OK: {num_valid} valid candidates, max_score={max_score:.4}");
    println!("Top candidates (score > 0):");
    for (((&s, &e), &valid), scores) in scored
        .starts
        .iter()
        .zip(&scored.ends)
        .zip(&scored.valid)
        .zip(&scored.scores)
    {
        if !valid {
            continue;
        }
        let best = scores.iter().cloned().fold(f32::MIN, f32::max);
        if best > 0.0 {
            let ws = s;
            let we = e;
            let tw = &batch.text_tokens[0];
            let span_text: String = tw.get(ws..we).map(|t| t.join(" ")).unwrap_or_default();
            for (qi, (name, _)) in queries.iter().enumerate() {
                if scores[qi] > 0.0 {
                    println!(
                        "  [{name}] ({ws}..{we}) \"{span_text}\" score={:.4}",
                        scores[qi]
                    );
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

#[test]
#[ignore = "debug collation comparison"]
fn test_collation_comparison() {
    let model_id = "fastino/gliner2.5-small-v1";
    let hf_config_path = download_from_hub(model_id, "config.json");
    let hf_config = std::fs::read_to_string(&hf_config_path).expect("read config.json");
    let mut config = gliner2_rs::config::presets::gliner25_small();
    config.boundary = gliner2_rs::config::BoundaryConfig::from_hf_config_json(&hf_config);

    // Load HF tokenizer
    let tok_path = download_from_hub(model_id, "tokenizer.json");
    let hf_tok = tokenizers::Tokenizer::from_file(&tok_path).expect("load tokenizer");

    let ws_tok = gliner2_rs::tokenizer::WhitespaceTokenizer::new();
    let collator = gliner2_rs::batch::ExtractorCollator::with_hf_tokenizer(
        ws_tok,
        hf_tok,
        false,
        config.max_len,
    );

    let schema_json = serde_json::json!({
        "entities": ["person", "organization", "location"]
    });

    let text = "Apple CEO Tim Cook announced the new iPhone 15 in Cupertino.";
    let samples = vec![(text.to_string(), schema_json)];
    let batch = collator.collate(&samples).expect("collate");

    println!("=== RUST COLLATION ===");
    println!("input_ids len: {}", batch.input_ids.dim(1).unwrap());
    let ids = batch
        .input_ids
        .flatten_all()
        .unwrap()
        .to_vec1::<i64>()
        .unwrap();
    let ids_u32: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
    println!("input_ids: {:?}", ids_u32);
    println!("text_word_indices: {:?}", batch.text_word_indices);
    println!("schema_special_indices: {:?}", batch.schema_special_indices);

    // Compare with Python:
    // Python input_ids: [287, 128003, 6967, 287, 128005, 604, 128005, 1416, 128005, 1250, 1263, 1263, 128002, 6038, 101312, 41718, 3712, 1577, 262, 353, 16998, 706, 267, 3189, 649, 41290, 323]
    // Python text_word_indices: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26]
    // Python query_marker_indices: [4, 6, 8]
    // Python start_mappings: [0, 6, 10, 14, 19, 29, 33, 37, 44, 47, 50, 59]
    // Python end_mappings: [5, 9, 13, 18, 28, 32, 36, 43, 46, 49, 59, 60]
}

/// Numeric parity vs Python reference dumps in /tmp/g25diag/.
#[test]
#[ignore = "numeric parity debug; requires /tmp/g25diag dumps"]
fn test_numeric_parity() {
    let model_id = "fastino/gliner2.5-small-v1";
    let hf_config_path = download_from_hub(model_id, "config.json");
    let hf_config = std::fs::read_to_string(&hf_config_path).expect("read config.json");
    let mut config = gliner25_small();
    config.boundary = BoundaryConfig::from_hf_config_json(&hf_config);

    let mut model = gliner2_rs::model::Extractor::new(&config).expect("construct extractor");
    let weights_path = download_from_hub(model_id, "model.safetensors");
    model.load_weights(&weights_path).expect("load weights");
    let boundary = model.boundary.as_ref().expect("boundary present");

    let tok_path = download_from_hub(model_id, "tokenizer.json");
    let hf_tok = tokenizers::Tokenizer::from_file(&tok_path).expect("load tokenizer");
    let ws_tok = gliner2_rs::tokenizer::WhitespaceTokenizer::new();
    let collator = gliner2_rs::batch::ExtractorCollator::with_hf_tokenizer(
        ws_tok,
        hf_tok,
        false,
        config.max_len,
    );

    let schema_json = serde_json::json!({
        "entities": ["person", "organization", "location"]
    });
    let text = "Apple CEO Tim Cook announced the new iPhone 15 in Cupertino.";
    let batch = collator
        .collate(&[(text.to_string(), schema_json)])
        .expect("collate")
        .to(candle_core::Device::Cpu, None)
        .expect("to device");

    let token_embs = model
        .encoder
        .forward(&batch.input_ids, &batch.attention_mask)
        .expect("encoder forward")
        .narrow(0, 0, 1)
        .unwrap()
        .squeeze(0)
        .unwrap();
    let h = token_embs.dims()[1];
    let rows = token_embs.to_vec2::<f32>().unwrap();

    // text states
    let count = batch.text_word_counts[0];
    let twi = batch
        .text_word_indices
        .as_ref()
        .unwrap()
        .to_vec2::<i64>()
        .unwrap();
    let word_positions: Vec<usize> = twi[0][..count].iter().map(|&v| v as usize).collect();
    let mut flat_text = Vec::with_capacity(count * h);
    for &w in &word_positions {
        flat_text.extend_from_slice(&rows[w]);
    }
    let text_states =
        candle_core::Tensor::from_slice(&flat_text, (count, h), &candle_core::Device::Cpu).unwrap();

    // query states ([E] markers only)
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
    let query_states =
        candle_core::Tensor::from_slice(&flat_queries, (q, h), &candle_core::Device::Cpu).unwrap();

    // --- Compare text_states with Python ---
    let py: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("/tmp/g25diag/text_states.json").unwrap())
            .unwrap();
    let py_data = py["data"].as_array().unwrap();
    let rs_data = text_states.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut max_diff: f32 = 0.0;
    for (a, b) in py_data.iter().zip(&rs_data) {
        max_diff = max_diff.max((a.as_f64().unwrap() as f32 - b).abs());
    }
    println!(
        "text_states shape rust=({count},{h}) python={:?} max_diff={max_diff:.6}",
        py["shape"]
    );

    let scored = boundary
        .score_sample(&text_states, count, &query_states)
        .expect("score_sample");

    // Compare pair scores per query
    let py_pair: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("/tmp/g25diag/pair_logits.json").unwrap())
            .unwrap();
    println!("python pair_logits shape={:?}", py_pair["shape"]);
    let names = ["person", "organization", "location"];
    for (qi, name) in names.iter().enumerate() {
        let mut top: Vec<(usize, usize, f32)> = Vec::new();
        for ci in 0..scored.starts.len() {
            if !scored.valid[ci] {
                continue;
            }
            top.push((scored.starts[ci], scored.ends[ci], scored.scores[ci][qi]));
        }
        top.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        println!("{name} top5: {:?}", &top[..top.len().min(5)]);
    }
}

/// Layer-by-layer parity bisection against Python per-layer dumps
/// (`/tmp/g25diag/encoder_stages.json`). Identifies the first stage where
/// divergence appears and how it grows through the stack.
#[test]
#[ignore = "layerwise parity debug; requires /tmp/g25diag dumps"]
fn test_layerwise_parity() {
    let model_id = "fastino/gliner2.5-small-v1";
    let hf_config_path = download_from_hub(model_id, "config.json");
    let hf_config = std::fs::read_to_string(&hf_config_path).expect("read config.json");
    let mut config = gliner25_small();
    config.boundary = BoundaryConfig::from_hf_config_json(&hf_config);

    let mut model = gliner2_rs::model::Extractor::new(&config).expect("construct extractor");
    let weights_path = download_from_hub(model_id, "model.safetensors");
    model.load_weights(&weights_path).expect("load weights");

    let tok_path = download_from_hub(model_id, "tokenizer.json");
    let hf_tok = tokenizers::Tokenizer::from_file(&tok_path).expect("load tokenizer");
    let ws_tok = gliner2_rs::tokenizer::WhitespaceTokenizer::new();
    let collator = gliner2_rs::batch::ExtractorCollator::with_hf_tokenizer(
        ws_tok,
        hf_tok,
        false,
        config.max_len,
    );

    let schema_json = serde_json::json!({
        "entities": ["person", "organization", "location"]
    });
    let text = "Apple CEO Tim Cook announced the new iPhone 15 in Cupertino.";
    let batch = collator
        .collate(&[(text.to_string(), schema_json)])
        .expect("collate")
        .to(candle_core::Device::Cpu, None)
        .expect("to device");

    let stages = model
        .encoder
        .forward_debug(&batch.input_ids, &batch.attention_mask)
        .expect("forward_debug");

    let py: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("/tmp/g25diag/encoder_stages.json").unwrap())
            .unwrap();

    let max_diff_vs = |data: &serde_json::Value, t: &candle_core::Tensor| -> f32 {
        let rs = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        fn flatten(v: &serde_json::Value, out: &mut Vec<f32>) {
            if let Some(arr) = v.as_array() {
                for x in arr {
                    flatten(x, out);
                }
            } else if let Some(n) = v.as_f64() {
                out.push(n as f32);
            }
        }
        let mut pyv = Vec::new();
        flatten(data, &mut pyv);
        assert_eq!(pyv.len(), rs.len(), "stage size mismatch");
        let mut md = 0.0f32;
        for (a, b) in pyv.iter().zip(&rs) {
            md = md.max((a - b).abs());
        }
        md
    };

    println!("stage | max_diff");
    for k in 0..12 {
        let input_stage = &stages[k];
        let output_stage = &stages[k + 1];
        let din = max_diff_vs(&py[format!("layer_{k}_in").as_str()], input_stage);
        let dout = max_diff_vs(&py[format!("layer_{k}_out").as_str()], output_stage);
        println!(
            "layer {k:2}: in={din:.6} out={dout:.6}  (delta {:+.6})",
            dout - din
        );
    }
}

/// Sub-stage parity inside each layer: selfattn context, attention output,
/// FFN gelu, layer output vs `/tmp/g25diag/encoder_substages.json`.
#[test]
#[ignore = "substage parity debug; requires /tmp/g25diag dumps"]
fn test_substage_parity() {
    let model_id = "fastino/gliner2.5-small-v1";
    let hf_config = std::fs::read_to_string(download_from_hub(model_id, "config.json")).unwrap();
    let mut config = gliner25_small();
    config.boundary = BoundaryConfig::from_hf_config_json(&hf_config);
    let mut model = gliner2_rs::model::Extractor::new(&config).unwrap();
    model
        .load_weights(download_from_hub(model_id, "model.safetensors"))
        .unwrap();

    let tok =
        tokenizers::Tokenizer::from_file(download_from_hub(model_id, "tokenizer.json")).unwrap();
    let collator = gliner2_rs::batch::ExtractorCollator::with_hf_tokenizer(
        gliner2_rs::tokenizer::WhitespaceTokenizer::new(),
        tok,
        false,
        config.max_len,
    );
    let schema_json = serde_json::json!({"entities": ["person", "organization", "location"]});
    let text = "Apple CEO Tim Cook announced the new iPhone 15 in Cupertino.";
    let batch = collator
        .collate(&[(text.to_string(), schema_json)])
        .unwrap()
        .to(candle_core::Device::Cpu, None)
        .unwrap();

    let stages = model
        .encoder
        .forward_substages(&batch.input_ids, &batch.attention_mask)
        .unwrap();

    let py: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string("/tmp/g25diag/encoder_substages.json").unwrap(),
    )
    .unwrap();
    fn flatten(v: &serde_json::Value, out: &mut Vec<f32>) {
        if let Some(arr) = v.as_array() {
            for x in arr {
                flatten(x, out);
            }
        } else if let Some(n) = v.as_f64() {
            out.push(n as f32);
        }
    }
    let md = |key: &str, t: &candle_core::Tensor| -> f32 {
        let rs = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut pyv = Vec::new();
        flatten(&py[key], &mut pyv);
        assert_eq!(pyv.len(), rs.len(), "{key} size mismatch");
        pyv.iter()
            .zip(&rs)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    };
    for (i, (ctx, attn, gelu, out)) in stages.iter().enumerate() {
        println!(
            "layer {i:2}: selfattn={:.6} attn={:.6} ffn_mid={:.6} ffn_out={:.6}",
            md(&format!("l{i}_selfattn"), ctx),
            md(&format!("l{i}_attn"), attn),
            md(&format!("l{i}_ffn_mid"), gelu),
            md(&format!("l{i}_ffn_out"), out),
        );
    }
}

/// Permanent full-matrix parity gate against committed fixtures
/// (`tests/fixtures/g25/`), regenerated via `scripts/dump_g25_fixtures.py`.
///
/// Asserts:
/// - encoder embedding output within 1e-5 of Python
/// - encoder final output within 1e-4 of Python
/// - pooled-candidate pair logits within 0.15 everywhere and within 0.05 on
///   decision-relevant scores (Python logit > -5)
#[test]
#[ignore = "downloads model weights (~500MB)"]
fn test_full_matrix_parity() {
    const TOL_ENC_IN: f32 = 1e-5;
    const TOL_ENC_OUT: f32 = 1e-4;
    const TOL_RELEVANT: f32 = 0.05;
    const TOL_GLOBAL: f32 = 0.15;

    let model_id = "fastino/gliner2.5-small-v1";
    let hf_config = std::fs::read_to_string(download_from_hub(model_id, "config.json")).unwrap();
    let mut config = gliner25_small();
    config.boundary = BoundaryConfig::from_hf_config_json(&hf_config);
    let mut model = gliner2_rs::model::Extractor::new(&config).unwrap();
    model
        .load_weights(download_from_hub(model_id, "model.safetensors"))
        .unwrap();
    let boundary = model.boundary.as_ref().unwrap();

    let tok =
        tokenizers::Tokenizer::from_file(download_from_hub(model_id, "tokenizer.json")).unwrap();
    let collator = gliner2_rs::batch::ExtractorCollator::with_hf_tokenizer(
        gliner2_rs::tokenizer::WhitespaceTokenizer::new(),
        tok,
        false,
        config.max_len,
    );

    let text = "Apple CEO Tim Cook announced the new iPhone 15 in Cupertino.";
    let schema_json = serde_json::json!({"entities": ["person", "organization", "location"]});
    let batch = collator
        .collate(&[(text.to_string(), schema_json)])
        .unwrap()
        .to(candle_core::Device::Cpu, None)
        .unwrap();

    // Encoder stage comparison.
    let stages = model
        .encoder
        .forward_debug(&batch.input_ids, &batch.attention_mask)
        .unwrap();
    let enc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("tests/fixtures/g25/encoder.json").unwrap())
            .unwrap();
    fn flatten(v: &serde_json::Value, out: &mut Vec<f32>) {
        if let Some(arr) = v.as_array() {
            for x in arr {
                flatten(x, out);
            }
        } else if let Some(n) = v.as_f64() {
            out.push(n as f32);
        }
    }
    let diff_vs = |fixture: &serde_json::Value, t: &candle_core::Tensor| -> f32 {
        let rs = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut fv = Vec::new();
        flatten(fixture, &mut fv);
        assert_eq!(fv.len(), rs.len());
        fv.iter()
            .zip(&rs)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    };
    let d_in = diff_vs(&enc["l0_in"], &stages[0]);
    let d_out = diff_vs(&enc["final"], &stages[12]);
    println!("encoder: l0_in={d_in:.2e} final={d_out:.2e}");
    assert!(d_in < TOL_ENC_IN, "embedding divergence {d_in}");
    assert!(d_out < TOL_ENC_OUT, "encoder divergence {d_out}");

    // Build word/query states like extract_sample does.
    let final_state = stages[12].squeeze(0).unwrap();
    let h = final_state.dims()[1];
    let rows = final_state.to_vec2::<f32>().unwrap();
    let count = batch.text_word_counts[0];
    let twi = batch
        .text_word_indices
        .as_ref()
        .unwrap()
        .to_vec2::<i64>()
        .unwrap();
    let words: Vec<usize> = twi[0][..count].iter().map(|&v| v as usize).collect();
    let mut flat_text = Vec::with_capacity(count * h);
    for &w in &words {
        flat_text.extend_from_slice(&rows[w]);
    }
    let text_states =
        candle_core::Tensor::from_slice(&flat_text, (count, h), &candle_core::Device::Cpu).unwrap();

    let specials = &batch.schema_special_indices[0][0];
    let schema_tokens = &batch.schema_tokens_list[0][0];
    let mut sp = 0;
    let mut pending: Option<usize> = None;
    let mut markers: Vec<usize> = Vec::new();
    for token in schema_tokens {
        if token.starts_with('[') && token.ends_with(']') {
            let pos = specials[sp];
            sp += 1;
            pending = if token == "[E]" { Some(pos) } else { None };
            continue;
        }
        if let Some(pos) = pending.take() {
            markers.push(pos);
        }
    }
    let qn = markers.len();
    let mut flat_q = Vec::with_capacity(qn * h);
    for pos in &markers {
        flat_q.extend_from_slice(&rows[*pos]);
    }
    let query_states =
        candle_core::Tensor::from_slice(&flat_q, (qn, h), &candle_core::Device::Cpu).unwrap();

    let scored = boundary
        .score_sample(&text_states, count, &query_states)
        .unwrap();
    use std::collections::HashMap;
    let mut rs_by_span: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
    for ci in 0..scored.starts.len() {
        if !scored.valid[ci] {
            continue;
        }
        rs_by_span
            .entry((scored.starts[ci], scored.ends[ci]))
            .or_default()
            .extend((0..qn).map(|qi| scored.scores[ci][qi]));
    }

    let pool: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("tests/fixtures/g25/pool.json").unwrap())
            .unwrap();
    let idx = pool["indices"].as_array().unwrap();
    let lg = pool["logits"].as_array().unwrap();
    let pyq_count = idx.len();
    let pyc = idx[0].as_array().unwrap().len();
    assert_eq!(pyq_count, qn);
    let mut diffs: Vec<(f32, f32, usize, usize)> = Vec::new();
    for ci in 0..pyc {
        for qi in 0..qn {
            let pair = &idx[qi][ci];
            let s = pair[0].as_i64().unwrap() as usize;
            let e = pair[1].as_i64().unwrap() as usize;
            let rs_row = match rs_by_span.get(&(s, e)) {
                Some(v) => v,
                None => continue,
            };
            let pyv = lg[qi][ci].as_f64().unwrap() as f32;
            if pyv.abs() > 100.0 {
                continue;
            } // MASK_LOGIT padding slot
            diffs.push((rs_row[qi], pyv, s, e));
        }
    }
    assert!(
        !diffs.is_empty(),
        "no overlapping candidates with fixture pool"
    );
    let gmax = diffs
        .iter()
        .map(|d| (d.0 - d.1).abs())
        .fold(0.0f32, f32::max);
    let rmax = diffs
        .iter()
        .filter(|d| d.1 > -5.0)
        .map(|d| (d.0 - d.1).abs())
        .fold(0.0f32, f32::max);
    println!(
        "pair logits: n={} global={gmax:.4} relevant={rmax:.4}",
        diffs.len()
    );
    assert!(gmax < TOL_GLOBAL, "global logit divergence {gmax}");
    assert!(rmax < TOL_RELEVANT, "decision-relevant divergence {rmax}");
}
/// Compare raw encoder token embeddings vs Python dump.
#[test]
#[ignore = "encoder parity debug; requires /tmp/g25diag dumps"]
fn test_encoder_parity() {
    let model_id = "fastino/gliner2.5-small-v1";
    let hf_config_path = download_from_hub(model_id, "config.json");
    let hf_config = std::fs::read_to_string(&hf_config_path).expect("read config.json");
    let mut config = gliner25_small();
    config.boundary = BoundaryConfig::from_hf_config_json(&hf_config);

    let mut model = gliner2_rs::model::Extractor::new(&config).expect("construct extractor");
    let weights_path = download_from_hub(model_id, "model.safetensors");
    model.load_weights(&weights_path).expect("load weights");

    let tok_path = download_from_hub(model_id, "tokenizer.json");
    let hf_tok = tokenizers::Tokenizer::from_file(&tok_path).expect("load tokenizer");
    let ws_tok = gliner2_rs::tokenizer::WhitespaceTokenizer::new();
    let collator = gliner2_rs::batch::ExtractorCollator::with_hf_tokenizer(
        ws_tok,
        hf_tok,
        false,
        config.max_len,
    );

    let schema_json = serde_json::json!({
        "entities": ["person", "organization", "location"]
    });
    let text = "Apple CEO Tim Cook announced the new iPhone 15 in Cupertino.";
    let batch = collator
        .collate(&[(text.to_string(), schema_json)])
        .expect("collate")
        .to(candle_core::Device::Cpu, None)
        .expect("to device");

    let embs = model
        .encoder
        .forward(&batch.input_ids, &batch.attention_mask)
        .expect("encoder forward")
        .narrow(0, 0, 1)
        .unwrap()
        .squeeze(0)
        .unwrap();
    let rs: Vec<Vec<f32>> = embs.to_vec2::<f32>().unwrap();

    let py: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("/tmp/g25diag/token_embs.json").unwrap())
            .unwrap();
    let py_shape: Vec<usize> = py["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let py_data: Vec<f32> = py["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    println!("rust shape {:?} python shape {:?}", embs.dims(), py_shape);
    let (s_len, s_h) = (py_shape[0], py_shape[1]);
    // per-position max diff
    let mut worst: Vec<(usize, f32)> = Vec::new();
    for pos in 0..s_len.min(rs.len()) {
        let mut md: f32 = 0.0;
        for j in 0..s_h {
            md = md.max((py_data[pos * s_h + j] - rs[pos][j]).abs());
        }
        worst.push((pos, md));
    }
    worst.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("worst positions: {:?}", &worst[..8.min(worst.len())]);
    let global: f32 = worst.iter().map(|(_, d)| *d).fold(0.0, f32::max);
    println!("global max_diff={global:.6}");
}

/// Staged encoder parity: embeddings + per-layer outputs vs Python dumps.
#[test]
#[ignore = "staged parity debug; requires /tmp/g25diag dumps"]
fn test_staged_parity() {
    let model_id = "fastino/gliner2.5-small-v1";
    let hf_config_path = download_from_hub(model_id, "config.json");
    let hf_config = std::fs::read_to_string(&hf_config_path).expect("read config.json");
    let mut config = gliner25_small();
    config.boundary = BoundaryConfig::from_hf_config_json(&hf_config);

    let mut model = gliner2_rs::model::Extractor::new(&config).expect("construct extractor");
    let weights_path = download_from_hub(model_id, "model.safetensors");
    model.load_weights(&weights_path).expect("load weights");

    let tok_path = download_from_hub(model_id, "tokenizer.json");
    let hf_tok = tokenizers::Tokenizer::from_file(&tok_path).expect("load tokenizer");
    let ws_tok = gliner2_rs::tokenizer::WhitespaceTokenizer::new();
    let collator = gliner2_rs::batch::ExtractorCollator::with_hf_tokenizer(
        ws_tok,
        hf_tok,
        false,
        config.max_len,
    );

    let schema_json = serde_json::json!({
        "entities": ["person", "organization", "location"]
    });
    let text = "Apple CEO Tim Cook announced the new iPhone 15 in Cupertino.";
    let batch = collator
        .collate(&[(text.to_string(), schema_json)])
        .expect("collate")
        .to(candle_core::Device::Cpu, None)
        .expect("to device");

    let stages = model
        .encoder
        .forward_debug(&batch.input_ids, &batch.attention_mask)
        .unwrap();

    fn compare(path: &str, rs_flat: &[f32]) -> f32 {
        let py: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let py_data: Vec<f32> = py["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let mut md: f32 = 0.0;
        for (a, b) in py_data.iter().zip(rs_flat) {
            md = md.max((a - b).abs());
        }
        md
    }

    // embeddings
    let emb: Vec<f32> = stages[0]
        .narrow(0, 0, 1)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    println!(
        "embeddings max_diff={:.6}",
        compare("/tmp/g25diag/embeddings.json", &emb)
    );
    for (i, st) in stages.iter().enumerate().skip(1) {
        let flat: Vec<f32> = st
            .narrow(0, 0, 1)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let path = if i == stages.len() - 1 {
            "/tmp/g25diag/token_embs.json".to_string()
        } else {
            format!("/tmp/g25diag/layer_{}.json", i - 1)
        };
        println!("stage_{i} ({path}) max_diff={:.6}", compare(&path, &flat));
    }
}
