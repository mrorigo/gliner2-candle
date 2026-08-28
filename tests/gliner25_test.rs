//! GLiNER2.5 integration tests.
//!
//! These tests download real GLiNER2.5 checkpoints from HuggingFace Hub
//! (Apache 2.0) and verify Phase-0 scaffolding: architecture detection,
//! config parsing, and zero-mismatch weight loading of the boundary model.
//!
//! Gated behind the `hub` feature / env var like the other hub tests.

use gliner2_rs::config::{Architecture, BoundaryConfig};
use std::path::PathBuf;

fn download_from_hub(repo_id: &str, filename: &str) -> PathBuf {
    use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};

    let repo = Repo::with_revision(repo_id.to_string(), RepoType::Model, "main".to_string());

    let api = ApiBuilder::new()
        .with_progress(true)
        .build()
        .expect("Failed to build HF API");

    let repo_api = api.repo(repo);
    repo_api
        .get(filename)
        .unwrap_or_else(|_| panic!("Failed to download {filename}"))
}

#[test]
fn test_gliner25_architecture_detection() {
    assert_eq!(
        Architecture::detect("fastino/gliner2-base-v1"),
        Architecture::Gliner2
    );
    assert_eq!(
        Architecture::detect("fastino/gliner2.5-base-v1"),
        Architecture::Gliner25
    );
    assert_eq!(
        Architecture::detect("fastino/gliner2.5-small-v1"),
        Architecture::Gliner25
    );
    assert_eq!(
        Architecture::detect("fastino/gliner2.5-multi-v1"),
        Architecture::Gliner25
    );

    let cfg = BoundaryConfig::default();
    assert_eq!(cfg.start_top_k, 24);
}

/// Load the real GLiNER2.5 base checkpoint and verify every boundary-head
/// tensor maps with correct shapes (Phase-0 acceptance criterion).
#[test]
#[ignore = "downloads ~770MB; run explicitly with --ignored"]
fn test_gliner25_base_checkpoint_loads() {
    let model_id = "fastino/gliner2.5-base-v1";

    // Parse upstream config.json for boundary settings.
    let hf_config_path = download_from_hub(model_id, "config.json");
    let hf_config =
        std::fs::read_to_string(&hf_config_path).expect("failed to read downloaded config.json");

    assert_eq!(
        Architecture::from_hf_config_json(&hf_config),
        Architecture::Gliner25,
        "upstream config.json must declare the boundary architecture"
    );

    let mut config = gliner2_rs::config::presets::gliner25_base();
    config.boundary = BoundaryConfig::from_hf_config_json(&hf_config);
    assert_eq!(config.boundary.start_top_k, 24);
    assert_eq!(config.boundary.pool_size, 192);

    // Build model and load weights (shape validation happens eagerly).
    let _device = candle_core::Device::Cpu;
    let mut model = gliner2_rs::model::Extractor::new(&config)
        .expect("failed to construct extractor for GLiNER2.5");
    assert!(
        model.boundary.is_none(),
        "boundary loads at weight-load time"
    );

    let weights_path = download_from_hub(model_id, "model.safetensors");
    model
        .load_weights(&weights_path)
        .expect("GLiNER2.5 base weights must load with zero shape mismatches");

    assert!(
        model.boundary.is_some(),
        "boundary head must be populated after loading"
    );
    assert!(model.is_loaded);
}
