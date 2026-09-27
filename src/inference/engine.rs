// Rust guideline compliant 2026-04-03
//! Inference engine for GLiNER2.
//!
//! This module provides the main `GLiNER2` struct that serves as the
//! user-facing API for information extraction. It coordinates model loading,
//! schema processing, batch inference, and result formatting.
//!
//! # Example
//!
//! ```ignore
//! use gliner2_candle::GLiNER2;
//! use gliner2_candle::schema::SchemaBuilder;
//!
//! // Load model
//! let model = GLiNER2::from_pretrained("fastino/gliner2-base-v1")?;
//!
//! // Build schema
//! let schema = SchemaBuilder::new()
//!     .entity("person").description("Names of people").done()
//!     .entity("company").done()
//!     .build()?;
//!
//! // Extract entities
//! let result = model.extract_entities("Apple CEO Tim Cook", &schema)?;
//! println!("{:?}", result);
//! ```

use std::path::Path;

use candle_core::{Device, Tensor};
use serde_json::Value as JsonValue;
use tokenizers::Tokenizer as HfTokenizer;

use crate::batch::{ExtractorCollator, PreprocessedBatch};
use crate::config::ExtractorConfig;
use crate::error::{GlinerError, Result};
use crate::model::Extractor;
use crate::schema::builder::SchemaBuilder;
use crate::schema::types::{RegexValidator, Schema};
use crate::tokenizer::WhitespaceTokenizer;

/// Extraction result for a single text.
pub type ExtractionResult = JsonValue;

/// Batch extraction results for multiple texts.
pub type BatchExtractionResult = Vec<ExtractionResult>;

/// Main GLiNER2 inference engine.
///
/// This struct provides the primary API for running GLiNER2 inference.
/// It handles model loading, schema processing, batch inference, and
/// result formatting.
///
/// # Thread Safety
///
/// `GLiNER2` is `Send + Sync` and can be shared across threads using `Arc`.
///
/// # Example
///
/// ```ignore
/// use gliner2_candle::GLiNER2;
///
/// let model = GLiNER2::from_pretrained("fastino/gliner2-base-v1")?;
/// let result = model.extract_entities(
///     "Apple CEO Tim Cook announced iPhone 15.",
///     &["company", "person", "product"]
/// )?;
/// ```
#[derive(Debug)]
pub struct GLiNER2 {
    /// The underlying extractor model.
    model: Extractor,
    /// Batch collator for inference.
    collator: ExtractorCollator,
    /// Default confidence threshold.
    default_threshold: f32,
    /// Device for inference.
    device: Device,
}

impl GLiNER2 {
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /// Create a new GLiNER2 engine from configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The extractor configuration.
    ///
    /// # Returns
    ///
    /// A new `GLiNER2` engine.
    ///
    /// # Errors
    ///
    /// Returns an error if model initialization fails.
    pub fn new(config: &ExtractorConfig) -> Result<Self> {
        let model = Extractor::new(config)?;
        let ws_tokenizer = WhitespaceTokenizer::new();

        // Try to load HuggingFace tokenizer (default behavior)
        let hf_tokenizer = Self::load_hf_tokenizer(config);
        let collator = if let Some(hf_tok) = hf_tokenizer {
            ExtractorCollator::with_hf_tokenizer(
                ws_tokenizer.clone(),
                hf_tok,
                false,
                config.max_len,
            )
        } else {
            ExtractorCollator::with_max_len(ws_tokenizer.clone(), false, config.max_len)
        };

        let device = match config.device.as_str() {
            "cpu" => Device::Cpu,
            "cuda" => Device::cuda_if_available(0).unwrap_or(Device::Cpu),
            _ => Device::Cpu,
        };

        Ok(Self {
            model,
            collator,
            default_threshold: 0.5,
            device,
        })
    }

    /// Load a GLiNER2 model from a pretrained repository or local path.
    ///
    /// # Arguments
    ///
    /// * `model_name_or_path` - HuggingFace model name or local path.
    ///
    /// # Returns
    ///
    /// A loaded `GLiNER2` engine.
    ///
    /// # Errors
    ///
    /// Returns an error if model initialization fails, model weights cannot be
    /// loaded, or a local path is invalid.
    pub fn from_pretrained(model_name_or_path: impl AsRef<Path>) -> Result<Self> {
        let input = model_name_or_path.as_ref();
        let input_str = input.to_string_lossy().to_string();
        let is_local = input.exists();
        let mut config = ExtractorConfig::new(input_str.clone());

        // Detect the architecture family (GLiNER2 vs GLiNER2.5 boundary).
        config.architecture = crate::config::Architecture::detect(&input_str);

        // For remote models, download and parse config.json for encoder dimensions.
        if !is_local {
            if let Some(hf_config_path) = Self::download_hf_config(&input_str)
                && let Ok(hf_config_str) = std::fs::read_to_string(&hf_config_path)
            {
                // config.json is authoritative: "span" vs "boundary". Name matching
                // alone misreads e.g. "GLiNER2.5-Decide", which is a span checkpoint.
                config.architecture =
                    crate::config::Architecture::from_hf_config_json(&hf_config_str);
                // Parse boundary config for GLiNER2.5
                config.boundary =
                    crate::config::BoundaryConfig::from_hf_config_json(&hf_config_str);
            }
            // Encoder dims are in a separate encoder_config/config.json
            if let Some(enc_path) = Self::download_encoder_config(&input_str)
                && let Ok(enc_str) = std::fs::read_to_string(&enc_path)
                && let Ok(enc) = serde_json::from_str::<serde_json::Value>(&enc_str)
            {
                if let Some(vs) = enc.get("vocab_size").and_then(|v| v.as_u64()) {
                    config.vocab_size = vs as usize;
                }
                if let Some(hs) = enc.get("hidden_size").and_then(|v| v.as_u64()) {
                    config.hidden_size = hs as usize;
                }
                if let Some(nl) = enc.get("num_hidden_layers").and_then(|v| v.as_u64()) {
                    config.num_hidden_layers = nl as usize;
                }
                if let Some(nh) = enc.get("num_attention_heads").and_then(|v| v.as_u64()) {
                    config.num_attention_heads = nh as usize;
                }
                if let Some(is) = enc.get("intermediate_size").and_then(|v| v.as_u64()) {
                    config.intermediate_size = is as usize;
                }
            }
        } else {
            config.boundary =
                crate::config::BoundaryConfig::detect(input.to_string_lossy().as_ref());
        }

        let mut model = Extractor::new(&config)?;

        // Load weights from local path or download from Hub for model IDs.
        if is_local {
            model.load_weights(input)?;
        } else if let Some(model_path) = Self::download_model_weights(&input_str) {
            model.load_weights(model_path)?;
        }

        let ws_tokenizer = WhitespaceTokenizer::new();

        // Try local tokenizer first for local paths; otherwise use config/HF Hub.
        let hf_tokenizer = if is_local {
            Self::load_hf_tokenizer_from_path(input).or_else(|| Self::load_hf_tokenizer(&config))
        } else {
            Self::load_hf_tokenizer(&config)
        };

        let collator = if let Some(hf_tok) = hf_tokenizer {
            ExtractorCollator::with_hf_tokenizer(
                ws_tokenizer.clone(),
                hf_tok,
                false,
                config.max_len,
            )
        } else {
            ExtractorCollator::with_max_len(ws_tokenizer.clone(), false, config.max_len)
        };

        let device = match config.device.as_str() {
            "cpu" => Device::Cpu,
            "cuda" => Device::cuda_if_available(0).unwrap_or(Device::Cpu),
            _ => Device::Cpu,
        };

        Ok(Self {
            model,
            collator,
            default_threshold: 0.5,
            device,
        })
    }

    /// Download model weights (`model.safetensors`) from HuggingFace Hub.
    fn download_model_weights(model_id: &str) -> Option<std::path::PathBuf> {
        use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};

        let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, "main".to_string());

        let api = ApiBuilder::new().with_progress(true).build().ok()?;

        let repo_api = api.repo(repo);
        match repo_api.get("model.safetensors") {
            Ok(path) => Some(path),
            Err(e) => {
                tracing::warn!("Failed to download model weights for '{}': {}", model_id, e);
                None
            }
        }
    }

    /// Download `config.json` from HuggingFace Hub for a remote model.
    fn download_hf_config(model_id: &str) -> Option<std::path::PathBuf> {
        use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};

        let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, "main".to_string());
        let api = ApiBuilder::new().with_progress(false).build().ok()?;
        let repo_api = api.repo(repo);
        match repo_api.get("config.json") {
            Ok(path) => Some(path),
            Err(e) => {
                tracing::warn!("Failed to download config.json for '{}': {}", model_id, e);
                None
            }
        }
    }

    /// Download `encoder_config/config.json` from HuggingFace Hub.
    fn download_encoder_config(model_id: &str) -> Option<std::path::PathBuf> {
        use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};

        let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, "main".to_string());
        let api = ApiBuilder::new().with_progress(false).build().ok()?;
        let repo_api = api.repo(repo);
        match repo_api.get("encoder_config/config.json") {
            Ok(path) => Some(path),
            Err(e) => {
                tracing::debug!("No encoder_config/config.json for '{}': {}", model_id, e);
                None
            }
        }
    }

    // -------------------------------------------------------------------------
    // Tokenizer Loading
    // -------------------------------------------------------------------------

    /// Load a HuggingFace tokenizer from config.
    ///
    /// Tries to load from `config.tokenizer_path` first, then from local model directory,
    /// then downloads from HuggingFace Hub using the model name.
    ///
    /// Returns `None` only if all loading methods fail (falls back to whitespace tokenizer).
    fn load_hf_tokenizer(config: &ExtractorConfig) -> Option<HfTokenizer> {
        // Try loading from explicit tokenizer path first
        if let Some(ref path) = config.tokenizer_path
            && let Some(tokenizer) = Self::load_hf_tokenizer_from_path(path)
        {
            tracing::info!("Loaded HuggingFace tokenizer from: {:?}", path);
            return Some(tokenizer);
        }

        // Try loading from model name as local directory
        let model_name = &config.model_name;
        let model_path = std::path::Path::new(model_name);
        if model_path.exists()
            && model_path.is_dir()
            && let Some(tokenizer) = Self::load_hf_tokenizer_from_path(model_path)
        {
            tracing::info!(
                "Loaded HuggingFace tokenizer from model directory: {}",
                model_name
            );
            return Some(tokenizer);
        }

        // Try common tokenizer file patterns in current directory
        let tokenizer_files = ["tokenizer.json", "tokenizer_config.json"];
        for file in &tokenizer_files {
            let path = std::path::Path::new(file);
            if path.exists()
                && let Ok(tokenizer) = HfTokenizer::from_file(path)
            {
                tracing::info!("Loaded HuggingFace tokenizer from: {}", file);
                return Some(tokenizer);
            }
        }

        // Download from HuggingFace Hub (default behavior)
        tracing::info!(
            "Downloading tokenizer for '{}' from HuggingFace Hub...",
            model_name
        );
        Self::download_hf_tokenizer(model_name)
    }

    /// Load a HuggingFace tokenizer from a directory path.
    ///
    /// Looks for `tokenizer.json` first (fast), then falls back to
    /// loading from the path directly if it's a JSON file.
    fn load_hf_tokenizer_from_path(path: impl AsRef<std::path::Path>) -> Option<HfTokenizer> {
        let path = path.as_ref();

        // Try loading tokenizer.json first (preferred, fast)
        let tokenizer_json = path.join("tokenizer.json");
        if tokenizer_json.exists()
            && let Ok(tokenizer) = HfTokenizer::from_file(&tokenizer_json)
        {
            return Some(tokenizer);
        }

        // Try loading from the path directly if it's a tokenizer.json file
        if path.extension().is_some_and(|ext| ext == "json")
            && let Ok(tokenizer) = HfTokenizer::from_file(path)
        {
            return Some(tokenizer);
        }

        None
    }

    /// Download a HuggingFace tokenizer from the Hub.
    ///
    /// Downloads `tokenizer.json` for the given model ID and loads it.
    /// The file is cached locally by `hf-hub` for future use.
    fn download_hf_tokenizer(model_id: &str) -> Option<HfTokenizer> {
        use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};

        let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, "main".to_string());

        let api = ApiBuilder::new().with_progress(true).build().ok()?;

        let repo_api = api.repo(repo);

        // Download tokenizer.json
        match repo_api.get("tokenizer.json") {
            Ok(tokenizer_path) => {
                tracing::info!("Downloaded tokenizer.json to: {:?}", tokenizer_path);
                HfTokenizer::from_file(&tokenizer_path).ok()
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to download tokenizer for '{}': {}. \
                     Falling back to whitespace tokenizer.",
                    model_id,
                    e
                );
                None
            }
        }
    }

    // -------------------------------------------------------------------------
    // Schema & Extraction
    // -------------------------------------------------------------------------

    /// Create a new schema builder.
    ///
    /// # Returns
    ///
    /// A new `SchemaBuilder` for constructing extraction schemas.
    pub fn create_schema(&self) -> SchemaBuilder {
        SchemaBuilder::new()
    }

    // -------------------------------------------------------------------------
    // Entity Extraction
    // -------------------------------------------------------------------------

    /// Extract entities from a single text.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text.
    /// * `entity_types` - Entity types to extract (names or with descriptions).
    /// * `threshold` - Confidence threshold (default: 0.5).
    /// * `include_confidence` - Whether to include confidence scores.
    /// * `include_spans` - Whether to include character positions.
    /// * `max_len` - Maximum token length (overrides config).
    ///
    /// # Returns
    ///
    /// Extraction result with entities.
    ///
    /// # Errors
    ///
    /// Returns an error if schema construction, batching, or inference fails.
    pub fn extract_entities(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: Option<f32>,
        include_confidence: bool,
        include_spans: bool,
        max_len: Option<usize>,
    ) -> Result<ExtractionResult> {
        let schema = self.build_entity_schema(entity_types)?;
        self.extract(
            text,
            &schema,
            threshold.unwrap_or(self.default_threshold),
            include_confidence,
            include_spans,
            max_len,
        )
    }

    /// Batch extract entities from multiple texts.
    ///
    /// # Arguments
    ///
    /// * `texts` - Input texts.
    /// * `entity_types` - Entity types to extract.
    /// * `batch_size` - Batch size for processing.
    /// * `threshold` - Confidence threshold.
    /// * `num_workers` - Number of parallel workers.
    /// * `include_confidence` - Whether to include confidence scores.
    /// * `include_spans` - Whether to include character positions.
    /// * `max_len` - Maximum token length.
    ///
    /// # Returns
    ///
    /// Batch extraction results.
    ///
    /// # Errors
    ///
    /// Returns an error if schema construction, batching, or inference fails.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_extract_entities(
        &self,
        texts: &[String],
        entity_types: &[&str],
        batch_size: usize,
        threshold: Option<f32>,
        num_workers: usize,
        include_confidence: bool,
        include_spans: bool,
        max_len: Option<usize>,
    ) -> Result<BatchExtractionResult> {
        let schema = self.build_entity_schema(entity_types)?;
        self.batch_extract(
            texts,
            &schema,
            batch_size,
            threshold.unwrap_or(self.default_threshold),
            num_workers,
            include_confidence,
            include_spans,
            max_len,
        )
    }

    // -------------------------------------------------------------------------
    // Classification
    // -------------------------------------------------------------------------

    /// Classify a single text.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text.
    /// * `tasks` - Classification tasks (task name -> labels).
    /// * `threshold` - Confidence threshold.
    /// * `include_confidence` - Whether to include confidence scores.
    /// * `max_len` - Maximum token length.
    ///
    /// # Returns
    ///
    /// Classification result.
    ///
    /// # Errors
    ///
    /// Returns an error if schema construction, batching, or inference fails.
    pub fn classify_text(
        &self,
        text: &str,
        tasks: &[(String, Vec<String>)],
        threshold: Option<f32>,
        include_confidence: bool,
        max_len: Option<usize>,
    ) -> Result<ExtractionResult> {
        let mut schema_builder = self.create_schema();
        for (task_name, labels) in tasks {
            schema_builder = schema_builder
                .classification(task_name.clone(), labels.clone())
                .threshold(threshold.unwrap_or(0.5))
                .done();
        }
        let schema = schema_builder.build()?;

        self.extract(
            text,
            &schema,
            threshold.unwrap_or(self.default_threshold),
            include_confidence,
            false,
            max_len,
        )
    }

    /// Batch classify multiple texts.
    ///
    /// # Arguments
    ///
    /// * `texts` - Input texts.
    /// * `tasks` - Classification tasks.
    /// * `batch_size` - Batch size.
    /// * `threshold` - Confidence threshold.
    /// * `num_workers` - Number of parallel workers.
    /// * `include_confidence` - Whether to include confidence scores.
    /// * `max_len` - Maximum token length.
    ///
    /// # Returns
    ///
    /// Batch classification results.
    ///
    /// # Errors
    ///
    /// Returns an error if schema construction, batching, or inference fails.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_classify_text(
        &self,
        texts: &[String],
        tasks: &[(String, Vec<String>)],
        batch_size: usize,
        threshold: Option<f32>,
        num_workers: usize,
        include_confidence: bool,
        max_len: Option<usize>,
    ) -> Result<BatchExtractionResult> {
        let mut schema_builder = self.create_schema();
        for (task_name, labels) in tasks {
            schema_builder = schema_builder
                .classification(task_name.clone(), labels.clone())
                .threshold(threshold.unwrap_or(0.5))
                .done();
        }
        let schema = schema_builder.build()?;

        self.batch_extract(
            texts,
            &schema,
            batch_size,
            threshold.unwrap_or(self.default_threshold),
            num_workers,
            include_confidence,
            false,
            max_len,
        )
    }

    // -------------------------------------------------------------------------
    // Relation Extraction
    // -------------------------------------------------------------------------

    /// Extract relations from a single text.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text.
    /// * `relation_types` - Relation types to extract.
    /// * `threshold` - Confidence threshold.
    /// * `include_confidence` - Whether to include confidence scores.
    /// * `include_spans` - Whether to include character positions.
    /// * `max_len` - Maximum token length.
    ///
    /// # Returns
    ///
    /// Extraction result with relations.
    ///
    /// # Errors
    ///
    /// Returns an error if schema construction, batching, or inference fails.
    pub fn extract_relations(
        &self,
        text: &str,
        relation_types: &[&str],
        threshold: Option<f32>,
        include_confidence: bool,
        include_spans: bool,
        max_len: Option<usize>,
    ) -> Result<ExtractionResult> {
        let mut schema_builder = self.create_schema();
        for rel_type in relation_types {
            schema_builder = schema_builder.relation(rel_type.to_string()).done();
        }
        let schema = schema_builder.build()?;

        self.extract(
            text,
            &schema,
            threshold.unwrap_or(self.default_threshold),
            include_confidence,
            include_spans,
            max_len,
        )
    }

    /// Batch extract relations from multiple texts.
    ///
    /// # Arguments
    ///
    /// * `texts` - Input texts.
    /// * `relation_types` - Relation types to extract.
    /// * `batch_size` - Batch size.
    /// * `threshold` - Confidence threshold.
    /// * `num_workers` - Number of parallel workers.
    /// * `include_confidence` - Whether to include confidence scores.
    /// * `include_spans` - Whether to include character positions.
    /// * `max_len` - Maximum token length.
    ///
    /// # Returns
    ///
    /// Batch extraction results.
    ///
    /// # Errors
    ///
    /// Returns an error if schema construction, batching, or inference fails.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_extract_relations(
        &self,
        texts: &[String],
        relation_types: &[&str],
        batch_size: usize,
        threshold: Option<f32>,
        num_workers: usize,
        include_confidence: bool,
        include_spans: bool,
        max_len: Option<usize>,
    ) -> Result<BatchExtractionResult> {
        let mut schema_builder = self.create_schema();
        for rel_type in relation_types {
            schema_builder = schema_builder.relation(rel_type.to_string()).done();
        }
        let schema = schema_builder.build()?;

        self.batch_extract(
            texts,
            &schema,
            batch_size,
            threshold.unwrap_or(self.default_threshold),
            num_workers,
            include_confidence,
            include_spans,
            max_len,
        )
    }

    // -------------------------------------------------------------------------
    // General Extraction
    // -------------------------------------------------------------------------

    /// Extract from a single text using a custom schema.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text.
    /// * `schema` - The extraction schema.
    /// * `threshold` - Confidence threshold.
    /// * `include_confidence` - Whether to include confidence scores.
    /// * `include_spans` - Whether to include character positions.
    /// * `max_len` - Maximum token length.
    ///
    /// # Returns
    ///
    /// Extraction result.
    ///
    /// # Errors
    ///
    /// Returns an error if batching or inference fails, or no result is
    /// produced for the input sample.
    pub fn extract(
        &self,
        text: &str,
        schema: &Schema,
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
        max_len: Option<usize>,
    ) -> Result<ExtractionResult> {
        let results = self.batch_extract(
            &[text.to_string()],
            schema,
            1,
            threshold,
            0,
            include_confidence,
            include_spans,
            max_len,
        )?;

        results
            .into_iter()
            .next()
            .ok_or_else(|| GlinerError::inference("No results returned from extraction"))
    }

    /// Batch extract from multiple texts using a schema.
    ///
    /// # Arguments
    ///
    /// * `texts` - Input texts.
    /// * `schema` - The extraction schema.
    /// * `batch_size` - Batch size for processing.
    /// * `threshold` - Confidence threshold.
    /// * `num_workers` - Number of parallel workers for preprocessing.
    /// * `include_confidence` - Whether to include confidence scores.
    /// * `include_spans` - Whether to include character positions.
    /// * `max_len` - Maximum token length.
    ///
    /// # Returns
    ///
    /// Batch extraction results.
    ///
    /// # Errors
    ///
    /// Returns an error if collation, model forward, or post-processing fails.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_extract(
        &self,
        texts: &[String],
        schema: &Schema,
        batch_size: usize,
        threshold: f32,
        _num_workers: usize,
        include_confidence: bool,
        include_spans: bool,
        max_len: Option<usize>,
    ) -> Result<BatchExtractionResult> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Set model to evaluation mode
        let model = &self.model;
        // Note: eval() requires mutable access, but we use immutable reference here
        // The is_training flag is only used internally and doesn't affect inference

        // Convert schema to dict format; on the boundary path, attribute
        // labels become hidden entity queries in the collated prompt.
        let is_boundary = model.architecture == crate::config::Architecture::Gliner25;
        let (schema_dict, attrs_runtime) = if is_boundary {
            match schema.expanded_with_attributes() {
                Ok((expanded, prompts)) => {
                    let runtime = (!prompts.is_empty()).then(|| {
                        crate::inference::boundary::AttributesRuntime {
                            prompt_by_label: prompts,
                            groups: schema
                                .entity_attribute_groups
                                .iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect(),
                        }
                    });
                    (expanded.to_dict(), runtime)
                }
                Err(e) => return Err(e),
            }
        } else {
            (schema.to_dict(), None)
        };
        let attrs_ref = attrs_runtime.as_ref();
        // Match the Python reference: chunk documents beyond the default chunk
        // size (384 words) rather than waiting for the full encoder window —
        // boundary-model recall collapses on long single-window inputs.
        const MAX_ENCODER_WORDS: usize = 4096usize.saturating_sub(64);
        let context_words = crate::chunking::DEFAULT_CHUNK_SIZE
            .min(
                max_len
                    .or(self.config().max_len)
                    .map_or(MAX_ENCODER_WORDS, |m| m.saturating_sub(64).max(32)),
            )
            .max(32);

        // Expand long documents into overlapping chunks (boundary path only).
        // Each entry maps to (original_text_index, Option<&[TextChunk]>, chunk_idx).
        struct Expanded {
            _original: usize,
            text: String,
        }
        let mut expanded: Vec<Expanded> = Vec::new();
        let mut chunk_lists: Vec<Option<Vec<crate::chunking::TextChunk>>> =
            Vec::with_capacity(texts.len());
        for (i, text) in texts.iter().enumerate() {
            let needs_chunk = is_boundary && text.split_whitespace().count() > context_words;
            if needs_chunk {
                let chunks = crate::chunking::split_text_into_chunks(
                    text,
                    context_words,
                    crate::chunking::DEFAULT_CHUNK_OVERLAP.min(context_words / 2),
                )?;
                for chunk in chunks.iter() {
                    expanded.push(Expanded {
                        _original: i,
                        text: chunk.text.clone(),
                    });
                }
                chunk_lists.push(Some(chunks));
            } else {
                expanded.push(Expanded {
                    _original: i,
                    text: text.clone(),
                });
                chunk_lists.push(None);
            }
        }

        // Process in batches
        let mut all_expanded_results: Vec<JsonValue> = Vec::with_capacity(expanded.len());
        for group in expanded.chunks(batch_size.max(1)) {
            let samples: Vec<(String, JsonValue)> = group
                .iter()
                .map(|e| (e.text.clone(), schema_dict.clone()))
                .collect();
            let results = self.process_batch(
                &samples,
                model,
                threshold,
                include_confidence,
                include_spans,
                max_len,
                attrs_ref,
            )?;
            all_expanded_results.extend(results);
        }

        // Regroup chunk results back to their original documents.
        let mut all_results: BatchExtractionResult = Vec::with_capacity(texts.len());
        let mut cursor = 0usize;
        for (i, text) in texts.iter().enumerate() {
            match &chunk_lists[i] {
                None => {
                    all_results.push(all_expanded_results[cursor].clone());
                    cursor += 1;
                }
                Some(chunks) => {
                    let chunk_results: Vec<JsonValue> =
                        all_expanded_results[cursor..cursor + chunks.len()].to_vec();
                    cursor += chunks.len();
                    let merged = crate::chunking::merge_chunk_results(
                        text,
                        chunks,
                        chunk_results,
                        include_confidence,
                        include_spans,
                        crate::chunking::MergePolicy::Allow,
                    )?;
                    all_results.push(merged);
                }
            }
        }

        Ok(all_results)
    }

    // -------------------------------------------------------------------------
    // Internal Methods
    // -------------------------------------------------------------------------

    /// Process a single batch of samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Batch of (text, schema) pairs.
    /// * `model` - The extractor model.
    /// * `threshold` - Confidence threshold.
    /// * `include_confidence` - Whether to include confidence scores.
    /// * `include_spans` - Whether to include character positions.
    /// * `max_len` - Maximum token length.
    ///
    /// # Returns
    ///
    /// Extraction results for the batch.
    #[allow(clippy::too_many_arguments)]
    fn process_batch(
        &self,
        samples: &[(String, JsonValue)],
        model: &Extractor,
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
        max_len: Option<usize>,
        attrs: Option<&crate::inference::boundary::AttributesRuntime>,
    ) -> Result<Vec<ExtractionResult>> {
        // Collate batch
        let collator = match max_len {
            Some(override_len) => self.collator.with_runtime_max_len(Some(override_len)),
            None => self.collator.clone(),
        };
        let batch = collator.collate(samples)?;

        // Move to device
        let batch = batch.to(self.device.clone(), None)?;

        // GLiNER2.5 boundary path.
        if model.architecture == crate::config::Architecture::Gliner25 {
            let boundary = model.boundary.as_ref().ok_or_else(|| {
                GlinerError::inference("GLiNER2.5 model weights not loaded (boundary head missing)")
            })?;
            let t_enc = std::time::Instant::now();
            let token_embs = model
                .encoder
                .forward(&batch.input_ids, &batch.attention_mask)?;
            let t_enc = t_enc.elapsed();
            let t_score = std::time::Instant::now();
            let mut results = Vec::with_capacity(batch.batch_size());
            for sample_idx in 0..batch.batch_size() {
                let states = token_embs.narrow(0, sample_idx, 1)?.squeeze(0)?;
                let sample_result = crate::inference::boundary::extract_sample(
                    boundary,
                    &states,
                    &batch,
                    sample_idx,
                    threshold,
                    include_confidence,
                    include_spans,
                    attrs,
                )?;
                let _ = batch.original_texts.get(sample_idx);
                results.push(sample_result);
            }
            if std::env::var("GLINER2_PROFILE").is_ok() {
                eprintln!(
                    "PROFILE encoder={t_enc:?} boundary_total={:?} batch={}",
                    t_score.elapsed(),
                    batch.batch_size()
                );
            }
            return Ok(results);
        }

        // Run forward pass
        let output = model.forward(&batch)?;

        // Extract results for each sample
        let mut results = Vec::with_capacity(batch.batch_size());

        for sample_idx in 0..batch.batch_size() {
            let result = self.extract_sample(
                &output,
                &batch,
                sample_idx,
                threshold,
                include_confidence,
                include_spans,
            )?;
            results.push(result);
        }

        Ok(results)
    }

    /// Extract results for a single sample from model output.
    ///
    /// # Arguments
    ///
    /// * `output` - Model output.
    /// * `batch` - Preprocessed batch.
    /// * `sample_idx` - Sample index.
    /// * `threshold` - Confidence threshold.
    /// * `include_confidence` - Whether to include confidence scores.
    /// * `include_spans` - Whether to include character positions.
    ///
    /// # Returns
    ///
    /// Extraction result for the sample.
    fn extract_sample(
        &self,
        output: &crate::model::ExtractorOutput,
        batch: &PreprocessedBatch,
        sample_idx: usize,
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<ExtractionResult> {
        let mut result = serde_json::Map::new();

        // Get task types for this sample
        let task_types = batch.sample_task_types(sample_idx).unwrap_or(&[]);

        // Process each schema/task
        for (schema_idx, task_type) in task_types.iter().enumerate() {
            let task_result = match task_type.as_str() {
                "entities" => self.extract_entities_from_output(
                    output,
                    batch,
                    sample_idx,
                    schema_idx,
                    threshold,
                    include_confidence,
                    include_spans,
                )?,
                "classifications" => self.extract_classification_from_output(
                    output,
                    batch,
                    sample_idx,
                    schema_idx,
                    threshold,
                    include_confidence,
                )?,
                "relations" => self.extract_relations_from_output(
                    output,
                    batch,
                    sample_idx,
                    schema_idx,
                    threshold,
                    include_confidence,
                    include_spans,
                )?,
                "json_structures" => self.extract_structures_from_output(
                    output,
                    batch,
                    sample_idx,
                    schema_idx,
                    threshold,
                    include_confidence,
                    include_spans,
                )?,
                _ => JsonValue::Null,
            };

            // Add to result based on task type
            match task_type.as_str() {
                "entities" => {
                    result.insert("entities".to_string(), task_result);
                }
                "classifications" => {
                    // Token index 2 is the folded prompt string; recover the task name.
                    if let Some(schema_tokens) = batch.schema_tokens(sample_idx, schema_idx)
                        && let Some(prompt_str) = schema_tokens.get(2)
                        && let Some(task_name) = batch
                            .original_schemas
                            .get(sample_idx)
                            .and_then(|s| Self::resolve_classification_task(s, prompt_str))
                    {
                        result.insert(task_name, task_result);
                    }
                }
                "relations" => {
                    if let Some(schema_tokens) = batch.schema_tokens(sample_idx, schema_idx)
                        && schema_tokens.len() > 2
                    {
                        let rel_name = &schema_tokens[2];
                        if let Some(relations) = result.get_mut("relation_extraction") {
                            if let Some(rel_obj) = relations.as_object_mut() {
                                rel_obj.insert(rel_name.clone(), task_result);
                            }
                        } else {
                            let mut rel_obj = serde_json::Map::new();
                            rel_obj.insert(schema_tokens[2].clone(), task_result);
                            result.insert(
                                "relation_extraction".to_string(),
                                JsonValue::Object(rel_obj),
                            );
                        }
                    }
                }
                "json_structures" => {
                    if let Some(schema_tokens) = batch.schema_tokens(sample_idx, schema_idx)
                        && schema_tokens.len() > 2
                    {
                        let struct_name = &schema_tokens[2];
                        result.insert(struct_name.clone(), task_result);
                    }
                }
                _ => {}
            }
        }

        Ok(JsonValue::Object(result))
    }

    /// Extract entities from model output.
    ///
    /// Implements the full GLiNER2 entity extraction pipeline:
    /// 1. Get span representations from model output
    /// 2. Get schema embeddings for entity types
    /// 3. Compute span scores via dot product
    /// 4. Apply threshold filtering
    /// 5. Extract entities with text spans
    #[allow(clippy::too_many_arguments)]
    fn extract_entities_from_output(
        &self,
        output: &crate::model::ExtractorOutput,
        batch: &PreprocessedBatch,
        sample_idx: usize,
        schema_idx: usize,
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<JsonValue> {
        let mut entities = serde_json::Map::new();

        // Get span representations for this sample
        let Some(span_outputs) = output.span_representations.as_ref() else {
            return Ok(JsonValue::Object(entities));
        };
        if sample_idx >= span_outputs.len() {
            return Ok(JsonValue::Object(entities));
        }
        let span_output = &span_outputs[sample_idx];

        // Get schema embeddings for this sample/schema
        let schema_embs = &output.schema_embeddings;
        if sample_idx >= schema_embs.len() {
            return Ok(JsonValue::Object(entities));
        }
        let sample_schema_embs = &schema_embs[sample_idx];
        if schema_idx >= sample_schema_embs.len() {
            return Ok(JsonValue::Object(entities));
        }
        let schema_tokens_embs = &sample_schema_embs[schema_idx];
        if schema_tokens_embs.is_empty() {
            return Ok(JsonValue::Object(entities));
        }

        // Get entity types from schema tokens
        let Some(schema_tokens) = batch.schema_tokens(sample_idx, schema_idx) else {
            return Ok(JsonValue::Object(entities));
        };

        // Find entity type tokens (tokens following [E] markers)
        // Schema format: ["(", "[P]", "entities", "(", "[E]", "location", "[E]", "organization", "[E]", "person", ")", ")"]
        let mut entity_types: Vec<String> = Vec::new();
        let mut entity_type_indices: Vec<usize> = Vec::new();
        let mut special_token_counter = 0;
        for (i, token) in schema_tokens.iter().enumerate() {
            if token.starts_with('[') && token.ends_with(']') {
                if token == "[E]" && i + 1 < schema_tokens.len() {
                    let entity_type = schema_tokens[i + 1].clone();
                    if !entity_type.is_empty() {
                        entity_types.push(entity_type);
                        entity_type_indices.push(special_token_counter);
                    }
                }
                special_token_counter += 1;
            }
        }

        if entity_types.is_empty() {
            return Ok(JsonValue::Object(entities));
        }

        // Get span rep shape: (seq_len, max_width, hidden_size)
        let span_rep = &span_output.span_rep;
        let span_mask = &span_output.span_mask;
        let spans_idx = &span_output.spans_idx;

        let dims = span_rep.dims();
        if dims.len() != 3 {
            return Ok(JsonValue::Object(entities));
        }
        let max_width = dims[1];
        let hidden_size = dims[2];

        // Get text tokens for span extraction
        let text_tokens = batch.sample_text_tokens(sample_idx).unwrap_or(&[]);
        let start_mappings = batch.sample_start_mapping(sample_idx).unwrap_or(&[]);
        let end_mappings = batch.sample_end_mapping(sample_idx).unwrap_or(&[]);

        // Use count_embed + einsum scoring mechanism (matching Python implementation)
        // Step 1: Get entity embeddings (skip [P] token at index 0)
        // entity_type_indices maps to positions in schema_tokens_embs
        let num_entity_types = entity_types.len();
        if num_entity_types == 0 || schema_tokens_embs.len() <= 1 {
            return Ok(JsonValue::Object(entities));
        }

        let mut entity_emb_data: Vec<f32> = Vec::with_capacity(num_entity_types * hidden_size);
        for &emb_idx in &entity_type_indices {
            if emb_idx < schema_tokens_embs.len() {
                let emb = &schema_tokens_embs[emb_idx];
                if let Ok(data) = emb.flatten_all()
                    && let Ok(vec) = data.to_vec1::<f32>()
                {
                    entity_emb_data.extend_from_slice(&vec);
                }
            } else {
                // Pad with zeros if embedding not found
                entity_emb_data.extend(std::iter::repeat_n(0.0f32, hidden_size));
            }
        }

        // Create entity embeddings tensor: (num_entity_types, hidden)
        let entity_embs_tensor = match Tensor::from_vec(
            entity_emb_data,
            (num_entity_types, hidden_size),
            &output.device,
        ) {
            Ok(t) => t,
            Err(_) => return Ok(JsonValue::Object(entities)),
        };

        // Step 2: Predict count using count_pred layer
        // Use [P] token embedding (schema_tokens_embs[0]) for count prediction
        let p_token_emb = &schema_tokens_embs[0];
        let pred_count = if let Ok(output) = self.model.count_pred.predict_count(p_token_emb) {
            output.count.clamp(1, 20)
        } else {
            5
        };
        let pred_count = pred_count.clamp(1, 20);

        // Step 3: Transform entity embeddings using count_embed
        // Output shape: (pred_count, num_entity_types, hidden)
        let Some(count_embed) = self.model.count_embed.as_ref() else {
            return Ok(JsonValue::Object(entities));
        };
        let struct_proj = match count_embed.forward(&entity_embs_tensor, pred_count) {
            Ok(out) => out.embeddings,
            Err(_) => return Ok(JsonValue::Object(entities)),
        };

        // Get struct_proj as flat data: (pred_count, num_entity_types, hidden)
        let struct_proj_data: Vec<f32> = match struct_proj.flatten_all() {
            Ok(t) => t.to_vec1().unwrap_or_default(),
            Err(_) => return Ok(JsonValue::Object(entities)),
        };

        // Get span mask dimensions
        let span_mask_dims = span_mask.dims();
        if span_mask_dims.len() != 2 {
            return Ok(JsonValue::Object(entities));
        }
        let mask_seq_len = span_mask_dims[0];
        let mask_max_width = span_mask_dims[1];

        // Get span mask as flat vector
        let mask_data: Vec<u32> = match span_mask.flatten_all() {
            Ok(t) => t.to_vec1().unwrap_or_default(),
            Err(_) => return Ok(JsonValue::Object(entities)),
        };

        // Get span reps as flat data: (seq_len, max_width, hidden)
        let span_rep_data: Vec<f32> = match span_rep.flatten_all() {
            Ok(t) => t.to_vec1().unwrap_or_default(),
            Err(_) => return Ok(JsonValue::Object(entities)),
        };

        // Get spans indices
        let spans_idx_data: Vec<u32> = match spans_idx.flatten_all() {
            Ok(t) => t.to_vec1().unwrap_or_default(),
            Err(_) => return Ok(JsonValue::Object(entities)),
        };

        // Step 4: Compute scores using einsum-like operation
        // Python: torch.einsum("lkd,bpd->bplk", span_rep, struct_proj)
        // span_rep: (seq_len, max_width, hidden) -> l, k, d
        // struct_proj: (pred_count, num_entity_types, hidden) -> b, p, d
        // Result: (batch=1, pred_count, seq_len, num_entity_types) -> b, p, l, k
        //
        // For each span (i, w) and entity type (entity_idx), compute:
        // score = sum_d(span_rep[i, w, d] * struct_proj[count, entity_idx, d])
        // Then take max over count positions and apply sigmoid

        // Precompute scores for all spans and entity types
        // scores[span_idx][entity_idx] = max over count of sigmoid(score)
        let total_spans = mask_seq_len * mask_max_width;
        let mut span_entity_scores: Vec<Vec<f32>> =
            vec![vec![0.0f32; num_entity_types]; total_spans];

        for i in 0..mask_seq_len {
            for w in 0..mask_max_width {
                let mask_idx = i * mask_max_width + w;
                if mask_idx >= mask_data.len() || mask_data[mask_idx] == 0 {
                    continue; // Invalid span
                }

                // Get span representation: (hidden,)
                let span_start = i * max_width * hidden_size + w * hidden_size;
                if span_start + hidden_size > span_rep_data.len() {
                    continue;
                }
                let span_rep_vec = &span_rep_data[span_start..span_start + hidden_size];

                // Compute scores for each entity type.
                // Match Python entities path: use count index 0 (no max-over-count aggregation).
                for (entity_idx, score_slot) in span_entity_scores[mask_idx]
                    .iter_mut()
                    .enumerate()
                    .take(num_entity_types)
                {
                    // Get struct_proj[0, entity_idx, :]
                    let struct_start = entity_idx * hidden_size;
                    if struct_start + hidden_size > struct_proj_data.len() {
                        continue;
                    }
                    let struct_vec = &struct_proj_data[struct_start..struct_start + hidden_size];

                    // Compute dot product
                    let score: f32 = span_rep_vec
                        .iter()
                        .zip(struct_vec.iter())
                        .map(|(a, b)| a * b)
                        .sum();

                    // Apply sigmoid
                    let prob = 1.0 / (1.0 + (-score).exp());
                    *score_slot = prob;
                }
            }
        }

        // Step 5: Extract entities for each entity type
        let original_text = batch.original_text(sample_idx).unwrap_or_default();
        for (entity_idx, entity_type) in entity_types.iter().enumerate() {
            let mut found_spans: Vec<(String, f32, usize, usize)> = Vec::new();

            for i in 0..mask_seq_len {
                for w in 0..mask_max_width {
                    let mask_idx = i * mask_max_width + w;
                    if mask_idx >= mask_data.len() || mask_data[mask_idx] == 0 {
                        continue;
                    }

                    let prob = span_entity_scores[mask_idx][entity_idx];
                    if prob >= threshold {
                        // Get span boundaries
                        let spans_flat_idx = i * max_width * 2 + w * 2;
                        if spans_flat_idx + 1 < spans_idx_data.len() {
                            let start_pos = spans_idx_data[spans_flat_idx] as usize;
                            let end_pos = spans_idx_data[spans_flat_idx + 1] as usize;

                            let char_start = if start_pos < start_mappings.len() {
                                start_mappings[start_pos]
                            } else {
                                0
                            };
                            let char_end = if end_pos < end_mappings.len() {
                                end_mappings[end_pos]
                            } else {
                                char_start
                            };

                            let entity_text = if !original_text.is_empty() && char_start <= char_end
                            {
                                safe_slice(original_text, char_start, char_end)
                            } else if start_pos < text_tokens.len() && end_pos < text_tokens.len() {
                                text_tokens[start_pos..=end_pos].join(" ")
                            } else if start_pos < text_tokens.len() {
                                text_tokens[start_pos].clone()
                            } else {
                                continue;
                            };

                            found_spans.push((entity_text, prob, char_start, char_end));
                        }
                    }
                }
            }

            let formatted = Self::format_spans(&mut found_spans, include_confidence, include_spans);
            entities.insert(entity_type.clone(), formatted);
        }

        Ok(JsonValue::Object(entities))
    }

    /// Extract classification from model output.
    fn extract_classification_from_output(
        &self,
        output: &crate::model::ExtractorOutput,
        batch: &PreprocessedBatch,
        sample_idx: usize,
        schema_idx: usize,
        threshold: f32,
        include_confidence: bool,
    ) -> Result<JsonValue> {
        // Get schema embeddings for this sample/schema.
        let sample_schema_embs = output
            .schema_embeddings
            .get(sample_idx)
            .and_then(|sample| sample.get(schema_idx))
            .ok_or_else(|| {
                GlinerError::inference("Missing schema embeddings for classification task")
            })?;
        if sample_schema_embs.is_empty() {
            return Ok(JsonValue::Null);
        }

        // Map each `[L]` label marker to its schema embedding slot. Python uses
        // `embs[1:]` (skipping `[P]`), which is the same set: the only bracket
        // tokens in a classification group are `[P]`, the `[L]` markers, and the
        // two closing parens. Token index 2 is the folded prompt string, which no
        // longer counts because it does not start with `[`.
        let schema_tokens = batch.schema_tokens(sample_idx, schema_idx).ok_or_else(|| {
            GlinerError::inference("Missing schema tokens for classification task")
        })?;
        let mut labels = Vec::new();
        let mut label_emb_indices = Vec::new();
        let mut special_token_counter = 0usize;
        for (i, token) in schema_tokens.iter().enumerate() {
            if token.starts_with('[') && token.ends_with(']') {
                if token == "[L]" && i + 1 < schema_tokens.len() {
                    labels.push(schema_tokens[i + 1].clone());
                    label_emb_indices.push(special_token_counter);
                }
                special_token_counter += 1;
            }
        }

        if labels.is_empty() {
            return Ok(JsonValue::Null);
        }

        let mut label_embs = Vec::with_capacity(label_emb_indices.len());
        for emb_idx in label_emb_indices {
            if let Some(emb) = sample_schema_embs.get(emb_idx) {
                label_embs.push(emb.clone());
            }
        }
        if label_embs.is_empty() {
            return Ok(JsonValue::Null);
        }

        let label_embs_tensor = Tensor::cat(&label_embs, 0).map_err(|e| {
            GlinerError::inference(format!("Failed to stack label embeddings: {e}"))
        })?;
        let logits = self.model.classifier.forward(&label_embs_tensor)?;
        let logits_vec: Vec<f32> = logits
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| {
                GlinerError::inference(format!("Failed to decode classification logits: {e}"))
            })?;
        let is_multi_label = Self::is_multilabel_task(batch, sample_idx, schema_idx);

        // Python: sigmoid for multi_label, softmax otherwise
        // (`GLiNER2._extract_classification_result`). The argmax is unaffected, but
        // the reported confidence is not.
        let probs: Vec<f32> = if is_multi_label {
            logits_vec
                .iter()
                .map(|v| 1.0 / (1.0 + (-v).exp()))
                .collect()
        } else {
            let max = logits_vec.iter().cloned().fold(f32::MIN, f32::max);
            let sum: f32 = logits_vec.iter().map(|v| (v - max).exp()).sum();
            logits_vec
                .iter()
                .map(|v| (v - max).exp() / sum.max(f32::MIN_POSITIVE))
                .collect()
        };

        if is_multi_label {
            let mut selected = Vec::new();
            for (i, label) in labels.iter().enumerate() {
                let prob = probs.get(i).copied().unwrap_or(0.0);
                if prob < threshold {
                    continue;
                }
                if include_confidence {
                    selected.push(serde_json::json!({
                        "label": label,
                        "confidence": prob
                    }));
                } else {
                    selected.push(JsonValue::String(label.clone()));
                }
            }
            return Ok(JsonValue::Array(selected));
        }

        // Single-label: pick the argmax.
        let mut best_idx = 0usize;
        let mut best_prob = f32::MIN;
        for (idx, prob) in probs.iter().enumerate() {
            if *prob > best_prob {
                best_prob = *prob;
                best_idx = idx;
            }
        }
        let best_label = labels
            .get(best_idx)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        if include_confidence {
            Ok(serde_json::json!({
                "label": best_label,
                "confidence": best_prob
            }))
        } else {
            Ok(JsonValue::String(best_label))
        }
    }

    /// Extract relations from model output.
    #[allow(clippy::too_many_arguments)]
    fn extract_relations_from_output(
        &self,
        output: &crate::model::ExtractorOutput,
        batch: &PreprocessedBatch,
        sample_idx: usize,
        schema_idx: usize,
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<JsonValue> {
        // Requires span representations.
        let span_outputs = match output.span_representations.as_ref() {
            Some(v) if sample_idx < v.len() => &v[sample_idx],
            _ => return Ok(JsonValue::Array(Vec::new())),
        };

        // Schema embeddings for this relation schema.
        let schema_tokens_embs = match output
            .schema_embeddings
            .get(sample_idx)
            .and_then(|s| s.get(schema_idx))
        {
            Some(v) if !v.is_empty() => v,
            _ => return Ok(JsonValue::Array(Vec::new())),
        };

        let schema_tokens = match batch.schema_tokens(sample_idx, schema_idx) {
            Some(v) => v,
            None => return Ok(JsonValue::Array(Vec::new())),
        };

        // Relation fields are tokens that follow [R].
        let mut field_names = Vec::new();
        let mut field_emb_indices = Vec::new();
        let mut special_token_counter = 0usize;
        for (i, token) in schema_tokens.iter().enumerate() {
            if token.starts_with('[') && token.ends_with(']') {
                if token == "[R]" && i + 1 < schema_tokens.len() {
                    field_names.push(schema_tokens[i + 1].clone());
                    field_emb_indices.push(special_token_counter);
                }
                special_token_counter += 1;
            }
        }
        if field_names.len() < 2 {
            return Ok(JsonValue::Array(Vec::new()));
        }

        let hidden_size = span_outputs.span_rep.dims().get(2).copied().unwrap_or(0);
        if hidden_size == 0 {
            return Ok(JsonValue::Array(Vec::new()));
        }

        let mut field_emb_data: Vec<f32> = Vec::with_capacity(field_names.len() * hidden_size);
        for &emb_idx in &field_emb_indices {
            if emb_idx < schema_tokens_embs.len() {
                let emb = &schema_tokens_embs[emb_idx];
                if let Ok(data) = emb.flatten_all()
                    && let Ok(vec) = data.to_vec1::<f32>()
                {
                    field_emb_data.extend_from_slice(&vec);
                }
            } else {
                field_emb_data.extend(std::iter::repeat_n(0.0f32, hidden_size));
            }
        }

        let field_embs_tensor = match Tensor::from_vec(
            field_emb_data,
            (field_names.len(), hidden_size),
            &output.device,
        ) {
            Ok(t) => t,
            Err(_) => return Ok(JsonValue::Array(Vec::new())),
        };

        let pred_count = Self::predicted_count(
            output,
            batch,
            sample_idx,
            schema_idx,
            schema_tokens_embs,
            &self.model,
        );
        let Some(count_embed) = self.model.count_embed.as_ref() else {
            return Ok(JsonValue::Array(Vec::new()));
        };
        let struct_proj = match count_embed.forward(&field_embs_tensor, pred_count) {
            Ok(out) => out.embeddings,
            Err(_) => return Ok(JsonValue::Array(Vec::new())),
        };

        let struct_proj_data: Vec<f32> = match struct_proj.flatten_all() {
            Ok(t) => t.to_vec1().unwrap_or_default(),
            Err(_) => return Ok(JsonValue::Array(Vec::new())),
        };
        let span_rep_data: Vec<f32> = match span_outputs.span_rep.flatten_all() {
            Ok(t) => t.to_vec1().unwrap_or_default(),
            Err(_) => return Ok(JsonValue::Array(Vec::new())),
        };
        let mask_data: Vec<u32> = match span_outputs.span_mask.flatten_all() {
            Ok(t) => t.to_vec1().unwrap_or_default(),
            Err(_) => return Ok(JsonValue::Array(Vec::new())),
        };
        let spans_idx_data: Vec<u32> = match span_outputs.spans_idx.flatten_all() {
            Ok(t) => t.to_vec1().unwrap_or_default(),
            Err(_) => return Ok(JsonValue::Array(Vec::new())),
        };

        let mask_dims = span_outputs.span_mask.dims();
        if mask_dims.len() != 2 {
            return Ok(JsonValue::Array(Vec::new()));
        }
        let mask_seq_len = mask_dims[0];
        let mask_max_width = mask_dims[1];
        let max_width = span_outputs
            .span_rep
            .dims()
            .get(1)
            .copied()
            .unwrap_or(mask_max_width);
        let total_spans = mask_seq_len * mask_max_width;

        let text_tokens = batch.sample_text_tokens(sample_idx).unwrap_or(&[]);
        let start_mappings = batch.sample_start_mapping(sample_idx).unwrap_or(&[]);
        let end_mappings = batch.sample_end_mapping(sample_idx).unwrap_or(&[]);
        let original_text = batch.original_text(sample_idx).unwrap_or_default();

        let rel_name = schema_tokens.get(2).cloned().unwrap_or_default();
        let rel_threshold =
            Self::relation_threshold(batch, sample_idx, &rel_name).unwrap_or(threshold);

        let mut instances = Vec::new();
        for inst in 0..pred_count {
            let mut top_fields: Vec<Option<(String, f32, usize, usize)>> =
                vec![None; field_names.len()];

            for (field_idx, top_field) in top_fields.iter_mut().enumerate().take(field_names.len())
            {
                let mut spans = Self::collect_scored_spans(
                    &span_rep_data,
                    &struct_proj_data,
                    &mask_data,
                    &spans_idx_data,
                    mask_seq_len,
                    mask_max_width,
                    max_width,
                    hidden_size,
                    field_names.len(),
                    inst,
                    field_idx,
                    total_spans,
                    original_text,
                    text_tokens,
                    start_mappings,
                    end_mappings,
                    rel_threshold,
                );

                spans.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                if let Some(first) = spans.into_iter().next() {
                    *top_field = Some(first);
                }
            }

            if top_fields.len() >= 2
                && let (Some(head), Some(tail)) = (&top_fields[0], &top_fields[1])
            {
                let rel_obj = if include_spans && include_confidence {
                    serde_json::json!({
                        "head": {"text": head.0, "confidence": head.1, "start": head.2, "end": head.3},
                        "tail": {"text": tail.0, "confidence": tail.1, "start": tail.2, "end": tail.3},
                    })
                } else if include_spans {
                    serde_json::json!({
                        "head": {"text": head.0, "start": head.2, "end": head.3},
                        "tail": {"text": tail.0, "start": tail.2, "end": tail.3},
                    })
                } else if include_confidence {
                    serde_json::json!({
                        "head": {"text": head.0, "confidence": head.1},
                        "tail": {"text": tail.0, "confidence": tail.1},
                    })
                } else {
                    serde_json::json!({
                        "head": head.0,
                        "tail": tail.0,
                    })
                };
                instances.push(rel_obj);
            }
        }

        Ok(JsonValue::Array(instances))
    }

    /// Extract structures from model output.
    #[allow(clippy::too_many_arguments)]
    fn extract_structures_from_output(
        &self,
        output: &crate::model::ExtractorOutput,
        batch: &PreprocessedBatch,
        sample_idx: usize,
        schema_idx: usize,
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<JsonValue> {
        // Requires span representations.
        let span_outputs = match output.span_representations.as_ref() {
            Some(v) if sample_idx < v.len() => &v[sample_idx],
            _ => return Ok(JsonValue::Array(Vec::new())),
        };

        let schema_tokens_embs = match output
            .schema_embeddings
            .get(sample_idx)
            .and_then(|s| s.get(schema_idx))
        {
            Some(v) if !v.is_empty() => v,
            _ => return Ok(JsonValue::Array(Vec::new())),
        };

        let schema_tokens = match batch.schema_tokens(sample_idx, schema_idx) {
            Some(v) => v,
            None => return Ok(JsonValue::Array(Vec::new())),
        };

        // Structure fields are tokens that follow [C].
        let mut field_names = Vec::new();
        let mut field_emb_indices = Vec::new();
        let mut special_token_counter = 0usize;
        for (i, token) in schema_tokens.iter().enumerate() {
            if token.starts_with('[') && token.ends_with(']') {
                if token == "[C]" && i + 1 < schema_tokens.len() {
                    field_names.push(schema_tokens[i + 1].clone());
                    field_emb_indices.push(special_token_counter);
                }
                special_token_counter += 1;
            }
        }
        if field_names.is_empty() {
            return Ok(JsonValue::Array(Vec::new()));
        }

        let hidden_size = span_outputs.span_rep.dims().get(2).copied().unwrap_or(0);
        if hidden_size == 0 {
            return Ok(JsonValue::Array(Vec::new()));
        }

        let mut field_emb_data: Vec<f32> = Vec::with_capacity(field_names.len() * hidden_size);
        for &emb_idx in &field_emb_indices {
            if emb_idx < schema_tokens_embs.len() {
                let emb = &schema_tokens_embs[emb_idx];
                if let Ok(data) = emb.flatten_all()
                    && let Ok(vec) = data.to_vec1::<f32>()
                {
                    field_emb_data.extend_from_slice(&vec);
                }
            } else {
                field_emb_data.extend(std::iter::repeat_n(0.0f32, hidden_size));
            }
        }

        let field_embs_tensor = match Tensor::from_vec(
            field_emb_data,
            (field_names.len(), hidden_size),
            &output.device,
        ) {
            Ok(t) => t,
            Err(_) => return Ok(JsonValue::Array(Vec::new())),
        };

        let pred_count = Self::predicted_count(
            output,
            batch,
            sample_idx,
            schema_idx,
            schema_tokens_embs,
            &self.model,
        );
        let Some(count_embed) = self.model.count_embed.as_ref() else {
            return Ok(JsonValue::Array(Vec::new()));
        };
        let struct_proj = match count_embed.forward(&field_embs_tensor, pred_count) {
            Ok(out) => out.embeddings,
            Err(_) => return Ok(JsonValue::Array(Vec::new())),
        };

        let struct_proj_data: Vec<f32> = match struct_proj.flatten_all() {
            Ok(t) => t.to_vec1().unwrap_or_default(),
            Err(_) => return Ok(JsonValue::Array(Vec::new())),
        };
        let span_rep_data: Vec<f32> = match span_outputs.span_rep.flatten_all() {
            Ok(t) => t.to_vec1().unwrap_or_default(),
            Err(_) => return Ok(JsonValue::Array(Vec::new())),
        };
        let mask_data: Vec<u32> = match span_outputs.span_mask.flatten_all() {
            Ok(t) => t.to_vec1().unwrap_or_default(),
            Err(_) => return Ok(JsonValue::Array(Vec::new())),
        };
        let spans_idx_data: Vec<u32> = match span_outputs.spans_idx.flatten_all() {
            Ok(t) => t.to_vec1().unwrap_or_default(),
            Err(_) => return Ok(JsonValue::Array(Vec::new())),
        };

        let mask_dims = span_outputs.span_mask.dims();
        if mask_dims.len() != 2 {
            return Ok(JsonValue::Array(Vec::new()));
        }
        let mask_seq_len = mask_dims[0];
        let mask_max_width = mask_dims[1];
        let max_width = span_outputs
            .span_rep
            .dims()
            .get(1)
            .copied()
            .unwrap_or(mask_max_width);
        let total_spans = mask_seq_len * mask_max_width;

        let original_text = batch.original_text(sample_idx).unwrap_or_default();
        let text_tokens = batch.sample_text_tokens(sample_idx).unwrap_or(&[]);
        let start_mappings = batch.sample_start_mapping(sample_idx).unwrap_or(&[]);
        let end_mappings = batch.sample_end_mapping(sample_idx).unwrap_or(&[]);

        let struct_name = schema_tokens.get(2).cloned().unwrap_or_default();
        let mut instances = Vec::new();
        for inst in 0..pred_count {
            let mut instance = serde_json::Map::new();

            for (field_idx, field_name) in field_names.iter().enumerate() {
                let (dtype, field_threshold, validators) =
                    Self::structure_field_metadata(batch, sample_idx, &struct_name, field_name);
                let mut spans = Self::collect_scored_spans(
                    &span_rep_data,
                    &struct_proj_data,
                    &mask_data,
                    &spans_idx_data,
                    mask_seq_len,
                    mask_max_width,
                    max_width,
                    hidden_size,
                    field_names.len(),
                    inst,
                    field_idx,
                    total_spans,
                    original_text,
                    text_tokens,
                    start_mappings,
                    end_mappings,
                    field_threshold.unwrap_or(threshold),
                );
                if !validators.is_empty() {
                    spans.retain(|(text, _, _, _)| self.apply_validators(text, &validators));
                }

                if dtype == "str" {
                    spans
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    if let Some((text, conf, start, end)) = spans.into_iter().next() {
                        let v = if include_spans && include_confidence {
                            serde_json::json!({ "text": text, "confidence": conf, "start": start, "end": end })
                        } else if include_spans {
                            serde_json::json!({ "text": text, "start": start, "end": end })
                        } else if include_confidence {
                            serde_json::json!({ "text": text, "confidence": conf })
                        } else {
                            JsonValue::String(text)
                        };
                        instance.insert(field_name.clone(), v);
                    } else {
                        instance.insert(field_name.clone(), JsonValue::Null);
                    }
                } else {
                    let formatted =
                        Self::format_spans(&mut spans, include_confidence, include_spans);
                    instance.insert(field_name.clone(), formatted);
                }
            }

            let has_content = instance.values().any(|v| match v {
                JsonValue::Array(arr) => !arr.is_empty(),
                JsonValue::Null => false,
                _ => true,
            });
            if has_content {
                instances.push(JsonValue::Object(instance));
            }
        }

        Ok(JsonValue::Array(instances))
    }

    fn predicted_count(
        output: &crate::model::ExtractorOutput,
        _batch: &PreprocessedBatch,
        sample_idx: usize,
        schema_idx: usize,
        schema_tokens_embs: &[Tensor],
        model: &Extractor,
    ) -> usize {
        if let Some(all_counts) = &output.count_predictions
            && let Some(sample_counts) = all_counts.get(sample_idx)
            && let Some(count) = sample_counts.get(schema_idx)
        {
            return *count;
        }

        if schema_tokens_embs.is_empty() {
            return 1;
        }

        let p_emb = &schema_tokens_embs[0];
        model
            .count_pred
            .predict_count(p_emb)
            .map(|out| out.count)
            .unwrap_or(1)
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_scored_spans(
        span_rep_data: &[f32],
        struct_proj_data: &[f32],
        mask_data: &[u32],
        spans_idx_data: &[u32],
        mask_seq_len: usize,
        mask_max_width: usize,
        max_width: usize,
        hidden_size: usize,
        num_fields: usize,
        count_idx: usize,
        field_idx: usize,
        total_spans: usize,
        original_text: &str,
        text_tokens: &[String],
        start_mappings: &[usize],
        end_mappings: &[usize],
        threshold: f32,
    ) -> Vec<(String, f32, usize, usize)> {
        let mut spans = Vec::new();
        for i in 0..mask_seq_len {
            for w in 0..mask_max_width {
                let mask_idx = i * mask_max_width + w;
                if mask_idx >= total_spans
                    || mask_idx >= mask_data.len()
                    || mask_data[mask_idx] == 0
                {
                    continue;
                }

                let span_start = i * max_width * hidden_size + w * hidden_size;
                if span_start + hidden_size > span_rep_data.len() {
                    continue;
                }
                let span_rep_vec = &span_rep_data[span_start..span_start + hidden_size];

                let struct_start = (count_idx * num_fields + field_idx) * hidden_size;
                if struct_start + hidden_size > struct_proj_data.len() {
                    continue;
                }
                let struct_vec = &struct_proj_data[struct_start..struct_start + hidden_size];

                let score: f32 = span_rep_vec
                    .iter()
                    .zip(struct_vec.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let prob = 1.0 / (1.0 + (-score).exp());
                if prob < threshold {
                    continue;
                }

                let spans_flat_idx = i * max_width * 2 + w * 2;
                if spans_flat_idx + 1 >= spans_idx_data.len() {
                    continue;
                }
                let start_pos = spans_idx_data[spans_flat_idx] as usize;
                let end_pos = spans_idx_data[spans_flat_idx + 1] as usize;

                let char_start = if start_pos < start_mappings.len() {
                    start_mappings[start_pos]
                } else {
                    0
                };
                let char_end = if end_pos < end_mappings.len() {
                    end_mappings[end_pos]
                } else {
                    char_start
                };

                let entity_text = if !original_text.is_empty() && char_start <= char_end {
                    safe_slice(original_text, char_start, char_end)
                } else if start_pos < text_tokens.len() && end_pos < text_tokens.len() {
                    text_tokens[start_pos..=end_pos].join(" ")
                } else if start_pos < text_tokens.len() {
                    text_tokens[start_pos].clone()
                } else {
                    continue;
                };

                spans.push((entity_text, prob, char_start, char_end));
            }
        }
        spans
    }

    fn format_spans(
        spans: &mut [(String, f32, usize, usize)],
        include_confidence: bool,
        include_spans: bool,
    ) -> JsonValue {
        if spans.is_empty() {
            return JsonValue::Array(Vec::new());
        }

        spans.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut selected: Vec<(String, f32, usize, usize)> = Vec::new();
        for candidate in spans.iter() {
            let overlap = selected
                .iter()
                .any(|s| !(candidate.3 <= s.2 || candidate.2 >= s.3));
            if !overlap {
                selected.push(candidate.clone());
            }
        }

        let formatted = selected
            .into_iter()
            .map(|(text, conf, start, end)| {
                if include_spans && include_confidence {
                    serde_json::json!({ "text": text, "confidence": conf, "start": start, "end": end })
                } else if include_spans {
                    serde_json::json!({ "text": text, "start": start, "end": end })
                } else if include_confidence {
                    serde_json::json!({ "text": text, "confidence": conf })
                } else {
                    JsonValue::String(text)
                }
            })
            .collect();

        JsonValue::Array(formatted)
    }

    fn relation_threshold(
        batch: &PreprocessedBatch,
        sample_idx: usize,
        rel_name: &str,
    ) -> Option<f32> {
        let schema_json = batch.original_schemas.get(sample_idx)?;
        schema_json
            .get("relation_metadata")
            .and_then(|v| v.as_object())
            .and_then(|meta| meta.get(rel_name))
            .and_then(|v| v.get("threshold"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
    }

    fn structure_field_metadata(
        batch: &PreprocessedBatch,
        sample_idx: usize,
        struct_name: &str,
        field_name: &str,
    ) -> (String, Option<f32>, Vec<RegexValidator>) {
        let Some(schema_json) = batch.original_schemas.get(sample_idx) else {
            return ("list".to_string(), None, Vec::new());
        };
        let Some(structs) = schema_json
            .get("json_structures")
            .and_then(|v| v.as_array())
        else {
            return ("list".to_string(), None, Vec::new());
        };

        for item in structs {
            let Some(obj) = item.as_object() else {
                continue;
            };
            let Some(fields) = obj.get(struct_name).and_then(|v| v.as_object()) else {
                continue;
            };
            let Some(field_cfg) = fields.get(field_name) else {
                continue;
            };

            if let Some(cfg_obj) = field_cfg.as_object() {
                let dtype = cfg_obj
                    .get("dtype")
                    .and_then(|v| v.as_str())
                    .unwrap_or("list")
                    .to_string();
                let threshold = cfg_obj
                    .get("threshold")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32);
                let validators = cfg_obj
                    .get("validators")
                    .and_then(|v| serde_json::from_value::<Vec<RegexValidator>>(v.clone()).ok())
                    .unwrap_or_default();
                return (dtype, threshold, validators);
            }
            return ("list".to_string(), None, Vec::new());
        }

        ("list".to_string(), None, Vec::new())
    }

    /// Recover the classification task name from the folded prompt string.
    ///
    /// Port of `GLiNER2._resolve_classification_config`. Token index 2 holds the
    /// whole prompt string (`"<task>"`, or `"<task>: <instruction>"`, plus
    /// appended `[DESCRIPTION]` / `[EXAMPLE]` segments), so an exact match is not
    /// enough. Prefers the longest task that prefixes the string, and requires the
    /// remainder to be empty or start with `':'`/`' '` so that `intent` does not
    /// shadow `intensity`.
    fn resolve_classification_task(schema_json: &JsonValue, prompt_str: &str) -> Option<String> {
        let items = schema_json.get("classifications")?.as_array()?;
        let mut best: Option<&str> = None;
        for item in items {
            let Some(task) = item.get("task").and_then(|t| t.as_str()) else {
                continue;
            };
            if task.is_empty() || !prompt_str.starts_with(task) {
                continue;
            }
            let rest = &prompt_str[task.len()..];
            let ok = rest.is_empty() || rest.starts_with(':') || rest.starts_with(' ');
            if ok && best.is_none_or(|b| task.len() > b.len()) {
                best = Some(task);
            }
        }
        if best.is_some() {
            return best.map(str::to_string);
        }
        // Fallback: any task that prefixes the string, first match wins.
        items.iter().find_map(|item| {
            let task = item.get("task")?.as_str()?;
            prompt_str.starts_with(task).then(|| task.to_string())
        })
    }

    fn is_multilabel_task(batch: &PreprocessedBatch, sample_idx: usize, schema_idx: usize) -> bool {
        let Some(prompt_str) = batch
            .schema_tokens(sample_idx, schema_idx)
            .and_then(|tokens| tokens.get(2))
            .cloned()
        else {
            return false;
        };

        let Some(schema_json) = batch.original_schemas.get(sample_idx) else {
            return false;
        };

        let Some(task) = Self::resolve_classification_task(schema_json, &prompt_str) else {
            return false;
        };

        schema_json
            .get("classifications")
            .and_then(|v| v.as_array())
            .and_then(|items| {
                items.iter().find_map(|item| {
                    let obj = item.as_object()?;
                    let t = obj.get("task")?.as_str()?;
                    (t == task).then(|| {
                        obj.get("multi_label")
                            .and_then(|m| m.as_bool())
                            .unwrap_or(false)
                    })
                })
            })
            .unwrap_or(false)
    }

    // -------------------------------------------------------------------------
    // Helper Methods
    // -------------------------------------------------------------------------

    /// Build an entity schema from entity type names.
    ///
    /// # Arguments
    ///
    /// * `entity_types` - Entity type names.
    ///
    /// # Returns
    ///
    /// A constructed schema.
    fn build_entity_schema(&self, entity_types: &[&str]) -> Result<Schema> {
        let mut builder = self.create_schema();
        for entity_type in entity_types {
            builder = builder.entity(entity_type.to_string()).done();
        }
        builder.build()
    }

    /// Apply regex validators to extracted text.
    ///
    /// # Arguments
    ///
    /// * `text` - The extracted text.
    /// * `validators` - Regex validators to apply.
    ///
    /// # Returns
    ///
    /// `true` if the text passes all validators.
    fn apply_validators(&self, text: &str, validators: &[RegexValidator]) -> bool {
        if validators.is_empty() {
            return true;
        }

        validators.iter().all(|v| v.validate(text).unwrap_or(false))
    }

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    /// Get the model configuration.
    pub fn config(&self) -> &ExtractorConfig {
        &self.model.config
    }

    /// Get the default threshold.
    pub fn default_threshold(&self) -> f32 {
        self.default_threshold
    }

    /// Set the default threshold.
    pub fn set_default_threshold(&mut self, threshold: f32) {
        self.default_threshold = threshold;
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Quantize the model to FP16.
    ///
    /// # Errors
    ///
    /// Returns an error if quantization fails.
    pub fn quantize(&mut self) -> Result<()> {
        self.model.quantize()
    }

    /// Compile the model for faster inference.
    ///
    /// # Errors
    ///
    /// Returns an error if compile setup fails.
    pub fn compile(&mut self) -> Result<()> {
        self.model.compile()
    }

    /// Load model weights from a safetensors file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the safetensors file.
    ///
    /// # Returns
    ///
    /// `Ok(())` if weights were loaded successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if the safetensors file cannot be read or weights
    /// cannot be applied to the model.
    pub fn load_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.model.load_weights(path)
    }
}

fn safe_slice(text: &str, start: usize, end: usize) -> String {
    let end = end.min(text.len());
    let start = start.min(end);
    if let Some(s) = text.get(start..end) {
        return s.to_string();
    }
    let mut chars = text.char_indices();
    let s = chars
        .find(|&(idx, _)| idx >= start)
        .map(|(idx, _)| idx)
        .unwrap_or(text.len());
    let mut chars = text.char_indices();
    let e = chars
        .find(|&(idx, _)| idx >= end)
        .map(|(idx, _)| idx)
        .unwrap_or(text.len());
    text.get(s..e).map(str::to_string).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gliner2_creation() {
        let config = ExtractorConfig::default();
        let engine = GLiNER2::new(&config);
        assert!(engine.is_ok());
        let engine = engine.unwrap();
        assert_eq!(engine.default_threshold(), 0.5);
    }

    #[test]
    fn test_schema_builder() {
        let config = ExtractorConfig::default();
        let engine = GLiNER2::new(&config).unwrap();
        let schema_builder = engine.create_schema();
        assert!(schema_builder.build().is_err()); // Empty schema is invalid

        let schema = engine
            .create_schema()
            .entity("person")
            .done()
            .entity("company")
            .done()
            .build();
        assert!(schema.is_ok());
    }

    #[test]
    fn test_build_entity_schema() {
        let config = ExtractorConfig::default();
        let engine = GLiNER2::new(&config).unwrap();
        let schema = engine.build_entity_schema(&["person", "company"]);
        assert!(schema.is_ok());
        let schema = schema.unwrap();
        assert_eq!(schema.entities.len(), 2);
    }

    #[test]
    fn test_apply_validators() {
        let config = ExtractorConfig::default();
        let engine = GLiNER2::new(&config).unwrap();

        // No validators - should pass
        assert!(engine.apply_validators("test", &[]));

        // With valid regex
        let validator = RegexValidator::new(r"^\d+$").unwrap();
        assert!(engine.apply_validators("123", std::slice::from_ref(&validator)));
        assert!(!engine.apply_validators("abc", &[validator]));
    }

    #[test]
    fn test_set_threshold() {
        let config = ExtractorConfig::default();
        let mut engine = GLiNER2::new(&config).unwrap();
        assert_eq!(engine.default_threshold(), 0.5);
        engine.set_default_threshold(0.7);
        assert_eq!(engine.default_threshold(), 0.7);
    }

    #[test]
    fn test_safe_slice_provenance() {
        let text = "Server handled 500 requests/second under load.";
        let start = 15;
        let end = 34; // "500 requests/second"
        let slice = safe_slice(text, start, end);
        assert_eq!(slice, "500 requests/second");
        assert_eq!(text.get(start..end), Some(slice.as_str()));
        assert_eq!(&text.as_bytes()[start..end], slice.as_bytes());

        // Unicode / multi-byte characters
        let unicode_text = "Apple 🍎 CEO Tim Cook announced Zürich results in 北京.";
        let start_zurich = unicode_text.find("Zürich").unwrap();
        let end_zurich = start_zurich + "Zürich".len();
        let zurich_slice = safe_slice(unicode_text, start_zurich, end_zurich);
        assert_eq!(zurich_slice, "Zürich");
        assert_eq!(
            unicode_text.get(start_zurich..end_zurich),
            Some(zurich_slice.as_str())
        );

        let start_bj = unicode_text.find("北京").unwrap();
        let end_bj = start_bj + "北京".len();
        let bj_slice = safe_slice(unicode_text, start_bj, end_bj);
        assert_eq!(bj_slice, "北京");
        assert_eq!(unicode_text.get(start_bj..end_bj), Some(bj_slice.as_str()));
    }

    #[test]
    fn test_format_spans_flags() {
        let spans = vec![
            ("Apple".to_string(), 0.95, 0, 5),
            ("Tim Cook".to_string(), 0.98, 10, 18),
            ("Apple CEO".to_string(), 0.80, 0, 9), // Overlaps with Apple (lower score)
        ];

        // 1. Neither spans nor confidence
        let res_bare = GLiNER2::format_spans(&mut spans.clone(), false, false);
        assert_eq!(res_bare, serde_json::json!(["Tim Cook", "Apple"]));

        // 2. Only spans
        let res_spans = GLiNER2::format_spans(&mut spans.clone(), false, true);
        assert_eq!(
            res_spans,
            serde_json::json!([
                {"text": "Tim Cook", "start": 10, "end": 18},
                {"text": "Apple", "start": 0, "end": 5}
            ])
        );

        // 3. Only confidence
        let res_conf = GLiNER2::format_spans(&mut spans.clone(), true, false);
        let conf_arr = res_conf.as_array().unwrap();
        assert_eq!(conf_arr.len(), 2);
        assert_eq!(conf_arr[0]["text"], "Tim Cook");
        assert!((conf_arr[0]["confidence"].as_f64().unwrap() - 0.98).abs() < 1e-4);
        assert_eq!(conf_arr[1]["text"], "Apple");
        assert!((conf_arr[1]["confidence"].as_f64().unwrap() - 0.95).abs() < 1e-4);

        // 4. Both spans and confidence
        let res_both = GLiNER2::format_spans(&mut spans.clone(), true, true);
        let both_arr = res_both.as_array().unwrap();
        assert_eq!(both_arr.len(), 2);
        assert_eq!(both_arr[0]["text"], "Tim Cook");
        assert_eq!(both_arr[0]["start"], 10);
        assert_eq!(both_arr[0]["end"], 18);
        assert!((both_arr[0]["confidence"].as_f64().unwrap() - 0.98).abs() < 1e-4);
        assert_eq!(both_arr[1]["text"], "Apple");
        assert_eq!(both_arr[1]["start"], 0);
        assert_eq!(both_arr[1]["end"], 5);
        assert!((both_arr[1]["confidence"].as_f64().unwrap() - 0.95).abs() < 1e-4);
    }
}
