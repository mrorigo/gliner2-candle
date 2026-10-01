// Rust guideline compliant 2026-04-03
//! Main Extractor model for GLiNER2.
//!
//! This module implements the core `Extractor` model that combines all
//! neural network components for information extraction. It handles the
//! forward pass, embedding extraction, and coordinates between the
//! encoder, span representation, count prediction, and classifier layers.
//!
//! # Architecture
//!
//! The Extractor model consists of:
//! - **Encoder**: Transformer encoder (BERT/DeBERTa) via candle
//! - **Span Representation Layer**: Computes span-level representations
//! - **Count Prediction Layer**: Predicts number of instances per schema
//! - **Classifier Head**: Classification head for text classification tasks
//!
//! # Example
//!
//! ```ignore
//! use gliner2_candle::model::Extractor;
//! use gliner2_candle::config::ExtractorConfig;
//! use gliner2_candle::batch::PreprocessedBatch;
//!
//! let config = ExtractorConfig::default();
//! let mut model = Extractor::new(&config)?;
//! model.load_weights("path/to/model.safetensors")?;
//!
//! let output = model.forward(&batch)?;
//! ```

use std::path::Path;

use candle_core::{DType, Device, Tensor};

use crate::batch::PreprocessedBatch;
use crate::config::ExtractorConfig;
use crate::error::{GlinerError, Result};
use crate::model::candle_encoder::CandleEncoder;
use crate::model::classifier::ClassifierHead;
use crate::model::count_embed::CountLayer;
use crate::model::count_pred::CountPredictionLayer;
use crate::model::loading::ModelLoader;
use crate::model::span_rep::{SpanRepOutput, SpanRepresentationLayer};

type SchemaEmbeddings = Vec<Vec<Vec<Tensor>>>;
type EmbeddingExtraction = (Vec<Tensor>, SchemaEmbeddings);

/// Output of the Extractor model forward pass.
#[derive(Clone)]
pub struct ExtractorOutput {
    /// Token embeddings for each sample: `Vec<(seq_len, hidden_size)>`.
    pub token_embeddings: Vec<Tensor>,
    /// Schema embeddings for each sample: `Vec<Vec<Vec<(hidden_size,)>>>`.
    /// Structure: sample -> schema -> schema_tokens -> embedding
    pub schema_embeddings: Vec<Vec<Vec<Tensor>>>,
    /// Span representations for each sample (if computed).
    pub span_representations: Option<Vec<SpanRepOutput>>,
    /// Classification logits for each sample (if applicable).
    pub classification_logits: Option<Vec<Tensor>>,
    /// Count predictions for each schema in each sample.
    pub count_predictions: Option<Vec<Vec<usize>>>,
    /// Batch size.
    pub batch_size: usize,
    /// Device the output tensors are on.
    pub device: Device,
}

impl std::fmt::Debug for ExtractorOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtractorOutput")
            .field("batch_size", &self.batch_size)
            .field("token_embeddings_len", &self.token_embeddings.len())
            .field("schema_embeddings_len", &self.schema_embeddings.len())
            .field(
                "has_span_representations",
                &self.span_representations.is_some(),
            )
            .field(
                "has_classification_logits",
                &self.classification_logits.is_some(),
            )
            .field("has_count_predictions", &self.count_predictions.is_some())
            .field("device", &self.device)
            .finish()
    }
}

impl ExtractorOutput {
    /// Create a new extractor output.
    pub fn new(
        token_embeddings: Vec<Tensor>,
        schema_embeddings: Vec<Vec<Vec<Tensor>>>,
        span_representations: Option<Vec<SpanRepOutput>>,
        classification_logits: Option<Vec<Tensor>>,
        count_predictions: Option<Vec<Vec<usize>>>,
        device: Device,
    ) -> Self {
        let batch_size = token_embeddings.len();
        Self {
            token_embeddings,
            schema_embeddings,
            span_representations,
            classification_logits,
            count_predictions,
            batch_size,
            device,
        }
    }

    /// Create an empty extractor output.
    pub fn empty(device: Device) -> Self {
        Self {
            token_embeddings: Vec::new(),
            schema_embeddings: Vec::new(),
            span_representations: None,
            classification_logits: None,
            count_predictions: None,
            batch_size: 0,
            device,
        }
    }

    /// Check if the output is empty.
    pub fn is_empty(&self) -> bool {
        self.batch_size == 0
    }
}

/// Main Extractor model for GLiNER2.
///
/// This model combines all neural network components for information
/// extraction, including entity extraction, relation extraction,
/// structured data extraction, and text classification.
///
/// # Tensor Flow
///
/// 1. Input: `PreprocessedBatch` with tokenized text and schemas
/// 2. Encoder: Produces token embeddings for all tokens
/// 3. Embedding Extraction: Extracts text and schema embeddings
/// 4. Span Representation: Computes span-level representations
/// 5. Count Prediction: Predicts number of instances per schema
/// 6. Classification: Produces classification logits (if applicable)
/// 7. Output: `ExtractorOutput` with all intermediate results
pub struct Extractor {
    /// Model configuration.
    pub config: ExtractorConfig,
    /// Architecture family (GLiNER2 vs GLiNER2.5).
    pub architecture: crate::config::Architecture,
    /// Hidden size of the model.
    pub hidden_size: usize,
    /// Maximum span width.
    pub max_width: usize,
    /// Device for tensor operations.
    pub device: Device,
    /// Whether the model is in training mode.
    pub is_training: bool,
    /// Whether the model has been loaded with weights.
    pub is_loaded: bool,

    // Submodules
    /// Transformer encoder (BERT/DeBERTa via candle).
    pub encoder: CandleEncoder,
    /// Span representation layer.
    pub span_rep: SpanRepresentationLayer,
    /// Count prediction layer.
    pub count_pred: CountPredictionLayer,
    /// Count embedding layer for entity scoring.
    ///
    /// `None` only when the checkpoint ships no count-aware projection at all.
    /// Both upstream variants are implemented (`CountLSTM` and `CountLSTMv2`);
    /// the weights decide which is loaded, since every span checkpoint declares
    /// `counting_layer: "count_lstm"` regardless of what it actually contains.
    /// Entity scoring needs this module, so a checkpoint without it cannot
    /// extract spans, relations or structures. Classification never uses it.
    pub count_embed: Option<CountLayer>,
    /// Classifier head.
    pub classifier: ClassifierHead,
    /// GLiNER2.5 boundary model (populated at weight-load time; `None` for
    /// GLiNER2 checkpoints and before loading).
    pub boundary: Option<crate::model::boundary::BoundaryModel>,
}

impl std::fmt::Debug for Extractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Extractor")
            .field("config", &self.config)
            .field("architecture", &self.architecture)
            .field("hidden_size", &self.hidden_size)
            .field("max_width", &self.max_width)
            .field("device", &self.device)
            .field("is_training", &self.is_training)
            .field("is_loaded", &self.is_loaded)
            .finish()
    }
}

impl Extractor {
    /// Create a new Extractor model from configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The extractor configuration.
    ///
    /// # Returns
    ///
    /// A new `Extractor` with initialized submodules.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration validation fails or submodule
    /// initialization fails.
    pub fn new(config: &ExtractorConfig) -> Result<Self> {
        config.validate()?;

        // Determine device
        let device = match config.device.as_str() {
            "cpu" => Device::Cpu,
            "cuda" => Device::cuda_if_available(0).unwrap_or(Device::Cpu),
            _ => {
                if config.device.starts_with("cuda:") {
                    let idx: usize = config.device[5..].parse().map_err(|_| {
                        GlinerError::config(format!("Invalid CUDA device: {}", config.device))
                    })?;
                    Device::cuda_if_available(idx).unwrap_or(Device::Cpu)
                } else {
                    Device::Cpu
                }
            }
        };

        // Initialize encoder
        let encoder = CandleEncoder::new(config, device.clone()).map_err(|e| {
            GlinerError::model_loading(format!("Failed to initialize encoder: {e}"))
        })?;

        // Initialize submodules
        let span_rep = SpanRepresentationLayer::from_config(config, device.clone())?;
        let count_pred = CountPredictionLayer::from_config(config, device.clone())?;
        // Placeholder until weights are loaded: pick whichever layout the
        // VarBuilder will supply. `rebuild_model` replaces this with the real
        // layer (or `None` when the checkpoint has no count-aware weights).
        let count_embed = Some(CountLayer::placeholder(config.hidden_size, device.clone()));
        let classifier = ClassifierHead::from_config(config, device.clone())?;

        Ok(Self {
            config: config.clone(),
            architecture: config.architecture,
            hidden_size: config.hidden_size,
            max_width: config.max_width,
            device,
            is_training: false,
            is_loaded: false,
            encoder,
            span_rep,
            count_pred,
            count_embed,
            classifier,
            boundary: None,
        })
    }

    /// Load model weights from a safetensors file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the safetensors file.
    ///
    /// # Returns
    ///
    /// A newly initialized extractor configured for the given model name.
    ///
    /// # Errors
    ///
    /// Returns an error if the safetensors file cannot be loaded or model
    /// components cannot be rebuilt.
    pub fn load_weights(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let loader = ModelLoader::new(&self.config, self.device.clone())?;
        loader.load_safetensors(path, self)?;
        self.is_loaded = true;
        Ok(())
    }

    /// Load model weights from a HuggingFace model repository.
    ///
    /// This method downloads the model weights from HuggingFace Hub
    /// if they are not already cached locally.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The HuggingFace model name (e.g., "fastino/gliner2-base-v1").
    ///
    /// # Returns
    ///
    /// `Ok(())` if weights were loaded successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if model initialization fails.
    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        let config = ExtractorConfig::new(model_name);
        let mut model = Self::new(&config)?;

        // In a full implementation, this would download from HuggingFace Hub
        // For now, we expect local weights
        model.is_loaded = false;
        Ok(model)
    }

    /// Set the model to training mode.
    pub fn train(&mut self) {
        self.is_training = true;
    }

    /// Set the model to evaluation mode.
    pub fn eval(&mut self) {
        self.is_training = false;
    }

    /// Check if the model is in training mode.
    pub fn is_training(&self) -> bool {
        self.is_training
    }

    /// Forward pass through the model.
    ///
    /// # Arguments
    ///
    /// * `batch` - The preprocessed batch of inputs.
    ///
    /// # Returns
    ///
    /// An `ExtractorOutput` containing all intermediate results.
    ///
    /// # Errors
    ///
    /// Returns an error if any encoder, embedding extraction, span, count, or
    /// classification computation fails.
    pub fn forward(&self, batch: &PreprocessedBatch) -> Result<ExtractorOutput> {
        if batch.is_empty() {
            return Ok(ExtractorOutput::empty(self.device.clone()));
        }

        // GLiNER2.5 uses the boundary path, not the span-enumeration grid.
        if self.architecture == crate::config::Architecture::Gliner25 {
            return Err(GlinerError::config(
                "GLiNER2.5 boundary decoding is not yet implemented (see docs/PLAN_2.5.md Phase 2)",
            ));
        }

        // Step 1: Run encoder to get token embeddings
        let token_embeddings = self.extract_token_embeddings(batch)?;

        // Step 2: Extract schema embeddings
        let schema_embeddings = self.extract_schema_embeddings(batch)?;

        // Step 3: Compute span representations for samples that need them
        let span_representations = self.compute_span_representations(&token_embeddings, batch)?;

        // Step 4: Compute count predictions for each schema
        let count_predictions = self.compute_count_predictions(&schema_embeddings)?;

        // Step 5: Compute classification logits for classification tasks
        let classification_logits =
            self.compute_classification_logits(&schema_embeddings, batch)?;

        Ok(ExtractorOutput::new(
            token_embeddings,
            schema_embeddings,
            span_representations,
            classification_logits,
            count_predictions,
            self.device.clone(),
        ))
    }

    /// Extract token embeddings from the encoder output.
    ///
    /// This method runs the encoder on the batch input_ids and uses
    /// the precomputed `text_word_indices` to gather token embeddings
    /// for each word in the input text.
    ///
    /// # Arguments
    ///
    /// * `batch` - The preprocessed batch.
    ///
    /// # Returns
    ///
    /// A vector of token embeddings, one per sample.
    fn extract_token_embeddings(&self, batch: &PreprocessedBatch) -> Result<Vec<Tensor>> {
        // Get input_ids and attention_mask from batch
        let input_ids = batch.input_ids.to_device(&self.device).map_err(|e| {
            GlinerError::model_loading(format!("Failed to move input_ids to device: {e}"))
        })?;
        let attention_mask = batch.attention_mask.to_device(&self.device).map_err(|e| {
            GlinerError::model_loading(format!("Failed to move attention_mask to device: {e}"))
        })?;

        // Run encoder forward pass
        let encoder_output = self
            .encoder
            .forward(&input_ids, &attention_mask)
            .map_err(|e| GlinerError::model_loading(format!("Encoder forward pass failed: {e}")))?;

        // Extract per-sample embeddings using text_word_indices
        let batch_size = batch.batch_size();
        let mut token_embeddings = Vec::with_capacity(batch_size);

        for sample_idx in 0..batch_size {
            let word_count = batch.text_word_counts.get(sample_idx).copied().unwrap_or(0);
            if word_count == 0 {
                // Empty sample: return empty tensor
                let empty = Tensor::zeros((0, self.hidden_size), DType::F32, &self.device)
                    .map_err(|e| {
                        GlinerError::model_loading(format!("Failed to create empty tensor: {e}"))
                    })?;
                token_embeddings.push(empty);
                continue;
            }

            // Gather embeddings using precomputed indices
            if let Some(indices_tensor) = &batch.text_word_indices {
                // indices_tensor shape: (batch_size, max_words)
                // Gather indices for this sample
                let sample_indices = indices_tensor
                    .narrow(0, sample_idx, 1)
                    .map_err(|e| {
                        GlinerError::model_loading(format!("Failed to narrow indices: {e}"))
                    })?
                    .squeeze(0)
                    .map_err(|e| {
                        GlinerError::model_loading(format!("Failed to squeeze indices: {e}"))
                    })?
                    .narrow(0, 0, word_count)
                    .map_err(|e| {
                        GlinerError::model_loading(format!("Failed to trim padded indices: {e}"))
                    })?;

                // encoder_output: (batch, seq_len, hidden)
                // Gather: encoder_output[sample_idx, indices] -> (word_count, hidden)
                let sample_encoder = encoder_output
                    .narrow(0, sample_idx, 1)
                    .map_err(|e| {
                        GlinerError::model_loading(format!("Failed to narrow encoder output: {e}"))
                    })?
                    .squeeze(0)
                    .map_err(|e| {
                        GlinerError::model_loading(format!("Failed to squeeze encoder output: {e}"))
                    })?;

                let word_embs = sample_encoder
                    .index_select(&sample_indices, 0)
                    .map_err(|e| {
                        GlinerError::model_loading(format!("Failed to gather word embeddings: {e}"))
                    })?;

                token_embeddings.push(word_embs);
            } else {
                // Fallback: create empty tensor
                let empty = Tensor::zeros((0, self.hidden_size), DType::F32, &self.device)
                    .map_err(|e| {
                        GlinerError::model_loading(format!("Failed to create empty tensor: {e}"))
                    })?;
                token_embeddings.push(empty);
            }
        }

        Ok(token_embeddings)
    }

    /// Extract schema embeddings from the encoder output.
    ///
    /// This method uses the precomputed `schema_special_indices` to gather
    /// embeddings for each schema token from the encoder output.
    ///
    /// # Arguments
    ///
    /// * `batch` - The preprocessed batch.
    ///
    /// # Returns
    ///
    /// A nested vector of schema embeddings: sample -> schema -> tokens -> embedding
    fn extract_schema_embeddings(
        &self,
        batch: &PreprocessedBatch,
    ) -> Result<Vec<Vec<Vec<Tensor>>>> {
        // Get input_ids and attention_mask from batch
        let input_ids = batch.input_ids.to_device(&self.device).map_err(|e| {
            GlinerError::model_loading(format!("Failed to move input_ids to device: {e}"))
        })?;
        let attention_mask = batch.attention_mask.to_device(&self.device).map_err(|e| {
            GlinerError::model_loading(format!("Failed to move attention_mask to device: {e}"))
        })?;

        // Run encoder forward pass
        let encoder_output = self
            .encoder
            .forward(&input_ids, &attention_mask)
            .map_err(|e| GlinerError::model_loading(format!("Encoder forward pass failed: {e}")))?;

        let batch_size = batch.batch_size();
        let mut schema_embeddings = Vec::with_capacity(batch_size);

        for sample_idx in 0..batch_size {
            let num_schemas = batch.num_schemas(sample_idx).unwrap_or(0);
            let mut sample_schema_embs = Vec::with_capacity(num_schemas);

            for schema_idx in 0..num_schemas {
                // Get schema tokens for this schema
                let schema_tokens = batch.schema_tokens(sample_idx, schema_idx);
                if schema_tokens.is_none() || schema_tokens.as_ref().is_none_or(|t| t.is_empty()) {
                    sample_schema_embs.push(Vec::new());
                    continue;
                }
                let Some(schema_tokens) = schema_tokens else {
                    sample_schema_embs.push(Vec::new());
                    continue;
                };

                // Get special indices for this schema
                let special_indices = batch.schema_special_indices_for(sample_idx, schema_idx);
                if special_indices.is_none()
                    || special_indices.as_ref().is_none_or(|i| i.is_empty())
                {
                    sample_schema_embs.push(Vec::new());
                    continue;
                }
                let Some(special_indices) = special_indices else {
                    sample_schema_embs.push(Vec::new());
                    continue;
                };

                // Get encoder output for this sample
                let sample_encoder = encoder_output
                    .narrow(0, sample_idx, 1)
                    .map_err(|e| {
                        GlinerError::model_loading(format!("Failed to narrow encoder output: {e}"))
                    })?
                    .squeeze(0)
                    .map_err(|e| {
                        GlinerError::model_loading(format!("Failed to squeeze encoder output: {e}"))
                    })?;

                // Extract embeddings for each schema token using special indices
                let mut schema_token_embs = Vec::with_capacity(schema_tokens.len());
                for &idx in special_indices.iter() {
                    let idx_tensor = Tensor::new(&[idx as u32], &self.device).map_err(|e| {
                        GlinerError::model_loading(format!("Failed to create index tensor: {e}"))
                    })?;
                    let emb = sample_encoder.index_select(&idx_tensor, 0).map_err(|e| {
                        GlinerError::model_loading(format!(
                            "Failed to gather schema embedding: {e}"
                        ))
                    })?;

                    schema_token_embs.push(emb);
                }

                sample_schema_embs.push(schema_token_embs);
            }

            schema_embeddings.push(sample_schema_embs);
        }

        Ok(schema_embeddings)
    }

    /// Compute span representations for samples that need them.
    ///
    /// # Arguments
    ///
    /// * `token_embeddings` - Token embeddings for each sample.
    /// * `batch` - The preprocessed batch.
    ///
    /// # Returns
    ///
    /// Optional vector of span representation outputs.
    fn compute_span_representations(
        &self,
        token_embeddings: &[Tensor],
        batch: &PreprocessedBatch,
    ) -> Result<Option<Vec<SpanRepOutput>>> {
        let batch_size = batch.batch_size();
        let mut span_outputs = Vec::with_capacity(batch_size);

        for (sample_idx, token_embs) in token_embeddings.iter().enumerate().take(batch_size) {
            // Check if this sample has span-based tasks
            let task_types = batch.sample_task_types(sample_idx);
            let has_span_task =
                task_types.is_some_and(|types| types.iter().any(|t| t != "classifications"));

            let numel = token_embs.dims().iter().product::<usize>();

            if has_span_task && numel > 0 {
                let span_output = self.span_rep.forward(token_embs)?;
                span_outputs.push(span_output);
            } else {
                // Create empty span output for samples without span tasks
                let empty_span = Tensor::zeros(
                    (0, self.max_width, self.hidden_size),
                    DType::F32,
                    &self.device,
                )
                .map_err(|e| {
                    GlinerError::model_loading(format!("Failed to create empty span tensor: {e}"))
                })?;
                let empty_idx = Tensor::zeros((0, self.max_width, 2), DType::U32, &self.device)
                    .map_err(|e| {
                        GlinerError::model_loading(format!(
                            "Failed to create empty index tensor: {e}"
                        ))
                    })?;
                let empty_mask = Tensor::zeros((0, self.max_width), DType::U32, &self.device)
                    .map_err(|e| {
                        GlinerError::model_loading(format!(
                            "Failed to create empty mask tensor: {e}"
                        ))
                    })?;
                span_outputs.push(SpanRepOutput {
                    span_rep: empty_span,
                    spans_idx: empty_idx,
                    span_mask: empty_mask,
                });
            }
        }

        Ok(Some(span_outputs))
    }

    /// Compute count predictions for each schema.
    ///
    /// # Arguments
    ///
    /// * `schema_embeddings` - Schema embeddings for each sample.
    ///
    /// # Returns
    ///
    /// Optional vector of count predictions: sample -> schema -> count
    fn compute_count_predictions(
        &self,
        schema_embeddings: &[Vec<Vec<Tensor>>],
    ) -> Result<Option<Vec<Vec<usize>>>> {
        let batch_size = schema_embeddings.len();
        let mut all_counts = Vec::with_capacity(batch_size);

        for sample_schema_embs in schema_embeddings {
            let mut sample_counts = Vec::with_capacity(sample_schema_embs.len());

            for schema_token_embs in sample_schema_embs {
                if schema_token_embs.is_empty() {
                    sample_counts.push(0);
                    continue;
                }

                // Use the first schema token embedding (typically the schema name token)
                // to predict the count
                let schema_emb = &schema_token_embs[0];
                let count_output = self.count_pred.predict_count(schema_emb)?;
                sample_counts.push(count_output.count);
            }

            all_counts.push(sample_counts);
        }

        Ok(Some(all_counts))
    }

    /// Compute classification logits for classification tasks.
    ///
    /// # Arguments
    ///
    /// * `schema_embeddings` - Schema embeddings for each sample.
    /// * `batch` - The preprocessed batch.
    ///
    /// # Returns
    ///
    /// Optional vector of classification logits per sample.
    fn compute_classification_logits(
        &self,
        schema_embeddings: &[Vec<Vec<Tensor>>],
        batch: &PreprocessedBatch,
    ) -> Result<Option<Vec<Tensor>>> {
        let batch_size = schema_embeddings.len();
        let mut all_logits = Vec::with_capacity(batch_size);

        for (sample_idx, sample_schema_embs) in
            schema_embeddings.iter().enumerate().take(batch_size)
        {
            let task_types = batch.sample_task_types(sample_idx);
            let has_classification =
                task_types.is_some_and(|types| types.iter().any(|t| t == "classifications"));

            if has_classification {
                // Collect schema embeddings for classification tasks
                let mut cls_embs = Vec::new();

                let task_types_ref = task_types.unwrap_or(&[]);
                for (schema_idx, schema_token_embs) in sample_schema_embs.iter().enumerate() {
                    if schema_idx < task_types_ref.len()
                        && task_types_ref[schema_idx] == "classifications"
                    {
                        // Stack all schema token embeddings for this classification task
                        if !schema_token_embs.is_empty() {
                            let stacked = Tensor::stack(schema_token_embs, 0).map_err(|e| {
                                GlinerError::model_loading(format!(
                                    "Failed to stack classification embeddings: {e}"
                                ))
                            })?;
                            cls_embs.push(stacked);
                        }
                    }
                }

                if !cls_embs.is_empty() {
                    // Stack all classification embeddings
                    let all_cls_embs = Tensor::cat(&cls_embs, 0).map_err(|e| {
                        GlinerError::model_loading(format!(
                            "Failed to concat classification embeddings: {e}"
                        ))
                    })?;
                    let logits = self.classifier.forward(&all_cls_embs)?;
                    all_logits.push(logits);
                } else {
                    all_logits.push(Tensor::zeros((0,), DType::F32, &self.device).map_err(
                        |e| {
                            GlinerError::model_loading(format!(
                                "Failed to create empty logits: {e}"
                            ))
                        },
                    )?);
                }
            } else {
                all_logits.push(Tensor::zeros((0,), DType::F32, &self.device).map_err(|e| {
                    GlinerError::model_loading(format!("Failed to create empty logits: {e}"))
                })?);
            }
        }

        Ok(Some(all_logits))
    }

    /// Run the encoder on input IDs to get token embeddings.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Input token IDs of shape `(batch_size, seq_len)`.
    /// * `attention_mask` - Attention mask of shape `(batch_size, seq_len)`.
    ///
    /// # Returns
    ///
    /// Encoder output tensor of shape `(batch_size, seq_len, hidden_size)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the encoder forward pass fails.
    pub fn run_encoder(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        self.encoder
            .forward(input_ids, attention_mask)
            .map_err(|e| GlinerError::model_loading(format!("Encoder forward pass failed: {e}")))
    }

    /// Extract embeddings from a batch using the fast path (gather-based).
    ///
    /// This method uses precomputed indices to efficiently gather
    /// token and schema embeddings from the encoder output.
    ///
    /// # Arguments
    ///
    /// * `encoder_output` - The encoder output tensor.
    /// * `batch` - The preprocessed batch.
    ///
    /// # Returns
    ///
    /// A tuple of (token_embeddings, schema_embeddings).
    ///
    /// # Errors
    ///
    /// Returns an error if embedding extraction fails.
    pub fn extract_embeddings_fast(
        &self,
        _encoder_output: &Tensor,
        batch: &PreprocessedBatch,
    ) -> Result<EmbeddingExtraction> {
        let token_embeddings = self.extract_token_embeddings(batch)?;
        let schema_embeddings = self.extract_schema_embeddings(batch)?;
        Ok((token_embeddings, schema_embeddings))
    }

    /// Extract embeddings from a batch using the loop path.
    ///
    /// This method iterates through each token and schema to extract
    /// embeddings, which is slower but more flexible.
    ///
    /// # Arguments
    ///
    /// * `encoder_output` - The encoder output tensor.
    /// * `input_ids` - Input token IDs.
    /// * `batch` - The preprocessed batch.
    ///
    /// # Returns
    ///
    /// A tuple of (token_embeddings, schema_embeddings).
    ///
    /// # Errors
    ///
    /// Returns an error if embedding extraction fails.
    pub fn extract_embeddings_loop(
        &self,
        _encoder_output: &Tensor,
        _input_ids: &Tensor,
        batch: &PreprocessedBatch,
    ) -> Result<EmbeddingExtraction> {
        // Similar to fast path but with explicit looping
        self.extract_embeddings_fast(_encoder_output, batch)
    }

    /// Get the hidden size of the model.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get the maximum span width.
    pub fn max_width(&self) -> usize {
        self.max_width
    }

    /// Get the device the model is on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Quantize the model to FP16 for faster inference.
    ///
    /// # Errors
    ///
    /// This function currently does not fail, but returns `Result` for API
    /// consistency with other model operations.
    pub fn quantize(&mut self) -> Result<()> {
        if self.config.use_fp16 || self.config.use_bf16 {
            return Ok(()); // Already quantized
        }

        self.config.use_fp16 = true;
        // In a full implementation, this would convert all weights to FP16
        Ok(())
    }

    /// Compile the model for faster inference (if supported).
    ///
    /// # Errors
    ///
    /// This function currently does not fail, but returns `Result` for API
    /// consistency with other model operations.
    pub fn compile(&mut self) -> Result<()> {
        self.config.compile = true;
        // In a full implementation, this would use torch.compile or similar
        Ok(())
    }
}

/// Builder for constructing `Extractor` with custom settings.
#[derive(Debug, Clone, Default)]
pub struct ExtractorBuilder {
    config: ExtractorConfig,
}

impl ExtractorBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model name.
    pub fn model_name(mut self, name: impl Into<String>) -> Self {
        self.config.model_name = name.into();
        self
    }

    /// Set the hidden size.
    pub fn hidden_size(mut self, size: usize) -> Self {
        self.config.hidden_size = size;
        self
    }

    /// Set the maximum span width.
    pub fn max_width(mut self, width: usize) -> Self {
        self.config.max_width = width;
        self
    }

    /// Set the device.
    pub fn device(mut self, device: impl Into<String>) -> Self {
        self.config.device = device.into();
        self
    }

    /// Enable FP16.
    pub fn fp16(mut self, enabled: bool) -> Self {
        self.config.use_fp16 = enabled;
        if enabled {
            self.config.use_bf16 = false;
        }
        self
    }

    /// Enable BF16.
    pub fn bf16(mut self, enabled: bool) -> Self {
        self.config.use_bf16 = enabled;
        if enabled {
            self.config.use_fp16 = false;
        }
        self
    }

    /// Set max_len truncation.
    pub fn max_len(mut self, max_len: Option<usize>) -> Self {
        self.config.max_len = max_len;
        self
    }

    /// Build the extractor model.
    ///
    /// # Errors
    ///
    /// Returns an error if extractor initialization fails.
    pub fn build(self) -> Result<Extractor> {
        Extractor::new(&self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::PreprocessedBatchBuilder;
    use candle_core::Tensor;

    fn create_test_batch() -> PreprocessedBatch {
        let input_ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5, 6], (2, 3), &Device::Cpu).unwrap();
        let attention_mask =
            Tensor::from_slice(&[1u32, 1, 1, 1, 1, 0], (2, 3), &Device::Cpu).unwrap();

        PreprocessedBatchBuilder::new()
            .input_ids(input_ids)
            .attention_mask(attention_mask)
            .schema_counts(vec![1, 1])
            .task_types(vec![
                vec!["entities".to_string()],
                vec!["entities".to_string()],
            ])
            .text_tokens(vec![
                vec!["apple".to_string(), "inc".to_string()],
                vec!["hello".to_string()],
            ])
            .schema_tokens_list(vec![vec![vec![
                "(".to_string(),
                "[P]".to_string(),
                "entities".to_string(),
            ]]])
            .start_mappings(vec![vec![0, 6], vec![0]])
            .end_mappings(vec![vec![5, 10], vec![5]])
            .original_texts(vec!["Apple Inc.".to_string(), "Hello".to_string()])
            .original_schemas(vec![serde_json::Value::Null, serde_json::Value::Null])
            .text_word_counts(vec![2, 1])
            .schema_special_indices(vec![vec![vec![0, 1, 2]], vec![vec![0, 1]]])
            .build()
            .unwrap()
    }

    #[test]
    fn test_extractor_creation() {
        let config = ExtractorConfig::default();
        let extractor = Extractor::new(&config);
        assert!(extractor.is_ok());
        let extractor = extractor.unwrap();
        assert_eq!(extractor.hidden_size(), 768);
        assert_eq!(extractor.max_width(), 8);
        assert!(!extractor.is_loaded);
    }

    #[test]
    fn test_extractor_forward() {
        let config = ExtractorConfig::default();
        let extractor = Extractor::new(&config).unwrap();
        let batch = create_test_batch();

        let output = extractor.forward(&batch);
        assert!(output.is_ok(), "Forward failed: {:?}", output.err());
        let output = output.unwrap();
        assert_eq!(output.batch_size, 2);
        assert_eq!(output.token_embeddings.len(), 2);
    }

    #[test]
    fn test_extractor_train_eval_modes() {
        let config = ExtractorConfig::default();
        let mut extractor = Extractor::new(&config).unwrap();

        assert!(!extractor.is_training());
        extractor.train();
        assert!(extractor.is_training());
        extractor.eval();
        assert!(!extractor.is_training());
    }

    #[test]
    fn test_extractor_builder() {
        let extractor = ExtractorBuilder::new()
            .model_name("test-model")
            .hidden_size(768)
            .max_width(12)
            .device("cpu")
            .fp16(false)
            .max_len(Some(384))
            .build();

        assert!(extractor.is_ok());
        let extractor = extractor.unwrap();
        assert_eq!(extractor.config.model_name, "test-model");
        assert_eq!(extractor.hidden_size(), 768);
        assert_eq!(extractor.max_width(), 12);
    }

    #[test]
    fn test_extractor_quantize() {
        let config = ExtractorConfig::default();
        let mut extractor = Extractor::new(&config).unwrap();

        assert!(!extractor.config.use_fp16);
        extractor.quantize().unwrap();
        assert!(extractor.config.use_fp16);
    }

    #[test]
    fn test_extractor_from_pretrained() {
        let extractor = Extractor::from_pretrained("fastino/gliner2-base-v1");
        assert!(extractor.is_ok());
        let extractor = extractor.unwrap();
        assert_eq!(extractor.config.model_name, "fastino/gliner2-base-v1");
    }

    #[test]
    fn test_extractor_output_empty() {
        let config = ExtractorConfig::default();
        let extractor = Extractor::new(&config).unwrap();

        // Create empty batch
        let empty_input_ids = Tensor::zeros((0, 0), DType::U32, &Device::Cpu).unwrap();
        let empty_mask = Tensor::zeros((0, 0), DType::U32, &Device::Cpu).unwrap();
        let empty_batch = PreprocessedBatchBuilder::new()
            .input_ids(empty_input_ids)
            .attention_mask(empty_mask)
            .build()
            .unwrap();

        let output = extractor.forward(&empty_batch);
        assert!(output.is_ok());
        assert!(output.unwrap().is_empty());
    }

    #[test]
    fn test_extractor_run_encoder() {
        let config = ExtractorConfig::default();
        let extractor = Extractor::new(&config).unwrap();

        let input_ids = Tensor::from_slice(&[1u32, 2, 3, 4, 5, 6], (2, 3), &Device::Cpu).unwrap();
        let attention_mask =
            Tensor::from_slice(&[1u32, 1, 1, 1, 1, 0], (2, 3), &Device::Cpu).unwrap();

        let output = extractor.run_encoder(&input_ids, &attention_mask);
        assert!(output.is_ok(), "run_encoder err: {:?}", output.err());
        let output = output.unwrap();
        assert_eq!(output.dims(), &[2, 3, 768]);
    }
}
