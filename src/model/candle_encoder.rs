// Rust guideline compliant 2026-04-03
//! Candle-based transformer encoder for GLiNER2.
//!
//! This module provides a wrapper around candle's BERT and DeBERTa V2 models,
//! serving as the text encoder for GLiNER2. It handles model construction,
//! weight loading from safetensors, and the forward pass.
//!
//! # Architecture
//!
//! The encoder transforms token IDs into contextualized token embeddings:
//! - Input: `input_ids` (batch, seq_len), `attention_mask` (batch, seq_len)
//! - Output: `last_hidden_state` (batch, seq_len, hidden_size)
//!
//! # Supported Models
//!
//! - **BERT**: Standard BERT architecture (e.g., bert-base-uncased)
//! - **DeBERTa V2**: DeBERTa V2 architecture (e.g., microsoft/deberta-v3-base)
//!
//! # Example
//!
//! ```ignore
//! use gliner2_candle::model::candle_encoder::CandleEncoder;
//! use gliner2_candle::config::ExtractorConfig;
//! use candle_core::Device;
//!
//! let config = ExtractorConfig::default();
//! let encoder = CandleEncoder::new(&config, Device::Cpu)?;
//!
//! // Forward pass
//! let embeddings = encoder.forward(&input_ids, &attention_mask)?;
//! ```

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{bert, debertav2};

use crate::config::ExtractorConfig;
use crate::error::{GlinerError, Result};

/// Supported encoder architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderType {
    /// Standard BERT architecture.
    Bert,
    /// DeBERTa V2 architecture.
    DebertaV2,
    /// DeBERTa V3 architecture (no token_type_embeddings).
    DebertaV3,
}

impl EncoderType {
    /// Detect encoder type from model name.
    ///
    /// Returns `DebertaV3` for DeBERTa-v3/GLiNER2 model IDs, `DebertaV2`
    /// for other DeBERTa names, otherwise `Bert`.
    pub fn from_model_name(model_name: &str) -> Self {
        let lower = model_name.to_lowercase();
        if lower.contains("deberta-v3") || lower.contains("gliner2") {
            EncoderType::DebertaV3
        } else if lower.contains("deberta") {
            EncoderType::DebertaV2
        } else {
            EncoderType::Bert
        }
    }

    /// Detect encoder type from weight names in a safetensors file.
    ///
    /// Reads the safetensors header to get weight names and detects the encoder type
    /// based on DeBERTa-specific patterns (e.g., `key_proj`, `query_proj`, `rel_embeddings`).
    ///
    /// # Errors
    ///
    /// Returns an error if the safetensors file cannot be read or parsed.
    pub fn from_safetensors_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        use std::io::Read;
        let path = path.as_ref();
        let mut file = std::fs::File::open(path).map_err(|e| {
            GlinerError::model_loading_with_path(format!("Failed to open safetensors: {e}"), path)
        })?;

        // Read header length (first 8 bytes)
        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes).map_err(|e| {
            GlinerError::model_loading_with_path(format!("Failed to read header: {e}"), path)
        })?;
        let header_len = u64::from_le_bytes(len_bytes) as usize;

        // Read header JSON
        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes).map_err(|e| {
            GlinerError::model_loading_with_path(format!("Failed to read header: {e}"), path)
        })?;

        let header: std::collections::HashMap<String, serde_json::Value> =
            serde_json::from_slice(&header_bytes).map_err(|e| {
                GlinerError::model_loading_with_path(format!("Failed to parse header: {e}"), path)
            })?;

        // Check for DeBERTa-specific weight patterns
        let has_rel_embeddings = header.keys().any(|name| name.contains("rel_embeddings"));
        let has_key_proj = header.keys().any(|name| name.contains("key_proj"));
        let has_token_type = header
            .keys()
            .any(|name| name.contains("token_type_embeddings"));

        // DeBERTa V3 has rel_embeddings but no token_type_embeddings
        if has_rel_embeddings && !has_token_type {
            Ok(EncoderType::DebertaV3)
        } else if has_key_proj || has_rel_embeddings {
            Ok(EncoderType::DebertaV2)
        } else {
            Ok(EncoderType::Bert)
        }
    }
}

/// Candle-based transformer encoder for GLiNER2.
///
/// This struct wraps either a BERT or DeBERTa V2 model from candle,
/// providing a unified interface for encoding token IDs into embeddings.
///
/// # Tensor Shapes
///
/// - Input `input_ids`: `(batch_size, seq_len)`
/// - Input `attention_mask`: `(batch_size, seq_len)`
/// - Output: `(batch_size, seq_len, hidden_size)`
pub struct CandleEncoder {
    /// The underlying model (BERT or DeBERTa V2).
    model: EncoderModel,
    /// Device for tensor operations.
    device: Device,
    /// Hidden size of the model.
    hidden_size: usize,
    /// Whether the model has been loaded with weights.
    is_loaded: bool,
}

/// Internal enum holding either BERT or DeBERTa V2 model.
enum EncoderModel {
    Bert(Box<bert::BertModel>),
    DebertaV2(Box<debertav2::DebertaV2Model>),
    DebertaV3(Box<crate::model::deberta_v3::DebertaV3Model>),
}

impl CandleEncoder {
    /// Create a new encoder from configuration with random/uninitialized weights.
    ///
    /// # Arguments
    ///
    /// * `config` - The extractor configuration.
    /// * `device` - The device to run the encoder on.
    ///
    /// # Returns
    ///
    /// A new `CandleEncoder` with uninitialized weights.
    ///
    /// # Errors
    ///
    /// Returns an error if encoder initialization fails.
    pub fn new(config: &ExtractorConfig, device: Device) -> Result<Self> {
        let encoder_type = EncoderType::from_model_name(&config.model_name);
        let hidden_size = config.hidden_size;

        // Create a VarBuilder with random initialization via VarMap
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = Self::load_model(vb, config, encoder_type)?;

        Ok(Self {
            model,
            device,
            hidden_size,
            is_loaded: false,
        })
    }

    /// Create a new encoder loaded from a VarBuilder.
    ///
    /// # Arguments
    ///
    /// * `vb` - The VarBuilder containing the weights.
    /// * `config` - The extractor configuration.
    /// * `device` - The device to run the encoder on.
    ///
    /// # Returns
    ///
    /// A new `CandleEncoder` with weights loaded from the VarBuilder.
    ///
    /// # Errors
    ///
    /// Returns an error if encoder construction fails for the selected backend.
    pub fn from_var_builder(
        vb: VarBuilder,
        config: &ExtractorConfig,
        device: Device,
    ) -> Result<Self> {
        // Detect encoder type by checking for DeBERTa-specific weight names
        // DeBERTa V3 has rel_embeddings but no token_type_embeddings
        let has_rel = vb.contains_tensor("encoder.encoder.rel_embeddings.weight");
        let has_token_type = vb.contains_tensor("encoder.embeddings.token_type_embeddings.weight");
        let has_key_proj =
            vb.contains_tensor("encoder.encoder.layer.0.attention.self.key_proj.weight");

        let encoder_type = if has_rel && !has_token_type {
            EncoderType::DebertaV3
        } else if has_key_proj || has_rel {
            EncoderType::DebertaV2
        } else {
            EncoderType::Bert
        };
        let hidden_size = config.hidden_size;

        let model = Self::load_model(vb, config, encoder_type)?;

        Ok(Self {
            model,
            device,
            hidden_size,
            is_loaded: true,
        })
    }

    /// Create a new encoder loaded from a safetensors file.
    ///
    /// # Arguments
    ///
    /// * `config` - The extractor configuration.
    /// * `path` - Path to the safetensors file.
    /// * `device` - The device to run the encoder on.
    ///
    /// # Returns
    ///
    /// A new `CandleEncoder` with weights loaded from the file.
    ///
    /// # Errors
    ///
    /// Returns an error if safetensors loading fails or encoder construction fails.
    pub fn from_safetensors(
        config: &ExtractorConfig,
        path: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self> {
        let path = path.as_ref();
        // Detect encoder type from actual weight names in the safetensors file
        let encoder_type = EncoderType::from_safetensors_path(path)?;
        let hidden_size = config.hidden_size;

        // Load weights from safetensors.
        // SAFETY: `path` is provided by the caller and read-only memory mapped.
        // candle validates safetensors headers and tensor offsets before
        // constructing tensors, returning an error on malformed files.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device) }
            .map_err(|e| {
                GlinerError::model_loading_with_path(
                    format!("Failed to load safetensors: {e}"),
                    path,
                )
            })?;

        let model = Self::load_model(vb, config, encoder_type)?;

        Ok(Self {
            model,
            device,
            hidden_size,
            is_loaded: true,
        })
    }

    /// Load the appropriate model type from VarBuilder.
    fn load_model(
        vb: VarBuilder,
        config: &ExtractorConfig,
        encoder_type: EncoderType,
    ) -> Result<EncoderModel> {
        // GLiNER2 weights are stored with "encoder." prefix in safetensors
        // (e.g., "encoder.embeddings.word_embeddings.weight")
        // Candle's BERT/DeBERTa models expect names without the prefix
        // (e.g., "embeddings.word_embeddings.weight")
        // So we add the "encoder" prefix to the VarBuilder to match the stored names.
        let vb = vb.pp("encoder");

        tracing::info!(
            "Loading {:?} encoder with vocab_size={}, hidden_size={}",
            encoder_type,
            config.vocab_size,
            config.hidden_size
        );

        match encoder_type {
            EncoderType::Bert => {
                let bert_config = Self::build_bert_config(config);
                let model = bert::BertModel::load(vb, &bert_config).map_err(|e| {
                    GlinerError::model_loading(format!("Failed to load BERT model: {e}"))
                })?;
                Ok(EncoderModel::Bert(Box::new(model)))
            }
            EncoderType::DebertaV2 => {
                let deberta_config = Self::build_deberta_config(config);
                let model = debertav2::DebertaV2Model::load(vb, &deberta_config).map_err(|e| {
                    GlinerError::model_loading(format!("Failed to load DeBERTa V2 model: {e}"))
                })?;
                Ok(EncoderModel::DebertaV2(Box::new(model)))
            }
            EncoderType::DebertaV3 => {
                let deberta_config = Self::build_deberta_v3_config(config);
                let model = crate::model::deberta_v3::DebertaV3Model::load(vb, &deberta_config)
                    .map_err(|e| {
                        GlinerError::model_loading(format!("Failed to load DeBERTa V3 model: {e}"))
                    })?;
                Ok(EncoderModel::DebertaV3(Box::new(model)))
            }
        }
    }

    /// Forward pass through the encoder.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Token IDs of shape `(batch_size, seq_len)`.
    /// * `attention_mask` - Attention mask of shape `(batch_size, seq_len)`.
    ///
    /// # Returns
    ///
    /// Token embeddings of shape `(batch_size, seq_len, hidden_size)`.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor creation fails or the backend forward pass fails.
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let token_type_ids = Tensor::zeros_like(input_ids).map_err(|e| {
            GlinerError::model_loading(format!("Failed to create token type IDs: {e}"))
        })?;

        match &self.model {
            EncoderModel::Bert(model) => model
                .forward(input_ids, &token_type_ids, Some(attention_mask))
                .map_err(|e| GlinerError::model_loading(format!("BERT forward pass failed: {e}"))),
            EncoderModel::DebertaV2(model) => model
                .forward(input_ids, None, Some(attention_mask.clone()))
                .map_err(|e| {
                    GlinerError::model_loading(format!("DeBERTa V2 forward pass failed: {e}"))
                }),
            EncoderModel::DebertaV3(model) => model
                .forward(input_ids, &token_type_ids, Some(attention_mask))
                .map_err(|e| {
                    GlinerError::model_loading(format!("DeBERTa V3 forward pass failed: {e}"))
                }),
        }
    }

    /// Debug staged forward: [embeddings, layer0, .., final].
    /// Per-layer sub-stages for parity bisection (DeBERTa V3 only).
    pub fn forward_substages(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Vec<(Tensor, Tensor, Tensor, Tensor)>> {
        match &self.model {
            EncoderModel::DebertaV3(model) => model
                .forward_substages(input_ids, Some(attention_mask))
                .map_err(|e| GlinerError::model_loading(format!("{e}"))),
            _ => Err(GlinerError::model_loading(
                "forward_substages only supports DebertaV3",
            )),
        }
    }

    pub fn forward_debug(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Vec<Tensor>> {
        match &self.model {
            EncoderModel::DebertaV3(model) => model
                .forward_debug(input_ids, Some(attention_mask))
                .map_err(|e| GlinerError::model_loading(format!("debug forward failed: {e}"))),
            _ => Err(GlinerError::model_loading(
                "forward_debug only supports DebertaV3",
            )),
        }
    }

    /// Get the hidden size of the encoder.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get the device the encoder is on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Check if the encoder has been loaded with weights.
    pub fn is_loaded(&self) -> bool {
        self.is_loaded
    }

    /// Load weights from a safetensors file into an existing encoder.
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
    /// Returns an error if the safetensors file cannot be loaded or model
    /// reconstruction fails.
    pub fn load_weights(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();

        // SAFETY: `path` points to the requested safetensors file and is only
        // mapped for immutable reads. candle performs format and bounds checks
        // and surfaces failures as `Result` errors.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &self.device) }
            .map_err(|e| {
                GlinerError::model_loading_with_path(
                    format!("Failed to load safetensors: {e}"),
                    path,
                )
            })?;

        let encoder_type = match &self.model {
            EncoderModel::Bert(_) => EncoderType::Bert,
            EncoderModel::DebertaV2(_) => EncoderType::DebertaV2,
            EncoderModel::DebertaV3(_) => EncoderType::DebertaV3,
        };

        // Rebuild config from current state
        let config = ExtractorConfig {
            hidden_size: self.hidden_size,
            ..Default::default()
        };

        self.model = Self::load_model(vb, &config, encoder_type)?;
        self.is_loaded = true;

        Ok(())
    }

    /// Build a candle BERT config from the extractor config.
    fn build_bert_config(config: &ExtractorConfig) -> bert::Config {
        bert::Config {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            intermediate_size: config.intermediate_size,
            hidden_act: Self::map_hidden_act(config.hidden_act),
            hidden_dropout_prob: config.hidden_dropout_prob as f64,
            max_position_embeddings: config.max_position_embeddings,
            type_vocab_size: config.type_vocab_size,
            initializer_range: 0.02,
            layer_norm_eps: config.layer_norm_eps as f64,
            pad_token_id: config.pad_token_id,
            position_embedding_type: bert::PositionEmbeddingType::Absolute,
            use_cache: true,
            classifier_dropout: None,
            model_type: Some("bert".to_string()),
        }
    }

    /// Build a DeBERTa V3 config from the extractor config.
    fn build_deberta_v3_config(
        config: &ExtractorConfig,
    ) -> crate::model::deberta_v3::DebertaV3Config {
        crate::model::deberta_v3::DebertaV3Config {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            intermediate_size: config.intermediate_size,
            hidden_dropout_prob: config.hidden_dropout_prob as f64,
            max_position_embeddings: config.max_position_embeddings,
            layer_norm_eps: 1e-7,
            pad_token_id: config.pad_token_id,
            share_att_key: true,
            relative_attention: true,
            max_relative_positions: -1,
            pos_att_type: vec!["p2c".to_string(), "c2p".to_string()],
            position_buckets: 256,
        }
    }

    /// Build a candle DeBERTa V2 config from the extractor config (for compatibility).
    fn build_deberta_config(config: &ExtractorConfig) -> debertav2::Config {
        debertav2::Config {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            intermediate_size: config.intermediate_size,
            hidden_act: Self::map_deberta_act(config.hidden_act),
            hidden_dropout_prob: config.hidden_dropout_prob as f64,
            attention_probs_dropout_prob: config.attention_probs_dropout_prob as f64,
            max_position_embeddings: config.max_position_embeddings,
            type_vocab_size: config.type_vocab_size,
            initializer_range: 0.02,
            layer_norm_eps: config.layer_norm_eps as f64,
            relative_attention: true,
            max_relative_positions: 512,
            pad_token_id: Some(config.pad_token_id),
            position_biased_input: false,
            pos_att_type: vec!["p2c".to_string(), "c2p".to_string()],
            position_buckets: None,
            share_att_key: None,
            attention_head_size: None,
            embedding_size: None,
            norm_rel_ebd: None,
            conv_kernel_size: None,
            conv_groups: None,
            conv_act: None,
            id2label: None,
            label2id: None,
            pooler_dropout: None,
            pooler_hidden_act: None,
            pooler_hidden_size: None,
            cls_dropout: None,
        }
    }

    /// Map the hidden activation function to candle's BERT enum.
    fn map_hidden_act(act: crate::config::HiddenActivation) -> bert::HiddenAct {
        match act {
            crate::config::HiddenActivation::Gelu => bert::HiddenAct::Gelu,
            crate::config::HiddenActivation::GeluApproximate => bert::HiddenAct::GeluApproximate,
            crate::config::HiddenActivation::Relu => bert::HiddenAct::Relu,
            crate::config::HiddenActivation::Silu => bert::HiddenAct::Gelu, // Fallback
        }
    }

    /// Map the hidden activation function to candle's DeBERTa enum.
    fn map_deberta_act(act: crate::config::HiddenActivation) -> debertav2::HiddenAct {
        match act {
            crate::config::HiddenActivation::Gelu => debertav2::HiddenAct::Gelu,
            crate::config::HiddenActivation::GeluApproximate => {
                debertav2::HiddenAct::GeluApproximate
            }
            crate::config::HiddenActivation::Relu => debertav2::HiddenAct::Relu,
            crate::config::HiddenActivation::Silu => debertav2::HiddenAct::Gelu, // Fallback
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ExtractorConfig;

    #[test]
    fn test_encoder_type_detection() {
        assert_eq!(
            EncoderType::from_model_name("bert-base-uncased"),
            EncoderType::Bert
        );
        assert_eq!(
            EncoderType::from_model_name("microsoft/deberta-v3-base"),
            EncoderType::DebertaV3
        );
        assert_eq!(
            EncoderType::from_model_name("fastino/gliner2-base-v1"),
            EncoderType::DebertaV3
        );
    }

    #[test]
    fn test_bert_config_mapping() {
        let config = ExtractorConfig::default();
        let bert_config = CandleEncoder::build_bert_config(&config);

        assert_eq!(bert_config.hidden_size, 768);
        assert_eq!(bert_config.num_hidden_layers, 12);
        assert_eq!(bert_config.num_attention_heads, 12);
        assert_eq!(bert_config.intermediate_size, 3072);
        assert_eq!(bert_config.vocab_size, 30522);
    }

    #[test]
    fn test_deberta_config_mapping() {
        let config = ExtractorConfig::default();
        let deberta_config = CandleEncoder::build_deberta_config(&config);

        assert_eq!(deberta_config.hidden_size, 768);
        assert_eq!(deberta_config.num_hidden_layers, 12);
        assert_eq!(deberta_config.num_attention_heads, 12);
        assert_eq!(deberta_config.intermediate_size, 3072);
        assert_eq!(deberta_config.vocab_size, 30522);
    }
}
