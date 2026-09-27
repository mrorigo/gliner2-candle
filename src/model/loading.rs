// Rust guideline compliant 2026-04-03
//! Model weight loading for GLiNER2.
//!
//! This module provides utilities for loading model weights from safetensors
//! files using candle's VarBuilder. It handles weight loading for all model
//! components: encoder, span representation, count prediction, and classifier.
//!
//! # Supported Formats
//!
//! - **Safetensors**: `.safetensors` files (preferred)
//! - **Sharded models**: Multiple `.safetensors` files
//!
//! # Weight Loading
//!
//! The loader uses candle's `VarBuilder::from_mmaped_safetensors()` to load
//! weights directly from safetensors files. Each model component is rebuilt
//! with the loaded weights via its `from_var_builder()` method.
//!
//! # Example
//!
//! ```ignore
//! use gliner2_candle::model::loading::ModelLoader;
//! use gliner2_candle::config::ExtractorConfig;
//! use candle_core::Device;
//!
//! let config = ExtractorConfig::default();
//! let loader = ModelLoader::new(&config, Device::Cpu)?;
//! loader.load_safetensors("model.safetensors", &mut model)?;
//! ```

use std::path::Path;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::config::ExtractorConfig;
use crate::error::{GlinerError, Result};
use crate::model::extractor::Extractor;

/// Model weight loader for GLiNER2.
///
/// This struct handles loading weights from safetensors files and rebuilding
/// model components with the loaded weights. It supports both single and
/// sharded safetensors files.
#[derive(Debug)]
pub struct ModelLoader {
    /// Model configuration.
    config: ExtractorConfig,
    /// Device to load tensors onto.
    device: Device,
}

impl ModelLoader {
    /// Create a new model loader.
    ///
    /// # Arguments
    ///
    /// * `config` - The extractor configuration.
    /// * `device` - The device to load tensors onto.
    ///
    /// # Errors
    ///
    /// This function currently does not fail, but returns `Result` for API
    /// consistency with other constructors.
    pub fn new(config: &ExtractorConfig, device: Device) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            device,
        })
    }

    /// Load weights from a safetensors file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the safetensors file.
    /// * `model` - The model to load weights into.
    ///
    /// # Returns
    ///
    /// `Ok(())` if weights were loaded successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is missing, unreadable, malformed, or if
    /// rebuilding model components fails.
    pub fn load_safetensors(&self, path: impl AsRef<Path>, model: &mut Extractor) -> Result<()> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(GlinerError::model_loading_with_path(
                "Safetensors file not found",
                path,
            ));
        }

        // Load weights via candle's VarBuilder.
        // SAFETY: `path` is validated to exist, and candle performs safetensors
        // parsing with bounds checks before exposing tensors. This call only
        // memory-maps immutable files and returns a `Result` on parse failure.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &self.device) }
            .map_err(|e| {
                GlinerError::model_loading_with_path(
                    format!("Failed to load safetensors: {e}"),
                    path,
                )
            })?;

        // Rebuild model components with loaded weights
        self.rebuild_model(vb, model)?;

        Ok(())
    }

    /// Load weights from multiple safetensors files (sharded models).
    ///
    /// # Arguments
    ///
    /// * `paths` - Paths to the safetensors files.
    /// * `model` - The model to load weights into.
    ///
    /// # Returns
    ///
    /// `Ok(())` if weights were loaded successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if no paths are provided, shard files cannot be read,
    /// or rebuilding model components fails.
    pub fn load_sharded_safetensors(
        &self,
        paths: &[impl AsRef<Path>],
        model: &mut Extractor,
    ) -> Result<()> {
        if paths.is_empty() {
            return Err(GlinerError::model_loading(
                "No safetensors files provided".to_string(),
            ));
        }

        let path_refs: Vec<&Path> = paths.iter().map(|p| p.as_ref()).collect();

        // Load all shards via candle's VarBuilder.
        // SAFETY: each path comes from `paths` and is only used for read-only
        // memory mapping. candle validates safetensors metadata and tensor
        // bounds and returns an error if any shard is invalid.
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&path_refs, DType::F32, &self.device) }
                .map_err(|e| {
                    GlinerError::model_loading(format!("Failed to load sharded safetensors: {e}"))
                })?;

        // Rebuild model components with loaded weights
        self.rebuild_model(vb, model)?;

        Ok(())
    }

    /// Load the optional count-embedding layer.
    ///
    /// Returns `Ok(None)` when the checkpoint ships no compatible weights.
    /// Checkpoints with `counting_layer: "count_lstm"` (e.g. the GLiNER2.5-Decide
    /// family) ship `count_embed.gru` + `count_embed.projector` instead of the
    /// historical `count_embed.transformer.*` block this port implements, so the
    /// probe below finds nothing and count guidance is simply unavailable. That
    /// leaves the count-guided entity scorer with no weights; classification
    /// decoding never touches this module.
    fn load_count_embed(
        &self,
        vb: VarBuilder,
    ) -> Result<Option<crate::model::count_embed::CountEmbedLayer>> {
        let count_embed_vb = vb.pp("count_embed");
        let hidden = self.config.hidden_size;
        const MAX_COUNT: usize = 20;
        const IN_PROJECTOR_DIM: usize = 128;

        if count_embed_vb
            .get((MAX_COUNT, hidden), "pos_embedding.weight")
            .is_err()
        {
            return Ok(None);
        }
        if count_embed_vb
            .pp("transformer")
            .pp("in_projector")
            .get((hidden, IN_PROJECTOR_DIM), "weight")
            .is_err()
        {
            return Ok(None);
        }

        crate::model::count_embed::CountEmbedLayer::from_var_builder(
            count_embed_vb,
            hidden,
            MAX_COUNT,
            self.device.clone(),
        )
        .map(Some)
        .map_err(|e| GlinerError::model_loading(format!("Failed to rebuild count_embed: {e}")))
    }

    /// Rebuild model components with weights from VarBuilder.
    ///
    /// # Arguments
    ///
    /// * `vb` - The VarBuilder containing the loaded weights.
    /// * `model` - The model to rebuild.
    fn rebuild_model(&self, vb: VarBuilder, model: &mut Extractor) -> Result<()> {
        // GLiNER2.5 boundary architecture: shared encoder + boundary head.
        if self.config.architecture == crate::config::Architecture::Gliner25 {
            model.encoder = crate::model::candle_encoder::CandleEncoder::from_var_builder(
                vb.clone(),
                &self.config,
                self.device.clone(),
            )
            .map_err(|e| GlinerError::model_loading(format!("Failed to rebuild encoder: {e}")))?;

            let boundary =
                crate::model::boundary::BoundaryModel::load(vb, &self.config, &self.device)
                    .map_err(|e| {
                        GlinerError::model_loading(format!("Failed to load boundary head: {e}"))
                    })?;
            model.boundary = Some(boundary);
            return Ok(());
        }

        // Rebuild encoder with loaded weights
        model.encoder = crate::model::candle_encoder::CandleEncoder::from_var_builder(
            vb.clone(),
            &self.config,
            self.device.clone(),
        )
        .map_err(|e| GlinerError::model_loading(format!("Failed to rebuild encoder: {e}")))?;

        // Rebuild span representation layer with loaded weights
        model.span_rep = crate::model::span_rep::SpanRepresentationLayer::from_var_builder(
            vb.clone(),
            &self.config,
            self.device.clone(),
        )
        .map_err(|e| GlinerError::model_loading(format!("Failed to rebuild span_rep: {e}")))?;

        // Rebuild count prediction layer with loaded weights
        model.count_pred = crate::model::count_pred::CountPredictionLayer::from_var_builder(
            vb.clone(),
            &self.config,
            self.device.clone(),
        )
        .map_err(|e| GlinerError::model_loading(format!("Failed to rebuild count_pred: {e}")))?;

        // Rebuild count embedding layer with loaded weights.
        model.count_embed = self.load_count_embed(vb.clone())?;

        // Rebuild classifier head with loaded weights
        model.classifier = crate::model::classifier::ClassifierHead::from_var_builder(
            vb,
            &self.config,
            self.device.clone(),
        )
        .map_err(|e| GlinerError::model_loading(format!("Failed to rebuild classifier: {e}")))?;

        Ok(())
    }

    /// Get the device the loader is using.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the model configuration.
    pub fn config(&self) -> &ExtractorConfig {
        &self.config
    }
}

/// Utility functions for working with model weights.
pub mod utils {
    use super::*;

    /// Check if a directory contains safetensors weights.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the directory.
    ///
    /// # Returns
    ///
    /// `true` if safetensors files are found.
    pub fn has_safetensors(path: impl AsRef<Path>) -> bool {
        let path = path.as_ref();
        if path.is_dir() {
            path.join("model.safetensors").exists()
                || path.join("model.safetensors.index.json").exists()
        } else {
            path.extension().is_some_and(|ext| ext == "safetensors")
        }
    }

    /// Get all safetensors files in a directory.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the directory.
    ///
    /// # Returns
    ///
    /// A vector of paths to safetensors files.
    pub fn get_safetensors_files(path: impl AsRef<Path>) -> Vec<std::path::PathBuf> {
        let path = path.as_ref();
        let mut files = if path.is_dir() {
            std::fs::read_dir(path)
                .ok()
                .into_iter()
                .flat_map(|entries| entries.flatten().map(|entry| entry.path()))
                .filter(|entry_path| {
                    entry_path
                        .extension()
                        .is_some_and(|ext| ext == "safetensors")
                })
                .collect()
        } else if path.exists() {
            vec![path.to_path_buf()]
        } else {
            Vec::new()
        };

        files.sort();
        files
    }

    /// Load a model from a HuggingFace model directory.
    ///
    /// This function automatically detects whether the model is sharded
    /// and loads all weights accordingly.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the model directory or file.
    /// * `config` - The extractor configuration.
    /// * `device` - The device to load tensors onto.
    ///
    /// # Returns
    ///
    /// A loaded `Extractor` model.
    ///
    /// # Errors
    ///
    /// Returns an error if model construction fails, no valid safetensors are
    /// found, or weight loading fails.
    pub fn load_from_hf_dir(
        model_path: impl AsRef<Path>,
        config: &ExtractorConfig,
        device: Device,
    ) -> Result<Extractor> {
        let model_path = model_path.as_ref();
        let mut model = Extractor::new(config)?;
        let loader = ModelLoader::new(config, device)?;

        if has_safetensors(model_path) {
            let files = get_safetensors_files(model_path);
            if files.is_empty() {
                return Err(GlinerError::model_loading_with_path(
                    "No safetensors files found",
                    model_path,
                ));
            }

            if files.len() == 1 {
                loader.load_safetensors(&files[0], &mut model)?;
            } else {
                loader.load_sharded_safetensors(&files, &mut model)?;
            }
        } else {
            return Err(GlinerError::model_loading_with_path(
                "No safetensors files found in directory",
                model_path,
            ));
        }

        Ok(model)
    }
}
