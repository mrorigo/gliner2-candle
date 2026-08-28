// Rust guideline compliant 2026-04-03
//! Model module for GLiNER2.
//!
//! This module provides the neural network components for the GLiNER2 model,
//! including the main `Extractor` model, span representation layer, count
//! prediction layers, classification head, and model weight loading utilities.
//!
//! # Architecture
//!
//! The GLiNER2 model consists of:
//! - **Encoder**: Transformer encoder (BERT-like) for token embeddings
//! - **Span Representation**: Computes span representations from token embeddings
//! - **Count Prediction**: Predicts number of instances for each schema
//! - **Classifier**: Classification head for classification tasks
//!
//! # Example
//!
//! ```ignore
//! use gliner2_candle::model::{Extractor, ExtractorConfig};
//!
//! let config = ExtractorConfig::default();
//! let model = Extractor::new(&config)?;
//! model.load_weights("path/to/weights.safetensors")?;
//! ```

pub mod boundary;
pub mod candle_encoder;
pub mod classifier;
pub mod count_embed;
pub mod count_pred;
pub mod deberta_v3;
pub mod extractor;
pub mod loading;
pub mod span_rep;
pub mod weight_mapping;

// Re-export main types
pub use boundary::BoundaryModel;
pub use candle_encoder::{CandleEncoder, EncoderType};
pub use classifier::ClassifierHead;
pub use count_pred::CountPredictionLayer;
pub use extractor::{Extractor, ExtractorOutput};
pub use loading::ModelLoader;
pub use span_rep::SpanRepresentationLayer;
pub use weight_mapping::{
    build_weight_map, is_classifier_weight, is_count_pred_weight, is_encoder_weight,
    is_span_rep_weight,
};
