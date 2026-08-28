// Rust guideline compliant 2026-04-03
//! GLiNER2 / GLiNER2.5 Rust Implementation
//!
//! A pure Rust (candle) port of the GLiNER2 span-enumeration and GLiNER2.5
//! boundary-prediction information extraction models, providing efficient
//! CPU-based inference for entity extraction, text classification, structured
//! data extraction, relation extraction, and span attributes — with numeric
//! parity against the Python reference.
//!
//! # Quick Start (GLiNER2)
//!
//! ```ignore
//! use gliner2_candle::GLiNER2;
//! use gliner2_candle::schema::SchemaBuilder;
//!
//! // Load model (sync; downloads weights from HuggingFace Hub on first run)
//! let model = GLiNER2::from_pretrained("fastino/gliner2-base-v1")?;
//!
//! // Extract entities
//! let schema = SchemaBuilder::new()
//!     .entities(vec!["person".to_string(), "company".to_string()])
//!     .build()?;
//!
//! let result = model.extract_entities("Apple CEO Tim Cook", &schema)?;
//! ```
//!
//! # Quick Start (GLiNER2.5)
//!
//! GLiNER2.5 uses the boundary pipeline via `engine.extract` with a
//! `Schema::new` + optional `AttributeGroup`s schema. See the README for a
//! full example.
//!
//! # Features
//!
//! - **GLiNER2**: span-enumeration entity extraction with confidence scores and span positions
//! - **GLiNER2.5**: boundary prediction with entities, classifications, relations, and span attributes
//! - **Span Attributes**: single-/multi-label attributes attached to retained spans
//! - **Text Classification**: single and multi-label classification
//! - **Relation Extraction**: relationship extraction between entities
//! - **Long Documents**: auto-chunking (>384 words) and merge in `batch_extract`
//! - **CPU Optimized**: fast inference on standard hardware without GPU

// -------------------------------------------------------------------------
// Public API
// -------------------------------------------------------------------------

pub mod batch;
pub mod chunking;
pub mod config;
pub mod constraints;
pub mod error;
pub mod inference;
pub mod model;
pub mod schema;
pub mod tokenizer;

// Re-export main types for convenience
pub use config::ExtractorConfig;
pub use error::{GlinerError, Result};
pub use inference::engine::GLiNER2;
pub use schema::SchemaBuilder;
pub use tokenizer::WhitespaceTokenizer;

// -------------------------------------------------------------------------
// Version
// -------------------------------------------------------------------------

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
