// Rust guideline compliant 2026-04-03
//! GLiNER2 Rust Implementation
//!
//! A Rust port of the GLiNER2 information extraction model, providing efficient
//! CPU-based inference for entity extraction, text classification, structured
//! data extraction, and relation extraction.
//!
//! # Quick Start
//!
//! ```ignore
//! use gliner2_rs::GLiNER2;
//! use gliner2_rs::schema::SchemaBuilder;
//!
//! // Load model
//! let model = GLiNER2::from_pretrained("fastino/gliner2-base-v1").await?;
//!
//! // Extract entities
//! let schema = SchemaBuilder::new()
//!     .entities(vec!["person".to_string(), "company".to_string()])
//!     .build()?;
//!
//! let result = model.extract_entities("Apple CEO Tim Cook", &schema).await?;
//! ```
//!
//! # Features
//!
//! - **Entity Extraction**: Named entity recognition with confidence scores and span positions
//! - **Text Classification**: Single and multi-label classification
//! - **Structured Data Extraction**: JSON structure parsing from text
//! - **Relation Extraction**: Relationship extraction between entities
//! - **Batch Processing**: Efficient batch inference with parallel preprocessing
//! - **CPU Optimized**: Fast inference on standard hardware without GPU

// -------------------------------------------------------------------------
// Public API
// -------------------------------------------------------------------------

pub mod batch;
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
