// Rust guideline compliant 2026-04-03
//! Inference module for GLiNER2.
//!
//! This module provides the main inference engine and extraction logic
//! for running GLiNER2 models. It coordinates model loading, schema
//! processing, batch inference, and result formatting.
//!
//! # Example
//!
//! ```ignore
//! use gliner2_rs::inference::GLiNER2;
//!
//! let model = GLiNER2::from_pretrained("fastino/gliner2-base-v1")?;
//! let result = model.extract_entities("Apple CEO Tim Cook", &["company", "person"])?;
//! ```

pub mod boundary;
pub mod engine;

// Re-export main types for convenience
pub use engine::GLiNER2;
