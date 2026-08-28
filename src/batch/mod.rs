// Rust guideline compliant 2026-04-03
//! Batch module for GLiNER2.
//!
//! This module provides data structures and utilities for batching
//! inputs during inference, including the `PreprocessedBatch` struct
//! that holds GPU-ready tensors and metadata, and the `ExtractorCollator`
//! that converts raw text/schema pairs into batches.
//!
//! # Example
//!
//! ```ignore
//! use gliner2_candle::batch::{PreprocessedBatch, ExtractorCollator};
//!
//! let collator = ExtractorCollator::new(&processor, false);
//! let batch = collator.collate(&samples);
//! ```

pub mod collator;
pub mod preprocessed;

// Re-export commonly used types
pub use collator::ExtractorCollator;
pub use preprocessed::PreprocessedBatch;
pub use preprocessed::PreprocessedBatchBuilder;
