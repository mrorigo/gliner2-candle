// Rust guideline compliant 2026-04-03
//! Error types for GLiNER2 Rust implementation.
//!
//! This module defines all error types used throughout the library,
//! providing detailed context for debugging and error handling.

use std::path::PathBuf;
use thiserror::Error;

/// Main result type alias for GLiNER2 operations.
pub type Result<T> = std::result::Result<T, GlinerError>;

/// Comprehensive error type for all GLiNER2 operations.
#[derive(Error, Debug)]
pub enum GlinerError {
    // -------------------------------------------------------------------------
    // Tokenizer Errors
    // -------------------------------------------------------------------------
    #[error("Tokenizer error: {message}")]
    Tokenizer {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    // -------------------------------------------------------------------------
    // Schema Errors
    // -------------------------------------------------------------------------
    #[error("Invalid schema: {message}")]
    InvalidSchema { message: String },

    #[error("Schema transformation error: {message}")]
    SchemaTransformation { message: String },

    // -------------------------------------------------------------------------
    // Model Loading Errors
    // -------------------------------------------------------------------------
    #[error("Model loading failed: {message}")]
    ModelLoading {
        message: String,
        path: Option<PathBuf>,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Weight loading error: {message}")]
    WeightLoading {
        message: String,
        layer_name: String,
        expected_shape: Vec<usize>,
        actual_shape: Vec<usize>,
    },

    #[error("Configuration error: {message}")]
    Config {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    // -------------------------------------------------------------------------
    // Inference Errors
    // -------------------------------------------------------------------------
    #[error("Inference failed: {message}")]
    Inference {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Batch processing error: {message}")]
    BatchProcessing { message: String },

    #[error("Dimension mismatch: expected {expected:?}, got {actual:?}")]
    DimensionMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    // -------------------------------------------------------------------------
    // Tensor/Backend Errors
    // -------------------------------------------------------------------------
    #[error("Tensor operation failed: {message}")]
    Tensor {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    // -------------------------------------------------------------------------
    // I/O Errors
    // -------------------------------------------------------------------------
    #[error("I/O error: {path:?}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("Download failed: {url}")]
    Download {
        url: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    // -------------------------------------------------------------------------
    // Serialization Errors
    // -------------------------------------------------------------------------
    #[error("Serialization error: {message}")]
    Serialization {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    // -------------------------------------------------------------------------
    // Validation Errors
    // -------------------------------------------------------------------------
    #[error("Validation error: {message}")]
    Validation { message: String },

    // -------------------------------------------------------------------------
    // Regex Validator Errors
    // -------------------------------------------------------------------------
    #[error("Regex validator error: {message}")]
    RegexValidator {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    // -------------------------------------------------------------------------
    // Generic/Other Errors
    // -------------------------------------------------------------------------
    #[error("{message}")]
    Other {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}

impl GlinerError {
    /// Create a tokenizer error.
    pub fn tokenizer(message: impl Into<String>) -> Self {
        Self::Tokenizer {
            message: message.into(),
            source: None,
        }
    }

    /// Create a tokenizer error with a source.
    pub fn tokenizer_with_source(
        message: impl Into<String>,
        source: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::Tokenizer {
            message: message.into(),
            source: Some(source.into()),
        }
    }

    /// Create an invalid schema error.
    pub fn invalid_schema(message: impl Into<String>) -> Self {
        Self::InvalidSchema {
            message: message.into(),
        }
    }

    /// Create a schema transformation error.
    pub fn schema_transformation(message: impl Into<String>) -> Self {
        Self::SchemaTransformation {
            message: message.into(),
        }
    }

    /// Create a model loading error.
    pub fn model_loading(message: impl Into<String>) -> Self {
        Self::ModelLoading {
            message: message.into(),
            path: None,
            source: None,
        }
    }

    /// Create a model loading error with path.
    pub fn model_loading_with_path(message: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        Self::ModelLoading {
            message: message.into(),
            path: Some(path.into()),
            source: None,
        }
    }

    /// Create a model loading error with source.
    pub fn model_loading_with_source(
        message: impl Into<String>,
        source: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::ModelLoading {
            message: message.into(),
            path: None,
            source: Some(source.into()),
        }
    }

    /// Create a weight loading error.
    pub fn weight_loading(
        layer_name: impl Into<String>,
        expected_shape: Vec<usize>,
        actual_shape: Vec<usize>,
    ) -> Self {
        let layer_name = layer_name.into();
        Self::WeightLoading {
            message: format!("Weight shape mismatch for layer '{}'", layer_name),
            layer_name,
            expected_shape,
            actual_shape,
        }
    }

    /// Create a configuration error.
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
            source: None,
        }
    }

    /// Create an inference error.
    pub fn inference(message: impl Into<String>) -> Self {
        Self::Inference {
            message: message.into(),
            source: None,
        }
    }

    /// Create an inference error with source.
    pub fn inference_with_source(
        message: impl Into<String>,
        source: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::Inference {
            message: message.into(),
            source: Some(source.into()),
        }
    }

    /// Create a batch processing error.
    pub fn batch_processing(message: impl Into<String>) -> Self {
        Self::BatchProcessing {
            message: message.into(),
        }
    }

    /// Create a dimension mismatch error.
    pub fn dimension_mismatch(expected: Vec<usize>, actual: Vec<usize>) -> Self {
        Self::DimensionMismatch { expected, actual }
    }

    /// Create a tensor error.
    pub fn tensor(message: impl Into<String>) -> Self {
        Self::Tensor {
            message: message.into(),
            source: None,
        }
    }

    /// Create a tensor error with source.
    pub fn tensor_with_source(
        message: impl Into<String>,
        source: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::Tensor {
            message: message.into(),
            source: Some(source.into()),
        }
    }

    /// Create an I/O error.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// Create a download error.
    pub fn download(url: impl Into<String>) -> Self {
        Self::Download {
            url: url.into(),
            source: None,
        }
    }

    /// Create a serialization error.
    pub fn serialization(message: impl Into<String>) -> Self {
        Self::Serialization {
            message: message.into(),
            source: None,
        }
    }

    /// Create a validation error.
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }

    /// Create a regex validator error.
    pub fn regex_validator(message: impl Into<String>) -> Self {
        Self::RegexValidator {
            message: message.into(),
            source: None,
        }
    }

    /// Create an other error.
    pub fn other(message: impl Into<String>) -> Self {
        Self::Other {
            message: message.into(),
            source: None,
        }
    }
}

// Convenience conversions

impl From<candle_core::Error> for GlinerError {
    fn from(err: candle_core::Error) -> Self {
        Self::Tensor {
            message: format!("candle error: {err}"),
            source: Some(Box::new(err)),
        }
    }
}

impl From<std::io::Error> for GlinerError {
    fn from(err: std::io::Error) -> Self {
        Self::Other {
            message: format!("I/O error: {err}"),
            source: Some(Box::new(err)),
        }
    }
}

impl From<serde_json::Error> for GlinerError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization {
            message: format!("JSON error: {err}"),
            source: Some(Box::new(err)),
        }
    }
}

impl From<regex::Error> for GlinerError {
    fn from(err: regex::Error) -> Self {
        Self::RegexValidator {
            message: format!("Regex error: {err}"),
            source: Some(Box::new(err)),
        }
    }
}

impl From<safetensors::SafeTensorError> for GlinerError {
    fn from(err: safetensors::SafeTensorError) -> Self {
        Self::WeightLoading {
            message: format!("Safetensors error: {err}"),
            layer_name: "unknown".to_string(),
            expected_shape: vec![],
            actual_shape: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = GlinerError::invalid_schema("Missing entity types");
        assert_eq!(format!("{err}"), "Invalid schema: Missing entity types");

        let err = GlinerError::dimension_mismatch(vec![3, 4], vec![3, 5]);
        assert_eq!(
            format!("{err}"),
            "Dimension mismatch: expected [3, 4], got [3, 5]"
        );
    }

    #[test]
    fn test_error_chain() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = GlinerError::io(PathBuf::from("/tmp/test.json"), io_err);
        assert!(format!("{err}").contains("/tmp/test.json"));
    }
}
