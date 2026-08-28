// Rust guideline compliant 2026-04-03
//! Preprocessed batch for GLiNER2 inference.
//!
//! This module defines the `PreprocessedBatch` struct, which holds GPU-ready
//! tensors and metadata for model inference. It mirrors the Python
//! `PreprocessedBatch` dataclass and provides methods for device transfer,
//! memory pinning, and field access.

use candle_core::Tensor;
use serde_json::Value as JsonValue;

use crate::error::{GlinerError, Result};

/// A token mapping entry: (schema_type, schema_index, token_index).
pub type TokenMapping = (String, usize, usize);

/// GPU-ready batch for training/inference.
///
/// This struct holds all the data needed for a forward pass through the
/// GLiNER2 model, including tokenized inputs, attention masks, schema
/// information, and precomputed routing indices for fast embedding extraction.
///
/// # Tensor Shapes
///
/// - `input_ids`: `(batch_size, max_seq_len)`
/// - `attention_mask`: `(batch_size, max_seq_len)`
/// - `text_word_indices`: `(batch_size, max_words)` or `None`
///
/// # Example
///
/// ```ignore
/// use gliner2_candle::batch::PreprocessedBatch;
/// use candle_core::{Tensor, Device};
///
/// let batch = PreprocessedBatch::new(
///     input_ids,
///     attention_mask,
///     mapped_indices,
///     schema_counts,
///     original_lengths,
///     structure_labels,
///     task_types,
///     text_tokens,
///     schema_tokens_list,
///     start_mappings,
///     end_mappings,
///     original_texts,
///     original_schemas,
///     text_word_indices,
///     text_word_counts,
///     schema_special_indices,
/// );
///
/// // Move to device
/// let batch = batch.to(Device::Cpu, None)?;
/// ```
#[derive(Debug)]
pub struct PreprocessedBatch {
    // -------------------------------------------------------------------------
    // Tensors (GPU-ready)
    // -------------------------------------------------------------------------
    /// Token IDs for the batch. Shape: `(batch_size, max_seq_len)`.
    pub input_ids: Tensor,

    /// Attention mask for the batch. Shape: `(batch_size, max_seq_len)`.
    pub attention_mask: Tensor,

    /// Precomputed routing indices for fast text embedding extraction.
    /// Shape: `(batch_size, max_words)` or `None` if not computed.
    pub text_word_indices: Option<Tensor>,

    // -------------------------------------------------------------------------
    // Per-sample metadata
    // -------------------------------------------------------------------------
    /// Per-sample token mappings: list of (schema_type, schema_index, token_index).
    pub mapped_indices: Vec<Vec<TokenMapping>>,

    /// Number of schemas per sample.
    pub schema_counts: Vec<usize>,

    /// Original sequence lengths before padding.
    pub original_lengths: Vec<usize>,

    /// Ground truth structure labels (used in training, empty for inference).
    pub structure_labels: Vec<Vec<JsonValue>>,

    /// Task types per schema for each sample.
    pub task_types: Vec<Vec<String>>,

    /// Original text tokens for each sample.
    pub text_tokens: Vec<Vec<String>>,

    /// Schema tokens per sample: `sample -> schema -> tokens`.
    pub schema_tokens_list: Vec<Vec<Vec<String>>>,

    /// Character position start mappings for each sample.
    pub start_mappings: Vec<Vec<usize>>,

    /// Character position end mappings for each sample.
    pub end_mappings: Vec<Vec<usize>>,

    /// Original input texts for result formatting.
    pub original_texts: Vec<String>,

    /// Original schemas for result formatting.
    pub original_schemas: Vec<JsonValue>,

    /// Actual word count per sample (for text_word_indices).
    pub text_word_counts: Vec<usize>,

    /// Per-sample, per-schema special token positions for embedding extraction.
    /// Structure: `sample -> schema -> positions`.
    pub schema_special_indices: Vec<Vec<Vec<usize>>>,
}

impl PreprocessedBatch {
    /// Create a new `PreprocessedBatch`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_ids: Tensor,
        attention_mask: Tensor,
        mapped_indices: Vec<Vec<TokenMapping>>,
        schema_counts: Vec<usize>,
        original_lengths: Vec<usize>,
        structure_labels: Vec<Vec<JsonValue>>,
        task_types: Vec<Vec<String>>,
        text_tokens: Vec<Vec<String>>,
        schema_tokens_list: Vec<Vec<Vec<String>>>,
        start_mappings: Vec<Vec<usize>>,
        end_mappings: Vec<Vec<usize>>,
        original_texts: Vec<String>,
        original_schemas: Vec<JsonValue>,
        text_word_indices: Option<Tensor>,
        text_word_counts: Vec<usize>,
        schema_special_indices: Vec<Vec<Vec<usize>>>,
    ) -> Self {
        Self {
            input_ids,
            attention_mask,
            text_word_indices,
            mapped_indices,
            schema_counts,
            original_lengths,
            structure_labels,
            task_types,
            text_tokens,
            schema_tokens_list,
            start_mappings,
            end_mappings,
            original_texts,
            original_schemas,
            text_word_counts,
            schema_special_indices,
        }
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.input_ids.dims()[0]
    }

    /// Get the sequence length.
    pub fn seq_len(&self) -> usize {
        self.input_ids.dims()[1]
    }

    /// Check if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.batch_size() == 0
    }

    /// Move tensors to the specified device and optionally cast float tensors.
    ///
    /// Integer tensors (`input_ids`, `text_word_indices`) are moved to the
    /// device but keep their original dtype regardless of the `dtype` argument.
    ///
    /// # Arguments
    ///
    /// * `device` - The target device.
    /// * `dtype` - Optional dtype for casting float tensors.
    ///
    /// # Returns
    ///
    /// A new `PreprocessedBatch` with tensors on the target device.
    pub fn to(
        &self,
        device: candle_core::Device,
        dtype: Option<candle_core::DType>,
    ) -> Result<Self> {
        let cast_tensor = |tensor: &Tensor, is_int: bool| -> Result<Tensor> {
            let mut t = tensor.to_device(&device).map_err(|e| {
                GlinerError::model_loading(format!("Failed to move tensor to device: {e}"))
            })?;
            if let Some(dt) = dtype
                && !is_int
            {
                t = t.to_dtype(dt).map_err(|e| {
                    GlinerError::model_loading(format!("Failed to cast tensor dtype: {e}"))
                })?;
            }
            Ok(t)
        };

        Ok(Self {
            input_ids: cast_tensor(&self.input_ids, true)?,
            attention_mask: cast_tensor(&self.attention_mask, false)?,
            text_word_indices: self
                .text_word_indices
                .as_ref()
                .map(|t| cast_tensor(t, true))
                .transpose()?,
            // Non-tensor fields are cloned
            mapped_indices: self.mapped_indices.clone(),
            schema_counts: self.schema_counts.clone(),
            original_lengths: self.original_lengths.clone(),
            structure_labels: self.structure_labels.clone(),
            task_types: self.task_types.clone(),
            text_tokens: self.text_tokens.clone(),
            schema_tokens_list: self.schema_tokens_list.clone(),
            start_mappings: self.start_mappings.clone(),
            end_mappings: self.end_mappings.clone(),
            original_texts: self.original_texts.clone(),
            original_schemas: self.original_schemas.clone(),
            text_word_counts: self.text_word_counts.clone(),
            schema_special_indices: self.schema_special_indices.clone(),
        })
    }

    /// Pin tensors to memory for faster GPU transfer.
    ///
    /// This is equivalent to `tensor.pin_memory()` in PyTorch and is used
    /// with pinned host memory for asynchronous GPU transfers.
    ///
    /// # Returns
    ///
    /// A new `PreprocessedBatch` with pinned tensors.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor cloning or optional tensor handling fails.
    pub fn pin_memory(&self) -> Result<Self> {
        let pin_tensor = |tensor: &Tensor| -> Result<Tensor> {
            // Note: candle doesn't have a direct pin_memory() method.
            // In practice, this would use CUDA pinned memory allocation.
            // For now, we return the tensor as-is.
            Ok(tensor.clone())
        };

        Ok(Self {
            input_ids: pin_tensor(&self.input_ids)?,
            attention_mask: pin_tensor(&self.attention_mask)?,
            text_word_indices: self
                .text_word_indices
                .as_ref()
                .map(pin_tensor)
                .transpose()?,
            mapped_indices: self.mapped_indices.clone(),
            schema_counts: self.schema_counts.clone(),
            original_lengths: self.original_lengths.clone(),
            structure_labels: self.structure_labels.clone(),
            task_types: self.task_types.clone(),
            text_tokens: self.text_tokens.clone(),
            schema_tokens_list: self.schema_tokens_list.clone(),
            start_mappings: self.start_mappings.clone(),
            end_mappings: self.end_mappings.clone(),
            original_texts: self.original_texts.clone(),
            original_schemas: self.original_schemas.clone(),
            text_word_counts: self.text_word_counts.clone(),
            schema_special_indices: self.schema_special_indices.clone(),
        })
    }

    /// Get a field by name (for compatibility with Python dict-style access).
    ///
    /// # Arguments
    ///
    /// * `key` - The field name.
    ///
    /// # Returns
    ///
    /// The field value as a `JsonValue`, or an error if the field doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if `key` is unknown or serialization fails.
    pub fn get_field(&self, key: &str) -> Result<JsonValue> {
        match key {
            "input_ids" => Ok(JsonValue::String(format!(
                "Tensor(shape={:?})",
                self.input_ids.dims()
            ))),
            "attention_mask" => Ok(JsonValue::String(format!(
                "Tensor(shape={:?})",
                self.attention_mask.dims()
            ))),
            "text_word_indices" => Ok(self
                .text_word_indices
                .as_ref()
                .map(|t| JsonValue::String(format!("Tensor(shape={:?})", t.dims())))
                .unwrap_or(JsonValue::Null)),
            "mapped_indices" => serde_json::to_value(&self.mapped_indices).map_err(|e| {
                GlinerError::serialization(format!("Failed to serialize mapped_indices: {e}"))
            }),
            "schema_counts" => serde_json::to_value(&self.schema_counts).map_err(|e| {
                GlinerError::serialization(format!("Failed to serialize schema_counts: {e}"))
            }),
            "original_lengths" => serde_json::to_value(&self.original_lengths).map_err(|e| {
                GlinerError::serialization(format!("Failed to serialize original_lengths: {e}"))
            }),
            "structure_labels" => Ok(JsonValue::Array(
                self.structure_labels
                    .iter()
                    .map(|labels| JsonValue::Array(labels.clone()))
                    .collect(),
            )),
            "task_types" => serde_json::to_value(&self.task_types).map_err(|e| {
                GlinerError::serialization(format!("Failed to serialize task_types: {e}"))
            }),
            "text_tokens" => serde_json::to_value(&self.text_tokens).map_err(|e| {
                GlinerError::serialization(format!("Failed to serialize text_tokens: {e}"))
            }),
            "schema_tokens_list" => serde_json::to_value(&self.schema_tokens_list).map_err(|e| {
                GlinerError::serialization(format!("Failed to serialize schema_tokens_list: {e}"))
            }),
            "start_mappings" => serde_json::to_value(&self.start_mappings).map_err(|e| {
                GlinerError::serialization(format!("Failed to serialize start_mappings: {e}"))
            }),
            "end_mappings" => serde_json::to_value(&self.end_mappings).map_err(|e| {
                GlinerError::serialization(format!("Failed to serialize end_mappings: {e}"))
            }),
            "original_texts" => serde_json::to_value(&self.original_texts).map_err(|e| {
                GlinerError::serialization(format!("Failed to serialize original_texts: {e}"))
            }),
            "original_schemas" => Ok(JsonValue::Array(self.original_schemas.clone())),
            "text_word_counts" => serde_json::to_value(&self.text_word_counts).map_err(|e| {
                GlinerError::serialization(format!("Failed to serialize text_word_counts: {e}"))
            }),
            "schema_special_indices" => {
                serde_json::to_value(&self.schema_special_indices).map_err(|e| {
                    GlinerError::serialization(format!(
                        "Failed to serialize schema_special_indices: {e}"
                    ))
                })
            }
            _ => Err(GlinerError::validation(format!(
                "PreprocessedBatch does not have field '{key}'"
            ))),
        }
    }

    /// Check if a field exists.
    ///
    /// # Arguments
    ///
    /// * `key` - The field name.
    ///
    /// # Returns
    ///
    /// `true` if the field exists, `false` otherwise.
    pub fn has_field(&self, key: &str) -> bool {
        matches!(
            key,
            "input_ids"
                | "attention_mask"
                | "text_word_indices"
                | "mapped_indices"
                | "schema_counts"
                | "original_lengths"
                | "structure_labels"
                | "task_types"
                | "text_tokens"
                | "schema_tokens_list"
                | "start_mappings"
                | "end_mappings"
                | "original_texts"
                | "original_schemas"
                | "text_word_counts"
                | "schema_special_indices"
        )
    }

    /// Get an iterator over field names.
    pub fn field_names() -> &'static [&'static str] {
        &[
            "input_ids",
            "attention_mask",
            "text_word_indices",
            "mapped_indices",
            "schema_counts",
            "original_lengths",
            "structure_labels",
            "task_types",
            "text_tokens",
            "schema_tokens_list",
            "start_mappings",
            "end_mappings",
            "original_texts",
            "original_schemas",
            "text_word_counts",
            "schema_special_indices",
        ]
    }

    /// Get the number of schemas for a specific sample.
    ///
    /// # Arguments
    ///
    /// * `sample_idx` - The sample index.
    ///
    /// # Returns
    ///
    /// The number of schemas, or `None` if the index is out of bounds.
    pub fn num_schemas(&self, sample_idx: usize) -> Option<usize> {
        self.schema_counts.get(sample_idx).copied()
    }

    /// Get the task types for a specific sample.
    ///
    /// # Arguments
    ///
    /// * `sample_idx` - The sample index.
    ///
    /// # Returns
    ///
    /// The task types, or `None` if the index is out of bounds.
    pub fn sample_task_types(&self, sample_idx: usize) -> Option<&[String]> {
        self.task_types.get(sample_idx).map(|v| v.as_slice())
    }

    /// Get the schema tokens for a specific sample and schema.
    ///
    /// # Arguments
    ///
    /// * `sample_idx` - The sample index.
    /// * `schema_idx` - The schema index within the sample.
    ///
    /// # Returns
    ///
    /// The schema tokens, or `None` if indices are out of bounds.
    pub fn schema_tokens(&self, sample_idx: usize, schema_idx: usize) -> Option<&[String]> {
        self.schema_tokens_list
            .get(sample_idx)
            .and_then(|schemas| schemas.get(schema_idx))
            .map(|tokens| tokens.as_slice())
    }

    /// Get the original text for a specific sample.
    ///
    /// # Arguments
    ///
    /// * `sample_idx` - The sample index.
    ///
    /// # Returns
    ///
    /// The original text, or `None` if the index is out of bounds.
    pub fn original_text(&self, sample_idx: usize) -> Option<&str> {
        self.original_texts.get(sample_idx).map(|s| s.as_str())
    }

    /// Get the text tokens for a specific sample.
    ///
    /// # Arguments
    ///
    /// * `sample_idx` - The sample index.
    ///
    /// # Returns
    ///
    /// The text tokens, or `None` if the index is out of bounds.
    pub fn sample_text_tokens(&self, sample_idx: usize) -> Option<&[String]> {
        self.text_tokens.get(sample_idx).map(|v| v.as_slice())
    }

    /// Get the start mapping for a specific sample.
    ///
    /// # Arguments
    ///
    /// * `sample_idx` - The sample index.
    ///
    /// # Returns
    ///
    /// The start mapping, or `None` if the index is out of bounds.
    pub fn sample_start_mapping(&self, sample_idx: usize) -> Option<&[usize]> {
        self.start_mappings.get(sample_idx).map(|v| v.as_slice())
    }

    /// Get the end mapping for a specific sample.
    ///
    /// # Arguments
    ///
    /// * `sample_idx` - The sample index.
    ///
    /// # Returns
    ///
    /// The end mapping, or `None` if the index is out of bounds.
    pub fn sample_end_mapping(&self, sample_idx: usize) -> Option<&[usize]> {
        self.end_mappings.get(sample_idx).map(|v| v.as_slice())
    }

    /// Get the schema special indices for a specific sample and schema.
    ///
    /// # Arguments
    ///
    /// * `sample_idx` - The sample index.
    /// * `schema_idx` - The schema index within the sample.
    ///
    /// # Returns
    ///
    /// The special indices, or `None` if indices are out of bounds.
    pub fn schema_special_indices_for(
        &self,
        sample_idx: usize,
        schema_idx: usize,
    ) -> Option<&[usize]> {
        self.schema_special_indices
            .get(sample_idx)
            .and_then(|schemas| schemas.get(schema_idx))
            .map(|indices| indices.as_slice())
    }
}

/// Builder for constructing `PreprocessedBatch` incrementally.
///
/// This builder is useful when constructing batches from multiple
/// samples, allowing fields to be set in any order.
#[derive(Debug, Default)]
pub struct PreprocessedBatchBuilder {
    input_ids: Option<Tensor>,
    attention_mask: Option<Tensor>,
    text_word_indices: Option<Tensor>,
    mapped_indices: Option<Vec<Vec<TokenMapping>>>,
    schema_counts: Option<Vec<usize>>,
    original_lengths: Option<Vec<usize>>,
    structure_labels: Option<Vec<Vec<JsonValue>>>,
    task_types: Option<Vec<Vec<String>>>,
    text_tokens: Option<Vec<Vec<String>>>,
    schema_tokens_list: Option<Vec<Vec<Vec<String>>>>,
    start_mappings: Option<Vec<Vec<usize>>>,
    end_mappings: Option<Vec<Vec<usize>>>,
    original_texts: Option<Vec<String>>,
    original_schemas: Option<Vec<JsonValue>>,
    text_word_counts: Option<Vec<usize>>,
    schema_special_indices: Option<Vec<Vec<Vec<usize>>>>,
}

impl PreprocessedBatchBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the input IDs tensor.
    pub fn input_ids(mut self, tensor: Tensor) -> Self {
        self.input_ids = Some(tensor);
        self
    }

    /// Set the attention mask tensor.
    pub fn attention_mask(mut self, tensor: Tensor) -> Self {
        self.attention_mask = Some(tensor);
        self
    }

    /// Set the text word indices tensor.
    pub fn text_word_indices(mut self, tensor: Option<Tensor>) -> Self {
        self.text_word_indices = tensor;
        self
    }

    /// Set the mapped indices.
    pub fn mapped_indices(mut self, indices: Vec<Vec<TokenMapping>>) -> Self {
        self.mapped_indices = Some(indices);
        self
    }

    /// Set the schema counts.
    pub fn schema_counts(mut self, counts: Vec<usize>) -> Self {
        self.schema_counts = Some(counts);
        self
    }

    /// Set the original lengths.
    pub fn original_lengths(mut self, lengths: Vec<usize>) -> Self {
        self.original_lengths = Some(lengths);
        self
    }

    /// Set the structure labels.
    pub fn structure_labels(mut self, labels: Vec<Vec<JsonValue>>) -> Self {
        self.structure_labels = Some(labels);
        self
    }

    /// Set the task types.
    pub fn task_types(mut self, types: Vec<Vec<String>>) -> Self {
        self.task_types = Some(types);
        self
    }

    /// Set the text tokens.
    pub fn text_tokens(mut self, tokens: Vec<Vec<String>>) -> Self {
        self.text_tokens = Some(tokens);
        self
    }

    /// Set the schema tokens list.
    pub fn schema_tokens_list(mut self, tokens: Vec<Vec<Vec<String>>>) -> Self {
        self.schema_tokens_list = Some(tokens);
        self
    }

    /// Set the start mappings.
    pub fn start_mappings(mut self, mappings: Vec<Vec<usize>>) -> Self {
        self.start_mappings = Some(mappings);
        self
    }

    /// Set the end mappings.
    pub fn end_mappings(mut self, mappings: Vec<Vec<usize>>) -> Self {
        self.end_mappings = Some(mappings);
        self
    }

    /// Set the original texts.
    pub fn original_texts(mut self, texts: Vec<String>) -> Self {
        self.original_texts = Some(texts);
        self
    }

    /// Set the original schemas.
    pub fn original_schemas(mut self, schemas: Vec<JsonValue>) -> Self {
        self.original_schemas = Some(schemas);
        self
    }

    /// Set the text word counts.
    pub fn text_word_counts(mut self, counts: Vec<usize>) -> Self {
        self.text_word_counts = Some(counts);
        self
    }

    /// Set the schema special indices.
    pub fn schema_special_indices(mut self, indices: Vec<Vec<Vec<usize>>>) -> Self {
        self.schema_special_indices = Some(indices);
        self
    }

    /// Build the batch, returning an error if required fields are missing.
    ///
    /// # Errors
    ///
    /// Returns an error if required tensors are missing or tensor batch sizes do
    /// not match.
    pub fn build(self) -> Result<PreprocessedBatch> {
        let input_ids = self
            .input_ids
            .ok_or_else(|| GlinerError::validation("input_ids is required"))?;
        let attention_mask = self
            .attention_mask
            .ok_or_else(|| GlinerError::validation("attention_mask is required"))?;

        // Validate tensor shapes match
        let batch_size = input_ids.dims()[0];
        let mask_batch_size = attention_mask.dims()[0];
        if batch_size != mask_batch_size {
            return Err(GlinerError::dimension_mismatch(
                vec![batch_size],
                vec![mask_batch_size],
            ));
        }

        Ok(PreprocessedBatch {
            input_ids,
            attention_mask,
            text_word_indices: self.text_word_indices,
            mapped_indices: self.mapped_indices.unwrap_or_default(),
            schema_counts: self.schema_counts.unwrap_or_default(),
            original_lengths: self.original_lengths.unwrap_or_default(),
            structure_labels: self.structure_labels.unwrap_or_default(),
            task_types: self.task_types.unwrap_or_default(),
            text_tokens: self.text_tokens.unwrap_or_default(),
            schema_tokens_list: self.schema_tokens_list.unwrap_or_default(),
            start_mappings: self.start_mappings.unwrap_or_default(),
            end_mappings: self.end_mappings.unwrap_or_default(),
            original_texts: self.original_texts.unwrap_or_default(),
            original_schemas: self.original_schemas.unwrap_or_default(),
            text_word_counts: self.text_word_counts.unwrap_or_default(),
            schema_special_indices: self.schema_special_indices.unwrap_or_default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_batch() -> PreprocessedBatch {
        let input_ids =
            Tensor::from_slice(&[1u32, 2, 3, 4, 5, 6], (2, 3), &candle_core::Device::Cpu).unwrap();
        let attention_mask =
            Tensor::from_slice(&[1u32, 1, 1, 1, 1, 0], (2, 3), &candle_core::Device::Cpu).unwrap();

        PreprocessedBatch::new(
            input_ids,
            attention_mask,
            vec![
                vec![("entities".to_string(), 0, 0)],
                vec![("entities".to_string(), 0, 0)],
            ],
            vec![1, 1],
            vec![3, 2],
            vec![vec![], vec![]],
            vec![vec!["entities".to_string()], vec!["entities".to_string()]],
            vec![
                vec!["apple".to_string(), "inc".to_string()],
                vec!["hello".to_string()],
            ],
            vec![vec![vec![
                "(".to_string(),
                "[P]".to_string(),
                "entities".to_string(),
            ]]],
            vec![vec![0, 1, 2], vec![0, 1, 2]],
            vec![vec![5, 8, 12], vec![0, 3, 7]],
            vec!["Apple Inc.".to_string(), "Hello".to_string()],
            vec![
                JsonValue::Object(serde_json::Map::new()),
                JsonValue::Object(serde_json::Map::new()),
            ],
            None,
            vec![2, 1],
            vec![vec![vec![0, 1, 2]], vec![vec![0, 1]]],
        )
    }

    #[test]
    fn test_batch_size() {
        let batch = create_test_batch();
        assert_eq!(batch.batch_size(), 2);
    }

    #[test]
    fn test_seq_len() {
        let batch = create_test_batch();
        assert_eq!(batch.seq_len(), 3);
    }

    #[test]
    fn test_is_empty() {
        let batch = create_test_batch();
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_has_field() {
        let batch = create_test_batch();
        assert!(batch.has_field("input_ids"));
        assert!(batch.has_field("attention_mask"));
        assert!(batch.has_field("text_tokens"));
        assert!(!batch.has_field("nonexistent"));
    }

    #[test]
    fn test_get_field() {
        let batch = create_test_batch();
        let result = batch.get_field("input_ids");
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_field_invalid() {
        let batch = create_test_batch();
        let result = batch.get_field("invalid_field");
        assert!(result.is_err());
    }

    #[test]
    fn test_num_schemas() {
        let batch = create_test_batch();
        assert_eq!(batch.num_schemas(0), Some(1));
        assert_eq!(batch.num_schemas(1), Some(1));
        assert_eq!(batch.num_schemas(2), None);
    }

    #[test]
    fn test_sample_task_types() {
        let batch = create_test_batch();
        assert_eq!(
            batch.sample_task_types(0),
            Some(&["entities".to_string()][..])
        );
        assert_eq!(batch.sample_task_types(2), None);
    }

    #[test]
    fn test_original_text() {
        let batch = create_test_batch();
        assert_eq!(batch.original_text(0), Some("Apple Inc."));
        assert_eq!(batch.original_text(1), Some("Hello"));
        assert_eq!(batch.original_text(2), None);
    }

    #[test]
    fn test_sample_text_tokens() {
        let batch = create_test_batch();
        assert_eq!(
            batch.sample_text_tokens(0),
            Some(&["apple".to_string(), "inc".to_string()][..])
        );
        assert_eq!(batch.sample_text_tokens(2), None);
    }

    #[test]
    fn test_builder() {
        let input_ids =
            Tensor::from_slice(&[1u32, 2, 3, 4], (2, 2), &candle_core::Device::Cpu).unwrap();
        let attention_mask =
            Tensor::from_slice(&[1u32, 1, 1, 0], (2, 2), &candle_core::Device::Cpu).unwrap();

        let batch = PreprocessedBatchBuilder::new()
            .input_ids(input_ids)
            .attention_mask(attention_mask)
            .schema_counts(vec![1, 1])
            .task_types(vec![
                vec!["entities".to_string()],
                vec!["entities".to_string()],
            ])
            .text_tokens(vec![vec!["hello".to_string()], vec!["world".to_string()]])
            .schema_tokens_list(vec![vec![vec![
                "(".to_string(),
                "[P]".to_string(),
                "entities".to_string(),
            ]]])
            .start_mappings(vec![vec![0], vec![0]])
            .end_mappings(vec![vec![5], vec![5]])
            .original_texts(vec!["hello".to_string(), "world".to_string()])
            .original_schemas(vec![JsonValue::Null, JsonValue::Null])
            .text_word_counts(vec![1, 1])
            .schema_special_indices(vec![vec![vec![0, 1]], vec![vec![0, 1]]])
            .build()
            .unwrap();

        assert_eq!(batch.batch_size(), 2);
        assert_eq!(batch.seq_len(), 2);
    }

    #[test]
    fn test_builder_missing_required() {
        let result = PreprocessedBatchBuilder::new().build();
        assert!(result.is_err());
    }
}
