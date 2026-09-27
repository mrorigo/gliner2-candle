// Rust guideline compliant 2026-04-03
//! ExtractorCollator for GLiNER2 batch preprocessing.
//!
//! This module provides the `ExtractorCollator` struct that converts raw
//! text/schema pairs into GPU-ready `PreprocessedBatch` objects. It handles
//! tokenization, schema encoding, padding, and index mapping for efficient
//! batch inference.
//!
//! # Example
//!
//! ```ignore
//! use gliner2_candle::batch::ExtractorCollator;
//! use gliner2_candle::schema::Schema;
//! use gliner2_candle::tokenizer::WhitespaceTokenizer;
//!
//! let tokenizer = WhitespaceTokenizer::new();
//! let collator = ExtractorCollator::new(&tokenizer, false);
//!
//! let samples = vec![
//!     ("Apple CEO Tim Cook", schema_dict),
//!     ("Google in Mountain View", schema_dict),
//! ];
//!
//! let batch = collator.collate(&samples)?;
//! ```

use candle_core::{Device, Tensor};
use serde_json::Value as JsonValue;
use tokenizers::Tokenizer as HfTokenizer;

use crate::batch::preprocessed::{PreprocessedBatch, TokenMapping};
use crate::error::{GlinerError, Result};
use crate::tokenizer::WhitespaceTokenizer;

/// Special tokens used in GLiNER2 schema encoding.
pub mod special_tokens {
    /// Opening parenthesis.
    pub const OPEN_PAREN: &str = "(";
    /// Closing parenthesis.
    pub const CLOSE_PAREN: &str = ")";
    /// Prompt token.
    pub const P_TOKEN: &str = "[P]";
    /// Entity token.
    pub const E_TOKEN: &str = "[E]";
    /// Classification token.
    ///
    /// Distinct from `C_TOKEN` (structure/JSON fields) in the tokenizer: `[L]` is
    /// id 128007, `[C]` is 128004. Both GLiNER2 reference revisions emit `[L]` for
    /// classification labels, so these must not be aliased.
    pub const L_TOKEN: &str = "[L]";
    /// Relation token.
    pub const R_TOKEN: &str = "[R]";
    /// Structure/JSON token.
    pub const C_TOKEN: &str = "[C]";
    /// Description token.
    pub const DESC_TOKEN: &str = "[DESCRIPTION]";
    /// Example token.
    pub const EXAMPLE_TOKEN: &str = "[EXAMPLE]";
    /// Output token.
    pub const OUTPUT_TOKEN: &str = "[OUTPUT]";
}

/// A single sample for collation: (text, schema).
pub type CollateSample = (String, JsonValue);

/// ExtractorCollator for converting text/schema pairs into batches.
///
/// This collator handles all preprocessing for GLiNER2 inference,
/// including:
/// - Whitespace tokenization of input text
/// - Schema encoding into token sequences
/// - Padding and batching
/// - Index mapping for embedding extraction
///
/// # Arguments
///
/// * `tokenizer` - The whitespace tokenizer to use.
/// * `is_training` - Whether in training mode (affects schema processing).
/// * `max_len` - Optional maximum token length for truncation.
#[derive(Debug, Clone)]
pub struct ExtractorCollator {
    /// The whitespace tokenizer for span boundary detection.
    tokenizer: WhitespaceTokenizer,
    /// HuggingFace tokenizer for token ID generation (optional).
    hf_tokenizer: Option<std::sync::Arc<HfTokenizer>>,
    /// Whether in training mode.
    _is_training: bool,
    /// Maximum token length (None = no limit).
    max_len: Option<usize>,
}

impl ExtractorCollator {
    /// Create a new collator.
    ///
    /// # Arguments
    ///
    /// * `tokenizer` - The whitespace tokenizer.
    /// * `is_training` - Whether in training mode.
    pub fn new(tokenizer: WhitespaceTokenizer, is_training: bool) -> Self {
        Self {
            tokenizer,
            hf_tokenizer: None,
            _is_training: is_training,
            max_len: None,
        }
    }

    /// Create a new collator with max_len truncation.
    ///
    /// # Arguments
    ///
    /// * `tokenizer` - The whitespace tokenizer.
    /// * `is_training` - Whether in training mode.
    /// * `max_len` - Maximum token length.
    pub fn with_max_len(
        tokenizer: WhitespaceTokenizer,
        is_training: bool,
        max_len: Option<usize>,
    ) -> Self {
        Self {
            tokenizer,
            hf_tokenizer: None,
            _is_training: is_training,
            max_len,
        }
    }

    /// Create a new collator with a HuggingFace tokenizer.
    ///
    /// # Arguments
    ///
    /// * `tokenizer` - The whitespace tokenizer for span boundaries.
    /// * `hf_tokenizer` - HuggingFace tokenizer for token ID generation.
    /// * `is_training` - Whether in training mode.
    /// * `max_len` - Optional maximum token length.
    pub fn with_hf_tokenizer(
        tokenizer: WhitespaceTokenizer,
        hf_tokenizer: HfTokenizer,
        is_training: bool,
        max_len: Option<usize>,
    ) -> Self {
        Self {
            tokenizer,
            hf_tokenizer: Some(std::sync::Arc::new(hf_tokenizer)),
            _is_training: is_training,
            max_len,
        }
    }

    /// Clone this collator but override `max_len` for a single call path.
    pub fn with_runtime_max_len(&self, max_len: Option<usize>) -> Self {
        let mut cloned = self.clone();
        cloned.max_len = max_len;
        cloned
    }

    /// Collate a batch of samples into a `PreprocessedBatch`.
    ///
    /// # Arguments
    ///
    /// * `samples` - A slice of (text, schema) pairs.
    ///
    /// # Returns
    ///
    /// A `PreprocessedBatch` ready for model inference.
    pub fn collate(&self, samples: &[CollateSample]) -> Result<PreprocessedBatch> {
        if samples.is_empty() {
            return Err(GlinerError::batch_processing("Cannot collate empty batch"));
        }

        // Process each sample
        let mut all_input_ids: Vec<Vec<i64>> = Vec::with_capacity(samples.len());
        let mut all_mapped_indices: Vec<Vec<TokenMapping>> = Vec::with_capacity(samples.len());
        let mut all_schema_counts: Vec<usize> = Vec::with_capacity(samples.len());
        let mut all_original_lengths: Vec<usize> = Vec::with_capacity(samples.len());
        let mut all_structure_labels: Vec<Vec<JsonValue>> = Vec::with_capacity(samples.len());
        let mut all_task_types: Vec<Vec<String>> = Vec::with_capacity(samples.len());
        let mut all_text_tokens: Vec<Vec<String>> = Vec::with_capacity(samples.len());
        let mut all_schema_tokens_list: Vec<Vec<Vec<String>>> = Vec::with_capacity(samples.len());
        let mut all_start_mappings: Vec<Vec<usize>> = Vec::with_capacity(samples.len());
        let mut all_end_mappings: Vec<Vec<usize>> = Vec::with_capacity(samples.len());
        let mut all_original_texts: Vec<String> = Vec::with_capacity(samples.len());
        let mut all_original_schemas: Vec<JsonValue> = Vec::with_capacity(samples.len());
        let mut all_text_word_indices: Vec<Vec<i64>> = Vec::with_capacity(samples.len());
        let mut all_text_word_counts: Vec<usize> = Vec::with_capacity(samples.len());
        let mut all_schema_special_indices: Vec<Vec<Vec<usize>>> =
            Vec::with_capacity(samples.len());

        for (text, schema) in samples {
            let processed = self.process_sample(text, schema)?;

            all_input_ids.push(processed.input_ids);
            all_mapped_indices.push(processed.mapped_indices);
            all_schema_counts.push(processed.schema_count);
            all_original_lengths.push(processed.original_length);
            all_structure_labels.push(processed.structure_labels);
            all_task_types.push(processed.task_types);
            all_text_tokens.push(processed.text_tokens);
            all_schema_tokens_list.push(processed.schema_tokens_list);
            all_start_mappings.push(processed.start_mapping);
            all_end_mappings.push(processed.end_mapping);
            all_original_texts.push(processed.original_text);
            all_original_schemas.push(processed.original_schema.clone());
            all_text_word_indices.push(processed.text_word_indices);
            all_text_word_counts.push(processed.text_word_count);
            all_schema_special_indices.push(processed.schema_special_indices);
        }

        // Find max sequence length for padding
        let max_seq_len = all_input_ids.iter().map(|ids| ids.len()).max().unwrap_or(0);

        // Pad input IDs and attention mask
        let mut padded_input_ids: Vec<i64> = Vec::with_capacity(samples.len() * max_seq_len);
        let mut attention_mask: Vec<i64> = Vec::with_capacity(samples.len() * max_seq_len);

        for ids in &all_input_ids {
            let len = ids.len();
            padded_input_ids.extend_from_slice(ids);
            // Pad with 0 (assuming 0 is pad token ID)
            padded_input_ids.extend(std::iter::repeat_n(0, max_seq_len - len));

            // Attention mask: 1 for real tokens, 0 for padding
            attention_mask.extend(std::iter::repeat_n(1, len));
            attention_mask.extend(std::iter::repeat_n(0, max_seq_len - len));
        }

        // Create tensors
        let input_ids_tensor = Tensor::from_slice(
            &padded_input_ids,
            (samples.len(), max_seq_len),
            &Device::Cpu,
        )
        .map_err(|e| {
            GlinerError::model_loading(format!("Failed to create input_ids tensor: {e}"))
        })?;

        let attention_mask_tensor =
            Tensor::from_slice(&attention_mask, (samples.len(), max_seq_len), &Device::Cpu)
                .map_err(|e| {
                    GlinerError::model_loading(format!(
                        "Failed to create attention_mask tensor: {e}"
                    ))
                })?;

        // Create text word indices tensor if all samples have the same max words
        let max_words = all_text_word_indices
            .iter()
            .map(|idx| idx.len())
            .max()
            .unwrap_or(0);
        let text_word_indices_tensor = if max_words > 0 {
            let mut flat_indices: Vec<i64> = Vec::with_capacity(samples.len() * max_words);
            for indices in &all_text_word_indices {
                flat_indices.extend_from_slice(indices);
                flat_indices.extend(std::iter::repeat_n(-1, max_words - indices.len()));
            }
            Some(
                Tensor::from_slice(&flat_indices, (samples.len(), max_words), &Device::Cpu)
                    .map_err(|e| {
                        GlinerError::model_loading(format!(
                            "Failed to create text_word_indices tensor: {e}"
                        ))
                    })?,
            )
        } else {
            None
        };

        // Build the batch
        Ok(PreprocessedBatch::new(
            input_ids_tensor,
            attention_mask_tensor,
            all_mapped_indices,
            all_schema_counts,
            all_original_lengths,
            all_structure_labels,
            all_task_types,
            all_text_tokens,
            all_schema_tokens_list,
            all_start_mappings,
            all_end_mappings,
            all_original_texts,
            all_original_schemas,
            text_word_indices_tensor,
            all_text_word_counts,
            all_schema_special_indices,
        ))
    }

    /// Process a single sample into tokenized form.
    fn process_sample(&self, text: &str, schema: &JsonValue) -> Result<ProcessedSample> {
        // Tokenize text with whitespace tokenizer for span boundaries
        let tokens = self.tokenizer.tokenize(text);
        let text_tokens: Vec<String> = tokens.iter().map(|t| t.text.clone()).collect();
        let (start_mapping, end_mapping) = WhitespaceTokenizer::build_mappings(&tokens);

        // Apply max_len truncation
        let truncated_tokens = if let Some(max_len) = self.max_len {
            if text_tokens.len() > max_len {
                &text_tokens[..max_len]
            } else {
                &text_tokens[..]
            }
        } else {
            &text_tokens[..]
        };

        // Encode schema into token sequences
        let schema_result = self.encode_schema(schema)?;

        // Build input sequence using HuggingFace tokenizer when available
        let mut input_ids: Vec<i64> = Vec::new();
        let mut mapped_indices: Vec<TokenMapping> = Vec::new();
        let mut text_word_indices: Vec<i64> = Vec::new();
        let mut schema_special_indices: Vec<Vec<usize>> = Vec::new();

        if let Some(hf_tok) = &self.hf_tokenizer {
            // Use HF tokenizer for proper subword tokenization
            // Follow Python approach: build combined sequence with SEP tokens,
            // tokenize each token individually, and track subword positions

            // Build combined token sequence matching Python implementation:
            // schema_tokens + [SEP_STRUCT] + schema_tokens + [SEP_STRUCT] + ... + [SEP_TEXT] + text_tokens
            let mut combined_tokens: Vec<String> = Vec::new();
            let num_schemas = schema_result.schema_tokens_list.len();

            for schema_tokens in &schema_result.schema_tokens_list {
                for token in schema_tokens {
                    combined_tokens.push(token.clone());
                }
                // Add SEP_STRUCT between schemas
                combined_tokens.push("[SEP_STRUCT]".to_string());
            }
            // Remove last SEP_STRUCT if we added any
            if combined_tokens.last().is_some_and(|t| t == "[SEP_STRUCT]") {
                combined_tokens.pop();
            }
            // Add SEP_TEXT before text tokens
            combined_tokens.push("[SEP_TEXT]".to_string());

            // Add text tokens
            for token in truncated_tokens {
                combined_tokens.push(token.clone());
            }

            // Initialize schema_special_indices for each schema
            for _ in 0..num_schemas {
                schema_special_indices.push(Vec::new());
            }

            // Tokenize each token into subwords and track positions
            // Follow Python logic: track seg_type ("schema", "sep", "text")
            let mut last_text_orig: Option<usize> = None;
            let mut current_schema = 0;
            let mut found_sep = false;
            let mut text_token_counter = 0;

            for (orig_idx, token) in combined_tokens.iter().enumerate() {
                // Determine segment type (matching Python implementation)
                let (seg_type, schema_idx) = if token == "[SEP_TEXT]" {
                    ("sep", num_schemas)
                } else if !found_sep {
                    if token == "[SEP_STRUCT]" {
                        current_schema += 1;
                    }
                    ("schema", current_schema)
                } else {
                    ("text", num_schemas)
                };

                if seg_type == "sep" {
                    found_sep = true;
                }

                // Tokenize this token into subwords and use the produced IDs directly.
                // This preserves tokenizer context (e.g. continuation pieces like "er", "tino").
                // Text words are lower-cased first, matching Python's word splitter
                // (`word_splitter(text, lower=True)`); character offsets still index
                // the original string because folding happens per-token after the split.
                let encode_str: String = if seg_type == "text" {
                    token.to_lowercase()
                } else {
                    token.clone()
                };
                let (sub_tokens, sub_token_ids): (Vec<String>, Vec<u32>) = hf_tok
                    .encode(encode_str.as_str(), false)
                    .map(|enc| (enc.get_tokens().to_vec(), enc.get_ids().to_vec()))
                    .unwrap_or_else(|_| (vec![token.clone()], vec![0u32]));

                let subword_pos_start = input_ids.len();

                for &token_id in &sub_token_ids {
                    input_ids.push(token_id as i64);
                }

                // Track routing indices (matching Python logic)
                if seg_type == "text" && !sub_tokens.is_empty() {
                    // New text word - record position of first subword
                    if last_text_orig != Some(orig_idx) {
                        text_word_indices.push(subword_pos_start as i64);
                        last_text_orig = Some(orig_idx);
                        // Add mapping for this text token
                        for _ in 0..sub_tokens.len() {
                            mapped_indices.push(("text".to_string(), 0, text_token_counter));
                        }
                        text_token_counter += 1;
                    }
                } else if seg_type == "schema" && !sub_tokens.is_empty() {
                    // Schema token - check if it's a special token
                    // Special tokens are those starting with '[' and ending with ']'
                    // but NOT SEP_STRUCT tokens
                    if token.starts_with('[') && token.ends_with(']') && token != "[SEP_STRUCT]" {
                        // Record the first subword position for this special token
                        if schema_idx < num_schemas {
                            schema_special_indices[schema_idx].push(subword_pos_start);
                        }
                    }
                }
            }
        } else {
            // Fallback: use placeholder token IDs
            for schema_tokens in &schema_result.schema_tokens_list {
                let schema_start_pos = input_ids.len();
                for token in schema_tokens {
                    input_ids.push(self.token_to_id(token));
                }
                let schema_end_pos = input_ids.len();
                schema_special_indices.push((schema_start_pos..schema_end_pos).collect());
            }

            // Add separator
            input_ids.push(self.token_to_id("[SEP]"));

            // Add text tokens
            for (token_idx, token) in truncated_tokens.iter().enumerate() {
                input_ids.push(self.token_to_id(token));
                mapped_indices.push(("text".to_string(), 0, token_idx));
                text_word_indices.push(input_ids.len() as i64 - 1);
            }

            // Add final separator
            input_ids.push(self.token_to_id("[SEP]"));
        }

        Ok(ProcessedSample {
            input_ids,
            mapped_indices,
            schema_count: schema_result.schema_tokens_list.len(),
            original_length: truncated_tokens.len() + 2, // +2 for separators
            structure_labels: schema_result.structure_labels,
            task_types: schema_result.task_types,
            text_tokens: truncated_tokens.to_vec(),
            schema_tokens_list: schema_result.schema_tokens_list,
            start_mapping: start_mapping[..truncated_tokens.len()].to_vec(),
            end_mapping: end_mapping[..truncated_tokens.len()].to_vec(),
            original_text: text.to_string(),
            original_schema: schema.clone(),
            text_word_indices,
            text_word_count: truncated_tokens.len(),
            schema_special_indices,
        })
    }

    /// Encode a schema into token sequences.
    fn encode_schema(&self, schema: &JsonValue) -> Result<SchemaEncodingResult> {
        let mut schema_tokens_list: Vec<Vec<String>> = Vec::new();
        let mut task_types: Vec<String> = Vec::new();
        let mut structure_labels: Vec<JsonValue> = Vec::new();

        // Parse entities - handle both array format (preserves order) and object format
        if let Some(entities) = schema.get("entities") {
            let entity_names: Vec<String> = if let Some(arr) = entities.as_array() {
                // Array format: ["person", "organization", "location"]
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            } else if let Some(obj) = entities.as_object() {
                // Object format: {"person": "", "organization": "", "location": ""}
                obj.keys().cloned().collect()
            } else {
                Vec::new()
            };

            if !entity_names.is_empty() {
                let tokens = self.build_entity_tokens(&entity_names, entities);
                schema_tokens_list.push(tokens);
                task_types.push("entities".to_string());
                structure_labels.push(JsonValue::Array(Vec::new()));
            }
        }

        // Parse classifications
        if let Some(classifications) = schema.get("classifications")
            && let Some(cls_array) = classifications.as_array()
        {
            for cls in cls_array {
                if let Some(cls_obj) = cls.as_object()
                    && let Some(task) = cls_obj.get("task").and_then(|v| v.as_str())
                {
                    let labels = cls_obj
                        .get("labels")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();

                    if !labels.is_empty() {
                        let tokens = self.build_classification_tokens(task, &labels, cls_obj);
                        schema_tokens_list.push(tokens);
                        task_types.push("classifications".to_string());
                        structure_labels.push(JsonValue::Array(Vec::new()));
                    }
                }
            }
        }

        // Parse structures
        if let Some(structures) = schema.get("json_structures")
            && let Some(struct_array) = structures.as_array()
        {
            for structure in struct_array {
                if let Some(struct_obj) = structure.as_object() {
                    for (parent, fields) in struct_obj {
                        if let Some(fields_obj) = fields.as_object() {
                            let field_names: Vec<String> = fields_obj.keys().cloned().collect();
                            if !field_names.is_empty() {
                                let tokens =
                                    self.build_structure_tokens(parent, &field_names, fields_obj);
                                schema_tokens_list.push(tokens);
                                task_types.push("json_structures".to_string());
                                structure_labels.push(JsonValue::Array(Vec::new()));
                            }
                        }
                    }
                }
            }
        }

        // Parse relations
        if let Some(relations) = schema.get("relations")
            && let Some(rel_array) = relations.as_array()
        {
            for relation in rel_array {
                if let Some(rel_obj) = relation.as_object() {
                    for (rel_name, fields) in rel_obj {
                        let field_names: Vec<String> = if let Some(meta_fields) = schema
                            .get("relation_metadata")
                            .and_then(|m| m.get(rel_name))
                            .and_then(|m| m.get("fields"))
                            .and_then(|f| f.as_array())
                        {
                            meta_fields
                                .iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        } else if let Some(fields_arr) = fields.as_array() {
                            fields_arr
                                .iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        } else if let Some(fields_obj) = fields.as_object() {
                            fields_obj.keys().cloned().collect()
                        } else {
                            vec!["head".to_string(), "tail".to_string()]
                        };

                        let tokens = self.build_relation_tokens(rel_name, &field_names);
                        schema_tokens_list.push(tokens);
                        task_types.push("relations".to_string());
                        structure_labels.push(JsonValue::Array(Vec::new()));
                    }
                }
            }
        }

        Ok(SchemaEncodingResult {
            schema_tokens_list,
            task_types,
            structure_labels,
        })
    }

    /// Build entity schema tokens.
    fn build_entity_tokens(
        &self,
        entity_names: &[String],
        entities_value: &JsonValue,
    ) -> Vec<String> {
        let mut tokens = vec![
            special_tokens::OPEN_PAREN.to_string(),
            special_tokens::P_TOKEN.to_string(),
            "entities".to_string(),
            special_tokens::OPEN_PAREN.to_string(),
        ];

        for name in entity_names {
            tokens.push(special_tokens::E_TOKEN.to_string());
            tokens.push(name.clone());

            // Add description if available
            if let Some(desc) = entities_value.get(name).and_then(|v| v.as_str())
                && !desc.is_empty()
            {
                tokens.push(special_tokens::DESC_TOKEN.to_string());
                tokens.push(format!("{name}: {desc}"));
            }
        }

        tokens.push(special_tokens::CLOSE_PAREN.to_string());
        tokens.push(special_tokens::CLOSE_PAREN.to_string());
        tokens
    }

    /// Build classification schema tokens.
    fn build_classification_tokens(
        &self,
        task: &str,
        labels: &[String],
        cls_obj: &serde_json::Map<String, JsonValue>,
    ) -> Vec<String> {
        // Python builds ONE prompt string at token index 2 and appends the
        // instruction, label descriptions and few-shot examples to it as plain
        // text (gliner2/processor.py::_transform_schema). Emitting them as separate
        // top-level schema tokens changes the token sequence and, because the
        // marker scan counts any `[...]` token, shifts the label embedding indices.
        let mut prompt_str = task.to_string();
        if let Some(prompt) = cls_obj.get("prompt").and_then(|v| v.as_str()) {
            prompt_str = format!("{task}: {prompt}");
        }
        if let Some(label_descs) = cls_obj
            .get("label_descriptions")
            .and_then(|v| v.as_object())
        {
            for label in labels {
                if let Some(desc) = label_descs.get(label).and_then(|v| v.as_str()) {
                    prompt_str
                        .push_str(&format!(" {} {label}: {desc}", special_tokens::DESC_TOKEN));
                }
            }
        }
        if let Some(examples) = cls_obj.get("examples").and_then(|v| v.as_array()) {
            for example in examples {
                if let Some(ex_array) = example.as_array()
                    && ex_array.len() >= 2
                    && let (Some(input), Some(output)) =
                        (ex_array[0].as_str(), ex_array[1].as_str())
                {
                    prompt_str.push_str(&format!(
                        " {} {input} {} {output}",
                        special_tokens::EXAMPLE_TOKEN,
                        special_tokens::OUTPUT_TOKEN
                    ));
                }
            }
        }

        let mut tokens = vec![
            special_tokens::OPEN_PAREN.to_string(),
            special_tokens::P_TOKEN.to_string(),
            prompt_str,
            special_tokens::OPEN_PAREN.to_string(),
        ];

        for label in labels {
            tokens.push(special_tokens::L_TOKEN.to_string());
            tokens.push(label.clone());
        }

        tokens.push(special_tokens::CLOSE_PAREN.to_string());
        tokens.push(special_tokens::CLOSE_PAREN.to_string());
        tokens
    }

    /// Build structure schema tokens.
    fn build_structure_tokens(
        &self,
        parent: &str,
        field_names: &[String],
        fields_obj: &serde_json::Map<String, JsonValue>,
    ) -> Vec<String> {
        let mut tokens = vec![
            special_tokens::OPEN_PAREN.to_string(),
            special_tokens::P_TOKEN.to_string(),
            parent.to_string(),
            special_tokens::OPEN_PAREN.to_string(),
        ];

        for field in field_names {
            tokens.push(special_tokens::C_TOKEN.to_string());
            tokens.push(field.clone());

            // Add description if available
            if let Some(field_obj) = fields_obj.get(field).and_then(|v| v.as_object())
                && let Some(desc) = field_obj.get("description").and_then(|v| v.as_str())
            {
                tokens.push(special_tokens::DESC_TOKEN.to_string());
                tokens.push(format!("{field}: {desc}"));
            }
        }

        tokens.push(special_tokens::CLOSE_PAREN.to_string());
        tokens.push(special_tokens::CLOSE_PAREN.to_string());
        tokens
    }

    /// Build relation schema tokens.
    fn build_relation_tokens(&self, rel_name: &str, field_names: &[String]) -> Vec<String> {
        let mut tokens = vec![
            special_tokens::OPEN_PAREN.to_string(),
            special_tokens::P_TOKEN.to_string(),
            rel_name.to_string(),
            special_tokens::OPEN_PAREN.to_string(),
        ];

        for field in field_names {
            tokens.push(special_tokens::R_TOKEN.to_string());
            tokens.push(field.clone());
        }

        tokens.push(special_tokens::CLOSE_PAREN.to_string());
        tokens.push(special_tokens::CLOSE_PAREN.to_string());
        tokens
    }

    /// Convert a token string to a token ID using the HuggingFace tokenizer.
    ///
    /// If a HuggingFace tokenizer is available, it encodes the token and returns
    /// the first token ID. Otherwise, falls back to a hash-based placeholder.
    fn token_to_id(&self, token: &str) -> i64 {
        if let Some(hf_tok) = &self.hf_tokenizer
            && let Ok(encoding) = hf_tok.encode(token, false)
            && let Some(&id) = encoding.get_ids().first()
        {
            return id as i64;
        }
        // Fallback: hash-based placeholder
        let hash = token
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        (hash % 30522) as i64
    }
}

/// Result of processing a single sample.
#[derive(Debug, Clone)]
struct ProcessedSample {
    input_ids: Vec<i64>,
    mapped_indices: Vec<TokenMapping>,
    schema_count: usize,
    original_length: usize,
    structure_labels: Vec<JsonValue>,
    task_types: Vec<String>,
    text_tokens: Vec<String>,
    schema_tokens_list: Vec<Vec<String>>,
    start_mapping: Vec<usize>,
    end_mapping: Vec<usize>,
    original_text: String,
    original_schema: JsonValue,
    text_word_indices: Vec<i64>,
    text_word_count: usize,
    schema_special_indices: Vec<Vec<usize>>,
}

/// Result of encoding a schema.
#[derive(Debug, Clone)]
struct SchemaEncodingResult {
    schema_tokens_list: Vec<Vec<String>>,
    task_types: Vec<String>,
    structure_labels: Vec<JsonValue>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_schema() -> JsonValue {
        serde_json::json!({
            "entities": {
                "person": "Names of people",
                "company": "Organization names"
            },
            "classifications": [
                {
                    "task": "sentiment",
                    "labels": ["positive", "negative", "neutral"],
                    "cls_threshold": 0.5
                }
            ]
        })
    }

    #[test]
    fn test_collate_single_sample() {
        let tokenizer = WhitespaceTokenizer::new();
        let collator = ExtractorCollator::new(tokenizer, false);
        let schema = create_test_schema();

        let samples = vec![(String::from("Apple CEO Tim Cook"), schema)];
        let batch = collator.collate(&samples);

        assert!(batch.is_ok());
        let batch = batch.unwrap();
        assert_eq!(batch.batch_size(), 1);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_collate_multiple_samples() {
        let tokenizer = WhitespaceTokenizer::new();
        let collator = ExtractorCollator::new(tokenizer, false);
        let schema = create_test_schema();

        let samples = vec![
            (String::from("Apple CEO Tim Cook"), schema.clone()),
            (String::from("Google in Mountain View"), schema.clone()),
            (String::from("Microsoft founded by Bill Gates"), schema),
        ];

        let batch = collator.collate(&samples);
        assert!(batch.is_ok());
        let batch = batch.unwrap();
        assert_eq!(batch.batch_size(), 3);
    }

    #[test]
    fn test_collate_empty_batch() {
        let tokenizer = WhitespaceTokenizer::new();
        let collator = ExtractorCollator::new(tokenizer, false);

        let samples: Vec<CollateSample> = vec![];
        let result = collator.collate(&samples);
        assert!(result.is_err());
    }

    #[test]
    fn test_collate_with_max_len() {
        let tokenizer = WhitespaceTokenizer::new();
        let collator = ExtractorCollator::with_max_len(tokenizer, false, Some(5));
        let schema = create_test_schema();

        let long_text =
            String::from("Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday");
        let samples = vec![(long_text, schema)];
        let batch = collator.collate(&samples);

        assert!(batch.is_ok());
        let batch = batch.unwrap();
        // Text tokens should be truncated to max_len
        assert_eq!(batch.sample_text_tokens(0).unwrap().len(), 5);
    }

    #[test]
    fn test_runtime_max_len_override() {
        let tokenizer = WhitespaceTokenizer::new();
        let base_collator = ExtractorCollator::with_max_len(tokenizer, false, None);
        let collator = base_collator.with_runtime_max_len(Some(3));
        let schema = create_test_schema();

        let long_text =
            String::from("Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday");
        let samples = vec![(long_text, schema)];
        let batch = collator.collate(&samples);

        assert!(batch.is_ok());
        let batch = batch.unwrap();
        assert_eq!(batch.sample_text_tokens(0).unwrap().len(), 3);
    }

    #[test]
    fn test_entity_token_building() {
        let tokenizer = WhitespaceTokenizer::new();
        let collator = ExtractorCollator::new(tokenizer, false);

        let entities = vec!["person".to_string(), "company".to_string()];
        let entities_obj = serde_json::json!({
            "person": "Names of people",
            "company": "Organization names"
        });

        let tokens = collator.build_entity_tokens(&entities, &entities_obj);

        assert!(tokens.contains(&"(".to_string()));
        assert!(tokens.contains(&"[P]".to_string()));
        assert!(tokens.contains(&"entities".to_string()));
        assert!(tokens.contains(&"[E]".to_string()));
        assert!(tokens.contains(&"person".to_string()));
        assert!(tokens.contains(&"company".to_string()));
        assert!(tokens.contains(&")".to_string()));
    }

    #[test]
    fn test_classification_token_building() {
        let tokenizer = WhitespaceTokenizer::new();
        let collator = ExtractorCollator::new(tokenizer, false);

        let labels = vec!["positive".to_string(), "negative".to_string()];
        let cls_obj = serde_json::json!({
            "task": "sentiment",
            "labels": ["positive", "negative"]
        })
        .as_object()
        .cloned()
        .unwrap();

        let tokens = collator.build_classification_tokens("sentiment", &labels, &cls_obj);

        // Layout: ["(", "[P]", "<folded prompt>", "(", "[L]", label, ..., ")", ")"]
        // `[L]`, not `[C]`: classification labels use a distinct tokenizer id.
        assert_eq!(
            tokens,
            vec![
                "(",
                "[P]",
                "sentiment",
                "(",
                "[L]",
                "positive",
                "[L]",
                "negative",
                ")",
                ")"
            ]
        );
    }

    #[test]
    fn test_classification_prompt_and_descriptions_fold_into_prompt_string() {
        let tokenizer = WhitespaceTokenizer::new();
        let collator = ExtractorCollator::new(tokenizer, false);

        let labels = vec!["card_lost".to_string()];
        let cls_obj = serde_json::json!({
            "task": "intent",
            "prompt": "What does the customer want?",
            "labels": ["card_lost"],
            "label_descriptions": { "card_lost": "The physical card is missing" }
        })
        .as_object()
        .cloned()
        .unwrap();

        let tokens = collator.build_classification_tokens("intent", &labels, &cls_obj);

        // Index 2 carries instruction + descriptions as plain text, not as separate
        // top-level schema tokens (see gliner2/processor.py::_transform_schema).
        assert_eq!(
            tokens[2],
            "intent: What does the customer want? \
             [DESCRIPTION] card_lost: The physical card is missing"
        );
        assert_eq!(tokens[3], "(");
        assert_eq!(tokens[4], "[L]");
        assert_eq!(tokens[5], "card_lost");
        // No stray structural tokens after the label.
        assert_eq!(tokens.len(), 8);
    }

    #[test]
    fn test_relation_token_building() {
        let tokenizer = WhitespaceTokenizer::new();
        let collator = ExtractorCollator::new(tokenizer, false);

        let tokens =
            collator.build_relation_tokens("works_for", &["head".to_string(), "tail".to_string()]);

        assert!(tokens.contains(&"(".to_string()));
        assert!(tokens.contains(&"[P]".to_string()));
        assert!(tokens.contains(&"works_for".to_string()));
        assert!(tokens.contains(&"[R]".to_string()));
        assert!(tokens.contains(&"head".to_string()));
        assert!(tokens.contains(&"tail".to_string()));
    }

    #[test]
    fn test_schema_encoding_entities_only() {
        let tokenizer = WhitespaceTokenizer::new();
        let collator = ExtractorCollator::new(tokenizer, false);
        let schema = serde_json::json!({
            "entities": {
                "person": "",
                "company": ""
            }
        });

        let result = collator.encode_schema(&schema);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.task_types, vec!["entities"]);
        assert_eq!(result.schema_tokens_list.len(), 1);
    }

    #[test]
    fn test_schema_encoding_mixed() {
        let tokenizer = WhitespaceTokenizer::new();
        let collator = ExtractorCollator::new(tokenizer, false);
        let schema = serde_json::json!({
            "entities": {
                "person": ""
            },
            "classifications": [
                {
                    "task": "sentiment",
                    "labels": ["positive", "negative"]
                }
            ],
            "relations": [
                {
                    "works_for": {
                        "head": "",
                        "tail": ""
                    }
                }
            ]
        });

        let result = collator.encode_schema(&schema);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.task_types.len(), 3);
        assert!(result.task_types.contains(&"entities".to_string()));
        assert!(result.task_types.contains(&"classifications".to_string()));
        assert!(result.task_types.contains(&"relations".to_string()));
    }
}
