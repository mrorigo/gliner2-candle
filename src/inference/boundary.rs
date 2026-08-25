//! GLiNER2.5 boundary-path inference orchestration.
//!
//! Bridges the collator/encoder pipeline to
//! [`BoundaryModel::score_sample`](crate::model::boundary::BoundaryModel):
//! builds word-level text states and per-type query states from the batch
//! metadata, scores candidates, and decodes entities/classifications.

use candle_core::Tensor;
use serde_json::{json, Value as JsonValue};

use crate::batch::preprocessed::PreprocessedBatch;
use crate::error::{GlinerError, Result};
use crate::model::boundary::{BoundaryModel, RecordDecoder, MASK_LOGIT};

/// One query aligned with a schema field.
struct QuerySpec {
    /// Schema group index this query belongs to.
    group: usize,
    /// Field name (entity type or classification label).
    name: String,
    /// Sequence position of this field's `[E]`/`[L]` marker token.
    marker_pos: usize,
}

/// Build query specs by walking each schema's token list in lockstep with its
/// special-token position indices.
///
/// Entity groups contribute one query per `[E]` marker; classification groups
/// one per `[L]` label. Other task types are skipped (relations/records come
/// in PLAN_2.5 Phase 3).
fn build_queries(batch: &PreprocessedBatch, sample_idx: usize) -> Result<Vec<QuerySpec>> {
    let mut specs = Vec::new();
    let num = batch.num_schemas(sample_idx).unwrap_or(0);
    for g in 0..num {
        let tokens = batch
            .schema_tokens(sample_idx, g)
            .ok_or_else(|| GlinerError::inference("missing schema tokens"))?;
        let specials = batch
            .schema_special_indices_for(sample_idx, g)
            .ok_or_else(|| GlinerError::inference("missing schema special indices"))?;
        let mut sp = 0usize;
        let mut pending: Option<(String, usize)> = None; // (marker kind, position)
        for token in tokens {
            if token.starts_with('[') && token.ends_with(']') {
                let pos = *specials.get(sp).ok_or_else(|| {
                    GlinerError::inference("special index out of sync with schema tokens")
                })?;
                sp += 1;
                pending = match token.as_str() {
                    "[E]" | "[L]" => Some((token.clone(), pos)),
                    _ => None,
                };
                continue;
            }
            if let Some((marker, marker_pos)) = pending.take() {
                specs.push(QuerySpec {
                    group: g,
                    name: token.clone(),
                    marker_pos,
                });
                let _ = marker;
            }
        }
    }
    Ok(specs)
}

/// Boundary-path extraction for a single collated sample.
///
/// Returns task results keyed like the span path (`entities`, task names).
pub(crate) fn extract_sample(
    boundary: &BoundaryModel,
    encoder_states: &Tensor, // [S, hidden]
    batch: &PreprocessedBatch,
    sample_idx: usize,
    threshold: f32,
    include_confidence: bool,
) -> Result<JsonValue> {
    // --- Word-level text states -------------------------------------------
    let seq_len = encoder_states.dims()[0];
    let word_rows = text_word_positions(batch, sample_idx, seq_len)?;
    let text_len = word_rows.len();
    let h = encoder_states.dims()[1];
    let mut flat_text = Vec::with_capacity(text_len * h);
    {
        let rows = encoder_states
            .to_vec2::<f32>()
            .map_err(|e| GlinerError::inference(format!("{e}")))?;
        for &w in &word_rows {
            flat_text.extend_from_slice(&rows[w]);
        }
    }
    let text_states = Tensor::from_slice(&flat_text, (text_len.max(1), h), encoder_states.device())
        .map_err(|e| GlinerError::inference(format!("{e}")))?;

    // --- Query states -------------------------------------------------------
    let specs = build_queries(batch, sample_idx)?;
    let q = specs.len();
    let mut flat_queries = Vec::with_capacity(q * h);
    if q > 0 {
        let rows = encoder_states
            .to_vec2::<f32>()
            .map_err(|e| GlinerError::inference(format!("{e}")))?;
        for spec in &specs {
            flat_queries.extend_from_slice(&rows[spec.marker_pos.min(seq_len - 1)]);
        }
    } else {
        return Ok(JsonValue::Object(Default::default()));
    }
    let query_states = Tensor::from_slice(&flat_queries, (q, h), encoder_states.device())
        .map_err(|e| GlinerError::inference(format!("{e}")))?;

    // --- Score --------------------------------------------------------------
    let scored = boundary.score_sample(&text_states, text_len, &query_states)?;

    // --- Decode per group -----------------------------------------------------
    let original_text = batch.original_text(sample_idx).unwrap_or_default();
    let starts_map = batch.sample_start_mapping(sample_idx);
    let ends_map = batch.sample_end_mapping(sample_idx);
    let task_types = batch.sample_task_types(sample_idx).unwrap_or(&[]);

    let mut result = serde_json::Map::new();
    for (group, task) in task_types.iter().enumerate() {
        match task.as_str() {
            "entities" => {
                let mut by_type: serde_json::Map<String, JsonValue> = Default::default();
                for (qi, spec) in specs.iter().enumerate() {
                    if spec.group != group {
                        continue;
                    }
                    let mut hits: Vec<(usize, usize, f32)> = Vec::new();
                    for ci in 0..scored.valid.len() {
                        let score = scored.scores[ci][qi];
                        if scored.valid[ci] && score > MASK_LOGIT / 2.0 {
                            let prob = sigmoid(score);
                            if prob >= threshold {
                                hits.push((scored.starts[ci], scored.ends[ci], prob));
                            }
                        }
                    }
                    hits.sort_by(|a, b| {
                        b.2.partial_cmp(&a.2)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then(a.0.cmp(&b.0))
                            .then(a.1.cmp(&b.1))
                    });
                    if hits.is_empty() {
                        continue;
                    }
                    let mut entries = Vec::with_capacity(hits.len());
                    for (s, e, conf) in hits {
                        let char_start = char_offset(starts_map, s.saturating_sub(1));
                        let char_end = char_offset(ends_map, (e - 1).min(text_len - 1));
                        let text = safe_slice(original_text, char_start, char_end);
                        if include_confidence {
                            entries.push(json!({
                                "text": text,
                                "start": char_start,
                                "end": char_end,
                                "confidence": conf,
                            }));
                        } else {
                            entries.push(json!(text));
                        }
                    }
                    by_type.insert(spec.name.clone(), JsonValue::Array(entries));
                }
                result.insert("entities".to_string(), JsonValue::Object(by_type));
            }
            "classifications" => {
                decode_classification(
                    boundary,
                    encoder_states,
                    batch,
                    sample_idx,
                    group,
                    &specs,
                    threshold,
                    include_confidence,
                    &mut result,
                )?;
            }
            "records" => {
                if let (Some(rd), Some(cand_states)) =
                    (boundary.record_decoder.as_ref(), scored.candidate_states.as_ref())
                {
                    let group_specs: Vec<&QuerySpec> =
                        specs.iter().filter(|s| s.group == group).collect();
                    if !group_specs.is_empty() {
                        let field_qids: Vec<usize> =
                            group_specs.iter().enumerate().map(|(i, _)| i).collect();
                        let cand_spans: Vec<(usize, usize)> = scored
                            .starts
                            .iter()
                            .zip(&scored.ends)
                            .map(|(&s, &e)| (s, e))
                            .collect();
                        let flat_queries: Vec<Vec<f32>> = {
                            let rows = query_states
                                .to_vec2::<f32>()
                                .map_err(|e| GlinerError::inference(format!("{e}")))?;
                            rows
                        };
                        let c_valid = scored.valid.clone();
                        let (obj_logits, assign_logits, inst_spans) =
                            rd.forward_group(
                                &flat_queries,
                                cand_states,
                                &cand_spans,
                                &c_valid,
                                &field_qids,
                            )?;
                        let decoded = RecordDecoder::decode_group(
                            &obj_logits,
                            &assign_logits,
                            &inst_spans,
                            &cand_spans,
                            &c_valid,
                            &field_qids,
                            threshold,
                            0.3,
                        );
                        let task_name = batch
                            .schema_tokens(sample_idx, group)
                            .and_then(|t| t.get(2).map(|s| s.to_string()))
                            .unwrap_or_else(|| format!("record_{group}"));
                        let entries: Vec<JsonValue> = decoded
                            .iter()
                            .map(|rec| {
                                let mut fields_json = serde_json::Map::new();
                                for (&qid, spans) in &rec.fields {
                                    if let Some(spec) = group_specs.get(qid) {
                                        let span_jsons: Vec<JsonValue> = spans
                                            .iter()
                                            .map(|&(s, e)| {
                                                let cs = char_offset(starts_map, s.saturating_sub(1));
                                                let ce = char_offset(ends_map, (e - 1).min(text_len - 1));
                                                json!({"text": safe_slice(original_text, cs, ce), "start": cs, "end": ce})
                                            })
                                            .collect();
                                        fields_json.insert(spec.name.clone(), JsonValue::Array(span_jsons));
                                    }
                                }
                                if include_confidence {
                                    json!({"fields": fields_json, "confidence": rec.score})
                                } else {
                                    json!({"fields": fields_json})
                                }
                            })
                            .collect();
                        result.insert(task_name, JsonValue::Array(entries));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(JsonValue::Object(result))
}

/// Classification decoding via the boundary classifier head.
#[allow(clippy::too_many_arguments)]
fn decode_classification(
    boundary: &BoundaryModel,
    encoder_states: &Tensor,
    batch: &PreprocessedBatch,
    sample_idx: usize,
    group: usize,
    specs: &[QuerySpec],
    threshold: f32,
    include_confidence: bool,
    result: &mut serde_json::Map<String, JsonValue>,
) -> Result<()> {
    let Some(classifier) = boundary.classifier_ref() else {
        return Ok(());
    };
    let group_specs: Vec<&QuerySpec> =
        specs.iter().filter(|s| s.group == group).collect();
    if group_specs.is_empty() {
        return Ok(());
    }
    // Re-run the classifier on this group's marker embeddings.
    let h = encoder_states.dims()[1];
    let seq_len = encoder_states.dims()[0];
    let rows = encoder_states
        .to_vec2::<f32>()
        .map_err(|e| GlinerError::inference(format!("{e}")))?;
    let mut flat = Vec::with_capacity(group_specs.len() * h);
    for spec in &group_specs {
        flat.extend_from_slice(&rows[spec.marker_pos.min(seq_len - 1)]);
    }
    let embs = Tensor::from_slice(&flat, (group_specs.len(), h), encoder_states.device())
        .map_err(|e| GlinerError::inference(format!("{e}")))?;
    let logits = classifier.forward(&embs)?;
    let logits = logits
        .squeeze(1)?
        .to_vec1::<f32>()
        .map_err(|e| GlinerError::inference(format!("{e}")))?;

    let schema_json = batch
        .original_schemas
        .get(sample_idx)
        .cloned()
        .unwrap_or(JsonValue::Null);
    let multi_label = schema_json
        .get("classifications")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.get(group))
        .and_then(|o| o.get("multi_label"))
        .and_then(|m| m.as_bool())
        .unwrap_or(false);

    let task_name = batch
        .schema_tokens(sample_idx, group)
        .and_then(|t| t.get(2).map(|s| s.to_string()))
        .unwrap_or_else(|| format!("task_{group}"));

    if multi_label {
        let selected: Vec<JsonValue> = group_specs
            .iter()
            .zip(&logits)
            .filter(|(_, lg)| sigmoid(**lg) >= threshold)
            .map(|(spec, &lg)| {
                if include_confidence {
                    json!({"label": spec.name, "confidence": sigmoid(lg)})
                } else {
                    json!(spec.name)
                }
            })
            .collect();
        result.insert(task_name, JsonValue::Array(selected));
    } else {
        let (best_qi, best_lg) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        let spec = &group_specs[best_qi];
        result.insert(
            task_name,
            if include_confidence {
                json!({"label": spec.name, "confidence": softmax_conf(*best_lg, &logits)})
            } else {
                json!(spec.name)
            },
        );
    }
    Ok(())
}

fn text_word_positions(
    batch: &PreprocessedBatch,
    sample_idx: usize,
    _seq_len: usize,
) -> Result<Vec<usize>> {
    // The collator records first-subword positions per whitespace word.
    let count = *batch.text_word_counts.get(sample_idx).ok_or_else(|| GlinerError::inference("missing text word count"))?;
    // Recompute positions from mapped indices is unreliable here; instead read
    // the tensor the collator produced.
    let tensor = batch
        .text_word_indices
        .as_ref()
        .ok_or_else(|| GlinerError::inference("missing text_word_indices tensor"))?;
    let all = tensor
        .to_vec2::<i64>()
        .map_err(|e| GlinerError::inference(format!("{e}")))?;
    let row = &all[sample_idx];
    Ok(row[..count.min(row.len())]
        .iter()
        .map(|&v| v.max(0) as usize)
        .collect())
}

fn char_offset(mapping: Option<&[usize]>, word_idx: usize) -> usize {
    mapping
        .and_then(|m| m.get(word_idx))
        .copied()
        .unwrap_or(0)
}

fn safe_slice(text: &str, start: usize, end: usize) -> String {
    let end = end.min(text.len());
    let start = start.min(end);
    text.get(start..end)
        .map(str::to_string)
        .unwrap_or_default()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax_conf(best: f32, all: &[f32]) -> f32 {
    let max = all.iter().cloned().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = all.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    ((best - max).exp()) / sum.max(f32::MIN_POSITIVE)
}
