//! GLiNER2.5 boundary-path inference orchestration.
//!
//! Bridges the collator/encoder pipeline to
//! [`BoundaryModel::score_sample`](crate::model::boundary::BoundaryModel):
//! builds word-level text states and per-type query states from the batch
//! metadata, scores candidates, and decodes entities/classifications.

use std::collections::{HashMap, HashSet};

use candle_core::Tensor;
use serde_json::{Value as JsonValue, json};

use crate::batch::preprocessed::PreprocessedBatch;
use crate::error::{GlinerError, Result};
use crate::model::boundary::{BoundaryModel, MASK_LOGIT, SharedPoolScores, decode_group_natural};
use crate::schema::types::{AttributeGroup, FieldCardinality, FieldDtype};

/// Resolved span-attribute metadata for one extraction call.
pub(crate) struct AttributesRuntime {
    /// Attribute label -> model-facing prompt label.
    pub prompt_by_label: HashMap<String, String>,
    /// Group definitions in declaration order.
    pub groups: Vec<(String, AttributeGroup)>,
}

impl AttributesRuntime {
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    fn prompt_labels(&self) -> HashSet<&str> {
        self.prompt_by_label.values().map(|s| s.as_str()).collect()
    }
}

/// One query aligned with a schema field.
struct QuerySpec {
    /// Schema group index this query belongs to.
    group: usize,
    /// Field name (entity type or classification label).
    name: String,
    /// Sequence position of this field's `[E]`/`[L]`/`[R]` marker token.
    marker_pos: usize,
}

/// Metadata for a single relation type within a schema group.
struct RelationSpec {
    /// Schema group index.
    group: usize,
    /// Human-readable relation type name.
    relation_name: String,
    /// Query index of the head entity role.
    head_qid: usize,
    /// Query index of the tail entity role.
    tail_qid: usize,
}

/// Build query specs and relation metadata by walking each schema's token list.
///
/// Entity groups contribute one query per `[E]` marker; classification groups
/// one per `[L]` label; relation groups one per `[R]` marker. Relation groups
/// also produce [`RelationSpec`] entries linking head/tail role query indices.
fn build_queries(
    batch: &PreprocessedBatch,
    sample_idx: usize,
) -> Result<(Vec<QuerySpec>, Vec<RelationSpec>)> {
    let mut specs = Vec::new();
    let mut relations = Vec::new();
    let num = batch.num_schemas(sample_idx).unwrap_or(0);
    for g in 0..num {
        let tokens = batch
            .schema_tokens(sample_idx, g)
            .ok_or_else(|| GlinerError::inference("missing schema tokens"))?;
        let specials = batch
            .schema_special_indices_for(sample_idx, g)
            .ok_or_else(|| GlinerError::inference("missing schema special indices"))?;
        let mut sp = 0usize;
        let mut pending: Option<(String, usize)> = None;
        let first_role_qid = specs.len();
        for token in tokens {
            if token.starts_with('[') && token.ends_with(']') {
                let pos = *specials.get(sp).ok_or_else(|| {
                    GlinerError::inference("special index out of sync with schema tokens")
                })?;
                sp += 1;
                pending = match token.as_str() {
                    "[E]" | "[L]" | "[C]" | "[R]" => Some((token.clone(), pos)),
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
        // If this group has relation roles ([R] markers), pair them up.
        let role_specs: Vec<&QuerySpec> = specs[first_role_qid..]
            .iter()
            .filter(|s| s.group == g)
            .collect();
        if role_specs.len() >= 2 {
            let relation_name = tokens
                .get(2)
                .cloned()
                .unwrap_or_else(|| format!("relation_{g}"));
            relations.push(RelationSpec {
                group: g,
                relation_name,
                head_qid: first_role_qid,
                tail_qid: first_role_qid + 1,
            });
        }
    }
    Ok((specs, relations))
}

/// Natural (anchor-driven) structured-record decode for one structure group.
///
/// Returns `Some(records)` only when the schema declares natural mode with a
/// valid anchor and a record decoder is available; otherwise `None` so the
/// caller falls back to the flat per-field span path.
#[allow(clippy::too_many_arguments)]
fn decode_structure_records(
    boundary: &BoundaryModel,
    encoder_states: &Tensor, // [S, hidden]
    scored: &SharedPoolScores,
    batch: &PreprocessedBatch,
    sample_idx: usize,
    group: usize,
    specs: &[QuerySpec],
    task_name: &str,
    threshold: f32,
    include_confidence: bool,
    include_spans: bool,
    original_text: &str,
    starts_map: Option<&[usize]>,
    ends_map: Option<&[usize]>,
    text_len: usize,
) -> Result<Option<JsonValue>> {
    // --- Resolve structure metadata (`__mode__`, `__anchor__`, cardinality) ---
    let Some(schemas) = batch.original_schemas.get(sample_idx) else {
        return Ok(None);
    };
    let Some(structure_obj) = schemas
        .get("json_structures")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            arr.iter().find(|el| {
                el.as_object()
                    .map(|o| o.contains_key(task_name))
                    .unwrap_or(false)
            })
        })
        .and_then(|el| el.as_object())
    else {
        return Ok(None);
    };
    let is_natural = structure_obj
        .get("__mode__")
        .and_then(|v| v.as_str())
        .map(|m| m.eq_ignore_ascii_case("natural"))
        .unwrap_or(false);
    if !is_natural {
        return Ok(None);
    }
    let anchor = structure_obj
        .get("__anchor__")
        .and_then(|v| v.as_str())
        .filter(|a| !a.is_empty())
        .map(|s| s.to_string());
    let Some(anchor) = anchor else {
        return Ok(None);
    };
    let Some(field_obj) = structure_obj.get(task_name).and_then(|v| v.as_object()) else {
        return Ok(None);
    };

    // Group's field specs in schema-token order (matches field candidate order).
    let fields: Vec<&QuerySpec> = specs.iter().filter(|s| s.group == group).collect();
    let qids: Vec<usize> = specs
        .iter()
        .enumerate()
        .filter(|(_, s)| s.group == group)
        .map(|(qi, _)| qi)
        .collect();
    let num_fields = fields.len();
    if num_fields == 0 {
        return Ok(None);
    }

    let anchor_field_idx = fields
        .iter()
        .position(|s| s.name == anchor)
        .unwrap_or(usize::MAX);
    if anchor_field_idx == usize::MAX {
        return Ok(None);
    }

    let decoder = match &boundary.record_decoder {
        Some(d) => d,
        None => {
            return Ok(None);
        }
    };
    let Some(ref states) = scored.candidate_states else {
        return Ok(None);
    };

    // --- Per-field candidate enumeration from the shared pool ---
    let mut field_candidates: Vec<Vec<usize>> = Vec::with_capacity(num_fields);
    let mut field_cardinality: Vec<FieldCardinality> = Vec::with_capacity(num_fields);
    for (f, spec) in fields.iter().enumerate() {
        let qi = qids[f];
        let mut cands: Vec<usize> = Vec::new();
        for ci in 0..scored.valid.len() {
            if scored.valid[ci] && scored.scores[ci][qi] > MASK_LOGIT / 2.0 {
                cands.push(ci);
            }
        }
        cands.sort_unstable();
        field_candidates.push(cands);

        let meta = field_obj.get(&spec.name);
        let dtype = meta
            .and_then(|v| v.get("dtype"))
            .and_then(|v| v.as_str())
            .map(|d| {
                if d.eq_ignore_ascii_case("list") {
                    FieldDtype::List
                } else {
                    FieldDtype::Str
                }
            });
        let card = meta
            .and_then(|v| v.get("cardinality"))
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<FieldCardinality>().ok())
            .or_else(|| dtype.map(FieldCardinality::for_dtype))
            .unwrap_or(FieldCardinality::ZeroOrOne);
        field_cardinality.push(card);
    }

    // Anchor candidates seed one instance each (index-aligned with object logits).
    let anchor_cands = &field_candidates[anchor_field_idx];
    if anchor_cands.is_empty() {
        return Ok(None);
    }
    let anchor_object_logits: Vec<f32> = anchor_cands
        .iter()
        .map(|&ci| scored.scores[ci][qids[anchor_field_idx]])
        .collect();
    let anchor_states: Vec<Vec<f32>> = anchor_cands.iter().map(|&ci| states[ci].clone()).collect();
    let anchor_spans: Vec<(usize, usize)> = anchor_cands
        .iter()
        .map(|&ci| (scored.starts[ci], scored.ends[ci]))
        .collect();
    let field_cand_states: Vec<Vec<Vec<f32>>> = field_candidates
        .iter()
        .map(|cs| cs.iter().map(|&ci| states[ci].clone()).collect())
        .collect();
    let field_spans: Vec<Vec<(usize, usize)>> = field_candidates
        .iter()
        .map(|cs| {
            cs.iter()
                .map(|&ci| (scored.starts[ci], scored.ends[ci]))
                .collect()
        })
        .collect();

    // Schema query states indexed by query id (spec.marker_pos rows).
    let rows = encoder_states
        .to_vec2::<f32>()
        .map_err(|e| GlinerError::inference(format!("{e}")))?;
    let seq_len = encoder_states.dims()[0];
    let mut query_states: Vec<Vec<f32>> = Vec::with_capacity(specs.len());
    for spec in specs {
        query_states.push(rows[spec.marker_pos.min(seq_len.saturating_sub(1))].clone());
    }

    let (object_logits, assign_logits, _) = decoder
        .forward_group_natural(
            &query_states,
            &qids,
            &anchor_object_logits,
            &anchor_states,
            &anchor_spans,
            &field_cand_states,
        )
        .map_err(|e| GlinerError::inference(format!("{e}")))?;

    let records = decode_group_natural(
        &object_logits,
        &assign_logits,
        &field_spans,
        anchor_field_idx,
        &field_cardinality,
        threshold,
        threshold,
    );

    // --- Emit JSON records ---
    let pair_temp = boundary.config.pair_temperature.max(f32::MIN_POSITIVE);
    let mut entries: Vec<JsonValue> = Vec::new();
    for rec in records {
        let mut rec_obj = serde_json::Map::new();
        for (f, spans) in &rec.fields {
            if *f >= fields.len() {
                continue;
            }
            let name = &fields[*f].name;
            let qi = qids[*f];
            let pool_cands = field_candidates.get(*f).cloned().unwrap_or_default();
            let mut span_jsons: Vec<JsonValue> = Vec::new();
            for (k, &(s, e)) in spans.iter().enumerate() {
                let assignment_conf = rec
                    .field_scores
                    .get(f)
                    .and_then(|v| v.get(k))
                    .copied()
                    .unwrap_or(0.0);
                // candidate-span score: sigmoid(pair_logit / pair_temperature).
                // Mirrors the reference `_candidate_span_probability` + the
                // `min(candidate, assignment)` combination in `_format_field`.
                let cand_prob = field_spans[*f]
                    .iter()
                    .position(|&sp| sp == (s, e))
                    .and_then(|ci| pool_cands.get(ci))
                    .and_then(|&pool_ci| scored.scores.get(pool_ci))
                    .and_then(|row| row.get(qi))
                    .map(|&lg| sigmoid(lg / pair_temp))
                    .unwrap_or(0.0);
                let conf = if *f == anchor_field_idx {
                    cand_prob
                } else {
                    cand_prob.min(assignment_conf)
                };
                let char_start = char_offset(starts_map, s.min(text_len.saturating_sub(1)));
                let char_end = char_offset(ends_map, (e.saturating_sub(1)).min(text_len - 1));
                let text = safe_slice(original_text, char_start, char_end);
                let span_val = if include_spans && include_confidence {
                    json!({"text": text, "start": char_start, "end": char_end, "confidence": conf})
                } else if include_spans {
                    json!({"text": text, "start": char_start, "end": char_end})
                } else if include_confidence {
                    json!({"text": text, "confidence": conf})
                } else {
                    json!(text)
                };
                span_jsons.push(span_val);
            }
            rec_obj.insert(name.clone(), JsonValue::Array(span_jsons));
        }
        if !rec_obj.is_empty() {
            entries.push(JsonValue::Object(rec_obj));
        }
    }

    Ok(Some(JsonValue::Array(entries)))
}

/// Decode the `"entities"` groups, recording canonical endpoint spans.
///
/// Relations read endpoints from the spans produced here, so this must run for
/// every entity group before the relation pass.
#[allow(clippy::too_many_arguments)]
fn decode_entities(
    boundary: &BoundaryModel,
    text_states: &Tensor,
    text_len: usize,
    query_states: &Tensor,
    specs: &[QuerySpec],
    scored: &SharedPoolScores,
    group: usize,
    threshold: f32,
    include_confidence: bool,
    include_spans: bool,
    original_text: &str,
    starts_map: Option<&[usize]>,
    ends_map: Option<&[usize]>,
    attrs: Option<&AttributesRuntime>,
    result: &mut serde_json::Map<String, JsonValue>,
    all_entity_spans_by_type: &mut HashMap<String, Vec<(usize, usize)>>,
    all_canonical_entity_spans: &mut HashSet<(usize, usize)>,
) -> Result<()> {
    let attr_prompts: HashSet<&str> = attrs.map(|a| a.prompt_labels()).unwrap_or_default();
    let has_attributes = attrs.is_some_and(|a| !a.is_empty());

    // name -> (entries, token coords per entry)
    #[allow(clippy::type_complexity)]
    let mut decoded_entities: Vec<(String, Vec<JsonValue>, Vec<(usize, usize)>)> = Vec::new();
    for (qi, spec) in specs.iter().enumerate() {
        if spec.group != group || attr_prompts.contains(spec.name.as_str()) {
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
        // Boundary default overlap policy "flat": keep the
        // maximum-total-score non-overlapping subset.
        let resolved = resolve_flat_spans(hits.into_iter().map(|(st, e, p)| (p, st, e)).collect());
        hits = resolved.into_iter().map(|(p, st, e)| (st, e, p)).collect();
        if hits.is_empty() {
            continue;
        }
        let mut entries = Vec::with_capacity(hits.len());
        let mut coords = Vec::with_capacity(hits.len());
        for (s, e, conf) in hits {
            let char_start = char_offset(starts_map, s.min(text_len.saturating_sub(1)));
            let char_end = char_offset(ends_map, (e - 1).min(text_len - 1));
            let text = safe_slice(original_text, char_start, char_end);
            let entry = if has_attributes {
                // Attributed format follows the flags strictly.
                let mut obj = serde_json::Map::new();
                obj.insert("text".into(), json!(text));
                if include_confidence {
                    obj.insert("confidence".into(), json!(conf));
                }
                if include_spans {
                    obj.insert("start".into(), json!(char_start));
                    obj.insert("end".into(), json!(char_end));
                }
                JsonValue::Object(obj)
            } else if include_spans && include_confidence {
                json!({
                    "text": text,
                    "start": char_start,
                    "end": char_end,
                    "confidence": conf,
                })
            } else if include_spans {
                json!({
                    "text": text,
                    "start": char_start,
                    "end": char_end,
                })
            } else if include_confidence {
                json!({
                    "text": text,
                    "confidence": conf,
                })
            } else {
                json!(text)
            };
            entries.push(entry);
            coords.push((s, e));
        }
        decoded_entities.push((spec.name.clone(), entries, coords));
    }

    if has_attributes {
        attach_entity_attributes(
            boundary,
            text_states,
            text_len,
            query_states,
            specs,
            attrs.expect("attrs"),
            &mut decoded_entities,
        )?;
    }

    let mut by_type: serde_json::Map<String, JsonValue> = Default::default();
    for (name, entries, coords) in decoded_entities {
        all_entity_spans_by_type
            .entry(name.clone())
            .or_default()
            .extend(coords.iter().copied());
        all_canonical_entity_spans.extend(coords.iter().copied());
        by_type.insert(name, JsonValue::Array(entries));
    }
    result.insert("entities".to_string(), JsonValue::Object(by_type));
    Ok(())
}

/// Decode one `"records"` / `"json_structures"` group into `result`.
///
/// Two paths: a natural/anchored record decode when the schema declares
/// `__mode__ == "natural"` with an `__anchor__`, otherwise a flat per-field span
/// list from the shared candidate pool.
#[allow(clippy::too_many_arguments)]
fn decode_records(
    boundary: &BoundaryModel,
    batch: &PreprocessedBatch,
    sample_idx: usize,
    encoder_states: &Tensor,
    text_len: usize,
    scored: &SharedPoolScores,
    specs: &[QuerySpec],
    group: usize,
    threshold: f32,
    include_confidence: bool,
    include_spans: bool,
    original_text: &str,
    starts_map: Option<&[usize]>,
    ends_map: Option<&[usize]>,
    result: &mut serde_json::Map<String, JsonValue>,
) -> Result<()> {
    let task_name = batch
        .schema_tokens(sample_idx, group)
        .and_then(|t| t.get(2).map(|s| s.to_string()))
        .unwrap_or_else(|| format!("record_{group}"));

    // Natural (anchor-driven) record decoding preserves instance
    // identity. Only engaged when the schema declares it.
    if let Some(records) = decode_structure_records(
        boundary,
        encoder_states,
        scored,
        batch,
        sample_idx,
        group,
        specs,
        &task_name,
        threshold,
        include_confidence,
        include_spans,
        original_text,
        starts_map,
        ends_map,
        text_len,
    )? {
        result.insert(task_name, records);
        return Ok(());
    }

    let mut entries: Vec<JsonValue> = Vec::new();
    let mut fields_json = serde_json::Map::new();

    for (qi, spec) in specs.iter().enumerate() {
        if spec.group != group {
            continue;
        }
        let mut hits: Vec<(usize, usize, f32)> = Vec::new();
        for ci in 0..scored.valid.len() {
            if !scored.valid[ci] {
                continue;
            }
            let score = scored.scores[ci][qi];
            if score > MASK_LOGIT / 2.0 {
                let prob = sigmoid(score);
                if prob >= threshold {
                    hits.push((scored.starts[ci], scored.ends[ci], prob));
                }
            }
        }
        let resolved = resolve_flat_spans(hits.into_iter().map(|(st, e, p)| (p, st, e)).collect());
        let mut span_jsons: Vec<JsonValue> = Vec::new();
        for (conf, s, e) in resolved {
            let char_start = char_offset(starts_map, s.min(text_len.saturating_sub(1)));
            let char_end = char_offset(ends_map, (e - 1).min(text_len - 1));
            let text = safe_slice(original_text, char_start, char_end);
            let span_val = if include_spans && include_confidence {
                json!({"text": text, "start": char_start, "end": char_end, "confidence": conf})
            } else if include_spans {
                json!({"text": text, "start": char_start, "end": char_end})
            } else if include_confidence {
                json!({"text": text, "confidence": conf})
            } else {
                json!(text)
            };
            span_jsons.push(span_val);
        }
        fields_json.insert(spec.name.clone(), JsonValue::Array(span_jsons));
    }

    if fields_json
        .values()
        .any(|v| matches!(v, JsonValue::Array(a) if !a.is_empty()))
    {
        entries.push(JsonValue::Object(fields_json));
    }

    result.insert(task_name, JsonValue::Array(entries));
    Ok(())
}

/// Decode one `"relations"` group into `result`.
///
/// Endpoints come from the canonical entity spans collected by the first pass,
/// so `extract_sample` must run the entity groups before calling this.
#[allow(clippy::too_many_arguments)]
fn decode_relations(
    boundary: &BoundaryModel,
    text_states: &Tensor,
    text_len: usize,
    hidden_size: usize,
    query_states: &Tensor,
    specs: &[QuerySpec],
    relation_specs: &[RelationSpec],
    scored: &SharedPoolScores,
    group: usize,
    threshold: f32,
    include_confidence: bool,
    include_spans: bool,
    original_text: &str,
    starts_map: Option<&[usize]>,
    ends_map: Option<&[usize]>,
    all_entity_spans_by_type: &HashMap<String, Vec<(usize, usize)>>,
    all_canonical_entity_spans: &HashSet<(usize, usize)>,
    result: &mut serde_json::Map<String, JsonValue>,
) -> Result<()> {
    if let Some(rs) = boundary.relation_scorer.as_ref() {
        // The relation scorer reads word-level hidden states
        // (H-dim), not the D-dim boundary encoder output.
        let word_states: Vec<Vec<f32>> = text_states
            .to_vec2::<f32>()
            .map_err(|e| GlinerError::inference(format!("{e}")))?;
        // Find the relation spec for this group.
        if let Some(rel) = relation_specs.iter().find(|r| r.group == group) {
            let h_qid = rel.head_qid;
            let t_qid = rel.tail_qid;
            if h_qid < specs.len() && t_qid < specs.len() {
                // Build relation query state (directional concat of head and tail query states).
                let qr: Vec<f32> = {
                    let rows = query_states
                        .to_vec2::<f32>()
                        .map_err(|e| GlinerError::inference(format!("{e}")))?;
                    let mut r = Vec::with_capacity(2 * hidden_size);
                    r.extend_from_slice(&rows[h_qid]);
                    r.extend_from_slice(&rows[t_qid]);
                    r
                };

                let h_name = &specs[h_qid].name;
                let t_name = &specs[t_qid].name;

                let (head_cands, tail_cands) = if !all_canonical_entity_spans.is_empty() {
                    let mut hc: Vec<(usize, usize)> =
                        if let Some(type_spans) = all_entity_spans_by_type.get(h_name) {
                            type_spans.clone()
                        } else {
                            all_canonical_entity_spans.iter().copied().collect()
                        };
                    hc.sort_unstable();
                    let mut tc: Vec<(usize, usize)> =
                        if let Some(type_spans) = all_entity_spans_by_type.get(t_name) {
                            type_spans.clone()
                        } else {
                            all_canonical_entity_spans.iter().copied().collect()
                        };
                    tc.sort_unstable();
                    (hc, tc)
                } else {
                    // Fallback when no entity group is in schema: use role query proposals
                    let mut head_hits: Vec<(usize, usize, f32)> = Vec::new();
                    for ci in 0..scored.valid.len() {
                        if !scored.valid[ci] {
                            continue;
                        }
                        let h_score = scored.scores[ci][h_qid];
                        if h_score > MASK_LOGIT / 2.0 {
                            let prob = sigmoid(h_score);
                            if prob >= threshold {
                                head_hits.push((scored.starts[ci], scored.ends[ci], prob));
                            }
                        }
                    }
                    let resolved_heads = resolve_flat_spans(
                        head_hits.into_iter().map(|(st, e, p)| (p, st, e)).collect(),
                    );
                    let hc: Vec<(usize, usize)> = resolved_heads
                        .into_iter()
                        .map(|(_, st, e)| (st, e))
                        .collect();

                    let mut tail_hits: Vec<(usize, usize, f32)> = Vec::new();
                    for ci in 0..scored.valid.len() {
                        if !scored.valid[ci] {
                            continue;
                        }
                        let t_score = scored.scores[ci][t_qid];
                        if t_score > MASK_LOGIT / 2.0 {
                            let prob = sigmoid(t_score);
                            if prob >= threshold {
                                tail_hits.push((scored.starts[ci], scored.ends[ci], prob));
                            }
                        }
                    }
                    let resolved_tails = resolve_flat_spans(
                        tail_hits.into_iter().map(|(st, e, p)| (p, st, e)).collect(),
                    );
                    let tc: Vec<(usize, usize)> = resolved_tails
                        .into_iter()
                        .map(|(_, st, e)| (st, e))
                        .collect();

                    (hc, tc)
                };

                // Generate all head×tail pairs, skip self-loops.
                let mut pair_heads: Vec<(usize, usize)> = Vec::new();
                let mut pair_tails: Vec<(usize, usize)> = Vec::new();
                for &hs in &head_cands {
                    for &ts in &tail_cands {
                        if hs == ts {
                            continue; // no self-loops
                        }
                        pair_heads.push(hs);
                        pair_tails.push(ts);
                    }
                }
                if !pair_heads.is_empty() {
                    let pair_scores =
                        rs.forward(&word_states, &qr, &pair_heads, &pair_tails, text_len)?;
                    // Decode pairs above threshold, ranked descending by score.
                    let mut scored_entries: Vec<(f32, JsonValue)> = Vec::new();
                    let mut seen_pairs: HashSet<((usize, usize), (usize, usize))> = HashSet::new();
                    for (idx, &score) in pair_scores.iter().enumerate() {
                        let prob = sigmoid(score);
                        if prob < threshold {
                            continue;
                        }
                        let (hs, he) = pair_heads[idx];
                        let (ts, te) = pair_tails[idx];
                        if !seen_pairs.insert(((hs, he), (ts, te))) {
                            continue;
                        }
                        let cs_h = char_offset(starts_map, hs.min(text_len.saturating_sub(1)));
                        let ce_h = char_offset(ends_map, (he - 1).min(text_len - 1));
                        let cs_t = char_offset(starts_map, ts.min(text_len.saturating_sub(1)));
                        let ce_t = char_offset(ends_map, (te - 1).min(text_len - 1));
                        let head_text = safe_slice(original_text, cs_h, ce_h);
                        let tail_text = safe_slice(original_text, cs_t, ce_t);
                        // Output shape mirrors the Python decoder:
                        // spans > confidence-only > bare pairs.
                        let obj = if include_spans {
                            let mut o = json!({
                                "head": {"text": head_text, "start": cs_h, "end": ce_h},
                                "tail": {"text": tail_text, "start": cs_t, "end": ce_t},
                            });
                            if include_confidence {
                                let conf = json!(prob);
                                o["head"]["confidence"] = conf.clone();
                                o["tail"]["confidence"] = conf;
                            }
                            o
                        } else if include_confidence {
                            json!({
                                "head": {"text": head_text, "confidence": prob},
                                "tail": {"text": tail_text, "confidence": prob},
                            })
                        } else {
                            json!([head_text, tail_text])
                        };
                        scored_entries.push((prob, obj));
                    }
                    scored_entries
                        .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    let entries: Vec<JsonValue> =
                        scored_entries.into_iter().map(|(_, obj)| obj).collect();
                    let mut rel_map = match result.remove("relation_extraction") {
                        Some(JsonValue::Object(m)) => m,
                        _ => serde_json::Map::new(),
                    };
                    rel_map.insert(rel.relation_name.clone(), JsonValue::Array(entries));
                    result.insert(
                        "relation_extraction".to_string(),
                        JsonValue::Object(rel_map),
                    );
                }
            }
        }
    };

    Ok(())
}

/// Boundary-path extraction for a single collated sample.
///
/// Returns task results keyed like the span path (`entities`, task names).
#[allow(clippy::too_many_arguments)]
pub(crate) fn extract_sample(
    boundary: &BoundaryModel,
    encoder_states: &Tensor, // [S, hidden]
    batch: &PreprocessedBatch,
    sample_idx: usize,
    threshold: f32,
    include_confidence: bool,
    include_spans: bool,
    attrs: Option<&AttributesRuntime>,
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
    let t0 = std::time::Instant::now();
    let (specs, relation_specs) = build_queries(batch, sample_idx)?;
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
    if std::env::var("GLINER2_PROFILE").is_ok() {
        eprintln!(
            "PROFILE   score_sample={:?} q={} text_len={}",
            t0.elapsed(),
            specs.len(),
            text_len
        );
    }

    // --- Decode per group -----------------------------------------------------
    let original_text = batch.original_text(sample_idx).unwrap_or_default();
    let starts_map = batch.sample_start_mapping(sample_idx);
    let ends_map = batch.sample_end_mapping(sample_idx);
    let task_types = batch.sample_task_types(sample_idx).unwrap_or(&[]);

    let mut result = serde_json::Map::new();
    let mut all_entity_spans_by_type: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
    let mut all_canonical_entity_spans: HashSet<(usize, usize)> = HashSet::new();

    // First pass: decode all entity groups so relation extraction has access to canonical endpoints
    // First pass: decode all entity groups so relation extraction has access
    // to canonical endpoints
    for (group, task) in task_types.iter().enumerate() {
        if task.as_str() == "entities" {
            decode_entities(
                boundary,
                &text_states,
                text_len,
                &query_states,
                &specs,
                &scored,
                group,
                threshold,
                include_confidence,
                include_spans,
                original_text,
                starts_map,
                ends_map,
                attrs,
                &mut result,
                &mut all_entity_spans_by_type,
                &mut all_canonical_entity_spans,
            )?;
        }
    }

    // Second pass: decode classifications, records, and relations
    for (group, task) in task_types.iter().enumerate() {
        match task.as_str() {
            "entities" => {
                // Handled in first pass
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
            "records" | "json_structures" => {
                decode_records(
                    boundary,
                    batch,
                    sample_idx,
                    encoder_states,
                    text_len,
                    &scored,
                    &specs,
                    group,
                    threshold,
                    include_confidence,
                    include_spans,
                    original_text,
                    starts_map,
                    ends_map,
                    &mut result,
                )?;
            }
            "relations" => {
                decode_relations(
                    boundary,
                    &text_states,
                    text_len,
                    h,
                    &query_states,
                    &specs,
                    &relation_specs,
                    &scored,
                    group,
                    threshold,
                    include_confidence,
                    include_spans,
                    original_text,
                    starts_map,
                    ends_map,
                    &all_entity_spans_by_type,
                    &all_canonical_entity_spans,
                    &mut result,
                )?;
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
    let group_specs: Vec<&QuerySpec> = specs.iter().filter(|s| s.group == group).collect();
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

/// Maximum-total-score non-overlapping subset of candidate spans.
///
/// Port of the Python shared resolver (`gliner2.inference.overlap`) with the
/// boundary-architecture default policy `flat`/`disallow`: exact-boundary
/// duplicates collapse to their highest-ranked representative, then weighted
/// interval scheduling maximizes total score with deterministic tie-breaking.
fn resolve_flat_spans(hits: Vec<(f32, usize, usize)>) -> Vec<(f32, usize, usize)> {
    if hits.is_empty() {
        return hits;
    }
    // Rank: descending score, ascending start/end, original order last.
    let mut ranked: Vec<(usize, (f32, usize, usize))> = hits.iter().cloned().enumerate().collect();
    ranked.sort_by(|a, b| {
        let (_, (sa, sta, ena)) = a;
        let (_, (sb, stb, enb)) = b;
        sb.partial_cmp(sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(sta.cmp(stb))
            .then(ena.cmp(enb))
            .then(a.0.cmp(&b.0))
    });
    let mut distinct: Vec<(usize, (f32, usize, usize))> = Vec::new();
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    for row in ranked {
        let key = (row.1.1, row.1.2);
        if seen.insert(key) {
            distinct.push(row);
        }
    }

    // Weighted interval scheduling over `distinct`.
    let n = distinct.len();
    let mut by_end: Vec<usize> = (0..n).collect();
    by_end.sort_by(|&a, &b| {
        let (_, (_, sa, ea)) = &distinct[a];
        let (_, (_, sb, eb)) = &distinct[b];
        ea.cmp(eb)
            .then(sa.cmp(sb))
            .then(
                distinct[b]
                    .1
                    .0
                    .partial_cmp(&distinct[a].1.0)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(a.cmp(&b))
    });
    let ends: Vec<usize> = by_end.iter().map(|&i| distinct[i].1.2).collect();

    // Predecessors: rightmost interval ending <= start(i).
    let mut preds = vec![usize::MAX; n];
    for (i, &idx) in by_end.iter().enumerate() {
        let start_i = distinct[idx].1.1;
        // bisect_right(ends, start_i, 0, i) - 1
        let mut lo = 0usize;
        let mut hi = i;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if ends[mid] <= start_i {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        preds[i] = lo.wrapping_sub(1); // usize::MAX when lo == 0
    }

    fn rank_key(
        row: &(usize, (f32, usize, usize)),
    ) -> (std::cmp::Reverse<u32>, usize, usize, usize) {
        // Approximate -score ordering via partial_cmp on f32 is unstable to
        // hash; compare through Reverse of bit pattern for total order.
        let bits = row.1.0.to_bits();
        (std::cmp::Reverse(bits), row.1.1, row.1.2, row.0)
    }

    // best[i] holds (total_score, selection as sorted-by-rank index vector).
    let mut best: Vec<(f64, Vec<usize>)> = Vec::with_capacity(n + 1);
    best.push((0.0, Vec::new()));
    let selection_key = |sel: &[usize]| -> Vec<(std::cmp::Reverse<u32>, usize, usize, usize)> {
        let mut rows: Vec<&(usize, (f32, usize, usize))> =
            sel.iter().map(|&i| &distinct[by_end[i]]).collect();
        rows.sort_by_key(|r| rank_key(r));
        rows.into_iter().map(rank_key).collect()
    };

    for i in 0..n {
        let idx = by_end[i];
        let (prev_score, prev_sel) = &best[preds[i].wrapping_add(1).min(best.len() - 1)];
        let mut with_sel = prev_sel.clone();
        with_sel.push(i);
        let with_score = prev_score + distinct[idx].1.0 as f64;
        let (without_score, without_sel) = &best[i];

        let take = if with_score > *without_score {
            true
        } else if with_score < *without_score {
            false
        } else if with_sel.len() != without_sel.len() {
            with_sel.len() > without_sel.len()
        } else {
            selection_key(&with_sel) < selection_key(without_sel)
        };
        if take {
            best.push((with_score, with_sel));
        } else {
            best.push((*without_score, without_sel.clone()));
        }
    }

    let final_selection = std::mem::take(&mut best[n].1);
    final_selection
        .into_iter()
        .map(|i| distinct[by_end[i]])
        .map(|(_, item)| item)
        .collect()
}

/// Re-score attribute-label queries at retained entity spans and attach the
/// per-group results to each decoded entry.
///
/// Port of the Python `_attach_entity_attributes`: unique (start, end) pairs
/// are scored against every configured attribute query via the explicit-spans
/// primitive, then reduced per group (sigmoid for multi_label, softmax
/// otherwise).
#[allow(clippy::type_complexity)]
fn attach_entity_attributes(
    boundary: &BoundaryModel,
    text_states: &Tensor,
    text_len: usize,
    query_states: &Tensor,
    specs: &[QuerySpec],
    attrs: &AttributesRuntime,
    decoded_entities: &mut [(String, Vec<JsonValue>, Vec<(usize, usize)>)],
) -> Result<()> {
    // Unique attribute rows in declaration order.
    let mut attr_rows: Vec<(String, String, usize)> = Vec::new(); // (label, prompt, qid)
    let mut row_by_prompt: HashMap<String, usize> = HashMap::new();
    for group in &attrs.groups {
        for label in &group.1.labels {
            let prompt = match attrs.prompt_by_label.get(label) {
                Some(p) => p.clone(),
                None => continue,
            };
            if row_by_prompt.contains_key(&prompt) {
                continue;
            }
            if let Some(qid) = specs.iter().position(|s| s.name == prompt) {
                row_by_prompt.insert(prompt.clone(), attr_rows.len());
                attr_rows.push((label.clone(), prompt, qid));
            }
        }
    }
    if attr_rows.is_empty() {
        return Ok(());
    }

    // Unique token-coordinate pairs across all retained entities.
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut pair_index: HashMap<(usize, usize), usize> = HashMap::new();
    for (_, _, coords) in decoded_entities.iter() {
        for &coord in coords {
            if let std::collections::hash_map::Entry::Vacant(e) = pair_index.entry(coord) {
                e.insert(pairs.len());
                pairs.push(coord);
            }
        }
    }
    if pairs.is_empty() {
        return Ok(());
    }

    // Query subset tensor: one row per attribute query.
    let h = query_states.dims()[1];
    let rows = query_states
        .to_vec2::<f32>()
        .map_err(|e| GlinerError::inference(format!("{e}")))?;
    let mut flat = Vec::with_capacity(attr_rows.len() * h);
    for (_, _, qid) in &attr_rows {
        flat.extend_from_slice(&rows[*qid]);
    }
    let attr_query_states = Tensor::from_slice(&flat, (attr_rows.len(), h), query_states.device())
        .map_err(|e| GlinerError::inference(format!("{e}")))?;

    let logits =
        boundary.score_explicit_spans(text_states, text_len, &attr_query_states, &pairs)?;

    // Attach per entity entry.
    for (entity_name, entries, coords) in decoded_entities.iter_mut() {
        for (entry, coord) in entries.iter_mut().zip(coords.iter()) {
            let obj = match entry.as_object_mut() {
                Some(o) => o,
                None => continue,
            };
            let column = match pair_index.get(coord) {
                Some(&c) => c,
                None => continue,
            };
            for (group_name, group) in &attrs.groups {
                if let Some(applies_to) = &group.applies_to
                    && !applies_to.contains(entity_name)
                {
                    continue;
                }
                let present: Vec<(&str, f32)> = group
                    .labels
                    .iter()
                    .filter_map(|l| {
                        let prompt = attrs.prompt_by_label.get(l)?;
                        let row = row_by_prompt.get(prompt.as_str())?;
                        Some((l.as_str(), logits.scores[column][*row]))
                    })
                    .collect();
                if present.is_empty() {
                    continue;
                }
                if group.multi_label {
                    let values: Vec<JsonValue> = present
                        .iter()
                        .filter_map(|&(label, value)| {
                            let prob = sigmoid(value);
                            (prob >= group.threshold)
                                .then(|| json!({"label": label, "confidence": prob}))
                        })
                        .collect();
                    obj.insert(group_name.clone(), JsonValue::Array(values));
                } else {
                    let probs: Vec<f32> = present.iter().map(|&(_, v)| sigmoid(v)).collect();
                    let best = probs
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    // Softmax over raw logits.
                    let max_logit = present
                        .iter()
                        .map(|&(_, v)| v)
                        .fold(f32::NEG_INFINITY, f32::max);
                    let sum_exp: f32 = present.iter().map(|&(_, v)| (v - max_logit).exp()).sum();
                    let best_prob = (present[best].1 - max_logit).exp() / sum_exp;
                    obj.insert(
                        group_name.clone(),
                        json!({"label": present[best].0, "confidence": best_prob}),
                    );
                    let _ = probs; // probabilities computed above only for argmax ordering
                }
            }
        }
    }
    Ok(())
}

fn text_word_positions(
    batch: &PreprocessedBatch,
    sample_idx: usize,
    _seq_len: usize,
) -> Result<Vec<usize>> {
    // The collator records first-subword positions per whitespace word.
    let count = *batch
        .text_word_counts
        .get(sample_idx)
        .ok_or_else(|| GlinerError::inference("missing text word count"))?;
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
    mapping.and_then(|m| m.get(word_idx)).copied().unwrap_or(0)
}

fn safe_slice(text: &str, start: usize, end: usize) -> String {
    let end = end.min(text.len());
    let start = start.min(end);
    text.get(start..end).map(str::to_string).unwrap_or_default()
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
