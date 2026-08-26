//! Long-document chunking utilities.
//!
//! Splits text into overlapping word windows, remaps chunk-local spans back to
//! document character offsets, and merges duplicate predictions across chunks.
//! Mirrors the semantics of `gliner2.inference.chunking` in the Python repo.

use serde_json::Value as JsonValue;

use crate::error::{GlinerError, Result};

/// Default maximum words per chunk (matches the Python default).
pub const DEFAULT_CHUNK_SIZE: usize = 384;
/// Default word overlap between adjacent chunks (matches the Python default).
pub const DEFAULT_CHUNK_OVERLAP: usize = 64;

/// A chunk of text with offsets into the original document.
#[derive(Debug, Clone, PartialEq)]
pub struct TextChunk {
    /// Chunk text (original casing), `text[start_char..end_char]`.
    pub text: String,
    /// Start character offset in the original document.
    pub start_char: usize,
    /// End character offset (exclusive) in the original document.
    pub end_char: usize,
    /// Index of the first word (inclusive) in the original word sequence.
    pub start_word: usize,
    /// Index of the last word (exclusive) in the original word sequence.
    pub end_word: usize,
}

/// Policy for resolving spans that overlap after merging chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MergePolicy {
    /// Keep every distinct span (Python `allow`).
    #[default]
    Allow,
    /// Maximum-total-confidence non-overlapping span selection
    /// (weighted interval scheduling; Python `disallow`).
    HighestConfidence,
    /// Drop spans strictly contained by another span (Python `longest`).
    LongestSpan,
}

/// Split `text` into overlapping word windows on whitespace boundaries.
///
/// Character offsets always index the original text. An empty document yields
/// a single empty chunk covering the whole text.
pub fn split_text_into_chunks(
    text: &str,
    chunk_size: usize,
    chunk_overlap: usize,
) -> Result<Vec<TextChunk>> {
    if chunk_size == 0 {
        return Err(GlinerError::inference("chunk_size must be greater than 0"));
    }
    if chunk_overlap >= chunk_size {
        return Err(GlinerError::inference(
            "chunk_overlap must be smaller than chunk_size",
        ));
    }

    // Whitespace word tokens with char offsets into the original text.
    let mut words: Vec<(usize, usize)> = Vec::new();
    let mut in_word = false;
    let mut start = 0usize;
    for (i, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if in_word {
                words.push((start, i));
                in_word = false;
            }
        } else if !in_word {
            start = i;
            in_word = true;
        }
    }
    if in_word {
        words.push((start, text.len()));
    }

    if words.is_empty() {
        return Ok(vec![TextChunk {
            text: text.to_string(),
            start_char: 0,
            end_char: text.len(),
            start_word: 0,
            end_word: 0,
        }]);
    }

    let step = chunk_size - chunk_overlap;
    let mut chunks = Vec::new();
    let mut start_word = 0usize;
    loop {
        let end_word = (start_word + chunk_size).min(words.len());
        let start_char = words[start_word].0;
        let end_char = words[end_word - 1].1;
        chunks.push(TextChunk {
            text: text[start_char..end_char].to_string(),
            start_char,
            end_char,
            start_word,
            end_word,
        });
        if end_word == words.len() {
            break;
        }
        start_word += step;
    }
    Ok(chunks)
}

fn is_span_dict(value: &serde_json::Map<String, JsonValue>) -> bool {
    value.contains_key("start") && value.contains_key("end")
}

/// Recursively shift chunk-local span offsets by `chunk.start_char`, refreshing
/// span `text` from the original document when offsets are valid.
pub fn remap_result_spans(result: &mut JsonValue, original_text: &str, chunk: &TextChunk) {
    match result {
        JsonValue::Array(items) => {
            for item in items {
                remap_result_spans(item, original_text, chunk);
            }
        }
        JsonValue::Object(map) => {
            for (_, value) in map.iter_mut() {
                remap_result_spans(value, original_text, chunk);
            }
            if is_span_dict(map) {
                let start = map.get("start").and_then(|v| v.as_u64()).unwrap_or(0) as usize
                    + chunk.start_char;
                let end =
                    map.get("end").and_then(|v| v.as_u64()).unwrap_or(0) as usize + chunk.start_char;
                map.insert("start".to_string(), JsonValue::from(start));
                map.insert("end".to_string(), JsonValue::from(end));
                if end <= original_text.len() && start <= end {
                    map.insert(
                        "text".to_string(),
                        JsonValue::from(original_text[start..end].to_string()),
                    );
                }
            }
        }
        _ => {}
    }
}

struct SpanItem {
    start: usize,
    end: usize,
    confidence: f32,
    value: JsonValue,
}

/// Merge per-chunk extraction results into one document-level result.
///
/// Spans with identical `(start, end)` boundaries collapse to their
/// highest-confidence representative; remaining overlaps are resolved with
/// `policy`. Non-span values (relations, classifications) dedupe on canonical
/// JSON equality keeping the highest-confidence copy.
pub fn merge_chunk_results(
    original_text: &str,
    chunks: &[TextChunk],
    mut chunk_results: Vec<JsonValue>,
    include_confidence: bool,
    include_spans: bool,
    policy: MergePolicy,
) -> Result<JsonValue> {
    if chunks.len() != chunk_results.len() {
        return Err(GlinerError::inference(
            "chunks and chunk_results must have the same length",
        ));
    }
    for (chunk, result) in chunks.iter().zip(chunk_results.iter_mut()) {
        remap_result_spans(result, original_text, chunk);
    }

    let merged = merge_objects(&chunk_results, policy)?;
    Ok(strip_span_metadata(&merged, include_confidence, include_spans))
}

fn merge_objects(results: &[JsonValue], policy: MergePolicy) -> Result<JsonValue> {
    let mut out = serde_json::Map::new();
    // Preserve first-seen key order across chunks.
    let mut keys: Vec<String> = Vec::new();
    for result in results {
        if let JsonValue::Object(map) = result {
            for key in map.keys() {
                if !keys.contains(key) {
                    keys.push(key.clone());
                }
            }
        }
    }
    for key in keys {
        let values: Vec<JsonValue> = results
            .iter()
            .filter_map(|r| r.get(&key).cloned())
            .collect();
        let merged_value = match key.as_str() {
            "entities" => merge_entity_values(&values, policy),
            _ => merge_generic_values(&values, policy),
        };
        out.insert(key, merged_value);
    }
    Ok(JsonValue::Object(out))
}

fn merge_entity_values(values: &[JsonValue], policy: MergePolicy) -> JsonValue {
    let mut labels: Vec<String> = Vec::new();
    for value in values {
        if let JsonValue::Object(map) = value {
            for label in map.keys() {
                if !labels.contains(label) {
                    labels.push(label.clone());
                }
            }
        }
    }
    let mut out = serde_json::Map::new();
    for label in labels {
        let mut items: Vec<JsonValue> = Vec::new();
        for value in values {
            if let Some(v) = value.get(&label) {
                match v {
                    JsonValue::Array(arr) => items.extend(arr.iter().cloned()),
                    other if !other.is_null() => items.push(other.clone()),
                    _ => {}
                }
            }
        }
        out.insert(label, dedupe_items(items, policy));
    }
    JsonValue::Object(out)
}

fn merge_generic_values(values: &[JsonValue], policy: MergePolicy) -> JsonValue {
    let non_empty: Vec<&JsonValue> = values
        .iter()
        .filter(|v| !v.is_null() && *v != &JsonValue::Object(Default::default()) && *v != &JsonValue::Array(vec![]))
        .collect();
    if non_empty.is_empty() {
        return values.first().cloned().unwrap_or(JsonValue::Null);
    }
    // Classification dicts: highest-confidence wins.
    if non_empty.iter().all(|v| {
        v.as_object()
            .map(|m| m.contains_key("confidence"))
            .unwrap_or(false)
    }) {
        return non_empty
            .iter()
            .max_by(|a, b| {
                let ca = a["confidence"].as_f64().unwrap_or(0.0);
                let cb = b["confidence"].as_f64().unwrap_or(0.0);
                ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .map(|v| v.clone())
            .unwrap_or_else(|| (*non_empty[0]).clone());
    }
    if non_empty.iter().all(|v| v.is_array()) {
        let items: Vec<JsonValue> = non_empty
            .iter()
            .flat_map(|v| v.as_array().unwrap().iter().cloned())
            .collect();
        return dedupe_items(items, policy);
    }
    if non_empty.iter().all(|v| v.is_object()) {
        let owned: Vec<JsonValue> = non_empty.iter().map(|v| (*v).clone()).collect();
        let refs: Vec<&JsonValue> = owned.iter().collect();
        return merge_nested_dicts(&refs, policy);
    }
    (*non_empty[0]).clone()
}

fn merge_nested_dicts(values: &[&JsonValue], policy: MergePolicy) -> JsonValue {
    let mut keys: Vec<String> = Vec::new();
    for value in values {
        if let JsonValue::Object(map) = value {
            for key in map.keys() {
                if !keys.contains(key) {
                    keys.push(key.clone());
                }
            }
        }
    }
    let mut out = serde_json::Map::new();
    for key in keys {
        let vals: Vec<JsonValue> = values
            .iter()
            .filter_map(|v| v.get(&key).cloned())
            .collect();
        out.insert(key, merge_generic_values(&vals, policy));
    }
    JsonValue::Object(out)
}

fn canonical_key(value: &JsonValue) -> String {
    match value {
        JsonValue::Object(map) => {
            let mut parts: Vec<String> = map
                .iter()
                .filter(|(k, _)| k.as_str() != "confidence")
                .map(|(k, v)| format!("{k}={}", canonical_key(v)))
                .collect();
            parts.sort();
            format!("{{{}}}", parts.join(","))
        }
        JsonValue::Array(items) => {
            let parts: Vec<String> = items.iter().map(canonical_key).collect();
            format!("[{}]", parts.join(","))
        }
        other => other.to_string(),
    }
}

fn representative_confidence(value: &JsonValue) -> f32 {
    fn deepest_confidence(value: &JsonValue) -> Option<f32> {
        match value {
            JsonValue::Object(map) => {
                if let Some(c) = map.get("confidence").and_then(|v| v.as_f64()) {
                    return Some(c as f32);
                }
                map.values().find_map(deepest_confidence)
            }
            JsonValue::Array(items) => items.iter().find_map(deepest_confidence),
            _ => None,
        }
    }
    deepest_confidence(value).unwrap_or(0.0)
}

fn item_start(item: &SpanItem) -> usize {
    item.start
}
fn item_end(item: &SpanItem) -> usize {
    item.end
}
fn item_score(item: &SpanItem) -> f32 {
    item.confidence
}

fn resolve_overlaps(mut items: Vec<SpanItem>, policy: MergePolicy) -> Vec<SpanItem> {
    // Collapse identical-boundary duplicates to the highest-score one.
    items.sort_by(|a, b| {
        item_score(b)
            .partial_cmp(&item_score(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(item_start(a).cmp(&item_start(b)))
            .then(item_end(a).cmp(&item_end(b)))
    });
    let mut distinct: Vec<SpanItem> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for item in items {
        if seen.insert((item.start, item.end)) {
            distinct.push(item);
        }
    }
    match policy {
        MergePolicy::Allow => distinct,
        MergePolicy::LongestSpan => {
            let keep_mask: Vec<bool> = distinct
                .iter()
                .enumerate()
                .map(|(i, cand)| {
                    !distinct.iter().enumerate().any(|(j, other)| {
                        j != i
                            && other.start <= cand.start
                            && cand.end <= other.end
                            && (other.start < cand.start || cand.end < other.end)
                    })
                })
                .collect();
            distinct.into_iter().zip(keep_mask).filter_map(|(it, k)| k.then_some(it)).collect()
        }
        MergePolicy::HighestConfidence => {
            // Weighted interval scheduling maximizing total confidence.
            let n = distinct.len();
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by_key(|&i| (distinct[i].end, distinct[i].start));
            // predecessors[i]: last interval in `order` ending <= start of order[i]
            const NONE: usize = usize::MAX;
            // pred_rank[rank]: rank (in `order` coords) of the last interval
            // whose end <= start of order[rank], or NONE.
            let mut pred_rank = vec![NONE; n];
            for (rank, &i) in order.iter().enumerate() {
                let mut lo = 0usize;
                let mut hi = rank;
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    if distinct[order[mid]].end <= distinct[i].start {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                pred_rank[rank] = if lo == 0 { NONE } else { lo - 1 };
            }
            // DP over ranks: best_score[r] = best total using first r intervals.
            let mut best_score = vec![0.0f32; n + 1];
            let mut take = vec![false; n];
            for rank in 0..n {
                let prev_rank = if pred_rank[rank] == NONE { 0 } else { pred_rank[rank] + 1 };
                let with = best_score[prev_rank] + distinct[order[rank]].confidence;
                let without = best_score[rank];
                if with > without {
                    best_score[rank + 1] = with;
                    take[rank] = true;
                } else {
                    best_score[rank + 1] = without;
                }
            }
            // Reconstruct selection.
            let mut selected = std::collections::HashSet::new();
            let mut rank = n;
            while rank > 0 {
                if take[rank - 1] {
                    selected.insert(order[rank - 1]);
                    rank = if pred_rank[rank - 1] == NONE { 0 } else { pred_rank[rank - 1] + 1 };
                } else {
                    rank -= 1;
                }
            }
            distinct
                .into_iter()
                .enumerate()
                .filter_map(|(i, it)| selected.contains(&i).then_some(it))
                .collect()
        }
    }
}

fn extract_spans(items: Vec<JsonValue>, policy: MergePolicy) -> Vec<SpanItem> {
    let spans: Vec<SpanItem> = items
        .into_iter()
        .filter_map(|item| {
            let obj = item.as_object()?;
            if !is_span_dict(obj) {
                return None;
            }
            Some(SpanItem {
                start: obj.get("start").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                end: obj.get("end").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                confidence: obj
                    .get("confidence")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32,
                value: item,
            })
        })
        .collect();
    resolve_overlaps(spans, policy)
}

fn dedupe_items(items: Vec<JsonValue>, policy: MergePolicy) -> JsonValue {
    let mut span_items: Vec<JsonValue> = Vec::new();
    let mut others: Vec<JsonValue> = Vec::new();
    for item in items {
        let is_span = item
            .as_object()
            .map(is_span_dict)
            .unwrap_or(false);
        if is_span {
            span_items.push(item);
        } else {
            others.push(item);
        }
    }

    let mut out: Vec<JsonValue> = Vec::new();
    for span in extract_spans(span_items, policy) {
        out.push(span.value);
    }
    out.sort_by(|a, b| {
        let sa = a["start"].as_u64().unwrap_or(0);
        let sb = b["start"].as_u64().unwrap_or(0);
        sa.cmp(&sb)
            .then(a["end"].as_u64().unwrap_or(0).cmp(&b["end"].as_u64().unwrap_or(0)))
    });

    // Non-span items: canonical-key dedupe keeping the higher-confidence copy.
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for item in others {
        let key = canonical_key(&item);
        match seen.get(&key) {
            None => {
                seen.insert(key, out.len());
                out.push(item);
            }
            Some(&idx) => {
                if representative_confidence(&item)
                    > representative_confidence(&out[idx])
                {
                    out[idx] = item;
                }
            }
        }
    }
    JsonValue::Array(out)
}

fn strip_span_metadata(value: &JsonValue, include_confidence: bool, include_spans: bool) -> JsonValue {
    match value {
        JsonValue::Array(items) => {
            JsonValue::Array(items.iter().map(|i| strip_span_metadata(i, include_confidence, include_spans)).collect())
        }
        JsonValue::Object(map) => {
            if is_span_dict(map) && !include_spans {
                // Reduce span dicts to text (+confidence when requested).
                let mut out = serde_json::Map::new();
                if let Some(t) = map.get("text") {
                    out.insert("text".to_string(), t.clone());
                }
                if include_confidence && let Some(c) = map.get("confidence") {
                    out.insert("confidence".to_string(), c.clone());
                }
                return JsonValue::Object(out);
            }
            let mut out = serde_json::Map::new();
            for (k, v) in map {
                if k == "confidence" && !include_confidence {
                    continue;
                }
                out.insert(k.clone(), strip_span_metadata(v, include_confidence, include_spans));
            }
            JsonValue::Object(out)
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_split_basic() {
        let text = "one two three four five";
        let chunks = split_text_into_chunks(text, 3, 1).unwrap();
        // step = 3 - 1 = 2 -> windows [0..3], [2..5]
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "one two three");
        assert_eq!(chunks[1].text, "three four five");
        assert_eq!(chunks[0].start_word, 0);
        assert_eq!(chunks[0].end_word, 3);
        assert_eq!(chunks.last().unwrap().end_word, 5);
    }

    #[test]
    fn test_split_offsets() {
        let text = "alpha beta  gamma"; // double space
        let chunks = split_text_into_chunks(text, 2, 1).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "alpha beta");
        assert_eq!(chunks[1].text, "beta  gamma");
        assert_eq!(chunks[1].start_char, 6);
        // offsets index original text including the double space
        assert!(text[chunks[1].start_char..].starts_with("beta"));
    }

    #[test]
    fn test_split_single_chunk_when_short() {
        let chunks = split_text_into_chunks("hello world", 384, 64).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hello world");
    }

    #[test]
    fn test_split_empty() {
        let chunks = split_text_into_chunks("", 384, 64).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "");
        assert_eq!(chunks[0].start_word, 0);
        assert_eq!(chunks[0].end_word, 0);
    }

    #[test]
    fn test_split_invalid_args() {
        assert!(split_text_into_chunks("a", 0, 0).is_err());
        assert!(split_text_into_chunks("a", 4, 4).is_err());
    }

    fn sample_result(start: usize, end: usize, text_: &str, conf: f64) -> JsonValue {
        json!({"entities": {"person": [{"start": start, "end": end, "text": text_, "confidence": conf}]}})
    }

    #[test]
    fn test_remap_and_merge_entities() {
        let text = "Tim Cook met Sundar Pichai at Apple.";
        let chunks = split_text_into_chunks(text, 4, 2).unwrap();
        let results: Vec<JsonValue> = chunks
            .iter()
            .map(|c| {
                // pretend each chunk found its first word as a person
                let local = &c.text.split_whitespace().next().unwrap().to_string();
                let s = 0;
                let e = local.len();
                sample_result(s, e, local, 0.9)
            })
            .collect();
        let merged = merge_chunk_results(text, &chunks, results, true, true, MergePolicy::Allow).unwrap();
        let people = merged["entities"]["person"].as_array().unwrap();
        assert!(!people.is_empty());
        // all remapped to doc offsets, text matches original slice
        for p in people {
            let s = p["start"].as_u64().unwrap() as usize;
            let e = p["end"].as_u64().unwrap() as usize;
            assert_eq!(&text[s..e], p["text"].as_str().unwrap());
        }
    }

    #[test]
    fn test_dedupe_identical_boundaries_keeps_highest_confidence() {
        let text = "Apple is a company.";
        let chunks = vec![TextChunk {
            text: text.to_string(),
            start_char: 0,
            end_char: text.len(),
            start_word: 0,
            end_word: 4,
        }];
        let results = vec![json!({"entities": {"person": [
            {"start": 0, "end": 5, "text": "Apple", "confidence": 0.7},
            {"start": 0, "end": 5, "text": "Apple", "confidence": 0.95},
        ]}})];
        let merged =
            merge_chunk_results(text, &chunks, results, true, true, MergePolicy::Allow).unwrap();
        let people = merged["entities"]["person"].as_array().unwrap();
        assert_eq!(people.len(), 1);
        assert!((people[0]["confidence"].as_f64().unwrap() - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_policy_disallow_resolves_overlap() {
        let text = "Chief Executive Officer Tim Cook";
        let chunks = vec![TextChunk {
            text: text.to_string(),
            start_char: 0,
            end_char: text.len(),
            start_word: 0,
            end_word: 5,
        }];
        // "Chief Executive Officer Tim Cook": Tim starts at 24.
        let results = vec![json!({"entities": {"person": [
            {"start": 0, "end": 27, "text": "Chief Executive Officer Tim", "confidence": 0.8},
            {"start": 24, "end": 32, "text": "Tim Cook", "confidence": 0.9},
            {"start": 24, "end": 26, "text": "Ti", "confidence": 0.85},
        ]}})];
        let merged =
            merge_chunk_results(text, &chunks, results, true, true, MergePolicy::HighestConfidence)
                .unwrap();
        let people = merged["entities"]["person"].as_array().unwrap();
        // Overlapping candidates: (0,27)=0.8, (24,32)=0.9, (24,26)=0.85 —
        // every pair overlaps, so the best non-overlapping set is the single
        // highest-confidence span "Tim Cook".
        assert_eq!(people.len(), 1);
        assert_eq!(people[0]["text"], "Tim Cook");
    }

    #[test]
    fn test_policy_longest_drops_contained() {
        let text = "New York City is big.";
        let chunks = vec![TextChunk {
            text: text.to_string(),
            start_char: 0,
            end_char: text.len(),
            start_word: 0,
            end_word: 4,
        }];
        let results = vec![json!({"entities": {"location": [
            {"start": 0, "end": 13, "text": "New York City", "confidence": 0.8},
            {"start": 0, "end": 8, "text": "New York", "confidence": 0.9},
        ]}})];
        let merged =
            merge_chunk_results(text, &chunks, results, true, true, MergePolicy::LongestSpan)
                .unwrap();
        let locs = merged["entities"]["location"].as_array().unwrap();
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0]["text"], "New York City");
    }

    #[test]
    fn test_strip_metadata() {
        let text = "Apple.";
        let chunks = vec![TextChunk {
            text: text.to_string(),
            start_char: 0,
            end_char: text.len(),
            start_word: 0,
            end_word: 1,
        }];
        let results = vec![sample_result(0, 5, "Apple", 0.9)];
        let merged = merge_chunk_results(text, &chunks, results, false, false, MergePolicy::Allow)
            .unwrap();
        let person = &merged["entities"]["person"];
        let arr = person.as_array().unwrap();
        assert_eq!(arr[0], json!({"text": "Apple"}));
    }

    #[test]
    fn test_classification_merge_picks_max_confidence() {
        let text = "happy days";
        let chunks = vec![
            TextChunk { text: text.into(), start_char: 0, end_char: text.len(), start_word: 0, end_word: 2 },
            TextChunk { text: text.into(), start_char: 0, end_char: text.len(), start_word: 0, end_word: 2 },
        ];
        let mk = |c: f64| json!({"sentiment": {"label": "positive", "confidence": c}});
        let results = vec![mk(0.6), mk(0.9)];
        let merged = merge_chunk_results(text, &chunks, results, true, false, MergePolicy::Allow)
            .unwrap();
        assert!((merged["sentiment"]["confidence"].as_f64().unwrap() - 0.9).abs() < 1e-9);
    }
}

