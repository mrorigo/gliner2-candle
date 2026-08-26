//! GLiNER2.5 performance smoke benchmark.

use gliner2_rs::schema::types::Schema;

#[test]
#[ignore = "benchmark; run with --ignored"]
fn bench_gliner25_latency() {
    let model_id = "fastino/gliner2.5-small-v1";
    let t0 = std::time::Instant::now();
    let engine = gliner2_rs::inference::engine::GLiNER2::from_pretrained(model_id)
        .expect("from_pretrained");
    println!("load time: {:?}", t0.elapsed());

    let schema = Schema::from_dict(&serde_json::json!({
        "entities": ["person", "organization", "location", "date", "product"]
    }))
    .expect("schema");

    let text = "Apple CEO Tim Cook announced the new iPhone 15 in Cupertino.";
    let _ = engine.extract(text, &schema, 0.5, false, false, None).expect("warmup");

    let n = 10;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        let _ = engine.extract(text, &schema, 0.5, false, false, None).unwrap();
    }
    println!("short ({} chars): {:.3}s/call", text.len(), t0.elapsed().as_secs_f64() / n as f64);

    // Batch of 8
    let texts: Vec<String> = (0..8).map(|i| format!("{text} Sample number {i} here.")).collect();
    let t0 = std::time::Instant::now();
    let _ = engine.batch_extract(&texts, &schema, 8, 0.5, 0, false, false, None).unwrap();
    println!("batch8 short: {:?}", t0.elapsed());

    // Medium doc (~450 words -> near max_len? small model max_len=4096 tokens; keep ~2000 words)
    let para = "Apple CEO Tim Cook announced the new iPhone 15 in Cupertino on Tuesday. \
        Google and Microsoft also sent representatives to the event in California.";
    let medium: String = std::iter::repeat(para).take(100).collect::<Vec<_>>().join(" ");
    let _ = engine.extract(&medium, &schema, 0.5, false, false, None).expect("warmup med");
    let t0 = std::time::Instant::now();
    let _ = engine.extract(&medium, &schema, 0.5, false, false, None).unwrap();
    println!("medium ({} chars): {:?} total", medium.len(), t0.elapsed());
}

/// Long-document auto-chunking: correctness + speed.
#[test]
#[ignore = "long-context benchmark"]
fn bench_long_context() {
    let engine = gliner2_rs::inference::engine::GLiNER2::from_pretrained("fastino/gliner2.5-small-v1")
        .expect("from_pretrained");
    let schema = gliner2_rs::schema::types::Schema::from_dict(&serde_json::json!({
        "entities": ["person", "organization", "location", "date", "product"]
    }))
    .expect("schema");

    let para = "Apple CEO Tim Cook announced the new iPhone 15 in Cupertino on Tuesday. \
        Google and Microsoft also sent representatives to the event in California.";
    let text: String = std::iter::repeat(para).take(100).collect::<Vec<_>>().join(" ");
    println!("words={}", text.split_whitespace().count());

    let t0 = std::time::Instant::now();
    let result = engine
        .extract(&text, &schema, 0.5, true, true, None)
        .expect("extract");
    println!("chunked extraction: {:?}", t0.elapsed());

    if let Some(o) = result["entities"].as_object() {
        for (k, v) in o {
            println!("label {k}: {}", v.as_array().map(|a| a.len()).unwrap_or(0));
        }
    } else {
        println!("entities is not an object: {result}");
    }
    let empty = Vec::new();
    let orgs = result["entities"]["organization"].as_array().unwrap_or(&empty);
    let persons = result["entities"]["person"].as_array().unwrap_or(&empty);
    let locs = result["entities"]["location"].as_array().unwrap_or(&empty);
    fn texts_of(v: &[serde_json::Value]) -> Vec<String> {
        v.iter().map(|e| e["text"].as_str().unwrap().to_string()).collect()
    }
    for (name, arr) in [("organizations", orgs), ("persons", persons), ("locations", locs)] {
        let mut t = texts_of(arr);
        t.sort();
        t.dedup();
        println!("{name} unique count={}: {:?}", t.len(), &t[..t.len().min(8)]);
    }

    // Sanity: every span must slice the original document cleanly.
    for (key, arr) in [("organization", orgs), ("person", persons), ("location", locs)] {
        for e in arr {
            let s = e["start"].as_u64().unwrap() as usize;
            let en = e["end"].as_u64().unwrap() as usize;
            assert_eq!(&text[s..en], e["text"].as_str().unwrap(), "{key} span mismatch");
        }
    }
}
