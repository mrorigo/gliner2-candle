#[test]
#[ignore = "long input quality bisect"]
fn long_quality_bisect() {
    let engine = gliner2_rs::inference::engine::GLiNER2::from_pretrained("fastino/gliner2.5-small-v1").unwrap();
    let schema = gliner2_rs::schema::types::Schema::from_dict(&serde_json::json!({
        "entities": ["person", "organization", "location", "date"]
    })).unwrap();
    let para = "Apple CEO Tim Cook announced the new iPhone 15 in Cupertino on Tuesday. \
        Google and Microsoft also sent representatives to the event in California.";
    for take in [5usize, 10, 20, 40, 80] {
        let text: String = std::iter::repeat(para).take(take).collect::<Vec<_>>().join(" ");
        let r = engine.extract(&text, &schema, 0.5, true, false, None).unwrap();
        let mut counts = Vec::new();
        if let Some(o) = r["entities"].as_object() {
            for (k, v) in o {
                counts.push(format!("{k}={}", v.as_array().map(|a| a.len()).unwrap_or(0)));
            }
        }
        counts.sort();
        println!("take={take} words={} -> {:?}", text.split_whitespace().count(), counts);
    }
}
