//! Task probe binary for testing model extraction capabilities across GLiNER2 and GLiNER2.5.
//!
//! Supports `GLINER_MODEL` env var (default: `fastino/gliner2.5-small-v1`).

use gliner2_candle::inference::engine::GLiNER2;
use gliner2_candle::schema::builder::SchemaBuilder;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_id =
        env::var("GLINER_MODEL").unwrap_or_else(|_| "fastino/gliner2.5-small-v1".to_string());
    println!("=== GLiNER2 Task Probe ===");
    println!("Model: {model_id}\n");

    let engine = GLiNER2::from_pretrained(&model_id)?;

    // 1. Entity extraction probe (with punctuation & compound metrics)
    println!("--- 1. Entity Extraction Probe ---");
    let text =
        "Server sustained 500 requests/second with 100ms latency. Apple CEO Tim Cook was pleased.";
    let entity_schema = SchemaBuilder::new()
        .entities(vec![
            "metric".to_string(),
            "person".to_string(),
            "company".to_string(),
        ])
        .build()?;
    let entity_res = engine.extract(text, &entity_schema, 0.4, true, true, None)?;
    println!("Result: {}\n", serde_json::to_string_pretty(&entity_res)?);

    // Verify span provenance
    if let Some(entities) = entity_res.get("entities").and_then(|v| v.as_object()) {
        for (etype, list) in entities {
            if let Some(arr) = list.as_array() {
                for item in arr {
                    if let (Some(t), Some(s), Some(e)) = (
                        item.get("text").and_then(|v| v.as_str()),
                        item.get("start").and_then(|v| v.as_u64()),
                        item.get("end").and_then(|v| v.as_u64()),
                    ) {
                        let slice = &text[s as usize..e as usize];
                        assert_eq!(
                            slice, t,
                            "Span text provenance mismatch for {etype}: '{t}' vs slice '{slice}'"
                        );
                        println!("  [Verified] {etype} '{t}' == text[{s}..{e}]");
                    }
                }
            }
        }
    }

    // 2. Classification probe
    println!("\n--- 2. Classification Probe ---");
    let cls_schema = SchemaBuilder::new()
        .classification("sentiment", vec!["positive".into(), "negative".into()])
        .done()
        .build()?;
    let cls_res = engine.extract(
        "The new release performs exceptionally well!",
        &cls_schema,
        0.5,
        true,
        false,
        None,
    )?;
    println!("Result: {}\n", serde_json::to_string_pretty(&cls_res)?);

    // 3. Structure probe
    println!("--- 3. JSON Structure Probe ---");
    let struct_schema = SchemaBuilder::new()
        .entities(vec!["product".to_string(), "company".to_string()])
        .structure("product_info")
        .field("name")
        .done_field()
        .field("company")
        .done_field()
        .done_structure()
        .build()?;
    let struct_res = engine.extract(
        "Apple launched iPhone in Cupertino.",
        &struct_schema,
        0.4,
        false,
        false,
        None,
    )?;
    println!("Result: {}\n", serde_json::to_string_pretty(&struct_res)?);

    // 4. Relation probe
    println!("--- 4. Relation Probe ---");
    let rel_schema = SchemaBuilder::new()
        .entities(vec!["person".to_string(), "company".to_string()])
        .relation("works_for")
        .done()
        .build()?;
    let rel_res = engine.extract(
        "Tim Cook works for Apple in Cupertino.",
        &rel_schema,
        0.5,
        false,
        false,
        None,
    )?;
    println!("Result: {}\n", serde_json::to_string_pretty(&rel_res)?);

    println!("All task probes executed successfully!");
    Ok(())
}
