use gliner2_candle::inference::engine::GLiNER2;
use gliner2_candle::schema::builder::SchemaBuilder;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_id = env::var("GLINER_MODEL").unwrap_or_else(|_| "fastino/gliner2.5-small-v1".to_string());
    println!("=== GLiNER2 Candle ===");
    println!("Model: {model_id}\n");

    let engine = GLiNER2::from_pretrained(&model_id)?;
    let text = "Apple CEO Tim Cook announced Zürich results with 500 requests/second.";
    let schema = SchemaBuilder::new()
        .entities(vec!["person".to_string(), "company".to_string(), "metric".to_string()])
        .build()?;
    let res = engine.extract(text, &schema, 0.4, true, true, None)?;
    println!("Extraction:\n{}", serde_json::to_string_pretty(&res)?);

    Ok(())
}
