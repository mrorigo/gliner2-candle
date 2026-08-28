use gliner2_candle::inference::engine::GLiNER2;
use gliner2_candle::schema::builder::SchemaBuilder;

#[test]
#[ignore]
fn dbg_cls() {
    let engine = GLiNER2::from_pretrained("fastino/gliner2.5-small-v1").unwrap();
    let schema = SchemaBuilder::new()
        .classification("sentiment", vec!["positive".into(), "negative".into()])
        .done()
        .build()
        .unwrap();
    println!("DICT {}", schema.to_dict());
    let r = engine
        .extract(
            "I absolutely loved the movie, it was wonderful.",
            &schema,
            0.5,
            true,
            false,
            None,
        )
        .unwrap();
    println!("RESULT {r}");
}
