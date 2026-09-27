use gliner2_candle::GLiNER2;
use gliner2_candle::schema::builder::SchemaBuilder;
use gliner2_candle::schema::types::{FieldCardinality, FieldDtype, StructureMode};

#[test]
fn scratch_natural_record() {
    let engine = GLiNER2::from_pretrained("fastino/gliner2.5-small-v1").expect("load");

    let schema = SchemaBuilder::new()
        .structure("decision")
        .mode(StructureMode::Natural)
        .anchor("person")
        .field("person")
        .dtype(FieldDtype::Str)
        .cardinality(FieldCardinality::RequiredOne)
        .done_field()
        .field("decision")
        .dtype(FieldDtype::Str)
        .cardinality(FieldCardinality::RequiredOne)
        .done_field()
        .field("design")
        .dtype(FieldDtype::Str)
        .cardinality(FieldCardinality::RequiredOne)
        .done_field()
        .done_structure()
        .build()
        .unwrap();

    let text = "Alice selected the red design, while Bob rejected the blue design.";
    let r = engine
        .extract(text, &schema, 0.4, false, false, None)
        .unwrap();
    println!("RESULT: {r}");
}
