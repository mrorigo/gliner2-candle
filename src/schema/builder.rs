// Rust guideline compliant 2026-04-03
//! Schema builder for GLiNER2.
//!
//! This module provides a fluent API for building extraction schemas,
//! matching the Python `Schema` class's builder pattern. It allows
//! users to construct complex schemas with entities, classifications,
//! structures, and relations using method chaining.
//!
//! # Example
//!
//! ```
//! use gliner2_candle::schema::builder::SchemaBuilder;
//! use gliner2_candle::schema::types::{EntityDef, ClassificationDef, FieldDef, RelationDef, FieldDtype};
//!
//! let schema = SchemaBuilder::new()
//!     .entity("person").description("Names of people").done()
//!     .entity("company").done()
//!     .classification("sentiment", vec!["positive".to_string(), "negative".to_string(), "neutral".to_string()])
//!         .multi_label(false)
//!         .threshold(0.5)
//!         .done()
//!     .structure("product_info")
//!         .field("name").dtype(FieldDtype::Str).done_field()
//!         .field("price").done_field()
//!         .done_structure()
//!     .relation("works_for").done()
//!     .build()
//!     .unwrap();
//! ```

use crate::error::Result;
use crate::schema::types::*;
use std::collections::HashMap;

/// Builder for constructing extraction schemas with a fluent API.
///
/// This builder mirrors the Python `Schema` class and provides
/// method chaining for convenient schema construction.
#[derive(Debug, Default)]
pub struct SchemaBuilder {
    schema: Schema,
    entity_order: Vec<String>,
    relation_order: Vec<String>,
    field_orders: HashMap<String, Vec<String>>,
    field_metadata: HashMap<String, FieldMetadata>,
    entity_metadata: HashMap<String, EntityMetadata>,
    relation_metadata: HashMap<String, RelationMetadata>,
    active_structure: Option<Box<StructureBuilder>>,
}

/// Metadata for a field.
#[derive(Debug, Clone)]
pub struct FieldMetadata {
    pub dtype: FieldDtype,
    pub threshold: Option<f32>,
    pub choices: Option<Vec<String>>,
    pub validators: Option<Vec<RegexValidator>>,
}

/// Metadata for an entity.
#[derive(Debug, Clone)]
pub struct EntityMetadata {
    pub dtype: FieldDtype,
    pub threshold: Option<f32>,
}

/// Metadata for a relation.
#[derive(Debug, Clone)]
pub struct RelationMetadata {
    pub threshold: Option<f32>,
}

impl SchemaBuilder {
    /// Create a new empty schema builder.
    pub fn new() -> Self {
        Self::default()
    }

    // -------------------------------------------------------------------------
    // Entity Methods
    // -------------------------------------------------------------------------

    /// Add a single entity type.
    ///
    /// # Arguments
    ///
    /// * `name` - The entity type name.
    ///
    /// # Returns
    ///
    /// An `EntityBuilder` for configuring the entity.
    ///
    /// # Example
    ///
    /// ```
    /// use gliner2_candle::schema::builder::SchemaBuilder;
    ///
    /// let schema = SchemaBuilder::new()
    ///     .entity("person")
    ///         .description("Names of people")
    ///         .threshold(0.6)
    ///         .done()
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn entity(mut self, name: impl Into<String>) -> EntityBuilder {
        let name = name.into();
        if !self.entity_order.contains(&name) {
            self.entity_order.push(name.clone());
        }
        EntityBuilder {
            parent: self,
            name,
            description: None,
            dtype: FieldDtype::List,
            threshold: None,
        }
    }

    /// Add multiple entity types at once.
    ///
    /// # Arguments
    ///
    /// * `names` - A vector of entity type names.
    ///
    /// # Example
    ///
    /// ```
    /// use gliner2_candle::schema::builder::SchemaBuilder;
    ///
    /// let schema = SchemaBuilder::new()
    ///     .entities(vec!["person".to_string(), "company".to_string(), "location".to_string()])
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn entities(mut self, names: Vec<String>) -> Self {
        for name in names {
            if !self.entity_order.contains(&name) {
                self.entity_order.push(name.clone());
            }
            self.schema.entities.push(EntityDef::new(name));
        }
        self
    }

    /// Add entities with descriptions from a HashMap.
    ///
    /// # Arguments
    ///
    /// * `entities` - A HashMap mapping entity names to descriptions.
    ///
    /// # Example
    ///
    /// ```
    /// use gliner2_candle::schema::builder::SchemaBuilder;
    /// use std::collections::HashMap;
    ///
    /// let mut entities = HashMap::new();
    /// entities.insert("person".to_string(), "Names of people".to_string());
    /// entities.insert("company".to_string(), "Organization names".to_string());
    ///
    /// let schema = SchemaBuilder::new()
    ///     .entities_with_descriptions(entities)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn entities_with_descriptions(mut self, entities: HashMap<String, String>) -> Self {
        for (name, desc) in entities {
            if !self.entity_order.contains(&name) {
                self.entity_order.push(name.clone());
            }
            self.schema
                .entities
                .push(EntityDef::new(&name).with_description(&desc));
            self.schema.entity_descriptions.insert(name.clone(), desc);
        }
        self
    }

    // -------------------------------------------------------------------------
    // Classification Methods
    // -------------------------------------------------------------------------

    /// Add a classification task.
    ///
    /// # Arguments
    ///
    /// * `task` - The task name.
    /// * `labels` - The possible labels.
    ///
    /// # Returns
    ///
    /// A `ClassificationBuilder` for configuring the classification.
    ///
    /// # Example
    ///
    /// ```
    /// use gliner2_candle::schema::builder::SchemaBuilder;
    ///
    /// let schema = SchemaBuilder::new()
    ///     .classification("sentiment", vec!["positive".to_string(), "negative".to_string(), "neutral".to_string()])
    ///         .multi_label(false)
    ///         .threshold(0.5)
    ///         .done()
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn classification(
        self,
        task: impl Into<String>,
        labels: Vec<String>,
    ) -> ClassificationBuilder {
        ClassificationBuilder {
            parent: self,
            task: task.into(),
            labels,
            multi_label: false,
            threshold: 0.5,
            label_descriptions: HashMap::new(),
            prompt: None,
            examples: Vec::new(),
        }
    }

    // -------------------------------------------------------------------------
    // Structure Methods
    // -------------------------------------------------------------------------

    /// Start building a JSON structure.
    ///
    /// # Arguments
    ///
    /// * `name` - The structure name.
    ///
    /// # Returns
    ///
    /// A `StructureBuilder` for adding fields.
    ///
    /// # Example
    ///
    /// ```
    /// use gliner2_candle::schema::builder::SchemaBuilder;
    /// use gliner2_candle::schema::types::FieldDtype;
    ///
    /// let schema = SchemaBuilder::new()
    ///     .structure("product")
    ///         .field("name").dtype(FieldDtype::Str).done_field()
    ///         .field("price").done_field()
    ///         .field("colors").done_field()
    ///         .done_structure()
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn structure(self, name: impl Into<String>) -> StructureBuilder {
        let name = name.into();
        StructureBuilder {
            parent: Box::new(self),
            name,
            fields: Vec::new(),
            descriptions: HashMap::new(),
            field_order: Vec::new(),
            mode: StructureMode::Default,
            anchor: None,
        }
    }

    // -------------------------------------------------------------------------
    // Relation Methods
    // -------------------------------------------------------------------------

    /// Add a relation type.
    ///
    /// # Arguments
    ///
    /// * `name` - The relation type name.
    ///
    /// # Returns
    ///
    /// A `RelationBuilder` for configuring the relation.
    ///
    /// # Example
    ///
    /// ```
    /// use gliner2_candle::schema::builder::SchemaBuilder;
    ///
    /// let schema = SchemaBuilder::new()
    ///     .relation("works_for")
    ///         .description("Employment relationship")
    ///         .done()
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn relation(mut self, name: impl Into<String>) -> RelationBuilder {
        let name = name.into();
        if !self.relation_order.contains(&name) {
            self.relation_order.push(name.clone());
        }
        RelationBuilder {
            parent: self,
            name,
            description: None,
            threshold: None,
            fields: vec!["head".to_string(), "tail".to_string()],
        }
    }

    /// Add multiple relation types at once.
    ///
    /// # Arguments
    ///
    /// * `names` - A vector of relation type names.
    pub fn relations(mut self, names: Vec<String>) -> Self {
        for name in names {
            if !self.relation_order.contains(&name) {
                self.relation_order.push(name.clone());
            }
            self.schema.relations.push(RelationDef::new(name));
        }
        self
    }

    // -------------------------------------------------------------------------
    // Build & Convert
    // -------------------------------------------------------------------------

    /// Build the schema, validating all settings.
    ///
    /// # Returns
    ///
    /// The constructed `Schema`, or an error if validation fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the assembled schema fails validation.
    pub fn build(mut self) -> Result<Schema> {
        // Finish any active structure
        if let Some(structure_builder) = self.active_structure.take() {
            let structure = structure_builder.finish_structure();
            self.schema.structures.push(structure);
        }

        self.schema.validate()?;
        Ok(self.schema)
    }

    /// Build the schema without validation (for internal use).
    pub fn build_unchecked(mut self) -> Schema {
        if let Some(structure_builder) = self.active_structure.take() {
            let structure = structure_builder.finish_structure();
            self.schema.structures.push(structure);
        }
        self.schema
    }

    /// Get the underlying schema without consuming the builder.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get the entity order.
    pub fn entity_order(&self) -> &[String] {
        &self.entity_order
    }

    /// Get the relation order.
    pub fn relation_order(&self) -> &[String] {
        &self.relation_order
    }

    /// Get the field orders for structures.
    pub fn field_orders(&self) -> &HashMap<String, Vec<String>> {
        &self.field_orders
    }

    /// Get the field metadata.
    pub fn field_metadata(&self) -> &HashMap<String, FieldMetadata> {
        &self.field_metadata
    }

    /// Get the entity metadata.
    pub fn entity_metadata(&self) -> &HashMap<String, EntityMetadata> {
        &self.entity_metadata
    }

    /// Get the relation metadata.
    pub fn relation_metadata(&self) -> &HashMap<String, RelationMetadata> {
        &self.relation_metadata
    }

    /// Store field metadata for a structure field.
    fn store_field_metadata(
        &mut self,
        parent: &str,
        field: &str,
        dtype: FieldDtype,
        threshold: Option<f32>,
        choices: Option<Vec<String>>,
        validators: Option<Vec<RegexValidator>>,
    ) {
        let key = format!("{parent}.{field}");
        self.field_metadata.insert(
            key,
            FieldMetadata {
                dtype,
                threshold,
                choices,
                validators,
            },
        );
    }

    /// Store entity metadata.
    fn store_entity_metadata(&mut self, name: &str, dtype: FieldDtype, threshold: Option<f32>) {
        self.entity_metadata
            .insert(name.to_string(), EntityMetadata { dtype, threshold });
    }

    /// Store relation metadata.
    fn store_relation_metadata(&mut self, name: &str, threshold: Option<f32>) {
        self.relation_metadata
            .insert(name.to_string(), RelationMetadata { threshold });
    }

    /// Store field order for a structure.
    fn store_field_order(&mut self, parent: &str, order: Vec<String>) {
        self.field_orders.insert(parent.to_string(), order);
    }
}

// -------------------------------------------------------------------------
// Entity Builder
// -------------------------------------------------------------------------

/// Builder for configuring a single entity type.
#[derive(Debug)]
pub struct EntityBuilder {
    parent: SchemaBuilder,
    name: String,
    description: Option<String>,
    dtype: FieldDtype,
    threshold: Option<f32>,
}

impl EntityBuilder {
    /// Set the entity description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the data type.
    pub fn dtype(mut self, dtype: FieldDtype) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set the confidence threshold.
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Finish the entity and return to the schema builder.
    pub fn done(mut self) -> SchemaBuilder {
        let entity = EntityDef::new(&self.name);
        let entity = if let Some(desc) = &self.description {
            entity.with_description(desc)
        } else {
            entity
        };
        let entity = entity.with_dtype(self.dtype);
        let entity = if let Some(threshold) = self.threshold {
            entity.with_threshold(threshold)
        } else {
            entity
        };

        self.parent.schema.entities.push(entity);
        self.parent
            .store_entity_metadata(&self.name, self.dtype, self.threshold);
        if let Some(desc) = self.description {
            self.parent
                .schema
                .entity_descriptions
                .insert(self.name, desc);
        }
        self.parent
    }
}

// -------------------------------------------------------------------------
// Classification Builder
// -------------------------------------------------------------------------

/// Builder for configuring a classification task.
#[derive(Debug)]
pub struct ClassificationBuilder {
    parent: SchemaBuilder,
    task: String,
    labels: Vec<String>,
    multi_label: bool,
    threshold: f32,
    label_descriptions: HashMap<String, String>,
    prompt: Option<String>,
    examples: Vec<(String, String)>,
}

impl ClassificationBuilder {
    /// Enable multi-label mode.
    pub fn multi_label(mut self, enabled: bool) -> Self {
        self.multi_label = enabled;
        self
    }

    /// Set the classification threshold.
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set label descriptions.
    pub fn label_descriptions(mut self, descs: HashMap<String, String>) -> Self {
        self.label_descriptions = descs;
        self
    }

    /// Set the prompt.
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Add a few-shot example.
    pub fn example(mut self, input: impl Into<String>, output: impl Into<String>) -> Self {
        self.examples.push((input.into(), output.into()));
        self
    }

    /// Add multiple few-shot examples.
    pub fn examples(mut self, examples: Vec<(String, String)>) -> Self {
        self.examples.extend(examples);
        self
    }

    /// Finish the classification and return to the schema builder.
    pub fn done(mut self) -> SchemaBuilder {
        let mut cls = ClassificationDef::new(&self.task, self.labels);
        cls = cls.multi_label(self.multi_label);
        cls = cls.with_threshold(self.threshold);

        if !self.label_descriptions.is_empty() {
            cls = cls.with_label_descriptions(self.label_descriptions);
        }
        if let Some(prompt) = self.prompt {
            cls.prompt = Some(prompt);
        }
        if !self.examples.is_empty() {
            cls.examples = Some(self.examples);
        }

        self.parent.schema.classifications.push(cls);
        self.parent
    }
}

// -------------------------------------------------------------------------
// Structure Builder
// -------------------------------------------------------------------------

/// Builder for configuring a JSON structure.
#[derive(Debug)]
pub struct StructureBuilder {
    parent: Box<SchemaBuilder>,
    name: String,
    fields: Vec<FieldDef>,
    descriptions: HashMap<String, String>,
    field_order: Vec<String>,
    mode: StructureMode,
    anchor: Option<String>,
}

impl StructureBuilder {
    /// Set the record decoding mode for the structure.
    pub fn mode(mut self, mode: StructureMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the anchor field for natural-mode records.
    pub fn anchor(mut self, anchor: impl Into<String>) -> Self {
        self.anchor = Some(anchor.into());
        self
    }

    /// Add a field to the structure.
    ///
    /// # Arguments
    ///
    /// * `name` - The field name.
    ///
    /// # Returns
    ///
    /// A `FieldBuilder` for configuring the field.
    pub fn field(mut self, name: impl Into<String>) -> FieldBuilder {
        let name = name.into();
        self.field_order.push(name.clone());
        FieldBuilder {
            parent: self,
            name,
            dtype: FieldDtype::List,
            cardinality: None,
            choices: None,
            description: None,
            threshold: None,
            validators: None,
        }
    }

    /// Finish the structure and return to the schema builder.
    pub fn done_structure(self) -> SchemaBuilder {
        let structure = StructureDef {
            name: self.name.clone(),
            fields: self.fields,
            descriptions: self.descriptions,
            mode: self.mode,
            anchor: self.anchor,
        };
        let mut parent = *self.parent;
        parent.schema.structures.push(structure);
        parent.store_field_order(&self.name, self.field_order);
        parent
    }

    /// Internal method to finish the structure.
    fn finish_structure(self) -> StructureDef {
        StructureDef {
            name: self.name,
            fields: self.fields,
            descriptions: self.descriptions,
            mode: self.mode,
            anchor: self.anchor,
        }
    }
}

// -------------------------------------------------------------------------
// Field Builder
// -------------------------------------------------------------------------

/// Builder for configuring a structure field.
#[derive(Debug)]
pub struct FieldBuilder {
    parent: StructureBuilder,
    name: String,
    dtype: FieldDtype,
    cardinality: Option<FieldCardinality>,
    choices: Option<Vec<String>>,
    description: Option<String>,
    threshold: Option<f32>,
    validators: Option<Vec<RegexValidator>>,
}

impl FieldBuilder {
    /// Set the data type.
    pub fn dtype(mut self, dtype: FieldDtype) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set the cardinality.
    pub fn cardinality(mut self, cardinality: FieldCardinality) -> Self {
        self.cardinality = Some(cardinality);
        self
    }

    /// Set predefined choices.
    pub fn choices(mut self, choices: Vec<String>) -> Self {
        self.choices = Some(choices);
        self
    }

    /// Set the description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the confidence threshold.
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Set validators.
    pub fn validators(mut self, validators: Vec<RegexValidator>) -> Self {
        self.validators = Some(validators);
        self
    }

    /// Finish the field and return to the structure builder.
    pub fn done_field(mut self) -> StructureBuilder {
        let mut field = FieldDef::new(&self.name).with_dtype(self.dtype);
        if let Some(cardinality) = self.cardinality {
            field = field.with_cardinality(cardinality);
        }

        if let Some(ref choices) = self.choices {
            field = field.with_choices(choices.clone());
        }
        if let Some(desc) = self.description {
            field = field.with_description(&desc);
            self.parent.descriptions.insert(self.name.clone(), desc);
        }
        if let Some(threshold) = self.threshold {
            field = field.with_threshold(threshold);
        }
        if let Some(ref validators) = self.validators {
            field = field.with_validators(validators.clone());
        }

        self.parent.fields.push(field);

        // Store metadata in parent schema builder
        self.parent.parent.store_field_metadata(
            &self.parent.name,
            &self.name,
            self.dtype,
            self.threshold,
            self.choices,
            self.validators,
        );

        self.parent
    }
}

// -------------------------------------------------------------------------
// Relation Builder
// -------------------------------------------------------------------------

/// Builder for configuring a relation type.
#[derive(Debug)]
pub struct RelationBuilder {
    parent: SchemaBuilder,
    name: String,
    description: Option<String>,
    threshold: Option<f32>,
    fields: Vec<String>,
}

impl RelationBuilder {
    /// Set the relation description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the confidence threshold.
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Set custom field names.
    pub fn fields(mut self, fields: Vec<String>) -> Self {
        self.fields = fields;
        self
    }

    /// Finish the relation and return to the schema builder.
    pub fn done(mut self) -> SchemaBuilder {
        let mut relation = RelationDef::new(&self.name);

        if let Some(desc) = self.description {
            relation = relation.with_description(&desc);
        }
        if let Some(threshold) = self.threshold {
            relation = relation.with_threshold(threshold);
        }
        if self.fields != vec!["head".to_string(), "tail".to_string()] {
            relation = relation.with_fields(self.fields);
        }

        self.parent.schema.relations.push(relation);
        self.parent
            .store_relation_metadata(&self.name, self.threshold);
        self.parent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_builder() {
        let schema = SchemaBuilder::new()
            .entity("person")
            .description("Names of people")
            .threshold(0.6)
            .done()
            .entity("company")
            .done()
            .build()
            .unwrap();

        assert_eq!(schema.entities.len(), 2);
        assert_eq!(schema.entities[0].name, "person");
        assert_eq!(
            schema.entities[0].description,
            Some("Names of people".to_string())
        );
        assert_eq!(schema.entities[0].threshold, Some(0.6));
        assert_eq!(schema.entities[1].name, "company");
    }

    #[test]
    fn test_classification_builder() {
        let schema = SchemaBuilder::new()
            .classification(
                "sentiment",
                vec!["positive".to_string(), "negative".to_string()],
            )
            .multi_label(false)
            .threshold(0.4)
            .done()
            .build()
            .unwrap();

        assert_eq!(schema.classifications.len(), 1);
        assert_eq!(schema.classifications[0].task, "sentiment");
        assert!(!schema.classifications[0].multi_label);
        assert!((schema.classifications[0].cls_threshold - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn test_structure_builder() {
        let schema = SchemaBuilder::new()
            .structure("product")
            .field("name")
            .dtype(FieldDtype::Str)
            .done_field()
            .field("price")
            .done_field()
            .field("colors")
            .done_field()
            .done_structure()
            .build()
            .unwrap();

        assert_eq!(schema.structures.len(), 1);
        assert_eq!(schema.structures[0].name, "product");
        assert_eq!(schema.structures[0].fields.len(), 3);
        assert_eq!(schema.structures[0].fields[0].name, "name");
        assert_eq!(schema.structures[0].fields[0].dtype, FieldDtype::Str);
    }

    #[test]
    fn test_relation_builder() {
        let schema = SchemaBuilder::new()
            .relation("works_for")
            .description("Employment relationship")
            .threshold(0.5)
            .done()
            .build()
            .unwrap();

        assert_eq!(schema.relations.len(), 1);
        assert_eq!(schema.relations[0].name, "works_for");
        assert_eq!(
            schema.relations[0].description,
            Some("Employment relationship".to_string())
        );
    }

    #[test]
    fn test_complex_schema() {
        let schema = SchemaBuilder::new()
            .entity("person")
            .description("People")
            .done()
            .entity("company")
            .done()
            .classification(
                "sentiment",
                vec!["positive".to_string(), "negative".to_string()],
            )
            .done()
            .structure("product")
            .field("name")
            .dtype(FieldDtype::Str)
            .done_field()
            .field("price")
            .done_field()
            .done_structure()
            .relation("works_for")
            .done()
            .build()
            .unwrap();

        assert_eq!(schema.entities.len(), 2);
        assert_eq!(schema.classifications.len(), 1);
        assert_eq!(schema.structures.len(), 1);
        assert_eq!(schema.relations.len(), 1);
    }

    #[test]
    fn test_entities_batch() {
        let schema = SchemaBuilder::new()
            .entities(vec![
                "person".to_string(),
                "company".to_string(),
                "location".to_string(),
            ])
            .build()
            .unwrap();

        assert_eq!(schema.entities.len(), 3);
    }

    #[test]
    fn test_relations_batch() {
        let schema = SchemaBuilder::new()
            .relations(vec!["works_for".to_string(), "founded_by".to_string()])
            .build()
            .unwrap();

        assert_eq!(schema.relations.len(), 2);
    }

    #[test]
    fn test_validation_error() {
        let result = SchemaBuilder::new().build();
        assert!(result.is_err());
    }
}
