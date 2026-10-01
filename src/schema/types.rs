// Rust guideline compliant 2026-04-03
//! Schema types for GLiNER2.
//!
//! This module defines all data structures used to represent extraction schemas,
//! including entities, classifications, relations, and structured data fields.
//! These types mirror the Python `Schema` class functionality.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{GlinerError, Result};

/// Data type for schema fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum FieldDtype {
    /// Single string value.
    Str,
    /// List of string values.
    #[default]
    List,
}

impl std::fmt::Display for FieldDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldDtype::Str => write!(f, "str"),
            FieldDtype::List => write!(f, "list"),
        }
    }
}

impl std::str::FromStr for FieldDtype {
    type Err = GlinerError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "str" => Ok(FieldDtype::Str),
            "list" => Ok(FieldDtype::List),
            _ => Err(GlinerError::validation(format!(
                "Invalid dtype: {s}. Expected 'str' or 'list'"
            ))),
        }
    }
}

/// Regex validator for post-processing extracted spans.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegexValidator {
    /// The regex pattern.
    pub pattern: String,
    /// Match mode: "full" for fullmatch, "partial" for search.
    #[serde(default = "default_match_mode")]
    pub mode: MatchMode,
    /// If true, exclude matches (invert the filter).
    #[serde(default)]
    pub exclude: bool,
    /// Regex flags (case-insensitive by default).
    #[serde(default = "default_flags")]
    pub flags: u32,
}

fn default_match_mode() -> MatchMode {
    MatchMode::Full
}

fn default_flags() -> u32 {
    regex::RegexBuilder::new("").build().map_or(0, |_| {
        // Case-insensitive flag
        regex::Regex::new("(?i)").map_or(0, |_| 0)
    })
}

/// Match mode for regex validators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchMode {
    /// Full match (entire string must match).
    Full,
    /// Partial match (substring can match).
    Partial,
}

impl Default for RegexValidator {
    fn default() -> Self {
        Self {
            pattern: String::new(),
            mode: MatchMode::Full,
            exclude: false,
            flags: 0,
        }
    }
}

impl RegexValidator {
    /// Create a new regex validator.
    ///
    /// # Errors
    ///
    /// Returns an error if the regex pattern is invalid.
    pub fn new(pattern: impl Into<String>) -> Result<Self> {
        let pattern = pattern.into();
        regex::Regex::new(&pattern).map_err(|_e| {
            GlinerError::regex_validator(format!("Invalid regex pattern: {pattern}"))
        })?;
        Ok(Self {
            pattern,
            ..Default::default()
        })
    }

    /// Validate text against the pattern.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored regex pattern is invalid.
    pub fn validate(&self, text: &str) -> Result<bool> {
        let re = regex::Regex::new(&self.pattern)
            .map_err(|e| GlinerError::regex_validator(format!("Invalid regex: {}", e)))?;

        let matched = match self.mode {
            MatchMode::Full => {
                re.is_match(text)
                    && re
                        .find(text)
                        .is_some_and(|m| m.start() == 0 && m.end() == text.len())
            }
            MatchMode::Partial => re.is_match(text),
        };

        Ok(if self.exclude { !matched } else { matched })
    }
}

/// Cardinality of a structure field: how many values it may hold and whether
/// absence is allowed. Mirrors the reference `gliner2` `cardinality` concept.
///
/// * Scalar fields (`Str` dtype) map to ``RequiredOne``/``ZeroOrOne``.
/// * List fields (`List` dtype) map to ``OneOrMore``/``ZeroOrMore``.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldCardinality {
    /// Exactly one value, always present.
    RequiredOne,
    /// At most one value, absence allowed.
    ZeroOrOne,
    /// One or more values.
    OneOrMore,
    /// Zero or more values.
    ZeroOrMore,
}

#[allow(clippy::derivable_impls)]
impl Default for FieldCardinality {
    fn default() -> Self {
        FieldCardinality::ZeroOrMore
    }
}

impl FieldCardinality {
    /// Whether the field holds a single scalar value (`str` dtype).
    pub fn is_scalar(self) -> bool {
        matches!(
            self,
            FieldCardinality::RequiredOne | FieldCardinality::ZeroOrOne
        )
    }

    /// Whether the model may leave the field unset.
    pub fn allows_absent(self) -> bool {
        matches!(
            self,
            FieldCardinality::ZeroOrOne | FieldCardinality::ZeroOrMore
        )
    }

    /// The canonical cardinality for a given dtype (used when unspecified).
    pub fn for_dtype(dtype: FieldDtype) -> Self {
        match dtype {
            FieldDtype::Str => FieldCardinality::ZeroOrOne,
            FieldDtype::List => FieldCardinality::ZeroOrMore,
        }
    }
}

impl std::fmt::Display for FieldCardinality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                FieldCardinality::RequiredOne => "required_one",
                FieldCardinality::ZeroOrOne => "zero_or_one",
                FieldCardinality::OneOrMore => "one_or_more",
                FieldCardinality::ZeroOrMore => "zero_or_more",
            }
        )
    }
}

impl std::str::FromStr for FieldCardinality {
    type Err = GlinerError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "required_one" => Ok(FieldCardinality::RequiredOne),
            "zero_or_one" => Ok(FieldCardinality::ZeroOrOne),
            "one_or_more" => Ok(FieldCardinality::OneOrMore),
            "zero_or_more" => Ok(FieldCardinality::ZeroOrMore),
            _ => Err(GlinerError::validation(format!("Invalid cardinality: {s}"))),
        }
    }
}

/// Record decoding mode for a structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructureMode {
    /// Default mode.
    #[default]
    Default,
    /// Natural mode: an anchor field defines the record instances and each
    /// remaining field is assigned to the instance it co-occurs with.
    Natural,
}

impl std::fmt::Display for StructureMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                StructureMode::Default => "default",
                StructureMode::Natural => "natural",
            }
        )
    }
}

/// Field definition for structured data extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDef {
    /// Field name.
    pub name: String,
    /// Data type (str or list).
    #[serde(default)]
    pub dtype: FieldDtype,
    /// Cardinality of the field (optional; inferred from dtype when unset).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cardinality: Option<FieldCardinality>,
    /// Predefined choices for classification-style fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub choices: Option<Vec<String>>,
    /// Field description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Confidence threshold for this field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
    /// Regex validators for post-processing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validators: Option<Vec<RegexValidator>>,
}

impl FieldDef {
    /// Resolve the effective cardinality (explicit or dtype-default).
    pub fn effective_cardinality(&self) -> FieldCardinality {
        self.cardinality
            .unwrap_or_else(|| FieldCardinality::for_dtype(self.dtype))
    }
}

impl FieldDef {
    /// Create a new field definition.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dtype: FieldDtype::List,
            cardinality: None,
            choices: None,
            description: None,
            threshold: None,
            validators: None,
        }
    }

    /// Set the data type.
    pub fn with_dtype(mut self, dtype: FieldDtype) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set the cardinality.
    pub fn with_cardinality(mut self, cardinality: FieldCardinality) -> Self {
        self.cardinality = Some(cardinality);
        self
    }

    /// Set predefined choices.
    pub fn with_choices(mut self, choices: Vec<String>) -> Self {
        self.choices = Some(choices);
        self
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Set validators.
    pub fn with_validators(mut self, validators: Vec<RegexValidator>) -> Self {
        self.validators = Some(validators);
        self
    }
}

/// Structure definition for JSON structure extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureDef {
    /// Structure name.
    pub name: String,
    /// Field definitions.
    pub fields: Vec<FieldDef>,
    /// Field descriptions.
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub descriptions: HashMap<String, String>,
    /// Record decoding mode.
    #[serde(default)]
    pub mode: StructureMode,
    /// Anchor field name for natural mode (defines record instances).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
}

impl StructureDef {
    /// Create a new structure definition.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            fields: Vec::new(),
            descriptions: HashMap::new(),
            mode: StructureMode::Default,
            anchor: None,
        }
    }

    /// Add a field to the structure.
    pub fn add_field(mut self, field: FieldDef) -> Self {
        self.fields.push(field);
        self
    }

    /// Set the record decoding mode.
    pub fn with_mode(mut self, mode: StructureMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the anchor field for natural mode.
    pub fn with_anchor(mut self, anchor: impl Into<String>) -> Self {
        self.anchor = Some(anchor.into());
        self
    }
}

/// Attribute group attached to extracted entity spans.
///
/// Mirrors the Python reference `gliner2.inference.schema.AttributeGroup`:
/// labels are registered as hidden entity queries and re-scored at retained
/// spans after decoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeGroup {
    /// Values available in this attribute group.
    pub labels: Vec<String>,
    /// Use independent sigmoid decisions instead of forcing one value.
    #[serde(default)]
    pub multi_label: bool,
    /// Selection cutoff for multi-label groups.
    #[serde(default = "default_attribute_threshold")]
    pub threshold: f32,
    /// Optional entity types this group applies to (all when `None`).
    #[serde(default)]
    pub applies_to: Option<Vec<String>>,
    /// Prefix model-facing values with the group name to reduce ambiguity
    /// while keeping returned values unqualified.
    #[serde(default)]
    pub qualify_labels: bool,
}

fn default_attribute_threshold() -> f32 {
    0.5
}

impl Default for AttributeGroup {
    fn default() -> Self {
        Self {
            labels: Vec::new(),
            multi_label: false,
            threshold: default_attribute_threshold(),
            applies_to: None,
            qualify_labels: false,
        }
    }
}

/// Entity definition with optional metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityDef {
    /// Entity type name.
    pub name: String,
    /// Entity description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Data type (str or list).
    #[serde(default)]
    pub dtype: FieldDtype,
    /// Confidence threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
}

impl EntityDef {
    /// Create a new entity definition.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            dtype: FieldDtype::List,
            threshold: None,
        }
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the dtype.
    pub fn with_dtype(mut self, dtype: FieldDtype) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set the threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }
}

/// Classification task definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationDef {
    /// Task name.
    pub task: String,
    /// Possible labels.
    pub labels: Vec<String>,
    /// Whether multiple labels can be selected.
    #[serde(default)]
    pub multi_label: bool,
    /// Classification threshold.
    #[serde(default = "default_cls_threshold")]
    pub cls_threshold: f32,
    /// Label descriptions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_descriptions: Option<HashMap<String, String>>,
    /// Prompt for the task.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Few-shot examples.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub examples: Option<Vec<(String, String)>>,
}

fn default_cls_threshold() -> f32 {
    0.5
}

impl ClassificationDef {
    /// Create a new classification definition.
    pub fn new(task: impl Into<String>, labels: Vec<String>) -> Self {
        Self {
            task: task.into(),
            labels,
            multi_label: false,
            cls_threshold: 0.5,
            label_descriptions: None,
            prompt: None,
            examples: None,
        }
    }

    /// Set multi-label mode.
    pub fn multi_label(mut self, enabled: bool) -> Self {
        self.multi_label = enabled;
        self
    }

    /// Set the threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.cls_threshold = threshold;
        self
    }

    /// Set label descriptions.
    pub fn with_label_descriptions(mut self, descs: HashMap<String, String>) -> Self {
        self.label_descriptions = Some(descs);
        self
    }
}

/// Relation definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationDef {
    /// Relation type name.
    pub name: String,
    /// Relation description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Confidence threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
    /// Field names (typically "head" and "tail").
    #[serde(default = "default_relation_fields")]
    pub fields: Vec<String>,
}

fn default_relation_fields() -> Vec<String> {
    vec!["head".to_string(), "tail".to_string()]
}

impl RelationDef {
    /// Create a new relation definition.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            threshold: None,
            fields: default_relation_fields(),
        }
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Set custom fields.
    pub fn with_fields(mut self, fields: Vec<String>) -> Self {
        self.fields = fields;
        self
    }
}

/// Task type for schema items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    /// Entity extraction.
    Entities,
    /// Classification.
    Classifications,
    /// JSON structure extraction.
    JsonStructures,
    /// Relation extraction.
    Relations,
}

impl std::fmt::Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskType::Entities => write!(f, "entities"),
            TaskType::Classifications => write!(f, "classifications"),
            TaskType::JsonStructures => write!(f, "json_structures"),
            TaskType::Relations => write!(f, "relations"),
        }
    }
}

impl std::str::FromStr for TaskType {
    type Err = GlinerError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "entities" => Ok(TaskType::Entities),
            "classifications" => Ok(TaskType::Classifications),
            "json_structures" => Ok(TaskType::JsonStructures),
            "relations" => Ok(TaskType::Relations),
            _ => Err(GlinerError::validation(format!("Invalid task type: {s}"))),
        }
    }
}

/// Complete schema for information extraction.
///
/// This struct represents a full extraction schema that can include
/// entities, classifications, structures, and relations.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Schema {
    /// Entity definitions.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub entities: Vec<EntityDef>,
    /// Classification task definitions.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub classifications: Vec<ClassificationDef>,
    /// Structure definitions.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub structures: Vec<StructureDef>,
    /// Relation definitions.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub relations: Vec<RelationDef>,
    /// Attribute groups keyed by group name.
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub entity_attribute_groups: HashMap<String, AttributeGroup>,
    /// Entity descriptions (legacy format).
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub entity_descriptions: HashMap<String, String>,
    /// Structure descriptions (legacy format).
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub json_descriptions: HashMap<String, HashMap<String, String>>,
}

fn attribute_prompt_label(group_name: &str, group: &AttributeGroup, label: &str) -> String {
    if group.qualify_labels {
        format!("{group_name}: {label}")
    } else {
        label.to_string()
    }
}

impl Schema {
    /// Create a new empty schema.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add entities to the schema.
    pub fn entities(mut self, entities: Vec<EntityDef>) -> Self {
        self.entities = entities;
        self
    }

    /// Add classifications to the schema.
    pub fn classifications(mut self, classifications: Vec<ClassificationDef>) -> Self {
        self.classifications = classifications;
        self
    }

    /// Add structures to the schema.
    pub fn structures(mut self, structures: Vec<StructureDef>) -> Self {
        self.structures = structures;
        self
    }

    /// Add relations to the schema.
    pub fn relations(mut self, relations: Vec<RelationDef>) -> Self {
        self.relations = relations;
        self
    }

    /// Attach attribute groups to entities declared by this schema.
    ///
    /// Mirrors the Python `Schema.entity_attributes`: attribute labels are
    /// registered as hidden entity queries (excluded from the public entity
    /// order) and re-scored at retained spans after decoding.
    pub fn entity_attributes(mut self, groups: HashMap<String, AttributeGroup>) -> Result<Self> {
        let content_entities: std::collections::HashSet<&String> =
            self.entities.iter().map(|e| &e.name).collect();
        if content_entities.is_empty() {
            return Err(GlinerError::invalid_schema(
                "entity_attributes() requires entities() to be called first",
            ));
        }

        const RESERVED: [&str; 4] = ["text", "confidence", "start", "end"];
        let mut seen: HashMap<&str, &str> = HashMap::new();
        for (group_name, group) in &groups {
            if group_name.is_empty() || RESERVED.contains(&group_name.as_str()) {
                return Err(GlinerError::invalid_schema(format!(
                    "Invalid attribute group name {group_name:?}: must be non-empty \
                     and not one of {RESERVED:?}"
                )));
            }
            if group.labels.is_empty() {
                return Err(GlinerError::invalid_schema(format!(
                    "Attribute group '{group_name}' has no labels"
                )));
            }
            if !(0.0..=1.0).contains(&group.threshold) {
                return Err(GlinerError::invalid_schema(format!(
                    "Attribute group '{group_name}' threshold must be in [0, 1], got {}",
                    group.threshold
                )));
            }
            if let Some(applies_to) = &group.applies_to {
                for entity in applies_to {
                    if !content_entities.contains(entity) {
                        return Err(GlinerError::invalid_schema(format!(
                            "Attribute group '{group_name}' applies to unknown entity: {entity}"
                        )));
                    }
                }
            }
            for label in &group.labels {
                if label.trim().is_empty() {
                    return Err(GlinerError::invalid_schema(format!(
                        "Attribute group '{group_name}' has an empty label"
                    )));
                }
                if let Some(prev_group) = seen.get(label.as_str()) {
                    return Err(GlinerError::invalid_schema(format!(
                        "Label '{label}' is in both '{prev_group}' and '{group_name}'"
                    )));
                }
                seen.insert(label, group_name);
            }
        }

        // Prompt-label collisions with declared entities.
        for (group_name, group) in &groups {
            for label in &group.labels {
                let prompt = attribute_prompt_label(group_name, group, label);
                if content_entities.contains(&prompt) {
                    return Err(GlinerError::invalid_schema(format!(
                        "Attribute labels collide with entity labels: {prompt}; use qualify_labels=true"
                    )));
                }
            }
        }

        self.entity_attribute_groups = groups;
        Ok(self)
    }

    /// Resolved attribute metadata for inference.
    ///
    /// Returns `(prompt_by_label, prompt_labels_sorted)` where prompt labels
    /// are the model-facing query names, sorted like the Python reference
    /// before insertion into the entity list.
    pub fn attribute_prompts(&self) -> (HashMap<String, String>, Vec<String>) {
        let mut prompts: HashMap<String, String> = HashMap::new();
        for (group_name, group) in &self.entity_attribute_groups {
            for label in &group.labels {
                prompts.insert(
                    label.clone(),
                    attribute_prompt_label(group_name, group, label),
                );
            }
        }
        let mut sorted: Vec<String> = {
            let mut v: Vec<String> = prompts.values().cloned().collect();
            v.sort();
            v.dedup();
            v
        };
        sorted.shrink_to_fit();
        (prompts, sorted)
    }

    /// A copy of this schema with hidden attribute queries appended as entity
    /// definitions, plus the label→prompt mapping.
    ///
    /// The returned schema should be used for collation; the original schema's
    /// public entity order is unchanged.
    pub fn expanded_with_attributes(&self) -> Result<(Schema, HashMap<String, String>)> {
        if self.entity_attribute_groups.is_empty() {
            return Ok((self.clone(), HashMap::new()));
        }
        let (prompts, sorted) = self.attribute_prompts();
        let mut expanded = self.clone();
        for prompt in &sorted {
            expanded.entities.push(EntityDef::new(prompt.clone()));
        }
        Ok((expanded, prompts))
    }

    /// Check if the schema is empty.
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
            && self.classifications.is_empty()
            && self.structures.is_empty()
            && self.relations.is_empty()
    }

    /// Get all task types in the schema.
    pub fn task_types(&self) -> Vec<TaskType> {
        [
            (!self.entities.is_empty(), TaskType::Entities),
            (!self.classifications.is_empty(), TaskType::Classifications),
            (!self.structures.is_empty(), TaskType::JsonStructures),
            (!self.relations.is_empty(), TaskType::Relations),
        ]
        .into_iter()
        .filter_map(|(enabled, task)| enabled.then_some(task))
        .collect()
    }

    /// Validate the schema.
    ///
    /// # Errors
    ///
    /// Returns an error if required schema fields are missing or thresholds are
    /// out of range.
    pub fn validate(&self) -> Result<()> {
        if self.is_empty() {
            return Err(GlinerError::invalid_schema(
                "Schema must have at least one task type",
            ));
        }

        // Validate entities
        for entity in &self.entities {
            if entity.name.is_empty() {
                return Err(GlinerError::invalid_schema("Entity name cannot be empty"));
            }
            if let Some(threshold) = entity.threshold
                && !(0.0..=1.0).contains(&threshold)
            {
                return Err(GlinerError::invalid_schema(format!(
                    "Entity threshold must be 0-1, got {threshold}"
                )));
            }
        }

        // Validate classifications
        for cls in &self.classifications {
            if cls.task.is_empty() {
                return Err(GlinerError::invalid_schema(
                    "Classification task name cannot be empty",
                ));
            }
            if cls.labels.is_empty() {
                return Err(GlinerError::invalid_schema(format!(
                    "Classification '{}' has no labels",
                    cls.task
                )));
            }
            if !(0.0..=1.0).contains(&cls.cls_threshold) {
                return Err(GlinerError::invalid_schema(format!(
                    "Classification threshold must be 0-1, got {}",
                    cls.cls_threshold
                )));
            }
        }

        // Validate structures
        for structure in &self.structures {
            if structure.name.is_empty() {
                return Err(GlinerError::invalid_schema(
                    "Structure name cannot be empty",
                ));
            }
            if structure.fields.is_empty() {
                return Err(GlinerError::invalid_schema(format!(
                    "Structure '{}' has no fields",
                    structure.name
                )));
            }
            if structure.mode == StructureMode::Natural {
                let anchor = structure.anchor.as_deref().ok_or_else(|| {
                    GlinerError::invalid_schema(format!(
                        "Structure '{}' is in natural mode but has no anchor field",
                        structure.name
                    ))
                })?;
                if !structure.fields.iter().any(|f| f.name == anchor) {
                    return Err(GlinerError::invalid_schema(format!(
                        "Structure '{}' anchor field '{anchor}' is not among its fields",
                        structure.name
                    )));
                }
            }
            for field in &structure.fields {
                if field.name.is_empty() {
                    return Err(GlinerError::invalid_schema("Field name cannot be empty"));
                }
                if let Some(card) = field.cardinality {
                    if card.is_scalar() != (field.dtype == FieldDtype::Str) {
                        return Err(GlinerError::invalid_schema(format!(
                            "Field '{}.{}' cardinality '{}' is incompatible with dtype '{}'",
                            structure.name, field.name, card, field.dtype
                        )));
                    }
                }
                if let Some(threshold) = field.threshold
                    && !(0.0..=1.0).contains(&threshold)
                {
                    return Err(GlinerError::invalid_schema(format!(
                        "Field threshold must be 0-1, got {threshold}"
                    )));
                }
            }
        }

        // Validate relations
        for relation in &self.relations {
            if relation.name.is_empty() {
                return Err(GlinerError::invalid_schema("Relation name cannot be empty"));
            }
            if let Some(threshold) = relation.threshold
                && !(0.0..=1.0).contains(&threshold)
            {
                return Err(GlinerError::invalid_schema(format!(
                    "Relation threshold must be 0-1, got {threshold}"
                )));
            }
        }

        Ok(())
    }

    /// Convert to dictionary format compatible with Python GLiNER2.
    pub fn to_dict(&self) -> serde_json::Value {
        let mut dict = serde_json::Map::new();

        // Entities - use array format to preserve order (BTreeMap sorts alphabetically)
        if !self.entities.is_empty() {
            let entity_names: Vec<serde_json::Value> = self
                .entities
                .iter()
                .map(|e| serde_json::Value::String(e.name.clone()))
                .collect();
            dict.insert(
                "entities".to_string(),
                serde_json::Value::Array(entity_names),
            );

            if !self.entity_descriptions.is_empty() {
                let descs: serde_json::Map<String, serde_json::Value> = self
                    .entity_descriptions
                    .iter()
                    .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
                    .collect();
                dict.insert(
                    "entity_descriptions".to_string(),
                    serde_json::Value::Object(descs),
                );
            }
        }

        // Classifications
        if !self.classifications.is_empty() {
            let classifications: Vec<serde_json::Value> = self
                .classifications
                .iter()
                .map(|cls| {
                    // The serde field names already match the keys the collator
                    // reads, and `examples` serializes as [[input, output]],
                    // which is what `build_classification_tokens` expects.
                    // Building the map by hand previously dropped
                    // `label_descriptions`, `prompt` and `examples` on the floor.
                    let mut obj = match serde_json::to_value(cls) {
                        Ok(serde_json::Value::Object(obj)) => obj,
                        _ => serde_json::Map::new(),
                    };
                    if !cls.multi_label {
                        obj.remove("multi_label");
                    }
                    let threshold = if cls.cls_threshold.is_finite() {
                        cls.cls_threshold.clamp(0.0, 1.0)
                    } else {
                        0.5
                    };
                    if let Some(number) = serde_json::Number::from_f64(threshold as f64) {
                        obj.insert(
                            "cls_threshold".to_string(),
                            serde_json::Value::Number(number),
                        );
                    }
                    serde_json::Value::Object(obj)
                })
                .collect();
            dict.insert(
                "classifications".to_string(),
                serde_json::Value::Array(classifications),
            );
        }

        // Structures
        if !self.structures.is_empty() {
            let structures: Vec<serde_json::Value> = self
                .structures
                .iter()
                .map(|s| {
                    let mut obj = serde_json::Map::new();
                    let mut fields = serde_json::Map::new();
                    for field in &s.fields {
                        let has_metadata = field.choices.is_some()
                            || field.description.is_some()
                            || field.threshold.is_some()
                            || field.validators.is_some()
                            || field.dtype != FieldDtype::List;

                        if has_metadata {
                            let mut field_obj = serde_json::Map::new();
                            field_obj.insert(
                                "value".to_string(),
                                serde_json::Value::String("".to_string()),
                            );
                            field_obj.insert(
                                "dtype".to_string(),
                                serde_json::Value::String(field.dtype.to_string()),
                            );
                            if let Some(card) = field.cardinality {
                                field_obj.insert(
                                    "cardinality".to_string(),
                                    serde_json::Value::String(card.to_string()),
                                );
                            }

                            if let Some(choices) = &field.choices {
                                field_obj.insert(
                                    "choices".to_string(),
                                    serde_json::Value::Array(
                                        choices
                                            .iter()
                                            .map(|c| serde_json::Value::String(c.clone()))
                                            .collect(),
                                    ),
                                );
                            }
                            if let Some(description) = &field.description {
                                field_obj.insert(
                                    "description".to_string(),
                                    serde_json::Value::String(description.clone()),
                                );
                            }
                            if let Some(threshold) = field.threshold
                                && let Some(n) = serde_json::Number::from_f64(threshold as f64)
                            {
                                field_obj
                                    .insert("threshold".to_string(), serde_json::Value::Number(n));
                            }
                            if let Some(validators) = &field.validators {
                                let vals = serde_json::to_value(validators)
                                    .unwrap_or(serde_json::Value::Array(vec![]));
                                field_obj.insert("validators".to_string(), vals);
                            }

                            fields.insert(field.name.clone(), serde_json::Value::Object(field_obj));
                        } else {
                            fields.insert(
                                field.name.clone(),
                                serde_json::Value::String("".to_string()),
                            );
                        }
                    }
                    obj.insert(s.name.clone(), serde_json::Value::Object(fields));
                    if s.mode == StructureMode::Natural {
                        obj.insert(
                            "__mode__".to_string(),
                            serde_json::Value::String("natural".to_string()),
                        );
                        if let Some(anchor) = &s.anchor {
                            obj.insert(
                                "__anchor__".to_string(),
                                serde_json::Value::String(anchor.clone()),
                            );
                        }
                    }
                    serde_json::Value::Object(obj)
                })
                .collect();
            dict.insert(
                "json_structures".to_string(),
                serde_json::Value::Array(structures),
            );

            if !self.json_descriptions.is_empty() {
                let descs: serde_json::Map<String, serde_json::Value> = self
                    .json_descriptions
                    .iter()
                    .map(|(k, v)| {
                        let inner: serde_json::Map<String, serde_json::Value> = v
                            .iter()
                            .map(|(fk, fv)| (fk.clone(), serde_json::Value::String(fv.clone())))
                            .collect();
                        (k.clone(), serde_json::Value::Object(inner))
                    })
                    .collect();
                dict.insert(
                    "json_descriptions".to_string(),
                    serde_json::Value::Object(descs),
                );
            }
        }

        // Relations
        if !self.relations.is_empty() {
            let relations: Vec<serde_json::Value> = self
                .relations
                .iter()
                .map(|r| {
                    let mut obj = serde_json::Map::new();
                    let mut fields = serde_json::Map::new();
                    for field_name in &r.fields {
                        fields.insert(
                            field_name.clone(),
                            serde_json::Value::String("".to_string()),
                        );
                    }
                    obj.insert(r.name.clone(), serde_json::Value::Object(fields));
                    serde_json::Value::Object(obj)
                })
                .collect();
            dict.insert("relations".to_string(), serde_json::Value::Array(relations));

            let relation_meta: serde_json::Map<String, serde_json::Value> = self
                .relations
                .iter()
                .map(|r| {
                    let mut meta = serde_json::Map::new();
                    if let Some(t) = r.threshold {
                        if let Some(n) = serde_json::Number::from_f64(t as f64) {
                            meta.insert("threshold".to_string(), serde_json::Value::Number(n));
                        }
                    }
                    meta.insert(
                        "fields".to_string(),
                        serde_json::Value::Array(
                            r.fields
                                .iter()
                                .map(|f| serde_json::Value::String(f.clone()))
                                .collect(),
                        ),
                    );
                    (r.name.clone(), serde_json::Value::Object(meta))
                })
                .collect();
            if !relation_meta.is_empty() {
                dict.insert(
                    "relation_metadata".to_string(),
                    serde_json::Value::Object(relation_meta),
                );
            }
        }

        serde_json::Value::Object(dict)
    }

    /// Create a schema from a dictionary (Python-compatible format).
    ///
    /// # Errors
    ///
    /// Returns an error if parsed schema content fails validation.
    pub fn from_dict(dict: &serde_json::Value) -> Result<Self> {
        let mut schema = Self::new();

        if let Some(obj) = dict.as_object() {
            // Parse entities
            if let Some(entities) = obj.get("entities") {
                if let Some(entities_obj) = entities.as_object() {
                    for (name, value) in entities_obj {
                        let mut entity = EntityDef::new(name);
                        if let Some(desc) = value.as_str()
                            && !desc.is_empty()
                        {
                            entity = entity.with_description(desc);
                        }
                        schema.entities.push(entity);
                    }
                } else if let Some(entities_arr) = entities.as_array() {
                    for value in entities_arr {
                        if let Some(name) = value.as_str() {
                            schema.entities.push(EntityDef::new(name));
                        }
                    }
                }
            }

            // Parse entity descriptions
            if let Some(descs) = obj.get("entity_descriptions")
                && let Some(descs_obj) = descs.as_object()
            {
                for (k, v) in descs_obj {
                    if let Some(desc) = v.as_str() {
                        schema
                            .entity_descriptions
                            .insert(k.clone(), desc.to_string());
                    }
                }
            }

            // Parse classifications
            if let Some(classifications) = obj.get("classifications")
                && let Some(cls_arr) = classifications.as_array()
            {
                for cls_value in cls_arr {
                    if let Some(cls_obj) = cls_value.as_object()
                        && let Some(task) = cls_obj.get("task").and_then(|v| v.as_str())
                    {
                        let labels = cls_obj
                            .get("labels")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();

                        let mut cls = ClassificationDef::new(task, labels);

                        if let Some(multi) = cls_obj.get("multi_label").and_then(|v| v.as_bool()) {
                            cls = cls.multi_label(multi);
                        }
                        if let Some(threshold) =
                            cls_obj.get("cls_threshold").and_then(|v| v.as_f64())
                        {
                            cls = cls.with_threshold(threshold as f32);
                        }
                        if let Some(descs) = cls_obj.get("label_descriptions").and_then(|v| v.as_object())
                        {
                            let map: HashMap<String, String> = descs
                                .iter()
                                .filter_map(|(k, v)| {
                                    v.as_str().map(|d| (k.clone(), d.to_string()))
                                })
                                .collect();
                            if !map.is_empty() {
                                cls = cls.with_label_descriptions(map);
                            }
                        }
                        if let Some(prompt) = cls_obj.get("prompt").and_then(|v| v.as_str()) {
                            cls.prompt = Some(prompt.to_string());
                        }
                        if let Some(examples) = cls_obj.get("examples").and_then(|v| v.as_array())
                        {
                            let pairs: Vec<(String, String)> = examples
                                .iter()
                                .filter_map(|e| {
                                    let pair = e.as_array()?;
                                    let input = pair.first()?.as_str()?;
                                    let output = pair.get(1)?.as_str()?;
                                    Some((input.to_string(), output.to_string()))
                                })
                                .collect();
                            if !pairs.is_empty() {
                                cls.examples = Some(pairs);
                            }
                        }

                        schema.classifications.push(cls);
                    }
                }
            }

            // Parse structures
            if let Some(structures) = obj.get("json_structures")
                && let Some(struct_arr) = structures.as_array()
            {
                for struct_value in struct_arr {
                    if let Some(struct_obj) = struct_value.as_object() {
                        let mut mode = StructureMode::Default;
                        let mut anchor: Option<String> = None;
                        for (name, fields_value) in struct_obj {
                            if name == "__mode__" {
                                if fields_value.as_str() == Some("natural") {
                                    mode = StructureMode::Natural;
                                }
                                continue;
                            }
                            if name == "__anchor__" {
                                anchor = fields_value.as_str().map(String::from);
                                continue;
                            }
                            let mut structure = StructureDef::new(name);
                            structure.mode = mode;
                            structure.anchor.clone_from(&anchor);
                            if let Some(fields_obj) = fields_value.as_object() {
                                for (field_name, field_value) in fields_obj {
                                    let mut field = FieldDef::new(field_name);
                                    if let Some(field_obj) = field_value.as_object() {
                                        if let Some(dtype) =
                                            field_obj.get("dtype").and_then(|v| v.as_str())
                                            && let Ok(dt) = dtype.parse::<FieldDtype>()
                                        {
                                            field = field.with_dtype(dt);
                                        }
                                        if let Some(card) =
                                            field_obj.get("cardinality").and_then(|v| v.as_str())
                                            && let Ok(c) = card.parse::<FieldCardinality>()
                                        {
                                            field = field.with_cardinality(c);
                                        }
                                        if let Some(choices) =
                                            field_obj.get("choices").and_then(|v| v.as_array())
                                        {
                                            let choice_strings: Vec<String> = choices
                                                .iter()
                                                .filter_map(|v| v.as_str().map(String::from))
                                                .collect();
                                            field = field.with_choices(choice_strings);
                                        }
                                    }
                                    structure.fields.push(field);
                                }
                            }
                            schema.structures.push(structure);
                        }
                    }
                }
            }

            // Parse structure descriptions
            if let Some(descs) = obj.get("json_descriptions")
                && let Some(descs_obj) = descs.as_object()
            {
                for (struct_name, fields_descs) in descs_obj {
                    if let Some(fields_obj) = fields_descs.as_object() {
                        let mut inner_map = HashMap::new();
                        for (field_name, desc) in fields_obj {
                            if let Some(desc_str) = desc.as_str() {
                                inner_map.insert(field_name.clone(), desc_str.to_string());
                            }
                        }
                        schema
                            .json_descriptions
                            .insert(struct_name.clone(), inner_map);
                    }
                }
            }

            // Parse relations
            if let Some(relations) = obj.get("relations") {
                if let Some(rel_arr) = relations.as_array() {
                    for rel_value in rel_arr {
                        if let Some(rel_obj) = rel_value.as_object() {
                            for (name, fields_val) in rel_obj {
                                let mut relation = RelationDef::new(name);
                                let field_names: Vec<String> = if let Some(meta_fields) = obj
                                    .get("relation_metadata")
                                    .and_then(|m| m.get(name))
                                    .and_then(|m| m.get("fields"))
                                    .and_then(|f| f.as_array())
                                {
                                    meta_fields
                                        .iter()
                                        .filter_map(|v| v.as_str().map(String::from))
                                        .collect()
                                } else if let Some(fields_arr) = fields_val.as_array() {
                                    fields_arr
                                        .iter()
                                        .filter_map(|v| v.as_str().map(String::from))
                                        .collect()
                                } else if let Some(fields_obj) = fields_val.as_object() {
                                    fields_obj.keys().cloned().collect()
                                } else {
                                    vec!["head".to_string(), "tail".to_string()]
                                };
                                if !field_names.is_empty() {
                                    relation = relation.with_fields(field_names);
                                }
                                schema.relations.push(relation);
                            }
                        }
                    }
                } else if let Some(rel_arr_str) = relations.as_array() {
                    for rel_value in rel_arr_str {
                        if let Some(name) = rel_value.as_str() {
                            schema.relations.push(RelationDef::new(name));
                        }
                    }
                }
            }
        }

        schema.validate()?;
        Ok(schema)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_def() {
        let entity = EntityDef::new("person")
            .with_description("Names of people")
            .with_threshold(0.6);

        assert_eq!(entity.name, "person");
        assert_eq!(entity.description, Some("Names of people".to_string()));
        assert_eq!(entity.threshold, Some(0.6));
    }

    #[test]
    fn test_classification_def() {
        let cls = ClassificationDef::new(
            "sentiment",
            vec!["positive".to_string(), "negative".to_string()],
        )
        .multi_label(true)
        .with_threshold(0.4);

        assert_eq!(cls.task, "sentiment");
        assert!(cls.multi_label);
        assert!((cls.cls_threshold - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn test_field_def() {
        let field = FieldDef::new("price")
            .with_dtype(FieldDtype::Str)
            .with_description("Product price");

        assert_eq!(field.name, "price");
        assert_eq!(field.dtype, FieldDtype::Str);
        assert_eq!(field.description, Some("Product price".to_string()));
    }

    #[test]
    fn test_schema_validation() {
        let schema = Schema::new();
        assert!(schema.validate().is_err());

        let schema = Schema::new().entities(vec![EntityDef::new("person")]);
        assert!(schema.validate().is_ok());
    }

    #[test]
    fn test_schema_to_dict() {
        let schema = Schema::new()
            .entities(vec![
                EntityDef::new("person").with_description("Names of people"),
                EntityDef::new("company"),
            ])
            .classifications(vec![ClassificationDef::new(
                "sentiment",
                vec!["positive".to_string(), "negative".to_string()],
            )]);

        let dict = schema.to_dict();
        assert!(dict.get("entities").is_some());
        assert!(dict.get("classifications").is_some());
    }

    /// `to_dict` used to hand-build the classification map field by field and
    /// silently drop `label_descriptions`, `prompt` and `examples`. The collator
    /// reads all three, so a schema built through `SchemaBuilder` never reached
    /// the model with any of them. Going through `to_dict` here is the point:
    /// tests that hand-build a `serde_json::Value` cannot catch this class of bug.
    #[test]
    fn test_to_dict_preserves_classification_prompt_metadata() {
        let mut descriptions = HashMap::new();
        descriptions.insert(
            "card_lost".to_string(),
            "the physical card is missing".to_string(),
        );
        let cls = ClassificationDef::new("intent", vec!["card_lost".to_string()])
            .with_label_descriptions(descriptions);
        let cls = ClassificationDef {
            prompt: Some("What does the customer want?".to_string()),
            examples: Some(vec![(
                "my card is gone".to_string(),
                "card_lost".to_string(),
            )]),
            ..cls
        };
        let schema = Schema::new().classifications(vec![cls]);

        let head = &schema.to_dict()["classifications"][0];
        assert_eq!(head["label_descriptions"]["card_lost"], "the physical card is missing");
        assert_eq!(head["prompt"], "What does the customer want?");
        assert_eq!(head["examples"][0][0], "my card is gone");
        assert_eq!(head["examples"][0][1], "card_lost");
    }

    #[test]
    fn test_from_dict_reads_classification_prompt_metadata() {
        let dict = serde_json::json!({
            "classifications": [{
                "task": "intent",
                "labels": ["card_lost"],
                "label_descriptions": {"card_lost": "the physical card is missing"},
                "prompt": "What does the customer want?",
                "examples": [["my card is gone", "card_lost"]]
            }]
        });

        let schema = Schema::from_dict(&dict).expect("valid schema");
        let cls = &schema.classifications[0];

        assert_eq!(
            cls.label_descriptions.as_ref().unwrap()["card_lost"],
            "the physical card is missing"
        );
        assert_eq!(cls.prompt.as_deref(), Some("What does the customer want?"));
        assert_eq!(
            cls.examples.as_ref().unwrap(),
            &vec![("my card is gone".to_string(), "card_lost".to_string())]
        );
    }

    #[test]
    fn test_schema_from_dict() {
        let dict = serde_json::json!({
            "entities": {
                "person": "Names of people",
                "company": ""
            },
            "classifications": [
                {
                    "task": "sentiment",
                    "labels": ["positive", "negative"],
                    "multi_label": false,
                    "cls_threshold": 0.5
                }
            ]
        });

        let schema = Schema::from_dict(&dict).unwrap();
        assert_eq!(schema.entities.len(), 2);
        assert_eq!(schema.classifications.len(), 1);
        assert_eq!(schema.classifications[0].task, "sentiment");
    }

    #[test]
    fn test_regex_validator() {
        let validator = RegexValidator::new(r"^\d+$").unwrap();
        assert!(validator.validate("123").unwrap());
        assert!(!validator.validate("abc").unwrap());
    }

    #[test]
    fn test_task_type_from_str() {
        assert_eq!("entities".parse::<TaskType>().unwrap(), TaskType::Entities);
        assert_eq!(
            "classifications".parse::<TaskType>().unwrap(),
            TaskType::Classifications
        );
        assert!("invalid".parse::<TaskType>().is_err());
    }

    #[test]
    fn test_field_dtype_from_str() {
        assert_eq!("str".parse::<FieldDtype>().unwrap(), FieldDtype::Str);
    }

    #[test]
    fn test_relation_fields_to_from_dict_roundtrip() {
        let schema = Schema {
            relations: vec![
                RelationDef::new("rate_limit")
                    .with_fields(vec!["system".to_string(), "metric_value".to_string()]),
            ],
            ..Default::default()
        };
        let dict = schema.to_dict();
        let rels = dict.get("relations").and_then(|v| v.as_array()).unwrap();
        assert_eq!(rels.len(), 1);
        let rel_obj = rels[0]
            .get("rate_limit")
            .and_then(|v| v.as_object())
            .unwrap();
        assert!(rel_obj.contains_key("system"));
        assert!(rel_obj.contains_key("metric_value"));

        let restored = Schema::from_dict(&dict).unwrap();
        assert_eq!(restored.relations.len(), 1);
        assert_eq!(restored.relations[0].name, "rate_limit");
        assert_eq!(
            restored.relations[0].fields,
            vec!["system".to_string(), "metric_value".to_string()]
        );
    }

    #[test]
    fn test_field_dtype_from_str_extra() {
        assert_eq!("list".parse::<FieldDtype>().unwrap(), FieldDtype::List);
        assert!("invalid".parse::<FieldDtype>().is_err());
    }
}
