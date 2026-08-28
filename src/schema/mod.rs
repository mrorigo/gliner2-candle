// Rust guideline compliant 2026-04-03
//! Schema module for GLiNER2.
//!
//! This module provides types and builders for defining extraction schemas,
//! including entities, classifications, structures, and relations.
//!
//! # Example
//!
//! ```
//! use gliner2_candle::schema::SchemaBuilder;
//! use gliner2_candle::schema::types::FieldDtype;
//!
//! let schema = SchemaBuilder::new()
//!     .entity("person").description("Names of people").done()
//!     .entity("company").done()
//!     .classification("sentiment", vec!["positive".to_string(), "negative".to_string()])
//!         .threshold(0.5)
//!         .done()
//!     .structure("product")
//!         .field("name").dtype(FieldDtype::Str).done_field()
//!         .field("price").done_field()
//!         .done_structure()
//!     .relation("works_for").done()
//!     .build()
//!     .unwrap();
//! ```

pub mod builder;
pub mod types;

// Re-export commonly used types for convenience
pub use builder::SchemaBuilder;
pub use types::{
    ClassificationDef, EntityDef, FieldDef, FieldDtype, MatchMode, RegexValidator, RelationDef,
    Schema, StructureDef, TaskType,
};
