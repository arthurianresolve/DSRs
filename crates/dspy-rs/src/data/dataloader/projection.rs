//! Typed projection and source-error ordering shared by all row readers.

use std::collections::HashSet;

use anyhow::{Result, anyhow};
use facet::{Facet, Type, UserType};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use crate::FieldType;
use crate::typesys::schema::field_type_from_shape;

use super::{DataLoadError, RowRecord, TypedLoadOptions, UnknownFieldPolicy};

pub(super) fn load<E>(
    opts: &TypedLoadOptions,
    visit: impl FnOnce(&mut TypedProjection<'_>) -> Result<(), DataLoadError>,
) -> Result<Vec<E>>
where
    E: DeserializeOwned + for<'f> Facet<'f>,
{
    let mut projection = TypedProjection::new::<E>(opts);
    visit(&mut projection)?;
    projection.finish()
}

fn resolve_source_field<'a>(field: &'a str, opts: &'a TypedLoadOptions) -> &'a str {
    opts.field_map
        .get(field)
        .map(String::as_str)
        .unwrap_or(field)
}

struct ProjectedField<'a> {
    name: &'static str,
    source: &'a str,
    field_type: FieldType,
    last_use: bool,
}

fn projection_fields<E: for<'f> Facet<'f>>(
    opts: &TypedLoadOptions,
) -> std::result::Result<Vec<ProjectedField<'_>>, DataLoadError> {
    let shape = E::SHAPE;
    let struct_type = match &shape.ty {
        Type::User(UserType::Struct(struct_type)) => struct_type,
        _ => {
            return Err(DataLoadError::Json(anyhow!(
                "row type `{}` must be a struct",
                shape.type_identifier
            )));
        }
    };
    let mut seen = HashSet::new();
    let mut fields: Vec<_> = struct_type
        .fields
        .iter()
        .rev()
        .filter(|field| !field.should_skip_deserializing())
        .map(|field| {
            let source = resolve_source_field(field.name, opts);
            ProjectedField {
                name: field.name,
                source,
                field_type: field_type_from_shape(field.shape()),
                last_use: seen.insert(source),
            }
        })
        .collect();
    fields.reverse();
    Ok(fields)
}

pub(super) struct TypedProjection<'a> {
    unknown_fields: UnknownFieldPolicy,
    fields: std::result::Result<Vec<ProjectedField<'a>>, DataLoadError>,
    rows: Vec<ProjectedRow>,
    error: Option<DataLoadError>,
}

impl<'a> TypedProjection<'a> {
    fn new<E: for<'f> Facet<'f>>(opts: &'a TypedLoadOptions) -> Self {
        Self {
            unknown_fields: opts.unknown_fields,
            fields: projection_fields::<E>(opts),
            rows: Vec::new(),
            error: None,
        }
    }

    #[cfg(feature = "data")]
    pub(super) fn source_fields(&self) -> Option<HashSet<&'a str>> {
        (self.unknown_fields == UnknownFieldPolicy::Ignore).then(|| {
            self.fields
                .as_deref()
                .unwrap_or_default()
                .iter()
                .map(|field| field.source)
                .collect()
        })
    }

    pub(super) fn push(&mut self, row: RowRecord) {
        if let Ok(fields) = &self.fields
            && self.error.is_none()
        {
            match project_row(row, fields, self.unknown_fields) {
                Ok(row) => self.rows.push(row),
                Err(error) => self.error = Some(error),
            }
        }
    }

    fn finish<E: DeserializeOwned>(self) -> Result<Vec<E>> {
        self.fields?;
        // Source validation completes before any user deserializer runs.
        // A deserialization error on an earlier row wins over a projection error.
        let examples = self
            .rows
            .into_iter()
            .map(ProjectedRow::deserialize)
            .collect::<std::result::Result<_, _>>()?;
        if let Some(error) = self.error {
            return Err(error.into());
        }
        Ok(examples)
    }
}

struct ProjectedRow {
    index: usize,
    fields: Map<String, Value>,
}

impl ProjectedRow {
    fn deserialize<E: DeserializeOwned>(self) -> std::result::Result<E, DataLoadError> {
        serde_json::from_value(Value::Object(self.fields)).map_err(|err| {
            DataLoadError::TypeMismatch {
                row: self.index,
                field: "row".to_string(),
                message: err.to_string(),
            }
        })
    }
}

fn project_row(
    mut row: RowRecord,
    fields: &[ProjectedField<'_>],
    unknown_fields: UnknownFieldPolicy,
) -> std::result::Result<ProjectedRow, DataLoadError> {
    let mut map = Map::with_capacity(fields.len());

    for field in fields {
        let value = if field.last_use {
            row.values.remove(field.source)
        } else {
            row.values.get(field.source).cloned()
        };
        match value {
            Some(value) => {
                map.insert(
                    field.name.to_string(),
                    coerce_row_value(value, &field.field_type),
                );
            }
            // Missing optional fields deserialize as `None`; anything else is an error.
            None if field.field_type.is_optional() => {}
            None => {
                return Err(DataLoadError::MissingField {
                    row: row.row_index,
                    field: field.name.to_string(),
                });
            }
        }
    }

    if unknown_fields == UnknownFieldPolicy::Error
        && let Some((field, _)) = row.values.into_iter().next()
    {
        return Err(DataLoadError::UnknownField {
            row: row.row_index,
            field,
        });
    }

    Ok(ProjectedRow {
        index: row.row_index,
        fields: map,
    })
}

/// Nudges a loosely-typed source value (common with CSV, where every cell starts as text)
/// toward the field's declared type before strict serde deserialization. This preserves the
/// tolerant loading behavior the vendored BAML stack used to provide: a `"4"` cell fills a
/// `String` field as `"4"` and an `i64` field as `4`. Values that don't cleanly convert are
/// passed through unchanged so serde surfaces a precise type-mismatch error.
fn coerce_row_value(value: Value, field_type: &FieldType) -> Value {
    match (field_type.peel(), value) {
        (FieldType::String, Value::Number(number)) => Value::String(number.to_string()),
        (FieldType::String, Value::Bool(flag)) => Value::String(flag.to_string()),
        (FieldType::Int, Value::String(text)) => text
            .trim()
            .parse::<i64>()
            .map(|parsed| Value::Number(parsed.into()))
            .unwrap_or_else(|_| Value::String(text)),
        (FieldType::Float, Value::String(text)) => text
            .trim()
            .parse::<f64>()
            .ok()
            .and_then(serde_json::Number::from_f64)
            .map(Value::Number)
            .unwrap_or(Value::String(text)),
        (FieldType::Bool, Value::String(text)) => {
            let trimmed = text.trim();
            if trimmed.eq_ignore_ascii_case("true") {
                Value::Bool(true)
            } else if trimmed.eq_ignore_ascii_case("false") {
                Value::Bool(false)
            } else {
                Value::String(text)
            }
        }
        (_, value) => value,
    }
}
