use dspy_rs::typesys::schema::{ClassDef, FieldDef};
use dspy_rs::typesys::{FieldType, OutputSchema, TypeTable};
use serde_json::Value;

pub fn compile(schema: &Value, name: &str) -> Option<OutputSchema> {
    let mut compiler = SchemaCompiler {
        definitions: schema.get("$defs").or_else(|| schema.get("definitions")),
        types: TypeTable::default(),
        next_inline_class: 0,
    };
    let target = compiler.field_type(schema, name, None, 0)?;
    Some(OutputSchema {
        target,
        types: compiler.types,
    })
}

pub fn needs_json_schema(schema: &OutputSchema) -> bool {
    hides_class_fields(&schema.target, true)
        || schema.types.classes.values().any(|class| {
            class
                .fields
                .iter()
                .any(|field| hides_class_fields(&field.field_type, true))
        })
}

fn hides_class_fields(ty: &FieldType, can_inline: bool) -> bool {
    // The compact renderer can inline a class through an optional or a direct list.
    match ty {
        FieldType::Class(_) => !can_inline,
        FieldType::Optional(inner) => hides_class_fields(inner, can_inline),
        FieldType::List(inner) => hides_class_fields(
            inner,
            can_inline && matches!(inner.as_ref(), FieldType::Class(_)),
        ),
        FieldType::Map(_, value) => hides_class_fields(value, false),
        FieldType::Union(choices) => choices
            .iter()
            .any(|choice| hides_class_fields(choice, false)),
        _ => false,
    }
}

struct SchemaCompiler<'a> {
    definitions: Option<&'a Value>,
    types: TypeTable,
    next_inline_class: usize,
}

impl SchemaCompiler<'_> {
    fn field_type(
        &mut self,
        schema: &Value,
        name: &str,
        reference_name: Option<&str>,
        depth: usize,
    ) -> Option<FieldType> {
        if depth >= 64 {
            return None;
        }
        if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
            let key = reference
                .strip_prefix("#/$defs/")
                .or_else(|| reference.strip_prefix("#/definitions/"))?
                .replace("~1", "/")
                .replace("~0", "~");
            let definition = self.definitions?.get(&key)?;
            return self.field_type(definition, &key, Some(&key), depth + 1);
        }
        if let Some(constant) = schema.get("const") {
            return Some(FieldType::Literal(constant.as_str()?.to_owned()));
        }
        if let Some(values) = schema.get("enum").and_then(Value::as_array) {
            let choices = values
                .iter()
                .map(|value| Some(FieldType::Literal(value.as_str()?.to_owned())))
                .collect::<Option<Vec<_>>>()?;
            return alternatives(choices);
        }
        if let Some(variants) = schema
            .get("anyOf")
            .or_else(|| schema.get("oneOf"))
            .and_then(Value::as_array)
        {
            let mut choices = Vec::new();
            let mut nullable = false;
            for variant in variants {
                if variant.get("type").and_then(Value::as_str) == Some("null") {
                    nullable = true;
                } else {
                    choices.push(self.field_type(variant, name, None, depth + 1)?);
                }
            }
            let ty = alternatives(choices)?;
            return Some(if nullable {
                FieldType::optional(ty)
            } else {
                ty
            });
        }
        if let Some([only]) = schema
            .get("allOf")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
        {
            return self.field_type(only, name, reference_name, depth + 1);
        }

        match schema.get("type").and_then(Value::as_str)? {
            "string" => Some(FieldType::String),
            "integer" => Some(FieldType::Int),
            "number" => Some(FieldType::Float),
            "boolean" => Some(FieldType::Bool),
            "array" => Some(FieldType::List(Box::new(self.field_type(
                schema.get("items")?,
                name,
                None,
                depth + 1,
            )?))),
            "object" => self.object_type(schema, name, reference_name, depth),
            _ => None,
        }
    }

    fn object_type(
        &mut self,
        schema: &Value,
        name: &str,
        reference_name: Option<&str>,
        depth: usize,
    ) -> Option<FieldType> {
        let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
            return Some(FieldType::Map(
                Box::new(FieldType::String),
                Box::new(self.field_type(
                    schema.get("additionalProperties")?,
                    name,
                    None,
                    depth + 1,
                )?),
            ));
        };
        let key = if let Some(reference_name) = reference_name {
            format!("ref:{reference_name}")
        } else {
            self.next_inline_class += 1;
            format!("inline:{}", self.next_inline_class)
        };
        if self.types.classes.contains_key(&key) {
            return Some(FieldType::Class(key));
        }

        let class = ClassDef {
            internal_name: key.clone(),
            rendered_name: schema
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or(name)
                .to_owned(),
            docs: description(schema),
            fields: Vec::new(),
            constraints: Vec::new(),
        };
        // Register the name before walking properties so recursive models can refer to it.
        self.types.classes.insert(key.clone(), class);
        let required = schema.get("required").and_then(Value::as_array);
        let mut fields = Vec::with_capacity(properties.len());
        for (name, prop) in properties {
            let mut ty = self.field_type(prop, name, None, depth + 1)?;
            if !required
                .is_some_and(|fields| fields.iter().any(|field| field.as_str() == Some(name)))
            {
                ty = FieldType::optional(ty);
            }
            fields.push(FieldDef {
                name: name.clone(),
                rendered_name: name.clone(),
                field_type: ty,
                docs: description(prop),
                constraints: Vec::new(),
            });
        }
        self.types.classes.get_mut(&key)?.fields = fields;
        Some(FieldType::Class(key))
    }
}

fn alternatives(mut items: Vec<FieldType>) -> Option<FieldType> {
    match items.len() {
        0 => None,
        1 => items.pop(),
        _ => Some(FieldType::Union(items)),
    }
}

fn description(schema: &Value) -> Option<String> {
    schema
        .get("description")
        .and_then(Value::as_str)
        .map(str::to_owned)
}
