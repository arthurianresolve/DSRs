//! JSON Schema projections for provider output and tool inputs.

use std::collections::HashSet;

use serde_json::{Map, Value, json};

use crate::ir::{FieldDef, SignatureDef};

use super::{FieldType, TypeTable};

/// Projects output fields and their reachable types into a strict JSON Schema.
/// Object properties use LM-facing aliases; parsers map these back to canonical names.
pub fn json_schema_for_outputs(def: &SignatureDef, types: &TypeTable) -> Value {
    project(&def.outputs, types, SchemaKind::StrictOutput)
}

/// Projects a tool or hole's inputs into a JSON Schema object.
/// Optional arguments can be omitted. Recursive classes use local references.
pub fn input_schema_of(def: &SignatureDef, types: &TypeTable) -> Value {
    project(&def.inputs, types, SchemaKind::ToolInput)
}

/// Whether this signature fits the strict JSON Schema subset accepted by providers.
/// Open maps use JSON mode or delimiter output.
pub fn json_schema_compatible(def: &SignatureDef, types: &TypeTable) -> bool {
    let mut visited = None;
    def.outputs
        .iter()
        .all(|field| compatible_type(&field.ty, types, &mut visited))
}

fn compatible_type<'a>(
    ty: &'a FieldType,
    types: &'a TypeTable,
    visited: &mut Option<HashSet<&'a str>>,
) -> bool {
    match ty {
        // Open maps are not valid in providers' strict schema subset. Keep them
        // open by selecting JSON mode or delimiter output instead of closing them.
        FieldType::Map(_, _) => false,
        FieldType::Optional(inner) | FieldType::List(inner) => {
            compatible_type(inner, types, visited)
        }
        FieldType::Union(items) => items
            .iter()
            .all(|item| compatible_type(item, types, visited)),
        FieldType::Class(name) if visited.get_or_insert_with(HashSet::new).insert(name) => {
            types.classes.get(name).is_none_or(|class| {
                class
                    .fields
                    .iter()
                    .all(|field| compatible_type(&field.field_type, types, visited))
            })
        }
        _ => true,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SchemaKind {
    StrictOutput,
    ToolInput,
}

fn project(fields: &[FieldDef], types: &TypeTable, kind: SchemaKind) -> Value {
    let mut projection = SchemaProjection {
        types,
        kind,
        defs: Map::new(),
        active_classes: Vec::new(),
    };
    let mut root = projection.object(
        fields
            .iter()
            .map(|field| (field.lm_name.as_ref(), &field.ty, field.docs.as_deref())),
    );
    if !projection.defs.is_empty() {
        root["$defs"] = Value::Object(projection.defs);
    }
    root
}

struct SchemaProjection<'a> {
    types: &'a TypeTable,
    kind: SchemaKind,
    defs: Map<String, Value>,
    active_classes: Vec<&'a str>,
}

impl<'a> SchemaProjection<'a> {
    fn object(
        &mut self,
        fields: impl IntoIterator<Item = (&'a str, &'a FieldType, Option<&'a str>)>,
    ) -> Value {
        let mut properties = Map::new();
        let mut required = Vec::new();
        let strict = self.kind == SchemaKind::StrictOutput;
        for (name, ty, docs) in fields {
            properties.insert(name.to_owned(), described(self.ty(ty), docs));
            if strict || !ty.is_optional() {
                required.push(Value::String(name.to_owned()));
            }
        }
        let mut object = json!({"type": "object"});
        object["properties"] = Value::Object(properties);
        object["required"] = Value::Array(required);
        if strict {
            object["additionalProperties"] = Value::Bool(false);
        }
        object
    }

    fn ty(&mut self, ty: &'a FieldType) -> Value {
        match ty {
            FieldType::String => json!({"type": "string"}),
            FieldType::Int => json!({"type": "integer"}),
            FieldType::Float => json!({"type": "number"}),
            FieldType::Bool => json!({"type": "boolean"}),
            FieldType::Literal(value) => json!({"type": "string", "const": value}),
            FieldType::List(inner) => {
                let mut schema = json!({"type": "array"});
                schema["items"] = self.ty(inner);
                schema
            }
            FieldType::Optional(inner) => {
                let schema = self.ty(inner);
                if self.kind == SchemaKind::StrictOutput {
                    // Strict providers require every property, so optional outputs are nullable.
                    alternatives(vec![schema, json!({"type": "null"})])
                } else {
                    schema
                }
            }
            FieldType::Map(_, value) => {
                let mut schema = json!({"type": "object"});
                schema["additionalProperties"] = self.ty(value);
                schema
            }
            FieldType::Class(name) => self.class(name),
            FieldType::Enum(name) => self.enum_type(name),
            FieldType::Union(items) => {
                alternatives(items.iter().map(|item| self.ty(item)).collect())
            }
        }
    }

    fn class(&mut self, name: &'a str) -> Value {
        if self.defs.contains_key(name) {
            return reference(name);
        }
        let strict = self.kind == SchemaKind::StrictOutput;
        let Some(class) = self.types.classes.get(name) else {
            return if strict {
                reference(name)
            } else {
                json!({"type": "object"})
            };
        };
        if self.active_classes.contains(&name) {
            // The active projection will replace this entry when it unwinds.
            self.defs.insert(name.to_owned(), Value::Null);
            return reference(name);
        }
        self.active_classes.push(name);
        let schema = self.object(class.fields.iter().map(|field| {
            (
                field.rendered_name.as_str(),
                &field.field_type,
                if strict { field.docs.as_deref() } else { None },
            )
        }));
        self.active_classes.pop();
        if strict || self.defs.contains_key(name) {
            let docs = if strict { class.docs.as_deref() } else { None };
            self.defs.insert(name.to_owned(), described(schema, docs));
            reference(name)
        } else {
            schema
        }
    }

    fn enum_type(&mut self, name: &str) -> Value {
        let strict = self.kind == SchemaKind::StrictOutput;
        if strict && self.defs.contains_key(name) {
            return reference(name);
        }
        let mut schema = json!({"type": "string"});
        let Some(enm) = self.types.enums.get(name) else {
            return if strict { reference(name) } else { schema };
        };
        schema["enum"] = Value::Array(
            enm.values
                .iter()
                .map(|value| {
                    Value::String(if strict {
                        value.rendered_name.clone()
                    } else {
                        value.name.clone()
                    })
                })
                .collect(),
        );
        if strict {
            self.defs
                .insert(name.to_owned(), described(schema, enm.docs.as_deref()));
            reference(name)
        } else {
            schema
        }
    }
}

fn described(mut schema: Value, docs: Option<&str>) -> Value {
    if let Some(docs) = docs {
        schema["description"] = Value::String(docs.to_owned());
    }
    schema
}

fn alternatives(schemas: Vec<Value>) -> Value {
    Value::Object(Map::from_iter([(
        "anyOf".to_owned(),
        Value::Array(schemas),
    )]))
}

fn reference(name: &str) -> Value {
    let token = name.replace('~', "~0").replace('/', "~1");
    json!({"$ref": format!("#/$defs/{token}")})
}

/// Maps LM-facing field aliases to canonical output keys recursively.
pub fn canonicalize_output_aliases(value: Value, def: &SignatureDef, types: &TypeTable) -> Value {
    let Value::Object(input) = value else {
        return value;
    };
    Value::Object(canonicalize_object(
        input,
        def.outputs
            .iter()
            .map(|f| (f.lm_name.as_ref(), f.name.as_ref(), &f.ty)),
        types,
    ))
}

/// Rewrites canonical output keys to LM-facing names in few-shot JSON examples.
pub fn alias_output_fields(value: Value, def: &SignatureDef, types: &TypeTable) -> Value {
    let Value::Object(input) = value else {
        return value;
    };
    Value::Object(alias_object(
        input,
        def.outputs
            .iter()
            .map(|f| (f.name.as_ref(), f.lm_name.as_ref(), &f.ty)),
        types,
    ))
}

fn alias_object<'a>(
    mut input: Map<String, Value>,
    fields: impl ExactSizeIterator<Item = (&'a str, &'a str, &'a FieldType)>,
    types: &TypeTable,
) -> Map<String, Value> {
    let single = fields.len() == 1;
    let mut out = Map::new();
    for (canonical, lm, ty) in fields {
        let selected = if single {
            input
                .swap_remove(canonical)
                .or_else(|| input.swap_remove(lm))
        } else {
            input.get(canonical).or_else(|| input.get(lm)).cloned()
        };
        let Some(mut item) = selected else {
            continue;
        };
        alias_nested(&mut item, ty, types);
        out.insert(lm.into(), item);
    }
    out
}

fn alias_nested(value: &mut Value, ty: &FieldType, types: &TypeTable) {
    match ty {
        FieldType::Optional(inner) => alias_nested(value, inner, types),
        FieldType::List(inner) => {
            if let Value::Array(values) = value {
                for item in values {
                    alias_nested(item, inner, types);
                }
            }
        }
        FieldType::Class(name) => {
            if let Some(class) = types.classes.get(name)
                && let Value::Object(input) = value
            {
                *input = alias_object(
                    std::mem::take(input),
                    class
                        .fields
                        .iter()
                        .map(|f| (f.name.as_str(), f.rendered_name.as_str(), &f.field_type)),
                    types,
                );
            }
        }
        FieldType::Map(_, item) => {
            if let Value::Object(values) = value {
                for item_value in values.values_mut() {
                    alias_nested(item_value, item, types);
                }
            }
        }
        FieldType::Union(items) => {
            for item in items {
                if let FieldType::Class(name) = item
                    && let Some(class) = types.classes.get(name)
                    && class.fields.iter().any(|field| {
                        value.get(&field.name).is_some()
                            || value.get(&field.rendered_name).is_some()
                    })
                {
                    alias_nested(value, item, types);
                    break;
                }
            }
        }
        _ => {}
    }
}

fn canonicalize_object<'a>(
    mut input: Map<String, Value>,
    fields: impl ExactSizeIterator<Item = (&'a str, &'a str, &'a FieldType)>,
    types: &TypeTable,
) -> Map<String, Value> {
    let single = fields.len() == 1;
    let mut out = Map::new();
    for (lm, canonical, ty) in fields {
        // An alias can also be another field's canonical key.
        let selected = if single {
            input
                .swap_remove(lm)
                .or_else(|| input.swap_remove(canonical))
        } else {
            input.get(lm).or_else(|| input.get(canonical)).cloned()
        };
        let Some(mut value) = selected else {
            continue;
        };
        canonicalize_nested(&mut value, ty, types);
        out.insert(canonical.into(), value);
    }
    out
}

fn canonicalize_nested(value: &mut Value, ty: &FieldType, types: &TypeTable) {
    match ty {
        FieldType::Optional(inner) => canonicalize_nested(value, inner, types),
        FieldType::List(inner) => if let Value::Array(values) = value { for value in values { canonicalize_nested(value, inner, types); } },
        FieldType::Class(name) => {
            if let Some(class) = types.classes.get(name)
                && let Value::Object(input) = value
            {
                *input = canonicalize_object(
                    std::mem::take(input),
                    class.fields.iter().map(|field| {
                        (field.rendered_name.as_str(), field.name.as_str(), &field.field_type)
                    }),
                    types,
                );
            }
        }
        FieldType::Map(_, item) => if let Value::Object(values) = value { for value in values.values_mut() { canonicalize_nested(value, item, types); } },
        FieldType::Union(items) => if let Some(item) = items.iter().find(|item| matches!(item, FieldType::Class(name) if value.get("$type").and_then(Value::as_str) == Some(name))) { canonicalize_nested(value, item, types); },
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ConstraintDef, FieldDef as IrFieldDef};
    use crate::typesys::{ClassDef, EnumDef, EnumValueDef};

    fn base_types() -> TypeTable {
        let mut types = TypeTable::default();
        types.classes.insert(
            "Node".into(),
            ClassDef {
                internal_name: "Node".into(),
                rendered_name: "Node".into(),
                docs: Some("linked node".into()),
                fields: vec![
                    crate::typesys::FieldDef {
                        name: "label".into(),
                        rendered_name: "label_alias".into(),
                        field_type: FieldType::String,
                        docs: Some("node label".into()),
                        constraints: vec![],
                    },
                    crate::typesys::FieldDef {
                        name: "next".into(),
                        rendered_name: "next".into(),
                        field_type: FieldType::Optional(Box::new(FieldType::Class("Node".into()))),
                        docs: None,
                        constraints: vec![],
                    },
                ],
                constraints: vec![],
            },
        );
        types.enums.insert(
            "Mood".into(),
            EnumDef {
                internal_name: "Mood".into(),
                rendered_name: "Mood".into(),
                docs: None,
                values: vec![EnumValueDef {
                    name: "Happy".into(),
                    rendered_name: "joyful".into(),
                    docs: None,
                }],
            },
        );
        types
    }

    #[test]
    fn tool_input_schema_keeps_recursive_classes_and_optional_arguments() {
        let def = SignatureDef::build("Tool")
            .input("root", FieldType::Class("Node".into()))
            .input("mood", FieldType::optional(FieldType::Enum("Mood".into())))
            .output("result", FieldType::String)
            .finish()
            .unwrap();

        let schema = input_schema_of(&def, &base_types());
        assert_eq!(schema["required"], json!(["root"]));
        assert_eq!(schema["properties"]["mood"]["enum"], json!(["Happy"]));
        assert_eq!(schema["properties"]["root"]["$ref"], "#/$defs/Node");
        let node = &schema["$defs"]["Node"];
        assert_eq!(node["properties"]["next"]["$ref"], "#/$defs/Node");
        assert_eq!(node["required"], json!(["label_alias"]));
    }

    #[test]
    fn schema_has_aliases_nullable_optional_enum_and_recursive_defs() {
        let def = SignatureDef::build("Out")
            .input("seed", FieldType::String)
            .output_full(IrFieldDef::new("root", FieldType::Class("Node".into())).aliased("tree"))
            .output(
                "maybe",
                FieldType::Optional(Box::new(FieldType::Enum("Mood".into()))),
            )
            .finish()
            .unwrap();
        let schema = json_schema_for_outputs(&def, &base_types());
        assert_eq!(schema["properties"]["tree"]["$ref"], "#/$defs/Node");
        assert_eq!(schema["properties"]["maybe"]["anyOf"][1]["type"], "null");
        assert_eq!(schema["required"], json!(["tree", "maybe"]));
        assert_eq!(
            schema["$defs"]["Node"]["properties"]["label_alias"]["description"],
            "node label"
        );
        assert_eq!(
            schema["$defs"]["Node"]["properties"]["next"]["anyOf"][1]["type"],
            "null"
        );
        assert_eq!(schema["$defs"]["Mood"]["enum"], json!(["joyful"]));
    }

    #[test]
    fn maps_remain_open_and_nested_non_string_maps_fall_back() {
        let def = SignatureDef::build("Out")
            .input("seed", FieldType::String)
            .output(
                "m",
                FieldType::Map(Box::new(FieldType::String), Box::new(FieldType::Int)),
            )
            .finish()
            .unwrap();
        let schema = json_schema_for_outputs(&def, &TypeTable::default());
        assert_eq!(
            schema["properties"]["m"]["additionalProperties"]["type"],
            "integer"
        );
        assert!(!json_schema_compatible(&def, &TypeTable::default()));
        let mut types = base_types();
        types
            .classes
            .get_mut("Node")
            .unwrap()
            .fields
            .push(crate::typesys::FieldDef {
                name: "map".into(),
                rendered_name: "map".into(),
                field_type: FieldType::Map(Box::new(FieldType::Int), Box::new(FieldType::String)),
                docs: None,
                constraints: vec![],
            });
        let nested = SignatureDef::build("Nested")
            .input("seed", FieldType::String)
            .output("n", FieldType::Class("Node".into()))
            .finish()
            .unwrap();
        assert!(!json_schema_compatible(&nested, &types));
    }

    #[test]
    fn empty_output_signature_projects_empty_object() {
        let def = SignatureDef {
            name: "Empty".into(),
            instruction: "".into(),
            inputs: Box::new([]),
            outputs: Box::new([]),
        };
        let schema = json_schema_for_outputs(&def, &TypeTable::default());
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"], json!({}));
        assert_eq!(schema["required"], json!([]));
        assert_eq!(schema["additionalProperties"], false);
    }

    #[test]
    fn aliases_canonicalize_recursively_without_string_round_trip() {
        let def = SignatureDef::build("Out")
            .input("seed", FieldType::String)
            .output_full(IrFieldDef::new("root", FieldType::Class("Node".into())).aliased("tree"))
            .output_full(
                IrFieldDef::new("verified", FieldType::Bool)
                    .with_constraint(ConstraintDef::check("affirmative", "this == true")),
            )
            .output_full(
                IrFieldDef::new("asserted", FieldType::Int)
                    .with_constraint(ConstraintDef::assert("this > 0")),
            )
            .finish()
            .unwrap();
        let types = base_types();
        let canonical = canonicalize_output_aliases(
            json!({"tree":{"label_alias":"a \"quoted\" value", "next":null},"verified":true,"asserted":1}),
            &def,
            &types,
        );
        let output = canonical.as_object().unwrap();
        let value = crate::typesys::coerce_value(
            output["root"].clone(),
            &FieldType::Class("Node".into()),
            &types,
        )
        .unwrap()
        .value;
        assert_eq!(value["label"], "a \"quoted\" value");
        assert_eq!(
            alias_output_fields(json!({"root":{"label":"x","next":null}}), &def, &types)["tree"]["label_alias"],
            "x"
        );
    }

    #[test]
    fn demo_aliases_can_share_the_same_source_key() {
        let def = SignatureDef::build("Out")
            .input("seed", FieldType::String)
            .output_full(IrFieldDef::new("first", FieldType::String).aliased("second"))
            .output_full(IrFieldDef::new("second", FieldType::String).aliased("third"))
            .finish()
            .unwrap();
        let types = TypeTable::default();
        assert_eq!(
            alias_output_fields(json!({"second": "shared"}), &def, &types),
            json!({"second": "shared", "third": "shared"})
        );
        assert_eq!(
            alias_output_fields(
                json!({"first": "canonical", "second": "shared"}),
                &def,
                &types
            ),
            json!({"second": "canonical", "third": "shared"})
        );
    }

    #[test]
    fn native_json_coercion_preserves_delimited_coercion_behavior() {
        let types = base_types();
        let node = crate::typesys::coerce_value(
            json!({"label_alias":"a \"quoted\" label"}),
            &FieldType::Class("Node".into()),
            &types,
        )
        .unwrap()
        .value;
        assert_eq!(node["label"], "a \"quoted\" label");
        assert_eq!(node["next"], Value::Null);

        let mood =
            crate::typesys::coerce_value(json!("JOYFUL"), &FieldType::Enum("Mood".into()), &types)
                .unwrap()
                .value;
        assert_eq!(mood, "Happy");

        let count = crate::typesys::coerce_value(json!("3.7"), &FieldType::Int, &types)
            .unwrap()
            .value;
        assert_eq!(count, 4);
    }
}
