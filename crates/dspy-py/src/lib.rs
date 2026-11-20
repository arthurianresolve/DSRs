use std::fmt::Write;

use dspy_rs::ChatAdapter;
use dspy_rs::typesys::{OutputSchema, coerce, schema_block, type_name};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use serde::Deserialize;
use serde_json::Value;

mod schema;

#[derive(Deserialize)]
struct OutputSpec {
    name: String,
    schema: Value,
}

struct OutputField {
    name: String,
    schema: Option<OutputSchema>,
}

#[pyclass(frozen)]
struct PreparedSignature {
    fields: Vec<OutputField>,
    structure: String,
}

#[pymethods]
impl PreparedSignature {
    #[new]
    fn new(inputs: Vec<String>, outputs_json: &str) -> PyResult<Self> {
        let outputs: Vec<OutputSpec> = serde_json::from_str(outputs_json)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let mut structure = String::from(
            "All interactions will be structured in the following way, with the appropriate values filled in.\n\n",
        );
        for name in inputs {
            write!(structure, "[[ ## {name} ## ]]\n{name}\n\n").unwrap();
        }
        let mut fields = Vec::with_capacity(outputs.len());
        for OutputSpec { name, schema } in outputs {
            if name == "completed" {
                return Err(PyValueError::new_err(
                    "completed is reserved for the end marker",
                ));
            }
            writeln!(structure, "[[ ## {name} ## ]]").unwrap();
            let compiled = schema::compile(&schema, &name);
            if let Some(typed) = compiled
                .as_ref()
                .filter(|typed| !schema::needs_json_schema(typed))
            {
                let label = type_name(&typed.target, Some(&typed.types));
                let block = schema_block(&typed.target, &typed.types);
                writeln!(
                    structure,
                    "Output field `{name}` should be of type: {label}"
                )
                .unwrap();
                if block != label {
                    write!(structure, "\n{block}\n").unwrap();
                }
            } else {
                writeln!(
                    structure,
                    "Output field `{name}` should match JSON Schema: {schema}"
                )
                .unwrap();
            }
            structure.push('\n');
            fields.push(OutputField {
                name,
                schema: compiled,
            });
        }
        structure.push_str("[[ ## completed ## ]]");
        Ok(Self { fields, structure })
    }

    fn format_field_structure(&self) -> &str {
        &self.structure
    }

    fn sections<'py>(&self, py: Python<'py>, completion: &str) -> PyResult<Bound<'py, PyTuple>> {
        let sections = ChatAdapter::parse_sections_flexible(completion);
        PyTuple::new(
            py,
            self.fields
                .iter()
                .map(|field| sections.get(field.name.as_str()).map(|text| text.as_ref())),
        )
    }

    fn coerce_text(&self, py: Python<'_>, index: usize, raw: &str) -> PyResult<Option<String>> {
        let field = self
            .fields
            .get(index)
            .ok_or_else(|| PyValueError::new_err("unknown output field"))?;
        let Some(schema) = &field.schema else {
            return Ok(None);
        };
        py.detach(|| {
            Ok(coerce(raw, &schema.target, &schema.types)
                .ok()
                .map(|value| value.value.to_string()))
        })
    }
}

#[pymodule]
fn _dsrs_dspy(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PreparedSignature>()
}
