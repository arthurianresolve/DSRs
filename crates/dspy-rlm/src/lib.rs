//! A DSRs module that lets an LM inspect typed inputs in a Python session.

use std::error::Error;
use std::fmt::Write;
use std::marker::PhantomData;
use std::sync::Arc;

use dspy_rs::ir::SignatureDef;
use dspy_rs::typesys::{json_schema_for_outputs, schema_block};
use dspy_rs::{
    CallMetadata, Chat, ChatAdapter, LM, LMResponse, LmError, Message, Module, OutputSchema,
    ParseError, Predict, PredictError, Predicted, Signature, ToolLoopMode, try_get_lm,
};
use futures::future::join_all;
use serde_json::Value;

mod runtime;
use runtime::{CellOutput, Session};

/// Limits for one reasoning session.
struct RlmConfig {
    max_iterations: usize,
    max_llm_calls: usize,
    max_output_chars: usize,
    enable_extraction_fallback: bool,
}

impl Default for RlmConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            max_llm_calls: 50,
            max_output_chars: 10_000,
            enable_extraction_fallback: true,
        }
    }
}

/// Runs model-generated Python cells against the input, then returns the typed submission.
///
/// The cells run in embedded CPython with the application's permissions. Use this
/// module with code and data sources that you trust to run in the host process.
pub struct Rlm<S: Signature> {
    config: RlmConfig,
    lm: Option<Arc<LM>>,
    sub_lm: Option<Arc<LM>>,
    instruction: String,
    signature: PhantomData<S>,
}

impl<S: Signature> Rlm<S> {
    pub fn builder() -> RlmBuilder<S> {
        RlmBuilder {
            module: Self {
                config: RlmConfig::default(),
                lm: None,
                sub_lm: None,
                instruction: S::instruction().to_owned(),
                signature: PhantomData,
            },
        }
    }

    #[allow(clippy::result_large_err)] // Match the Module error type.
    pub async fn call(&self, input: S::Input) -> Result<Predicted<S::Output>, PredictError> {
        <Self as Module>::call(self, input).await
    }
}

impl<S: Signature> Module for Rlm<S> {
    type Input = S::Input;
    type Output = S::Output;

    async fn forward(&self, input: S::Input) -> Result<Predicted<S::Output>, PredictError> {
        if self.config.max_iterations == 0 || self.config.max_output_chars == 0 {
            return Err(internal_error(
                "RLM iteration and output limits must be positive",
            ));
        }
        let lm = self
            .lm
            .clone()
            .or_else(try_get_lm)
            .ok_or_else(|| internal_error("Configure an LM for the RLM action loop"))?;
        let sub_lm = self.sub_lm.as_deref().unwrap_or(&lm);
        let input_json = serde_json::to_string(&input).map_err(internal_error)?;
        drop(input);
        let definition = SignatureDef::of::<S>();
        let types = SignatureDef::types_of::<S>();
        let output_schema = json_schema_for_outputs(definition, types);
        let input_schema = OutputSchema::from_shape(S::input_shape());
        let system = format!(
            "{}\n\nYou have a persistent Python session. Return a Python cell each turn. Inspect the input with print and continue from the results. Input fields are globals and are also available in the inputs dictionary. Use llm_query(prompt) or llm_query_batched(prompts) for selected passages. Finish with SUBMIT using the named output fields.\n\nInput type:\n{}\n\nSUBMIT schema:\n{output_schema}",
            self.instruction,
            schema_block(&input_schema.target, &input_schema.types),
        );
        let output_limit = self.config.max_output_chars;
        let (mut session, preview) =
            tokio::task::spawn_blocking(move || Session::new(input_json, output_limit))
                .await
                .map_err(internal_error)?
                .map_err(internal_error)?;
        let mut chat = Chat::new(vec![Message::system(system), Message::user(preview)]);

        let mut metadata = CallMetadata::default();
        let mut remaining = self.config.max_llm_calls;
        for _ in 0..self.config.max_iterations {
            let mut response = lm
                .call_with_tool_loop_mode(chat, Vec::new(), ToolLoopMode::CallerManaged)
                .await
                .map_err(|source| PredictError::Lm {
                    source: LmError::Provider {
                        provider: lm.config.model.clone(),
                        message: source.to_string(),
                        source: Some(source.into()),
                    },
                })?;
            record_call(&mut metadata, &mut response);
            let source = response.output.text_content();
            append_response(&mut metadata.raw_response, &source);
            chat = response.chat;

            let CellOutput {
                stdout,
                error,
                submission,
            } = run_cell(
                &mut session,
                python_source(source),
                sub_lm,
                &mut remaining,
                &mut metadata,
            )
            .await
            .map_err(internal_error)?;

            let mut feedback =
                format!("Python output:\n{stdout}\nSubmodel calls remaining: {remaining}.");
            if let Some(error) = error {
                write!(feedback, "\nPython error: {error}").expect("writing to String");
            }
            if let Some(submission) = submission {
                match parse_submission::<S>(&submission, &mut metadata) {
                    Ok(output) => return Ok(Predicted::new(output, metadata)),
                    Err(error) => {
                        write!(feedback, "\nSUBMIT rejected: {error}").expect("writing to String")
                    }
                }
            }
            chat.push_message(Message::user(feedback));
        }

        if !self.config.enable_extraction_fallback {
            return Err(PredictError::Parse {
                source: ParseError::ValidationFailed {
                    feedback: format!(
                        "RLM reached {} iterations without a valid submission",
                        self.config.max_iterations
                    ),
                },
                raw_response: metadata.raw_response,
                lm_usage: metadata.lm_usage,
            });
        }

        let instruction = format!(
            "{}\n\nUse the Python execution transcript to answer the task.",
            self.instruction,
        );
        chat.messages[0] =
            Message::system(ChatAdapter.build_system_def(definition, types, Some(&instruction)));
        let finalize = Predict::<S>::builder().named("rlm.finalize").build();
        let (prediction, _) = finalize.call_and_parse_with_lm(chat, lm).await?;
        let (output, final_metadata) = prediction.into_parts();
        append_response(&mut metadata.raw_response, &final_metadata.raw_response);
        metadata.lm_usage = metadata.lm_usage + final_metadata.lm_usage;
        metadata.call_ids.extend(final_metadata.call_ids);
        metadata.tool_calls.extend(final_metadata.tool_calls);
        metadata
            .tool_executions
            .extend(final_metadata.tool_executions);
        metadata.span_id = final_metadata.span_id;
        metadata.field_meta = final_metadata.field_meta;
        Ok(Predicted::new(output, metadata))
    }
}

pub struct RlmBuilder<S: Signature> {
    module: Rlm<S>,
}

impl<S: Signature> RlmBuilder<S> {
    pub fn lm(mut self, lm: impl Into<Arc<LM>>) -> Self {
        self.module.lm = Some(lm.into());
        self
    }

    pub fn sub_lm(mut self, lm: impl Into<Arc<LM>>) -> Self {
        self.module.sub_lm = Some(lm.into());
        self
    }

    pub fn instruction(mut self, instruction: impl Into<String>) -> Self {
        self.module.instruction = instruction.into();
        self
    }

    pub fn max_iterations(mut self, limit: usize) -> Self {
        self.module.config.max_iterations = limit;
        self
    }

    pub fn max_llm_calls(mut self, limit: usize) -> Self {
        self.module.config.max_llm_calls = limit;
        self
    }

    pub fn max_output_chars(mut self, limit: usize) -> Self {
        self.module.config.max_output_chars = limit;
        self
    }

    pub fn enable_extraction_fallback(mut self, enabled: bool) -> Self {
        self.module.config.enable_extraction_fallback = enabled;
        self
    }

    pub fn build(self) -> Rlm<S> {
        self.module
    }
}

fn python_source(response: String) -> String {
    let mut python = String::new();
    let mut inside = false;
    let mut has_fences = false;
    for line in response.lines() {
        match (inside, line.trim()) {
            (true, "```") => inside = false,
            (true, _) => {
                python.push_str(line);
                python.push('\n');
            }
            (false, "```" | "```py" | "```python") => {
                inside = true;
                has_fences = true;
            }
            (false, _) => {}
        }
    }
    if has_fences { python } else { response }
}

async fn run_cell(
    session: &mut Session,
    source: String,
    sub_lm: &LM,
    remaining: &mut usize,
    metadata: &mut CallMetadata,
) -> Result<CellOutput, Box<dyn Error + Send + Sync>> {
    let mut worker = session.execute(source);
    loop {
        tokio::select! {
            result = &mut worker => return Ok(result??),
            Some(mut query) = session.queries.recv() => {
                let count = query.prompts.len().min(*remaining);
                *remaining -= count;
                query.prompts.truncate(count);
                let results = join_all(query.prompts.into_iter().map(|prompt| async move {
                    let chat = Chat::new(vec![Message::user(prompt)]);
                    sub_lm.call_with_tool_loop_mode(chat, Vec::new(), ToolLoopMode::CallerManaged).await
                }))
                .await;
                let answers = results.into_iter().map(|result| {
                    result.map(|mut response| {
                        record_call(metadata, &mut response);
                        response.output.text_content()
                    }).map_err(|error| error.to_string())
                }).collect::<Vec<_>>();
                let _ = query.response.send(answers.into_iter().collect());
            }
        }
    }
}

fn record_call(metadata: &mut CallMetadata, response: &mut LMResponse) {
    metadata.lm_usage = metadata.lm_usage + response.usage;
    if let Some(history) = response.history.take() {
        metadata.call_ids.push(history.snapshot().id);
    }
    metadata.tool_calls.append(&mut response.tool_calls);
    metadata
        .tool_executions
        .append(&mut response.tool_executions);
}

fn append_response(transcript: &mut String, response: &str) {
    if !transcript.is_empty() {
        transcript.push_str("\n\n");
    }
    transcript.push_str(response);
}

fn parse_submission<S: Signature>(
    submission: &str,
    metadata: &mut CallMetadata,
) -> Result<S::Output, String> {
    let (values, mut fields) = ChatAdapter
        .parse_output_json_def(
            SignatureDef::of::<S>(),
            SignatureDef::types_of::<S>(),
            &Message::assistant(submission),
        )
        .map_err(|error| match error {
            ParseError::Multiple { errors, .. } => errors
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n"),
            other => other.to_string(),
        })?;
    let output =
        serde_json::from_value(Value::Object(values)).map_err(|error| error.to_string())?;
    metadata.field_meta = S::schema()
        .output_fields()
        .iter()
        .filter_map(|field| {
            let name = field.path().iter().last().unwrap_or(field.lm_name);
            fields
                .shift_remove(name)
                .map(|meta| (field.rust_name.clone(), meta))
        })
        .collect();
    Ok(output)
}

fn internal_error(source: impl Into<Box<dyn Error + Send + Sync>>) -> PredictError {
    PredictError::Params {
        name: "Rlm".into(),
        source: source.into(),
    }
}
