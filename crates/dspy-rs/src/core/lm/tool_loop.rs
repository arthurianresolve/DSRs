//! Tool declarations, prompt protocol, and automatic tool rounds.

use anyhow::Result;
use rig::{
    completion::{AssistantContent, CompletionRequest, CompletionResponse},
    message::{ToolCall, ToolChoice},
    tool::ToolDyn,
};
use std::{borrow::Cow, collections::HashMap, sync::Arc};
use tracing::{debug, trace, warn};

use super::{LM, LmUsage, Message, PreparedCallOptions};
use crate::core::history::CallProgress;
use crate::trace::SpanEvent;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolLoopMode {
    Auto,
    CallerManaged,
}

/// Failures of the automatic tool loop. Missing tools are sent back to the
/// model as recoverable feedback; limit and execution failures stop the call.
#[derive(Debug, thiserror::Error)]
pub enum ToolLoopError {
    #[error("Max tool iterations reached (limit {limit})")]
    IterationLimit { limit: usize },
    #[error("Tool '{name}' not found")]
    UnknownTool { name: String },
    #[error("tool `{name}` execution failed ({context}): {source}")]
    Execution {
        name: String,
        context: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

/// Pre-fetched tool definitions plus name-indexed executors.
///
/// Fetching a rig `ToolDyn` definition costs a boxed-future allocation and a
/// fresh `ToolDefinition` build — doing that per tool on every LM call is pure
/// waste when the tool set is fixed. Build a `ToolSet` once (e.g.
/// [`Predict`](crate::Predict) caches one per instance) and reuse it across calls
/// via [`LM::call_with_toolset`].
#[derive(Clone, Default)]
pub struct ToolSet {
    definitions: Vec<rig::completion::ToolDefinition>,
    by_name: HashMap<String, Arc<dyn ToolDyn>>,
}

impl ToolSet {
    /// Fetches every tool's definition once and indexes the executors by
    /// definition name. Duplicate names keep the first tool.
    pub async fn build(tools: &[Arc<dyn ToolDyn>]) -> Self {
        let mut definitions = Vec::with_capacity(tools.len());
        let mut by_name: HashMap<String, Arc<dyn ToolDyn>> = HashMap::with_capacity(tools.len());
        for tool in tools {
            let definition = tool.definition(String::new()).await;
            by_name
                .entry(definition.name.clone())
                .or_insert_with(|| Arc::clone(tool));
            definitions.push(definition);
        }
        Self {
            definitions,
            by_name,
        }
    }

    /// Builds a `ToolSet` from pre-built definitions with **no executors** —
    /// for caller-managed loops where the caller executes tools itself (the IR
    /// interpreter's `AgentLoop` builds definitions from declared tool
    /// signatures and overlay-resolved descriptions).
    pub fn from_definitions(definitions: Vec<rig::completion::ToolDefinition>) -> Self {
        Self {
            definitions,
            by_name: HashMap::new(),
        }
    }

    /// Collapses `tools` into a single sandboxed `run_js` tool (Code Mode):
    /// instead of emitting one JSON tool call per step, the model writes
    /// JavaScript against the tools as a JS API and composes their results in
    /// one execution. Drop the returned set into any tool loop
    /// ([`LM::call_with_toolset`], `Predict`) exactly like a normal `ToolSet`.
    ///
    /// Errors if two tool names mangle to the same JS identifier (see
    /// [`dsrs_tools::js_identifier`]).
    pub async fn code_mode(
        tools: Vec<Arc<dyn ToolDyn>>,
        config: dsrs_tools::SandboxConfig,
    ) -> Result<Self> {
        let tool = dsrs_tools::CodeModeTool::new(tools, config).await?;
        Ok(Self::build(&[Arc::new(tool) as Arc<dyn ToolDyn>]).await)
    }

    pub fn is_empty(&self) -> bool {
        self.definitions.is_empty()
    }

    pub fn definitions(&self) -> &[rig::completion::ToolDefinition] {
        &self.definitions
    }
}

pub(super) struct ToolRound {
    pub(super) calls: Vec<ToolCall>,
    pub(super) content: rig::OneOrMany<AssistantContent>,
    pub(super) usage: LmUsage,
}

pub(super) struct ToolLoopResult {
    pub(super) message: Message,
    pub(super) chat_history: Vec<rig::message::Message>,
    pub(super) tool_calls: Vec<ToolCall>,
    pub(super) tool_executions: Vec<String>,
}

/// One executed tool call with the details the trace format records.
struct ToolRunRecord {
    call: ToolCall,
    result: String,
    duration_us: u64,
    /// Failure reported back to the model as text (e.g. tool not found).
    error: Option<String>,
}

impl ToolRunRecord {
    fn to_event(&self) -> SpanEvent {
        SpanEvent::ToolRun {
            id: self.call.id.clone(),
            name: self.call.function.name.clone(),
            args: self.call.function.arguments.clone(),
            result: self.result.clone(),
            duration_us: self.duration_us,
            error: self.error.clone(),
        }
    }
}

/// Converts grouped rig assistant content into a single [`Message`] for event
/// recording, preserving reasoning and tool-call blocks.
fn assistant_message_from_content(content: &rig::OneOrMany<AssistantContent>) -> Message {
    Message::from(rig::message::Message::Assistant {
        id: None,
        content: content.clone(),
    })
}

/// What the model actually wants to do, extracted from a potentially multi-block response.
/// Reasoning blocks are preserved in `full_content` for faithful history replay.
pub(super) enum ChoiceAction {
    /// Terminal text response (possibly preceded by reasoning).
    Text(String),
    /// One or more tool calls to execute. Carries the full `OneOrMany` so
    /// reasoning blocks are preserved when we push the assistant turn into
    /// chat history. Supports parallel tool calling (Anthropic multi-tool-use,
    /// OpenAI parallel function calls).
    ToolCalls {
        calls: Vec<ToolCall>,
        full_content: Box<rig::OneOrMany<AssistantContent>>,
        assistant_text: Option<String>,
    },
}

pub(super) fn classify_choice(choice: rig::OneOrMany<AssistantContent>) -> Result<ChoiceAction> {
    let mut text: Option<String> = None;
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut has_image = false;

    for item in choice.iter() {
        match item {
            AssistantContent::ToolCall(tc) => {
                tool_calls.push(tc.clone());
            }
            AssistantContent::Text(t) => {
                text = Some(t.text.clone());
            }
            AssistantContent::Reasoning(_) => {}
            AssistantContent::Image(_) => has_image = true,
        }
    }

    if !tool_calls.is_empty() {
        return Ok(ChoiceAction::ToolCalls {
            calls: tool_calls,
            full_content: Box::new(choice),
            assistant_text: text,
        });
    }

    if let Some(t) = text {
        return Ok(ChoiceAction::Text(t));
    }

    if has_image {
        return Err(crate::LmCallError::UnsupportedContent.into());
    }

    // Fallback: only reasoning blocks — extract display text
    let display = choice
        .iter()
        .filter_map(|item| match item {
            AssistantContent::Reasoning(r) => Some(r.display_text()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    Ok(ChoiceAction::Text(display))
}

/// Render tools and exchanges for endpoints accepting ordinary chat messages only.
pub(super) fn format_prompt_tools(request: &mut CompletionRequest) -> Result<()> {
    if !request.tools.is_empty() {
        let declarations = serde_json::to_string(&request.tools)?;
        let instruction = format!(
            "Available tools: {declarations}\nTo request tools return ONLY a JSON object {{\"tool_calls\":[{{\"id\":\"unique-id\",\"name\":\"tool-name\",\"arguments\":{{}}}}]}}. Arguments must be a JSON object. Tool results will be supplied as text. When no tool is needed, follow the task's final output format. Tool choice policy: {:?}.",
            request.tool_choice
        );
        request.preamble = Some(match request.preamble.take() {
            Some(preamble) => format!("{preamble}\n\n{instruction}"),
            None => instruction,
        });
        request.tools.clear();
        request.tool_choice = None;
        request.output_schema = None;
        request.additional_params = None;
    }
    request.chat_history = rig::OneOrMany::many(
        request
            .chat_history
            .iter()
            .cloned()
            .map(prompt_tool_message),
    )
    .expect("nonempty chat history");
    Ok(())
}

fn prompt_tool_message(message: rig::message::Message) -> rig::message::Message {
    match message {
        rig::message::Message::Assistant { id, content } => {
            let has_calls = content
                .iter()
                .any(|c| matches!(c, AssistantContent::ToolCall(_)));
            let content = content.into_iter().filter_map(|item| match item {
                AssistantContent::Text(text) => {
                    if has_calls
                        && prompt_tool_value(&text.text)
                            .is_some_and(|v| v.get("tool_calls").is_some())
                    {
                        None
                    } else {
                        Some(AssistantContent::Text(text))
                    }
                }
                AssistantContent::ToolCall(call) => {
                    let value = serde_json::json!({"tool_calls":[{
                        "id":call.id, "name":call.function.name,
                        "arguments":call.function.arguments
                    }]});
                    Some(AssistantContent::text(value.to_string()))
                }
                other => Some(other),
            });
            rig::message::Message::Assistant {
                id,
                content: rig::OneOrMany::many(content).expect("nonempty assistant content"),
            }
        }
        rig::message::Message::User { content } => {
            let content = content.into_iter().map(|item| match item {
                rig::message::UserContent::ToolResult(result) => {
                    rig::message::UserContent::text(format!(
                        "Tool result: {}",
                        serde_json::to_string(&result).expect("serializable tool result")
                    ))
                }
                other => other,
            });
            rig::message::Message::User {
                content: rig::OneOrMany::many(content).expect("nonempty user content"),
            }
        }
    }
}

/// Accept field/code-fence wrappers and trailing commas. Quoted arguments are
/// preserved byte-for-byte; invalid argument types are still rejected.
fn prompt_tool_value(text: &str) -> Option<serde_json::Value> {
    if let Ok(value) = serde_json::from_str(text) {
        return Some(value);
    }
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    let bytes = text.get(start..=end)?.as_bytes();
    let mut repaired = Vec::with_capacity(bytes.len());
    let (mut quoted, mut escaped) = (false, false);
    for (index, byte) in bytes.iter().copied().enumerate() {
        if escaped {
            escaped = false;
            repaired.push(byte);
            continue;
        }
        if quoted && byte == b'\\' {
            escaped = true;
            repaired.push(byte);
            continue;
        }
        if byte == b'"' {
            quoted = !quoted;
        }
        if !quoted
            && byte == b','
            && bytes[index + 1..]
                .iter()
                .find(|b| !b.is_ascii_whitespace())
                .is_some_and(|b| *b == b'}' || *b == b']')
        {
            continue;
        }
        repaired.push(byte);
    }
    serde_json::from_slice(&repaired).ok()
}

pub(super) fn decode_prompt_tools(
    response: &mut CompletionResponse<()>,
    required: bool,
) -> Result<()> {
    let mut choice = Vec::new();
    let mut ids = std::collections::HashSet::new();
    for item in response.choice.iter() {
        choice.push(item.clone());
        let AssistantContent::Text(text) = item else {
            continue;
        };
        let Some(value) = prompt_tool_value(&text.text) else {
            continue;
        };
        let Some(calls) = value.get("tool_calls") else {
            continue;
        };
        let calls = calls
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("tool_calls must be an array"))?;
        for call in calls {
            let id = call
                .get("id")
                .and_then(serde_json::Value::as_str)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| anyhow::anyhow!("tool call needs an id"))?;
            anyhow::ensure!(ids.insert(id.to_string()), "duplicate tool call id");
            let name = call
                .get("name")
                .and_then(serde_json::Value::as_str)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| anyhow::anyhow!("tool call needs a name"))?;
            let arguments = call
                .get("arguments")
                .filter(|v| v.is_object())
                .ok_or_else(|| anyhow::anyhow!("tool arguments must be an object"))?;
            choice.push(AssistantContent::tool_call(id, name, arguments.clone()));
        }
    }
    anyhow::ensure!(
        !required
            || choice
                .iter()
                .any(|c| matches!(c, AssistantContent::ToolCall(_))),
        "a tool call was required"
    );
    response.choice = rig::OneOrMany::many(choice).expect("provider choice is nonempty");
    Ok(())
}

impl LM {
    /// Execute all tool calls in a batch concurrently, returning results paired with
    /// their calls in request order.
    async fn execute_tool_batch(
        tools_by_name: &HashMap<String, Arc<dyn ToolDyn>>,
        calls: &[ToolCall],
        context: &str,
    ) -> Result<Vec<ToolRunRecord>> {
        let executions = calls.iter().map(|tc| {
            let tool = tools_by_name.get(&tc.function.name).cloned();
            async move {
                let started = std::time::Instant::now();
                let (result, error) = match tool {
                    Some(tool) => {
                        let result =
                            tool.call(tc.function.arguments.to_string())
                                .await
                                .map_err(|err| ToolLoopError::Execution {
                                    name: tc.function.name.clone(),
                                    context: context.to_owned(),
                                    source: Box::new(err),
                                })?;
                        (result, None)
                    }
                    None => {
                        warn!(tool = %tc.function.name, context, "tool not found");
                        let message = ToolLoopError::UnknownTool {
                            name: tc.function.name.clone(),
                        }
                        .to_string();
                        (message.clone(), Some(message))
                    }
                };
                trace!(tool = %tc.function.name, result_len = result.len(), "tool executed");
                Ok::<_, anyhow::Error>(ToolRunRecord {
                    call: tc.clone(),
                    result,
                    duration_us: started.elapsed().as_micros() as u64,
                    error,
                })
            }
        });

        futures::future::try_join_all(executions).await
    }

    /// Push tool results into chat history as a single User message.
    fn push_tool_results(chat_history: &mut Vec<rig::message::Message>, results: &[ToolRunRecord]) {
        use rig::OneOrMany;
        use rig::message::UserContent;

        let tool_result_contents: Vec<UserContent> = results
            .iter()
            .map(|record| {
                let tc = &record.call;
                if let Some(call_id) = &tc.call_id {
                    UserContent::tool_result_with_call_id(
                        tc.id.clone(),
                        call_id.clone(),
                        OneOrMany::one(record.result.clone().into()),
                    )
                } else {
                    UserContent::tool_result(
                        tc.id.clone(),
                        OneOrMany::one(record.result.clone().into()),
                    )
                }
            })
            .collect();

        chat_history.push(rig::message::Message::User {
            content: OneOrMany::many(tool_result_contents).expect("results should not be empty"),
        });
    }

    #[tracing::instrument(
        name = "dsrs.lm.tools.loop",
        level = "debug",
        skip_all,
        fields(
            initial_tool_count = round.calls.len(),
            max_iterations = self.config.max_tool_iterations as usize
        )
    )]
    pub(super) async fn execute_tool_loop(
        &self,
        mut round: ToolRound,
        tools: &ToolSet,
        mut chat_history: Vec<rig::message::Message>,
        system_prompt: &str,
        options: PreparedCallOptions<'_>,
        progress: &mut CallProgress,
    ) -> Result<ToolLoopResult> {
        let max_iterations = self.config.max_tool_iterations as usize;
        let mut all_tool_calls = Vec::new();
        let mut all_tool_executions = Vec::new();
        let mut executed = 0;

        loop {
            progress.events.push(SpanEvent::Exchange {
                message: assistant_message_from_content(&round.content),
                usage: round.usage,
            });
            if executed == max_iterations {
                warn!(max_iterations, "max tool iterations reached");
                return Err(ToolLoopError::IterationLimit {
                    limit: max_iterations,
                }
                .into());
            }

            let context = if executed == 0 {
                Cow::Borrowed("initial")
            } else {
                Cow::Owned(format!("iteration {executed}"))
            };
            debug!(
                round = executed,
                count = round.calls.len(),
                "executing tool calls"
            );
            let results = Self::execute_tool_batch(&tools.by_name, &round.calls, &context).await?;
            for record in &results {
                all_tool_calls.push(record.call.clone());
                all_tool_executions.push(record.result.clone());
                progress.events.push(record.to_event());
            }

            // Providers with signed reasoning expect the complete assistant turn back.
            chat_history.push(rig::message::Message::Assistant {
                id: None,
                content: round.content,
            });
            Self::push_tool_results(&mut chat_history, &results);
            executed += 1;

            // A text completion remains allowed after the final tool round.
            let response = self
                .completion_with_retry(
                    || {
                        self.build_completion_request(
                            system_prompt,
                            &chat_history,
                            tools.definitions(),
                            Some(ToolChoice::Auto),
                            options,
                        )
                    },
                    progress,
                )
                .await?;

            let round_usage = LmUsage::from(response.usage);
            let accumulated_usage = progress.total_usage();
            debug!(
                rounds = executed,
                prompt_tokens = accumulated_usage.prompt_tokens,
                completion_tokens = accumulated_usage.completion_tokens,
                total_tokens = accumulated_usage.total_tokens,
                "tool loop usage updated"
            );

            // Scan ALL content blocks — don't just look at .first(), since
            // responses can be [Reasoning, ToolCall] or [Reasoning, Text].
            match classify_choice(response.choice)? {
                ChoiceAction::Text(text) => {
                    debug!(rounds = executed, "tool loop completed with text");
                    let message = Message::assistant(text);
                    progress.events.push(SpanEvent::Exchange {
                        message: message.clone(),
                        usage: round_usage,
                    });
                    return Ok(ToolLoopResult {
                        message,
                        chat_history,
                        tool_calls: all_tool_calls,
                        tool_executions: all_tool_executions,
                    });
                }
                ChoiceAction::ToolCalls {
                    calls,
                    full_content,
                    ..
                } => {
                    round = ToolRound {
                        calls,
                        content: *full_content,
                        usage: round_usage,
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_tools_enforce_required_calls_and_unique_ids_across_blocks() {
        let mut response = CompletionResponse {
            choice: rig::OneOrMany::one(AssistantContent::text("done")),
            usage: rig::completion::Usage::new(),
            raw_response: (),
            message_id: None,
        };
        assert!(decode_prompt_tools(&mut response, true).is_err());
        let text = AssistantContent::text(
            r#"{"tool_calls":[{"id":"same","name":"echo","arguments":{}}]}"#,
        );
        response.choice = rig::OneOrMany::many([text.clone(), text]).unwrap();
        assert!(decode_prompt_tools(&mut response, false).is_err());
    }

    #[test]
    fn prompt_envelopes_allow_wrappers_without_modifying_argument_strings() {
        let value = prompt_tool_value("[[ ## calls ## ]]\n{\"tool_calls\":[{\"id\":\"id\",\"name\":\"echo\",\"arguments\":{\"text\":\",}\",},},],}\n[[ ## completed ## ]]").unwrap();
        assert_eq!(value["tool_calls"][0]["arguments"]["text"], ",}");
    }
}
