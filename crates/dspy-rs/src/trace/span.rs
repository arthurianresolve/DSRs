//! The unified trace format (RFC 0001): one [`Span`] per `Predict` invocation,
//! one [`Trace`] per rollout.
//!
//! Spans are addressed by the same name the params system uses (fx slot name or
//! facet dotted path), interned per-trace as a [`CompId`]. The wire form is
//! JSONL (see [`Trace::to_jsonl`]); the in-memory form always speaks interned ids.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use crate::core::lm::url_origin;
use crate::utils::hash::{HashWriter, StableHasher, stable_hash_debug};
use crate::{Chat, ContentBlock, LMConfig, LmUsage, Message, Role};

/// Index of a span within its trace. Dense, assigned in insertion order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SpanId(pub u32);

/// Per-trace interned component index. The component *name* is the same string
/// the params system addresses: an fx slot name (`"drafter"`) or a facet dotted
/// path (`"pipeline.rater"`). `Trace::components` maps `CompId` -> name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CompId(pub u32);

/// Per-trace interned prompt prefix (system message + demo turns). One entry per
/// distinct (component, candidate) configuration — demos serialize once per
/// trace, not once per span.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PrefixId(pub u32);

/// Per-trace interned model configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(pub u32);

pub type JsonMap = serde_json::Map<String, serde_json::Value>;

/// One `Predict` invocation: one rendered prompt in, one parsed output out,
/// with the tool loop's inner structure as ordered events.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Span {
    pub id: SpanId,
    /// Who ran. Resolve to the name via [`Trace::component_name`].
    pub component: CompId,
    /// 0-based invocation index of this component within this rollout.
    /// `(component, seq)` is unique per trace — this is what makes loops
    /// addressable and what replay keys on.
    pub seq: u32,
    /// Enclosing span when this invocation ran within a parent span's future
    /// scope. Top-level sibling invocations each have no parent.
    pub parent: Option<SpanId>,

    // ---- request (recorded eagerly at span open) ----
    /// Interned system+demos prefix. `None` when the call had no prefix
    /// (e.g. a chat-level continuation on a caller-owned chat).
    pub prefix: Option<PrefixId>,
    /// The live suffix of the rendered prompt: the user turn (and, for
    /// multi-turn continuations, the full caller-provided history).
    /// `prefix + suffix` reconstructs the exact `Chat` sent.
    pub suffix: Vec<Message>,
    /// Signature input fields as JSON. `None` for continuations where no typed
    /// input exists.
    pub input: Option<JsonMap>,
    pub model: ModelId,
    /// Stable hash over (redacted model config ++ full rendered prompt),
    /// computed at span close. The replay key and determinism check.
    pub request_hash: u64,

    // ---- execution (filled at span close) ----
    /// Ordered provider exchanges, tool executions, and orchestration edits.
    pub events: Vec<SpanEvent>,
    /// Final assistant text, pre-parse.
    pub raw_output: Option<String>,
    /// Parsed signature output fields as JSON. `None` = call or parse failed.
    pub output: Option<JsonMap>,
    /// Aggregated across all exchanges.
    pub usage: LmUsage,
    pub error: Option<SpanError>,
    /// Span-level metric result, when the metric attached one after the
    /// rollout ran (see `TypedMetric::evaluate_spans`). `None` means no
    /// per-span credit was assigned — demo harvesting then falls back to the
    /// whole-rollout score in [`TraceOutcome::eval`]. Additive under §5.1's
    /// rules — no format version bump; absent on the wire when `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval: Option<Eval>,

    // ---- timing ----
    /// Microseconds since UNIX epoch.
    pub started_at_us: u64,
    pub duration_us: u64,

    /// False when any text field was truncated at serialization (§5.4) or the
    /// span was redacted. Replay refuses incomplete spans.
    #[serde(default = "default_true")]
    pub complete: bool,
}

fn default_true() -> bool {
    true
}

/// Ordered events inside a span. Tagged; readers skip unknown tags — this is
/// the streaming seam (RFC 0001 §6).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
#[non_exhaustive]
pub enum SpanEvent {
    /// One provider round-trip: the full assistant message (text, tool-call
    /// blocks, reasoning blocks) and that round-trip's own usage.
    Exchange { message: Message, usage: LmUsage },
    /// One tool execution between exchanges.
    ToolRun {
        /// Provider tool-call id.
        id: String,
        name: String,
        args: serde_json::Value,
        /// Tool result text as fed back to the model.
        result: String,
        duration_us: u64,
        /// Tool-level failure that was reported back to the model as text.
        error: Option<String>,
    },
    /// History removed before a provider exchange. Indices refer to the chat
    /// after applying preceding events in this span.
    HistoryPruned { start: usize, count: usize },
    /// A message added by orchestration outside a provider exchange.
    ChatMessage { message: Message },
    /// Unknown tag from a newer writer; preserved as a placeholder on read,
    /// dropped from the canonical JSONL on re-serialize.
    #[doc(hidden)]
    #[serde(other)]
    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpanError {
    pub kind: SpanErrorKind,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanErrorKind {
    /// Provider/network failure.
    Lm,
    /// Response received but output parsing failed. `raw_output` is still
    /// populated in this case — parse failures are prime reflection material.
    Parse,
    /// A tool execution aborted the loop.
    Tool,
    /// Scope ended while the span was open (task cancelled).
    Cancelled,
}

impl SpanErrorKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Lm => "lm",
            Self::Parse => "parse",
            Self::Tool => "tool",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Interned model configuration: [`LMConfig`] minus live state, minus secrets.
/// `api_key` is `#[serde(skip)]` on `LMConfig` already; `base_url` is
/// additionally reduced to origin (scheme+host+port) at intern time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelEntry {
    pub config: LMConfig,
    /// Identity hash of the redacted config and endpoint; part of `request_hash`.
    /// Endpoints with private path, query, or userinfo use a process-local key,
    /// so their strict replay identity cannot survive a process restart.
    pub config_hash: u64,
}

impl ModelEntry {
    /// Builds the redacted, hashable entry from a live config.
    pub fn from_config(config: &LMConfig) -> Self {
        // A process-local key keeps URL credentials out of the exported hash
        // while still distinguishing custom endpoints during live replay.
        static ENDPOINT_HASHER: LazyLock<std::collections::hash_map::RandomState> =
            LazyLock::new(std::collections::hash_map::RandomState::new);
        let endpoint_identity = config.base_url.as_ref().map(|url| {
            let origin = url_origin(url);
            if *url == origin || *url == format!("{origin}/v1") || *url == format!("{origin}/v1/") {
                return url.clone();
            }
            let mut hasher = ENDPOINT_HASHER.build_hasher();
            url.hash(&mut hasher);
            format!("private:{}", hasher.finish())
        });
        let config = Self::redacted_config(config);
        let config_hash = stable_hash_debug(&(&config, endpoint_identity));
        Self {
            config,
            config_hash,
        }
    }

    pub(crate) fn redacted_config(config: &LMConfig) -> LMConfig {
        let mut config = config.clone();
        config.temperature = crate::core::sampling::effective_temperature(config.temperature);
        config.api_key = None;
        config.base_url = config.base_url.as_deref().map(url_origin);
        config
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrefixEntry {
    /// System message + demo user/assistant turns, exactly as the predictor
    /// rendered them.
    pub messages: Vec<Message>,
}

/// One rollout: ordered spans plus per-trace intern tables.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Trace {
    pub meta: TraceMeta,
    /// `CompId` -> component name (fx name / dotted path).
    pub components: Vec<String>,
    /// RFC 0001 §1's reserved join column, parallel to `components`: the
    /// global [`ParamId`](crate::ir::ParamId)s of each component's slots in a
    /// program (per RFC 0002 §3.3). Empty until
    /// [`attach_program`](Trace::attach_program) fills it; a `None` entry is
    /// a component the program has no leaf for (static-lane harnesses leave
    /// the whole column empty). Additive — no format version bump.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub param_ids: Vec<Option<Vec<crate::ir::ParamId>>>,
    pub models: Vec<ModelEntry>,
    pub prefixes: Vec<PrefixEntry>,
    /// Insertion order == start order == (for the sequential case) topological
    /// order — the same invariant the old Graph had.
    pub spans: Vec<Span>,
    /// Rollout-level result, filled after the scope closes / metric runs.
    pub outcome: Option<TraceOutcome>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TraceMeta {
    /// Format version. This RFC: 1.
    pub v: u32,
    /// Unique id string, minted at capture start.
    pub trace_id: String,
    pub started_at_us: u64,
    /// Hash of the candidate under evaluation, when the harness recorded one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_hash: Option<u64>,
    /// Rollout input (the example's input fields), when the harness entry
    /// point recorded it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<JsonMap>,
    /// Free-form run tags ("optimizer": "gepa", "gen": "3").
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tags: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TraceOutcome {
    /// Rollout output fields (the final prediction, serialized).
    pub output: Option<JsonMap>,
    pub error: Option<String>,
    /// Metric result, if a metric ran over this rollout.
    pub eval: Option<Eval>,
    pub duration_us: u64,
}

/// The metric result type: one score, optional textual feedback.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Eval {
    pub score: f64,
    pub feedback: Option<String>,
}

impl Eval {
    /// Score-only result. Sufficient for COPRO and MIPROv2; GEPA requires
    /// [`with_feedback`](Eval::with_feedback).
    pub fn score(score: f64) -> Self {
        Self {
            score,
            feedback: None,
        }
    }

    /// Score plus textual feedback explaining *why* — GEPA's reflection input.
    pub fn with_feedback(score: f64, feedback: impl Into<String>) -> Self {
        Self {
            score,
            feedback: Some(feedback.into()),
        }
    }
}

impl Trace {
    pub fn component_id(&self, name: &str) -> Option<CompId> {
        self.components
            .iter()
            .position(|c| c == name)
            .map(|idx| CompId(idx as u32))
    }

    pub fn component_name(&self, id: CompId) -> &str {
        &self.components[id.0 as usize]
    }

    /// Every invocation of one component, in order — GEPA's `pred_trace`.
    pub fn for_component<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Span> + 'a {
        let id = self.component_id(name);
        self.spans
            .iter()
            .filter(move |span| Some(span.component) == id)
    }

    /// Reconstructs the exact rendered prompt of a span (prefix ++ suffix).
    pub fn prompt(&self, span: &Span) -> Vec<Message> {
        let prefix = span
            .prefix
            .map(|id| self.prefixes[id.0 as usize].messages.as_slice())
            .unwrap_or(&[]);
        let mut messages = Vec::with_capacity(prefix.len() + span.suffix.len());
        messages.extend(prefix.iter().cloned());
        messages.extend(span.suffix.iter().cloned());
        messages
    }

    /// Resolves a span's model config.
    pub fn model(&self, span: &Span) -> &LMConfig {
        &self.models[span.model.0 as usize].config
    }

    /// Spans that completed with parsed output.
    pub fn successes(&self) -> impl Iterator<Item = &'_ Span> + '_ {
        self.spans.iter().filter(|span| span.output.is_some())
    }

    /// Fills the [`param_ids`](Trace::param_ids) join column against a
    /// program (RFC 0002 §3.3): for each interned component whose name is one
    /// of the program's leaf names, the entry becomes that leaf's node-owned
    /// [`ParamId`](crate::ir::ParamId)s in id order — the same entities
    /// `Program::param_id("<leaf>.<slot>")` addresses. Components the program
    /// doesn't know (static-lane names, judge calls) get `None`.
    ///
    /// One addressing story: a span's `component` string == the leaf name ==
    /// the `ParamPath` prefix, so after attaching, spans join to optimizable
    /// slots without string surgery.
    pub fn attach_program(&mut self, program: &crate::ir::Program) {
        let mut by_leaf: std::collections::HashMap<&str, Vec<crate::ir::ParamId>> =
            std::collections::HashMap::new();
        for (id, slot) in program.params.iter() {
            if let crate::ir::ParamOwner::Node(node) = slot.owner
                && let Some(leaf) = program.leaf_name(node)
            {
                by_leaf.entry(leaf).or_default().push(id);
            }
        }
        self.param_ids = self
            .components
            .iter()
            .map(|name| by_leaf.get(name.as_str()).cloned())
            .collect();
    }
}

impl Span {
    /// Rebuilds everything the policy emitted for this span from its events:
    /// each `Exchange`'s assistant message verbatim, with consecutive
    /// `ToolRun`s batched into a single user-role tool-result turn — the same
    /// shape the live tool loop feeds back to the model.
    ///
    /// Used by the RL export's `completion` field. Orchestration edits affect
    /// the continuation chat independently of the policy's completions.
    pub fn completion_messages(&self) -> Vec<Message> {
        let mut messages = Vec::new();
        self.append_messages(&mut messages, false)
            .expect("completion export does not apply history edits");
        messages
    }

    pub(crate) fn replay_chat(&self, mut messages: Vec<Message>) -> Option<Chat> {
        if !self.append_messages(&mut messages, true)?
            && let Some(raw) = self.raw_output.as_deref().filter(|raw| !raw.is_empty())
        {
            messages.push(Message::assistant(raw));
        }
        Some(Chat::new(messages))
    }

    /// Returns whether an event appended a message. A malformed history range
    /// in an imported trace cannot produce a replayable chat.
    fn append_messages(&self, out: &mut Vec<Message>, apply_edits: bool) -> Option<bool> {
        fn flush(out: &mut Vec<Message>, pending: &mut Vec<ContentBlock>) {
            if !pending.is_empty() {
                out.push(Message::with_content(Role::User, std::mem::take(pending)));
            }
        }

        let mut pending = Vec::new();
        let mut appended = false;
        let mut exchange: Option<&Message> = None;
        for event in &self.events {
            match event {
                SpanEvent::Exchange { message, .. } => {
                    flush(out, &mut pending);
                    out.push(message.clone());
                    exchange = Some(message);
                    appended = true;
                }
                SpanEvent::ToolRun { id, result, .. } => {
                    let call_id = exchange
                        .and_then(|message| message.tool_calls_iter().find(|call| call.id == *id))
                        .and_then(|call| call.call_id.as_deref());
                    pending.push(ContentBlock::tool_result_text(id, call_id, result.clone()));
                    appended = true;
                }
                SpanEvent::HistoryPruned { start, count } if apply_edits => {
                    flush(out, &mut pending);
                    let range = *start..start.checked_add(*count)?;
                    out.get(range.clone())?;
                    out.drain(range);
                }
                SpanEvent::ChatMessage { message } if apply_edits => {
                    flush(out, &mut pending);
                    out.push(message.clone());
                    appended = true;
                }
                _ => {}
            }
        }
        flush(out, &mut pending);
        Some(appended)
    }

    pub(crate) fn recorded_tool_calls(&self) -> Vec<rig::message::ToolCall> {
        self.events
            .iter()
            .filter_map(|event| match event {
                SpanEvent::Exchange { message, .. } => Some(message),
                _ => None,
            })
            .flat_map(Message::tool_calls_iter)
            .cloned()
            .collect()
    }
}

#[derive(Clone)]
pub(crate) struct RequestHashState(StableHasher);

impl RequestHashState {
    pub(crate) fn new(config_hash: u64, sampling_id: Option<&str>) -> Self {
        let mut hasher = StableHasher::new();
        hasher.write(&config_hash.to_le_bytes());
        if let Some(id) = sampling_id {
            hasher.write(&(id.len() as u64).to_le_bytes());
            hasher.write(id.as_bytes());
        }
        Self(hasher)
    }

    pub(crate) fn append(&mut self, messages: &[Message]) {
        let mut writer = HashWriter(&mut self.0);
        for message in messages {
            let _ = write!(writer, "{message:?}");
        }
    }

    pub(crate) fn finish(self) -> u64 {
        self.0.finish()
    }
}

/// Computes a span's `request_hash`: stable hash over the redacted model config
/// hash and the full rendered prompt (prefix ++ suffix), streamed through the
/// messages' `Debug` representation.
pub(crate) fn request_hash(config_hash: u64, prefix: &[Message], suffix: &[Message]) -> u64 {
    let sampling = crate::core::sampling::current_sampling_options();
    let mut state = RequestHashState::new(config_hash, sampling.id.as_deref());
    state.append(prefix);
    state.append(suffix);
    state.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_origin_strips_secrets_and_paths() {
        assert_eq!(
            url_origin("https://user:pass@vllm.internal:8000/v1?api_key=sk-123"),
            "https://vllm.internal:8000"
        );
        assert_eq!(url_origin("http://localhost:8080"), "http://localhost:8080");
        assert_eq!(url_origin("localhost:8080/v1"), "localhost:8080");
    }

    #[test]
    fn model_entry_redacts_api_key() {
        let config = LMConfig {
            api_key: Some("sk-secret".to_string()),
            base_url: Some("https://token@host/path".to_string()),
            ..LMConfig::default()
        };
        let entry = ModelEntry::from_config(&config);
        assert_eq!(entry.config.api_key, None);
        assert_eq!(entry.config.base_url.as_deref(), Some("https://host"));
        assert_ne!(
            entry.config_hash,
            stable_hash_debug(&(entry.config.clone(), config.base_url))
        );
    }

    #[test]
    fn request_hash_changes_with_prompt_and_config() {
        let prefix = vec![Message::system("sys")];
        let suffix = vec![Message::user("hi")];
        let base = request_hash(1, &prefix, &suffix);
        assert_eq!(base, request_hash(1, &prefix, &suffix));
        assert_ne!(base, request_hash(2, &prefix, &suffix));
        assert_ne!(base, request_hash(1, &prefix, &[Message::user("bye")]));
    }

    #[test]
    fn output_format_changes_strict_replay_request_hash() {
        let def = crate::ir::SignatureDef::build("ReplayFormat")
            .input("question", crate::typesys::FieldType::String)
            .output("answer", crate::typesys::FieldType::String)
            .finish()
            .unwrap();
        let types = crate::typesys::TypeTable::default();
        let adapter = crate::ChatAdapter;
        let schema_prompt = adapter.build_system_json_def(&def, &types, None, true);
        let json_mode_prompt = adapter.build_system_json_def(&def, &types, None, false);
        let model_hash = ModelEntry::from_config(&LMConfig::default()).config_hash;
        let schema_hash = request_hash(model_hash, &[Message::system(schema_prompt)], &[]);
        let json_mode_hash = request_hash(model_hash, &[Message::system(json_mode_prompt)], &[]);
        assert_ne!(schema_hash, json_mode_hash);
    }

    #[test]
    fn exact_custom_endpoint_changes_trace_identity_without_serializing_path() {
        let left = LMConfig {
            base_url: Some("https://api.example/v1/first".into()),
            ..LMConfig::default()
        };
        let right = LMConfig {
            base_url: Some("https://api.example/v1/second".into()),
            ..LMConfig::default()
        };
        let left_entry = ModelEntry::from_config(&left);
        let right_entry = ModelEntry::from_config(&right);
        assert_eq!(
            left_entry.config_hash,
            ModelEntry::from_config(&left).config_hash
        );
        assert_eq!(left_entry.config.base_url, right_entry.config.base_url);
        assert_ne!(left_entry.config_hash, right_entry.config_hash);
        assert!(!format!("{:?}", left_entry.config).contains("/v1/first"));
        let public_guess = stable_hash_debug(&(left_entry.config.clone(), left.base_url));
        assert_ne!(left_entry.config_hash, public_guess);

        let common = LMConfig {
            base_url: Some("http://localhost/v1".into()),
            ..LMConfig::default()
        };
        let common_entry = ModelEntry::from_config(&common);
        assert_eq!(
            common_entry.config_hash,
            stable_hash_debug(&(common_entry.config.clone(), common.base_url))
        );
    }

    #[tokio::test]
    async fn request_hash_includes_sampling_identity_and_temperature() {
        let prompt = [Message::user("hello")];
        let config = LMConfig::default();
        let plain = request_hash(ModelEntry::from_config(&config).config_hash, &[], &prompt);
        let first = crate::with_sampling_options(
            crate::SamplingOptions {
                id: Some("sample-a".into()),
                temperature: Some(0.2),
            },
            async { request_hash(ModelEntry::from_config(&config).config_hash, &[], &prompt) },
        )
        .await;
        let second = crate::with_sampling_options(
            crate::SamplingOptions {
                id: Some("sample-b".into()),
                temperature: Some(0.2),
            },
            async { request_hash(ModelEntry::from_config(&config).config_hash, &[], &prompt) },
        )
        .await;
        let changed_temperature = crate::with_sampling_options(
            crate::SamplingOptions {
                id: Some("sample-a".into()),
                temperature: Some(0.8),
            },
            async { request_hash(ModelEntry::from_config(&config).config_hash, &[], &prompt) },
        )
        .await;
        assert_ne!(plain, first);
        assert_ne!(first, second);
        assert_ne!(first, changed_temperature);
    }
}
