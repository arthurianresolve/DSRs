//! The IR interpreter (RFC 0002 §3): async evaluation of a loaded [`Program`].
//!
//! - **Loading** ([`Interpreter::load`]) front-loads every check: validation,
//!   capability grants, model binding, host-tool binding, and the full
//!   dsrs-tools lifecycle (parse → compile → register) for every sandboxed
//!   tool and hole. A hole that doesn't compile fails the *load*, not the call.
//! - **Run state** is a call-scoped `Cx`, never stored on the interpreter —
//!   concurrent runs (and concurrent overlays) share one `Arc<Program>`.
//! - **Overlay read-through**: instruction/demos/model/context/code resolve
//!   through the optional [`Overlay`](crate::ir::params::Overlay) at render
//!   time. The program is never mutated.
//! - **Spans** (RFC 0001): each leaf evaluation emits one span through the
//!   ambient capture scope, `component` = the program-unique leaf name. An
//!   `AgentLoop` is one span with N `Exchange`/`ToolRun` events; a `Hole` is
//!   one span whose model is the reserved `sandbox:quickjs` config.

use std::collections::{HashMap, HashSet};
use std::pin::pin;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};
use std::time::Instant;

use cranelift_entity::{PrimaryMap, SecondaryMap};
use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde_json::Value;
use tokio::sync::OnceCell;

use crate::adapter::chat::{ChatAdapter, PreparedInputTemplates};
use crate::core::FieldMeta;
use crate::ir::graph::{
    AgentLoopNode, Binding, BudgetPolicy, CapSet, HoleImpl, HoleNode, ModelId, Node, NodeId,
    PortRef, PredictNode, Program, SigId, ToolId, ToolKind,
};
use crate::ir::params::{ContextPolicy, DemoRow, Overlay, ParamId, ParamOwner, ParamValue};
use crate::ir::sig::SignatureDef;
use crate::ir::validate::{ValidateError, json_matches_type};
use crate::trace::capture::{begin_span_lazy, in_span};
use crate::trace::replay::PrepareReplayOutput;
use crate::trace::{JsonMap, SpanEvent, SpanGuard, SpanOutcome, SpanRequest};
pub use crate::typesys::json_schema::input_schema_of;
use crate::typesys::{FieldType, TypeTable, json_schema_compatible, json_schema_for_outputs};
use crate::{
    Chat, LM, LMCallOptions, LMConfig, LmError, LmUsage, Message, Role, ToolLoopMode, ToolSet,
};

// ---------------------------------------------------------------------------
// Budgets
// ---------------------------------------------------------------------------

/// Run-level spend limits. `None` = unlimited.
#[derive(Clone, Copy, Debug, Default)]
pub struct Budget {
    pub max_lm_calls: Option<u32>,
    pub max_tokens: Option<u64>,
    pub deadline: Option<Instant>,
}

impl Budget {
    pub fn unlimited() -> Self {
        Self::default()
    }
}

/// Budget reservation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("budget exhausted")]
pub struct Exhausted;

/// Check-before-call metering: calls and deadline are hard-gated pre-call;
/// token budgets are soft (checked against accumulated usage, since usage is
/// only known post-hoc). Reservation recurses into the parent — an
/// `AgentLoop` chains a child meter under the run meter.
#[derive(Debug)]
pub struct BudgetMeter {
    limits: Budget,
    lm_calls: AtomicU32,
    tokens: AtomicU64,
    parent: Option<Arc<BudgetMeter>>,
}

impl BudgetMeter {
    pub fn new(limits: Budget) -> Self {
        Self {
            limits,
            lm_calls: AtomicU32::new(0),
            tokens: AtomicU64::new(0),
            parent: None,
        }
    }

    /// A child meter whose reservations also count against `parent`.
    pub fn child(parent: &Arc<BudgetMeter>, limits: Budget) -> Self {
        Self {
            parent: Some(Arc::clone(parent)),
            ..Self::new(limits)
        }
    }

    fn check_soft(&self) -> Result<(), Exhausted> {
        if let Some(deadline) = self.limits.deadline
            && Instant::now() >= deadline
        {
            return Err(Exhausted);
        }
        if let Some(max) = self.limits.max_tokens
            && self.tokens.load(Ordering::Relaxed) >= max
        {
            return Err(Exhausted);
        }
        Ok(())
    }

    /// Reserves one LM call against this meter and every ancestor.
    pub fn try_reserve_call(&self) -> Result<(), Exhausted> {
        self.check_soft()?;
        if let Some(max) = self.limits.max_lm_calls {
            self.lm_calls
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |calls| {
                    (calls < max).then(|| calls + 1)
                })
                .map_err(|_| Exhausted)?;
        } else {
            self.lm_calls.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(parent) = &self.parent
            && let Err(exhausted) = parent.try_reserve_call()
        {
            self.lm_calls.fetch_sub(1, Ordering::Relaxed);
            return Err(exhausted);
        }
        Ok(())
    }

    /// Records observed usage on this meter and every ancestor.
    pub fn record_usage(&self, usage: &LmUsage) {
        let _ = self
            .tokens
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |tokens| {
                Some(tokens.saturating_add(usage.total_tokens))
            });
        if let Some(parent) = &self.parent {
            parent.record_usage(usage);
        }
    }

    pub fn lm_calls(&self) -> u32 {
        self.lm_calls.load(Ordering::Relaxed)
    }

    pub fn tokens(&self) -> u64 {
        self.tokens.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Leaf metadata
// ---------------------------------------------------------------------------

/// Parse/coercion metadata from one successful `Predict`-leaf evaluation,
/// collected by [`Interpreter::run_collecting`] in execution order.
///
/// This is what `Predict<S>` reassembles into [`CallMetadata`](crate::CallMetadata):
/// the same [`FieldMeta`] type (jsonish coercion [`Flag`](crate::Flag)s +
/// `#[check]` [`ConstraintResult`](crate::ConstraintResult)s) that
/// `ChatAdapter::parse_output_def` produces, so a typed `Predict<S>` routed
/// through the interpreter loses none of `Predicted`'s metadata contract.
///
/// Scope and semantics:
/// - `Predict` and `AgentLoop` leaves report; `Hole` leaves make no LM call
///   and parse no delimited text. An `AgentLoop` reports once per evaluation
///   with loop-accumulated usage and its tool call/execution record;
///   `field_meta` is empty when the final output came from stop-tool args
///   (no parse metadata exists for those).
/// - Only *successful* evaluations report — a failed attempt inside `Retry`
///   propagates its error and leaves no outcome; the succeeding attempt
///   reports one. A leaf re-evaluated by `Refine`/`Loop` reports once per
///   evaluation.
/// - A replay-served leaf reports raw text and usage from the recorded span
///   with **empty** `field_meta`, matching the static lane ("served
///   predictions carry no per-field parse metadata").
#[derive(Debug, Clone)]
pub struct LeafOutcome {
    /// Persistent invocation IDs, in call order. Replay adds no stored calls.
    pub call_ids: Vec<String>,
    /// The program-unique leaf name (the trace span `component`).
    pub name: String,
    /// The full text the LM returned, before parsing.
    pub raw_response: String,
    /// Per-field parse details keyed by canonical field name
    /// (`FieldDef::name`) — raw section text, coercion flags, check results.
    pub field_meta: IndexMap<String, FieldMeta>,
    /// Token usage for this leaf's LM call.
    pub usage: LmUsage,
    /// Stable hash of the redacted model config used — identical to the
    /// trace's [`ModelEntry::config_hash`](crate::trace::ModelEntry) for the
    /// same config.
    pub model_config_hash: u64,
    /// The trace span this evaluation recorded, when a capture scope was
    /// active — what `Predict` surfaces as `CallMetadata::span_id`.
    pub span_id: Option<crate::trace::SpanId>,
    /// Executed calls in an `AgentLoop` and requested signature tools in a
    /// `Predict` leaf. Agent stop tools remain in the chat transcript.
    pub tool_calls: Vec<rig::message::ToolCall>,
    /// Results of executed tool calls. For agents these align with [`tool_calls`](Self::tool_calls).
    pub tool_executions: Vec<String>,
}

/// Program output plus per-leaf metadata, returned by
/// [`Interpreter::run_collecting`].
#[derive(Debug, Clone)]
pub struct RunOutput {
    /// The program's output map — exactly what [`Interpreter::run`] returns.
    pub output: JsonMap,
    /// One entry per successful `Predict`-leaf evaluation, in execution
    /// order. `ForkJoin` branches append in declared branch order.
    pub leaves: Vec<LeafOutcome>,
}

// ---------------------------------------------------------------------------
// Conversation surface (RFC 0004 §1–2)
// ---------------------------------------------------------------------------

/// One caller-driven conversation turn, returned by
/// [`Interpreter::run_conversation_caller_managed`] and
/// [`Interpreter::resume_conversation`].
#[derive(Debug)]
pub enum ConversationTurn {
    /// The leaf produced its final output; `chat` is the conversation
    /// extended with everything the turn generated.
    Complete { run: RunOutput, chat: Chat },
    /// The model requested tool calls the caller must execute. Run them and
    /// feed the results back through
    /// [`Interpreter::resume_conversation`].
    Suspended(ToolSuspension),
}

/// A caller-managed agent turn suspended on pending tool calls (RFC 0004 §2).
///
/// Everything the loop needs to continue travels inside: the open trace span,
/// the loop and run budget meters, the accumulated event/usage record, and
/// the turn cursor. Dropping a suspension without resuming closes its span as
/// `Cancelled` — the same contract as cancelling a run mid-flight.
pub struct ToolSuspension {
    calls: Vec<rig::message::ToolCall>,
    chat: Chat,
    state: Box<SuspendState>,
}

impl ToolSuspension {
    /// The tool calls the model requested, in request order. Stop-tool calls
    /// never appear here — they complete the turn instead of suspending it.
    pub fn calls(&self) -> &[rig::message::ToolCall] {
        &self.calls
    }

    /// The conversation so far, including the assistant tool-call turn.
    pub fn chat(&self) -> &Chat {
        &self.chat
    }
}

impl std::fmt::Debug for ToolSuspension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolSuspension")
            .field("calls", &self.calls)
            .field("chat_len", &self.chat.len())
            .finish_non_exhaustive()
    }
}

/// The private half of a [`ToolSuspension`]: resumption state for
/// [`Interpreter::resume_conversation`].
struct SuspendState {
    turn: AgentTurn,
    /// The loop turn `resume_conversation` re-enters at.
    next_turn: u32,
    /// When the loop suspended — the `ToolRun` events recorded at resume
    /// meter the time the calls were outstanding.
    suspended_at: Instant,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("program failed validation")]
    Invalid(#[from] ValidateError),
    #[error("program caps exceed host grants: missing {missing:?}")]
    CapsExceedGrants { missing: Vec<String> },
    #[error("model `{name}` could not be bound: {message}")]
    Model { name: String, message: String },
    #[error("model `{name}` uses an artifact URL not approved by the host")]
    ModelEndpointDenied { name: String },
    #[error("host tool `{name}` is not bound in the runtime environment")]
    HostToolUnbound { name: String },
    #[error("host binding `{name}` requires capabilities absent from the artifact: {missing:?}")]
    HostBindingCapsUnderdeclared { name: String, missing: Vec<String> },
    #[error("host (extern) hole `{name}` is not bound in the runtime environment")]
    HostHoleUnbound { name: String },
    #[error("program contains sandboxed code but the environment has no sandbox executor")]
    SandboxMissing,
    #[error("sandboxed code at `{at}` failed to register")]
    Register {
        at: String,
        #[source]
        source: dsrs_tools::RegisterError,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("lm error at `{at}`: {source}")]
    Lm {
        at: Box<str>,
        #[source]
        source: LmError,
    },
    #[error("parse error at `{at}`")]
    Parse {
        at: Box<str>,
        raw: String,
        /// The structured def-lane parse failure, when the leaf's response
        /// text was parsed (`Predict`/`AgentLoop` field parsing); `None` for
        /// structural coercions with no section parse (stop-tool args, hole
        /// outputs).
        source: Option<Box<crate::ParseError>>,
        /// Usage accumulated by the failing leaf evaluation.
        usage: LmUsage,
    },
    #[error("tool `{tool}` failed at `{at}`: {message}")]
    Tool {
        at: Box<str>,
        tool: Box<str>,
        message: String,
    },
    #[error("hole `{at}` failed")]
    Hole {
        at: Box<str>,
        #[source]
        source: dsrs_tools::ExecError,
    },
    #[error("capability `{cap}` denied at `{at}`")]
    CapabilityDenied { at: Box<str>, cap: Box<str> },
    #[error("budget exhausted at `{at}`")]
    Budget { at: Box<str> },
    #[error("route unmatched at `{at}`: `{value}`")]
    Route { at: Box<str>, value: String },
    #[error("cancelled")]
    Cancelled,
    /// Additive to RFC 0002: overlay minted against a different program.
    #[error("overlay minted against program {expected:016x}, run against {got:016x}")]
    Overlay { expected: u64, got: u64 },
    /// Additive to RFC 0002: run input rejected against the program signature.
    #[error("invalid input at `{at}`: {message}")]
    Input { at: Box<str>, message: String },
    /// Additive to RFC 0002: interpreter invariant violation.
    #[error("internal interpreter error at `{at}`: {message}")]
    Internal { at: Box<str>, message: String },
    /// RFC 0003 M-1: a strict replay scope refused this call.
    #[error("replay refused at `{at}`")]
    Replay {
        at: Box<str>,
        #[source]
        source: crate::trace::ReplayError,
    },
}

impl RunError {
    /// Whether `Retry`/`Refine` may intercept this error (RFC 0002 §3.4:
    /// `Lm | Parse | Tool | Hole`; `Budget`/`CapabilityDenied` are not
    /// retryable).
    pub fn retryable(&self) -> bool {
        matches!(
            self,
            Self::Lm { .. } | Self::Parse { .. } | Self::Tool { .. } | Self::Hole { .. }
        )
    }
}

// ---------------------------------------------------------------------------
// Runtime environment
// ---------------------------------------------------------------------------

/// What the host supplies at load: live models, host tool bindings, the
/// sandbox, and the capability grants. Secrets never transit the artifact —
/// model configs are bound to clients here, from host-held keys/env vars.
#[derive(Default)]
pub struct RuntimeEnv {
    /// Pre-bound live models by declared model name (`"fast"`). Models not
    /// bound here are constructed from their `ModelDef.config` at load.
    pub models: HashMap<String, Arc<LM>>,
    /// Exact artifact `base_url` values permitted when no host model is bound.
    /// Empty by default; the artifact cannot grant itself network destinations.
    pub allowed_model_base_urls: HashSet<String>,
    /// Host tool bindings and host-owned capability requirements.
    host_tools: HashMap<String, (Arc<dyn rig::tool::ToolDyn>, CapSet)>,
    /// Host (extern) hole bindings by leaf name, consulted once at load
    /// (RFC 0003 §4.1). The fn receives the hole's resolved input map and
    /// returns a JSON value coerced against the hole's output signature.
    host_holes: HashMap<String, (HostHoleFn, CapSet)>,
    /// The sandbox executing holes and sandboxed tools (QuickJS in v1).
    /// Required iff the program carries sandboxed code.
    pub sandbox: Option<Arc<dyn dsrs_tools::Executor>>,
    /// What this host permits. `program.caps ⊄ grants` ⇒ load refused.
    pub grants: CapSet,
    /// Code Mode: when set, every `AgentLoop` presents its non-stop tools as
    /// a single sandboxed `run_js` tool (with this sandbox config) instead of
    /// N JSON tools; the model writes JavaScript that calls them as globals.
    ///
    /// This is a `RuntimeEnv` binding option, not a `ToolKind` variant, by
    /// design: code mode is a host presentation/execution strategy, not
    /// program semantics — the same artifact (same tools, same genes, same
    /// program hash) must run identically either way, so it cannot live in
    /// the serialized program. `ToolKind` is also per-tool, while code mode
    /// collapses a whole loop's tool surface; and leaving the closed enum
    /// untouched keeps the `.dsrs` text format stable.
    pub code_mode: Option<dsrs_tools::SandboxConfig>,
}

impl RuntimeEnv {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bind_model(mut self, name: &str, lm: Arc<LM>) -> Self {
        self.models.insert(name.to_string(), lm);
        self
    }

    pub fn allow_model_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.allowed_model_base_urls.insert(base_url.into());
        self
    }

    /// Bind a native tool with requirements owned by the host, independent of
    /// any capability claim in a loaded artifact. Pass `&[]` only for tools
    /// that genuinely need no capability.
    pub fn bind_host_tool(
        mut self,
        name: &str,
        tool: Arc<dyn rig::tool::ToolDyn>,
        required_caps: &[&str],
    ) -> Self {
        self.host_tools.insert(
            name.to_string(),
            (tool, required_caps.iter().copied().collect()),
        );
        self
    }

    /// Binds a native implementation for an extern hole by leaf name.
    pub fn bind_host_hole<F, Fut>(mut self, name: &str, required_caps: &[&str], f: F) -> Self
    where
        F: Fn(JsonMap) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, String>> + Send + 'static,
    {
        self.host_holes.insert(
            name.to_string(),
            (
                Arc::new(move |input| Box::pin(f(input))),
                required_caps.iter().copied().collect(),
            ),
        );
        self
    }

    pub fn with_sandbox(mut self, sandbox: Arc<dyn dsrs_tools::Executor>) -> Self {
        self.sandbox = Some(sandbox);
        self
    }

    pub fn grant(mut self, cap: &str) -> Self {
        self.grants.insert(cap);
        self
    }

    /// Enables Code Mode for every `AgentLoop` (see
    /// [`code_mode`](Self::code_mode)): tools are presented to the model as a
    /// JS API behind one `run_js` tool executing under `config`.
    pub fn with_code_mode(mut self, config: dsrs_tools::SandboxConfig) -> Self {
        self.code_mode = Some(config);
        self
    }
}

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum ToolExec {
    Host(Arc<dyn rig::tool::ToolDyn>),
    Sandboxed(ParamId),
}

type SandboxRegistrationKey = (ParamId, u64);
type SandboxRegistrationResult = Result<String, dsrs_tools::RegisterError>;
type SandboxRegistrationCell = Arc<OnceCell<SandboxRegistrationResult>>;

#[derive(Default)]
struct SandboxRegistrations(Mutex<HashMap<SandboxRegistrationKey, SandboxRegistrationCell>>);

impl SandboxRegistrations {
    async fn get_or_register(
        &self,
        key: SandboxRegistrationKey,
        register: impl Future<Output = SandboxRegistrationResult>,
    ) -> SandboxRegistrationResult {
        let cell = {
            let mut entries = self.0.lock().unwrap();
            let cell = entries.entry(key).or_default();
            if let Some(result) = cell.get() {
                return result.clone();
            }
            Arc::clone(cell)
        };
        let pending = SandboxRegistration {
            registry: self,
            key,
            cell: Some(cell),
        };
        pending
            .cell
            .as_ref()
            .unwrap()
            .get_or_init(|| register)
            .await
            .clone()
    }
}

struct SandboxRegistration<'a> {
    registry: &'a SandboxRegistrations,
    key: SandboxRegistrationKey,
    cell: Option<SandboxRegistrationCell>,
}

impl Drop for SandboxRegistration<'_> {
    fn drop(&mut self) {
        let cell = self.cell.take().unwrap();
        if matches!(cell.get(), Some(Ok(_))) {
            return;
        }
        let mut entries = self.registry.0.lock().unwrap();
        // Only the map and this caller still own the failed or cancelled cell.
        if Arc::strong_count(&cell) == 2 && !matches!(cell.get(), Some(Ok(_))) {
            entries.remove(&self.key);
        }
        drop(cell);
    }
}

/// A native extern-hole implementation: resolved input map in, JSON value out
/// (coerced against the hole's output signature by the interpreter).
pub type HostHoleFn =
    Arc<dyn Fn(JsonMap) -> BoxFuture<'static, Result<Value, String>> + Send + Sync>;

/// A loaded, executable program: validated graph + bound models/tools +
/// registered sandbox code. Cheap to share; run state never lives here.
pub struct Interpreter {
    program: Arc<Program>,
    /// One last-used instruction/system pair per signature. Dynamic candidates
    /// replace entries rather than growing an unbounded process-wide cache.
    system_prompts: RwLock<HashMap<SigId, (String, OutputFormat, String)>>,
    /// At most one rendered demo set per leaf. The overlay is never kept alive by the cache.
    demo_messages: RwLock<HashMap<NodeId, CachedDemos>>,
    input_templates: SecondaryMap<SigId, Option<Arc<PreparedInputTemplates>>>,
    schemas: PrimaryMap<SigId, PreparedSchemas>,
    models: SecondaryMap<ModelId, Option<Arc<LM>>>,
    tool_exec: SecondaryMap<ToolId, Option<ToolExec>>,
    /// Extern-hole bindings by leaf name, verified complete at load.
    host_holes: HashMap<String, HostHoleFn>,
    sandbox: Option<Arc<dyn dsrs_tools::Executor>>,
    /// Registrations are shared within a declared code slot and source hash.
    registered: SandboxRegistrations,
    /// Code Mode sandbox config (see [`RuntimeEnv::code_mode`]).
    code_mode: Option<dsrs_tools::SandboxConfig>,
}

impl std::fmt::Debug for Interpreter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Interpreter")
            .field("program", &self.program.meta.name)
            .field("program_hash", &self.program.meta.program_hash)
            .finish_non_exhaustive()
    }
}

impl Interpreter {
    /// Load-time checks, in order — nothing lazy, nothing at call time:
    /// 1. [`Program::validate`] (§2.3 rules)
    /// 2. `program.caps ⊆ env.grants` (WIT rule: no ambient authority)
    /// 3. every model bindable (pre-bound by name, else client-constructible)
    /// 4. every `ToolKind::Host` name bound
    /// 5. every sandboxed tool and hole registered through the dsrs-tools
    ///    lifecycle (parse → compile → register). A hole that doesn't compile
    ///    fails the LOAD, not the call.
    pub fn load(
        program: Program,
        env: RuntimeEnv,
    ) -> impl std::future::Future<Output = Result<Self, LoadError>> {
        Self::load_shared(Arc::new(program), env)
    }

    /// Loads independent runtime bindings over an existing immutable program.
    /// Every validation and capability check is identical to `load`; only the
    /// graph allocation is shared. Per-run state remains local to each call.
    pub async fn load_shared(program: Arc<Program>, env: RuntimeEnv) -> Result<Self, LoadError> {
        program.validate()?;

        if !program.caps.is_subset(&env.grants) {
            return Err(LoadError::CapsExceedGrants {
                missing: program.caps.missing_from(&env.grants),
            });
        }

        let mut models: SecondaryMap<ModelId, Option<Arc<LM>>> = SecondaryMap::new();
        for (id, def) in program.models.iter() {
            let lm = match env.models.get(&*def.name) {
                Some(lm) => Arc::clone(lm),
                None => {
                    if def
                        .config
                        .base_url
                        .as_ref()
                        .is_some_and(|url| !env.allowed_model_base_urls.contains(url))
                    {
                        return Err(LoadError::ModelEndpointDenied {
                            name: def.name.to_string(),
                        });
                    }
                    Arc::new(LM::from_config(def.config.clone()).await.map_err(|err| {
                        LoadError::Model {
                            name: def.name.to_string(),
                            message: err.to_string(),
                        }
                    })?)
                }
            };
            models[id] = Some(lm);
        }

        let mut tool_exec: SecondaryMap<ToolId, Option<ToolExec>> = SecondaryMap::new();
        let mut sandboxed: Vec<(String, ParamId, crate::ir::graph::SigId, CapSet)> = Vec::new();
        for (id, tool) in program.tools.iter() {
            let name = program.syms.get(tool.name).to_string();
            match tool.kind {
                ToolKind::Host => {
                    let (bound, required) = env
                        .host_tools
                        .get(&name)
                        .ok_or(LoadError::HostToolUnbound { name: name.clone() })?;
                    if !required.is_subset(&tool.caps) {
                        return Err(LoadError::HostBindingCapsUnderdeclared {
                            name,
                            missing: required.missing_from(&tool.caps),
                        });
                    }
                    tool_exec[id] = Some(ToolExec::Host(Arc::clone(bound)));
                }
                ToolKind::Sandboxed { code } => {
                    sandboxed.push((name, code, tool.sig, tool.caps.clone()));
                    tool_exec[id] = Some(ToolExec::Sandboxed(code));
                }
            }
        }
        let mut host_holes: HashMap<String, HostHoleFn> = HashMap::new();
        for (_, node) in program.nodes.iter() {
            if let Node::Hole(hole) = node {
                let name = program.syms.get(hole.name).to_string();
                match hole.imp {
                    HoleImpl::Sandboxed { code } => {
                        sandboxed.push((name, code, hole.sig, hole.caps.clone()))
                    }
                    HoleImpl::Host { .. } => {
                        let (bound, required) = env
                            .host_holes
                            .get(&name)
                            .ok_or(LoadError::HostHoleUnbound { name: name.clone() })?;
                        if !required.is_subset(&hole.caps) {
                            return Err(LoadError::HostBindingCapsUnderdeclared {
                                name,
                                missing: required.missing_from(&hole.caps),
                            });
                        }
                        host_holes.insert(name, Arc::clone(bound));
                    }
                }
            }
        }

        // Code Mode: JS-identifier collisions among a loop's non-stop tool
        // names are a load-time refusal (nothing lazy, nothing at call time).
        if env.code_mode.is_some() {
            for (_, node) in program.nodes.iter() {
                let Node::AgentLoop(agent) = node else {
                    continue;
                };
                let mut seen: HashMap<String, String> = HashMap::new();
                for &tool_id in agent.tools.iter() {
                    if agent.stop.stop_tools.contains(&tool_id) {
                        continue;
                    }
                    let name = program.syms.get(program.tools[tool_id].name).to_string();
                    let js_name = dsrs_tools::js_identifier(&name);
                    if let Some(previous) = seen.insert(js_name.clone(), name.clone()) {
                        return Err(LoadError::Register {
                            at: program.syms.get(agent.name).to_string(),
                            source: dsrs_tools::RegisterError::InvalidCapability {
                                name: js_name.clone(),
                                reason: format!(
                                    "tool names `{previous}` and `{name}` both mangle to the JS identifier `{js_name}`"
                                ),
                            },
                        });
                    }
                }
            }
        }

        let mut input_templates = SecondaryMap::new();
        let mut schemas = PrimaryMap::new();
        for (sig, def) in program.sigs.iter() {
            schemas.push(PreparedSchemas::new(def));
            if let Some(templates) = PreparedInputTemplates::new(def) {
                input_templates[sig] = Some(Arc::new(templates));
            }
        }

        let sandbox = env.sandbox;
        let mut registered = HashMap::new();
        if !sandboxed.is_empty() {
            let executor = sandbox.as_ref().ok_or(LoadError::SandboxMissing)?;
            for (base, code, sig, caps) in sandboxed {
                let ParamValue::Code { source, hash, .. } = &program.params[code].default else {
                    unreachable!("validated: code slots hold Code values");
                };
                let name = register_code(
                    executor.as_ref(),
                    &base,
                    source,
                    *hash,
                    || {
                        schemas[sig]
                            .tool_input(&program.sigs[sig], &program.types)
                            .clone()
                    },
                    &caps,
                )
                .await
                .map_err(|source| LoadError::Register {
                    at: base.clone(),
                    source,
                })?;
                registered.insert((code, *hash), Arc::new(OnceCell::new_with(Some(Ok(name)))));
            }
        }

        Ok(Self {
            program,
            system_prompts: RwLock::new(HashMap::new()),
            demo_messages: RwLock::new(HashMap::new()),
            input_templates,
            schemas,
            models,
            tool_exec,
            host_holes,
            sandbox,
            registered: SandboxRegistrations(Mutex::new(registered)),
            code_mode: env.code_mode,
        })
    }

    pub fn program(&self) -> &Arc<Program> {
        &self.program
    }

    fn output_format(&self, sig: SigId, lm: &LM, allow_scalar: bool) -> OutputFormat {
        let prepared = &self.schemas[sig];
        if prepared.tool_calls {
            return OutputFormat::Delimited;
        }
        if lm.supports_output_schema()
            && *prepared.compatible.get_or_init(|| {
                json_schema_compatible(&self.program.sigs[sig], &self.program.types)
            })
        {
            if allow_scalar && prepared.scalar && lm.supports_scalar_output_schema() {
                OutputFormat::JsonScalar
            } else {
                OutputFormat::JsonSchema
            }
        } else if lm.supports_json_mode() {
            OutputFormat::JsonMode
        } else {
            OutputFormat::Delimited
        }
    }

    fn output_options(&self, sig: SigId, format: OutputFormat) -> &LMCallOptions {
        static DELIMITED: LMCallOptions = LMCallOptions {
            output_schema: None,
            json_mode: false,
            force_output_schema: false,
        };
        static JSON_MODE: LMCallOptions = LMCallOptions {
            output_schema: None,
            json_mode: true,
            force_output_schema: false,
        };
        let prepared = &self.schemas[sig];
        let options = match format {
            OutputFormat::Delimited => return &DELIMITED,
            OutputFormat::JsonMode => return &JSON_MODE,
            OutputFormat::JsonSchema => &prepared.object_options,
            OutputFormat::JsonScalar => &prepared.scalar_options,
        };
        options.get_or_init(|| {
            let def = &self.program.sigs[sig];
            let mut schema = json_schema_for_outputs(def, &self.program.types);
            if format == OutputFormat::JsonScalar {
                schema = std::mem::take(&mut schema["properties"][&*def.outputs[0].lm_name]);
            }
            Box::new(LMCallOptions {
                output_schema: Some(schema),
                ..LMCallOptions::default()
            })
        })
    }

    fn tool_input_schema(&self, sig: SigId) -> &Value {
        self.schemas[sig].tool_input(&self.program.sigs[sig], &self.program.types)
    }

    fn system_prompt(&self, sig: SigId, instruction: &str, format: OutputFormat) -> String {
        if let Some((cached_instruction, cached_format, prompt)) = self
            .system_prompts
            .read()
            .expect("system prompt cache poisoned")
            .get(&sig)
            && cached_instruction == instruction
            && *cached_format == format
        {
            return prompt.clone();
        }
        let def = &self.program.sigs[sig];
        let prompt = if format == OutputFormat::Delimited {
            ChatAdapter.build_system_def(def, &self.program.types, Some(instruction))
        } else if format == OutputFormat::JsonScalar {
            ChatAdapter.build_system_scalar_def(def, &self.program.types, instruction)
        } else {
            ChatAdapter.build_system_json_def(
                def,
                &self.program.types,
                Some(instruction),
                format == OutputFormat::JsonSchema,
            )
        };
        self.system_prompts
            .write()
            .expect("system prompt cache poisoned")
            .insert(sig, (instruction.to_owned(), format, prompt.clone()));
        prompt
    }

    /// Evaluates the program on `input`, reading parameters through `overlay`
    /// (never mutating the program), metering spend against `budget`.
    pub async fn run(
        &self,
        input: JsonMap,
        overlay: Option<Arc<Overlay>>,
        budget: Budget,
    ) -> Result<JsonMap, RunError> {
        self.run_inner(input, overlay, budget, false, None)
            .await
            .map(|out| out.output)
    }

    /// Like [`run`](Interpreter::run), additionally collecting a
    /// [`LeafOutcome`] per successful `Predict`-leaf evaluation (raw response,
    /// per-field flags and check results, usage, model config hash). See
    /// [`LeafOutcome`] for the exact scope and semantics.
    pub async fn run_collecting(
        &self,
        input: JsonMap,
        overlay: Option<Arc<Overlay>>,
        budget: Budget,
    ) -> Result<RunOutput, RunError> {
        self.run_inner(input, overlay, budget, true, None).await
    }

    pub(crate) async fn run_collecting_prepared(
        &self,
        input: JsonMap,
        overlay: Option<Arc<Overlay>>,
        budget: Budget,
        prepare: PrepareReplayOutput<'_>,
    ) -> Result<RunOutput, RunError> {
        self.run_inner(input, overlay, budget, true, Some(prepare))
            .await
    }

    /// Conversation-in/conversation-out evaluation (RFC 0004 §1): one turn
    /// with the program's single leaf over a caller-owned [`Chat`].
    ///
    /// A turn is not a run: each call evaluates the leaf once against the
    /// given conversation and returns the extended conversation alongside the
    /// turn's [`RunOutput`] (one [`LeafOutcome`], same semantics as
    /// [`run_collecting`](Self::run_collecting)). Call again with the
    /// returned chat — plus your follow-up message or a fresh `input` — for
    /// the next turn. Each turn records one trace span (`seq` increments per
    /// turn), meters against its own `budget`, and replays like any other
    /// leaf evaluation: the span keys on the full chat sent, so a recorded
    /// conversation serves turn by turn.
    ///
    /// `chat`/`input` combinations:
    /// - **empty chat + `Some(input)`** — opening turn: renders system +
    ///   demos + the formatted input, exactly like
    ///   [`conversation_opening`](Self::conversation_opening).
    /// - **non-empty chat + `None`** — continuation: the chat is sent as-is
    ///   (append your follow-up first).
    /// - **non-empty chat + `Some(input)`** — typed continuation: the
    ///   formatted input is appended as the next user message.
    ///
    /// Tools on an agent leaf dispatch through their bound executors, exactly
    /// like [`run`](Self::run). For the suspend-on-tools variant see
    /// [`run_conversation_caller_managed`](Self::run_conversation_caller_managed).
    ///
    /// Only single-leaf programs (what `Predict<S>` compiles to) have a
    /// conversation surface; a multi-node graph is refused with
    /// [`RunError::Input`].
    pub async fn run_conversation(
        &self,
        chat: Chat,
        input: Option<JsonMap>,
        overlay: Option<Arc<Overlay>>,
        budget: Budget,
    ) -> Result<(RunOutput, Chat), RunError> {
        pin!(self.run_conversation_prepared(chat, input, overlay, budget, None)).await
    }

    pub(crate) async fn run_conversation_prepared(
        &self,
        chat: Chat,
        input: Option<JsonMap>,
        overlay: Option<Arc<Overlay>>,
        budget: Budget,
        prepare: Option<PrepareReplayOutput<'_>>,
    ) -> Result<(RunOutput, Chat), RunError> {
        match self
            .conversation_turn(chat, input, overlay, budget, false, prepare)
            .await?
        {
            ConversationTurn::Complete { run, chat } => Ok((run, chat)),
            ConversationTurn::Suspended(_) => {
                unreachable!("dispatching conversations never suspend")
            }
        }
    }

    /// [`run_conversation`](Self::run_conversation) in caller-managed mode
    /// (RFC 0004 §2): when the model requests tool calls on an agent leaf,
    /// the loop *suspends* instead of dispatching — you get the pending calls
    /// plus a resumption token ([`ToolSuspension`]) and feed results back
    /// through [`resume_conversation`](Self::resume_conversation).
    ///
    /// Everything else is identical to dispatching mode: one span per turn
    /// with the same `Exchange`/`ToolRun` event stream, the same meters (the
    /// node budget chained under the run budget), and the same stop-tool
    /// semantics — a stop-tool call completes the turn, it never suspends.
    /// Two deliberate differences: Code Mode does not apply (the caller
    /// executes tools, so the host's execution strategy can't), and a replay
    /// scope never suspends (served turns have every tool effect baked in).
    pub async fn run_conversation_caller_managed(
        &self,
        chat: Chat,
        input: Option<JsonMap>,
        overlay: Option<Arc<Overlay>>,
        budget: Budget,
    ) -> Result<ConversationTurn, RunError> {
        self.conversation_turn(chat, input, overlay, budget, true, None)
            .await
    }

    /// Continues a suspended caller-managed turn with the results of the
    /// pending tool calls, aligned with [`ToolSuspension::calls`].
    /// Resume against the same loaded program that produced the suspension.
    ///
    /// Results land in the conversation and the trace exactly as dispatched
    /// executions would: one `ToolRun` event per call (metering the time the
    /// suspension was outstanding), one batched tool-result user turn, and
    /// the loop continues under the same meters and turn budget. Feed a
    /// failed tool's error text as its result to keep dispatching mode's
    /// LATM-style conversational repair.
    pub async fn resume_conversation(
        &self,
        suspension: ToolSuspension,
        results: Vec<String>,
    ) -> Result<ConversationTurn, RunError> {
        let ToolSuspension {
            calls,
            mut chat,
            state,
        } = suspension;
        let SuspendState {
            mut turn,
            next_turn,
            suspended_at,
        } = *state;
        let invalid = if !Arc::ptr_eq(&turn.program, &self.program) {
            Some("suspension belongs to another loaded program".to_string())
        } else if results.len() != calls.len() {
            Some(format!(
                "expected {} tool results, got {}",
                calls.len(),
                results.len()
            ))
        } else {
            None
        };
        if let Some(message) = invalid {
            let error = RunError::Input {
                at: turn.at.into(),
                message,
            };
            return Err(turn.run.finish_error(
                turn.guard,
                crate::trace::SpanErrorKind::Tool,
                error,
            ));
        }

        // Feed the caller's results back: same event, record, and
        // conversation shape as dispatched executions.
        let duration_us = suspended_at.elapsed().as_micros() as u64;
        let blocks = calls
            .iter()
            .zip(results)
            .map(|(call, result)| {
                let result = clip_tool_result(result, &turn.surface.policy);
                turn.run.record_tool_result(call, result, duration_us, None)
            })
            .collect();
        chat.push_message(Message::with_content(Role::User, blocks));
        self.drive_agent_turn(turn, chat, next_turn).await
    }

    /// Renders the opening [`Chat`] of a conversation with the program's
    /// leaf — system + demos (+ the agent playbook) + the formatted `input`
    /// turn — reading instruction/demos through `overlay` exactly like a run
    /// would. Inspect or edit the result, then hand it to
    /// [`run_conversation`](Self::run_conversation); or skip this and pass
    /// `run_conversation` an empty chat plus the input — same rendering.
    pub fn conversation_opening(
        &self,
        input: &JsonMap,
        overlay: Option<Arc<Overlay>>,
    ) -> Result<Chat, RunError> {
        self.check_overlay(overlay.as_ref())?;
        let node = self.conversation_leaf()?;
        let at = self
            .program
            .leaf_name(node)
            .expect("conversation leaves are named")
            .to_string();
        self.validate_input(&at, self.leaf_sig(node), input)?;
        let cx = self.conversation_cx(
            overlay,
            Arc::new(BudgetMeter::new(Budget::unlimited())),
            None,
        );
        let (mut messages, suffix) = self.render_leaf_opening(node, input, &cx)?;
        messages.extend(suffix);
        Ok(Chat::new(messages))
    }

    /// One conversation turn against the program's single leaf: assembles
    /// this turn's messages, opens the span, consults replay, and evaluates
    /// the leaf (single exchange for `Predict`, the agent loop for
    /// `AgentLoop` — suspending on tool calls when `suspend_on_tools`).
    async fn conversation_turn(
        &self,
        mut chat: Chat,
        input: Option<JsonMap>,
        overlay: Option<Arc<Overlay>>,
        budget: Budget,
        suspend_on_tools: bool,
        prepare: Option<PrepareReplayOutput<'_>>,
    ) -> Result<ConversationTurn, RunError> {
        self.check_overlay(overlay.as_ref())?;
        let node = self.conversation_leaf()?;
        let p = &*self.program;
        let at = p
            .leaf_name(node)
            .expect("conversation leaves are named")
            .to_string();
        let sig = self.leaf_sig_id(node);
        let def = &p.sigs[sig];
        let cx = self.conversation_cx(overlay.clone(), Arc::new(BudgetMeter::new(budget)), prepare);
        let lm = match &p.nodes[node] {
            Node::Predict(n) => self.p_model(&at, &cx, n.model)?,
            Node::AgentLoop(n) => self.p_model(&at, &cx, n.model)?,
            _ => unreachable!("conversation_leaf returns only leaves"),
        };
        let format = self.output_format(sig, &lm, matches!(&p.nodes[node], Node::Predict(_)));

        // This turn's messages, plus the rendered prefix when we own it (an
        // opening turn interns system+demos; a caller-owned continuation has
        // no known prefix split — the span records the full chat as suffix).
        let (prefix, suffix) = if chat.is_empty() {
            let input_map = input.as_ref().ok_or_else(|| RunError::Input {
                at: at.clone().into(),
                message: "an empty conversation needs an opening `input`".to_string(),
            })?;
            self.validate_input(&at, def, input_map)?;
            self.render_leaf_opening(node, input_map, &cx)?
        } else {
            if let Some(input_map) = input.as_ref() {
                self.validate_input(&at, def, input_map)?;
                let rendered =
                    render_input(def, input_map, format, self.input_templates[sig].as_deref())
                        .map_err(|err| RunError::Input {
                            at: at.clone().into(),
                            message: format!("{err:#}"),
                        })?;
                chat.push_message(Message::user(rendered));
            }
            (Vec::new(), chat.messages)
        };
        let prefix_len = if prefix.is_empty() {
            // Continuation: shield the leading system prompt from history
            // truncation, the only thing `prefix_len` is used for downstream.
            suffix
                .iter()
                .take_while(|message| message.role == Role::System)
                .count()
        } else {
            prefix.len()
        };

        let captured_prefix_len = prefix.len();
        let mut messages = prefix;
        messages.extend(suffix);

        let opening = TurnRequest {
            at: &at,
            prefix: &messages[..captured_prefix_len],
            suffix: &messages[captured_prefix_len..],
            input: input.as_ref(),
            model: &lm.config,
        };
        let agent = match &p.nodes[node] {
            Node::AgentLoop(n) => Some(self.prepare_agent(n, &cx, !suspend_on_tools)),
            _ => None,
        };
        let (mut guard, replay) = match &agent {
            Some(agent) => self.begin_agent_request(&cx, agent, opening, prefix_len),
            None => (
                opening.open_span(|| None),
                crate::trace::replay::intercept(&at, &lm.config, &messages, |output| {
                    self.prepare_replay_output(def, cx.prepare_replay, output)
                }),
            ),
        };
        match replay {
            Some(crate::trace::replay::ReplayDirective::Serve(span)) => {
                return Ok(self.serve_conversation_turn(
                    &at,
                    &lm,
                    span.span(),
                    messages,
                    guard,
                    agent.is_some(),
                ));
            }
            Some(crate::trace::replay::ReplayDirective::Refuse(err)) => {
                if let Some(guard) = guard {
                    guard.finish(span_error(
                        crate::trace::SpanErrorKind::Lm,
                        err.to_string(),
                        Vec::new(),
                        None,
                        LmUsage::default(),
                    ));
                }
                return Err(RunError::Replay {
                    at: at.into(),
                    source: err,
                });
            }
            Some(crate::trace::replay::ReplayDirective::Live) | None => {}
        }

        match agent {
            None => {
                let (output, leaf, chat) = self
                    .exchange_predict(
                        PredictExchange {
                            at: &at,
                            sig,
                            format,
                            lm: &lm,
                            meter: &cx.meter,
                            guard,
                        },
                        messages,
                        |messages| ChatAdapter.signature_tools_from_messages(def, messages),
                        true,
                    )
                    .await?;
                Ok(ConversationTurn::Complete {
                    run: RunOutput {
                        output,
                        leaves: leaf.into_iter().collect(),
                    },
                    chat,
                })
            }
            Some(agent) => {
                let n = agent.node;
                let mut run = AgentRun::default();
                let surface = run.finish_on_setup_error(
                    &mut guard,
                    self.build_agent_surface(&at, &cx, agent).await,
                )?;
                let meter = Arc::new(BudgetMeter::child(&cx.meter, node_budget(&n.budget)));
                let model_config_hash =
                    crate::trace::ModelEntry::from_config(&lm.config).config_hash;
                self.drive_agent_turn(
                    AgentTurn {
                        program: Arc::clone(&self.program),
                        node,
                        at,
                        lm,
                        sampling: crate::current_sampling_options(),
                        model_config_hash,
                        surface,
                        meter,
                        run_meter: Arc::clone(&cx.meter),
                        prefix_len,
                        guard,
                        run,
                    },
                    Chat::new(messages),
                    0,
                )
                .await
            }
        }
    }

    async fn exchange_predict(
        &self,
        exchange: PredictExchange<'_>,
        messages: Vec<Message>,
        declarations: impl FnOnce(&[Message]) -> anyhow::Result<Vec<rig::completion::ToolDefinition>>,
        collect_leaf: bool,
    ) -> Result<(JsonMap, Option<LeafOutcome>, Chat), RunError> {
        let PredictExchange {
            at,
            sig,
            format,
            lm,
            meter,
            guard,
        } = exchange;
        if meter.try_reserve_call().is_err() {
            if let Some(guard) = guard {
                guard.finish(span_error(
                    crate::trace::SpanErrorKind::Lm,
                    "budget exhausted".to_string(),
                    Vec::new(),
                    None,
                    LmUsage::default(),
                ));
            }
            return Err(RunError::Budget { at: at.into() });
        }

        let types = &self.program.types;
        let def = &self.program.sigs[sig];
        let options = self.output_options(sig, format);
        let declarations = declarations(&messages).map_err(|error| RunError::Input {
            at: at.into(),
            message: error.to_string(),
        })?;
        let toolset = ToolSet::from_definitions(declarations);
        let response = match in_span(
            guard.as_ref(),
            pin!(lm.call_with_toolset_and_options_ref(
                Chat::new(messages),
                &toolset,
                if toolset.is_empty() {
                    ToolLoopMode::Auto
                } else {
                    ToolLoopMode::CallerManaged
                },
                options,
            )),
        )
        .await
        {
            Ok(response) => response,
            Err(err) => {
                let message = err.to_string();
                if let Some(guard) = guard {
                    guard.finish(span_error(
                        crate::trace::SpanErrorKind::Lm,
                        message.clone(),
                        Vec::new(),
                        None,
                        LmUsage::default(),
                    ));
                }
                return Err(RunError::Lm {
                    at: at.into(),
                    source: LmError::Provider {
                        provider: lm.config.model.clone(),
                        message,
                        source: Some(err.reallocate_into_boxed_dyn_error_without_backtrace()),
                    },
                });
            }
        };
        meter.record_usage(&response.usage);

        let parsed = ChatAdapter
            .parse_signature_tool_output(def, types, &response.tool_calls)
            .unwrap_or_else(|| parse_response(def, types, &response.output, format));
        if let Some(history) = &response.history {
            history.parsed(&parsed).await;
        }
        match parsed {
            Ok((output, metas)) => {
                let raw = if collect_leaf || guard.is_some() {
                    response.output.content()
                } else {
                    String::new()
                };
                let leaf = collect_leaf.then(|| LeafOutcome {
                    name: at.to_string(),
                    raw_response: raw.clone(),
                    field_meta: metas,
                    call_ids: response.history.iter().map(|h| h.id()).collect(),
                    usage: response.usage,
                    model_config_hash: crate::trace::ModelEntry::from_config(&lm.config)
                        .config_hash,
                    span_id: guard.as_ref().map(|guard| guard.id()),
                    tool_calls: response.tool_calls,
                    tool_executions: response.tool_executions,
                });
                if let Some(guard) = guard {
                    guard.finish(SpanOutcome {
                        events: response.events,
                        raw_output: Some(raw),
                        output: Some(output.clone()),
                        usage: response.usage,
                        error: None,
                    });
                }
                Ok((output, leaf, response.chat))
            }
            Err(err) => {
                let raw = response.output.content();
                if let Some(guard) = guard {
                    guard.finish(span_error(
                        crate::trace::SpanErrorKind::Parse,
                        err.to_string(),
                        response.events,
                        Some(raw.clone()),
                        response.usage,
                    ));
                }
                Err(RunError::Parse {
                    at: at.into(),
                    raw,
                    source: Some(Box::new(err)),
                    usage: response.usage,
                })
            }
        }
    }

    /// Serves one conversation turn from the recorded output and chat edits.
    fn serve_conversation_turn(
        &self,
        at: &str,
        lm: &Arc<LM>,
        span: &crate::trace::Span,
        messages: Vec<Message>,
        guard: Option<crate::trace::SpanGuard>,
        agent: bool,
    ) -> ConversationTurn {
        let chat = span.replay_chat(messages);
        let output = span
            .output
            .clone()
            .expect("replay serves only spans with parsed output");
        let (tool_calls, tool_executions) = if agent {
            span.recorded_tools()
        } else {
            (span.requested_tool_calls(), Vec::new())
        };
        let leaf = LeafOutcome {
            name: at.to_string(),
            raw_response: span.raw_output.clone().unwrap_or_default(),
            call_ids: Vec::new(),
            field_meta: IndexMap::new(),
            usage: span.usage,
            model_config_hash: crate::trace::ModelEntry::from_config(&lm.config).config_hash,
            span_id: guard.as_ref().map(|guard| guard.id()),
            tool_calls,
            tool_executions,
        };
        if let Some(guard) = guard {
            guard.finish(SpanOutcome {
                events: span.events.clone(),
                raw_output: span.raw_output.clone(),
                output: Some(output.clone()),
                usage: span.usage,
                error: None,
            });
        }
        ConversationTurn::Complete {
            run: RunOutput {
                output,
                leaves: vec![leaf],
            },
            chat,
        }
    }

    /// Starts or resumes a conversation with the same resolved agent state.
    async fn drive_agent_turn(
        &self,
        mut turn: AgentTurn,
        chat: Chat,
        next_turn: u32,
    ) -> Result<ConversationTurn, RunError> {
        let Node::AgentLoop(n) = &turn.program.nodes[turn.node] else {
            unreachable!("agent turns start at an agent leaf");
        };
        let lc = AgentLoopCx {
            at: &turn.at,
            n,
            def: &turn.program.sigs[n.sig],
            lm: &turn.lm,
            surface: &turn.surface,
            prefix_len: turn.prefix_len,
            meter: &turn.meter,
            run_meter: &turn.run_meter,
        };
        let outcome = crate::with_sampling_options(
            turn.sampling.clone(),
            pin!(in_span(
                turn.guard.as_ref(),
                pin!(self.agent_loop(&lc, chat, &mut turn.run, next_turn)),
            )),
        )
        .await;
        match outcome {
            Ok(LoopOutcome::Done(done)) => {
                let (output, leaf, chat) =
                    turn.run
                        .finish(turn.guard, done, Some((&turn.at, turn.model_config_hash)));
                Ok(ConversationTurn::Complete {
                    run: RunOutput {
                        output,
                        leaves: leaf.into_iter().collect(),
                    },
                    chat,
                })
            }
            Ok(LoopOutcome::Suspend {
                calls,
                chat,
                next_turn,
            }) => Ok(ConversationTurn::Suspended(ToolSuspension {
                calls,
                chat,
                state: Box::new(SuspendState {
                    turn,
                    next_turn,
                    suspended_at: Instant::now(),
                }),
            })),
            Err(err) => Err(turn.run.finish_loop_error(turn.guard, err)),
        }
    }

    /// The conversation leaf: the program's single `Predict` or `AgentLoop`
    /// node, unwrapped through single-child `Seq`s. The conversation surface
    /// deliberately covers only 1-leaf programs (what `Predict<S>` compiles
    /// to) — a multi-node graph has no single conversation to own.
    fn conversation_leaf(&self) -> Result<NodeId, RunError> {
        let mut node = self.program.root;
        loop {
            match &self.program.nodes[node] {
                Node::Predict(_) | Node::AgentLoop(_) => return Ok(node),
                Node::Seq(n) if n.body.len() == 1 => node = n.body[0],
                _ => {
                    return Err(RunError::Input {
                        at: "$".into(),
                        message: "the conversation surface requires a single-leaf program \
                                  (one predict or agent node)"
                            .to_string(),
                    });
                }
            }
        }
    }

    /// The declared signature of a conversation leaf.
    fn leaf_sig(&self, node: NodeId) -> &SignatureDef {
        &self.program.sigs[self.leaf_sig_id(node)]
    }

    fn leaf_sig_id(&self, node: NodeId) -> SigId {
        match &self.program.nodes[node] {
            Node::Predict(n) => n.sig,
            Node::AgentLoop(n) => n.sig,
            _ => unreachable!("conversation_leaf returns only leaves"),
        }
    }

    /// Renders a conversation's opening (prefix, suffix) for the leaf with
    /// overlay-resolved instruction/demos (and the agent playbook) — exactly
    /// what a map-in evaluation of the same leaf would send.
    fn render_leaf_opening(
        &self,
        node: NodeId,
        input: &JsonMap,
        cx: &Cx<'_>,
    ) -> Result<(Vec<Message>, Vec<Message>), RunError> {
        let p = &*self.program;
        let (name, model, sig, allow_scalar, playbook) = match &p.nodes[node] {
            Node::Predict(n) => (n.name, n.model, n.sig, true, None),
            Node::AgentLoop(n) => (
                n.name,
                n.model,
                n.sig,
                false,
                self.p_context(cx, n.context_policy).playbook,
            ),
            _ => unreachable!("conversation_leaf returns only leaves"),
        };
        let lm = self.p_model(p.syms.get(name), cx, model)?;
        let format = self.output_format(sig, &lm, allow_scalar);
        self.render_prompt(node, cx, input, format, playbook.as_deref())
    }

    fn render_prompt(
        &self,
        node: NodeId,
        cx: &Cx<'_>,
        input: &JsonMap,
        format: OutputFormat,
        playbook: Option<&str>,
    ) -> Result<(Vec<Message>, Vec<Message>), RunError> {
        let (sig, instruction, demo_slot) = match &self.program.nodes[node] {
            Node::Predict(n) => (n.sig, n.instruction, n.demos),
            Node::AgentLoop(n) => (n.sig, n.instruction, n.demos),
            _ => unreachable!("only prediction and agent nodes render prompts"),
        };
        let def = &self.program.sigs[sig];
        let templates = self.input_templates[sig].as_deref();
        let input_error = |err: anyhow::Error| RunError::Input {
            at: self.program.node_display(node).into(),
            message: format!("{err:#}"),
        };
        let mut system = self.system_prompt(sig, &self.p_text(cx, instruction), format);
        if let Some(playbook) = playbook {
            system.push_str("\n\n");
            system.push_str(playbook);
        }
        let rows = self.p_demos(cx, demo_slot);
        let mut prefix = Vec::with_capacity(1 + rows.len() * 2);
        prefix.push(Message::system(system));
        if !rows.is_empty() {
            let demos = self
                .rendered_demos(node, sig, rows, format, cx.overlay.as_ref())
                .map_err(input_error)?;
            prefix.extend_from_slice(&demos);
        }
        let mut suffix = ChatAdapter
            .signature_history(def, input)
            .map_err(input_error)?;
        suffix.push(Message::user(
            render_input(def, input, format, templates).map_err(input_error)?,
        ));
        Ok((prefix, suffix))
    }

    fn rendered_demos(
        &self,
        node: NodeId,
        sig: SigId,
        rows: &[DemoRow],
        format: OutputFormat,
        overlay: Option<&Arc<Overlay>>,
    ) -> anyhow::Result<Arc<[Message]>> {
        if let Some(cached) = self
            .demo_messages
            .read()
            .expect("demo cache poisoned")
            .get(&node)
            && cached.matches(format, overlay)
        {
            return Ok(Arc::clone(&cached.messages));
        }
        let messages: Arc<[Message]> = render_demos(
            &self.program.sigs[sig],
            &self.program.types,
            rows,
            format,
            self.input_templates[sig].as_deref(),
        )?
        .into();
        self.demo_messages
            .write()
            .expect("demo cache poisoned")
            .insert(
                node,
                CachedDemos {
                    format,
                    overlay: overlay.map(Arc::downgrade),
                    messages: Arc::clone(&messages),
                },
            );
        Ok(messages)
    }

    /// A conversation's run state: overlay + run meter, no frames, no
    /// collection — conversation turns build their [`LeafOutcome`] directly.
    fn conversation_cx<'a>(
        &self,
        overlay: Option<Arc<Overlay>>,
        meter: Arc<BudgetMeter>,
        prepare_replay: Option<PrepareReplayOutput<'a>>,
    ) -> Cx<'a> {
        Cx {
            overlay,
            meter,
            frames: SecondaryMap::new(),
            inputs: Vec::new(),
            parent: None,
            feedback: None,
            refine_feedback: None,
            leaves: None,
            prepare_replay,
        }
    }

    /// Overlay/program pairing check, shared by every entry point.
    fn check_overlay(&self, overlay: Option<&Arc<Overlay>>) -> Result<(), RunError> {
        if let Some(overlay) = overlay
            && overlay.base != self.program.meta.program_hash
        {
            return Err(RunError::Overlay {
                expected: overlay.base,
                got: self.program.meta.program_hash,
            });
        }
        Ok(())
    }

    /// Validates an input map against a signature's declared input fields.
    fn validate_input(
        &self,
        at: &str,
        sig: &SignatureDef,
        input: &JsonMap,
    ) -> Result<(), RunError> {
        for field in sig.inputs.iter() {
            match input.get(&*field.name) {
                Some(value) => {
                    if !json_matches_type(value, &field.ty, &self.program.types) {
                        return Err(RunError::Input {
                            at: at.into(),
                            message: format!(
                                "field `{}` does not match its declared type",
                                field.name
                            ),
                        });
                    }
                }
                None if field.ty.is_optional() => {}
                None => {
                    return Err(RunError::Input {
                        at: at.into(),
                        message: format!("missing input field `{}`", field.name),
                    });
                }
            }
        }
        Ok(())
    }

    fn prepare_replay_output(
        &self,
        sig: &SignatureDef,
        prepare: Option<PrepareReplayOutput<'_>>,
        output: &JsonMap,
    ) -> Result<(), String> {
        for field in sig.outputs.iter() {
            match output.get(&*field.name) {
                Some(value) if json_matches_type(value, &field.ty, &self.program.types) => {}
                Some(_) => {
                    return Err(format!(
                        "field `{}` does not match its declared type",
                        field.name
                    ));
                }
                None if field.ty.is_optional() => {}
                None => return Err(format!("missing output field `{}`", field.name)),
            }
        }
        if let Some(prepare) = prepare {
            prepare(output)?;
        }
        Ok(())
    }

    async fn run_inner(
        &self,
        input: JsonMap,
        overlay: Option<Arc<Overlay>>,
        budget: Budget,
        collect: bool,
        prepare_replay: Option<PrepareReplayOutput<'_>>,
    ) -> Result<RunOutput, RunError> {
        self.check_overlay(overlay.as_ref())?;

        // Input surface check against the program's external signature.
        self.validate_input("$", &self.program.sigs[self.program.sig], &input)?;

        let mut cx = Cx {
            overlay,
            meter: Arc::new(BudgetMeter::new(budget)),
            frames: SecondaryMap::new(),
            inputs: vec![input],
            parent: None,
            feedback: None,
            refine_feedback: None,
            leaves: collect.then(Vec::new),
            prepare_replay,
        };
        let output = self.eval(self.program.root, &mut cx).await?;
        Ok(RunOutput {
            output,
            leaves: cx.leaves.unwrap_or_default(),
        })
    }

    // -- node dispatch --------------------------------------------------------

    fn eval<'a>(
        &'a self,
        id: NodeId,
        cx: &'a mut Cx<'_>,
    ) -> BoxFuture<'a, Result<JsonMap, RunError>> {
        Box::pin(async move {
            let out = match &self.program.nodes[id] {
                Node::Predict(n) => self.eval_predict(id, n, cx).await?,
                Node::AgentLoop(n) => self.eval_agent(id, n, cx).await?,
                Node::Hole(n) => self.eval_hole(id, n, cx).await?,
                Node::Seq(n) => {
                    for &child in n.body.iter() {
                        self.eval(child, cx).await?;
                    }
                    self.resolve_exports(&n.out, cx)?
                }
                Node::ForkJoin(n) => {
                    let parent = &*cx;
                    let futures = n.branches.iter().map(|&branch| async move {
                        let mut branch_cx = parent.branch();
                        self.eval(branch, &mut branch_cx).await?;
                        Ok::<_, RunError>((branch_cx.frames, branch_cx.leaves))
                    });
                    // First error aborts siblings: try_join_all drops the
                    // remaining futures, whose open spans record Cancelled
                    // via guard-drop (RFC 0001 §3.1).
                    let all_frames = futures::future::try_join_all(futures).await?;
                    for (mut frames, leaves) in all_frames {
                        for (node, out) in frames.iter_mut() {
                            if let Some(out) = out.take()
                                && cx.frame(node).is_none()
                            {
                                cx.frames[node] = Some(out);
                            }
                        }
                        // Branch outcomes merge in declared branch order.
                        if let (Some(collected), Some(branch_leaves)) = (cx.leaves.as_mut(), leaves)
                        {
                            collected.extend(branch_leaves);
                        }
                    }
                    self.resolve_exports(&n.join, cx)?
                }
                Node::Route(n) => {
                    let at = self.program.node_display(id);
                    let value = self.resolve_port(&at, &n.on, cx)?;
                    let discriminant = match value.as_str() {
                        Some(s) => s.to_string(),
                        None => value.to_string(),
                    };
                    let arm = n
                        .arms
                        .iter()
                        .find(|(variant, _)| {
                            self.program.syms.get(*variant) == discriminant.as_str()
                        })
                        .map(|(_, arm)| *arm)
                        .or(n.default);
                    match arm {
                        Some(arm) => self.eval(arm, cx).await?,
                        None => {
                            return Err(RunError::Route {
                                at: at.into(),
                                value: discriminant,
                            });
                        }
                    }
                }
                Node::Retry(n) => {
                    let mut attempt = 0u32;
                    loop {
                        let options = crate::current_sampling_options().child(
                            format!("retry-{}", self.program.node_display(id)),
                            attempt as usize,
                        );
                        match crate::with_sampling_options(options, self.eval(n.child, cx)).await {
                            Ok(out) => break out,
                            Err(err) if err.retryable() && attempt + 1 < n.max_attempts.get() => {
                                attempt += 1;
                                if n.feedback && matches!(err, RunError::Parse { .. }) {
                                    cx.feedback = Some(format!(
                                        "Your previous response could not be parsed: {err}. \
                                         Respond again, following the required \
                                         `[[ ## field ## ]]` output format exactly."
                                    ));
                                }
                                if n.backoff_ms > 0 {
                                    tokio::time::sleep(std::time::Duration::from_millis(
                                        n.backoff_ms as u64,
                                    ))
                                    .await;
                                }
                            }
                            Err(err) => return Err(err),
                        }
                    }
                }
                Node::Refine(n) => {
                    let saved = cx.refine_feedback.take();
                    let result = self.eval_refine(n, cx).await;
                    cx.refine_feedback = saved;
                    result?
                }
                Node::Loop(n) => {
                    let enclosing = cx.input().cloned().unwrap_or_default();
                    let mut carry = JsonMap::new();
                    let at = self.program.node_display(id);
                    let mut result = None;
                    for iter in 0..n.max_iters.get() {
                        let mut frame = enclosing.clone();
                        frame.extend(carry.clone());
                        cx.inputs.push(frame);
                        let body = self.eval(n.body, cx).await;
                        if body.is_err() {
                            cx.inputs.pop();
                        }
                        body?;
                        // Rebind carried values from this iteration.
                        let mut next = JsonMap::new();
                        for b in n.carry.iter() {
                            let name = self.program.syms.get(b.dst).to_string();
                            next.insert(name, self.resolve_port(&at, &b.src, cx)?);
                        }
                        let continue_ = match &n.while_ {
                            Some(port) => {
                                self.resolve_port(&at, port, cx)?.as_bool().unwrap_or(false)
                            }
                            None => true,
                        };
                        cx.inputs.pop();
                        carry = next;
                        if !continue_ || iter + 1 == n.max_iters.get() {
                            // Exports resolve in the final iteration's scope
                            // (body outputs + final carry visible).
                            let mut frame = enclosing.clone();
                            frame.extend(carry.clone());
                            cx.inputs.push(frame);
                            let out = self.resolve_exports(&n.out, cx);
                            cx.inputs.pop();
                            result = Some(out?);
                            break;
                        }
                    }
                    result.expect("bounded loops always take the exit branch")
                }
            };
            cx.frames[id] = Some(out.clone());
            Ok(out)
        })
    }

    async fn eval_refine(
        &self,
        n: &crate::ir::graph::RefineNode,
        cx: &mut Cx<'_>,
    ) -> Result<JsonMap, RunError> {
        let mut best: Option<(f64, usize, JsonMap, Frames)> = None;
        for round in 0..n.max_rounds.get() {
            let child_scope = crate::current_sampling_options().child(
                format!("refine-{}", self.program.node_display(n.child)),
                round as usize,
            );
            let out = crate::with_sampling_options(child_scope, self.eval(n.child, cx)).await?;
            let judge_scope = crate::current_sampling_options().child(
                format!("refine-judge-{}", self.program.node_display(n.judge)),
                round as usize,
            );
            let judge_out =
                crate::with_sampling_options(judge_scope, self.eval(n.judge, cx)).await?;
            let score = judge_out
                .get("score")
                .and_then(Value::as_f64)
                .unwrap_or(f64::NEG_INFINITY);
            if best
                .as_ref()
                .map_or(true, |(best_score, _, _, _)| score > *best_score)
            {
                best = Some((score, round as usize, out.clone(), cx.frames.clone()));
            }
            if score >= n.threshold {
                let key = format!(
                    "refine.{}.selected_round",
                    self.program.node_display(n.child)
                );
                crate::trace::capture::set_trace_tag(key, (round + 1).to_string());
                crate::trace::capture::set_trace_tag(
                    format!(
                        "refine.{}.selected_score",
                        self.program.node_display(n.child)
                    ),
                    score.to_string(),
                );
                return Ok(out);
            }
            let feedback = judge_out.get("feedback").cloned().unwrap_or(Value::Null);
            let field = self.program.syms.get(n.feedback_field).to_string();
            cx.refine_feedback = Some((n.child, field, feedback));
        }
        let (score, round, out, frames) = best.expect("refine has at least one bounded attempt");
        cx.frames = frames;
        crate::trace::capture::set_trace_tag(
            format!(
                "refine.{}.selected_round",
                self.program.node_display(n.child)
            ),
            (round + 1).to_string(),
        );
        crate::trace::capture::set_trace_tag(
            format!(
                "refine.{}.selected_score",
                self.program.node_display(n.child)
            ),
            score.to_string(),
        );
        Ok(out)
    }

    // -- leaves ---------------------------------------------------------------

    async fn eval_predict(
        &self,
        id: NodeId,
        n: &PredictNode,
        cx: &mut Cx<'_>,
    ) -> Result<JsonMap, RunError> {
        let p = &*self.program;
        let at = p.syms.get(n.name).to_string();
        let def = &p.sigs[n.sig];
        let input = self.resolve_bindings(&at, Some(id), &n.binding, cx)?;

        let lm = self.p_model(&at, cx, n.model)?;
        let format = self.output_format(n.sig, &lm, true);
        let (prefix, mut suffix) = self.render_prompt(id, cx, &input, format, None)?;
        if let Some(feedback) = cx.feedback.take() {
            suffix.push(Message::user(feedback));
        }
        let guard = begin_span_lazy(|| SpanRequest {
            component: &at,
            prefix: Some(&prefix),
            suffix: &suffix,
            input: Some(input.clone()),
            model: &lm.config,
            request_hash: None,
        });

        let mut messages = prefix;
        messages.extend(suffix);

        // Replay scope (RFC 0001 §4d/e, RFC 0003 M-1): intercept above the
        // LM. A served call reserves no budget, constructs no client, and
        // reaches no provider.
        let replay = crate::trace::replay::intercept(&at, &lm.config, &messages, |output| {
            self.prepare_replay_output(def, cx.prepare_replay, output)
        });
        match replay {
            Some(crate::trace::replay::ReplayDirective::Serve(span)) => {
                let span = span.span();
                let output = span
                    .output
                    .clone()
                    .expect("replay serves only spans with parsed output");
                // Parity with the static lane's `serve_recorded_span`: served
                // predictions carry no per-field parse metadata — the
                // recording stores the parsed output, not the parser's
                // field-level bookkeeping.
                if let Some(leaves) = cx.leaves.as_mut() {
                    leaves.push(LeafOutcome {
                        name: at.clone(),
                        raw_response: span.raw_output.clone().unwrap_or_default(),
                        call_ids: Vec::new(),
                        field_meta: IndexMap::new(),
                        usage: span.usage,
                        model_config_hash: crate::trace::ModelEntry::from_config(&lm.config)
                            .config_hash,
                        span_id: guard.as_ref().map(|guard| guard.id()),
                        tool_calls: span.requested_tool_calls(),
                        tool_executions: Vec::new(),
                    });
                }
                if let Some(guard) = guard {
                    guard.finish(SpanOutcome {
                        events: span.events.clone(),
                        raw_output: span.raw_output.clone(),
                        output: Some(output.clone()),
                        usage: span.usage,
                        error: None,
                    });
                }
                return Ok(output);
            }
            Some(crate::trace::replay::ReplayDirective::Refuse(err)) => {
                if let Some(guard) = guard {
                    guard.finish(span_error(
                        crate::trace::SpanErrorKind::Lm,
                        err.to_string(),
                        Vec::new(),
                        None,
                        LmUsage::default(),
                    ));
                }
                return Err(RunError::Replay {
                    at: at.into(),
                    source: err,
                });
            }
            Some(crate::trace::replay::ReplayDirective::Live) | None => {}
        }

        let (output, leaf, _) = self
            .exchange_predict(
                PredictExchange {
                    at: &at,
                    sig: n.sig,
                    format,
                    lm: &lm,
                    meter: &cx.meter,
                    guard,
                },
                messages,
                |_| ChatAdapter.signature_tools(def, &input),
                cx.leaves.is_some(),
            )
            .await?;
        if let Some(leaves) = cx.leaves.as_mut() {
            leaves.extend(leaf);
        }
        Ok(output)
    }

    async fn eval_hole(
        &self,
        id: NodeId,
        n: &HoleNode,
        cx: &mut Cx<'_>,
    ) -> Result<JsonMap, RunError> {
        let p = &*self.program;
        let at = p.syms.get(n.name);
        let def = &p.sigs[n.sig];
        let input = self.resolve_bindings(at, Some(id), &n.binding, cx)?;

        // Resolve the implementation up front — its hash is part of the
        // replay preimage (RFC 0003 §4.4).
        let imp = match &n.imp {
            HoleImpl::Sandboxed { code } => match self.p_value(cx, *code) {
                ParamValue::Code { source, hash, .. } => ResolvedHoleImpl::Sandboxed {
                    code: *code,
                    source,
                    hash: *hash,
                },
                other => {
                    return Err(RunError::Internal {
                        at: at.into(),
                        message: format!("code slot resolved to {:?}", other.kind()),
                    });
                }
            },
            HoleImpl::Host { hash } => ResolvedHoleImpl::Host { hash: *hash },
        };
        let (guard, replay) = {
            let hash = std::cell::OnceCell::new();
            let request_hash = || *hash.get_or_init(|| hole_request_hash(&imp, &input, &n.caps));
            let model = std::cell::OnceCell::new();
            let guard = begin_span_lazy(|| SpanRequest {
                component: at,
                prefix: None,
                suffix: &[],
                input: Some(input.clone()),
                model: model.get_or_init(|| LMConfig {
                    model: match &imp {
                        ResolvedHoleImpl::Sandboxed { .. } => "sandbox:quickjs".into(),
                        ResolvedHoleImpl::Host { .. } => "host:extern".into(),
                    },
                    ..LMConfig::default()
                }),
                request_hash: Some(request_hash()),
            });
            (
                guard,
                crate::trace::replay::intercept_hashed(at, request_hash, |output| {
                    self.prepare_replay_output(def, cx.prepare_replay, output)
                }),
            )
        };

        // Replay (RFC 0003 M-1): a hole keys on (impl ++ input ++ caps); a
        // served span returns the recorded, already-coerced output without
        // touching the sandbox or the host fn.
        match replay {
            Some(crate::trace::replay::ReplayDirective::Serve(span)) => {
                let span = span.span();
                let output = span
                    .output
                    .clone()
                    .expect("replay serves only spans with parsed output");
                if let Some(guard) = guard {
                    guard.finish(SpanOutcome {
                        events: span.events.clone(),
                        raw_output: span.raw_output.clone(),
                        output: Some(output.clone()),
                        usage: span.usage,
                        error: None,
                    });
                }
                return Ok(output);
            }
            Some(crate::trace::replay::ReplayDirective::Refuse(err)) => {
                if let Some(guard) = guard {
                    guard.finish(span_error(
                        crate::trace::SpanErrorKind::Tool,
                        err.to_string(),
                        Vec::new(),
                        None,
                        LmUsage::default(),
                    ));
                }
                return Err(RunError::Replay {
                    at: at.into(),
                    source: err,
                });
            }
            Some(crate::trace::replay::ReplayDirective::Live) | None => {}
        }

        let observation = guard.map(|guard| (guard, Value::Object(input.clone()), Instant::now()));
        let result = in_span(
            observation.as_ref().map(|(guard, _, _)| guard),
            pin!(self.execute_hole_impl(at, &imp, input, n.sig, &n.caps)),
        )
        .await;
        let observation = observation
            .map(|(guard, args, started)| (guard, args, started.elapsed().as_micros() as u64));

        match result {
            Ok((invoked_as, value)) => {
                let raw = value.to_string();
                let output = coerce_outputs(at, def, &p.types, value, &raw);
                let Some((guard, args, duration_us)) = observation else {
                    return output;
                };
                let events = vec![SpanEvent::ToolRun {
                    id: String::new(),
                    name: invoked_as,
                    args,
                    result: raw.clone(),
                    duration_us,
                    error: None,
                }];
                guard.finish(match &output {
                    Ok(output) => SpanOutcome {
                        events,
                        raw_output: Some(raw),
                        output: Some(output.clone()),
                        usage: LmUsage::default(),
                        error: None,
                    },
                    Err(error) => span_error(
                        crate::trace::SpanErrorKind::Parse,
                        error.to_string(),
                        events,
                        Some(raw),
                        LmUsage::default(),
                    ),
                });
                output
            }
            Err(err) => {
                if let Some((guard, _, _)) = observation {
                    guard.finish(span_error(
                        crate::trace::SpanErrorKind::Tool,
                        err.to_string(),
                        Vec::new(),
                        None,
                        LmUsage::default(),
                    ));
                }
                Err(err)
            }
        }
    }

    /// Executes a hole's resolved implementation. Returns the name it was
    /// invoked as (the registered sandbox tool name, or the leaf name for
    /// host holes) and the raw JSON result.
    async fn execute_hole_impl(
        &self,
        at: &str,
        imp: &ResolvedHoleImpl<'_>,
        input: JsonMap,
        sig: SigId,
        caps: &CapSet,
    ) -> Result<(String, Value), RunError> {
        match imp {
            ResolvedHoleImpl::Sandboxed { code, source, hash } => {
                // The capability gate is constructive (§3.5): caps were
                // checked at build/load and the sandbox only carries granted
                // host functions — undeclared authority is unreachable here.
                let sandbox = self.sandbox.as_ref().ok_or_else(|| RunError::Internal {
                    at: at.into(),
                    message: "hole evaluated without a sandbox (load should have refused)".into(),
                })?;
                let name = self
                    .ensure_registered(at, sandbox.as_ref(), *code, (source, *hash), sig, caps)
                    .await?;
                let value = sandbox
                    .execute_scoped(
                        dsrs_tools::ToolInvocation::new(name.clone(), Value::Object(input)),
                        &caps.iter().map(str::to_string).collect::<Vec<_>>(),
                    )
                    .await
                    .map_err(|source| RunError::Hole {
                        at: at.into(),
                        source,
                    })?;
                Ok((name, value))
            }
            ResolvedHoleImpl::Host { .. } => {
                let host = self.host_holes.get(at).ok_or_else(|| RunError::Internal {
                    at: at.into(),
                    message: "host hole not bound (load should have refused)".into(),
                })?;
                let value = host(input).await.map_err(|message| RunError::Hole {
                    at: at.into(),
                    source: dsrs_tools::ExecError::Internal { message },
                })?;
                Ok((at.to_string(), value))
            }
        }
    }

    async fn eval_agent(
        &self,
        id: NodeId,
        n: &AgentLoopNode,
        cx: &mut Cx<'_>,
    ) -> Result<JsonMap, RunError> {
        let p = &*self.program;
        let at = p.syms.get(n.name).to_string();
        let def = &p.sigs[n.sig];
        let input = self.resolve_bindings(&at, Some(id), &n.binding, cx)?;

        let agent = self.prepare_agent(n, cx, true);
        let lm = self.p_model(&at, cx, n.model)?;
        let format = self.output_format(n.sig, &lm, false);
        let (prefix, mut suffix) =
            self.render_prompt(id, cx, &input, format, agent.policy.playbook.as_deref())?;
        if let Some(feedback) = cx.feedback.take() {
            suffix.push(Message::user(feedback));
        }
        let prefix_len = prefix.len();
        let mut messages = prefix;
        messages.extend(suffix);
        let chat = Chat::new(messages);
        let opening = TurnRequest {
            at: &at,
            prefix: &chat.messages[..prefix_len],
            suffix: &chat.messages[prefix_len..],
            input: Some(&input),
            model: &lm.config,
        };
        let (mut guard, replay) = self.begin_agent_request(cx, &agent, opening, prefix_len);

        match replay {
            Some(crate::trace::replay::ReplayDirective::Serve(span)) => {
                let span = span.span();
                let output = span
                    .output
                    .clone()
                    .expect("replay serves only spans with parsed output");
                if let Some(leaves) = cx.leaves.as_mut() {
                    // Served evaluations carry no per-field parse metadata;
                    // the tool record is reconstructed from the recording.
                    let (tool_calls, tool_executions) = span.recorded_tools();
                    leaves.push(LeafOutcome {
                        name: at.clone(),
                        raw_response: span.raw_output.clone().unwrap_or_default(),
                        call_ids: Vec::new(),
                        field_meta: IndexMap::new(),
                        usage: span.usage,
                        model_config_hash: crate::trace::ModelEntry::from_config(&lm.config)
                            .config_hash,
                        span_id: guard.as_ref().map(|guard| guard.id()),
                        tool_calls,
                        tool_executions,
                    });
                }
                if let Some(guard) = guard {
                    guard.finish(SpanOutcome {
                        events: span.events.clone(),
                        raw_output: span.raw_output.clone(),
                        output: Some(output.clone()),
                        usage: span.usage,
                        error: None,
                    });
                }
                return Ok(output);
            }
            Some(crate::trace::replay::ReplayDirective::Refuse(err)) => {
                if let Some(guard) = guard {
                    guard.finish(span_error(
                        crate::trace::SpanErrorKind::Lm,
                        err.to_string(),
                        Vec::new(),
                        None,
                        LmUsage::default(),
                    ));
                }
                return Err(RunError::Replay {
                    at: at.into(),
                    source: err,
                });
            }
            Some(crate::trace::replay::ReplayDirective::Live) | None => {}
        }

        let mut run = AgentRun::default();
        let surface =
            run.finish_on_setup_error(&mut guard, self.build_agent_surface(&at, cx, agent).await)?;
        let meter = Arc::new(BudgetMeter::child(&cx.meter, node_budget(&n.budget)));
        let loop_cx = AgentLoopCx {
            at: &at,
            n,
            def,
            lm: &lm,
            surface: &surface,
            prefix_len,
            meter: &meter,
            run_meter: &cx.meter,
        };
        let outcome = in_span(
            guard.as_ref(),
            pin!(self.agent_loop(&loop_cx, chat, &mut run, 0)),
        )
        .await;
        match outcome {
            Ok(LoopOutcome::Done(done)) => {
                let collect = cx.leaves.is_some().then(|| {
                    (
                        at.as_str(),
                        crate::trace::ModelEntry::from_config(&lm.config).config_hash,
                    )
                });
                let (output, leaf, _) = run.finish(guard, done, collect);
                if let Some(leaves) = cx.leaves.as_mut() {
                    leaves.extend(leaf);
                }
                Ok(output)
            }
            Ok(LoopOutcome::Suspend { .. }) => {
                unreachable!("dispatching loops never suspend")
            }
            Err(err) => Err(run.finish_loop_error(guard, err)),
        }
    }

    /// Runs the agent loop from `start_turn`. Dispatching mode
    /// executes tool calls through its dispatcher. Caller-managed turns
    /// return [`LoopOutcome::Suspend`],
    /// leaving execution to the caller
    /// ([`resume_conversation`](Self::resume_conversation) re-enters here at
    /// `next_turn`). Stop-tool and budget semantics are identical in both
    /// modes.
    async fn agent_loop(
        &self,
        lc: &AgentLoopCx<'_>,
        mut chat: Chat,
        run: &mut AgentRun,
        start_turn: u32,
    ) -> Result<LoopOutcome, RunError> {
        let types = &self.program.types;
        let format = self.output_format(lc.n.sig, lc.lm, false);

        for turn in start_turn..lc.n.stop.max_turns.get() {
            if lc.meter.try_reserve_call().is_err() {
                return match lc.n.budget.on_exhausted {
                    BudgetPolicy::Fail => Err(RunError::Budget { at: lc.at.into() }),
                    BudgetPolicy::Finalize => self.finalize(lc, chat, run).await,
                };
            }

            run.events
                .extend(prune_history(&mut chat, lc.prefix_len, &lc.surface.policy));
            let options = self.output_options(lc.n.sig, format);
            let response =
                lm_call_toolset(lc.lm, chat, &lc.surface.toolset, lc.at, options).await?;
            lc.meter.record_usage(&response.usage);
            run.usage = run.usage + response.usage;
            run.events.extend(response.events);
            run.call_ids.extend(response.history.iter().map(|h| h.id()));
            chat = response.chat;

            if !response.tool_calls.is_empty() {
                // Stop-tool check: its args become the raw final output.
                if let Some(call) = response
                    .tool_calls
                    .iter()
                    .find(|call| lc.surface.stop_names.contains(&call.function.name))
                {
                    let args = call.function.arguments.clone();
                    let raw = args.to_string();
                    let output = coerce_outputs(lc.at, lc.def, types, args, &raw)?;
                    return Ok(LoopOutcome::Done(AgentCompletion {
                        output,
                        raw,
                        field_meta: IndexMap::new(),
                        chat,
                    }));
                }

                let Some(dispatch) = &lc.surface.dispatch else {
                    return Ok(LoopOutcome::Suspend {
                        calls: response.tool_calls,
                        chat,
                        next_turn: turn + 1,
                    });
                };

                let mut blocks = Vec::with_capacity(response.tool_calls.len());
                for call in &response.tool_calls {
                    let started = Instant::now();
                    let (result, error) = self.execute_agent_tool(lc, dispatch, call).await;
                    blocks.push(run.record_tool_result(
                        call,
                        result,
                        started.elapsed().as_micros() as u64,
                        error,
                    ));
                }
                chat.push_message(Message::with_content(Role::User, blocks));
                continue;
            }

            // Text turn.
            if lc.n.stop.until_parse {
                let raw = response.output.content();
                let parsed = parse_response(lc.def, types, &response.output, format);
                if let Some(history) = &response.history {
                    history.parsed(&parsed).await;
                }
                match parsed {
                    Ok((output, metas)) => {
                        return Ok(LoopOutcome::Done(AgentCompletion {
                            output,
                            raw,
                            field_meta: metas,
                            chat,
                        }));
                    }
                    Err(err) if turn + 1 < lc.n.stop.max_turns.get() => {
                        run.push_prompt(
                            &mut chat,
                            format!(
                                "Your response could not be parsed: {err}. Respond again, \
                             producing every output field in the required \
                             required output format."
                            ),
                        );
                    }
                    Err(err) => {
                        return Err(RunError::Parse {
                            at: lc.at.into(),
                            raw,
                            source: Some(Box::new(err)),
                            usage: run.usage,
                        });
                    }
                }
            }
        }

        // Turns exhausted without an accepted answer.
        match lc.n.budget.on_exhausted {
            BudgetPolicy::Fail => Err(RunError::Budget { at: lc.at.into() }),
            BudgetPolicy::Finalize => self.finalize(lc, chat, run).await,
        }
    }

    /// One forced tool-less closing round-trip ("wrap up now"), reserved
    /// against the RUN meter — the node budget is spent by definition.
    async fn finalize(
        &self,
        lc: &AgentLoopCx<'_>,
        mut chat: Chat,
        run: &mut AgentRun,
    ) -> Result<LoopOutcome, RunError> {
        lc.run_meter
            .try_reserve_call()
            .map_err(|_| RunError::Budget { at: lc.at.into() })?;
        let format = self.output_format(lc.n.sig, lc.lm, false);
        let options = self.output_options(lc.n.sig, format);
        run.push_prompt(
            &mut chat,
            "Budget exhausted — wrap up now. Produce the final output fields in the required \
             required output format, without calling any tools.",
        );
        let response = lc
            .lm
            .call_with_toolset_and_options_ref(
                chat,
                &ToolSet::default(),
                ToolLoopMode::Auto,
                options,
            )
            .await
            .map_err(|err| RunError::Lm {
                at: lc.at.into(),
                source: LmError::Provider {
                    provider: lc.lm.config.model.clone(),
                    message: err.to_string(),
                    source: Some(err.reallocate_into_boxed_dyn_error_without_backtrace()),
                },
            })?;
        lc.run_meter.record_usage(&response.usage);
        run.usage = run.usage + response.usage;
        run.events.extend(response.events);
        run.call_ids.extend(response.history.iter().map(|h| h.id()));
        let raw = response.output.content();
        let parsed = parse_response(lc.def, &self.program.types, &response.output, format);
        if let Some(history) = &response.history {
            history.parsed(&parsed).await;
        }
        let (output, metas) = parsed.map_err(|err| RunError::Parse {
            at: lc.at.into(),
            raw: raw.clone(),
            source: Some(Box::new(err)),
            usage: run.usage,
        })?;
        Ok(LoopOutcome::Done(AgentCompletion {
            output,
            raw,
            field_meta: metas,
            chat: response.chat,
        }))
    }

    /// Executes one agent tool call. Failures are conversational: the error
    /// text goes back to the model (LATM-style repair), never up the tree.
    async fn execute_agent_tool(
        &self,
        lc: &AgentLoopCx<'_>,
        dispatch: &AgentDispatch,
        call: &rig::message::ToolCall,
    ) -> (String, Option<String>) {
        let outcome: Result<String, String> = match dispatch.code_mode.as_ref() {
            Some(surface) if call.function.name == dsrs_tools::RUN_JS_TOOL_NAME => {
                execute_code_mode_script(surface, &call.function.arguments).await
            }
            _ => self.dispatch_agent_tool(lc.at, dispatch, call).await,
        };
        let (text, error) = match outcome {
            Ok(text) => (text, None),
            Err(message) => (message.clone(), Some(message)),
        };
        (clip_tool_result(text, &lc.surface.policy), error)
    }

    /// Routes one tool call to its bound executor (host `ToolDyn` or the
    /// sandbox), by declared name.
    async fn dispatch_agent_tool(
        &self,
        at: &str,
        dispatch: &AgentDispatch,
        call: &rig::message::ToolCall,
    ) -> Result<String, String> {
        let name = call.function.name.as_str();
        match dispatch.by_name.get(name) {
            None => Err(format!("Tool '{name}' not found")),
            Some(&tool_id) => match &self.tool_exec[tool_id] {
                Some(ToolExec::Host(tool)) => tool
                    .call(call.function.arguments.to_string())
                    .await
                    .map_err(|err| format!("tool `{name}` failed: {err}")),
                Some(ToolExec::Sandboxed(code)) => {
                    self.execute_sandboxed_tool(at, dispatch, tool_id, *code, call)
                        .await
                }
                None => Err(format!("tool `{name}` is not bound")),
            },
        }
    }

    async fn execute_sandboxed_tool(
        &self,
        at: &str,
        dispatch: &AgentDispatch,
        tool_id: ToolId,
        code: ParamId,
        call: &rig::message::ToolCall,
    ) -> Result<String, String> {
        let sandbox = self
            .sandbox
            .as_ref()
            .ok_or_else(|| "sandbox unavailable".to_string())?;
        let (source, hash) = dispatch
            .sandbox_code
            .get(&tool_id)
            .ok_or_else(|| "sandboxed tool has no code".to_string())?;
        let tool = &self.program.tools[tool_id];
        let registered_name = self
            .ensure_registered(
                at,
                sandbox.as_ref(),
                code,
                (source, *hash),
                tool.sig,
                &tool.caps,
            )
            .await
            .map_err(|err| err.to_string())?;
        sandbox
            .execute_scoped(
                dsrs_tools::ToolInvocation::new(registered_name, call.function.arguments.clone()),
                &tool.caps.iter().map(str::to_string).collect::<Vec<_>>(),
            )
            .await
            .map(|value| value.to_string())
            .map_err(|err| err.to_llm_json())
    }

    fn prepare_agent<'a>(
        &self,
        node: &'a AgentLoopNode,
        cx: &Cx<'_>,
        dispatch_tools: bool,
    ) -> PreparedAgent<'a> {
        PreparedAgent {
            node,
            tools: self.p_tool_set(cx, node.tool_set, &node.tools),
            policy: self.p_context(cx, node.context_policy),
            execution: if dispatch_tools {
                AgentExecution::Dispatch(self.code_mode)
            } else {
                AgentExecution::CallerManaged
            },
        }
    }

    fn begin_agent_request(
        &self,
        cx: &Cx<'_>,
        agent: &PreparedAgent<'_>,
        opening: TurnRequest<'_>,
        prefix_len: usize,
    ) -> (
        Option<SpanGuard>,
        Option<crate::trace::replay::ReplayDirective>,
    ) {
        use crate::trace::replay::{ReplayDirective, intercept_agent};
        use std::cell::OnceCell;

        let prompt = OnceCell::new();
        let opening_hash = || *prompt.get_or_init(|| opening.prompt_hash());
        let legacy = OnceCell::new();
        let legacy_agent_hash =
            || *legacy.get_or_init(|| self.agent_request_hash_v2(cx, agent, opening_hash()));
        let protected_prefix_len = agent.policy.max_history_turns.map(|_| prefix_len);
        let request_hash = || {
            crate::utils::hash::stable_hash_debug(&(
                "dsrs-agent-v3",
                legacy_agent_hash(),
                protected_prefix_len.map(|length| length as u64),
            ))
        };
        let replay = intercept_agent(
            opening.at,
            opening.prefix.len() + opening.suffix.len(),
            protected_prefix_len,
            opening_hash,
            legacy_agent_hash,
            request_hash,
            |output| {
                self.prepare_replay_output(
                    &self.program.sigs[agent.node.sig],
                    cx.prepare_replay,
                    output,
                )
            },
        );
        let guard = opening.open_span(|| {
            // Copies without chat edits retain their original matching contract.
            Some(match &replay {
                Some(ReplayDirective::Serve(span)) if !span.span().has_chat_edits() => {
                    span.span().request_hash
                }
                _ => request_hash(),
            })
        });
        (guard, replay)
    }

    /// Materializes the prepared tool selection for live execution.
    async fn build_agent_surface(
        &self,
        at: &str,
        cx: &Cx<'_>,
        agent: PreparedAgent<'_>,
    ) -> Result<AgentSurface, RunError> {
        let p = &*self.program;
        let tools = &agent.tools;
        let mut definitions = Vec::with_capacity(tools.len());
        let mut dispatch =
            matches!(agent.execution, AgentExecution::Dispatch(_)).then(AgentDispatch::default);
        for &tool_id in tools.iter() {
            let tool = &p.tools[tool_id];
            let name = p.syms.get(tool.name).to_string();
            if let Some(dispatch) = &mut dispatch {
                dispatch.by_name.insert(name.clone(), tool_id);
                if let ToolKind::Sandboxed { code } = tool.kind
                    && let ParamValue::Code { source, hash, .. } = self.p_value(cx, code)
                {
                    dispatch
                        .sandbox_code
                        .insert(tool_id, (source.clone(), *hash));
                }
            }
            definitions.push(rig::completion::ToolDefinition {
                name,
                description: self.p_text(cx, tool.desc),
                parameters: self.tool_input_schema(tool.sig).clone(),
            });
        }
        let stop_names: Vec<String> = agent
            .node
            .stop
            .stop_tools
            .iter()
            .map(|&t| p.syms.get(p.tools[t].name).to_string())
            .collect();
        // Code Mode: collapse the non-stop tool surface into one `run_js`
        // definition (stop tools stay individual — the loop must see their
        // calls by name to end).
        if let Some(dispatch) = &mut dispatch
            && let Some(config) = agent.execution.code_mode()
        {
            dispatch.code_mode = self
                .build_code_mode_surface(
                    at,
                    config,
                    tools,
                    &mut definitions,
                    &dispatch.sandbox_code,
                    &stop_names,
                )
                .await?;
        }
        Ok(AgentSurface {
            toolset: ToolSet::from_definitions(definitions),
            stop_names,
            policy: agent.policy,
            dispatch,
        })
    }

    fn agent_request_hash_v2(
        &self,
        cx: &Cx<'_>,
        agent: &PreparedAgent<'_>,
        opening_hash: u64,
    ) -> u64 {
        use crate::utils::hash::{HashWriter, StableHasher};
        use std::fmt::Write as _;
        use std::hash::Hasher as _;

        let p = &self.program;
        let n = agent.node;
        let mut hasher = StableHasher::new();
        let mut writer = HashWriter(&mut hasher);
        let _ = write!(
            writer,
            "dsrs-agent-v2 {opening_hash:016x} {:?} {:?} {:?} {} {}",
            agent.policy,
            agent.execution.code_mode(),
            n.budget,
            n.stop.max_turns,
            n.stop.until_parse
        );
        for &id in n.stop.stop_tools.iter() {
            let _ = write!(writer, " stop {:?}", p.syms.get(p.tools[id].name));
        }
        for &id in agent.tools.iter() {
            let tool = &p.tools[id];
            let code = match tool.kind {
                ToolKind::Host => None,
                ToolKind::Sandboxed { code } => match self.p_value(cx, code) {
                    ParamValue::Code { hash, .. } => Some(*hash),
                    _ => None,
                },
            };
            let _ = write!(
                writer,
                " tool {:?} {:?} {:?} {:?} {code:?}",
                p.syms.get(tool.name),
                self.p_value(cx, tool.desc),
                self.tool_input_schema(tool.sig),
                tool.caps
            );
        }
        hasher.finish()
    }

    /// Builds one agent loop's Code Mode surface: wraps every non-stop tool
    /// as a sandbox [`Capability`](dsrs_tools::Capability) (host tools call
    /// straight through `ToolDyn`; sandboxed tools route through the
    /// executor under their registered names) and replaces their definitions
    /// with a single `run_js` definition whose description is generated from
    /// the *overlay-resolved* tool descriptions — `ToolDesc` genes keep
    /// flowing into the surface the model sees. A loop with only stop tools
    /// keeps its individual tool definitions.
    async fn build_code_mode_surface(
        &self,
        at: &str,
        config: dsrs_tools::SandboxConfig,
        tools: &[ToolId],
        definitions: &mut Vec<rig::completion::ToolDefinition>,
        sandbox_code: &HashMap<ToolId, (String, u64)>,
        stop_names: &[String],
    ) -> Result<Option<CodeModeSurface>, RunError> {
        let p = &*self.program;
        let internal = |message: String| RunError::Internal {
            at: at.into(),
            message,
        };
        let mut kept = Vec::new();
        let mut apis = Vec::new();
        let mut capabilities = Vec::new();
        // `definitions[i]` was built from `tools[i]` (the ToolSet-selected
        // surface) — same order.
        for (&tool_id, definition) in tools.iter().zip(std::mem::take(definitions)) {
            if stop_names.contains(&definition.name) {
                kept.push(definition);
                continue;
            }
            // Collisions were refused at load; names here are unique.
            let js_name = dsrs_tools::js_identifier(&definition.name);
            let capability = match &self.tool_exec[tool_id] {
                Some(ToolExec::Host(tool)) => dsrs_tools::Capability::wrap_tool(
                    js_name.clone(),
                    definition.description.clone(),
                    definition.name.clone(),
                    Arc::clone(tool),
                ),
                Some(ToolExec::Sandboxed(code)) => {
                    let sandbox = Arc::clone(self.sandbox.as_ref().ok_or_else(|| {
                        internal("sandbox unavailable (load should have refused)".to_string())
                    })?);
                    let (source, hash) = sandbox_code.get(&tool_id).ok_or_else(|| {
                        internal(format!("sandboxed tool `{}` has no code", definition.name))
                    })?;
                    let registered = self
                        .ensure_registered(
                            at,
                            sandbox.as_ref(),
                            *code,
                            (source, *hash),
                            p.tools[tool_id].sig,
                            &p.tools[tool_id].caps,
                        )
                        .await?;
                    let tool_name = definition.name.clone();
                    let caps: Vec<String> =
                        p.tools[tool_id].caps.iter().map(str::to_string).collect();
                    dsrs_tools::Capability::new(
                        js_name.clone(),
                        definition.description.clone(),
                        move |args| {
                            let sandbox = Arc::clone(&sandbox);
                            let registered = registered.clone();
                            let tool_name = tool_name.clone();
                            let caps = caps.clone();
                            async move {
                                sandbox
                                    .execute_scoped(
                                        dsrs_tools::ToolInvocation::new(registered, args),
                                        &caps,
                                    )
                                    .await
                                    .map_err(|err| format!("tool `{tool_name}` failed: {err}"))
                            }
                        },
                    )
                }
                None => {
                    return Err(internal(format!(
                        "tool `{}` is not bound (load should have refused)",
                        definition.name
                    )));
                }
            };
            apis.push(dsrs_tools::ToolApi {
                js_name,
                description: definition.description,
                parameters: definition.parameters,
            });
            capabilities.push(capability);
        }
        if apis.is_empty() {
            *definitions = kept;
            return Ok(None);
        }
        let run_js = rig::completion::ToolDefinition {
            name: dsrs_tools::RUN_JS_TOOL_NAME.to_string(),
            description: dsrs_tools::code_mode_description(&apis),
            parameters: dsrs_tools::run_js_parameters(),
        };
        *definitions = std::iter::once(run_js).chain(kept).collect();
        Ok(Some(CodeModeSurface {
            capabilities,
            config,
        }))
    }

    // -- param + port resolution ----------------------------------------------

    fn p_value<'a>(&'a self, cx: &'a Cx<'_>, id: ParamId) -> &'a ParamValue {
        match &cx.overlay {
            Some(overlay) => overlay.resolve(&self.program, id),
            None => &self.program.params[id].default,
        }
    }

    fn p_text(&self, cx: &Cx<'_>, id: ParamId) -> String {
        match self.p_value(cx, id) {
            ParamValue::Instruction { text } | ParamValue::ToolDesc { text } => text.clone(),
            other => panic!("text slot resolved to {:?}", other.kind()),
        }
    }

    fn p_demos<'a>(&'a self, cx: &'a Cx<'_>, id: ParamId) -> &'a [DemoRow] {
        match self.p_value(cx, id) {
            ParamValue::Demos { rows } => rows,
            other => panic!("demos slot resolved to {:?}", other.kind()),
        }
    }

    /// The ToolSet gene's effective selection. Membership in `declared` was
    /// enforced when the value entered ([`Overlay::set`] / load-time
    /// validation); the filter here only keeps downstream arena indexing
    /// total against hostile ids.
    fn p_tool_set(&self, cx: &Cx<'_>, id: ParamId, declared: &[ToolId]) -> Box<[ToolId]> {
        match self.p_value(cx, id) {
            ParamValue::ToolSet { tools } => tools
                .iter()
                .copied()
                .filter(|tool| declared.contains(tool))
                .collect(),
            other => panic!("tool_set slot resolved to {:?}", other.kind()),
        }
    }

    fn p_context(&self, cx: &Cx<'_>, id: ParamId) -> ContextPolicy {
        match self.p_value(cx, id) {
            ParamValue::ContextPolicy { policy } => policy.clone(),
            other => panic!("context slot resolved to {:?}", other.kind()),
        }
    }

    fn p_model(&self, at: &str, cx: &Cx<'_>, id: ParamId) -> Result<Arc<LM>, RunError> {
        let model = match self.p_value(cx, id) {
            ParamValue::ModelRef { model } => *model,
            other => {
                return Err(RunError::Internal {
                    at: at.into(),
                    message: format!("model slot resolved to {:?}", other.kind()),
                });
            }
        };
        self.models[model]
            .as_ref()
            .cloned()
            .ok_or_else(|| RunError::Internal {
                at: at.into(),
                message: format!("model {model} not bound (load should have refused)"),
            })
    }

    async fn ensure_registered(
        &self,
        at: &str,
        sandbox: &dyn dsrs_tools::Executor,
        code: ParamId,
        (source, hash): (&str, u64),
        sig: SigId,
        caps: &CapSet,
    ) -> Result<String, RunError> {
        self.registered
            .get_or_register((code, hash), async {
                let base = match self.program.params[code].owner {
                    ParamOwner::Node(_) => at,
                    ParamOwner::Tool(id) => self.program.syms.get(self.program.tools[id].name),
                };
                register_code(
                    sandbox,
                    base,
                    source,
                    hash,
                    || self.tool_input_schema(sig).clone(),
                    caps,
                )
                .await
            })
            .await
            .map_err(|err| RunError::Hole {
                at: at.into(),
                source: dsrs_tools::ExecError::Internal {
                    message: format!("overlay code failed to register: {err}"),
                },
            })
    }

    fn resolve_bindings(
        &self,
        at: &str,
        node: Option<NodeId>,
        binds: &[Binding],
        cx: &Cx<'_>,
    ) -> Result<JsonMap, RunError> {
        let mut map = JsonMap::new();
        for b in binds {
            let name = self.program.syms.get(b.dst).to_string();
            map.insert(name, self.resolve_port(at, &b.src, cx)?);
        }
        // Refine feedback injection targets one leaf's input field.
        if let (Some(node), Some((target, field, value))) = (node, &cx.refine_feedback)
            && *target == node
        {
            map.insert(field.clone(), value.clone());
        }
        Ok(map)
    }

    fn resolve_exports(&self, binds: &[Binding], cx: &Cx<'_>) -> Result<JsonMap, RunError> {
        let mut map = JsonMap::new();
        for b in binds {
            let name = self.program.syms.get(b.dst).to_string();
            map.insert(name, self.resolve_port("out", &b.src, cx)?);
        }
        Ok(map)
    }

    fn resolve_port(&self, at: &str, port: &PortRef, cx: &Cx<'_>) -> Result<Value, RunError> {
        match port {
            PortRef::Lit(value) => Ok(value.clone()),
            // `^field` and `$.field` share the merged scope frame at runtime:
            // loop iteration frames are (enclosing inputs ⊕ carry), and the
            // v1 shadow rule makes the distinction purely one of intent.
            PortRef::Input(sym) | PortRef::Carried(sym) => {
                let name = self.program.syms.get(*sym);
                Ok(cx
                    .input()
                    .and_then(|frame| frame.get(name))
                    .cloned()
                    .unwrap_or(Value::Null))
            }
            PortRef::Out { node, field } => {
                let name = self.program.syms.get(*field);
                let frame = cx.frame(*node).ok_or_else(|| RunError::Internal {
                    at: at.into(),
                    message: format!(
                        "port references {} before it produced output",
                        self.program.node_display(*node)
                    ),
                })?;
                Ok(frame.get(name).cloned().unwrap_or(Value::Null))
            }
        }
    }
}

/// Mutable per-invocation loop state: the span's event stream, accumulated
/// usage, and the tool call/execution record surfaced through
/// [`LeafOutcome`].
#[derive(Default)]
struct AgentRun {
    call_ids: Vec<String>,
    events: Vec<SpanEvent>,
    usage: LmUsage,
    tool_calls: Vec<rig::message::ToolCall>,
    tool_executions: Vec<String>,
}

impl AgentRun {
    fn record_tool_result(
        &mut self,
        call: &rig::message::ToolCall,
        result: String,
        duration_us: u64,
        error: Option<String>,
    ) -> crate::ContentBlock {
        self.events.push(SpanEvent::ToolRun {
            id: call.id.clone(),
            name: call.function.name.clone(),
            args: call.function.arguments.clone(),
            result: result.clone(),
            duration_us,
            error,
        });
        self.tool_calls.push(call.clone());
        self.tool_executions.push(result.clone());
        crate::ContentBlock::tool_result_text(&call.id, call.call_id.as_deref(), result)
    }

    fn push_prompt(&mut self, chat: &mut Chat, prompt: impl Into<String>) {
        let message = Message::user(prompt);
        self.events.push(SpanEvent::ChatMessage {
            message: message.clone(),
        });
        chat.push_message(message);
    }

    fn finish_on_setup_error(
        &mut self,
        guard: &mut Option<SpanGuard>,
        surface: Result<AgentSurface, RunError>,
    ) -> Result<AgentSurface, RunError> {
        surface.map_err(|err| {
            std::mem::take(self).finish_error(guard.take(), crate::trace::SpanErrorKind::Tool, err)
        })
    }

    fn finish(
        self,
        guard: Option<SpanGuard>,
        done: AgentCompletion,
        leaf: Option<(&str, u64)>,
    ) -> (JsonMap, Option<LeafOutcome>, Chat) {
        let AgentCompletion {
            output,
            raw,
            field_meta,
            chat,
        } = done;
        let leaf = leaf.map(|(at, model_config_hash)| LeafOutcome {
            name: at.to_string(),
            raw_response: raw.clone(),
            field_meta,
            call_ids: self.call_ids,
            usage: self.usage,
            model_config_hash,
            span_id: guard.as_ref().map(SpanGuard::id),
            tool_calls: self.tool_calls,
            tool_executions: self.tool_executions,
        });
        if let Some(guard) = guard {
            guard.finish(SpanOutcome {
                events: self.events,
                raw_output: Some(raw),
                output: Some(output.clone()),
                usage: self.usage,
                error: None,
            });
        }
        (output, leaf, chat)
    }

    fn finish_loop_error(self, guard: Option<SpanGuard>, error: RunError) -> RunError {
        use crate::trace::SpanErrorKind;
        let kind = match &error {
            RunError::Parse { .. } => SpanErrorKind::Parse,
            RunError::Tool { .. } => SpanErrorKind::Tool,
            _ => SpanErrorKind::Lm,
        };
        self.finish_error(guard, kind, error)
    }

    fn finish_error(
        self,
        guard: Option<SpanGuard>,
        kind: crate::trace::SpanErrorKind,
        error: RunError,
    ) -> RunError {
        if let Some(guard) = guard {
            guard.finish(span_error(
                kind,
                error.to_string(),
                self.events,
                None,
                self.usage,
            ));
        }
        error
    }
}

/// How one `agent_loop` invocation ended (short of an error): the accepted
/// final output, or — caller-managed conversations only — a suspension on
/// pending tool calls.
enum LoopOutcome {
    Done(AgentCompletion),
    Suspend {
        calls: Vec<rig::message::ToolCall>,
        chat: Chat,
        next_turn: u32,
    },
}

struct AgentCompletion {
    output: JsonMap,
    raw: String,
    field_meta: IndexMap<String, FieldMeta>,
    chat: Chat,
}

/// One agent loop's resolved tool surface (see
/// [`Interpreter::build_agent_surface`]).
struct AgentSurface {
    toolset: ToolSet,
    stop_names: Vec<String>,
    policy: ContextPolicy,
    dispatch: Option<AgentDispatch>,
}

#[derive(Default)]
struct AgentDispatch {
    by_name: HashMap<String, ToolId>,
    sandbox_code: HashMap<ToolId, (String, u64)>,
    code_mode: Option<CodeModeSurface>,
}

struct PreparedAgent<'a> {
    node: &'a AgentLoopNode,
    tools: Box<[ToolId]>,
    policy: ContextPolicy,
    execution: AgentExecution,
}

enum AgentExecution {
    CallerManaged,
    Dispatch(Option<dsrs_tools::SandboxConfig>),
}

impl AgentExecution {
    fn code_mode(&self) -> Option<dsrs_tools::SandboxConfig> {
        match self {
            Self::CallerManaged => None,
            Self::Dispatch(config) => *config,
        }
    }
}

struct TurnRequest<'a> {
    at: &'a str,
    prefix: &'a [Message],
    suffix: &'a [Message],
    input: Option<&'a JsonMap>,
    model: &'a LMConfig,
}

impl TurnRequest<'_> {
    fn open_span(&self, request_hash: impl FnOnce() -> Option<u64>) -> Option<SpanGuard> {
        begin_span_lazy(|| SpanRequest {
            component: self.at,
            prefix: (!self.prefix.is_empty()).then_some(self.prefix),
            suffix: self.suffix,
            input: self.input.cloned(),
            model: self.model,
            request_hash: request_hash(),
        })
    }

    fn prompt_hash(&self) -> u64 {
        let model_hash = crate::trace::ModelEntry::from_config(self.model).config_hash;
        crate::trace::span::request_hash(model_hash, self.prefix, self.suffix)
    }
}

struct PredictExchange<'a> {
    at: &'a str,
    sig: SigId,
    format: OutputFormat,
    lm: &'a LM,
    meter: &'a BudgetMeter,
    guard: Option<SpanGuard>,
}

/// Resolved conversation state retained across caller-managed tool rounds.
struct AgentTurn {
    program: Arc<Program>,
    node: NodeId,
    at: String,
    lm: Arc<LM>,
    sampling: crate::SamplingOptions,
    model_config_hash: u64,
    surface: AgentSurface,
    meter: Arc<BudgetMeter>,
    run_meter: Arc<BudgetMeter>,
    prefix_len: usize,
    guard: Option<SpanGuard>,
    run: AgentRun,
}

/// Borrowed context for one agent-loop invocation — keeps `agent_loop` and
/// its helpers at a sane arity.
struct AgentLoopCx<'a> {
    at: &'a str,
    n: &'a AgentLoopNode,
    def: &'a SignatureDef,
    lm: &'a Arc<LM>,
    surface: &'a AgentSurface,
    prefix_len: usize,
    meter: &'a Arc<BudgetMeter>,
    run_meter: &'a Arc<BudgetMeter>,
}

/// One agent loop's Code Mode surface: the wrapped tool capabilities and the
/// sandbox config `run_js` scripts execute under.
struct CodeModeSurface {
    capabilities: Vec<dsrs_tools::Capability>,
    config: dsrs_tools::SandboxConfig,
}

/// Executes one `run_js` call against the loop's Code Mode surface. Failures
/// are conversational, like every agent tool failure.
async fn execute_code_mode_script(
    surface: &CodeModeSurface,
    args: &Value,
) -> Result<String, String> {
    let Some(code) = args.get("code").and_then(Value::as_str) else {
        return Err(format!(
            "{} requires a string `code` argument",
            dsrs_tools::RUN_JS_TOOL_NAME
        ));
    };
    dsrs_tools::run_script(code, surface.capabilities.clone(), surface.config)
        .await
        .map(|value| value.to_string())
        .map_err(|err| err.to_llm_json())
}

async fn lm_call_toolset(
    lm: &Arc<LM>,
    chat: Chat,
    toolset: &ToolSet,
    at: &str,
    options: &LMCallOptions,
) -> Result<crate::core::lm::LMResponse, RunError> {
    lm.call_with_toolset_and_options_ref(chat, toolset, ToolLoopMode::CallerManaged, options)
        .await
        .map_err(|err| RunError::Lm {
            at: at.into(),
            source: LmError::Provider {
                provider: lm.config.model.clone(),
                message: err.to_string(),
                source: Some(err.reallocate_into_boxed_dyn_error_without_backtrace()),
            },
        })
}

// ---------------------------------------------------------------------------
// Rendering and coercion helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputFormat {
    Delimited,
    JsonSchema,
    JsonScalar,
    JsonMode,
}

struct CachedDemos {
    format: OutputFormat,
    overlay: Option<Weak<Overlay>>,
    messages: Arc<[Message]>,
}

impl CachedDemos {
    fn matches(&self, format: OutputFormat, overlay: Option<&Arc<Overlay>>) -> bool {
        self.format == format
            && match (&self.overlay, overlay) {
                (None, None) => true,
                // Holding a Weak prevents address reuse, including after Arc::make_mut.
                (Some(previous), Some(current)) => {
                    std::ptr::eq(previous.as_ptr(), Arc::as_ptr(current))
                }
                _ => false,
            }
    }
}

struct PreparedSchemas {
    tool_calls: bool,
    scalar: bool,
    compatible: OnceLock<bool>,
    object_options: OnceLock<Box<LMCallOptions>>,
    scalar_options: OnceLock<Box<LMCallOptions>>,
    tool_input: OnceLock<Box<Value>>,
}

impl PreparedSchemas {
    fn new(def: &SignatureDef) -> Self {
        Self {
            tool_calls: def
                .outputs
                .iter()
                .any(|field| field.special_kind() == Some(crate::SpecialField::ToolCalls)),
            scalar: matches!(&*def.outputs, [field]
                if matches!(field.ty, FieldType::String | FieldType::Literal(_))),
            compatible: OnceLock::new(),
            object_options: OnceLock::new(),
            scalar_options: OnceLock::new(),
            tool_input: OnceLock::new(),
        }
    }

    fn tool_input(&self, def: &SignatureDef, types: &TypeTable) -> &Value {
        self.tool_input
            .get_or_init(|| Box::new(input_schema_of(def, types)))
    }
}

fn render_demos(
    def: &SignatureDef,
    types: &TypeTable,
    demos: &[DemoRow],
    format: OutputFormat,
    templates: Option<&PreparedInputTemplates>,
) -> anyhow::Result<Vec<Message>> {
    let adapter = ChatAdapter;
    let mut prefix = Vec::with_capacity(demos.len() * 2);
    for row in demos {
        prefix.push(Message::user(render_input(
            def, &row.input, format, templates,
        )?));
        prefix.push(Message::assistant(if format == OutputFormat::JsonScalar {
            let field = &def.outputs[0];
            row.output
                .get(field.name.as_ref())
                .ok_or_else(|| anyhow::anyhow!("demo missing output field `{}`", field.name))?
                .to_string()
        } else if format == OutputFormat::JsonSchema {
            adapter.format_output_json_schema_def(def, types, &row.output)
        } else if format != OutputFormat::Delimited {
            adapter.format_output_json_def(def, types, &row.output)
        } else {
            adapter.format_output_def(def, &row.output)
        }));
    }
    Ok(prefix)
}

fn render_input(
    def: &SignatureDef,
    input: &JsonMap,
    format: OutputFormat,
    templates: Option<&PreparedInputTemplates>,
) -> anyhow::Result<String> {
    match format {
        OutputFormat::Delimited => ChatAdapter.try_format_input_prepared(def, input, templates),
        OutputFormat::JsonScalar => ChatAdapter.try_format_input_scalar_def(def, input, templates),
        _ => ChatAdapter.try_format_input_json_prepared(def, input, templates),
    }
}

fn parse_response(
    def: &SignatureDef,
    types: &TypeTable,
    response: &Message,
    format: OutputFormat,
) -> Result<(JsonMap, IndexMap<String, FieldMeta>), crate::core::ParseError> {
    match format {
        OutputFormat::Delimited => ChatAdapter.parse_output_def(def, types, response),
        OutputFormat::JsonScalar => ChatAdapter.parse_output_scalar_def(def, types, response),
        _ => ChatAdapter.parse_output_json_def(def, types, response),
    }
}

fn span_error(
    kind: crate::trace::SpanErrorKind,
    message: String,
    events: Vec<SpanEvent>,
    raw_output: Option<String>,
    usage: LmUsage,
) -> SpanOutcome {
    SpanOutcome {
        events,
        raw_output,
        output: None,
        usage,
        error: Some(crate::trace::SpanError { kind, message }),
    }
}

/// Coerces an already-JSON value (hole result, stop-tool args) into the
/// signature's output map. Structural, with the standard widenings.
fn coerce_outputs(
    at: &str,
    def: &SignatureDef,
    types: &TypeTable,
    value: Value,
    raw: &str,
) -> Result<JsonMap, RunError> {
    let Value::Object(mut object) = value else {
        return Err(RunError::Parse {
            at: at.into(),
            raw: raw.to_string(),
            source: None,
            usage: LmUsage::default(),
        });
    };
    let mut output = JsonMap::new();
    for field in def.outputs.iter() {
        match object.remove(&*field.name) {
            Some(v)
                if json_matches_type(&v, &field.ty, types)
                    && crate::typesys::coerce::validate_json_constraints(&v, &field.ty, types)
                        .is_ok() =>
            {
                output.insert(field.name.to_string(), v);
            }
            None if field.ty.is_optional() => {
                output.insert(field.name.to_string(), Value::Null);
            }
            _ => {
                return Err(RunError::Parse {
                    at: at.into(),
                    raw: raw.to_string(),
                    source: None,
                    usage: LmUsage::default(),
                });
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod sandbox_registration_tests {
    use super::*;
    use cranelift_entity::EntityRef;

    #[tokio::test]
    async fn waiters_share_a_failure_and_later_calls_can_retry() {
        let registry = SandboxRegistrations::default();
        let key = (ParamId::new(0), 42);
        let (finish, wait) = tokio::sync::oneshot::channel();
        let mut leader = Box::pin(registry.get_or_register(key, async {
            wait.await.unwrap();
            Err(dsrs_tools::RegisterError::Compile {
                message: "fixture failure".into(),
            })
        }));
        assert!(futures::poll!(&mut leader).is_pending());
        let mut follower = Box::pin(registry.get_or_register(key, async {
            panic!("the existing registration should report the shared failure")
        }));
        assert!(futures::poll!(&mut follower).is_pending());
        finish.send(()).unwrap();
        let (first, second) = tokio::join!(leader, follower);
        for result in [first, second] {
            assert!(matches!(
                result,
                Err(dsrs_tools::RegisterError::Compile { message }) if message == "fixture failure"
            ));
        }
        assert!(registry.0.lock().unwrap().is_empty());
        assert_eq!(
            registry
                .get_or_register(key, async { Ok("retry".into()) })
                .await
                .unwrap(),
            "retry"
        );
    }

    #[tokio::test]
    async fn cancellation_releases_registration_and_other_slots_progress() {
        let registry = SandboxRegistrations::default();
        let key = (ParamId::new(0), 42);
        let other_key = (ParamId::new(1), 42);
        let mut leader = Box::pin(registry.get_or_register(key, std::future::pending()));
        let mut waiter = Box::pin(registry.get_or_register(key, async { Ok("successor".into()) }));
        let mut other = Box::pin(registry.get_or_register(other_key, async { Ok("other".into()) }));
        assert!(futures::poll!(&mut leader).is_pending());
        assert!(futures::poll!(&mut waiter).is_pending());
        assert!(matches!(
            futures::poll!(&mut other),
            std::task::Poll::Ready(Ok(name)) if name == "other"
        ));
        drop(leader);
        assert_eq!(waiter.await.unwrap(), "successor");

        let abandoned_key = (key.0, 43);
        let mut abandoned =
            Box::pin(registry.get_or_register(abandoned_key, std::future::pending()));
        assert!(futures::poll!(&mut abandoned).is_pending());
        drop(abandoned);
        assert!(!registry.0.lock().unwrap().contains_key(&abandoned_key));
    }
}

#[cfg(test)]
mod output_constraint_tests {
    use super::*;
    use crate::typesys::{ClassDef, Constraint, ConstraintKind, FieldDef as TypeFieldDef};

    #[test]
    fn structured_stop_output_enforces_nested_assert() {
        let mut types = TypeTable::default();
        types.classes.insert(
            "Score".into(),
            ClassDef {
                internal_name: "Score".into(),
                rendered_name: "Score".into(),
                docs: None,
                fields: vec![TypeFieldDef {
                    name: "value".into(),
                    rendered_name: "value".into(),
                    field_type: crate::typesys::FieldType::Int,
                    docs: None,
                    constraints: vec![Constraint {
                        level: ConstraintKind::Assert,
                        label: Some("positive".into()),
                        expression: "this > 0".into(),
                    }],
                }],
                constraints: Vec::new(),
            },
        );
        let def = SignatureDef::build("Result")
            .input("question", crate::typesys::FieldType::String)
            .output("score", crate::typesys::FieldType::Class("Score".into()))
            .finish()
            .unwrap();
        let invalid = serde_json::json!({"score":{"value":0}});
        let raw = invalid.to_string();
        assert!(matches!(
            coerce_outputs("stop", &def, &types, invalid, &raw),
            Err(RunError::Parse { .. })
        ));
        let valid = serde_json::json!({"score":{"value":1}});
        let raw = valid.to_string();
        assert!(coerce_outputs("stop", &def, &types, valid, &raw).is_ok());
    }
}

#[cfg(test)]
mod budget_meter_tests {
    use super::*;

    #[test]
    fn oversized_usage_keeps_the_parent_token_budget_exhausted() {
        let parent = Arc::new(BudgetMeter::new(Budget {
            max_tokens: Some(u64::MAX),
            ..Budget::unlimited()
        }));
        let child = BudgetMeter::child(&parent, Budget::unlimited());
        let first = LmUsage {
            prompt_tokens: u64::MAX,
            completion_tokens: u64::MAX,
            total_tokens: u64::MAX - 1,
        };
        let second = LmUsage {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
        };
        child.record_usage(&first);
        child.record_usage(&second);
        assert_eq!(child.tokens(), u64::MAX);
        assert_eq!(parent.tokens(), u64::MAX);
        assert_eq!(child.try_reserve_call(), Err(Exhausted));
        assert_eq!(child.lm_calls(), 0);

        let combined = first + second;
        assert_eq!(combined.total_tokens, u64::MAX);
        assert_eq!(combined.prompt_tokens, u64::MAX);
        assert_eq!(combined.completion_tokens, u64::MAX);
    }

    #[test]
    fn maximum_call_cap_rejects_an_exhausted_reservation() {
        let meter = BudgetMeter::new(Budget {
            max_lm_calls: Some(u32::MAX),
            ..Budget::unlimited()
        });
        meter.lm_calls.store(u32::MAX, Ordering::Relaxed);
        assert_eq!(meter.try_reserve_call(), Err(Exhausted));
        assert_eq!(meter.lm_calls(), u32::MAX);
    }
}

fn node_budget(budget: &crate::ir::graph::NodeBudget) -> Budget {
    Budget {
        max_lm_calls: budget.max_lm_calls,
        max_tokens: budget.max_tokens,
        deadline: budget
            .deadline_ms
            .map(|ms| Instant::now() + std::time::Duration::from_millis(ms)),
    }
}

/// `ContextPolicy.max_history_turns`: keep the rendered prefix plus the
/// trailing N conversation messages.
fn prune_history(chat: &mut Chat, prefix_len: usize, policy: &ContextPolicy) -> Option<SpanEvent> {
    let max_turns = policy.max_history_turns? as usize;
    let count = chat
        .len()
        .saturating_sub(prefix_len)
        .saturating_sub(max_turns);
    if count == 0 {
        return None;
    }
    chat.messages.drain(prefix_len..prefix_len + count);
    Some(SpanEvent::HistoryPruned {
        start: prefix_len,
        count,
    })
}

/// Applies `ContextPolicy.tool_result_max_bytes` to one tool result — the
/// same clip whether the tool was dispatched or the result came back through
/// [`Interpreter::resume_conversation`].
fn clip_tool_result(mut text: String, policy: &ContextPolicy) -> String {
    if let Some(max) = policy.tool_result_max_bytes {
        let max = max as usize;
        if text.len() > max {
            let mut cut = max;
            while cut > 0 && !text.is_char_boundary(cut) {
                cut -= 1;
            }
            text.truncate(cut);
            text.push_str("… [truncated]");
        }
    }
    text
}

/// A hole's implementation, resolved through the overlay (sandboxed code may
/// be a candidate's Code gene; host holes are fixed by their content hash).
enum ResolvedHoleImpl<'a> {
    Sandboxed {
        code: ParamId,
        source: &'a str,
        hash: u64,
    },
    Host {
        hash: u64,
    },
}

/// The hole replay preimage (RFC 0003 §4.4): impl discriminant ++ impl/code
/// hash ++ canonical input (sorted keys) ++ sorted caps. Holes render no
/// prompt, so the config+prompt hash every other leaf uses would be identical
/// for every hole in every program — this preimage is what makes hole spans
/// individually addressable by replay and divergence detection.
fn hole_request_hash(imp: &ResolvedHoleImpl<'_>, input: &JsonMap, caps: &CapSet) -> u64 {
    use std::hash::Hasher as _;
    let mut hasher = crate::utils::hash::StableHasher::new();
    let (discriminant, hash) = match imp {
        ResolvedHoleImpl::Sandboxed { hash, .. } => (0u8, *hash),
        ResolvedHoleImpl::Host { hash } => (1u8, *hash),
    };
    hasher.write_u8(discriminant);
    hasher.write_u64(hash);
    let mut fields: Vec<(&String, &Value)> = input.iter().collect();
    fields.sort_by_key(|(key, _)| *key);
    for (key, value) in fields {
        hasher.write(key.as_bytes());
        hasher.write(value.to_string().as_bytes());
    }
    for cap in caps.iter() {
        hasher.write(cap.as_bytes());
    }
    hasher.finish()
}

/// Registers a code gene in the sandbox under a content-addressed name
/// (`<base>-<hash hex>`), deduplicating against an already-registered tool.
async fn register_code(
    sandbox: &dyn dsrs_tools::Executor,
    base: &str,
    source: &str,
    hash: u64,
    params: impl FnOnce() -> Value,
    caps: &CapSet,
) -> Result<String, dsrs_tools::RegisterError> {
    let sanitized: String = base
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(40)
        .collect();
    let name = format!("{sanitized}-{hash:016x}");
    if sandbox.tool(&name).is_some() {
        // Content-addressed: same name ⇒ same source hash ⇒ already usable.
        return Ok(name);
    }
    let tool_source = dsrs_tools::ToolSource::new(
        name.clone(),
        format!("dsrs code gene `{base}`"),
        params(),
        source,
    );
    let allowed: Vec<String> = caps.iter().map(str::to_string).collect();
    match sandbox.register_scoped(tool_source, &allowed).await {
        Ok(_) => Ok(name),
        Err(dsrs_tools::RegisterError::Duplicate { name: registered })
            if registered == name && sandbox.tool(&name).is_some() =>
        {
            Ok(name)
        }
        Err(error) => Err(error),
    }
}

// ---------------------------------------------------------------------------
// Run state
// ---------------------------------------------------------------------------

type Frames = SecondaryMap<NodeId, Option<JsonMap>>;

/// Call-scoped run state; never stored on the interpreter. Concurrent runs
/// (with different overlays) share one `Arc<Program>`.
struct Cx<'a> {
    overlay: Option<Arc<Overlay>>,
    meter: Arc<BudgetMeter>,
    /// Completed node outputs — `PortRef::Out` resolution.
    frames: Frames,
    /// Scope-input stack (`$.field`): program input at root, merged carry
    /// frames inside loops.
    inputs: Vec<JsonMap>,
    /// A fork's parent state stays immutable until every branch returns.
    parent: Option<&'a Cx<'a>>,
    /// Retry corrective turn, consumed by the next leaf evaluation.
    feedback: Option<String>,
    /// Refine feedback injection: (child leaf, input field, value).
    refine_feedback: Option<(NodeId, String, Value)>,
    /// `Some` when the caller asked for per-leaf metadata
    /// ([`Interpreter::run_collecting`]); successful `Predict` leaves push
    /// here in execution order. `None` = plain `run`, zero collection cost.
    leaves: Option<Vec<LeafOutcome>>,
    prepare_replay: Option<PrepareReplayOutput<'a>>,
}

impl Cx<'_> {
    fn frame(&self, node: NodeId) -> Option<&JsonMap> {
        self.frames[node]
            .as_ref()
            .or_else(|| self.parent.and_then(|parent| parent.frame(node)))
    }

    fn input(&self) -> Option<&JsonMap> {
        self.inputs
            .last()
            .or_else(|| self.parent.and_then(Cx::input))
    }

    /// Shares the parent read-only state until join. Branch writes, feedback,
    /// and leaf collection belong to the new context.
    fn branch(&self) -> Cx<'_> {
        Cx {
            overlay: self.overlay.clone(),
            meter: Arc::clone(&self.meter),
            frames: SecondaryMap::new(),
            inputs: Vec::new(),
            parent: Some(self),
            feedback: None,
            refine_feedback: None,
            leaves: self.leaves.as_ref().map(|_| Vec::new()),
            prepare_replay: self.prepare_replay,
        }
    }
}
