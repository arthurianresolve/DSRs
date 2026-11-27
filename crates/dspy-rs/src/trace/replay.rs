//! Replay scope: serve `Predict` calls from a recorded [`Trace`] (RFC 0001 §4d/§4e).
//!
//! A trace is a complete set of canned LM responses. [`replay`] opens a
//! task-local scope symmetric to [`capture`](crate::trace::capture); every
//! `Predict` call inside it is intercepted **above** the LM: the call's
//! `request_hash` (same preimage the spans record — redacted model config ++
//! full rendered prompt) is compared against the next recorded span for that
//! component, and on a match the recorded parsed output is served without
//! constructing a client, calling a provider, or executing a tool — tool
//! effects are already baked into the recorded events.
//!
//! Two modes ([`ReplayMode`]):
//!
//! - **`Strict`** — every call must match its recording; any mismatch is a
//!   typed [`ReplayError`] surfaced as
//!   [`PredictError::Replay`](crate::PredictError::Replay). Temporal's
//!   determinism check; what fixtures and CI use:
//!
//!   ```ignore
//!   let (out, trace) = capture(|| pipeline(input.clone())).await;   // once, live
//!   std::fs::write("fixtures/qa.trace.jsonl", trace.to_jsonl()?)?;
//!
//!   // in tests, forever, zero API calls:
//!   let trace = Trace::from_jsonl(&std::fs::read_to_string("fixtures/qa.trace.jsonl")?)?;
//!   let (out, report) = trace::replay(&trace, ReplayMode::Strict, || pipeline(input)).await;
//!   assert_eq!(report.live, 0);
//!   ```
//!
//! - **`UntilDivergence`** — serve while hashes match; the first mismatching
//!   call *and every call after it* go to the live LM. Divergence is
//!   *detected*, never declared: a mutated parameter changes the rendered
//!   prompt (and therefore the hash) of exactly the calls it affects, so the
//!   unchanged prefix of a pipeline replays free and only the counterfactual
//!   suffix spends tokens (§4e):
//!
//!   ```ignore
//!   let mut params = fx::Params::new();
//!   params.set_instruction("refiner", mutated_instruction);
//!   let ((result, new_trace), report) = trace::replay(&base_trace, ReplayMode::UntilDivergence, || {
//!       capture(|| fx::with_params(params, pipeline(input)))
//!   }).await;
//!   // spans before the first "refiner" call: served free from base_trace
//!   // "refiner" and everything after: live LM calls only
//!   ```
//!
//! Once diverged, the session stays live even if a later call's hash happens
//! to match the recording again — downstream state may differ in ways the
//! prompt does not capture. Limitation (RFC §4e): served spans never re-run
//! tools, but post-divergence live calls do execute tools against the real
//! world; counterfactual replay of side-effectful harnesses needs a
//! resettable environment.
//!
//! Like `capture`, the scope is task-local: spawned subtasks do not inherit
//! it, and nested scopes are exclusive (innermost wins).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::task_local;

use crate::trace::TRACE_FORMAT_VERSION;
use crate::trace::span::{CompId, ModelEntry, Span, SpanId, Trace, request_hash};
use crate::{LMConfig, Message, Role};

task_local! {
    static ACTIVE: ReplaySession;
}

/// How a replay scope treats a call that does not match its recording.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayMode {
    /// Every call must match: the next recorded span for `(component, seq)`
    /// must exist, be complete, and have an equal `request_hash`. Any mismatch
    /// is a typed error. For fixtures and CI.
    Strict,
    /// Serve recorded spans while hashes match; on the first mismatch, that
    /// call and ALL subsequent calls go to the live LM. For counterfactual
    /// replay.
    UntilDivergence,
}

/// Why a call could not be served from the recording.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ReplayError {
    /// The live request's hash differs from the recorded span's — the prompt
    /// or model config changed relative to the recording.
    #[error(
        "replay divergence at `{component}` seq {seq}: recorded span {expected_span:?} \
         has request_hash {expected_hash:#018x}, live call hashed {got_hash:#018x}"
    )]
    Divergence {
        component: String,
        seq: u32,
        /// The recorded span the live call was checked against.
        expected_span: SpanId,
        expected_hash: u64,
        got_hash: u64,
    },

    /// The recorded span is unusable due to missing output, an unsupported
    /// trace version, invalid chat edits, truncation, or redaction.
    #[error(
        "recorded span {span:?} for `{component}` seq {seq} is incomplete or incompatible and cannot be replayed"
    )]
    Incomplete {
        component: String,
        seq: u32,
        span: SpanId,
    },

    /// The trace has no recorded span for `(component, seq)` — the live run
    /// makes more calls than the recording (or calls a component the
    /// recording never saw).
    #[error("no recorded span for `{component}` seq {seq}: the trace is exhausted")]
    Exhausted { component: String, seq: u32 },

    /// The recorded span matched but its stored output does not deserialize
    /// into the signature's output type (schema drift since recording).
    #[error(
        "recorded output of span {span:?} for `{component}` seq {seq} \
         does not fit the signature output type: {message}"
    )]
    OutputDecode {
        component: String,
        seq: u32,
        span: SpanId,
        message: String,
    },
}

/// What a replay scope did, returned by [`replay`] alongside the closure's
/// result.
#[derive(Clone, Debug, Default)]
pub struct ReplayReport {
    /// Calls served from the recording (zero provider calls).
    pub served: usize,
    /// Calls that went to the live LM (always 0 in `Strict` mode).
    pub live: usize,
    /// The recorded span at which divergence was first detected, when the
    /// mismatch could be attributed to one (hash mismatch / incomplete span).
    pub diverged_at: Option<SpanId>,
    /// The first mismatch, verbatim — the error `Strict` mode surfaced, or
    /// the reason `UntilDivergence` switched live.
    pub divergence: Option<ReplayError>,
}

/// Shared handle to an in-progress replay session.
#[derive(Clone)]
pub(crate) struct ReplaySession(Arc<Mutex<ReplayInner>>);

struct ReplayInner {
    trace: Arc<Trace>,
    span_index: Option<HashMap<(CompId, u32), usize>>,
    mode: ReplayMode,
    /// Per-component next-seq cursors in the recording's component table.
    cursors: HashMap<CompId, u32>,
    /// `UntilDivergence` only: once true, every call goes live.
    diverged: bool,
    report: ReplayReport,
}

/// What `Predict` should do with an intercepted call.
pub(crate) enum ReplayDirective {
    /// Serve this recorded span verbatim — no client, no provider, no tools.
    Serve(RecordedSpan),
    /// Call the live LM (post-divergence `UntilDivergence`).
    Live,
    /// Refuse the call (`Strict`): surface the typed error.
    Refuse(ReplayError),
}

pub(crate) struct RecordedSpan {
    trace: Arc<Trace>,
    position: usize,
}

impl RecordedSpan {
    pub(crate) fn span(&self) -> &Span {
        &self.trace.spans[self.position]
    }
}

impl ReplaySession {
    fn new(trace: Arc<Trace>, mode: ReplayMode) -> Self {
        // Small fixtures are cheaper to scan than to index.
        const INDEX_MIN_SPANS: usize = 1024;
        let span_index = (trace.spans.len() >= INDEX_MIN_SPANS).then(|| {
            let mut index = HashMap::with_capacity(trace.spans.len());
            for (position, span) in trace.spans.iter().enumerate() {
                index.entry((span.component, span.seq)).or_insert(position);
            }
            index
        });
        Self(Arc::new(Mutex::new(ReplayInner {
            trace,
            span_index,
            mode,
            cursors: HashMap::new(),
            diverged: false,
            report: ReplayReport::default(),
        })))
    }

    fn finish(self) -> ReplayReport {
        match Arc::try_unwrap(self.0) {
            Ok(mutex) => mutex.into_inner().unwrap().report,
            // Fallback: clone if an orphaned task still holds a reference.
            Err(arc) => arc.lock().unwrap().report.clone(),
        }
    }

    /// Decides the fate of one `Predict` call: serve, go live, or refuse.
    fn decide(&self, component: &str, config: &LMConfig, chat: &[Message]) -> ReplayDirective {
        // The same preimage `TraceSink::close` hashes: redacted-config hash ++
        // full rendered prompt. The prefix/suffix split does not matter — the
        // hash streams prefix ++ suffix, and `chat` is exactly that.
        self.decide_request(component, Some(chat.len()), |_, recorded| {
            let current = request_hash(ModelEntry::from_config(config).config_hash, &[], chat);
            (current != recorded.request_hash).then_some(current)
        })
    }

    /// [`decide`] with a caller-computed `request_hash` — the seam for leaves
    /// whose identity is not prompt-shaped (holes: impl hash ++ canonical
    /// input ++ caps, RFC 0003 §4.4). The recorded span's `request_hash` was
    /// stamped from the same preimage at capture time.
    fn decide_hashed(&self, component: &str, hash: impl FnOnce() -> u64) -> ReplayDirective {
        self.decide_request(component, None, |_, recorded| {
            let current = hash();
            (current != recorded.request_hash).then_some(current)
        })
    }

    fn decide_request(
        &self,
        component: &str,
        chat_len: Option<usize>,
        mismatch: impl FnOnce(&Trace, &Span) -> Option<u64>,
    ) -> ReplayDirective {
        let mut inner = self.0.lock().unwrap();

        if inner.diverged {
            inner.report.live += 1;
            return ReplayDirective::Live;
        }

        let component_id = inner.trace.component_id(component);
        let seq = component_id
            .and_then(|id| inner.cursors.get(&id).copied())
            .unwrap_or(0);
        let recorded = component_id.and_then(|id| {
            let position = match &inner.span_index {
                Some(index) => index.get(&(id, seq)).copied(),
                None => inner
                    .trace
                    .spans
                    .iter()
                    .position(|span| span.component == id && span.seq == seq),
            }?;
            Some((id, position))
        });
        let (id, position) = match recorded {
            Some(recorded) => recorded,
            None => {
                return inner.mismatch(
                    None,
                    ReplayError::Exhausted {
                        component: component.to_string(),
                        seq,
                    },
                );
            }
        };
        let span = &inner.trace.spans[position];
        let span_id = span.id;
        let incomplete = || ReplayError::Incomplete {
            component: component.to_string(),
            seq,
            span: span_id,
        };

        if !span.complete || span.output.is_none() || inner.trace.meta.v > TRACE_FORMAT_VERSION {
            return inner.mismatch(Some(span_id), incomplete());
        }

        if let Some(got_hash) = mismatch(&inner.trace, span) {
            let expected_hash = span.request_hash;
            return inner.mismatch(
                Some(span_id),
                ReplayError::Divergence {
                    component: component.to_string(),
                    seq,
                    expected_span: span_id,
                    expected_hash,
                    got_hash,
                },
            );
        }

        if chat_len.is_some_and(|len| !span.chat_edits_valid(len)) {
            return inner.mismatch(Some(span_id), incomplete());
        }

        let span = RecordedSpan {
            trace: Arc::clone(&inner.trace),
            position,
        };
        inner.cursors.insert(id, seq + 1);
        inner.report.served += 1;
        ReplayDirective::Serve(span)
    }
}

impl ReplayInner {
    /// Routes a mismatch by mode: `Strict` refuses, `UntilDivergence` flips
    /// the session live. Only the first mismatch is recorded in the report.
    fn mismatch(&mut self, at: Option<SpanId>, error: ReplayError) -> ReplayDirective {
        if self.report.divergence.is_none() {
            self.report.diverged_at = at;
            self.report.divergence = Some(error.clone());
        }
        match self.mode {
            ReplayMode::Strict => ReplayDirective::Refuse(error),
            ReplayMode::UntilDivergence => {
                self.diverged = true;
                self.report.live += 1;
                ReplayDirective::Live
            }
        }
    }
}

/// Runs `f` with `trace` as the canned-response source for every `Predict`
/// call on this task. Returns the closure's result and a [`ReplayReport`]
/// of what was served versus live.
///
/// Task-local scoping mirrors [`capture`](crate::trace::capture): spawned
/// subtasks do not inherit the scope; nesting is exclusive (innermost wins).
/// Compose with `capture` (replay outside, capture inside) to record the
/// counterfactual rollout while serving its unchanged prefix from `trace`.
pub async fn replay<F, Fut, R>(trace: &Trace, mode: ReplayMode, f: F) -> (R, ReplayReport)
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = R>,
{
    replay_shared(Arc::new(trace.clone()), mode, f).await
}

/// Runs a replay scope over an already owned recording.
///
/// Pass clones of the same `Arc<Trace>` to reuse a recording without copying
/// its payload. Each call has its own cursors, mode, and report, with the same
/// task-local scope as [`replay`].
pub async fn replay_shared<F, Fut, R>(
    trace: Arc<Trace>,
    mode: ReplayMode,
    f: F,
) -> (R, ReplayReport)
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = R>,
{
    let session = ReplaySession::new(trace, mode);
    let result = ACTIVE.scope(session.clone(), f()).await;
    (result, session.finish())
}

/// Returns `true` if the current task is inside a [`replay`] scope.
pub fn is_replaying() -> bool {
    ACTIVE.try_with(|_| ()).is_ok()
}

/// Consults the active replay scope, if any, about one `Predict` call.
/// `None` means no scope is active — proceed live, unconditionally.
///
/// Called by `Predict::call_and_parse_with_input` with the exact `Chat` it
/// would send and the config of the LM it would send it to.
pub(crate) fn intercept(
    component: &str,
    config: &LMConfig,
    chat: &[Message],
) -> Option<ReplayDirective> {
    let session = ACTIVE.try_with(|session| session.clone()).ok()?;
    Some(session.decide(component, config, chat))
}

/// [`intercept`] with a caller-computed `request_hash` — used by leaves whose
/// replay identity is not prompt-shaped (interpreter holes, RFC 0003 §4.4).
pub(crate) fn intercept_hashed(
    component: &str,
    hash: impl FnOnce() -> u64,
) -> Option<ReplayDirective> {
    let session = ACTIVE.try_with(|session| session.clone()).ok()?;
    Some(session.decide_hashed(component, hash))
}

/// Earlier agent hashes can match when the recording protects the same
/// opening prefix. A supplied continuation protects its leading systems.
pub(crate) fn intercept_agent(
    component: &str,
    chat_len: usize,
    protected_prefix_len: Option<usize>,
    opening_hash: impl FnOnce() -> u64,
    legacy_agent_hash: impl FnOnce() -> u64,
    request_hash: impl FnOnce() -> u64,
) -> Option<ReplayDirective> {
    let session = ACTIVE.try_with(Clone::clone).ok()?;
    Some(
        session.decide_request(component, Some(chat_len), |trace, span| {
            let same_prefix = |len| {
                let recorded_len = match span.prefix {
                    Some(id) => trace
                        .prefixes
                        .get(id.0 as usize)
                        .map(|entry| entry.messages.len()),
                    None => Some(
                        span.suffix
                            .iter()
                            .take_while(|msg| msg.role == Role::System)
                            .count(),
                    ),
                };
                recorded_len == Some(len)
            };
            if opening_hash() == span.request_hash && protected_prefix_len.is_none_or(same_prefix) {
                return None;
            }
            let current = request_hash();
            if current == span.request_hash
                || (legacy_agent_hash() == span.request_hash
                    && protected_prefix_len.is_none_or(same_prefix))
            {
                None
            } else {
                Some(current)
            }
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::capture::{SpanRequest, begin_span, capture};

    #[tokio::test]
    async fn agent_replay_accepts_the_legacy_opening_hash() {
        let (_, trace) = capture(|| async {
            begin_span(SpanRequest {
                component: "agent",
                prefix: None,
                suffix: &[],
                input: None,
                model: &LMConfig::default(),
                request_hash: Some(1),
            })
            .unwrap()
            .finish(crate::trace::SpanOutcome {
                events: vec![],
                raw_output: None,
                output: Some(crate::trace::JsonMap::new()),
                usage: crate::LmUsage::default(),
                error: None,
            });
        })
        .await;
        let (result, report) = replay(&trace, ReplayMode::Strict, || async {
            intercept_agent(
                "agent",
                0,
                None,
                || 1,
                || panic!("legacy match needs only the opening hash"),
                || panic!("legacy match needs only the opening hash"),
            )
        })
        .await;
        assert!(matches!(result, Some(ReplayDirective::Serve(_))));
        assert_eq!(report.served, 1);
    }

    #[tokio::test]
    async fn legacy_agent_keys_also_require_the_same_protected_prefix() {
        let prefix = [
            Message::system("agent"),
            Message::user("demo"),
            Message::assistant("example"),
        ];
        let (_, trace) = capture(|| async {
            begin_span(SpanRequest {
                component: "agent",
                prefix: Some(&prefix),
                suffix: &[Message::user("question")],
                input: None,
                model: &LMConfig::default(),
                request_hash: Some(2),
            })
            .unwrap()
            .finish(crate::trace::SpanOutcome {
                events: Vec::new(),
                raw_output: None,
                output: Some(crate::trace::JsonMap::new()),
                usage: crate::LmUsage::default(),
                error: None,
            });
        })
        .await;
        let (different, report) = replay(&trace, ReplayMode::Strict, || async {
            intercept_agent("agent", 4, Some(1), || 1, || 2, || 3)
        })
        .await;
        assert!(matches!(
            different,
            Some(ReplayDirective::Refuse(ReplayError::Divergence { .. }))
        ));
        assert_eq!(report.served, 0);
        let (same, report) = replay(&trace, ReplayMode::Strict, || async {
            intercept_agent("agent", 4, Some(3), || 1, || 2, || 4)
        })
        .await;
        assert!(matches!(same, Some(ReplayDirective::Serve(_))));
        assert_eq!(report.served, 1);
    }

    #[tokio::test]
    async fn replay_skips_hashing_for_incomplete_exhausted_and_live_calls() {
        let (_, trace) = capture(|| async {
            drop(begin_span(SpanRequest {
                component: "predict",
                prefix: None,
                suffix: &[],
                input: None,
                model: &LMConfig::default(),
                request_hash: Some(1),
            }));
        })
        .await;
        let trace = Arc::new(trace);
        let unused_hash = || panic!("request hash is unnecessary");
        let strict = ReplaySession::new(Arc::clone(&trace), ReplayMode::Strict);
        assert!(matches!(
            strict.decide_hashed("predict", unused_hash),
            ReplayDirective::Refuse(ReplayError::Incomplete { .. })
        ));
        assert!(matches!(
            strict.decide_hashed("missing", unused_hash),
            ReplayDirective::Refuse(ReplayError::Exhausted { .. })
        ));
        let until_divergence = ReplaySession::new(trace, ReplayMode::UntilDivergence);
        for _ in 0..2 {
            assert!(matches!(
                until_divergence.decide_hashed("predict", unused_hash),
                ReplayDirective::Live
            ));
        }
        assert_eq!(until_divergence.finish().live, 2);
    }
}
