//! What an optimizer optimizes: [`OptimizeTarget`], the lane-erased pair of
//! (thing under optimization, evaluation harness).
//!
//! Two lanes, one currency:
//!
//! - [`OptimizeTarget::module`] — a typed [`Module`] (+ [`Predictors`]
//!   discovery), a trainset slice, and a [`TypedMetric`]. Candidate injection
//!   is ambient ([`fx::with_params`](crate::fx::with_params)) — evaluation
//!   never mutates the module; the single mutation is the caller-driven
//!   [`install`](OptimizeTarget::install) of the winner at the end.
//! - [`OptimizeTarget::program`] — an interpreter-loaded IR
//!   [`Program`](crate::ir::Program), labeled [`DemoRow`] examples, and a
//!   [`ProgramMetric`]. Candidates read through
//!   [`ir::Overlay`](crate::ir::Overlay)s at render time; the winner is
//!   retrievable as an overlay ([`winner_overlay`](OptimizeTarget::winner_overlay))
//!   for [`Program::bake`](crate::ir::Program::bake).
//!
//! Construction runs the **naming pass** (module lane): every leaf the module
//! declares via [`Predictors`] is stamped with its declared name
//! ([`PredictorInfo::set_trace_name`]), so trace spans, candidate entries, and
//! persistence all address the same names. The target also snapshots
//! [`LeafInfo`] (schema text, current instruction, demos) — the read surface
//! strategies build candidates from — and computes the baseline identity.
//! Module-lane installation refreshes that snapshot from the installed state.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Result, anyhow};
use futures::future::LocalBoxFuture;
use serde::Serialize;
use serde_json::Value;

use crate::Module;
use crate::core::{
    PredictState, PredictorInfo, PredictorStateUpdate, Predictors, ToInput,
    apply_predictor_state_updates,
};
use crate::evaluate::{Eval, TypedMetric, evaluator::rollout_traced};
use crate::ir::graph::Node;
use crate::ir::interp::{Budget as RunBudget, Interpreter};
use crate::ir::params::{DemoRow, Overlay, ParamValue};
use crate::optimizer::engine::{
    BoundCandidate, Candidate, CandidatePayload, RolloutDetail, canonical_hash,
};
use crate::trace::{JsonMap, Trace, TraceMeta, TraceOutcome, capture_with_meta};

/// How a program-lane strategy tells the engine what "good" means: score one
/// interpreter output (`JsonMap` of the program's output signature fields)
/// against a labeled example.
///
/// The JSON-native sibling of [`TypedMetric`](crate::evaluate::TypedMetric) —
/// loaded programs have no static output type, so the metric sees the same
/// value model the interpreter produces. The rollout's captured [`Trace`] is
/// always provided; LM-as-judge metrics run outside the capture scope exactly
/// as in the module lane.
#[allow(async_fn_in_trait)]
pub trait ProgramMetric: Send + Sync {
    async fn evaluate(
        &self,
        example: &DemoRow,
        output: &JsonMap,
        trace: Option<&Trace>,
    ) -> Result<Eval>;
}

/// The read surface strategies build candidates from: one optimizable leaf's
/// name, current values, and field contract. Snapshotted at target
/// construction (candidates never mutate the target mid-run, so the snapshot
/// stays valid for the whole optimization).
#[derive(Clone, Debug)]
pub struct LeafInfo {
    /// The leaf's canonical name (trace component / candidate key).
    pub name: String,
    /// Current effective instruction (override or default).
    pub instruction: String,
    /// The signature/default instruction (ignoring overrides).
    pub default_instruction: String,
    /// Current demos as flat JSON rows.
    pub demos: Vec<JsonMap>,
    /// Input fields as `(lm name, docs)` pairs.
    pub input_fields: Vec<(String, String)>,
    /// Output fields as `(lm name, docs)` pairs.
    pub output_fields: Vec<(String, String)>,
}

impl LeafInfo {
    fn from_predictor(name: &str, info: &dyn PredictorInfo) -> Self {
        let schema = info.schema();
        Self {
            name: name.to_string(),
            instruction: info.instruction(),
            default_instruction: info.default_instruction(),
            demos: info.demos_as_json(),
            input_fields: schema
                .input_fields()
                .iter()
                .map(|field| (field.lm_name.to_string(), field.docs.clone()))
                .collect(),
            output_fields: schema
                .output_fields()
                .iter()
                .map(|field| (field.lm_name.to_string(), field.docs.clone()))
                .collect(),
        }
    }

    /// Renders the leaf's input/output contract for reflection prompts
    /// (GEPA/SIMBA format).
    pub fn schema_for_reflection(&self) -> String {
        let mut result = String::new();
        result.push_str("Input fields:\n");
        for (name, docs) in &self.input_fields {
            let docs = if docs.is_empty() {
                "No description"
            } else {
                docs
            };
            result.push_str(&format!("  - {name}: {docs}\n"));
        }
        result.push_str("Output fields:\n");
        for (name, docs) in &self.output_fields {
            let docs = if docs.is_empty() {
                "No description"
            } else {
                docs
            };
            result.push_str(&format!("  - {name}: {docs}\n"));
        }
        result
    }
}

/// The lane-erased evaluation surface the engine drives. Object-safe so the
/// [`Optimizer`](crate::optimizer::Optimizer) trait can be too.
trait Lane {
    fn num_examples(&self) -> usize;
    fn example_uid(&self, idx: usize) -> u64;
    fn leaves(&self) -> &[LeafInfo];
    fn baseline(&self) -> u64;
    fn bind_params(&self, params: &Arc<crate::fx::Params>) -> Result<BoundCandidate>;
    fn bind_overlay(&self, overlay: &Arc<Overlay>) -> Result<BoundCandidate>;
    fn run<'s>(
        &'s self,
        idx: usize,
        bound: BoundCandidate,
        candidate_hash: u64,
        detail: RolloutDetail,
        observe_capture: &'s dyn Fn(&Trace),
    ) -> LocalBoxFuture<'s, Result<(Eval, Trace)>>;
    fn output<'s>(&'s self, idx: usize, bound: BoundCandidate)
    -> LocalBoxFuture<'s, Result<Value>>;
    fn install(&mut self, winner: &Candidate) -> Result<()>;
    fn winner_overlay(&self) -> Option<Arc<Overlay>>;
}

// ---------------------------------------------------------------------------
// Module lane
// ---------------------------------------------------------------------------

struct ModuleLane<'a, E, M, MT> {
    module: &'a mut M,
    /// Validation examples first (empty when no valset), trainset after.
    val: &'a [E],
    train: &'a [E],
    metric: &'a MT,
    uids: Vec<u64>,
    snapshot: ModuleSnapshot,
}

impl<'a, E, M, MT> ModuleLane<'a, E, M, MT>
where
    E: ToInput<M::Input> + Serialize + Sync,
    M: Module + Predictors,
    MT: TypedMetric<E, M>,
{
    fn example(&self, idx: usize) -> &E {
        if idx < self.val.len() {
            &self.val[idx]
        } else {
            &self.train[idx - self.val.len()]
        }
    }
}

struct ModuleSnapshot {
    leaves: Vec<LeafInfo>,
    baseline: u64,
}

impl ModuleSnapshot {
    fn read(named: &[(String, &dyn PredictorInfo)]) -> Self {
        let states: std::collections::BTreeMap<&str, PredictState> = named
            .iter()
            .map(|(name, info)| (name.as_str(), info.dump_state()))
            .collect();
        Self {
            leaves: named
                .iter()
                .map(|(name, info)| LeafInfo::from_predictor(name, *info))
                .collect(),
            baseline: canonical_hash(&states),
        }
    }
}

impl<'a, E, M, MT> Lane for ModuleLane<'a, E, M, MT>
where
    E: ToInput<M::Input> + Serialize + Sync,
    M: Module + Predictors,
    MT: TypedMetric<E, M>,
{
    fn num_examples(&self) -> usize {
        self.val.len() + self.train.len()
    }

    fn example_uid(&self, idx: usize) -> u64 {
        self.uids[idx]
    }

    fn leaves(&self) -> &[LeafInfo] {
        &self.snapshot.leaves
    }

    fn baseline(&self) -> u64 {
        self.snapshot.baseline
    }

    fn bind_params(&self, params: &Arc<crate::fx::Params>) -> Result<BoundCandidate> {
        Ok(BoundCandidate::Params(Arc::clone(params)))
    }

    fn bind_overlay(&self, _overlay: &Arc<Overlay>) -> Result<BoundCandidate> {
        Err(anyhow!(
            "an ir::Overlay candidate cannot be evaluated on a module target; \
             register a Candidate instead"
        ))
    }

    fn run<'s>(
        &'s self,
        idx: usize,
        bound: BoundCandidate,
        candidate_hash: u64,
        detail: RolloutDetail,
        observe_capture: &'s dyn Fn(&Trace),
    ) -> LocalBoxFuture<'s, Result<(Eval, Trace)>> {
        Box::pin(async move {
            let BoundCandidate::Params(params) = bound else {
                return Err(anyhow!("module lane received a non-params candidate"));
            };
            let example = self.example(idx);
            let meta = TraceMeta {
                candidate_hash: Some(candidate_hash),
                ..TraceMeta::default()
            };
            let module: &M = &*self.module;
            // The candidate is scoped ambiently around the whole traced
            // rollout: each Predict leaf binds its own entry at call time.
            crate::fx::with_params_shared(
                params,
                rollout_traced(
                    module,
                    example,
                    self.metric,
                    meta,
                    detail.includes_trace(),
                    observe_capture,
                ),
            )
            .await
        })
    }

    fn output<'s>(
        &'s self,
        idx: usize,
        bound: BoundCandidate,
    ) -> LocalBoxFuture<'s, Result<Value>> {
        Box::pin(async move {
            let BoundCandidate::Params(params) = bound else {
                return Err(anyhow!("module lane received a non-params candidate"));
            };
            let example = self.example(idx);
            let input = example.to_input()?;
            let module: &M = &*self.module;
            let predicted = crate::fx::with_params_shared(params, module.call(input))
                .await
                .map_err(|err| anyhow!("{err}"))?;
            Ok(serde_json::to_value(predicted.into_inner()).unwrap_or(Value::Null))
        })
    }

    fn install(&mut self, winner: &Candidate) -> Result<()> {
        if winner.is_empty() {
            return Ok(());
        }
        let mut leaves = HashMap::new();
        for (name, predictor) in self.module.predictors_mut() {
            leaves.entry(name).or_insert(predictor);
        }
        let mut updates = Vec::with_capacity(winner.slots.len());
        for (name, slot) in &winner.slots {
            let Some((name, predictor)) = leaves.remove_entry(name) else {
                return Err(anyhow!("predictor `{name}` not found in the module"));
            };
            let original = predictor.dump_state();
            let proposed = PredictState {
                instruction_override: if slot.clear_instruction {
                    None
                } else {
                    slot.instruction
                        .as_ref()
                        .or(original.instruction_override.as_ref())
                        .cloned()
                },
                demos: slot.demos.as_ref().unwrap_or(&original.demos).clone(),
            };
            updates.push(PredictorStateUpdate {
                name,
                predictor,
                original,
                proposed,
            });
        }
        let result = apply_predictor_state_updates(&mut updates, "install winner on");
        drop(updates);
        drop(leaves);
        // A custom predictor may fail during rollback. Read the actual state
        // even when installation reports an error.
        self.snapshot = ModuleSnapshot::read(&self.module.predictors());
        result
    }

    fn winner_overlay(&self) -> Option<Arc<Overlay>> {
        None
    }
}

// ---------------------------------------------------------------------------
// Program lane
// ---------------------------------------------------------------------------

/// An immutable set of rows paired with its content fingerprints.
pub(super) struct ProgramExamples<'a> {
    rows: &'a [DemoRow],
    uids: Cow<'a, [u64]>,
    order: u64,
}

impl<'a> ProgramExamples<'a> {
    pub(super) fn new(rows: &'a [DemoRow]) -> Self {
        let uids: Vec<_> = rows.iter().map(canonical_hash).collect();
        Self {
            rows,
            order: canonical_hash(&uids),
            uids: Cow::Owned(uids),
        }
    }

    pub(super) fn borrowed(&self) -> ProgramExamples<'_> {
        ProgramExamples {
            rows: self.rows,
            uids: Cow::Borrowed(&self.uids),
            order: self.order,
        }
    }
}

struct ProgramLane<'a, MT> {
    interp: &'a Interpreter,
    examples: ProgramExamples<'a>,
    metric: &'a MT,
    leaves: Vec<LeafInfo>,
    program_tag: String,
    winner: Option<Arc<Overlay>>,
}

impl<'a, MT: ProgramMetric> Lane for ProgramLane<'a, MT> {
    fn num_examples(&self) -> usize {
        self.examples.rows.len()
    }

    fn example_uid(&self, idx: usize) -> u64 {
        self.examples.uids[idx]
    }

    fn leaves(&self) -> &[LeafInfo] {
        &self.leaves
    }

    fn baseline(&self) -> u64 {
        self.interp.program().meta.program_hash
    }

    fn bind_params(&self, params: &Arc<crate::fx::Params>) -> Result<BoundCandidate> {
        let overlay = params
            .bind(self.interp.program())
            .map_err(|err| anyhow!("failed to bind candidate against program: {err}"))?;
        Ok(BoundCandidate::Overlay(Arc::new(overlay)))
    }

    fn bind_overlay(&self, overlay: &Arc<Overlay>) -> Result<BoundCandidate> {
        Ok(BoundCandidate::Overlay(Arc::clone(overlay)))
    }

    fn run<'s>(
        &'s self,
        idx: usize,
        bound: BoundCandidate,
        candidate_hash: u64,
        detail: RolloutDetail,
        observe_capture: &'s dyn Fn(&Trace),
    ) -> LocalBoxFuture<'s, Result<(Eval, Trace)>> {
        Box::pin(async move {
            let BoundCandidate::Overlay(overlay) = bound else {
                return Err(anyhow!("program lane received a non-overlay candidate"));
            };
            let example = &self.examples.rows[idx];
            let meta = TraceMeta {
                candidate_hash: Some(candidate_hash),
                input: Some(example.input.clone()),
                tags: [("program".to_string(), self.program_tag.clone())]
                    .into_iter()
                    .collect(),
                ..TraceMeta::default()
            };
            let started = Instant::now();
            let (result, mut trace) = capture_with_meta(meta, || {
                self.interp
                    .run(example.input.clone(), Some(overlay), RunBudget::unlimited())
            })
            .await;
            observe_capture(&trace);
            let output =
                result.map_err(|err| anyhow!("candidate failed on example {idx}: {err}"))?;
            // Metric runs outside the capture scope so LM-as-judge metrics
            // don't pollute the execution trace.
            let eval = self.metric.evaluate(example, &output, Some(&trace)).await?;
            if detail.includes_trace() {
                trace.outcome = Some(TraceOutcome {
                    output: Some(output),
                    error: None,
                    eval: Some(eval.clone()),
                    duration_us: started.elapsed().as_micros() as u64,
                });
            }
            Ok((eval, trace))
        })
    }

    fn output<'s>(
        &'s self,
        idx: usize,
        bound: BoundCandidate,
    ) -> LocalBoxFuture<'s, Result<Value>> {
        Box::pin(async move {
            let BoundCandidate::Overlay(overlay) = bound else {
                return Err(anyhow!("program lane received a non-overlay candidate"));
            };
            let example = &self.examples.rows[idx];
            let output = self
                .interp
                .run(example.input.clone(), Some(overlay), RunBudget::unlimited())
                .await
                .map_err(|err| anyhow!("{err}"))?;
            Ok(Value::Object(output))
        })
    }

    fn install(&mut self, winner: &Candidate) -> Result<()> {
        let overlay = winner
            .to_params()
            .bind(self.interp.program())
            .map_err(|err| anyhow!("failed to bind winner against program: {err}"))?;
        self.winner = Some(Arc::new(overlay));
        Ok(())
    }

    fn winner_overlay(&self) -> Option<Arc<Overlay>> {
        self.winner.clone()
    }
}

/// Leaf snapshot for a loaded program: one [`LeafInfo`] per `predict`/`agent`
/// node, current values read from the param slot defaults.
fn program_leaves(interp: &Interpreter) -> Vec<LeafInfo> {
    let program = interp.program();
    let mut leaves = Vec::new();
    for (node_id, node) in program.nodes.iter() {
        let sig = match node {
            Node::Predict(n) => n.sig,
            Node::AgentLoop(n) => n.sig,
            _ => continue,
        };
        let Some(name) = program.leaf_name(node_id) else {
            continue;
        };
        let def = &program.sigs[sig];
        let instruction = program
            .param_id(&format!("{name}.instruction"))
            .and_then(|id| match &program.params[id].default {
                ParamValue::Instruction { text } => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_else(|| def.instruction.to_string());
        let demos = program
            .param_id(&format!("{name}.demos"))
            .and_then(|id| match &program.params[id].default {
                ParamValue::Demos { rows } => Some(
                    rows.iter()
                        .map(|row| {
                            let mut flat = row.input.clone();
                            flat.extend(row.output.iter().map(|(k, v)| (k.clone(), v.clone())));
                            flat
                        })
                        .collect(),
                ),
                _ => None,
            })
            .unwrap_or_default();
        let field_pairs = |fields: &[crate::ir::sig::FieldDef]| {
            fields
                .iter()
                .map(|field| {
                    (
                        field.lm_name.to_string(),
                        field.docs.as_deref().unwrap_or("").to_string(),
                    )
                })
                .collect()
        };
        leaves.push(LeafInfo {
            name: name.to_string(),
            default_instruction: instruction.clone(),
            instruction,
            demos,
            input_fields: field_pairs(&def.inputs),
            output_fields: field_pairs(&def.outputs),
        });
    }
    leaves
}

// ---------------------------------------------------------------------------
// OptimizeTarget
// ---------------------------------------------------------------------------

/// Identity of a module or interpreter and its metric for score reuse.
///
/// Keep a context while the execution bindings and scoring behavior are
/// equivalent. Example order and subsets can change within a context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EvaluationContext(u64);

impl EvaluationContext {
    /// Starts an independent evaluation context.
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for EvaluationContext {
    fn default() -> Self {
        Self::new()
    }
}

/// The thing an [`Optimizer`](crate::optimizer::Optimizer) optimizes: a
/// module or a program, packaged with its example set and metric. See the
/// module docs for the two lanes.
pub struct OptimizeTarget<'a> {
    lane: Box<dyn Lane + 'a>,
    context: EvaluationContext,
    example_order: u64,
    /// Number of leading example columns that are the validation set, when a
    /// separate valset was supplied.
    val_len: Option<usize>,
}

impl<'a> OptimizeTarget<'a> {
    /// A module-lane target: typed module + trainset (by reference) + metric.
    ///
    /// Runs the naming pass: each leaf declared by [`Predictors`] is stamped
    /// with its declared name so traces, candidates, and persistence agree.
    pub fn module<E, M, MT>(module: &'a mut M, trainset: &'a [E], metric: &'a MT) -> Self
    where
        E: ToInput<M::Input> + Serialize + Sync,
        M: Module + Predictors,
        MT: TypedMetric<E, M>,
    {
        Self::module_with_valset(module, trainset, None, metric)
    }

    /// [`module`](Self::module) with an optional validation set.
    ///
    /// When `valset` is `Some`, its examples become the *leading* columns of
    /// the target ([`val_columns`](Self::val_columns)) and the trainset the
    /// trailing ones ([`train_columns`](Self::train_columns)) — the layout
    /// GEPA's Pareto bookkeeping uses. When `None`, both views cover the
    /// whole trainset.
    pub fn module_with_valset<E, M, MT>(
        module: &'a mut M,
        trainset: &'a [E],
        valset: Option<&'a [E]>,
        metric: &'a MT,
    ) -> Self
    where
        E: ToInput<M::Input> + Serialize + Sync,
        M: Module + Predictors,
        MT: TypedMetric<E, M>,
    {
        // Naming pass: stamp each leaf with its declared name (once per run).
        for (name, info) in module.predictors_mut() {
            info.set_trace_name(&name);
        }
        let snapshot = ModuleSnapshot::read(&module.predictors());
        let val_len = valset.map(<[E]>::len);
        let val = valset.unwrap_or(&[]);
        let uids: Vec<_> = val
            .iter()
            .chain(trainset.iter())
            .map(canonical_hash)
            .collect();

        Self {
            context: EvaluationContext::new(),
            example_order: canonical_hash(&(val_len, &uids)),
            lane: Box::new(ModuleLane {
                module,
                val,
                train: trainset,
                metric,
                uids,
                snapshot,
            }),
            val_len,
        }
    }

    /// A program-lane target: interpreter-loaded program + labeled examples
    /// (by reference) + JSON metric.
    pub fn program<MT>(interp: &'a Interpreter, examples: &'a [DemoRow], metric: &'a MT) -> Self
    where
        MT: ProgramMetric,
    {
        Self::program_with_examples(
            interp,
            ProgramExamples::new(examples),
            metric,
            EvaluationContext::new(),
        )
    }

    pub(super) fn program_with_examples<MT>(
        interp: &'a Interpreter,
        examples: ProgramExamples<'a>,
        metric: &'a MT,
        context: EvaluationContext,
    ) -> Self
    where
        MT: ProgramMetric,
    {
        let leaves = program_leaves(interp);
        let program_tag = format!("{:016x}", interp.program().meta.program_hash);
        Self {
            context,
            example_order: examples.order,
            lane: Box::new(ProgramLane {
                interp,
                examples,
                metric,
                leaves,
                program_tag,
                winner: None,
            }),
            val_len: None,
        }
    }

    /// Reuses an evaluation context across targets built with equivalent
    /// execution bindings and metrics, for example after reordering examples.
    /// Constructors otherwise give each target its own context.
    pub fn with_evaluation_context(mut self, context: EvaluationContext) -> Self {
        self.context = context;
        self
    }

    pub(super) fn evaluation_context_id(&self) -> u64 {
        self.context.0
    }

    /// The optimizable leaves' read surface. Module leaves reflect installed
    /// predictor state. Program leaves describe the interpreter's base program.
    pub fn leaves(&self) -> &[LeafInfo] {
        self.lane.leaves()
    }

    pub fn num_examples(&self) -> usize {
        self.lane.num_examples()
    }

    /// Whether this target carries a distinct validation set.
    pub fn has_valset(&self) -> bool {
        self.val_len.is_some()
    }

    /// The scoring columns: the validation prefix when a valset was supplied,
    /// else every example.
    pub fn val_columns(&self) -> Vec<usize> {
        match self.val_len {
            Some(len) => (0..len).collect(),
            None => (0..self.num_examples()).collect(),
        }
    }

    /// The minibatch-sampling pool: the trainset suffix when a valset was
    /// supplied, else every example.
    pub fn train_columns(&self) -> Vec<usize> {
        match self.val_len {
            Some(len) => (len..self.num_examples()).collect(),
            None => (0..self.num_examples()).collect(),
        }
    }

    /// Installs the winning candidate — the **one** mutation of the run.
    ///
    /// Module lane: merges each slot into the named leaf's state through
    /// [`PredictorInfo::load_state`]. Program lane: binds the winner to an
    /// overlay retrievable via [`winner_overlay`](Self::winner_overlay)
    /// (bake it with [`Program::bake`](crate::ir::Program::bake)).
    pub fn install(&mut self, winner: &Candidate) -> Result<()> {
        self.lane.install(winner)
    }

    /// The installed winner as a bound overlay (program lane only).
    pub fn winner_overlay(&self) -> Option<Arc<Overlay>> {
        self.lane.winner_overlay()
    }

    pub(crate) fn baseline(&self) -> u64 {
        self.lane.baseline()
    }

    pub(super) fn example_order(&self) -> u64 {
        self.example_order
    }

    pub(crate) fn example_uid(&self, idx: usize) -> u64 {
        self.lane.example_uid(idx)
    }

    pub(crate) fn bind(&self, payload: &CandidatePayload) -> Result<BoundCandidate> {
        match payload {
            CandidatePayload::Params { params, .. } => self.lane.bind_params(params),
            CandidatePayload::Overlay(overlay) => self.lane.bind_overlay(overlay),
        }
    }

    pub(super) fn validate_example_indices(&self, indices: &[usize]) -> Result<()> {
        let num_examples = self.num_examples();
        if let Some(&bad) = indices.iter().find(|&&idx| idx >= num_examples) {
            return Err(anyhow!(
                "example index {bad} out of range ({num_examples} examples)"
            ));
        }
        Ok(())
    }

    pub(super) async fn run(
        &self,
        idx: usize,
        bound: BoundCandidate,
        candidate_hash: u64,
        detail: RolloutDetail,
        observe_capture: &dyn Fn(&Trace),
    ) -> Result<(Eval, Trace)> {
        self.lane
            .run(idx, bound, candidate_hash, detail, observe_capture)
            .await
    }

    /// Runs the given examples under `candidate` and returns the bare output
    /// values (no metric, no trace capture) — GEPA's best-output collection.
    /// Sequential, in index order.
    pub async fn candidate_outputs(
        &self,
        indices: &[usize],
        candidate: &Candidate,
    ) -> Result<Vec<Value>> {
        self.validate_example_indices(indices)?;
        let bound = self.lane.bind_params(&Arc::new(candidate.to_params()))?;
        let mut outputs = Vec::with_capacity(indices.len());
        for &idx in indices {
            outputs.push(self.lane.output(idx, bound.clone()).await?);
        }
        Ok(outputs)
    }
}
