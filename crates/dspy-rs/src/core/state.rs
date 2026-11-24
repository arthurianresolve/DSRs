//! Persistence for optimized module state.
//!
//! After an optimizer tunes a module, the improved instructions and demos live only
//! in memory. [`ModuleState`] snapshots every [`Predict`](crate::Predict) leaf into a
//! serializable value so an optimized program can be saved to disk and reloaded
//! without re-running optimization:
//!
//! ```ignore
//! // After optimization:
//! ModuleState::from_module(&module)?.save("optimized.json")?;
//!
//! // In production:
//! let mut module = MyPipeline::new();
//! ModuleState::load("optimized.json")?.apply(&mut module)?;
//! ```
//!
//! Leaves are discovered through the explicit [`Predictors`] contract — state
//! is keyed by the names the module declares for its leaves.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::core::{PredictorInfo, Predictors};
use crate::trace::JsonMap;

/// Serializable snapshot of a [`crate::Predict`]'s optimizable state.
///
/// Contains demos (as flat JSON rows) and the instruction override.
/// Produced by optimizers when they tune a predictor; persist a whole module's
/// worth of these with [`ModuleState`], or inject them per call tree with
/// [`fx::Params`](crate::fx::Params).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PredictState {
    /// Demo rows as flat JSON objects: field name → value, with input and
    /// output fields merged into one object. This is the serde boundary for
    /// demos-as-data — rows are split back into the predictor's typed
    /// `Demo<S>` via the signature schema on load.
    #[serde(default)]
    pub demos: Vec<JsonMap>,
    /// The instruction override, if any.
    #[serde(default)]
    pub instruction_override: Option<String>,
}

/// Serializable snapshot of every [`Predict`](crate::Predict) leaf in a module,
/// keyed by the leaf name the module declares via [`Predictors`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModuleState {
    /// Per-predictor state, keyed by leaf name (`BTreeMap` keeps JSON output stable).
    pub predictors: BTreeMap<String, PredictState>,
}

pub(crate) struct PredictorStateUpdate<'a> {
    pub name: String,
    pub predictor: &'a mut dyn PredictorInfo,
    pub original: PredictState,
    pub proposed: PredictState,
}

pub(crate) fn apply_predictor_state_updates(
    updates: &mut [PredictorStateUpdate<'_>],
    action: &str,
) -> Result<()> {
    for update in updates.iter() {
        update
            .predictor
            .validate_state(&update.proposed)
            .map_err(|err| anyhow!("failed to {action} `{}`: {err}", update.name))?;
    }

    for index in 0..updates.len() {
        let update = &mut updates[index];
        if let Err(err) = update
            .predictor
            .load_state(std::mem::take(&mut update.proposed))
        {
            let message = format!("failed to {action} `{}`: {err}", update.name);
            let mut rollback_errors = Vec::new();
            for attempted in &mut updates[..=index] {
                if let Err(err) = attempted
                    .predictor
                    .load_state(std::mem::take(&mut attempted.original))
                {
                    rollback_errors.push(format!("`{}`: {err}", attempted.name));
                }
            }
            if rollback_errors.is_empty() {
                return Err(anyhow!("{message}"));
            }
            return Err(anyhow!(
                "{message}; rollback failed for {}",
                rollback_errors.join(", ")
            ));
        }
    }
    Ok(())
}

impl ModuleState {
    /// Snapshots the current state (instruction overrides + demos) of every
    /// `Predict` leaf in `module`.
    pub fn from_module<M>(module: &M) -> Result<Self>
    where
        M: Predictors + ?Sized,
    {
        let mut predictors = BTreeMap::new();
        for (name, info) in module.predictors() {
            predictors.insert(name, info.dump_state());
        }
        Ok(Self { predictors })
    }

    /// Applies this state to a module, stamping each restored leaf's trace
    /// name with its declared name after all loads succeed.
    ///
    /// Every name in the state must resolve to a `Predict` leaf in `module` —
    /// unknown names are an error, since they mean the saved state and the module
    /// structure have diverged. Predictors not named in the state are left untouched.
    /// Built-in `Predict` leaves validate demo rows before mutation. If an
    /// unexpected load error occurs, previously loaded leaves are restored
    /// from their original snapshots; a failed rollback is reported.
    pub fn apply<M>(&self, module: &mut M) -> Result<()>
    where
        M: Predictors + ?Sized,
    {
        let mut remaining: BTreeMap<&str, &PredictState> = self
            .predictors
            .iter()
            .map(|(name, state)| (name.as_str(), state))
            .collect();
        let predictors = module.predictors_mut();
        for (name, _) in &predictors {
            remaining.remove(name.as_str());
        }
        if !remaining.is_empty() {
            let missing = remaining.keys().copied().collect::<Vec<_>>().join("`, `");
            return Err(anyhow!(
                "state refers to predictors not present in the module: `{missing}`"
            ));
        }

        let mut updates: Vec<_> = predictors
            .into_iter()
            .filter_map(|(name, predictor)| {
                let proposed = self.predictors.get(&name)?.clone();
                Some(PredictorStateUpdate {
                    name,
                    original: predictor.dump_state(),
                    predictor,
                    proposed,
                })
            })
            .collect();
        apply_predictor_state_updates(&mut updates, "load state for")?;
        for update in updates {
            update.predictor.set_trace_name(&update.name);
        }
        Ok(())
    }

    /// Serializes to pretty-printed JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("failed to serialize module state")
    }

    /// Deserializes from JSON produced by [`to_json`](ModuleState::to_json).
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).context("failed to deserialize module state")
    }

    /// Writes the state to a JSON file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        std::fs::write(path, self.to_json()?)
            .with_context(|| format!("failed to write module state to `{}`", path.display()))
    }

    /// Reads state from a JSON file written by [`save`](ModuleState::save).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read module state from `{}`", path.display()))?;
        Self::from_json(&json)
    }
}
