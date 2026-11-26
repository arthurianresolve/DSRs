//! `dsrs fmt`: canonical-print a `.dsrs` artifact.
//!
//! The canonical form is `Program::to_dsrs` — the same text that seals
//! `program_hash` (minus lineage) and that `bake` writes. Formatting is
//! therefore parse→print, never token shuffling: an artifact that doesn't
//! parse doesn't format.

use std::path::Path;

use anyhow::Context;
use dspy_rs::ir::Program;

/// Outcome of [`fmt_file`].
#[derive(Debug, PartialEq, Eq)]
pub enum FmtOutcome {
    /// `--write`: the file already was canonical; nothing touched.
    Unchanged,
    /// `--write`: the file was rewritten in canonical form.
    Rewrote,
    /// No `--write`: the canonical text, for stdout.
    Canonical(String),
}

/// Parses `path` and canonically prints it. With `write`, rewrites the file
/// in place (only when the bytes differ); otherwise returns the text.
pub fn fmt_file(path: impl AsRef<Path>, write: bool) -> anyhow::Result<FmtOutcome> {
    fmt_file_with_options(path, write, false)
}

/// Formats a file, explicitly allowing model URL redaction when requested.
pub fn fmt_file_with_options(
    path: impl AsRef<Path>,
    write: bool,
    redact_model_urls: bool,
) -> anyhow::Result<FmtOutcome> {
    let path = path.as_ref();
    let program = Program::load_dsrs(path)?;
    if !write {
        return Ok(FmtOutcome::Canonical(program.to_dsrs()));
    }
    let formatted = if redact_model_urls {
        program.to_dsrs()
    } else {
        program.to_dsrs_for_rewrite().map_err(|changed| {
            anyhow::anyhow!(
                "model base URL would change on export for {}: pass --redact-model-urls to write a sanitized artifact",
                changed.join(", ")
            )
        })?
    };
    let original = std::fs::read_to_string(path)
        .with_context(|| format!("failed to re-read `{}`", path.display()))?;
    if original == formatted {
        return Ok(FmtOutcome::Unchanged);
    }
    std::fs::write(path, formatted)
        .with_context(|| format!("failed to write `{}`", path.display()))?;
    Ok(FmtOutcome::Rewrote)
}
