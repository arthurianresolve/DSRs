//! JSONL wire format for [`Trace`]: header line, span lines, optional footer.
//!
//! ```text
//! {"h":{...meta...},"components":[...],"models":[...],"prefixes":[...]}
//! {"id":0,"component":0,"seq":0,...}
//! {"f":{...outcome...}}
//! ```
//!
//! The footer is a separate line so evals can be attached after spans are
//! written without rewriting the file. Oversized text fields are truncated at
//! serialization time (never during capture) with a content-hash marker;
//! truncated spans are marked `complete = false` and refuse strict replay.

use std::io::{self, Write};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::trace::span::{
    ModelEntry, PrefixEntry, Span, SpanEvent, Trace, TraceMeta, TraceOutcome,
};
use crate::utils::hash::stable_hash_debug;
use crate::{ContentBlock, Message};

/// Highest format version this reader understands.
pub const TRACE_FORMAT_VERSION: u32 = 1;

/// Per-message-text and raw-output budget (bytes).
const TEXT_BUDGET: usize = 64 * 1024;
/// Tool args/result budget (bytes).
const TOOL_BUDGET: usize = 16 * 1024;

#[derive(Serialize)]
struct HeaderRef<'a> {
    h: &'a TraceMeta,
    components: &'a [String],
    /// RFC 0001 §1's reserved join column — additive, omitted when empty.
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    param_ids: &'a [Option<Vec<crate::ir::ParamId>>],
    models: &'a [ModelEntry],
    prefixes: &'a [PrefixEntry],
}

#[derive(Deserialize)]
struct HeaderOwned {
    h: TraceMeta,
    #[serde(default)]
    components: Vec<String>,
    #[serde(default)]
    param_ids: Vec<Option<Vec<crate::ir::ParamId>>>,
    #[serde(default)]
    models: Vec<ModelEntry>,
    #[serde(default)]
    prefixes: Vec<PrefixEntry>,
}

#[derive(Serialize, Deserialize)]
struct Footer<T> {
    f: T,
}

impl Trace {
    /// Serializes to canonical JSONL: one header line, one line per span, one
    /// footer line when an outcome is recorded.
    pub fn to_jsonl(&self) -> Result<String> {
        let mut out = Vec::new();
        serde_json::to_writer(
            &mut out,
            &HeaderRef {
                h: &self.meta,
                components: &self.components,
                param_ids: &self.param_ids,
                models: &self.models,
                prefixes: &self.prefixes,
            },
        )?;
        out.push(b'\n');
        for span in &self.spans {
            match truncated_span(span) {
                Some(truncated) => serde_json::to_writer(&mut out, &truncated)?,
                None => serde_json::to_writer(&mut out, span)?,
            }
            out.push(b'\n');
        }
        if let Some(outcome) = &self.outcome {
            serde_json::to_writer(&mut out, &Footer { f: outcome })?;
            out.push(b'\n');
        }
        Ok(String::from_utf8(out)?)
    }

    /// Parses JSONL produced by [`to_jsonl`](Trace::to_jsonl). Unknown fields
    /// and unknown span-event tags are skipped; a version newer than this
    /// reader understands is rejected.
    pub fn from_jsonl(jsonl: &str) -> Result<Trace> {
        let mut lines = jsonl.lines().filter(|line| !line.trim().is_empty());
        let header_line = lines.next().ok_or_else(|| anyhow!("empty trace file"))?;
        let header: HeaderOwned =
            serde_json::from_str(header_line).context("failed to parse trace header line")?;
        if header.h.v > TRACE_FORMAT_VERSION {
            return Err(anyhow!(
                "trace format version {} is newer than supported version {}",
                header.h.v,
                TRACE_FORMAT_VERSION
            ));
        }

        let mut trace = Trace {
            meta: header.h,
            components: header.components,
            param_ids: header.param_ids,
            models: header.models,
            prefixes: header.prefixes,
            spans: Vec::new(),
            outcome: None,
        };

        for line in lines {
            // The footer decoder takes precedence. Objects without a literal
            // or escaped `f` key can go straight to the span decoder.
            let could_be_footer = !line.trim_start().starts_with('{')
                || line.contains("\"f\"")
                || line.contains(r"\u0066");
            if could_be_footer
                && let Ok(footer) = serde_json::from_str::<Footer<TraceOutcome>>(line)
            {
                trace.outcome = Some(footer.f);
                continue;
            }
            let mut span: Span =
                serde_json::from_str(line).context("failed to parse trace span line")?;
            // Unknown-tag placeholders don't survive re-serialization; drop
            // them here so a read-write cycle is stable.
            span.events
                .retain(|event| !matches!(event, SpanEvent::Unknown));
            trace.spans.push(span);
        }
        Ok(trace)
    }
}

fn truncate_marker(text: &str, budget: usize) -> String {
    let hash = stable_hash_debug(&text);
    let mut cut = budget;
    while cut > 0 && !text.is_char_boundary(cut) {
        cut -= 1;
    }
    format!(
        "{}…[truncated fnv1a={hash:016x} len={}]",
        &text[..cut],
        text.len()
    )
}

fn truncate_text(text: &mut String, budget: usize) -> bool {
    if text.len() <= budget {
        return false;
    }
    *text = truncate_marker(text, budget);
    true
}

fn truncate_messages(messages: &mut [Message]) -> bool {
    let mut truncated = false;
    for message in messages {
        for block in &mut message.content {
            if let ContentBlock::Text { text } = block {
                truncated |= truncate_text(text, TEXT_BUDGET);
            }
        }
    }
    truncated
}

fn messages_over_budget(messages: &[Message]) -> bool {
    messages
        .iter()
        .flat_map(|message| &message.content)
        .any(|block| matches!(block, ContentBlock::Text { text } if text.len() > TEXT_BUDGET))
}

#[derive(Default)]
struct ByteCount(usize);

impl Write for ByteCount {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self.0.saturating_add(bytes.len());
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn tool_args_over_budget(args: &serde_json::Value) -> bool {
    let mut count = ByteCount::default();
    serde_json::to_writer(&mut count, args).is_ok() && count.0 > TOOL_BUDGET
}

/// Returns a budget-respecting copy of the span when any field overflows,
/// `None` when the span fits as-is. Truncation clears `complete`.
fn truncated_span(span: &Span) -> Option<Span> {
    let over_budget = messages_over_budget(&span.suffix)
        || span
            .raw_output
            .as_ref()
            .is_some_and(|text| text.len() > TEXT_BUDGET)
        || span.events.iter().any(|event| match event {
            SpanEvent::ToolRun { args, result, .. } => {
                result.len() > TOOL_BUDGET || tool_args_over_budget(args)
            }
            SpanEvent::ChatMessage { message } => {
                messages_over_budget(std::slice::from_ref(message))
            }
            _ => false,
        });
    if !over_budget {
        return None;
    }

    let mut span = span.clone();
    truncate_messages(&mut span.suffix);
    if let Some(raw_output) = &mut span.raw_output {
        truncate_text(raw_output, TEXT_BUDGET);
    }
    for event in &mut span.events {
        match event {
            SpanEvent::ToolRun { args, result, .. } => {
                truncate_text(result, TOOL_BUDGET);
                let serialized = serde_json::to_string(args).unwrap_or_default();
                if serialized.len() > TOOL_BUDGET {
                    *args = serde_json::Value::String(truncate_marker(&serialized, TOOL_BUDGET));
                }
            }
            SpanEvent::ChatMessage { message } => {
                truncate_messages(std::slice::from_mut(message));
            }
            _ => {}
        }
    }
    span.complete = false;
    Some(span)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::span::{CompId, ModelId, SpanId};
    use crate::{LMConfig, LmUsage};

    fn test_span(suffix_text: &str) -> Span {
        Span {
            id: SpanId(0),
            component: CompId(0),
            seq: 0,
            parent: None,
            prefix: None,
            suffix: vec![Message::user(suffix_text)],
            input: None,
            model: ModelId(0),
            request_hash: 7,
            events: vec![SpanEvent::Exchange {
                message: Message::assistant("out"),
                usage: LmUsage::default(),
            }],
            raw_output: Some("out".to_string()),
            output: None,
            usage: LmUsage::default(),
            error: None,
            eval: None,
            started_at_us: 1,
            duration_us: 2,
            complete: true,
        }
    }

    fn test_trace(span: Span) -> Trace {
        Trace {
            meta: TraceMeta {
                v: 1,
                trace_id: "t".to_string(),
                started_at_us: 1,
                ..TraceMeta::default()
            },
            components: vec!["comp".to_string()],
            models: vec![ModelEntry::from_config(&LMConfig::default())],
            prefixes: Vec::new(),
            spans: vec![span],
            ..Trace::default()
        }
    }

    #[test]
    fn trace_reader_prefers_a_valid_escaped_footer_over_span_fields() {
        let jsonl = test_trace(test_span("hello")).to_jsonl().unwrap();
        let mut lines = jsonl.lines();
        let header = lines.next().unwrap();
        let span = lines.next().unwrap().strip_suffix('}').unwrap();
        let footer = serde_json::to_string(&TraceOutcome {
            duration_us: 5,
            ..TraceOutcome::default()
        })
        .unwrap();

        let mixed = format!("{header}\n{span},\"\\u0066\":{footer}}}\n");
        let parsed = Trace::from_jsonl(&mixed).unwrap();
        assert!(parsed.spans.is_empty());
        assert_eq!(parsed.outcome.unwrap().duration_us, 5);

        let malformed_footer = format!("{header}\n{span},\"f\":false}}\n");
        let parsed = Trace::from_jsonl(&malformed_footer).unwrap();
        assert_eq!(parsed.spans.len(), 1);
        assert!(parsed.outcome.is_none());
    }

    #[test]
    fn tool_argument_budget_counts_escaped_utf8_bytes() {
        let mut text = format!("🚀\n{}", "x".repeat(TOOL_BUDGET - 8));
        assert!(!tool_args_over_budget(&serde_json::json!(text)));
        text.push('x');
        assert!(tool_args_over_budget(&serde_json::json!(text)));
    }

    #[test]
    fn oversized_suffix_is_truncated_and_marked_incomplete() {
        let big = "x".repeat(TEXT_BUDGET * 2);
        let trace = test_trace(test_span(&big));
        let jsonl = trace.to_jsonl().expect("serialize");
        let parsed = Trace::from_jsonl(&jsonl).expect("deserialize");
        let span = &parsed.spans[0];
        assert!(!span.complete);
        let text = span.suffix[0].text_content();
        assert!(text.len() < big.len());
        assert!(text.contains("truncated fnv1a="));
        assert!(text.contains(&format!("len={}", big.len())));
    }

    #[test]
    fn oversized_chat_edit_is_truncated_and_marked_incomplete() {
        let mut span = test_span("hello");
        span.events.push(SpanEvent::ChatMessage {
            message: Message::user("x".repeat(TEXT_BUDGET * 2)),
        });
        let parsed = Trace::from_jsonl(&test_trace(span).to_jsonl().unwrap()).unwrap();
        let span = &parsed.spans[0];
        assert!(!span.complete);
        let Some(SpanEvent::ChatMessage { message }) = span.events.last() else {
            panic!("expected a chat edit");
        };
        assert!(message.text_content().contains("truncated fnv1a="));
    }

    #[test]
    fn jsonl_writes_the_canonical_header_span_and_footer_bytes() {
        let mut trace = test_trace(test_span("Zürich\nquoted \"value\""));
        trace.outcome = Some(TraceOutcome {
            output: Some(serde_json::Map::from_iter([(
                "answer".to_string(),
                serde_json::Value::String("東京".to_string()),
            )])),
            error: None,
            eval: Some(crate::Eval::score(1.0)),
            duration_us: 12,
        });
        let header = serde_json::to_string(&HeaderRef {
            h: &trace.meta,
            components: &trace.components,
            param_ids: &trace.param_ids,
            models: &trace.models,
            prefixes: &trace.prefixes,
        })
        .unwrap();
        let span = serde_json::to_string(&trace.spans[0]).unwrap();
        let footer = serde_json::to_string(trace.outcome.as_ref().unwrap()).unwrap();
        assert_eq!(
            trace.to_jsonl().unwrap(),
            format!("{header}\n{span}\n{{\"f\":{footer}}}\n")
        );
    }

    #[test]
    fn newer_version_is_rejected() {
        let mut trace = test_trace(test_span("hi"));
        trace.meta.v = 99;
        let jsonl = trace.to_jsonl().expect("serialize");
        let err = Trace::from_jsonl(&jsonl).expect_err("should reject v99");
        assert!(err.to_string().contains("newer than supported"));
    }

    #[test]
    fn span_eval_roundtrips_and_stays_off_the_wire_when_absent() {
        use crate::trace::span::Eval;

        // Absent span eval: the span line carries no `eval` key at all, so
        // pre-RFC-0004 traces and eval-free traces serialize byte-identically.
        let trace = test_trace(test_span("hi"));
        let jsonl = trace.to_jsonl().expect("serialize");
        let span_line = jsonl.lines().nth(1).expect("span line");
        assert!(!span_line.contains("\"eval\""));

        // Present span eval: survives a write/read cycle intact.
        let mut span = test_span("hi");
        span.eval = Some(Eval::with_feedback(0.25, "partial credit"));
        let trace = test_trace(span);
        let parsed = Trace::from_jsonl(&trace.to_jsonl().expect("serialize")).expect("deserialize");
        assert_eq!(
            parsed.spans[0].eval,
            Some(Eval::with_feedback(0.25, "partial credit"))
        );
    }

    #[test]
    fn unknown_event_tags_are_skipped_on_read() {
        let trace = test_trace(test_span("hi"));
        let mut jsonl = trace.to_jsonl().expect("serialize");
        // Simulate a newer writer emitting an unknown event tag.
        jsonl = jsonl.replace(
            "\"events\":[",
            "\"events\":[{\"t\":\"future_event\",\"payload\":1},",
        );
        let parsed = Trace::from_jsonl(&jsonl).expect("unknown tags should be skipped");
        assert_eq!(parsed.spans[0].events.len(), 1);
        assert!(matches!(
            parsed.spans[0].events[0],
            SpanEvent::Exchange { .. }
        ));
    }
}
