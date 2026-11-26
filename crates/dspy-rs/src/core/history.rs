//! Bounded execution history, independent of response caching and trace capture.
use std::collections::VecDeque;
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use super::storage::{StoreSession, StoreWrite, current_stores, start_write};
use crate::trace::{JsonMap, SpanError, SpanErrorKind, SpanEvent};
use crate::{Chat, LMConfig, LMResponse, LmUsage, Message};

pub const DEFAULT_HISTORY_CAPACITY: usize = 100;

/// One logical LM invocation, including cache hits and failed calls.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub id: String,
    #[serde(default)]
    pub trace_id: Option<String>,
    pub timestamp_us: u64,
    pub duration_us: u64,
    pub model: String,
    /// Generation settings with credentials removed and endpoint reduced to origin.
    pub config: LMConfig,
    pub messages: Vec<Message>,
    /// JSON rendering of the input messages, for compatibility with cache inspection.
    pub prompt: String,
    pub response: Option<Message>,
    pub raw_output: Option<String>,
    pub output: Option<JsonMap>,
    /// None when usage could not be recovered after a failed/cancelled call.
    pub usage: Option<LmUsage>,
    /// None when pricing is unavailable; token usage is not treated as cost.
    pub cost: Option<f64>,
    pub cache_hit: bool,
    pub module_path: Vec<String>,
    pub events: Vec<SpanEvent>,
    pub error: Option<SpanError>,
}

/// Shared live record. Parsed output is attached by the interpreter after the LM returns.
#[derive(Clone, Debug)]
pub struct HistoryRecord {
    entry: Arc<Mutex<HistoryEntry>>,
}

impl HistoryRecord {
    fn persist(&self, stores: Vec<Arc<StoreSession>>) -> StoreWrite<()> {
        if stores.is_empty() {
            return StoreWrite::Ready(());
        }
        let record = self.clone();
        let snapshot = OnceLock::new();
        start_write(stores, (), move |store, ()| {
            store.persist(snapshot.get_or_init(|| record.snapshot()));
        })
    }
    /// The invocation ID used to join prediction metadata to this record.
    pub fn id(&self) -> String {
        self.entry
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .id
            .clone()
    }

    pub fn snapshot(&self) -> HistoryEntry {
        let mut entry = self.entry.lock().unwrap_or_else(|p| p.into_inner());
        // Even an empty chat renders to "[]". Materialize once, on inspection
        // or persistence, from the messages captured at invocation time.
        if entry.prompt.is_empty() {
            entry.prompt = Chat::messages_json(&entry.messages).to_string();
        }
        entry.clone()
    }

    pub(crate) async fn parsed<T, E: std::fmt::Display>(&self, result: &Result<(JsonMap, T), E>) {
        {
            let mut entry = self.entry.lock().unwrap_or_else(|p| p.into_inner());
            match result {
                Ok((output, _)) => {
                    entry.output = Some(output.clone());
                    entry.error = None;
                }
                Err(error) => {
                    entry.error = Some(SpanError {
                        kind: SpanErrorKind::Parse,
                        message: error.to_string(),
                    })
                }
            }
        }
        self.persist(current_stores()).finish().await;
    }
}

#[derive(Debug)]
struct Window {
    capacity: usize,
    entries: VecDeque<HistoryRecord>,
}

/// A bounded history store. Clones share the same store; snapshots are newest first.
#[derive(Clone, Debug)]
pub struct HistoryBuffer(Arc<Mutex<Window>>);

impl Default for HistoryBuffer {
    fn default() -> Self {
        Self::new(DEFAULT_HISTORY_CAPACITY)
    }
}

impl HistoryBuffer {
    pub fn new(capacity: usize) -> Self {
        Self(Arc::new(Mutex::new(Window {
            capacity,
            entries: VecDeque::new(),
        })))
    }

    pub fn set_capacity(&self, capacity: usize) {
        let mut window = self.0.lock().unwrap_or_else(|p| p.into_inner());
        window.capacity = capacity;
        while window.entries.len() > capacity {
            window.entries.pop_front();
        }
    }

    pub fn clear(&self) {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entries
            .clear();
    }

    fn is_enabled(&self) -> bool {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).capacity != 0
    }

    pub fn inspect(&self, n: usize) -> Vec<HistoryEntry> {
        let records = self
            .0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entries
            .iter()
            .rev()
            .take(n)
            .cloned()
            .collect::<Vec<_>>();
        records.iter().map(HistoryRecord::snapshot).collect()
    }

    fn push(&self, record: HistoryRecord) {
        let mut window = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if window.capacity == 0 {
            return;
        }
        window.entries.push_back(record);
        while window.entries.len() > window.capacity {
            window.entries.pop_front();
        }
    }

    /// Collects LM invocations made by this future and its nested scopes.
    /// Spawned tasks do not inherit the caller stack.
    pub async fn scope<F: std::future::Future>(
        &self,
        name: impl Into<String>,
        future: F,
    ) -> F::Output {
        let mut scopes = CALLERS.try_with(Clone::clone).unwrap_or_default();
        scopes.push(Caller {
            name: name.into(),
            buffer: self.clone(),
        });
        CALLERS.scope(scopes, future).await
    }
}

#[derive(Clone)]
struct Caller {
    name: String,
    buffer: HistoryBuffer,
}
tokio::task_local! { static CALLERS: Vec<Caller>; }
static GLOBAL: LazyLock<HistoryBuffer> = LazyLock::new(HistoryBuffer::default);

pub fn inspect_history(n: usize) -> Vec<HistoryEntry> {
    GLOBAL.inspect(n)
}
pub fn clear_history() {
    GLOBAL.clear();
}
pub fn set_history_capacity(capacity: usize) {
    GLOBAL.set_capacity(capacity);
}

#[derive(Default)]
pub(crate) struct CallProgress {
    pub(crate) events: Vec<SpanEvent>,
    usage: Option<LmUsage>,
}

impl CallProgress {
    pub(crate) fn observe_usage(&mut self, usage: LmUsage) {
        self.usage = Some(self.total_usage() + usage);
    }

    pub(crate) fn total_usage(&self) -> LmUsage {
        self.usage.unwrap_or_default()
    }

    fn record_error(&mut self, entry: &mut HistoryEntry, error: SpanError) {
        entry.events = std::mem::take(&mut self.events);
        entry.usage = self.usage;
        entry.error = Some(error);
    }
}

struct HistoryCapture {
    record: HistoryRecord,
    stores: Vec<Arc<StoreSession>>,
    sinks: Vec<HistoryBuffer>,
    started: Instant,
}

impl HistoryCapture {
    fn publish(self, update: impl FnOnce(&mut HistoryEntry, &HistoryRecord)) -> StoreWrite<()> {
        {
            let mut entry = self.record.entry.lock().unwrap_or_else(|p| p.into_inner());
            update(&mut entry, &self.record);
            entry.duration_us = self.started.elapsed().as_micros() as u64;
        }
        for sink in self.sinks {
            sink.push(self.record.clone());
        }
        self.record.persist(self.stores)
    }
}

pub(crate) struct HistoryCall {
    pub(crate) progress: CallProgress,
    capture: Option<HistoryCapture>,
}

impl HistoryCall {
    pub(crate) fn begin(config: &LMConfig, chat: &Chat, lm_history: &HistoryBuffer) -> Self {
        let stores = current_stores();
        let enabled = !stores.is_empty()
            || lm_history.is_enabled()
            || GLOBAL.is_enabled()
            || CALLERS
                .try_with(|callers| callers.iter().any(|caller| caller.buffer.is_enabled()))
                .unwrap_or(false);
        if !enabled {
            return Self {
                progress: CallProgress::default(),
                capture: None,
            };
        }
        let callers = CALLERS.try_with(Clone::clone).unwrap_or_default();
        let record = HistoryRecord {
            entry: Arc::new(Mutex::new(HistoryEntry {
                id: crate::trace::capture::mint_trace_id(),
                trace_id: crate::trace::capture::current_trace_id(),
                timestamp_us: crate::trace::capture::now_us(),
                duration_us: 0,
                model: config.model.clone(),
                config: crate::trace::ModelEntry::redacted_config(config),
                messages: chat.messages.clone(),
                prompt: String::new(),
                response: None,
                raw_output: None,
                output: None,
                usage: None,
                cost: None,
                cache_hit: false,
                module_path: callers.iter().map(|c| c.name.clone()).collect(),
                events: Vec::new(),
                error: None,
            })),
        };
        let mut sinks = vec![lm_history.clone(), GLOBAL.clone()];
        for caller in callers {
            if !sinks
                .iter()
                .any(|sink| Arc::ptr_eq(&sink.0, &caller.buffer.0))
            {
                sinks.push(caller.buffer);
            }
        }
        Self {
            progress: CallProgress::default(),
            capture: Some(HistoryCapture {
                record,
                stores,
                sinks,
                started: Instant::now(),
            }),
        }
    }

    pub(crate) async fn finish(mut self, result: &mut anyhow::Result<LMResponse>) {
        if let Some(capture) = self.capture.take() {
            let write = capture.publish(|entry, record| match result {
                Ok(response) => {
                    entry.response = response
                        .events
                        .iter()
                        .rev()
                        .find_map(|event| match event {
                            SpanEvent::Exchange { message, .. } => Some(message.clone()),
                            _ => None,
                        })
                        .or_else(|| Some(response.output.clone()));
                    entry.raw_output = entry.response.as_ref().map(Message::content);
                    entry.cache_hit = response.cache_hit;
                    entry.usage = Some(if response.cache_hit {
                        LmUsage::default()
                    } else {
                        response.usage
                    });
                    entry.events = response.events.clone();
                    response.history = Some(record.clone());
                }
                Err(error) => self.progress.record_error(
                    entry,
                    SpanError {
                        kind: SpanErrorKind::Lm,
                        message: error.to_string(),
                    },
                ),
            });
            write.finish().await;
        }
    }
}

impl Drop for HistoryCall {
    fn drop(&mut self) {
        if let Some(capture) = self.capture.take() {
            let write = capture.publish(|entry, _| {
                self.progress.record_error(
                    entry,
                    SpanError {
                        kind: SpanErrorKind::Cancelled,
                        message: "LM invocation ended before completion".into(),
                    },
                );
            });
            // A dropped call still submits its record. The scope drains it.
            drop(write);
        }
    }
}
