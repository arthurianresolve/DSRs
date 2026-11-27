//! Persistent invocation storage, independent of caching and history retention.
use crate::HistoryEntry;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::fs::{File, OpenOptions};
use std::future::Future;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{BufRead, BufReader, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use tokio::sync::Notify;

mod writes;
use writes::WriteQueue;
pub(crate) use writes::{StoreWrite, start_write};

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("interaction storage I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid interaction journal: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported interaction journal version {0}")]
    Version(u32),
    #[error("interaction persistence failed: {source} ({additional} additional failures)")]
    Session {
        #[source]
        source: Box<StorageError>,
        additional: usize,
    },
}

impl StorageError {
    // Each scope returns an owned error. The first subscriber receives the
    // original, and nested subscribers receive its kind and diagnostic.
    fn copy_for_scope(&self) -> Self {
        match self {
            Self::Io(error) => Self::Io(match error.raw_os_error() {
                Some(code) => std::io::Error::from_raw_os_error(code),
                None => std::io::Error::new(error.kind(), error.to_string()),
            }),
            Self::Json(error) => Self::Json(serde::de::Error::custom(error)),
            Self::Version(version) => Self::Version(*version),
            Self::Session { source, additional } => Self::Session {
                source: Box::new(source.copy_for_scope()),
                additional: *additional,
            },
        }
    }
}

/// Parsing upserts the same invocation ID rather than creating another call.
pub trait InteractionStore: std::fmt::Debug + Send + Sync {
    fn upsert(&self, entry: &HistoryEntry) -> Result<(), StorageError>;
    fn get(&self, id: &str) -> Result<Option<HistoryEntry>, StorageError>;
    /// Newest invocations first, with at most one revision per ID.
    fn recent(&self, limit: usize) -> Result<Vec<HistoryEntry>, StorageError>;
    fn upsert_trace(&self, trace: &crate::Trace) -> Result<(), StorageError>;
    fn get_trace(&self, id: &str) -> Result<Option<crate::Trace>, StorageError>;

    /// A process-local key grouping handles that write to the same backend.
    /// The default groups writes by the shared store instance.
    #[doc(hidden)]
    fn write_affinity(&self) -> Option<u64> {
        None
    }
}

/// Versioned, append-only JSONL with durable updates and OS file locking.
/// Each handle validates and indexes complete records on open and as the journal
/// grows. A torn final line is repaired on append. Close all handles before
/// editing or replacing existing records.
/// Contains prompts/responses: choose a path with appropriate access permissions.
#[derive(Debug)]
pub struct JsonlInteractionStore {
    journal: Mutex<Journal>,
    write_affinity: Option<u64>,
}

#[derive(Serialize, Deserialize)]
struct Revision<E, T> {
    version: u32,
    #[serde(flatten)]
    record: Record<E, T>,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum Record<E = HistoryEntry, T = crate::Trace> {
    Invocation { entry: E },
    Execution { trace: T },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InvocationRevision {
    version: u32,
    entry: HistoryEntry,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TraceRevision {
    version: u32,
    trace: crate::Trace,
}

enum RecordKey {
    Invocation { id: String, timestamp: u64 },
    Execution { id: String },
}

impl Record {
    fn into_key(self) -> RecordKey {
        match self {
            Self::Invocation { entry } => RecordKey::Invocation {
                id: entry.id,
                timestamp: entry.timestamp_us,
            },
            Self::Execution { trace } => RecordKey::Execution {
                id: trace.meta.trace_id,
            },
        }
    }
}

struct LockedFile<'a>(&'a mut File);
impl Drop for LockedFile<'_> {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

#[derive(Debug)]
struct Journal {
    file: File,
    index: JournalIndex,
}

#[derive(Debug, Default)]
struct JournalIndex {
    // The offset just after the last validated newline. A partial or failed
    // append leaves this boundary available for the next refresh.
    complete: u64,
    modified: Option<SystemTime>,
    invocations: HashMap<String, (u64, RecordLocation)>,
    newest: BTreeSet<(u64, String)>,
    traces: HashMap<String, RecordLocation>,
}

#[derive(Clone, Copy, Debug)]
struct RecordLocation {
    offset: u64,
    len: usize,
}

fn decode_record(line: &[u8]) -> Result<Record, StorageError> {
    // The writer emits these envelopes in a fixed order. A direct decode
    // avoids buffering their payloads as content for the untagged enum.
    if line.starts_with(b"{\"version\":1,\"entry\":")
        && let Ok(InvocationRevision { version: 1, entry }) = serde_json::from_slice(line)
    {
        return Ok(Record::Invocation { entry });
    }
    if line.starts_with(b"{\"version\":1,\"trace\":")
        && let Ok(TraceRevision { version: 1, trace }) = serde_json::from_slice(line)
    {
        return Ok(Record::Execution { trace });
    }
    // Older or externally written envelopes keep the existing decode rules.
    let revision: Revision<HistoryEntry, crate::Trace> = serde_json::from_slice(line)?;
    if revision.version != 1 {
        return Err(StorageError::Version(revision.version));
    }
    Ok(revision.record)
}

impl RecordLocation {
    fn read(self, file: &mut File) -> Result<Record, StorageError> {
        file.seek(SeekFrom::Start(self.offset))?;
        let mut line = vec![0; self.len];
        file.read_exact(&mut line)?;
        decode_record(&line)
    }

    fn invocation(self, file: &mut File, id: &str) -> Result<HistoryEntry, StorageError> {
        match self.read(file)? {
            Record::Invocation { entry } if entry.id == id => Ok(entry),
            _ => Err(changed_record()),
        }
    }
}

fn changed_record() -> StorageError {
    std::io::Error::new(ErrorKind::InvalidData, "indexed interaction record changed").into()
}

impl JournalIndex {
    fn refresh(&mut self, file: &mut File) -> Result<u64, StorageError> {
        let meta = file.metadata()?;
        let len = meta.len();
        let modified = meta.modified().ok();
        if len < self.complete || (len == self.complete && modified != self.modified) {
            *self = Self::default();
        }
        if len > self.complete {
            file.seek(SeekFrom::Start(self.complete))?;
            let mut reader = BufReader::new(file);
            let mut line = Vec::new();
            loop {
                line.clear();
                reader.read_until(b'\n', &mut line)?;
                if line.last() != Some(&b'\n') {
                    break;
                }
                if line.len() > 1 {
                    let location = RecordLocation {
                        offset: self.complete,
                        len: line.len(),
                    };
                    self.record(decode_record(&line)?.into_key(), location);
                }
                self.complete += line.len() as u64;
            }
        }
        self.modified = modified;
        Ok(len)
    }

    fn record(&mut self, key: RecordKey, location: RecordLocation) {
        match key {
            RecordKey::Invocation { id, timestamp } => {
                if let Some((old, _)) = self.invocations.insert(id.clone(), (timestamp, location)) {
                    if old == timestamp {
                        return;
                    }
                    self.newest.remove(&(old, id.clone()));
                }
                self.newest.insert((timestamp, id));
            }
            RecordKey::Execution { id } => {
                self.traces.insert(id, location);
            }
        }
    }
}

impl JsonlInteractionStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        // Use the open file's identity so aliases and independently opened
        // handles share admission without retaining another file descriptor.
        let write_affinity = file
            .try_clone()
            .and_then(same_file::Handle::from_file)
            .ok()
            .map(|handle| {
                let mut hasher = DefaultHasher::new();
                handle.hash(&mut hasher);
                hasher.finish()
            });
        let store = Self {
            journal: Mutex::new(Journal {
                file,
                index: JournalIndex::default(),
            }),
            write_affinity,
        };
        store.locked(|_, _, _| Ok(()))?;
        Ok(store)
    }

    fn locked<T>(
        &self,
        f: impl FnOnce(&mut File, &mut JournalIndex, u64) -> Result<T, StorageError>,
    ) -> Result<T, StorageError> {
        let mut journal = self.journal.lock().unwrap_or_else(|p| p.into_inner());
        let Journal { file, index } = &mut *journal;
        file.lock()?;
        let guard = LockedFile(file);
        let len = index.refresh(guard.0)?;
        f(guard.0, index, len)
    }

    fn append(&self, record: Record<&HistoryEntry, &crate::Trace>) -> Result<(), StorageError> {
        let mut bytes = serde_json::to_vec(&Revision { version: 1, record })?;
        bytes.push(b'\n');
        // Validate the encoded revision before it can change the journal.
        let key = decode_record(&bytes)?.into_key();
        self.locked(|file, index, len| {
            if len > index.complete {
                file.set_len(index.complete)?;
            }
            file.seek(SeekFrom::Start(index.complete))?;
            file.write_all(&bytes)?;
            file.sync_data()?;
            index.record(
                key,
                RecordLocation {
                    offset: index.complete,
                    len: bytes.len(),
                },
            );
            index.complete += bytes.len() as u64;
            index.modified = file.metadata().and_then(|meta| meta.modified()).ok();
            Ok(())
        })
    }
}

impl InteractionStore for JsonlInteractionStore {
    fn write_affinity(&self) -> Option<u64> {
        self.write_affinity
    }

    fn upsert(&self, entry: &HistoryEntry) -> Result<(), StorageError> {
        self.append(Record::Invocation { entry })
    }
    fn upsert_trace(&self, trace: &crate::Trace) -> Result<(), StorageError> {
        self.append(Record::Execution { trace })
    }
    fn get_trace(&self, id: &str) -> Result<Option<crate::Trace>, StorageError> {
        self.locked(|file, index, _| {
            index
                .traces
                .get(id)
                .map(|location| match location.read(file)? {
                    Record::Execution { trace } if trace.meta.trace_id == id => Ok(trace),
                    _ => Err(changed_record()),
                })
                .transpose()
        })
    }
    fn get(&self, id: &str) -> Result<Option<HistoryEntry>, StorageError> {
        self.locked(|file, index, _| {
            index
                .invocations
                .get(id)
                .map(|(_, location)| location.invocation(file, id))
                .transpose()
        })
    }
    fn recent(&self, limit: usize) -> Result<Vec<HistoryEntry>, StorageError> {
        self.locked(|file, index, _| {
            index
                .newest
                .iter()
                .rev()
                .take(limit)
                .map(|(_, id)| index.invocations[id].1.invocation(file, id))
                .collect()
        })
    }
}

#[derive(Debug)]
pub(crate) struct StoreSession {
    queue: Arc<WriteQueue>,
    store: Arc<dyn InteractionStore>,
    errors: Mutex<SessionErrors>,
    pending: AtomicUsize,
    drained: Notify,
}

#[derive(Debug, Default)]
struct SessionErrors {
    first: Option<StorageError>,
    additional: usize,
}

impl StoreSession {
    fn record_error(&self, error: StorageError) {
        let mut errors = self.errors.lock().unwrap_or_else(|p| p.into_inner());
        if errors.first.is_none() {
            errors.first = Some(error);
        } else {
            errors.additional += 1;
        }
    }

    async fn drain(&self) {
        while self.pending.load(Ordering::Acquire) != 0 {
            self.drained.notified().await;
        }
    }
}

tokio::task_local! { static STORES: Vec<Arc<StoreSession>>; }
pub(crate) fn current_stores() -> Vec<Arc<StoreSession>> {
    STORES.try_with(Clone::clone).unwrap_or_default()
}

pub(crate) async fn persist_trace(trace: crate::Trace) -> crate::Trace {
    let stores = current_stores();
    if stores.is_empty() {
        return trace;
    }
    start_write(stores, trace, |store, trace| {
        store.upsert_trace(trace).inspect_err(|error| {
            tracing::error!(%error, trace_id = %trace.meta.trace_id, "trace persistence failed");
        })
    })
    .finish()
    .await
}

/// Persist calls, cache hits, failures, cancellation and subsequent parse results.
/// History capacity may be zero. Nested scopes sharing a store instance share
/// each write. Spawned tasks need their own scope. Storage errors are returned
/// after the future finishes (provider errors remain inside T).
/// With Tokio, writes run on an independent blocking
/// pool with per-backend admission. A completed scope waits for durability,
/// including writes from dropped calls.
pub async fn with_interaction_store<T>(
    store: Arc<dyn InteractionStore>,
    future: impl Future<Output = T>,
) -> Result<T, StorageError> {
    let session = Arc::new(StoreSession {
        queue: WriteQueue::for_store(&store),
        store,
        errors: Mutex::default(),
        pending: AtomicUsize::new(0),
        drained: Notify::new(),
    });
    let mut stores = current_stores();
    stores.push(session.clone());
    let output = STORES.scope(stores, future).await;
    session.drain().await;
    let mut errors = session.errors.lock().unwrap_or_else(|p| p.into_inner());
    match errors.first.take() {
        Some(source) => Err(StorageError::Session {
            source: Box::new(source),
            additional: errors.additional,
        }),
        None => Ok(output),
    }
}
