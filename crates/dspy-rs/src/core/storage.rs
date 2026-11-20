//! Persistent invocation storage, independent of caching and history retention.
use crate::HistoryEntry;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::fs::{File, OpenOptions};
use std::future::Future;
use std::io::{BufRead, BufReader, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

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

/// Parsing upserts the same invocation ID rather than creating another call.
pub trait InteractionStore: std::fmt::Debug + Send + Sync {
    fn upsert(&self, entry: &HistoryEntry) -> Result<(), StorageError>;
    fn get(&self, id: &str) -> Result<Option<HistoryEntry>, StorageError>;
    /// Newest invocations first, with at most one revision per ID.
    fn recent(&self, limit: usize) -> Result<Vec<HistoryEntry>, StorageError>;
    fn upsert_trace(&self, trace: &crate::Trace) -> Result<(), StorageError>;
    fn get_trace(&self, id: &str) -> Result<Option<crate::Trace>, StorageError>;
}

/// Versioned, append-only JSONL with durable updates and OS file locking.
/// Each handle validates and indexes complete records on open and as the journal
/// grows. A torn final line is repaired on append. Close all handles before
/// editing or replacing existing records.
/// Contains prompts/responses: choose a path with appropriate access permissions.
#[derive(Debug)]
pub struct JsonlInteractionStore {
    journal: Mutex<Journal>,
}

#[derive(Serialize, Deserialize)]
struct Revision {
    version: u32,
    #[serde(flatten)]
    record: Record,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum Record {
    Invocation { entry: HistoryEntry },
    Execution { trace: crate::Trace },
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
    let revision: Revision = serde_json::from_slice(line)?;
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
                    self.record(decode_record(&line)?, location);
                }
                self.complete += line.len() as u64;
            }
        }
        self.modified = modified;
        Ok(len)
    }

    fn record(&mut self, record: Record, location: RecordLocation) {
        match record {
            Record::Invocation { entry } => {
                let timestamp = entry.timestamp_us;
                let id = entry.id;
                if let Some((old, _)) = self.invocations.insert(id.clone(), (timestamp, location)) {
                    if old == timestamp {
                        return;
                    }
                    self.newest.remove(&(old, id.clone()));
                }
                self.newest.insert((timestamp, id));
            }
            Record::Execution { trace } => {
                self.traces.insert(trace.meta.trace_id, location);
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
        let store = Self {
            journal: Mutex::new(Journal {
                file,
                index: JournalIndex::default(),
            }),
        };
        store.locked(|_, _, _| Ok(()))?;
        Ok(store)
    }

    fn locked<T>(
        &self,
        f: impl FnOnce(&mut File, &JournalIndex, u64) -> Result<T, StorageError>,
    ) -> Result<T, StorageError> {
        let mut journal = self.journal.lock().unwrap_or_else(|p| p.into_inner());
        let Journal { file, index } = &mut *journal;
        file.lock()?;
        let guard = LockedFile(file);
        let len = index.refresh(guard.0)?;
        f(guard.0, index, len)
    }

    fn append(&self, record: Record) -> Result<(), StorageError> {
        let mut bytes = serde_json::to_vec(&Revision { version: 1, record })?;
        bytes.push(b'\n');
        self.locked(|file, index, len| {
            if len > index.complete {
                file.set_len(index.complete)?;
            }
            file.seek(SeekFrom::Start(index.complete))?;
            file.write_all(&bytes)?;
            file.sync_data()?;
            Ok(())
        })
    }
}

impl InteractionStore for JsonlInteractionStore {
    fn upsert(&self, entry: &HistoryEntry) -> Result<(), StorageError> {
        self.append(Record::Invocation {
            entry: entry.clone(),
        })
    }
    fn upsert_trace(&self, trace: &crate::Trace) -> Result<(), StorageError> {
        self.append(Record::Execution {
            trace: trace.clone(),
        })
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
    store: Arc<dyn InteractionStore>,
    errors: Mutex<Vec<StorageError>>,
}
impl StoreSession {
    fn persist_trace(&self, trace: &crate::Trace) {
        if let Err(error) = self.store.upsert_trace(trace) {
            tracing::error!(%error, trace_id = %trace.meta.trace_id, "trace persistence failed");
            self.errors
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(error);
        }
    }
    pub(crate) fn persist(&self, entry: &HistoryEntry) {
        if let Err(error) = self.store.upsert(entry) {
            tracing::error!(%error, call_id = %entry.id, "interaction persistence failed");
            self.errors
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(error);
        }
    }
}
tokio::task_local! { static STORES: Vec<Arc<StoreSession>>; }
pub(crate) fn current_stores() -> Vec<Arc<StoreSession>> {
    STORES.try_with(Clone::clone).unwrap_or_default()
}

pub(crate) fn persist_trace(trace: &crate::Trace) {
    for session in current_stores() {
        session.persist_trace(trace);
    }
}

/// Persist calls, cache hits, failures, cancellation and subsequent parse results.
/// History capacity may be zero. Nested scopes compose; spawned tasks need their
/// own scope. Storage errors are returned after the future finishes (provider
/// errors remain inside T). Writes are synchronous and durable before return.
pub async fn with_interaction_store<T>(
    store: Arc<dyn InteractionStore>,
    future: impl Future<Output = T>,
) -> Result<T, StorageError> {
    let session = Arc::new(StoreSession {
        store,
        errors: Mutex::new(Vec::new()),
    });
    let mut stores = current_stores();
    stores.push(session.clone());
    let output = STORES.scope(stores, future).await;
    let mut errors = session.errors.lock().unwrap_or_else(|p| p.into_inner());
    if errors.is_empty() {
        Ok(output)
    } else {
        let additional = errors.len() - 1;
        Err(StorageError::Session {
            source: Box::new(errors.remove(0)),
            additional,
        })
    }
}
