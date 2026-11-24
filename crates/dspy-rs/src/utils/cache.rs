use std::sync::Arc;

#[cfg(feature = "inflight-dedup")]
use std::{
    collections::{HashMap, hash_map::Entry},
    sync::Mutex,
};

use anyhow::Result;
use foyer::{BlockEngineBuilder, DeviceBuilder, FsDeviceBuilder, HybridCache, HybridCacheBuilder};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;
#[cfg(feature = "inflight-dedup")]
use tokio::sync::watch;
use tracing::{debug, trace, warn};

use crate::LmUsage;

/// Disk-tier initialization and lookup failures, with the backend cause intact.
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("response cache {operation} failed: {source}")]
    Backend {
        operation: &'static str,
        #[source]
        source: anyhow::Error,
    },
}

impl CacheError {
    fn backend(operation: &'static str, source: impl Into<anyhow::Error>) -> Self {
        Self::Backend {
            operation,
            source: source.into(),
        }
    }
}

/// Response-cache key: a 64-bit hash over the prompt + generation parameters.
///
/// Hashed keys keep foyer lookups and disk serialization O(1) in prompt size.
/// The hash is process-stable, which matches the cache's lifetime (the disk
/// tier lives in a per-process temp directory). Keys are produced by
/// [`LM`](crate::LM) from the rendered chat — callers never build them by hand.
pub type CacheKey = u64;

#[cfg(feature = "inflight-dedup")]
type PendingResponses = Mutex<HashMap<CacheKey, watch::Sender<FlightCompletion>>>;

#[cfg(feature = "inflight-dedup")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlightCompletion {
    CheckCache,
    Uncacheable,
}

#[cfg(feature = "inflight-dedup")]
pub(crate) enum InflightResponse<'a> {
    Leader(PendingResponse<'a>),
    Follower(watch::Receiver<FlightCompletion>),
}

#[cfg(feature = "inflight-dedup")]
pub(crate) enum SharedResponse<'a> {
    Cached(String),
    Leader(PendingResponse<'a>),
    Independent,
}

#[cfg(feature = "inflight-dedup")]
pub(crate) struct PendingResponse<'a> {
    key: CacheKey,
    requests: &'a PendingResponses,
    completion: FlightCompletion,
}

#[cfg(feature = "inflight-dedup")]
impl PendingResponse<'_> {
    pub(crate) fn finish_uncached(mut self) {
        self.completion = FlightCompletion::Uncacheable;
    }
}

#[cfg(feature = "inflight-dedup")]
impl Drop for PendingResponse<'_> {
    fn drop(&mut self) {
        let sender = self
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&self.key);
        // Notify followers after the map has been unlocked. The default
        // completion is a channel close, including on failure or cancellation.
        if let Some(sender) = sender
            && self.completion == FlightCompletion::Uncacheable
        {
            sender.send_replace(self.completion);
        }
    }
}

const MEMORY_CAPACITY: usize = 256 * 1024 * 1024;
const DISK_CAPACITY: usize = 1024 * 1024 * 1024;

/// A cached prompt-response pair.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheEntry {
    /// The formatted prompt that was sent to the LM.
    pub prompt: String,
    /// Token usage recorded for the original (uncached) call.
    #[serde(default)]
    pub usage: LmUsage,
    /// The raw assistant text of the response, so [`LM::call`](crate::LM) can
    /// replay a cached completion through the normal parse path.
    #[serde(default)]
    pub raw_output: Option<String>,
}

/// Hybrid memory + disk LM response cache.
///
/// Uses [foyer](https://docs.rs/foyer) with 256MB memory and 1GB disk (in a
/// temp directory owned by the cache for its whole lifetime). If the disk
/// tier cannot be initialized, the cache degrades to memory-only with a
/// warning instead of panicking. Execution history is owned independently
/// by [`LM`](crate::LM).
///
/// All methods take `&self`: the foyer cache is internally synchronized.
///
/// Created automatically by [`LM`](crate::LM) — you don't construct this directly.
#[derive(Clone)]
pub struct ResponseCache {
    handler: HybridCache<CacheKey, CacheEntry>,
    #[cfg(feature = "inflight-dedup")]
    inflight: Arc<PendingResponses>,
    /// Keeps the disk tier's temp directory alive: `TempDir` deletes the
    /// directory on drop, so it must outlive every clone of the cache.
    /// `None` when running memory-only.
    _disk_dir: Option<Arc<TempDir>>,
}

impl ResponseCache {
    #[tracing::instrument(name = "dsrs.cache.new", level = "debug")]
    pub async fn new() -> Self {
        let (handler, disk_dir) = match Self::try_build_hybrid().await {
            Ok((handler, dir)) => (handler, Some(Arc::new(dir))),
            Err(error) => {
                warn!(
                    error = %error,
                    "disk cache tier unavailable; falling back to memory-only response cache"
                );
                (Self::build_memory_only().await, None)
            }
        };

        let cache = Self {
            handler,
            #[cfg(feature = "inflight-dedup")]
            inflight: Arc::default(),
            _disk_dir: disk_dir,
        };
        debug!(
            disk_tier = cache._disk_dir.is_some(),
            "response cache initialized"
        );
        cache
    }

    /// Builds the full memory + disk hybrid, returning the `TempDir` guard
    /// that must be held for as long as the cache lives.
    async fn try_build_hybrid() -> Result<(HybridCache<CacheKey, CacheEntry>, TempDir)> {
        let dir =
            tempfile::tempdir().map_err(|error| CacheError::backend("create directory", error))?;

        let device = FsDeviceBuilder::new(dir.path())
            .with_capacity(DISK_CAPACITY)
            .build()
            .map_err(|error| CacheError::backend("create device", error))?;

        let hybrid = HybridCacheBuilder::new()
            .memory(MEMORY_CAPACITY)
            .storage()
            .with_engine_config(BlockEngineBuilder::new(device))
            .build()
            .await
            .map_err(|error| CacheError::backend("initialize disk tier", error))?;
        Ok((hybrid, dir))
    }

    /// Memory-only fallback: foyer's storage phase defaults to a noop engine,
    /// which cannot fail to build (no I/O involved).
    async fn build_memory_only() -> HybridCache<CacheKey, CacheEntry> {
        HybridCacheBuilder::new()
            .memory(MEMORY_CAPACITY)
            .storage()
            .build()
            .await
            .expect("memory-only foyer cache construction cannot fail")
    }

    #[cfg(test)]
    pub(crate) async fn memory_only_for_test() -> Self {
        Self {
            handler: Self::build_memory_only().await,
            #[cfg(feature = "inflight-dedup")]
            inflight: Arc::default(),
            _disk_dir: None,
        }
    }

    #[cfg(feature = "inflight-dedup")]
    pub(crate) fn register_inflight(&self, key: CacheKey) -> InflightResponse<'_> {
        let mut requests = self
            .inflight
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match requests.entry(key) {
            Entry::Occupied(entry) => InflightResponse::Follower(entry.get().subscribe()),
            Entry::Vacant(entry) => {
                let (sender, _) = watch::channel(FlightCompletion::CheckCache);
                entry.insert(sender);
                InflightResponse::Leader(PendingResponse {
                    key,
                    requests: &self.inflight,
                    completion: FlightCompletion::CheckCache,
                })
            }
        }
    }

    #[cfg(feature = "inflight-dedup")]
    pub(crate) async fn shared_response(&self, key: CacheKey) -> Result<SharedResponse<'_>> {
        if let Some(output) = self.get_response(key).await? {
            return Ok(SharedResponse::Cached(output));
        }

        // A miss after the first flight permits one shared handoff. Callers
        // still make their own attempts if both flights fail to cache a value.
        for _ in 0..2 {
            match self.register_inflight(key) {
                InflightResponse::Leader(pending) => {
                    // Another flight may have filled the cache before registration.
                    return Ok(match self.get_response(key).await? {
                        Some(output) => SharedResponse::Cached(output),
                        None => SharedResponse::Leader(pending),
                    });
                }
                InflightResponse::Follower(mut finished) => {
                    let _ = finished.changed().await;
                    if *finished.borrow() == FlightCompletion::Uncacheable {
                        return Ok(SharedResponse::Independent);
                    }
                    if let Some(output) = self.get_response(key).await? {
                        return Ok(SharedResponse::Cached(output));
                    }
                }
            }
        }
        Ok(SharedResponse::Independent)
    }

    /// Fetches the full cached entry (including raw output) for a key.
    #[tracing::instrument(name = "dsrs.cache.get_entry", level = "trace", skip(self))]
    pub async fn get_entry(&self, key: CacheKey) -> Result<Option<CacheEntry>> {
        let value = self
            .handler
            .get(&key)
            .await
            .map_err(|error| CacheError::backend("read", error))?
            .map(|v| v.value().clone());
        trace!(hit = value.is_some(), "cache lookup complete");
        Ok(value)
    }

    /// Replays only the response, without copying the stored prompt and usage.
    #[tracing::instrument(name = "dsrs.cache.get_response", level = "trace", skip(self))]
    pub(crate) async fn get_response(&self, key: CacheKey) -> Result<Option<String>> {
        let response = self
            .handler
            .get(&key)
            .await
            .map_err(|error| CacheError::backend("read", error))?
            .and_then(|entry| entry.value().raw_output.clone());
        trace!(hit = response.is_some(), "cache lookup complete");
        Ok(response)
    }

    /// Inserts an entry synchronously — the direct path used by [`LM::call`](crate::LM).
    #[tracing::instrument(name = "dsrs.cache.insert_entry", level = "trace", skip(self, entry))]
    pub fn insert_entry(&self, key: CacheKey, entry: CacheEntry) {
        let prompt_len = entry.prompt.len();
        self.handler.insert(key, entry);
        trace!(prompt_len, "cache entry inserted");
    }
}

#[cfg(all(test, feature = "inflight-dedup"))]
mod tests {
    use super::*;
    use std::future::{Future, poll_fn};
    use std::pin::Pin;
    use std::task::Poll;
    use std::time::Duration;

    const KEY: CacheKey = 17;

    fn lead(cache: &ResponseCache) -> PendingResponse<'_> {
        let InflightResponse::Leader(pending) = cache.register_inflight(KEY) else {
            panic!("expected an available flight");
        };
        pending
    }

    async fn subscribed(cache: &ResponseCache, mut waiter: Pin<&mut impl Future>) {
        tokio::time::timeout(
            Duration::from_secs(5),
            poll_fn(|cx| {
                assert!(waiter.as_mut().poll(cx).is_pending());
                // The initial cache lookup can yield before registration.
                let flights = cache.inflight.lock().unwrap();
                if flights
                    .get(&KEY)
                    .is_some_and(|sender| sender.receiver_count() > 0)
                {
                    Poll::Ready(())
                } else {
                    Poll::Pending
                }
            }),
        )
        .await
        .expect("waiter should subscribe to the current flight");
    }

    #[tokio::test]
    async fn uncacheable_flight_releases_waiters_for_independent_calls() {
        let cache = ResponseCache::memory_only_for_test().await;
        let pending = lead(&cache);
        let mut waiter = std::pin::pin!(cache.shared_response(KEY));
        subscribed(&cache, waiter.as_mut()).await;
        pending.finish_uncached();
        assert!(matches!(waiter.await.unwrap(), SharedResponse::Independent));
    }

    #[tokio::test]
    async fn waiter_can_proceed_after_two_empty_flights() {
        let cache = ResponseCache::memory_only_for_test().await;
        let initial = lead(&cache);
        let mut waiter = std::pin::pin!(cache.shared_response(KEY));
        subscribed(&cache, waiter.as_mut()).await;
        drop(initial);

        let replacement = lead(&cache);
        subscribed(&cache, waiter.as_mut()).await;
        drop(replacement);
        assert!(matches!(waiter.await.unwrap(), SharedResponse::Independent));
    }
}
