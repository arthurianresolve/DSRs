//! Schedules durable writes without occupying workers while a backend is busy.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, LazyLock, Mutex, Weak};

use futures::future::join_all;
use tokio::runtime::{Builder, Handle, Runtime};
use tokio::task::JoinHandle;

use super::{InteractionStore, StoreSession};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum StoreKey {
    Backend(u64),
    Instance(usize),
}

#[derive(Debug)]
pub(super) struct WriteQueue {
    key: StoreKey,
    turn: tokio::sync::Mutex<()>,
}

static QUEUES: LazyLock<Mutex<HashMap<StoreKey, Weak<WriteQueue>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

impl WriteQueue {
    pub(super) fn for_store(store: &Arc<dyn InteractionStore>) -> Arc<Self> {
        let key = match store.write_affinity() {
            Some(key) => StoreKey::Backend(key),
            None => StoreKey::Instance(Arc::as_ptr(store).cast::<()>() as usize),
        };
        let mut queues = QUEUES.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(queue) = queues.get(&key).and_then(Weak::upgrade) {
            return queue;
        }
        let queue = Arc::new(Self {
            key,
            turn: tokio::sync::Mutex::new(()),
        });
        queues.insert(key, Arc::downgrade(&queue));
        queue
    }
}

impl Drop for WriteQueue {
    fn drop(&mut self) {
        let mut queues = QUEUES.lock().unwrap_or_else(|p| p.into_inner());
        if queues
            .get(&self.key)
            .is_some_and(|queue| std::ptr::eq(queue.as_ptr(), self))
        {
            queues.remove(&self.key);
        }
    }
}

/// The scope drains writes even when an invocation drops its completion handle.
struct PendingWrites(Vec<Arc<StoreSession>>);

impl PendingWrites {
    fn new(stores: Vec<Arc<StoreSession>>) -> Self {
        for session in &stores {
            session.pending.fetch_add(1, Ordering::Relaxed);
        }
        Self(stores)
    }
}

impl Drop for PendingWrites {
    fn drop(&mut self) {
        for session in &self.0 {
            if session.pending.fetch_sub(1, Ordering::AcqRel) == 1 {
                session.drained.notify_one();
            }
        }
    }
}

pub(crate) enum StoreWrite<T> {
    Ready(T),
    Pending(JoinHandle<T>),
}

impl<T> StoreWrite<T> {
    pub(crate) async fn finish(self) -> T {
        match self {
            Self::Ready(value) => value,
            Self::Pending(task) => task.await.expect("interaction storage task failed"),
        }
    }
}

// Both admission and I/O survive cancellation on the caller's runtime.
static WRITER: LazyLock<std::io::Result<Runtime>> = LazyLock::new(|| {
    Builder::new_multi_thread()
        .worker_threads(1)
        .max_blocking_threads(4)
        .thread_name("dsrs-store-writer")
        .build()
});

pub(crate) fn start_write<T: Send + Sync + 'static>(
    stores: Vec<Arc<StoreSession>>,
    value: T,
    write: impl Fn(&StoreSession, &T) + Send + Sync + 'static,
) -> StoreWrite<T> {
    let inline = || {
        for store in &stores {
            write(store, &value);
        }
    };
    let Ok(caller) = Handle::try_current() else {
        inline();
        return StoreWrite::Ready(value);
    };
    let writer = match &*WRITER {
        Ok(writer) => writer,
        Err(error) => {
            tracing::error!(%error, "interaction writer unavailable");
            inline();
            return StoreWrite::Ready(value);
        }
    };
    let pending = PendingWrites::new(stores);
    StoreWrite::Pending(writer.spawn(async move {
        let value = Arc::new(value);
        let write = Arc::new(write);
        let jobs = pending.0.iter().map(|store| {
            let value = Arc::clone(&value);
            let write = Arc::clone(&write);
            let caller = caller.clone();
            async move {
                let _turn = store.queue.turn.lock().await;
                let store = Arc::clone(store);
                tokio::task::spawn_blocking(move || {
                    let _caller = caller.enter();
                    write(&store, &value);
                })
                .await
            }
        });
        for result in join_all(jobs).await {
            result.expect("interaction storage write failed");
        }
        Arc::into_inner(value).expect("all interaction storage writes have finished")
    }))
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use crate::core::storage::{JsonlInteractionStore, current_stores, with_interaction_store};
    use std::sync::mpsc;
    use std::time::Duration;
    use tokio::sync::Notify;

    #[tokio::test]
    async fn a_busy_journal_leaves_workers_available_for_another_journal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("busy.jsonl");
        let alias = dir.path().join("busy-alias.jsonl");
        let busy: Arc<dyn InteractionStore> = Arc::new(JsonlInteractionStore::open(&path).unwrap());
        std::fs::hard_link(path, &alias).unwrap();
        let same: Arc<dyn InteractionStore> = Arc::new(JsonlInteractionStore::open(alias).unwrap());
        let other: Arc<dyn InteractionStore> =
            Arc::new(JsonlInteractionStore::open(dir.path().join("other.jsonl")).unwrap());
        assert!(Arc::ptr_eq(
            &WriteQueue::for_store(&busy),
            &WriteQueue::for_store(&same),
        ));

        let (release, wait) = mpsc::channel::<()>();
        let blocked = Arc::new((Notify::new(), Mutex::new(wait)));
        let queued = |store| {
            with_interaction_store(store, async {
                let writes: Vec<_> = (0..4)
                    .map(|_| {
                        start_write(current_stores(), Arc::clone(&blocked), |_, blocked| {
                            blocked.0.notify_one();
                            let _ = blocked.1.lock().unwrap().recv();
                        })
                    })
                    .collect();
                for write in writes {
                    write.finish().await;
                }
            })
        };
        let mut first = std::pin::pin!(queued(busy));
        let mut alias = std::pin::pin!(queued(same));
        assert!(futures::poll!(&mut first).is_pending());
        assert!(futures::poll!(&mut alias).is_pending());
        let deadline = Duration::from_secs(5);
        tokio::time::timeout(deadline, blocked.0.notified())
            .await
            .unwrap();
        let progress = tokio::time::timeout(
            deadline,
            with_interaction_store(other, async {
                start_write(current_stores(), 42, |_, _| {}).finish().await
            }),
        )
        .await;
        drop(release);
        first.await.unwrap();
        alias.await.unwrap();
        assert_eq!(
            progress.expect("independent journal has a worker").unwrap(),
            42
        );
    }
}
