// These tests exercise native file locks and durable filesystem I/O.
#![cfg(not(miri))]

use dspy_rs::{
    Chat, InteractionStore, JsonlInteractionStore, LM, LMClient, Message, Module, Predict,
    Signature, TestCompletionModel, with_interaction_store,
};
use rig::completion::AssistantContent;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Arc;

#[derive(Signature, Clone, Debug)]
/// Answer the question.
struct QA {
    #[input]
    question: String,
    #[output]
    answer: String,
}

async fn lm(responses: Vec<AssistantContent>) -> LM {
    LM::builder()
        .base_url("http://127.0.0.1:1/v1".into())
        .model("storage".into())
        .cache(false)
        .build()
        .await
        .unwrap()
        .with_client(LMClient::Test(TestCompletionModel::new(responses)))
        .await
        .unwrap()
}

fn answer(value: &str) -> AssistantContent {
    AssistantContent::text(format!(
        "[[ ## answer ## ]]\n{value}\n[[ ## completed ## ]]"
    ))
}

#[tokio::test]
async fn reopens_complete_typed_records_with_history_disabled() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("interactions.jsonl");
    let store = Arc::new(JsonlInteractionStore::open(&path).unwrap());
    let lm = lm(vec![answer("Paris")]).await;
    lm.history.set_capacity(0);
    dspy_rs::set_history_capacity(0);
    let predict = Predict::<QA>::builder().lm(lm).build();
    predict.set_history_capacity(0);
    let output = with_interaction_store(
        store.clone(),
        predict.call(QAInput {
            question: "France?".into(),
        }),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(output.answer, "Paris");
    let records = store.recent(10).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].output.as_ref().unwrap()["answer"], "Paris");
    assert!(records[0].error.is_none());
    let id = records[0].id.clone();
    assert_eq!(output.metadata().call_ids, [id.clone()]);
    drop(store);
    let reopened = JsonlInteractionStore::open(&path).unwrap();
    assert_eq!(
        reopened.get(&id).unwrap().unwrap().output.unwrap()["answer"],
        "Paris"
    );
}

#[tokio::test]
async fn failed_parsing_updates_the_same_record() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(JsonlInteractionStore::open(dir.path().join("calls.jsonl")).unwrap());
    let predict = Predict::<QA>::new().with_lm(lm(vec![AssistantContent::text("invalid")]).await);
    assert!(
        with_interaction_store(
            store.clone(),
            predict.call(QAInput {
                question: "?".into()
            })
        )
        .await
        .unwrap()
        .is_err()
    );
    let records = store.recent(10).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(
        records[0].error.as_ref().unwrap().kind,
        dspy_rs::SpanErrorKind::Parse
    );
}

#[tokio::test]
async fn torn_tail_is_recovered_but_corruption_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("calls.jsonl");
    let store = Arc::new(JsonlInteractionStore::open(&path).unwrap());
    let lm = lm(vec![answer("one"), answer("two")]).await;
    let call = || lm.call(Chat::new(vec![Message::user("question")]), vec![]);
    with_interaction_store(store.clone(), call())
        .await
        .unwrap()
        .unwrap();
    OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"{torn")
        .unwrap();
    with_interaction_store(store.clone(), call())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        JsonlInteractionStore::open(&path)
            .unwrap()
            .recent(10)
            .unwrap()
            .len(),
        2
    );
    OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"bad record\n")
        .unwrap();
    assert!(store.recent(0).is_err());
    assert!(JsonlInteractionStore::open(&path).is_err());
}

#[test]
fn invalid_revision_leaves_the_journal_readable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trace.jsonl");
    let store = JsonlInteractionStore::open(&path).unwrap();
    let mut trace = dspy_rs::Trace::default();
    trace.meta.trace_id = "saved".into();
    store.upsert_trace(&trace).unwrap();
    let previous = std::fs::read(&path).unwrap();

    trace.meta.trace_id = "invalid-score".into();
    trace.outcome = Some(dspy_rs::TraceOutcome {
        eval: Some(dspy_rs::Eval::score(f64::NAN)),
        duration_us: 0,
        output: None,
        error: None,
    });
    assert!(matches!(
        store.upsert_trace(&trace),
        Err(dspy_rs::StorageError::Json(_))
    ));
    assert_eq!(std::fs::read(&path).unwrap(), previous);
    assert!(store.get_trace("invalid-score").unwrap().is_none());
    trace.outcome = None;
    store.upsert_trace(&trace).unwrap();
    let reopened = JsonlInteractionStore::open(&path).unwrap();
    assert!(reopened.get_trace("saved").unwrap().is_some());
    assert!(reopened.get_trace("invalid-score").unwrap().is_some());
}

#[tokio::test]
async fn independent_handles_do_not_lose_concurrent_updates() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("calls.jsonl");
    let a = Arc::new(JsonlInteractionStore::open(&path).unwrap());
    let b = Arc::new(JsonlInteractionStore::open(&path).unwrap());
    let lm = lm(vec![answer("a"), answer("b")]).await;
    let (one, two) = tokio::join!(
        with_interaction_store(
            a.clone(),
            lm.call(Chat::new(vec![Message::user("a")]), vec![])
        ),
        with_interaction_store(b, lm.call(Chat::new(vec![Message::user("b")]), vec![]))
    );
    one.unwrap().unwrap();
    two.unwrap().unwrap();
    let records = a.recent(10).unwrap();
    assert_eq!(records.len(), 2);
    assert_ne!(records[0].id, records[1].id);
    let barrier = Arc::new(std::sync::Barrier::new(7));
    let writers = (0..6)
        .map(|_| JsonlInteractionStore::open(&path))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    std::thread::scope(|scope| {
        for (index, writer) in writers.into_iter().enumerate() {
            let mut entry = records[0].clone();
            entry.id = format!("{}-writer-{index}", entry.id);
            let barrier = barrier.clone();
            scope.spawn(move || {
                barrier.wait();
                writer.upsert(&entry).unwrap();
            });
        }
        barrier.wait();
    });
    assert_eq!(a.recent(20).unwrap().len(), 8);
}

#[tokio::test]
async fn open_handles_read_the_latest_revision_and_reorder_recent_entries() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("calls.jsonl");
    let reader = JsonlInteractionStore::open(&path).unwrap();
    let writer = Arc::new(JsonlInteractionStore::open(&path).unwrap());
    let lm = lm(vec![answer("a")]).await;
    with_interaction_store(
        writer.clone(),
        lm.call(Chat::new(vec![Message::user("a")]), vec![]),
    )
    .await
    .unwrap()
    .unwrap();
    let mut entry = reader.recent(1).unwrap().pop().unwrap();
    let original = entry.id.clone();

    entry.id.push_str("-new");
    writer.upsert(&entry).unwrap();
    assert_eq!(reader.recent(1).unwrap()[0].id, entry.id);
    let newer = entry.id.clone();

    entry.id = original.clone();
    entry.timestamp_us += 1;
    entry.raw_output = Some("revised".into());
    writer.upsert(&entry).unwrap();
    assert_eq!(
        reader
            .get(&original)
            .unwrap()
            .unwrap()
            .raw_output
            .as_deref(),
        Some("revised")
    );
    let ids: Vec<_> = reader
        .recent(10)
        .unwrap()
        .into_iter()
        .map(|record| record.id)
        .collect();
    assert_eq!(ids, [original, newer]);

    let (_, mut trace) = dspy_rs::capture(|| async {}).await;
    let trace_id = trace.meta.trace_id.clone();
    writer.upsert_trace(&trace).unwrap();
    assert!(
        reader
            .get_trace(&trace_id)
            .unwrap()
            .unwrap()
            .meta
            .tags
            .is_empty()
    );
    trace.meta.tags.insert("status".into(), "updated".into());
    writer.upsert_trace(&trace).unwrap();
    assert_eq!(
        reader.get_trace(&trace_id).unwrap().unwrap().meta.tags,
        trace.meta.tags
    );
}

#[derive(Debug)]
struct Broken;
impl InteractionStore for Broken {
    fn upsert_trace(&self, _: &dspy_rs::Trace) -> Result<(), dspy_rs::StorageError> {
        Err(std::io::Error::other("disk full").into())
    }
    fn get_trace(&self, _: &str) -> Result<Option<dspy_rs::Trace>, dspy_rs::StorageError> {
        unreachable!()
    }
    fn upsert(&self, _: &dspy_rs::HistoryEntry) -> Result<(), dspy_rs::StorageError> {
        Err(std::io::Error::other("disk full").into())
    }
    fn get(&self, _: &str) -> Result<Option<dspy_rs::HistoryEntry>, dspy_rs::StorageError> {
        unreachable!()
    }
    fn recent(&self, _: usize) -> Result<Vec<dspy_rs::HistoryEntry>, dspy_rs::StorageError> {
        unreachable!()
    }
}

#[derive(Debug)]
struct WaitingStore {
    started: std::sync::Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    release: std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
}

impl InteractionStore for WaitingStore {
    fn upsert(&self, _: &dspy_rs::HistoryEntry) -> Result<(), dspy_rs::StorageError> {
        self.started
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .send(())
            .unwrap();
        self.release
            .lock()
            .unwrap()
            .recv_timeout(std::time::Duration::from_secs(5))
            .map_err(std::io::Error::other)?;
        Ok(())
    }
    fn get(&self, _: &str) -> Result<Option<dspy_rs::HistoryEntry>, dspy_rs::StorageError> {
        unreachable!()
    }
    fn recent(&self, _: usize) -> Result<Vec<dspy_rs::HistoryEntry>, dspy_rs::StorageError> {
        unreachable!()
    }
    fn upsert_trace(&self, _: &dspy_rs::Trace) -> Result<(), dspy_rs::StorageError> {
        unreachable!()
    }
    fn get_trace(&self, _: &str) -> Result<Option<dspy_rs::Trace>, dspy_rs::StorageError> {
        unreachable!()
    }
}

#[rstest::rstest]
#[case(false)]
#[case(true)]
#[tokio::test(flavor = "current_thread")]
async fn pending_storage_io_allows_other_futures_to_progress(#[case] cancel: bool) {
    let lm = if cancel {
        lm(vec![])
            .await
            .with_client(LMClient::custom(Pending))
            .await
            .unwrap()
    } else {
        lm(vec![answer("Paris")]).await
    };
    let (started, waiting) = tokio::sync::oneshot::channel();
    let (release, released) = std::sync::mpsc::channel();
    let store = Arc::new(WaitingStore {
        started: std::sync::Mutex::new(Some(started)),
        release: std::sync::Mutex::new(released),
    });
    let scope = with_interaction_store(store, async {
        let mut call = std::pin::pin!(lm.call(Chat::new(vec![Message::user("?")]), vec![]));
        if cancel {
            assert!(futures::poll!(&mut call).is_pending());
        } else {
            call.await.unwrap();
        }
    });
    let mut scope = std::pin::pin!(scope);
    // The scope must be pending even when the LM call was already dropped.
    assert!(futures::poll!(&mut scope).is_pending());
    let (result, ()) = tokio::join!(scope, async {
        tokio::time::timeout(std::time::Duration::from_secs(5), waiting)
            .await
            .unwrap()
            .unwrap();
        release.send(()).unwrap();
    });
    result.unwrap();
}

#[tokio::test]
async fn captured_traces_are_reopened_and_joined_to_call_ids() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("calls.jsonl");
    let store = Arc::new(JsonlInteractionStore::open(&path).unwrap());
    let predict = Predict::<QA>::new().with_lm(lm(vec![answer("Paris")]).await);
    let (result, trace) = with_interaction_store(
        store.clone(),
        Box::pin(with_interaction_store(
            store.clone(),
            dspy_rs::capture(|| {
                predict.call(QAInput {
                    question: "France?".into(),
                })
            }),
        )),
    )
    .await
    .unwrap()
    .unwrap();
    let journal = std::fs::read_to_string(&path).unwrap();
    assert_eq!(
        journal.lines().count(),
        3,
        "one call, parse update, and trace"
    );
    let id = result.unwrap().metadata().call_ids[0].clone();
    let handle = Arc::downgrade(&store);
    drop(store);
    assert!(handle.upgrade().is_none(), "history retained the store");
    let reopened = JsonlInteractionStore::open(&path).unwrap();
    let entry = reopened.get(&id).unwrap().unwrap();
    assert_eq!(
        entry.trace_id.as_deref(),
        Some(trace.meta.trace_id.as_str())
    );
    let mut saved = reopened.get_trace(&trace.meta.trace_id).unwrap().unwrap();
    assert_eq!(saved.spans.len(), 1);
    assert_eq!(saved.spans[0].output.as_ref().unwrap()["answer"], "Paris");

    saved.meta.v = 1;
    saved.spans[0].events.push(dspy_rs::SpanEvent::ChatMessage {
        message: Message::user("one more question"),
    });
    reopened.upsert_trace(&saved).unwrap();
    let journal = std::fs::read_to_string(&path).unwrap();
    let last: serde_json::Value = serde_json::from_str(journal.lines().last().unwrap()).unwrap();
    let portable = dspy_rs::Trace::from_jsonl(&saved.to_jsonl().unwrap()).unwrap();
    assert_eq!(portable.meta.v, 2);
    assert_eq!(last["trace"]["meta"]["v"], portable.meta.v);
}

#[tokio::test]
async fn persistence_failure_is_returned_to_the_caller() {
    let lm = lm(vec![answer("Paris")]).await;
    let result = with_interaction_store(
        Arc::new(Broken),
        lm.call(Chat::new(vec![Message::user("?")]), vec![]),
    )
    .await;
    let error = result.unwrap_err();
    assert!(error.to_string().contains("disk full"));
    let cause = std::error::Error::source(&error).unwrap();
    assert!(
        cause
            .source()
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .is_some()
    );
}

#[tokio::test]
async fn persistence_keeps_the_first_error_and_counts_later_failures() {
    let result = with_interaction_store(Arc::new(Broken), async {
        for _ in 0..5 {
            dspy_rs::capture(|| async {}).await;
        }
    })
    .await
    .unwrap_err();
    assert!(matches!(
        result,
        dspy_rs::StorageError::Session { source, additional: 4 }
            if source.to_string().contains("disk full")
    ));
}

#[tokio::test]
async fn nested_scopes_sharing_a_store_both_receive_write_failures() {
    let store = Arc::new(Broken);
    let check = |error| {
        assert!(matches!(
            error,
            dspy_rs::StorageError::Session { source, additional: 0 }
                if matches!(*source, dspy_rs::StorageError::Io(ref io) if io.to_string() == "disk full")
        ));
    };
    let outer = with_interaction_store(store.clone(), async {
        let inner = with_interaction_store(store, dspy_rs::capture(|| async {})).await;
        check(inner.unwrap_err());
    })
    .await;
    check(outer.unwrap_err());
}

#[tokio::test]
async fn v1_journal_keeps_legacy_envelope_precedence() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("calls.jsonl");
    let store = Arc::new(JsonlInteractionStore::open(&path).unwrap());
    let lm = lm(vec![answer("red")]).await;
    with_interaction_store(
        store.clone(),
        lm.call(Chat::new(vec![Message::user("?")]), vec![]),
    )
    .await
    .unwrap()
    .unwrap();
    let entry = store.recent(1).unwrap().pop().unwrap();
    drop(store);

    let mut trace = dspy_rs::Trace::default();
    trace.meta.trace_id = "legacy-trace".into();
    let trace = serde_json::to_string(&trace).unwrap();
    let record = format!(
        "{{\"version\":1,\"trace\":{trace},\"entry\":{},\"extra\":true}}\n",
        serde_json::to_string(&entry).unwrap()
    );
    let with_reordered_trace =
        format!("{record}{{\"extra\":true,\"trace\":{trace},\"version\":1}}\n");
    std::fs::write(&path, record).unwrap();
    let reader = JsonlInteractionStore::open(&path).unwrap();
    assert_eq!(reader.recent(1).unwrap()[0].id, entry.id);
    assert!(reader.get_trace("legacy-trace").unwrap().is_none());
    drop(reader);

    std::fs::write(&path, with_reordered_trace).unwrap();
    let reader = JsonlInteractionStore::open(&path).unwrap();
    assert!(reader.get_trace("legacy-trace").unwrap().is_some());
}

struct Pending;
impl dspy_rs::CompletionProvider for Pending {
    async fn completion(
        &self,
        _: rig::completion::CompletionRequest,
    ) -> Result<rig::completion::CompletionResponse<()>, rig::completion::CompletionError> {
        std::future::pending().await
    }
}

#[tokio::test]
async fn cancellation_and_provider_failures_are_persistent() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(JsonlInteractionStore::open(dir.path().join("calls.jsonl")).unwrap());
    let broken = lm(vec![]).await;
    let pending = lm(vec![])
        .await
        .with_client(LMClient::custom(Pending))
        .await
        .unwrap();
    let chat = || Chat::new(vec![Message::user("?")]);
    with_interaction_store(store.clone(), async {
        assert!(broken.call(chat(), vec![]).await.is_err());
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_millis(10),
                pending.call(chat(), vec![])
            )
            .await
            .is_err()
        );
    })
    .await
    .unwrap();
    let records = store.recent(10).unwrap();
    assert_eq!(records.len(), 2);
    assert!(
        records
            .iter()
            .any(|r| r.error.as_ref().unwrap().kind == dspy_rs::SpanErrorKind::Cancelled)
    );
    assert!(
        records
            .iter()
            .any(|r| r.error.as_ref().unwrap().kind == dspy_rs::SpanErrorKind::Lm)
    );
}

#[cfg(feature = "cache")]
#[tokio::test]
async fn cache_hits_have_separate_ids_and_generation_settings_miss() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(JsonlInteractionStore::open(dir.path().join("calls.jsonl")).unwrap());
    let provider = TestCompletionModel::new([answer("a"), answer("b")]);
    let mut lm = LM::builder()
        .base_url("http://127.0.0.1:1/v1".into())
        .model("storage-cache".into())
        .cache(true)
        .build()
        .await
        .unwrap()
        .with_client(LMClient::Test(provider.clone()))
        .await
        .unwrap();
    with_interaction_store(store.clone(), async {
        for _ in 0..2 {
            lm.call(Chat::new(vec![Message::user("same")]), vec![])
                .await
                .unwrap();
        }
        lm.config.temperature = 0.2;
        lm.call(Chat::new(vec![Message::user("same")]), vec![])
            .await
            .unwrap();
    })
    .await
    .unwrap();
    let records = store.recent(10).unwrap();
    assert_eq!(records.len(), 3);
    assert_eq!(records.iter().filter(|r| r.cache_hit).count(), 1);
    assert_eq!(provider.call_count(), 2);
    assert_eq!(records[0].config.temperature, 0.2);
}
