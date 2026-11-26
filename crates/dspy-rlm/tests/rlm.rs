use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use dspy_rlm::Rlm;
use dspy_rs::{
    CompletionProvider, HistoryEntry, InteractionStore, LM, LMClient, PredictError, Signature,
    StorageError, TestCompletionModel, Trace, with_interaction_store,
};
use rig::completion::{
    AssistantContent, CompletionError, CompletionRequest, CompletionResponse, Usage,
};
use tokio::sync::Barrier;

#[dspy_rs::Schema]
#[derive(Debug, Clone)]
struct Paper {
    title: String,
    body: String,
}

/// Find papers relevant to the search query.
#[derive(Debug, Clone, Signature)]
struct Search {
    #[input]
    papers: Vec<Paper>,
    #[input]
    query: String,
    #[output]
    #[alias("selected_titles")]
    titles: Vec<String>,
}

async fn test_lm(replies: &[&str]) -> (LM, TestCompletionModel) {
    let provider = TestCompletionModel::new(replies.iter().copied().map(AssistantContent::text));
    provider.set_usage(Usage {
        input_tokens: 2,
        output_tokens: 1,
        total_tokens: 3,
        ..Usage::new()
    });
    let lm = LM::builder()
        .base_url("http://127.0.0.1:1/v1".into())
        .model("rlm-test".into())
        .build()
        .await
        .unwrap()
        .with_client(LMClient::Test(provider.clone()))
        .await
        .unwrap();
    (lm, provider)
}

fn input(title: &str) -> SearchInput {
    SearchInput {
        papers: vec![Paper {
            title: title.into(),
            body: "abstract not requested by the model".repeat(2_000),
        }],
        query: title.to_owned(),
    }
}

struct BatchedClient {
    provider: TestCompletionModel,
    started: Barrier,
}

impl CompletionProvider for BatchedClient {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<()>, CompletionError> {
        let response = self.provider.completion(request).await;
        self.started.wait().await;
        response
    }
}

#[tokio::test]
async fn keeps_large_inputs_in_python_and_preserves_session_across_cells() {
    let (lm, provider) = test_lm(&[
        "```python\nmatches = [p['title'] for p in papers if query in p['title']]\n```\n```py\nmatches\n```",
        "SUBMIT(selected_titles=matches)",
    ])
    .await;
    let prediction = Rlm::<Search>::builder()
        .lm(lm)
        .build()
        .call(input("Rust"))
        .await
        .unwrap();

    assert_eq!(prediction.titles, ["Rust"]);
    assert_eq!(prediction.metadata().lm_usage.total_tokens, 6);
    assert!(prediction.metadata().field_meta().contains_key("titles"));
    let prompt = format!("{:?}", provider.last_request().unwrap());
    assert!(prompt.contains("list of 1 items"));
    assert!(prompt.contains("['Rust']"));
    assert!(!prompt.contains("abstract not requested by the model"));
}

#[tokio::test]
async fn reused_action_prompts_preserve_instance_instructions_and_fresh_inputs() {
    let (lm, provider) = test_lm(&["SUBMIT(selected_titles=[query])"; 3]).await;
    let lm = Arc::new(lm);
    let first = Rlm::<Search>::builder()
        .lm(Arc::clone(&lm))
        .instruction("Use the primary selection policy.")
        .build();
    let second = Rlm::<Search>::builder()
        .lm(lm)
        .instruction("Use the alternate selection policy.")
        .build();

    let prediction = first.call(input("Rust")).await.unwrap();
    assert_eq!(prediction.titles, ["Rust"]);
    let first_preamble = provider.last_request().unwrap().preamble.unwrap();
    assert!(first_preamble.starts_with("Use the primary selection policy."));

    let prediction = second.call(input("Python")).await.unwrap();
    assert_eq!(prediction.titles, ["Python"]);
    let second_preamble = provider.last_request().unwrap().preamble.unwrap();
    assert!(second_preamble.starts_with("Use the alternate selection policy."));

    let prediction = first.call(input("Zig")).await.unwrap();
    assert_eq!(prediction.titles, ["Zig"]);
    assert_eq!(
        provider.last_request().unwrap().preamble.as_deref(),
        Some(first_preamble.as_str())
    );
    assert_eq!(provider.call_count(), 3);
}

#[tokio::test]
async fn batches_submodel_calls_and_keeps_the_prefix_that_fits_the_budget() {
    let (action_lm, action_provider) = test_lm(&[
        "notes = llm_query_batched(['first', 'second', 'over budget'])",
        "SUBMIT(selected_titles=notes)",
    ])
    .await;
    let (sub_lm, sub_provider) = test_lm(&["first result", "second result"]).await;
    let sub_lm = sub_lm
        .with_client(LMClient::custom(BatchedClient {
            provider: sub_provider.clone(),
            started: Barrier::new(2),
        }))
        .await
        .unwrap();
    let rlm = Rlm::<Search>::builder()
        .lm(action_lm)
        .sub_lm(sub_lm)
        .max_llm_calls(2)
        .build();
    let prediction = tokio::time::timeout(Duration::from_secs(10), rlm.call(input("Rust")))
        .await
        .expect("the Python call completes on a single-thread runtime")
        .unwrap();

    assert_eq!(prediction.titles, ["first result", "second result"]);
    assert_eq!(action_provider.call_count(), 2);
    assert_eq!(sub_provider.call_count(), 2);
    assert_eq!(prediction.metadata().lm_usage.total_tokens, 12);
    assert!(
        format!("{:?}", action_provider.last_request().unwrap())
            .contains("Submodel calls remaining: 0")
    );
}

#[tokio::test]
async fn feeds_invalid_submissions_back_to_the_model_and_preserves_variables() {
    let (lm, provider) = test_lm(&[
        "saved = [papers[0]['title']]\nSUBMIT(unknown=saved)",
        "SUBMIT(selected_titles=saved)",
    ])
    .await;
    let prediction = Rlm::<Search>::builder()
        .lm(lm)
        .build()
        .call(input("Rust"))
        .await
        .unwrap();

    assert_eq!(prediction.titles, ["Rust"]);
    let prompt = format!("{:?}", provider.last_request().unwrap());
    assert!(prompt.contains("SUBMIT rejected"));
    assert!(prompt.contains("not found in response"));
}

#[test]
fn stored_submodel_calls_complete_on_a_single_blocking_worker() {
    #[derive(Debug, Default)]
    struct InvocationCount(AtomicUsize);

    impl InteractionStore for InvocationCount {
        fn upsert(&self, _: &HistoryEntry) -> Result<(), StorageError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn get(&self, _: &str) -> Result<Option<HistoryEntry>, StorageError> {
            unreachable!()
        }

        fn recent(&self, _: usize) -> Result<Vec<HistoryEntry>, StorageError> {
            unreachable!()
        }

        fn upsert_trace(&self, _: &Trace) -> Result<(), StorageError> {
            unreachable!()
        }

        fn get_trace(&self, _: &str) -> Result<Option<Trace>, StorageError> {
            unreachable!()
        }
    }

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .max_blocking_threads(1)
        .build()
        .unwrap()
        .block_on(async {
            let (lm, _) = test_lm(&["SUBMIT(selected_titles=[llm_query(query)])"]).await;
            let (sub_lm, _) = test_lm(&["Rust"]).await;
            let rlm = Rlm::<Search>::builder().lm(lm).sub_lm(sub_lm).build();
            let store = Arc::new(InvocationCount::default());
            let prediction = tokio::time::timeout(
                Duration::from_secs(10),
                with_interaction_store(store.clone(), rlm.call(input("Rust"))),
            )
            .await
            .expect("the nested call can persist while Python holds the blocking worker")
            .unwrap()
            .unwrap();

            assert_eq!(prediction.titles, ["Rust"]);
            assert_eq!(store.0.load(Ordering::Relaxed), 2);
        });
}

#[tokio::test]
async fn finalizes_through_predict_using_the_execution_transcript() {
    let (lm, provider) = test_lm(&[
        "print('selected: ' + papers[0]['title'])",
        "[[ ## selected_titles ## ]]\n[\"Rust\"]\n[[ ## completed ## ]]",
    ])
    .await;
    let prediction = Rlm::<Search>::builder()
        .lm(lm)
        .max_iterations(1)
        .build()
        .call(input("Rust"))
        .await
        .unwrap();

    assert_eq!(prediction.titles, ["Rust"]);
    assert_eq!(prediction.metadata().lm_usage.total_tokens, 6);
    let prompt = format!("{:?}", provider.last_request().unwrap());
    assert!(prompt.contains("selected: Rust"));
    assert!(prompt.contains("execution transcript"));
}

#[tokio::test]
async fn stops_at_the_iteration_limit_when_final_extraction_is_disabled() {
    let (lm, provider) = test_lm(&["print('still working')"]).await;
    let error = match Rlm::<Search>::builder()
        .lm(lm)
        .max_iterations(1)
        .enable_extraction_fallback(false)
        .build()
        .call(input("Rust"))
        .await
    {
        Ok(_) => panic!("expected iteration limit error"),
        Err(error) => error,
    };

    let PredictError::Parse {
        source, lm_usage, ..
    } = error
    else {
        panic!("expected a prediction parse error");
    };
    assert!(source.to_string().contains("reached 1 iterations"));
    assert_eq!(lm_usage.total_tokens, 3);
    assert_eq!(provider.call_count(), 1);
}
