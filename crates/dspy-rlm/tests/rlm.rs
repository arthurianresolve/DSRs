use std::time::Duration;

use dspy_rlm::Rlm;
use dspy_rs::{CompletionProvider, LM, LMClient, PredictError, Signature, TestCompletionModel};
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
