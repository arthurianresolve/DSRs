use std::time::Duration;

use dspy_rs::{CallMetadata, Module, PredictError, Predicted, Schema, forward_all};
use tokio::time::sleep;

struct DelayEcho;

#[derive(Clone, Debug, PartialEq, Eq)]
#[Schema]
struct DelayInput {
    value: i64,
    delay_ms: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[Schema]
struct DelayOutput {
    value: i64,
}

impl Module for DelayEcho {
    type Input = DelayInput;
    type Output = DelayOutput;

    async fn forward(&self, input: Self::Input) -> Result<Predicted<Self::Output>, PredictError> {
        sleep(Duration::from_millis(input.delay_ms.max(0) as u64)).await;
        if input.value < 0 {
            return Err(PredictError::Params {
                name: input.value.to_string(),
                source: std::io::Error::other("fixture rejection").into(),
            });
        }
        Ok(Predicted::new(
            DelayOutput { value: input.value },
            CallMetadata::default(),
        ))
    }
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn forward_all_preserves_input_order() {
    let module = DelayEcho;
    let inputs = vec![
        DelayInput {
            value: 0,
            delay_ms: 60,
        },
        DelayInput {
            value: 1,
            delay_ms: 10,
        },
        DelayInput {
            value: 2,
            delay_ms: 40,
        },
        DelayInput {
            value: 3,
            delay_ms: 5,
        },
    ];

    let outcomes = forward_all(&module, inputs, 2).await;
    let outputs = outcomes
        .into_iter()
        .map(|outcome| outcome.expect("forward should succeed").into_inner().value)
        .collect::<Vec<_>>();

    assert_eq!(outputs, vec![0, 1, 2, 3]);
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn zero_concurrency_finishes_with_ordered_individual_results() {
    let (empty, results) = tokio::time::timeout(Duration::from_secs(5), async {
        let empty = forward_all(&DelayEcho, Vec::new(), 0).await;
        let inputs = [-1, 2, -3]
            .map(|value| DelayInput { value, delay_ms: 0 })
            .to_vec();
        (empty, forward_all(&DelayEcho, inputs, 0).await)
    })
    .await
    .unwrap();
    assert!(empty.is_empty());
    assert!(matches!(&results[0], Err(PredictError::Params { name, .. }) if name == "-1"));
    assert_eq!(results[1].as_ref().unwrap().value, 2);
    assert!(matches!(&results[2], Err(PredictError::Params { name, .. }) if name == "-3"));
}
