//! Offline three-stage HotpotQA reflection workload. See benchmarks/cross-framework/hotpotqa.
use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use dspy_rs::ir::SignatureDef;
use dspy_rs::{
    ChatAdapter, JsonMap, LM, LMClient, Message, Predict, Signature, TestCompletionModel,
};
use rig::{completion::AssistantContent, message::Text};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Signature, Clone, Debug)]
/// Answer the question using the supplied context.
struct Answer {
    #[input]
    question: String,
    #[input]
    context: String,
    #[output]
    answer: String,
}

#[derive(Signature, Clone, Debug)]
/// Critique the answer against the supplied context.
struct Reflect {
    #[input]
    question: String,
    #[input]
    context: String,
    #[input]
    answer: String,
    #[output]
    critique: String,
}

#[derive(Signature, Clone, Debug)]
/// Revise the answer using the critique and supplied context.
struct Revise {
    #[input]
    question: String,
    #[input]
    context: String,
    #[input]
    draft: String,
    #[input]
    critique: String,
    #[output]
    answer: String,
}

const CRITIQUE: &str = "The answer agrees with the supporting context.";
const FIXTURE: &str = include_str!("../../../benchmarks/cross-framework/hotpotqa/fixture.json");

#[derive(Deserialize)]
struct Fixture {
    examples: Vec<Example>,
}

#[derive(Deserialize)]
struct Example {
    id: String,
    question: String,
    context: String,
    answer: String,
}

fn response(field: &str, value: &str) -> String {
    format!("[[ ## {field} ## ]]\n{value}\n\n[[ ## completed ## ]]\n")
}

fn adapter_profile<S: Signature>(input: Value, raw: String, iterations: usize) -> Value {
    let input: JsonMap = serde_json::from_value(input).unwrap();
    let def = SignatureDef::of::<S>();
    let types = SignatureDef::types_of::<S>();
    let message = Message::new("assistant", &raw);
    let adapter = ChatAdapter;
    let mut format_ns = 0;
    let mut parse_ns = 0;
    for i in 0..iterations + 20 {
        let start = Instant::now();
        std::hint::black_box(adapter.build_system_def(def, types, None));
        std::hint::black_box(adapter.format_input_def(def, &input));
        let formatted = start.elapsed().as_nanos();
        let start = Instant::now();
        std::hint::black_box(adapter.parse_output_def(def, types, &message).unwrap());
        if i >= 20 {
            format_ns += formatted;
            parse_ns += start.elapsed().as_nanos();
        }
    }
    json!({"format_ns_per_call":format_ns as f64 / iterations as f64,
        "parse_ns_per_call":parse_ns as f64 / iterations as f64})
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let iterations: usize = args.first().map(|s| s.parse()).transpose()?.unwrap_or(200);
    let wait_ms: u64 = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(0);
    ensure!(iterations > 0, "iterations must be positive");
    let fixture: Fixture = serde_json::from_str(FIXTURE)?;
    if args.iter().any(|arg| arg == "--profile-adapter") {
        ensure!(wait_ms == 0, "adapter replay has no simulated waiting");
        let profiles: Vec<_> = fixture.examples.iter().map(|e| json!({"id":e.id,
            "answer":adapter_profile::<Answer>(json!({"question":e.question,"context":e.context}), response("answer", &e.answer),iterations),
            "reflect":adapter_profile::<Reflect>(json!({"question":e.question,"context":e.context,"answer":e.answer}),response("critique",CRITIQUE),iterations),
            "revise":adapter_profile::<Revise>(json!({"question":e.question,"context":e.context,"draft":e.answer,"critique":CRITIQUE}),response("answer",&e.answer),iterations)
        })).collect();
        println!(
            "{}",
            json!({"framework":"dsrs","kind":"isolated_adapter_replay","iterations":iterations,"warmup":20,"profiles":profiles})
        );
        return Ok(());
    }

    let replies = (0..iterations + 20).flat_map(|_| {
        fixture.examples.iter().flat_map(|e| {
            [
                response("answer", &e.answer),
                response("critique", CRITIQUE),
                response("answer", &e.answer),
            ]
            .map(|text| AssistantContent::Text(Text { text }))
        })
    });
    let lm = LM::builder()
        .model("openai:mock".to_string())
        .api_key("unused-offline".into())
        .cache(false)
        .max_retries(0)
        .build()
        .await?
        .with_client(LMClient::Test(TestCompletionModel::new(replies)))
        .await?;
    lm.history.set_capacity(0);
    dspy_rs::set_history_capacity(0);
    let answer = Predict::<Answer>::builder()
        .lm(lm.clone())
        .named("answer")
        .build();
    let reflect = Predict::<Reflect>::builder()
        .lm(lm.clone())
        .named("reflect")
        .build();
    let revise = Predict::<Revise>::builder().lm(lm).named("revise").build();
    answer.set_history_capacity(0);
    reflect.set_history_capacity(0);
    revise.set_history_capacity(0);

    let mut samples = Vec::new();
    for iteration in 0..iterations + 20 {
        for e in &fixture.examples {
            let pipeline_start = Instant::now();
            let mut stage_ns = [0; 3];
            let mut wait_ns = 0;
            let start = Instant::now();
            if wait_ms > 0 {
                tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                wait_ns += start.elapsed().as_nanos();
            }
            let first = answer
                .call(AnswerInput {
                    question: e.question.clone(),
                    context: e.context.clone(),
                })
                .await?
                .into_inner();
            stage_ns[0] = start.elapsed().as_nanos();
            ensure!(
                first.answer == e.answer,
                "initial answer mismatch for {}",
                e.id
            );
            let start = Instant::now();
            if wait_ms > 0 {
                tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                wait_ns += start.elapsed().as_nanos();
            }
            let feedback = reflect
                .call(ReflectInput {
                    question: e.question.clone(),
                    context: e.context.clone(),
                    answer: first.answer.clone(),
                })
                .await?
                .into_inner();
            stage_ns[1] = start.elapsed().as_nanos();
            ensure!(
                feedback.critique == CRITIQUE,
                "critique mismatch for {}",
                e.id
            );
            let start = Instant::now();
            if wait_ms > 0 {
                tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                wait_ns += start.elapsed().as_nanos();
            }
            let final_answer = revise
                .call(ReviseInput {
                    question: e.question.clone(),
                    context: e.context.clone(),
                    draft: first.answer,
                    critique: feedback.critique,
                })
                .await?
                .into_inner();
            stage_ns[2] = start.elapsed().as_nanos();
            ensure!(
                final_answer.answer == e.answer,
                "revised answer mismatch for {}",
                e.id
            );
            if iteration >= 20 {
                samples.push(json!({"id":e.id,"answer":final_answer.answer,"pipeline_ns":pipeline_start.elapsed().as_nanos() as u64,
                    "stage_ns":stage_ns.map(|n|n as u64),"simulated_wait_ns":wait_ns as u64}));
            }
        }
    }
    println!(
        "{}",
        json!({"framework":"dsrs","kind":"pipeline_timing","iterations":iterations,"warmup":20,
        "simulated_wait_ms_per_stage":wait_ms,"cache":false,"history":false,"samples":samples})
    );
    Ok(())
}
