//! Offline and live HotpotQA reflection workload. See benchmarks/cross-framework/hotpotqa.
use std::io::{BufRead, Write};
use std::num::NonZeroU32;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing_subscriber::{
    Layer,
    layer::{Context, SubscriberExt},
    registry::LookupSpan,
};

use anyhow::{Context as _, Result, ensure};
use dspy_rs::ir::SignatureDef;
use dspy_rs::{
    ChatAdapter, Demo, JsonMap, LM, LMClient, Message, Module, Predict, Signature,
    TestCompletionModel,
};
use rig::{completion::AssistantContent, message::Text};
use serde::Deserialize;
use serde_json::{Value, json};

#[path = "support/hotpotqa_retrieval.rs"]
mod retrieval;
#[path = "support/hotpotqa_scoring.rs"]
mod scoring;

#[derive(Signature, Clone, Debug)]
/// Answer the question using the supplied context. Put a short answer without explanation in the answer output field. Do not copy input fields.
struct Answer {
    #[input]
    question: String,
    #[input]
    context: String,
    #[output]
    answer: String,
}

#[derive(Signature, Clone, Debug)]
/// Critique the answer against the supplied context. Put one brief sentence in the critique output field. Do not copy input fields.
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
/// Revise the answer using the critique and supplied context. Put a short answer without explanation in the answer output field. Do not copy input fields.
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
const DEMONSTRATIONS: &str =
    include_str!("../../../benchmarks/cross-framework/hotpotqa/demonstrations.json");
const SUPPORTED_ANSWERS: &str =
    include_str!("../../../benchmarks/cross-framework/hotpotqa/supported-answer-annotations.json");

fn predictors(
    lm: &LM,
    demonstrations: &[&retrieval::Demonstration],
    extractive: bool,
) -> (Predict<Answer>, Predict<Reflect>, Predict<Revise>) {
    let mut answer = Predict::<Answer>::builder().lm(lm.clone()).named("answer");
    let mut reflect = Predict::<Reflect>::builder()
        .lm(lm.clone())
        .named("reflect");
    let mut revise = Predict::<Revise>::builder().lm(lm.clone()).named("revise");
    if extractive {
        let instruction = "For an office or position, give only the office title without its country or jurisdiction. Keep words integral to the title.";
        answer = answer.instruction(format!("{}\n\n{instruction}", Answer::instruction()));
        revise = revise.instruction(format!("{}\n\n{instruction}", Revise::instruction()));
    }
    for demo in demonstrations {
        answer = answer.demo(Demo::new(
            AnswerInput {
                question: demo.question.clone(),
                context: demo.context.clone(),
            },
            AnswerOutput {
                answer: demo.answer.clone(),
            },
        ));
        reflect = reflect.demo(Demo::new(
            ReflectInput {
                question: demo.question.clone(),
                context: demo.context.clone(),
                answer: demo.draft.clone(),
            },
            ReflectOutput {
                critique: demo.critique.clone(),
            },
        ));
        revise = revise.demo(Demo::new(
            ReviseInput {
                question: demo.question.clone(),
                context: demo.context.clone(),
                draft: demo.draft.clone(),
                critique: demo.critique.clone(),
            },
            ReviseOutput {
                answer: demo.answer.clone(),
            },
        ));
    }
    let predictors = (answer.build(), reflect.build(), revise.build());
    predictors.0.set_history_capacity(0);
    predictors.1.set_history_capacity(0);
    predictors.2.set_history_capacity(0);
    predictors
}

#[derive(Clone, Default)]
struct AdapterTimings {
    adapter: Arc<Mutex<Vec<(String, u64)>>>,
    provider: Arc<Mutex<Vec<u64>>>,
}
impl<S: tracing::Subscriber + for<'a> LookupSpan<'a>> Layer<S> for AdapterTimings {
    fn on_new_span(
        &self,
        _: &tracing::span::Attributes<'_>,
        id: &tracing::span::Id,
        ctx: Context<'_, S>,
    ) {
        if let Some(span) = ctx.span(id) {
            if span.metadata().name() == "dsrs.provider.completion" {
                span.extensions_mut().insert(Instant::now());
            }
        }
    }
    fn on_enter(&self, id: &tracing::span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            if span.metadata().name().starts_with("dsrs.adapter.") {
                span.extensions_mut().insert(Instant::now());
            }
        }
    }
    fn on_exit(&self, id: &tracing::span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            if span.metadata().name().starts_with("dsrs.adapter.") {
                if let Some(start) = span.extensions_mut().remove::<Instant>() {
                    self.adapter.lock().unwrap().push((
                        span.metadata().name().to_string(),
                        start.elapsed().as_nanos() as u64,
                    ));
                }
            }
        }
    }
    fn on_close(&self, id: tracing::span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(&id) {
            if span.metadata().name() == "dsrs.provider.completion" {
                if let Some(start) = span.extensions_mut().remove::<Instant>() {
                    self.provider
                        .lock()
                        .unwrap()
                        .push(start.elapsed().as_nanos() as u64);
                }
            }
        }
    }
}

fn barrier(label: &str) -> Result<()> {
    if std::env::var_os("DSRS_PROFILE_BARRIER").is_some() {
        println!("{label}");
        std::io::stdout().flush()?;
        let mut line = String::new();
        std::io::stdin().lock().read_line(&mut line)?;
        ensure!(line.trim() == "continue", "controller barrier closed");
    }
    Ok(())
}

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
async fn main() -> Result<ExitCode> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let iterations: usize = args.first().map(|s| s.parse()).transpose()?.unwrap_or(200);
    let wait_ms: u64 = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(0);
    let live = args.iter().any(|s| s == "--live");
    let instrumented = args.iter().any(|s| s == "--profile-runtime");
    let smoke = args.iter().any(|s| s == "--smoke");
    let quality_check = args.iter().any(|s| s == "--check-quality");
    let protocol_check = quality_check || args.iter().any(|s| s == "--check-protocol");
    let retrieval_recipe = args.iter().any(|s| s == "--retrieval-recipe");
    ensure!(
        !quality_check || live,
        "--check-quality requires a live provider"
    );
    ensure!(
        !quality_check || !args.iter().any(|s| s == "--profile-adapter"),
        "quality checks require pipeline execution, not isolated adapter replay"
    );
    let fixture_path = args
        .iter()
        .position(|s| s == "--fixture")
        .map(|index| {
            args.get(index + 1)
                .context("--fixture requires a JSON file path")
        })
        .transpose()?;
    let supported_answers_path = args
        .iter()
        .position(|s| s == "--supported-answers")
        .map(|index| {
            args.get(index + 1)
                .context("--supported-answers requires a JSON file path")
        })
        .transpose()?;
    ensure!(
        !(retrieval_recipe || fixture_path.is_some()) || protocol_check,
        "retrieval recipes and custom fixtures require --check-protocol or --check-quality; they change the workload"
    );
    ensure!(
        supported_answers_path.is_none() || protocol_check,
        "custom supported-answer annotations require --check-protocol or --check-quality"
    );
    ensure!(
        !retrieval_recipe || !args.iter().any(|s| s == "--profile-adapter"),
        "retrieval recipes apply to pipeline diagnostics, not isolated adapter replay"
    );
    let json_output = args.iter().any(|s| s == "--json-output");
    ensure!(
        !json_output || live,
        "JSON output mode requires a live provider"
    );
    ensure!(
        !(smoke && protocol_check),
        "choose smoke or a full protocol check"
    );
    ensure!(!smoke || live, "smoke mode requires a live provider");
    let warmup = if smoke || protocol_check {
        0
    } else if live {
        1
    } else {
        20
    };
    ensure!(
        !live || wait_ms == 0,
        "live profiling must not inject simulated waiting"
    );
    let adapter_timings = AdapterTimings::default();
    if instrumented || live {
        let layer = adapter_timings
            .clone()
            .with_filter(tracing_subscriber::filter::filter_fn(move |m| {
                (live && m.name() == "dsrs.provider.completion")
                    || (instrumented && m.name().starts_with("dsrs.adapter."))
            }));
        tracing::subscriber::set_global_default(tracing_subscriber::registry().with(layer))?;
    }
    ensure!(iterations > 0, "iterations must be positive");
    let fixture_text = fixture_path.map(std::fs::read_to_string).transpose()?;
    let supported_answers_text = supported_answers_path
        .map(std::fs::read_to_string)
        .transpose()?;
    let supported_answers = scoring::SupportedAnswerAnnotations::from_json(
        supported_answers_text
            .as_deref()
            .unwrap_or(SUPPORTED_ANSWERS),
    )?;
    let mut fixture: Fixture = serde_json::from_str(fixture_text.as_deref().unwrap_or(FIXTURE))?;
    ensure!(
        !fixture.examples.is_empty(),
        "fixture must contain at least one example"
    );
    let demonstrations: Vec<retrieval::Demonstration> = if retrieval_recipe {
        serde_json::from_str(DEMONSTRATIONS)?
    } else {
        Vec::new()
    };
    if smoke {
        fixture.examples.truncate(1);
    }
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
        return Ok(ExitCode::SUCCESS);
    }

    let replies = (0..iterations + warmup).flat_map(|_| {
        fixture.examples.iter().flat_map(|e| {
            [
                response("answer", &e.answer),
                response("critique", CRITIQUE),
                response("answer", &e.answer),
            ]
            .map(|text| AssistantContent::Text(Text { text }))
        })
    });
    let waits = adapter_timings.provider.clone();
    let base_url = std::env::var("DSRS_PROFILE_BASE_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:18084/v1".into());
    let model = std::env::var("DSRS_PROFILE_MODEL").unwrap_or_else(|_| "llama3.2:3b".into());
    let lm = if live {
        LM::builder()
            .base_url(base_url.clone())
            .model(model.clone())
            .temperature(0.0)
            .max_tokens(128)
            .cache(false)
            .max_retries(0)
            .build()
            .await?
            .with_custom_output_schema(json_output)
            .with_scalar_output_schema(false)
    } else {
        LM::builder()
            .model("openai:mock".to_string())
            .api_key("unused-offline".into())
            .cache(false)
            .max_retries(0)
            .build()
            .await?
            .with_client(LMClient::Test(TestCompletionModel::new(replies)))
            .await?
    };
    lm.history.set_capacity(0);
    dspy_rs::set_history_capacity(0);
    let baseline = predictors(&lm, &[], false);

    let mut samples = Vec::new();
    for iteration in 0..iterations + warmup {
        if live && iteration == warmup {
            barrier("READY")?;
        }
        for e in &fixture.examples {
            let tuned = retrieval_recipe.then(|| {
                let (answer, reflect, revise) = predictors(
                    &lm,
                    &retrieval::select_demonstrations(&e.question, &demonstrations),
                    true,
                );
                (
                    answer,
                    reflect,
                    revise
                        .validated()
                        .max_attempts(NonZeroU32::new(3).unwrap())
                        .validator(|input: &ReviseInput, output: &ReviseOutput| {
                            retrieval::validate_answer(
                                &input.question,
                                &input.context,
                                &output.answer,
                            )
                        }),
                )
            });
            let reflect = tuned
                .as_ref()
                .map(|(_, reflect, _)| reflect)
                .unwrap_or(&baseline.1);
            let context = if retrieval_recipe {
                retrieval::retrieve_context(&e.question, &e.context)
            } else {
                e.context.clone()
            };
            let provider_start = waits.lock().unwrap().len();
            let adapter_start = adapter_timings.adapter.lock().unwrap().len();
            let pipeline_start = Instant::now();
            let mut stage_ns = [0; 3];
            let mut wait_ns = 0;
            let start = Instant::now();
            if wait_ms > 0 {
                tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                wait_ns += start.elapsed().as_nanos();
            }
            let first_input = AnswerInput {
                question: e.question.clone(),
                context: context.clone(),
            };
            let first = match &tuned {
                Some((answer, _, _)) => answer.call(first_input).await?,
                None => baseline.0.call(first_input).await?,
            }
            .into_inner();
            stage_ns[0] = start.elapsed().as_nanos();
            ensure!(
                live || first.answer == e.answer,
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
                    context: context.clone(),
                    answer: first.answer.clone(),
                })
                .await?
                .into_inner();
            stage_ns[1] = start.elapsed().as_nanos();
            ensure!(
                live || feedback.critique == CRITIQUE,
                "critique mismatch for {}",
                e.id
            );
            let start = Instant::now();
            if wait_ms > 0 {
                tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                wait_ns += start.elapsed().as_nanos();
            }
            let revise_input = ReviseInput {
                question: e.question.clone(),
                context: context.clone(),
                draft: first.answer.clone(),
                critique: feedback.critique.clone(),
            };
            let final_answer = match &tuned {
                Some((_, _, revise)) => revise.call(revise_input).await?,
                None => baseline.2.call(revise_input).await?,
            }
            .into_inner();
            let answer = if retrieval_recipe {
                retrieval::canonical_answer(&e.question, &final_answer.answer)
            } else {
                &final_answer.answer
            };
            stage_ns[2] = start.elapsed().as_nanos();
            ensure!(
                live || answer == e.answer,
                "revised answer mismatch for {}",
                e.id
            );
            if iteration >= warmup {
                let pipeline_ns = pipeline_start.elapsed().as_nanos() as u64;
                let score =
                    supported_answers.score(&e.id, &e.question, &e.context, &e.answer, answer)?;
                let provider_ns = waits.lock().unwrap()[provider_start..].to_vec();
                let adapter_events =
                    adapter_timings.adapter.lock().unwrap()[adapter_start..].to_vec();
                samples.push(json!({"id":e.id,"answer":answer,"pipeline_ns":pipeline_ns,
                    "stage_ns":stage_ns.map(|n|n as u64),"simulated_wait_ns":wait_ns as u64,
                    "provider_ns":provider_ns,"adapter_events":adapter_events,"expected_answer":e.answer,
                    "exact_match":score.exact_match,"supported_answer_match":score.supported_answer_match,
                    "supported_answer_annotation":score.annotation_applied,"accepted_answers":score.accepted_answers,
                    "draft":first.answer,"critique":feedback.critique,
                    "revised_answer":final_answer.answer,
                    "context":context}));
            }
        }
    }
    let strict_quality_passed = samples
        .iter()
        .all(|sample| sample["exact_match"].as_bool() == Some(true));
    let supported_quality_passed = samples
        .iter()
        .all(|sample| sample["supported_answer_match"].as_bool() == Some(true));
    println!(
        "{}",
        json!({"framework":"dsrs","kind":if protocol_check {"protocol_check"} else {"pipeline_timing"},"output_format":if json_output {"json_schema"} else {"delimiter"},"iterations":iterations,"warmup":warmup,"live":live,"smoke":smoke,"instrumented":instrumented,
        "simulated_wait_ms_per_stage":wait_ms,"cache":false,"history":false,"samples":samples,
        "recipe":if retrieval_recipe { "retrieved_context_and_demonstrations" } else { "baseline_reflection" },
        "scoring":{"strict_metric":"hotpotqa_normalized_exact_match","quality_status_uses":"strict_metric",
            "supported_metric":scoring::SUPPORTED_METRIC,
            "supported_annotations":if supported_answers_path.is_some() {"custom"} else {"default"},
            "supported_annotation_set":supported_answers.id()},
        "quality_status":if strict_quality_passed { "passed" } else { "failed" },
        "supported_answer_quality_status":if supported_quality_passed { "passed" } else { "failed" }})
    );
    if live {
        barrier("DONE")?;
    }
    Ok(if quality_check && !strict_quality_passed {
        ExitCode::from(2)
    } else {
        ExitCode::SUCCESS
    })
}
