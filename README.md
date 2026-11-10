<div align="center">
<img width="768" alt="DSRs" src="https://github.com/user-attachments/assets/bdb80520-216e-4742-b016-b71ca6eaac03" />

# DSRs

Typed LM programs and prompt optimization in Rust.

[Documentation](https://dsrs.herumbshandilya.com) · [API reference](https://docs.rs/dspy-rs) · [Examples](crates/dspy-rs/examples/) · [Issues](https://github.com/krypticmouse/DSRs/issues)
</div>

DSRs takes inspiration from [DSPy](https://github.com/stanfordnlp/dspy). A signature declares typed inputs and outputs; a predictor formats a prompt, calls an LM and parses the result. Modules compose those calls into pipelines. Optimizers tune instructions and demonstrations against your examples and metric.

The API is under active development. This README describes this checkout; a published crate may not yet contain every API shown here. See the [public export migration notes](docs/docs/api/index.mdx) when upgrading.

## Installation

For a released version:

```toml
[dependencies]
dspy-rs = "0.7.3"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
anyhow = "1"
serde_json = "1"
```

Use `cargo add dspy-rs`, or `cargo add dspy-rs --rename dsrs` if you prefer `use dsrs::...` imports. To use this checkout from another local project, replace the version dependency with `dspy-rs = { path = "../DSRs/crates/dspy-rs" }`.

The default features are `data` (CSV, Parquet and Hugging Face ingestion) and `cache` (optional disk response caching). JSON/JSONL loading and bounded execution history also work with `default-features = false`. Caching is off per LM unless `.cache(true)` is set; that setting requires the `cache` feature.

## First typed call and prompt inspection

Set `OPENAI_API_KEY` in your environment, then run the [complete example](crates/dspy-rs/examples/00-quickstart.rs):

```bash
cargo run -p dspy-rs --no-default-features --example 00-quickstart
```

For an offline prompt preview, append `-- --inspect-only`. It constructs the chat and exits before making a provider request; no API key is required.

```rust
use dspy_rs::{LM, Predict, Signature};

/// Answer questions accurately and concisely.
#[derive(Signature, Clone, Debug)]
struct QA {
    #[input]
    question: String,
    #[output]
    answer: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let inspect_only = std::env::args().any(|arg| arg == "--inspect-only");
    let api_key = if inspect_only {
        "unused-for-prompt-inspection".to_string()
    } else {
        std::env::var("OPENAI_API_KEY")?
    };
    let lm = LM::builder()
        .model("openai:gpt-4o-mini".to_string())
        .api_key(api_key)
        .build()
        .await?;
    let predict = Predict::<QA>::builder().lm(lm.clone()).build();
    let input = QAInput {
        question: "What is the capital of France?".into(),
    };

    // Building a chat formats the prompt without sending a provider request.
    let chat = predict.build_chat(&input).await?;
    println!("{}", serde_json::to_string_pretty(&chat.to_json())?);
    if inspect_only {
        return Ok(());
    }

    let (prediction, _conversation) = predict.call_and_parse(chat).await?;
    println!("Answer: {}", prediction.answer);
    for entry in lm.inspect_history(1).await {
        println!("Raw output: {:?}", entry.raw_output);
        println!("Parsed output: {:?}", entry.output);
    }
    Ok(())
}
```

The derive generates `QAInput` and `QAOutput`. A call returns `Predicted<QAOutput>`: typed output fields through `Deref`, plus usage, raw response and field metadata through `.metadata()`. Use `.into_inner()` to move the owned output into the next stage.

Prompt inspection does not create an execution-history entry. Completed calls, cache hits and failures do. History is bounded, independent of response caching, and is not a durable database. Avoid printing prompts or responses containing private data in production logs.

## Predictors and modules

`Predict<S>` implements `Module` and represents one typed LM task. `ChainOfThought<S>` wraps that task and adds a typed `reasoning` output. A custom `Module` implements `forward` to compose stages using ordinary async Rust. Callers use `Module::call`.

Declare optimizer-visible leaves with `dspy_rs::predictors!(MyPipeline { first, second });`. A `Facet` derive alone does not discover predictor fields. The [module guide](docs/docs/components/modules.mdx) contains a complete two-stage implementation.

For example, move an output to the next predictor rather than cloning it:

```rust
let analysis = analyzer.call(AnalyzeInput { text: document }).await?.into_inner();
let summary = summarizer.call(SummarizeInput {
    key_points: analysis.key_points,
}).await?;
```

This fragment assumes your `Analyze` and `Summarize` signatures. For runnable pipeline code, see [01-simple.rs](crates/dspy-rs/examples/01-simple.rs).

## Configuration, validation and optimization

Attach an LM with `Predict::builder().lm(lm).build()`, or install a global default with `configure(lm)`. Call overrides and task-local defaults are documented in the [LM guide](docs/docs/components/lm.mdx). Use `.model("ollama:llama3.2".to_string())` for local Ollama; a custom OpenAI-compatible server uses `base_url` with its API prefix, such as `http://localhost:8000/v1`.

Signatures support typed schemas, formatting, `#[check]` and `#[assert]` constraints. `Predict::validated()` adds application callbacks and bounded feedback retries; see [Predict](docs/docs/components/predict.mdx) for retry and tool-side-effect restrictions.

COPRO, MIPROv2, GEPA, SIMBA and Structural implement the optimizer interfaces. Their configuration and training contracts differ; use the [optimizer guide](docs/docs/components/optimizers.mdx) and runnable examples rather than copying builder options between optimizers. Structural edits apply to interpreter-loaded programs.

## Examples

Run commands from the repository root. Provider examples require credentials or a configured local model; compiling an example does not verify its live provider behavior.

| Example | Purpose | Extra features |
|---|---|---|
| [00-quickstart](crates/dspy-rs/examples/00-quickstart.rs) | Typed QA, offline prompt preview, parsed history | None |
| [01-simple](crates/dspy-rs/examples/01-simple.rs) | Composed QA and rating module | None |
| [03-evaluate-hotpotqa](crates/dspy-rs/examples/03-evaluate-hotpotqa.rs) | Typed dataset evaluation | `data` |
| [04-optimize-hotpotqa](crates/dspy-rs/examples/04-optimize-hotpotqa.rs) | COPRO training and saved state | `data` |
| [07-inspect-history](crates/dspy-rs/examples/07-inspect-history.rs) | Invocation history | None |
| [08-optimize-mipro](crates/dspy-rs/examples/08-optimize-mipro.rs) | MIPROv2 training | `data` |
| [12-tracing](crates/dspy-rs/examples/12-tracing.rs) | Capturing execution traces | None |
| [97-perf-microbench](crates/dspy-rs/examples/97-perf-microbench.rs) | Offline framework microbenchmarks | `cache` |

For example: `cargo run -p dspy-rs --no-default-features --features data --example 03-evaluate-hotpotqa`. See the [example index](docs/docs/guides/examples.mdx) for the full list.

Performance harnesses measure the workloads stated in their protocols. Mocked calls establish framework overhead; they do not establish live inference speed or task quality. See [cross-framework benchmarks](benchmarks/cross-framework/README.md).

## Development

```bash
cargo check --workspace --all-targets --locked
cargo test -p dspy-rs --locked
cargo check -p dspy-rs --all-targets --no-default-features --locked
cargo fmt --all -- --check
```

Use the repository's compatible Rust toolchain and checked-in lockfile. See [documentation development](docs/README.md) and [architecture](docs/docs/getting-started/how-dsrs-thinks.mdx).

Licensed under [Apache-2.0](LICENSE). Inspired by DSPy and built with the Rust ecosystem.
