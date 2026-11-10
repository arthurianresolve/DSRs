# HotpotQA reflection profiling

This offline workload runs three dependent typed predictions: answer a question,
critique that answer against its context, then revise it using the critique.
DSRs uses `Predict` and the in-process `TestCompletionModel`; DSPy uses `Predict`,
`ChatAdapter` and `DummyLM`. Each stage's deterministic response is checked.
There are no tool calls, retrieval, network requests or model inference.

The fixture contains the first three HotpotQA distractor validation questions,
their answers and supporting sentences. Both frameworks read the same bytes,
use the same instructions, fields and responses, and execute sequentially.
Framework-specific prompt syntax and typed conversion remain part of the work.
Caching is off and history retention is disabled in both; the implementations'
remaining history bookkeeping may differ. Startup, imports and LM construction
are excluded from recorded pipeline times.

## Run

From the repository root, with a compatible Rust toolchain and Python 3.12:

```text
python -m venv .venv-perf
.venv-perf/Scripts/python -m pip install -r benchmarks/cross-framework/hotpotqa/requirements.txt
cargo build -p dspy-rs --release --no-default-features --locked --example bench-hotpotqa-reflection
.venv-perf/Scripts/python benchmarks/cross-framework/hotpotqa/run.py --rust target/release/examples/bench-hotpotqa-reflection.exe --out ../hotpotqa-evidence
```

On Unix, use `.venv-perf/bin/python` and the executable without `.exe`.
Keep the environment outside the checkout or ignored, and commit the harness
before measuring. The runner requires a clean checkout and a new output
directory. Record the executable's build command, compiler, CPU and power policy
alongside its manifest; the manifest hashes the executable but cannot establish
which source produced an arbitrary executable supplied by the caller.
The requirement pins DSPy; the manifest records all resolved Python packages.

The runner alternates ABBA order over four runs per framework per timing phase:

* **CPU workload:** 20 warmup passes, then 200 measured passes over all three
  fixtures. Each pass makes three typed predictions per question.
* **Simulated waiting:** 20 warmup and 20 measured passes, with a requested 5 ms
  sleep before each stage. The measured sleep duration includes timer and
  scheduler overshoot. It happens outside the mock provider and measures
  pipeline waiting behavior, not actual HTTP or inference latency.
* **Adapter replay:** a separate instrumented pass measures direct prompt
  formatting and response parsing using each stage's fixed inputs. This is
  isolated attribution, not an additive breakdown of the pipeline timings:
  predictors may reuse cached program/system-prompt state. Do not subtract
  replay times from the pipeline to infer provider or idle time.

Raw per-question pipeline and stage timings, observed simulated waiting, output
checks, adapter profiles, executable/input hashes and runtime versions are
retained. The summary reports medians of run means and the individual run means;
it does not claim confidence intervals or universal speedups. Use uninstrumented
pipeline runs for timing comparisons. Profiled adapter replay is descriptive.

The mock's perfect agreement with the supplied answers is a harness check,
not task-quality evidence. This does not satisfy #22's live inference speedup
target or identify a production provider bottleneck. A live comparison must
separately pin a provider/model, token budgets, retrieval, concurrency, cache
policy and a quality threshold, then collect actual request timing and usage.

## Fixture attribution

Source: [HotpotQA](https://hotpotqa.github.io/), Yang et al., *HotpotQA: A Dataset
for Diverse, Explainable Multi-hop Question Answering* (2018), distributed in
the [Hugging Face dataset](https://huggingface.co/datasets/hotpotqa/hotpot_qa),
revision `1908d6afbbead072334abe2965f91bd2709910ab`.
The fixture records source IDs and retains only question, answer and supporting
sentences; other passages and metadata were removed. This adapted dataset
fixture is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/),
as stated by the dataset card. That license applies to `fixture.json`; benchmark
code remains under the repository's Apache-2.0 license.
