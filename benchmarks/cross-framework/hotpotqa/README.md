# HotpotQA reflection profiling

## Live provider and I/O profiling

`live.py` runs both frameworks against the same local Ollama model through an
instrumented HTTP proxy. The proxy translates their OpenAI-compatible requests
to Ollama's native `/api/chat` endpoint so the report can retain server-reported
loading, prompt-evaluation and generation timings and token counts. It does not
generate, substitute or repair model answers. Both clients use provider-native
JSON Schema output, identical fixture/task instructions, temperature 0, 128 output tokens,
seed 0 at the proxy, four model CPU threads, a 2,048-token context, no response cache/history/retries,
and one sequential request per stage. DSRs opts into the known endpoint's schema
support and disables scalar JSON output for object parity. DSPy uses JSONAdapter
formatting/parsing with its pinned schema exporter and no second-request
fallback. Raw provider JSON must be valid before a run can be accepted, even if
a client parser would tolerate or repair malformed JSON.

Build a clean committed checkout using a locked release build, retaining the
command/toolchain/exit and the executable hash. Then, with the pinned Python
environment and a running local Ollama model:

```text
cargo build -p dspy-rs --release --no-default-features --locked --example bench-hotpotqa-reflection
python benchmarks/cross-framework/hotpotqa/live.py --rust target/release/examples/bench-hotpotqa-reflection.exe --ollama http://127.0.0.1:11434 --model llama3.2:3b --samples 2 --wsl-distro Ubuntu-24.04 --out ../hotpotqa-live-evidence
```

Omit `--wsl-distro` on other systems. Measurements wait for compiler-idle windows;
compiler activity detected during a run invalidates it and is retained as a
failure. The controller waits and repeats the complete warmup/measured attempt,
up to `--interference-retries` additional attempts (default three). Accepted runs
keep their planned ABBA order; every attempt has a distinct artifact name and
an acceptance log. Provider/parse failures stop immediately, and incorrect
answers are retained rather than retried. No other benchmarks/model requests should run concurrently. The
controller checks once per second, so the observer itself introduces some host
overhead and cannot exclude every short unrelated task.
Requests keep the model loaded for 30 minutes, longer than the controller's
15-minute idle-wait limit, so waiting after warmup does not unload the model.

Two runs per framework follow ABBA order. Each process warms one complete pass
over the three fixtures, then times one pass. A separate instrumented pass per
framework records actual adapter formatting/parsing, including Rust's system
prompt cache behavior. Adapter instrumentation is excluded from the ABBA runs;
lightweight SDK timers remain enabled in both. Rust records the full lifetime
of the provider-dispatch span, including awaits, while Python times LM.forward.
Instrumented runs provide
attribution, not the cross-framework timing comparison. No simulated sleeps are
inserted in live mode.

Both client executables accept `--check-protocol` to run all three fixtures
without warmup, using their configured `DSRS_PROFILE_BASE_URL` and
`DSRS_PROFILE_MODEL`. Run with `1 0 --live --json-output --check-protocol`; these outputs are
labelled `protocol_check` and are correctness diagnostics, not timing evidence.
The task instructions explicitly name the output fields in both
frameworks. A regression checks that these instructions remain identical.

The report retains pipeline/stage wall time, provider-SDK elapsed time, actual
adapter spans, proxy/server timings, generated tokens, process CPU time and
answers, raw response JSON and provider stop reasons. CPU is sampled at barriers after warmup and before process exit;
initialization/imports/model construction are excluded. Process-interval wall
time also includes result serialization and controller handoff. Its
`max(0, wall - process CPU)` is a **lower bound on non-CPU wall time**, not a
precise scheduler idle measurement: overlapping worker CPU and descheduling
prevent that interpretation. Request timing and server phases identify waiting
triggers; SDK time includes transport and provider waiting, not CPU alone.
Process CPU counters have OS-dependent resolution. In this Windows run they
advance in 15.625 ms increments; a zero delta means below the observed counter
resolution, not zero work. Interpret the non-CPU lower bound at that precision.
Pipeline-minus-SDK wall time is a residual, not isolated adapter CPU. Server
phase times are Ollama's reported elapsed durations, not hardware CPU counters.

Every successful measured pipeline must make exactly three real HTTP requests.
Normalized final-answer exact match is reported for every fixture; incorrect
answers are retained. Parse/provider failures stop the run and retain diagnostic
artifacts. The small supporting-sentence fixture excludes retrieval and does not
establish HotpotQA benchmark accuracy, production bottlenecks, confidence
intervals or general speedup guarantees. Cold model/prompt costs are excluded.
The manifest records model digest/version, executable/workload hashes, commit,
toolchain and Python package versions; retain the build record alongside it.

The completed local-model run and its limitations are documented in
[the retained live results](evidence/2026-11-11-live/RESULTS.md).
Its [answer-quality review](evidence/2026-11-11-live/QUALITY_REVIEW.md) distinguishes
formatting mismatches, an ambiguous reference question and factual model errors.
The [Qwen3-0.6B-Tasks rerun](evidence/2026-11-11-live/model-comparison/RESULTS.md)
completed fresh native pipelines with 1/3 exact matches each. It verifies field
propagation against real replies but still fails answer quality; its browser
transport makes it unsuitable for latency comparisons.
The [DSRs failure investigation](evidence/2026-11-11-live/model-comparison/diagnosis/RESULTS.md)
also checks the previous full prompt, alternative layouts, grounding, sampling
and a larger model. Its unsuccessful candidates are retained; they have not been
adopted as runtime fixes.

Measurement completion and answer quality are separate outcomes. After all
planned runs, the controller writes `outcome.json`. Its quality gate requires
every final answer in both timing and instrumented passes to match the reference;
it exits with status 2 when quality fails. Failed answers remain in the reports
and are never retried to obtain a passing score. A passed quality gate establishes
eligibility for further comparison, not proof of a speedup. The earlier retained
experiment predates this gate: its native controller exit was 0, but retrospective
quality evaluation fails. The raw experiment files and exact-match scores remain
unchanged.

## DSRs retrieval and validation diagnostics

DSRs supports an opt-in `--retrieval-recipe` for protocol/quality checks. It ranks
the supplied context documents, follows document-title links and selects up to
two synthetic demonstrations by their input questions. Answer/revision stages
use bounded application validation: yes/no format, extraction from the retrieved
context and comma-delimited `other than X,` exclusions. These rules do not prove
that every context-supported answer resolves the question.

```powershell
$env:DSRS_PROFILE_BASE_URL = "http://your-local-provider/v1"
$env:DSRS_PROFILE_MODEL = "Qwen3-0.6B-Tasks"
cargo run -p dspy-rs --release --no-default-features --example bench-hotpotqa-reflection -- 1 0 --live --json-output --check-quality --retrieval-recipe
```

`--check-quality` prints the prediction report and exits 2 if any reference answer
does not match. An exhausted validation loop returns an error. Add
`--fixture benchmarks/cross-framework/hotpotqa/validation-fixture.json` to run
the four separate synthetic validation cases, which are absent from the training
demonstrations. Custom fixtures and this recipe require diagnostic mode; they
cannot silently change the timing workload.

On the retained Qwen3-0.6B-Tasks run, the recipe improved original strict exact
matches from 1/3 to 2/3 and passed 4/4 fresh cases. The original position question
remains ambiguous, and its unchanged strict quality gate still fails. A separate
exclusion question exhausted three attempts and failed explicitly. See the
[correction report](evidence/2026-11-11-live/model-comparison/repair/RESULTS.md)
for real-provider traces, exact native request verification and limitations.
These diagnostic results establish no speedup.

## Offline controlled workload

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
