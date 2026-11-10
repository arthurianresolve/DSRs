# Issue backlog fixes and local performance evidence — 2026-11-10

This batch adds explicit public exports, typed validation callbacks with feedback
retries, example feature gates, per-call render-context sharing, reproducible
HotpotQA profiling, and updated getting-started documentation. The changes are
stacked on `fix-optional-response-cache` (PR #134). The work below does not close
the entire compiler, storage or provider-compatibility backlog.

## Branches and scope

| Branch | Parent branch | Change |
|---|---|---|
| `fix-example-feature-flags` | `fix-optional-response-cache` | Correct example feature requirements and obsolete `dataloaders` commands; addresses the build/documentation portion of #85 |
| `fix-explicit-public-exports` | `fix-example-feature-flags` | Named public facade exports and API migration notes for #76 |
| `fix-validation-feedback` | `fix-explicit-public-exports` | Typed validation callbacks, bounded corrective turns and usage accounting for #44 |
| `fix-render-context-ownership` | `fix-validation-feedback` | Share one alias-aware MiniJinja input value per formatting call; measured #24/#25 clone hotspot |
| `fix-hotpotqa-profiling` | `fix-render-context-ownership` | Offline three-stage reflection benchmark, retained timings and separate adapter replay for #23 |
| `fix-documentation-checklist` | `fix-hotpotqa-profiling` | Quickstart, Predictor/Module distinction, LM selection, prompt/history inspection, README and example index for #12; this results archive |

The quickstart code in the README and guide exactly matches `00-quickstart.rs`
(apart from the example's header comment). Local Markdown links were checked.
The minimal-feature release example compiled and its `--inspect-only` path ran
without credentials or a provider request. Live calls still require a served
model or valid provider credentials.

## Rendering comparison

Baseline: `ab48894f22320af8da9510bcbc5f8db79b9cdacd` (harness only).
Candidate: `c0a5820090792fac5580aec8c64fc7de7cd3b191`.
Both builds used the identical committed harness, lockfile and
`--release --no-default-features --locked -j1` configuration. Binaries were copied
immediately after each clean-source build and hashed. See the
[protocol](../../rendering/README.md), [manifest](rendering/manifest.json),
[build records](rendering/baseline-build.json), and [summary](rendering/summary.json).

Eight samples per binary alternated ABBA order; each sample used 100 warmups and
2,000 timed iterations per case. Allocation profiling used a separate
200-iteration run. Complete output bytes matched in all six cases; the
[SHA-256 output hashes](rendering/output-sha256.json) are retained. Timing results
are medians of run means, not confidence intervals.

| Workload | Baseline µs/call | Candidate µs/call | Candidate/baseline time | Requested allocation bytes before → after |
|---|---:|---:|---:|---:|
| Plain, 1 field | 66.54 | 63.34 | 0.952 | 131,928 → 131,928 |
| Plain, 8 fields | 8.71 | 9.17 | 1.053 | 135,708 → 135,708 |
| Plain, 16 fields | 8.40 | 8.06 | 0.959 | 140,034 → 140,034 |
| Jinja, 1 field | 71.09 | 62.06 | 0.873 | 269,518 → 233,632 |
| Jinja, 8 fields | 587.25 | 392.71 | 0.669 | 1,340,772 → 477,469 |
| Jinja, 16 fields | 1,164.53 | 684.14 | 0.587 | 2,811,482 → 757,601 |

Every case also includes a 32 KiB plain payload. The wide Jinja cases took about
33% and 41% less time and requested about 64% and 73% fewer allocation bytes.
All predeclared local acceptance checks passed: plain controls within 10%,
Jinja latency at least 10% lower, and wide-Jinja requested allocation bytes at
least 50% lower. Allocation bytes sum successful allocation/reallocation
requests; they are not peak live memory or RSS. Do not infer scaling across
the plain-width rows from this allocation-sensitive microbenchmark.

The 43 targeted Rust tests passed: input formatting (13), dynamic rendering (6),
prompt goldens (4), prompt composition (8), adapter parsing (2), and validation
feedback (10). The [test transcript](rendering/validation.txt) records these
results. Earlier stack checks covered default and minimal feature compilation,
workspace compilation, export compile-fail contracts, and validation/history
integration. A complete workspace test suite was not rerun after this rendering
change.

The context-sharing change overlaps the input-rendering work in open upstream
[PR #103](https://github.com/krypticmouse/DSRs/pull/103). Coordinate that merge:
retain its fallible rendering/error handling when reconciling the shared-context
implementation. This branch preserves the current base branch's error behavior;
it does not replace PR #103's security and resource-limit fixes.

## HotpotQA reflection profile

Clean source: `8534e97cb95f93adc9b66b1037ba39886b5b61ab`.
The [build record](hotpotqa/build.json) and [manifest](hotpotqa/manifest.json)
record executable, fixture, script and lockfile hashes, exact runtime versions
and the resolved Python environment. The
[protocol](../../cross-framework/hotpotqa/README.md) specifies the fixed fixture,
instructions and three dependent predictions: answer, critique, revise.

Both frameworks used the same three HotpotQA questions and supporting contexts,
deterministic checked responses, disabled response caching, and disabled history
retention. This is a framework workload; it has no retrieval, tools, HTTP traffic
or inference. Perfect output agreement is a harness check, not task-quality
evidence. Startup/import/construction time is excluded.

Each timing phase used four runs per framework in ABBA order. The CPU phase
measured 200 passes over all three questions after 20 warmup passes; the waiting
phase used 20 measured and 20 warmup passes. The latter requested a 5 ms sleep
outside the mock provider before each stage. See [all run means](hotpotqa/summary.json)
and the raw per-question JSON samples in this directory.

| Phase | Framework | Pipeline ms/question | Answer stage ms | Critique stage ms | Revise stage ms | Observed simulated wait ms/question |
|---|---|---:|---:|---:|---:|---:|
| CPU | DSRs | 0.199 | 0.063 | 0.065 | 0.070 | 0 |
| CPU | DSPy 3.2.1 | 5.126 | 1.578 | 1.684 | 1.842 | 0 |
| Simulated wait | DSRs | 45.288 | 14.641 | 15.051 | 15.381 | 44.647 |
| Simulated wait | DSPy 3.2.1 | 19.452 | 6.435 | 6.497 | 6.540 | 15.932 |

Values are medians of run means; independently computed stage medians need not
sum to the pipeline median. DSRs' mock CPU pipeline took less time on this host.
Its waiting phase was slower: most of its time was measured inside Tokio sleeps,
far above the requested total of 15 ms. DSPy used Python `time.sleep` and observed
much less overshoot. These are different runtime timers, not a common remote
service. The result points to timer/scheduling behavior to investigate on this
host; it does not prove a particular kernel cause or a production I/O regression.
No timer-resolution or global power-policy changes were made to improve scores.

A separate adapter replay measured direct formatting and parsing over each
fixed stage input. Mean format/parse times across the nine cases were about
7.00/2.53 µs for DSRs and 323.65/26.68 µs for DSPy. These are instrumented,
isolated profiles, not an additive decomposition of the pipeline: predictors
can reuse program and prompt state. Do not subtract them to estimate provider
or idle time.

## Conditions and remaining scope

Host: Windows 11, Intel Core i7-10610U (4 cores/8 logical processors), Balanced
power policy. Rust/Cargo 1.98.1, LLVM 22.1.8, MSVC target; Python 3.12.11 with
DSPy 3.2.1. There was no CPU affinity, frequency lock or exclusive-host control.
Another repository's Rust build was present at the start of the reflection run.
Raw samples show material variation; treat these as local descriptive results.
No thermal, RSS or hardware-counter measurements were collected.

| Issue | Remaining scope |
|---|---|
| #22 | A twofold improvement to live inference/I/O is unverified. This batch measures framework work and one rendering optimization; simulated waiting exposes a separate timer concern. |
| #23 | Reproducible offline reflection profiling is added. Actual provider/network bottleneck and idle attribution require a separately controlled live workload. |
| #24 | The measured repeated input-context construction is fixed. No claim is made about every memory-access path or a universal speedup. |
| #25 | One clone hotspot and the current ownership boundaries are addressed. A repository-wide ownership redesign is not complete. |
| #26, #28 | Skipped: existing typed DAG/structural-edit foundations do not define an acceptance contract for automatically fusing/splitting LM prompts. Generic shape-compatible graph rewrites cannot establish task-quality equivalence. No compiler/fusion closure is claimed. |
| #85 | Feature-dependent builds and documented commands are corrected. The broader historical provider/output-format failures remain unverified. |
| #55 | Skipped: first-class persistent interaction storage, backend integration and the broader artifact/dataset storage work were not implemented. Bounded history is not a persistent store. |
| #72 | Skipped: signature-special tool/history fields, provider capability selection and prompt-based tool fallback were not implemented. Existing native tool execution and invocation history do not complete that migration. |
| #83 | Skipped as explicitly requested. |

No issues were closed, and no claims are made that the skipped work is resolved.
