# Real-provider reflection profiling

## Scope and result

This run completes the real provider/I/O attribution work for #23. Both clients
execute Answer → Reflect → Revise over the same three supporting-sentence
HotpotQA fixtures, using a real local CPU-only `llama3.2:3b` model. It measures
this workload and identifies its waiting triggers; it does not establish a
general Rust/Python speedup or HotpotQA accuracy.

All six planned runs passed the protocol and measurement gates. There were
108 actual model requests: 54 warmup and 54 measured requests, of which 36
belong to the four timing runs and 18 to the separate instrumented runs.
Every response was valid JSON. No compiler interference was detected and no
interference retry was needed. Incorrect answers were retained: both clients
scored **0/6 exact matches in timing runs**, and 0/3 in their instrumented pass.

## Identities and method

- Source: `arthurianresolve/DSRs`, branch `fix-hotpotqa-profiling`, clean commit
  `bda64b29974cdd22e4e8004976a7e5c8f4d2263c` before measurement.
- Executable SHA-256:
  `f92fad4d85f0e59e727677ed3e981bd2fdd730f37a7113aaf2861d49d0e85783`.
- Locked, minimal-feature release build, one Cargo job, incremental compilation
  disabled. Rust 1.99.1, Cargo 1.99.1, LLVM 23.1.1; Python 3.14 and DSPy 3.2.1.
  Full versions, build command and lock hash are in `build.json` and `manifest.json`.
- Windows 11 x86-64 client; Ubuntu-24.04 on WSL2 serves portable Ollama 0.40.16.
  Intel family 6/model 142/stepping 12, four physical cores/eight logical CPUs,
  balanced Windows power policy. Frequency/power were recorded, not pinned.
- Model: `llama3.2:3b`, Q4_K_M, CPU-only, digest
  `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`.
- Provider-native JSON Schema in both frameworks; temperature 0, 128 output
  tokens, server seed 0, four model threads and a 2,048-token context. The proxy
  checks the actual wire parameters. Response caches, history and provider
  retries are disabled; the model remains loaded with a 30-minute keep-alive.
- Each process warms one full three-case pass, then measures one full pass.
  Timing order is DSRs → DSPy → DSPy → DSRs. Each framework then has one separate
  instrumented run. SDK timers remain active in timing runs; adapter spans do not.
- Compiler activity is checked before runs, after warmup, once per second during
  measurement and at the end, on Windows and WSL. Accepted intervals have zero
  observed Cargo/rustc activity. This does not prove complete host isolation or
  exclude sub-second activity and ordinary OS noise.

All workload/harness hashes were checked after measurement. The immutable
executable was copied outside Cargo's mutable target directory before use.

## Timing results

Each row aggregates six pipelines and 18 HTTP requests from the two timing runs.
The three different fixtures account for much of the within-run range. The
small sample supports descriptive summaries, not confidence intervals.

| Metric | DSRs | DSPy |
| --- | ---: | ---: |
| Mean pipeline wall time | 21.074 s | 37.097 s |
| Median pipeline wall time | 20.188 s | 33.796 s |
| Pipeline range | 10.897–34.517 s | 20.695–57.047 s |
| First run mean | 20.418 s | 36.951 s |
| Second run mean | 21.729 s | 37.243 s |
| Total pipeline wall time, six pipelines | 126.442 s | 222.584 s |
| Mean Answer / Reflect / Revise stage | 5.744 / 7.596 / 7.733 s | 10.813 / 13.131 / 13.153 s |
| Mean server prompt processing per request | 4.292 s | 9.200 s |
| Mean server token generation per request | 2.685 s | 3.071 s |
| Mean server loading per request | 1.954 ms | 2.021 ms |
| Mean prompt / generated tokens per request | 315.56 / 20.33 | 391.78 / 21.89 |
| Pipeline wall time inside provider SDK calls | 99.998% | 99.967% |
| Pipeline wall time outside SDK calls, mean | 0.497 ms | 12.317 ms |
| Final-answer exact matches | 0/6 | 0/6 |

The observed ratio of mean pipeline wall times is DSPy/DSRs = **1.760**.
It is not an isolated runtime speedup: framework-rendered prompts differ,
token counts and generated answers differ, the model's prompt-cache behavior
is not reset between requests, and neither client meets the exact-match quality
target. These results do **not** demonstrate a 2× improvement, a universally
faster implementation, or faster execution of an equally correct program.

## Adapter attribution and waiting

The instrumented runs each contain three pipelines/nine real requests and
exactly one parse span per request. These timings are kept separate from the
ABBA comparison. Rust's warmed system-prompt cache is part of the measured path.

| Mean adapter elapsed time per request | DSRs | DSPy |
| --- | ---: | ---: |
| Formatting | 6.89 µs | 621.42 µs |
| Parsing | 18.17 µs | 44.77 µs |

The measured bottleneck is model service time, primarily prompt processing,
followed by token generation. In timing runs the server reports mean total
request times of 7.013 s for DSRs and 12.331 s for DSPy. The proxy's upstream
HTTP intervals average 7.022 s and 12.341 s. That gap includes loopback transport
and request/response handling; it is not a pure network measurement. Warm model
loading is small here. Cold-start loading is outside this experiment.

Each stage awaits its model response, and the next stage needs its output.
Those sequential dependencies and the model's prompt/generation work explain
the client's long waiting intervals. The elapsed provider-SDK span includes
serialization, transport and inference waiting. Pipeline-minus-SDK time is a
wall-time residual, not isolated adapter CPU or proof of a specific scheduler
idle state. The explicit adapter spans above supply direct adapter attribution.

Across the two timing intervals, client process wall/CPU counters were:

| Client measurement | DSRs | DSPy |
| --- | ---: | ---: |
| Wall time including result handoff | 126.444 s | 222.587 s |
| Observed process CPU, including interpreter descendants | 15.625 ms | 281.250 ms |
| `max(0, wall - CPU)` | 126.428 s | 222.306 s |

The last row estimates a lower bound on non-CPU wall time at the counters'
precision; it is not exact scheduler idle time. Windows CPU values here advance
in 15.625 ms increments. One DSRs run reports a zero delta, meaning below the
observed counter resolution, not no work. CPU work in the WSL model server and
the separate controller/proxy is outside the client CPU counter. Server phase
durations are elapsed durations, not hardware CPU counters.

## Retained evidence and reproduction

- `manifest.json`: source, hashes, model identity, environment, package versions
  and declared policy; `build.json`: native build exit, toolchain, flags, power
  policy, executable and lock hashes.
- `runtime.json`: portable server binary hash, WSL kernel and verified cleanup.
  The temporary Ollama runtime, both downloaded models and the keys generated
  for this test were removed after measurement; no system package was installed.
- `raw/`: original six accepted reports, stderr, attempt logs, controller summary
  and all 108 provider records, including answers, token counts and stop reasons.
  `analysis.json` recomputes the timing and attribution summaries above.
- `protocol/`: the two clients' full three-fixture protocol checks on the measured
  executable. Each made nine real requests; both scored 0/3 exact matches.
- `diagnostics/`: earlier incomplete delimiter experiments and protocol failures.
  Compiler interference, missing output markers and a dropped HTTP token limit
  invalidated those attempts. The token-limit bug is corrected in #133. These
  diagnostic attempts used different source/output settings and are excluded
  from every timing table and ratio here. They must not be pooled with this run.
- `artifact-sha256.json`: retained-file checksums. Original build/test logs remain
  alongside the artifacts; `validation.json` records the completed checks.

Use the live commands and pinned environment in the parent README. Rebuild the
recorded source with its locked release flags, retain/freeze the executable hash,
start the same local model and run `live.py --samples 2 --wsl-distro Ubuntu-24.04`
with a new output directory outside the checkout. Do not compare a changed
workload or pool failed attempts into the accepted sample. The three-case fixture
excludes retrieval and cannot establish production bottlenecks or benchmark-wide
quality. The tiny sample, changing power/frequency and model/prompt differences
also limit performance conclusions.
