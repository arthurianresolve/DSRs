# Qwen3-0.6B-Tasks answer-quality rerun

The earlier [DSRs correction](repair/completion/RESULTS.md) passed the original
strict references by filtering a valid alternate office. The
[follow-up review](repair/completion/followup/RESULTS.md) restores that evidence
and reports the resulting ambiguous strict mismatches. This report preserves
the earlier unmodified baseline results.

## Result

Both native reflection pipelines completed, each making nine real provider
requests over the same three HotpotQA fixtures. **DSRs: 1/3 exact matches;
DSPy: 1/3 exact matches. Answer quality failed.** Native exit 0 establishes
protocol completion, not a passing answer-quality gate.

The subsequent [DSRs failure investigation](diagnosis/RESULTS.md) tests prompt,
sampling, reasoning, grounding and model alternatives. It identifies a correct
draft being overwritten after incorrect criticism, but finds no verified fix.
The later [DSRs correction and validation report](repair/RESULTS.md) documents
the compact-demonstration fix and an opt-in retrieval recipe, including its
remaining strict-score mismatch and bounded validation failures.

| Question/reference | DSRs final answer | DSPy final answer |
| --- | --- | --- |
| Same nationality? `yes` | `Yes` (match) | `Yes` (match) |
| Government position? `Chief of Protocol` | `United States ambassador to Ghana and to Czechoslovakia` (reference mismatch; supported alternative in the supplied context) | `Shirley Temple` (wrong answer type: person rather than position) |
| Science fantasy series? `Animorphs` | `The Hork-Bajir Chronicles` (wrong companion book) | A sentence identifying The Hork-Bajir Chronicles as the series (wrong companion book) |

The second question is ambiguous: its context names multiple government roles.
The unchanged strict reference still rejects the supported DSRs alternative.
No references were changed and no semantic scores replace exact match.
Both first-stage nationality answers were `No`; both revisions became `Yes`.
Critiques contain contradictory or incorrect claims. The book/series error
survives reflection in both pipelines.

All 18 raw JSON replies equal the parsed draft, critique and final values.
All 18 requests contain the exact fixture question/context, with fresh drafts
and critiques propagated into subsequent requests. These observations support
a generation/prompt-behavior explanation for these errors, rather than parser
substitution. They do not isolate model choice from prompt differences or prove
that every framework path is correct.

## Method and identity

- Date: 2026-11-11. Existing user-provided VM model service; no VM, model or
  container was installed, created or reconfigured.
- Model API identity: `Qwen3-0.6B-Tasks`; reported 751,632,384 parameters,
  Q8_0, 798,801,920 bytes. Existing server disables thinking through
  `--chat-template-kwargs {"enable_thinking":false}`.
- Same frozen DSRs executable as the earlier Llama experiment, SHA-256
  `f92fad4d85f0e59e727677ed3e981bd2fdd730f37a7113aaf2861d49d0e85783`,
  source `bda64b29974cdd22e4e8004976a7e5c8f4d2263c`; native DSPy 3.2.1 client.
- One fresh three-question Answer → Reflect → Revise pass per framework,
  no warmup, temperature 0, 128-token limit, native JSON Schema responses.
  The bridge adds seed 0 to both frameworks' provider requests.
- Requests were captured from the running native clients and carried through
  the Proxmox browser shell/guest agent to the private existing model backend.
  Each model reply was returned to its waiting native client before its next
  stage executed. This is a fresh pipeline, not prerecorded stage replay.
- **Quality evidence only.** Browser handoff and guest-agent delay are inside
  client elapsed times. Those numbers must not be used for latency comparisons.
  DSRs/DSPy requests were paired per stage and run DSRs first on the provider;
  cache state and prompt differences also prevent performance conclusions.
- `tasks-live/` retains the actual requests, raw responses, native stdout,
  stderr, summary and exact transport controller. `validation.json` records
  the input, propagation and raw-to-parsed checks.

This single small pass does not establish a model ranking, overall task accuracy,
equal-quality speedup or universal 2× improvement. The older Llama run remains
0/6 per framework; this new pass is 1/3 per framework, with different model and
transport. It is not a controlled performance comparison with that experiment.

## Diagnostics retained separately

- `tasks-response.json` is an 18-request fixed-prompt replay: final prompts
  yielded 0/3 reference matches per framework. `prompts.json` holds those
  requests, including the earlier Llama drafts/critiques. Those intermediate
  inputs were deliberately fixed; this is **not** the fresh pipeline above.
- `qwen-default-response.json` retains a Qwen3-32B default-thinking probe:
  the first call used all 128 generated tokens on reasoning, returned empty
  answer content and stopped at the token limit. This is a budget/configuration
  failure, not an answer-quality score.
- `qwen-nothink-partial.json` retains three completed non-thinking Qwen3-32B
  probe requests before the user switched the requested model. Incomplete;
  no aggregate quality conclusion.
- `tasks-transport-failure/` retains the first native attempt. Six paired
  rounds reached the model, then browser handoff exceeded its 240-second
  deadline. Native clients exited 1 on the resulting HTTP 504. That attempt
  is excluded from quality scores. The complete attempt uses a 900-second
  handoff deadline; provider timeout remains 120 seconds.

Raw artifacts were retained before five named temporary test directories were
removed from the VM. No pre-existing model service configuration was changed.
