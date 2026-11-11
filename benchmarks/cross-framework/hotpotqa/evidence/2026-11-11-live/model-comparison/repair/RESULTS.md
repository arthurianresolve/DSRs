# DSRs corrections and bounded quality validation

## Changes

The SDK's simple native-schema prompt asks for compact JSON, but its few-shot
assistant messages were pretty-printed. DSRs now uses compact JSON for those
demonstrations too. Aliases are preserved. JSON mode, complex native outputs and
scalar outputs retain their existing formatting. The focused schema tests cover
the actual typed predictor chat, rather than only the formatter in isolation.

The DSRs HotpotQA example also has an opt-in `--retrieval-recipe`:

- Select at most two demonstrations by input-question terms, without reading
  their output labels during selection. The four synthetic training traces are
  in `demonstrations.json`; they contain no original HotpotQA rows or named entities.
- Rank supplied titled context documents by query terms, preserve ties and
  follow explicit document-title links. Untitled/no-overlap input retains the
  complete context. This keeps the linked actress/position evidence while
  removing distracting companion-book paragraphs from the series question.
- Use the existing `ValidatedPredict` on answer/revision. Boolean questions must
  return `yes` or `no`; other answers must be contiguous phrases from the
  retrieved context. A comma-delimited `other than X,` constraint rejects X.
  Feedback contains those input-derived rules, never a scoring reference.
  Each validated stage has at most three total attempts and fails explicitly
  if none is acceptable. The critic's opinion is not a validation rule.

The recipe is restricted to protocol/quality diagnostics because it changes
the workload. Baseline reflection remains reproducible without that option.
`--check-quality` requires a live provider, retains the prediction JSON and exits
2 on reference mismatches; exhausted validation returns an error, not a successful
answer. Custom fixtures are supported for these diagnostics.

## Measured results

| Condition | Strict reference exact matches | Native result |
| --- | --- | --- |
| Original baseline, three HotpotQA questions | 1/3 | Quality failed, exit 2 |
| Corrected DSRs recipe, same original questions/references | **2/3** | Quality failed, exit 2 |
| Four additional cases fixed before the trial, absent from demonstrations | **4/4** | Quality passed, exit 0 |
| Clarified position question explicitly excluding ambassador | No successful answer | Three live provider attempts rejected; validation error, exit 1 |

The corrected recipe returns `yes`, `ambassador` and `Animorphs` on the original
cases. The remaining strict mismatch is `ambassador` versus `Chief of Protocol`.
The supplied context supports both positions; the original question does not
distinguish them. The reference and strict score were **not changed** and the
overall original-fixture quality gate still fails. No semantic replacement score
or general accuracy claim is made.

The live exclusion run returned `ambassador`, then `mayor`, then `mayor`.
The application rejected the excluded role and the two answers absent from its
context, then stopped. This verifies rejection and bounded feedback; it is not
a successful factual answer to that question. The model remains unable to solve
every tested question.

## Verification method

1. Capture requests from the compiled native example. Issue fresh real model
   requests for every dependent stage using those templates; feed each actual
   draft/critique into its next stage. Preserve native assistant text arrays.
2. Replay those real replies through the final native executable, asserting
   equality of every request body (apart from seed 0 added by the test bridge).
   Baseline/original/fresh native runs verify 9/9/12 requests respectively and
   verify both passing and failing quality exit statuses.
3. Separately run the final native executable with the real provider through a
   browser/guest-agent bridge on the exclusion case. Its three requests and
   validation feedback are actual native execution, not a simulated retry.

The prototype used string/compact JSON demonstrations. Native capture exposed
the pretty-print difference and a nationality regression, so that first native
run was rejected. The final native-formatted real-provider run, and subsequent
exact-wire replay, verify the corrected format and recipe. All preceding trials
remain available, including failed alternatives. A setup run cancelled before
any provider call is recorded separately.

The final executable SHA-256 is
`26233217c80db3842f9c3526fbef171ebe7532df3c345d58f8bb1263a2b0761e`.
`validation.json` records the tests, source hashes and native exits. Native replay
times include local replay overhead; guest-agent tests include browser transport.
**None of these runs is performance evidence.** Earlier frozen performance
results are unchanged and cannot be applied to this different recipe.

## Model provenance control

The existing model's GGUF hash is
`361cc68159042c36ebff7715dc5a2e4612153e88f3e9c9c234820849d6dc9e1d`.
Its header identifies Qwen3-0.6B. A temporary server using the same installed
llama.cpp image tested [Qwen's official Q8 model](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/blob/main/Qwen3-0.6B-Q8_0.gguf),
verified against published SHA-256
`9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`.
It also scored 1/3 on the original workflow. The different local GGUF was therefore
not established as the cause. That temporary server was bound only to the VM's
loopback interface; its container and downloaded model were removed. Seven named
test directories were removed after remote/local checksums matched, and both
pre-existing model services remained running.

## Limits

This is an example recipe, not a general-purpose entailment verifier. Being an
allowed context span does not establish that an answer resolves the question,
and a canonical yes/no string does not establish its truth. Retrieval can omit
useful evidence. The four fresh cases are small synthetic diagnostics, not a
representative held-out benchmark. The guards guarantee their stated mechanical
constraints and bounded failure, not universal model correctness or a 2× speedup.
