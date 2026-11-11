# Completed native DSRs QA correction

The corrected `--retrieval-recipe` passes the unchanged original strict quality
gate using the existing Qwen3-0.6B-Tasks service. All additional positive checks
pass, including the clarified position question. The baseline remains available
for reproduction and still scores 1/3.

## What was fixed

1. **Validate the completed revision.** The first draft can be wrong: correcting
   it is the purpose of reflection. The previous recipe applied factual
   validation to that intermediate draft and stopped before the critic could
   repair it. The typed draft now reaches reflection; the final revision has
   the bounded validation loop. In both native office runs, the model first
   says `mayor`; its critic identifies the context's correct office and revision
   produces it. An excluded or unsupported final answer still fails validation.
2. **Retrieve the relevant relation.** Broad position questions prefer explicit
   office-holding statements. Questions asking about appointments retain those
   appointment statements. Input-derived exclusions are applied first, and
   negated statements do not displace positive appointments. Acting-role
   questions keep their context. Unknown relations retain the original context.
3. **Preserve connecting evidence.** Retrieval retains documents named in the
   question and individual incoming reference lines. The series case now keeps
   the companion-book-to-series link without restoring every distracting book
   paragraph. The output remains supported by the supplied text.
4. **Canonicalize office titles.** For position/office questions only, the
   application removes a trailing `of [the] <territory>` or `for [the] <territory>`
   suffix using pinned English Unicode CLDR names. The result is a substring of
   the raw answer. Integral title words and unknown jurisdictions remain intact.
   The raw `revised_answer` is retained alongside the canonical `answer`.
   [Data provenance and license](../../../../../data/README.md) describe the
   unmodified input data. Evaluation references and the scorer were not changed.

The original position question is still underspecified: its context supports
both ambassadorships and Chief of Protocol. This recipe declares an office-holding
preference for broad position questions, selecting a supported reference answer.
It does not establish that the original reference is the only valid reading.

## Native results

| Run | Strict matches | Native exit | Real provider requests |
| --- | ---: | ---: | ---: |
| Unmodified baseline | 1/3 | 2 | 9 |
| Corrected recipe, original questions | **3/3** | **0** | 9 |
| Four earlier synthetic cases | **4/4** | **0** | 12 |
| Clarified position question | **1/1** | **0** | 3 |
| Eight additional synthetic boundary cases | **8/8** | **0** | 24 |
| Repeat of original and clarified questions | **4/4** | **0** | 12 |
| Only stated office is explicitly excluded | No successful answer | **1** | 5 |

There are 16 distinct positive cases and four repeated predictions, all passing.
The last run receives `governor` as its draft. Its critic incorrectly calls that
an acceptable answer despite the exclusion. Final validation rejects `governor`,
then rejects the unsupported `mayor` twice, and stops after three final attempts.
The application does not trust the critic as its acceptance rule.

All **74 native requests** use real provider replies, with valid JSON and normal
provider stop reasons. `relation-recipe/native-live-fixed/` retains every request,
raw response, fixture, native stdout/stderr and exit summary. These are direct
native executions through a browser/guest-agent bridge, not replayed model calls.
The bridge adds seed 0 and verifies the native request body before returning the
provider's actual reply. Its response payload checksums were verified.

## Regressions and provenance

- Ten Rust example tests pass, covering linked documents, incoming evidence,
  retrieval fallback, exclusions, appointment selection, negation, acting roles,
  answer constraints and title canonicalization.
- Eleven profiling-harness checks pass in the pinned Python environment.
- Three native CLI regressions use a separate local mock provider. They verify
  repair of an unsupported draft, rejection after three final attempts, and
  isolation of scoring references. Changing the evaluation label leaves every
  request and the prediction unchanged; it only changes the quality exit.
- `build-record.json` pins the optimized executable hash, toolchain, source
  hashes, lockfile and build flags. `verify.py` checks the retained real replies,
  dependent-stage inputs, raw/canonical answers, strict results and cleanup.
- The model file SHA-256 is
  `361cc68159042c36ebff7715dc5a2e4612153e88f3e9c9c234820849d6dc9e1d`.
  The existing model and provider configuration were preserved.

## Rejected and diagnostic trials

All preceding trials are retained, including failures:

- `grounded-choices/`: allowing every context span still permits irrelevant
  answers such as `United States`; this constraint was not adopted.
- `role-demonstrations/`: richer demonstrations still return an excluded
  ambassadorship. They were not adopted.
- `focused-role/` and `relation-recipe/results.json`: provider trials established
  the useful relation focus, with separate synthetic cases, before native checks.
- `title-normalization/`: model-based shortening copies country qualifiers.
  It was replaced by the deterministic territory rule.
- `relation-recipe/native-live/`: the first native attempt rejects its office
  draft before reflection. It was stopped after 18 real requests; the unsent
  nineteenth request is retained as cancelled setup. Its initial cancellation
  note conjectured a response-repeatability problem. Review of the earlier trace
  showed the same `mayor` draft there too; that earlier pipeline had allowed its
  critic to correct it. The defect was validation placement.
- `cache-repeatability/`: all six office probes return `mayor`, with prompt
  caching both disabled and enabled. The response counters confirm that the
  setting took effect. This did not support changing provider cache policy.

The first 13 request bodies of the rejected and corrected native trials are
identical. The corrected workflow then sends the office draft to a critic;
the rejected workflow instead asks the draft predictor to retry it.

## Scope and cleanup

This remains an opt-in QA recipe for protocol/quality diagnostics. Its English
relation rules and territory normalization are application policy. The small
synthetic cases are development diagnostics, not representative HotpotQA accuracy.
Context-span validation does not prove entailment, and a yes/no string does not
prove factual truth. These results resolve the demonstrated workflow failures;
they do not guarantee arbitrary questions or models.

Browser/guest-agent transport includes operator handoff time. The recipe also
changes prompts and retrieval. **These runs are not performance evidence.**
Previously frozen timing results were not remeasured on this head.

After local and remote hashes matched for all 98 files, eight named temporary VM
directories were removed. Both pre-existing model services remained running.
No model installation or persistent service reconfiguration was needed.
`relation-recipe/cleanup-plan.json` and `cleanup-result.json` retain that evidence.
