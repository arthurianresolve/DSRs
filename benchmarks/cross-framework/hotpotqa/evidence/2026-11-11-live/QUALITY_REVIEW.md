# Answer-quality verification

## Finding

The real `llama3.2:3b` reflection workflow fails the reference-answer quality
check. The profiling implementation's request/response path passes the specific
checks below. Those are separate findings; successful timing collection should
not have been described as an overall successful implementation outcome.

This post-run qualitative review was performed by OpenAI Codex against the
fixture's questions and supplied context. It is not human annotation or a
validated automatic semantic grader. The canonical exact-match results remain
**0/6 for each framework**. No answers, references or timing samples were changed.

## Review of the timing answers

Each question occurs twice per framework. The answers repeat across both timing
runs and the separate instrumented pass.

| Question | DSRs answer and assessment | DSPy answer and assessment |
| --- | --- | --- |
| Were Scott Derrickson and Ed Wood of the same nationality? Reference: `yes`. | “Yes, both were American filmmakers.” Correct meaning, but longer than the reference. Both men are described as American in the context. | “Both Scott Derrickson and Ed Wood were American filmmakers.” Correct meaning, but longer than the reference. |
| What government position was held by the actress who portrayed Corliss Archer? Reference: `Chief of Protocol`. | “United States Ambassador to Ghana and Czechoslovakia.” The context supports these positions too. The question is underspecified; this is a supported alternative, not an equivalent name for Chief of Protocol. | “First Lady.” Incorrect: the supplied context identifies Shirley Temple's ambassadorships and Chief of Protocol role, not First Lady. |
| Which science fantasy young adult series has companion books about enslaved worlds and alien species? Reference: `Animorphs`. | “The Hork-Bajir Chronicles by K. A. Applegate.” Incorrect: the context explicitly identifies this as a companion book, not the requested series. | Identifies The Hork-Bajir Chronicles as a “companion series to Animorphs.” Incorrect: it mislabels the companion book as a series. Merely mentioning Animorphs does not identify it as the answer. |

Over the six timing answers, this review classifies DSRs as two correct in
meaning, two supported alternatives to an ambiguous question, and two incorrect.
DSPy has two correct in meaning and four incorrect. These categories are a
qualitative explanation of the failures, not replacement benchmark scores.

All 42 retained final answers were reviewed: 36 from the warmup and measured
passes, plus six from the earlier protocol checks. Exact texts, classifications,
reasons and source record indexes are in `quality-review/answer-review.json`.

## Checks separating generation from implementation behavior

1. All 54 measured raw model replies match the parsed initial answers, critiques
   and final answers exactly. The 18 raw replies in the two protocol checks also
   match their parsed results. The incorrect terms appear in the model-generated
   JSON; the adapters did not substitute them.
2. The frozen executable used for the measurements was run against a controlled
   local HTTP provider. DSPy was checked in the same way. For all three questions
   and all three stages, the provider verified the exact question/context inputs,
   output schema, temperature and token limit. Distinct synthetic draft and
   critique values verified their propagation into subsequent requests.
3. With known final answers supplied by that provider, both clients produced
   **3/3 exact final matches**, making nine requests each. The retained check
   script, requests and results are under `quality-review/pipeline-check/`.
   This is a plumbing regression check with controlled replies, not evidence of
   model accuracy or real-provider performance.

These checks support a generation/prompt-behavior explanation for the observed
errors. They do not prove that model choice alone is responsible: the two
frameworks render different prompts, and the model responds to those prompts.
The critique stage also asks for unnecessary elaboration on initially correct
`Yes` answers; the final answers preserve the meaning but lose exact match.

## Corrected acceptance and remaining model comparison

The original controller returned 0 after successful protocol/timing collection.
It lacked an answer-quality acceptance gate. The updated controller records
measurement completion separately, writes `outcome.json`, and returns 2 if any
final answer fails the reference check. It retains failures without retrying.
Applying that gate retrospectively to the retained summaries gives **quality
failed**, with `equal_quality_comparison_eligible: false`.

The observed latency ratio remains descriptive profiling data. It is not evidence
of a successful, equally correct optimized program. An alternative-model test
with fixed questions, context, instructions and generation settings is still
needed to isolate the effect of model choice. The requested alternative-model
test has not run: browser control failed before attachment, and the supplied
model API requires authenticated access. No remote VM was created or changed.
