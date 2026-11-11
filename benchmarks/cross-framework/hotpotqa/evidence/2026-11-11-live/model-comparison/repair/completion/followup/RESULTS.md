# Follow-up review: preserve evidence and report ambiguity

This is a follow-up to the [earlier native correction](../RESULTS.md), built from
parent commit `d4fd0797031b3251ef8406840bf451ecd5cd0249`. That earlier run
scored 3/3 against the original references. Its office filter removed a valid
ambassadorship, and its series retrieval omitted the passage about enslaved
worlds. The score was real for that executable, but it overstated how much of
the supplied evidence the recipe preserved.

## Implementation

- Incoming document links now retain additional passages that cover question
  terms missing from the initially selected text. In the original series case,
  the companion-book link and its enslaved-world passage are both retained;
  the continuity paragraph is still omitted.
- Broad office questions still put explicit office-holding statements last to
  help the model find them. Other eligible offices remain visible. An excluded
  role is omitted when another eligible office remains; final validation
  enforces the exclusion in all cases. The original office context now contains
  both the ambassadorship and Chief of Protocol.
- Initial drafts still reach reflection. The three-attempt validator applies to
  final revisions, with the original reference labels and exact-match scorer
  unchanged.

## Frozen native run

The locked, optimized example executable has SHA-256
`e0614ca92609f469cd6b0f43403c6ef0715d53e1fd76a04d49be3929e0684f80`.
It used the existing VM107 `Qwen3-0.6B-Tasks` service. The bridge added seed 0,
checked each returned payload SHA-256, and gave the native process the actual
provider reply. All 74 replies contain valid JSON and report a normal `stop`.

| Run | Strict matches | Native exit | Provider requests |
| --- | ---: | ---: | ---: |
| Unmodified baseline | 1/3 | 2 | 9 |
| Revised recipe, original questions | **2/3** | **2** | 9 |
| Four earlier synthetic cases | **4/4** | **0** | 12 |
| Clarified original office question | **1/1** | **0** | 3 |
| Eight prospective synthetic cases | **7/8** | **2** | 24 |
| Repeat original plus clarified | **3/4** | **2** | 12 |
| Only stated office explicitly excluded | No answer | **1** | 5 |

The two distinct strict mismatches are ambiguous office questions. In the
original fixture, Shirley Temple was named United States ambassador and served
as Chief of Protocol. The model answered `ambassador`; the reference says
`Chief of Protocol`. In the prospective fixture, Elena Brook was named
ambassador and served as Consul General. The model again answered
`ambassador`; the reference says `Consul General`. Both answers are supported by
their contexts. These broad questions do not specify which office to select.
The clarified original question explicitly excludes ambassador and passes.
The strict gate therefore exits 2 for the original and prospective sets; no
labels were changed or ambiguous mismatches counted as exact matches.

For the negative case, the draft was `governor` despite the question's exclusion.
The critic called it correct. Final validation rejected `governor` and then two
unsupported `mayor` answers, returning exit 1 after five total requests.

## Verification and limits

- Eleven focused Rust example tests, eleven Python harness tests and three
  native CLI mock-provider regressions pass. Build and test logs are retained.
- [`verify.py`](verify.py) checks the current source hashes, fixture identities,
  every native request and response against its individual VM payload file,
  the transport checksum receipts, dependent-stage prompts, native stdout,
  exact-match counts, exits, and the artifact manifest. `build-record.json`
  retains both the Windows build-input hashes and committed Git blob hashes
  where checkout line endings differ.
- `native-live/` retains 74 payload files, request records, checksum receipts,
  fixtures, stdout/stderr and summaries. `diagnostics/` retains the initial
  smoke result, a partial run stopped after the same ambiguity appeared, and
  the first complete run before the final executable was frozen. The accepted
  result above uses only the final executable named in `build-record.json`.
- The model file was identified and hashed in the earlier report; this run did
  not rehash it. Recorded payloads and checksums are internally auditable, not
  an independent attestation by the provider.
- Context-span validation does not prove entailment, and a yes/no string does
  not prove factual truth. These small diagnostic cases establish no general
  HotpotQA accuracy. Browser and guest-agent transport includes handoff time,
  so these runs provide **no performance comparison**. The old frozen timing
  results were not remeasured on this source.

The broader office ambiguity cannot be eliminated by an input-only heuristic
without choosing among two context-supported answers. A future strict quality
workload needs an unambiguous question or an evaluation rule that accepts both;
either change requires a new, explicitly identified workload.
