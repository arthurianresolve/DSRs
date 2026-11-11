# Retained diagnostic runs

These runs preceded the final executable and are not accepted as its results.

- `native-review-smoke/` records the first revised original-fixture run: 2/3
  strict matches with `ambassador` as the context-supported alternate answer.
- `native-review-complete/` records a partial full run. It was stopped after
  the original 2/3 result; request 19 was prepared but never sent to VM107.
- `native-review-final/` records the first full 74-request attempt. Its results
  match the accepted final run, but a subsequent source edit changed the
  optimized executable hash. The final run was repeated with the frozen binary.
- `semantic-prompt/` records a rejected nine-request candidate. It added an
  explicit instruction to prefer `served as` over `named` for held-office
  questions in all three predictor stages. The original strict score fell to
  1/3: the model still chose `ambassador` and changed the nationality answer
  from `yes` to `no`. The instruction was reverted; its request bodies and
  checked VM replies are retained here.

Request logs, available native stdout, summaries and state files are retained
here. The rejected semantic prompt also keeps its nine VM payload files and
transport checksum receipts. The final run's payloads and receipts are in
[`../native-live/`](../native-live/).
