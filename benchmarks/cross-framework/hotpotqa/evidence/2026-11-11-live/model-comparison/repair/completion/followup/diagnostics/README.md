# Retained diagnostic runs

These runs preceded the final executable and are not accepted as its results.

- `native-review-smoke/` records the first revised original-fixture run: 2/3
  strict matches with `ambassador` as the context-supported alternate answer.
- `native-review-complete/` records a partial full run. It was stopped after
  the original 2/3 result; request 19 was prepared but never sent to VM107.
- `native-review-final/` records the first full 74-request attempt. Its results
  match the accepted final run, but a subsequent source edit changed the
  optimized executable hash. The final run was repeated with the frozen binary.

Request logs, available native stdout, summaries and state files are retained
here. The final run's per-request VM payload files and checksum receipts are in
[`../native-live/`](../native-live/).
