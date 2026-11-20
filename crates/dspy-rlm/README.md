# DSRs RLM

`dspy_rlm::Rlm` runs Python cells proposed by a language model against typed
DSRs inputs. The session retains variables across turns. Collections are
represented as Python lists and dictionaries, and the model sees a compact
summary. It can inspect the data with Python and call a submodel on selected
passages using `llm_query` or `llm_query_batched`.

The model finishes with `SUBMIT(**fields)`. DSRs validates the submission
against the signature's output fields, aliases, and constraints, then returns
a typed `Predicted` value. With final extraction enabled, the turn limit
triggers a `Predict` call over the execution transcript.

## Build and use

Use CPython 3.14 with Rust 1.97.1, 1.98.1, 1.99.1, or 1.100.0. PyO3 links against the
interpreter. Set `PYO3_PYTHON` if more than one interpreter is installed. On
Windows, include the directory that contains the interpreter DLL on `PATH` when
running the application or tests.

Use both crates from the same checkout in your application's `Cargo.toml`.

```toml
[dependencies]
dspy-rs = { path = "DSRs/crates/dspy-rs", default-features = false }
dspy-rlm = { path = "DSRs/crates/dspy-rlm" }
```

Create an RLM with your existing LM client and call it from an async function.

```rust
use dspy_rlm::Rlm;
use dspy_rs::Signature;

/// Select the entries that best match the query.
#[derive(Debug, Clone, Signature)]
struct Shortlist {
    #[input]
    entries: Vec<String>,
    #[input]
    query: String,
    #[output]
    matches: Vec<String>,
}

let rlm = Rlm::<Shortlist>::builder()
    .lm(lm)
    .max_iterations(8)
    .max_llm_calls(12)
    .build();
let result = rlm.call(ShortlistInput { entries, query }).await?;
```

The action LM defaults to DSRs' scoped or configured LM. Use `.sub_lm(...)` to
select a separate model for queries. A batch returns results in prompt order for
the prefix that fits the remaining query budget. The per-session limits cover
action turns, submodel calls, and text returned from each Python cell.

Embedded Python runs with the application's permissions. Run the embedding
process with the filesystem and network access you intend to give model-generated
code.

Run the tests from the repository root.

```sh
cargo test --locked --manifest-path crates/dspy-rlm/Cargo.toml
```
