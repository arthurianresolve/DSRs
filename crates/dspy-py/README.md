# DSRs DSPy adapter

`DSRSBAMLAdapter` uses DSRs schema rendering and text coercion for DSPy chat
signatures. It validates clean JSON against the signature's Pydantic annotations
and repairs common formatting errors before validation.
Examples and tool or history handling use DSPy's existing chat adapter.

## Install from a checkout

Use Python 3.10 or newer. The adapter supports Rust 1.97.1, 1.98.1, 1.99.1, and
1.100.0. Run the install commands from the repository root with the repository's
toolchain, or set `RUSTUP_TOOLCHAIN` to select a different supported version.

```sh
uv venv
uv pip install ./crates/dspy-py
```

```python
import dspy
from dsrs_dspy import DSRSBAMLAdapter

dspy.configure(lm=dspy.LM("openai/gpt-4.1-mini"), adapter=DSRSBAMLAdapter())
```

Invalid or missing output fields surface as DSPy's `AdapterParseError`.
The adapter disables DSPy's automatic JSON-adapter retry by default.

This package builds separately from the Rust workspace. Build and test a host
release wheel from the repository root.

```sh
uv pip install "maturin>=1.16,<2" pytest
uv run --no-project maturin build --release --locked --manifest-path crates/dspy-py/Cargo.toml --out target/dspy-wheels
uv pip install "<path-to-built-wheel>"
uv run --no-project python -m pytest crates/dspy-py/tests
```
