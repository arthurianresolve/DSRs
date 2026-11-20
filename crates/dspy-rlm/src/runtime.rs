use std::ffi::CString;
use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyModule;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

pub(crate) struct Query {
    pub prompts: Vec<String>,
    pub response: oneshot::Sender<Result<Vec<String>, String>>,
}

#[pyclass(frozen)]
struct QueryTool(mpsc::Sender<Query>);

#[pymethods]
impl QueryTool {
    fn __call__(&self, py: Python<'_>, prompts: Vec<String>) -> PyResult<Vec<String>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }
        py.detach(|| {
            let (response, result) = oneshot::channel();
            self.0
                .blocking_send(Query { prompts, response })
                .map_err(|_| PyRuntimeError::new_err("The RLM session ended"))?;
            result
                .blocking_recv()
                .map_err(|_| PyRuntimeError::new_err("The RLM session ended"))?
                .map_err(PyRuntimeError::new_err)
        })
    }
}

pub(crate) struct CellOutput {
    pub stdout: String,
    pub error: Option<String>,
    pub submission: Option<String>,
}

pub(crate) struct Session {
    python: Arc<Py<PyAny>>,
    pub queries: mpsc::Receiver<Query>,
}

impl Session {
    pub fn new(input: String, output_limit: usize) -> PyResult<(Self, String)> {
        static SESSION_CLASS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
        let (sender, queries) = mpsc::channel(1);
        let (python, preview) = Python::attach(|py| {
            let session_class = SESSION_CLASS.get_or_try_init(py, || {
                let source = CString::new(include_str!("repl.py")).expect("valid REPL source");
                Ok::<_, PyErr>(
                    PyModule::from_code(py, &source, c"dspy_rlm_repl.py", c"_dspy_rlm")?
                        .getattr("Session")?
                        .unbind(),
                )
            })?;
            let query = Py::new(py, QueryTool(sender))?;
            let session = session_class.bind(py).call1((input, query, output_limit))?;
            let preview = session.getattr("preview")?.extract()?;
            Ok::<_, PyErr>((session.unbind(), preview))
        })?;
        Ok((
            Self {
                python: Arc::new(python),
                queries,
            },
            preview,
        ))
    }

    pub fn execute(&self, source: String) -> JoinHandle<PyResult<CellOutput>> {
        let python = Arc::clone(&self.python);
        tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                let result = python.bind(py).call_method1("run", (source,))?;
                let (stdout, error, submission) = result.extract()?;
                Ok(CellOutput {
                    stdout,
                    error,
                    submission,
                })
            })
        })
    }
}
