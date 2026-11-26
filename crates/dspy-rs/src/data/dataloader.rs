use anyhow::{Context, Result, anyhow};
#[cfg(feature = "data")]
use arrow::array::{
    Array, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
#[cfg(feature = "data")]
use csv::{ReaderBuilder, StringRecord};
#[cfg(feature = "data")]
use hf_hub::api::sync::Api;
#[cfg(feature = "data")]
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use reqwest;
use serde::de::{DeserializeSeed, Error as _, IgnoredAny, SeqAccess, Visitor};
use std::any::TypeId;
use std::cell::Cell;
use std::collections::HashMap;
#[cfg(feature = "data")]
use std::collections::HashSet;
use std::fs;
#[cfg(feature = "data")]
use std::io::Cursor;
use std::io::{BufRead, BufReader, Read};
#[cfg(feature = "data")]
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tracing::debug;

use crate::data::utils::is_url;
use facet::Facet;
use serde::de::DeserializeOwned;

mod projection;
use serde_json::Value;

/// Controls how typed loaders handle source fields that are not part of the target row struct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnknownFieldPolicy {
    /// Ignore extra source fields that are not consumed by the signature.
    #[default]
    Ignore,
    /// Fail the load when a row contains any extra source field.
    Error,
}

/// Options for shape-driven typed loading.
///
/// `field_map` remaps row-struct fields to source fields:
/// - key: row struct field name
/// - value: source field/column name in the file/dataset
///
/// `unknown_fields` controls whether extra source fields are ignored or rejected.
#[derive(Debug, Clone)]
pub struct TypedLoadOptions {
    pub field_map: HashMap<String, String>,
    pub unknown_fields: UnknownFieldPolicy,
}

impl Default for TypedLoadOptions {
    fn default() -> Self {
        Self {
            field_map: HashMap::new(),
            unknown_fields: UnknownFieldPolicy::Ignore,
        }
    }
}

/// Limits applied to HTTP JSON and CSV dataset loads. Local files are unaffected.
#[derive(Debug, Clone, Copy)]
pub struct RemoteLoadLimits {
    pub max_bytes: usize,
    pub max_rows: usize,
    pub timeout: Duration,
}

impl Default for RemoteLoadLimits {
    fn default() -> Self {
        Self {
            max_bytes: 16 * 1024 * 1024,
            max_rows: 50_000,
            timeout: Duration::from_secs(30),
        }
    }
}

/// Raw parsed row passed to custom mapper closures in `load_*_with` APIs.
///
/// Values are normalized into `serde_json::Value` so mappers can deserialize
/// directly into strongly typed Rust values using [`RowRecord::get`].
#[derive(Debug, Clone)]
pub struct RowRecord {
    /// 1-based row index in the loaded stream after filtering empty rows.
    pub row_index: usize,
    /// Parsed key-value payload for the row.
    pub values: HashMap<String, serde_json::Value>,
}

impl RowRecord {
    /// Deserialize a typed value from a row field.
    ///
    /// Returns [`DataLoadError::MissingField`] if the key is absent.
    ///
    /// Returns [`DataLoadError::TypeMismatch`] on deserialization failure.
    /// `String` reads return string values as-is and format other values as JSON.
    pub fn get<T: serde::de::DeserializeOwned + 'static>(
        &self,
        key: &str,
    ) -> std::result::Result<T, DataLoadError> {
        let value = self
            .values
            .get(key)
            .ok_or_else(|| DataLoadError::MissingField {
                row: self.row_index,
                field: key.to_string(),
            })?;

        let value = if TypeId::of::<T>() == TypeId::of::<String>() {
            match value {
                Value::String(_) => value.clone(),
                Value::Number(number) => Value::String(number.to_string()),
                Value::Bool(flag) => Value::String(flag.to_string()),
                other => Value::String(other.to_string()),
            }
        } else {
            value.clone()
        };
        serde_json::from_value(value).map_err(|err| DataLoadError::TypeMismatch {
            row: self.row_index,
            field: key.to_string(),
            message: err.to_string(),
        })
    }
}

/// Row-aware errors produced by typed data loading.
#[derive(Debug, thiserror::Error)]
pub enum DataLoadError {
    /// Source read/download failure.
    #[error("I/O error: {0}")]
    Io(#[source] anyhow::Error),
    /// CSV parser failure.
    #[error("CSV error: {0}")]
    Csv(#[source] anyhow::Error),
    /// JSON/JSONL parser failure.
    #[error("JSON error: {0}")]
    Json(#[source] anyhow::Error),
    /// Parquet parser failure.
    #[error("Parquet error: {0}")]
    Parquet(#[source] anyhow::Error),
    /// HuggingFace Hub listing or file retrieval failure.
    #[error("HuggingFace error: {0}")]
    Hf(#[source] anyhow::Error),
    /// Required row-struct field was missing from a row.
    #[error("missing field `{field}` at row {row}")]
    MissingField { row: usize, field: String },
    /// Row had an unexpected extra field when unknown-field policy is `Error`.
    #[error("unknown field `{field}` at row {row}")]
    UnknownField { row: usize, field: String },
    /// Field existed but could not be converted to required type.
    #[error("type mismatch for field `{field}` at row {row}: {message}")]
    TypeMismatch {
        row: usize,
        field: String,
        message: String,
    },
    /// Custom mapper closure returned an error.
    #[error("mapper error at row {row}: {source}")]
    Mapper {
        row: usize,
        #[source]
        source: anyhow::Error,
    },
}

/// Typed dataset ingress for JSON/CSV/Parquet/HuggingFace sources.
///
/// JSON/JSONL loading is always available; the CSV, Parquet, and HuggingFace
/// loaders require the (default-on) `data` feature, which carries the heavy
/// arrow/parquet/hf-hub/csv dependency stack.
///
/// Canonical public contract:
/// - Returns `Vec<E>` for any row struct `E: Deserialize + Facet` — rows are
///   *your* type, not a signature-shaped pair, so they can carry gold labels
///   and metric-only fields freely.
/// - Uses `E`'s Facet shape for required fields and type-aware coercion.
/// - Supports field remapping via [`TypedLoadOptions::field_map`].
/// - Reports row-aware failures through [`DataLoadError`].
pub struct DataLoader;

impl DataLoader {
    #[tracing::instrument(
        name = "dsrs.data.load_json",
        level = "debug",
        skip(path, opts),
        fields(
            is_url = is_url(path),
            lines,
            field_map_entries = opts.field_map.len(),
            unknown_fields = ?opts.unknown_fields
        )
    )]
    /// Load typed rows from JSON array/object or JSONL.
    ///
    /// `lines = true` treats the file as JSONL (`one object per line`).
    ///
    /// # Errors
    /// Returns [`DataLoadError`] wrapped in `anyhow::Error` for parse, schema,
    /// mapping, and conversion failures.
    pub fn load_json<E>(path: &str, lines: bool, opts: TypedLoadOptions) -> Result<Vec<E>>
    where
        E: DeserializeOwned + for<'f> Facet<'f>,
    {
        Self::load_json_with_limits(path, lines, opts, RemoteLoadLimits::default())
    }

    /// Load JSON with custom limits for HTTP sources.
    pub fn load_json_with_limits<E>(
        path: &str,
        lines: bool,
        opts: TypedLoadOptions,
        limits: RemoteLoadLimits,
    ) -> Result<Vec<E>>
    where
        E: DeserializeOwned + for<'f> Facet<'f>,
    {
        let examples = projection::load(&opts, |rows| {
            Self::visit_json_rows(path, lines, limits, |row| rows.push(row))
        })?;
        debug!(examples = examples.len(), "typed json examples loaded");
        Ok(examples)
    }

    #[tracing::instrument(
        name = "dsrs.data.load_json_with",
        level = "debug",
        skip(path, opts, mapper),
        fields(
            is_url = is_url(path),
            lines,
            field_map_entries = opts.field_map.len(),
            unknown_fields = ?opts.unknown_fields
        )
    )]
    /// Load rows from JSON/JSONL and map each row via a custom closure.
    ///
    /// This bypasses schema-driven conversion and gives full control to the caller.
    /// `opts` is accepted for API parity with non-mapper loaders.
    pub fn load_json_with<E, F>(
        path: &str,
        lines: bool,
        opts: TypedLoadOptions,
        mapper: F,
    ) -> Result<Vec<E>>
    where
        F: Fn(&RowRecord) -> Result<E>,
    {
        Self::load_json_with_mapper_limits(path, lines, opts, RemoteLoadLimits::default(), mapper)
    }

    /// Map JSON rows with custom limits for HTTP sources.
    pub fn load_json_with_mapper_limits<E, F>(
        path: &str,
        lines: bool,
        _opts: TypedLoadOptions,
        limits: RemoteLoadLimits,
        mapper: F,
    ) -> Result<Vec<E>>
    where
        F: Fn(&RowRecord) -> Result<E>,
    {
        let mut rows = Vec::new();
        Self::visit_json_rows(path, lines, limits, |row| rows.push(row))?;
        let examples = Self::rows_with_mapper(rows, mapper)?;
        debug!(
            examples = examples.len(),
            "typed json examples loaded via mapper"
        );
        Ok(examples)
    }

    #[cfg(feature = "data")]
    #[tracing::instrument(
        name = "dsrs.data.load_csv",
        level = "debug",
        skip(path, opts),
        fields(
            is_url = is_url(path),
            delimiter,
            has_headers,
            field_map_entries = opts.field_map.len(),
            unknown_fields = ?opts.unknown_fields
        )
    )]
    /// Load typed rows from CSV.
    ///
    /// When `has_headers` is `false`, fields are exposed as `column_{idx}` for
    /// mapper-based paths. Shape-based paths should typically use headers.
    pub fn load_csv<E>(
        path: &str,
        delimiter: char,
        has_headers: bool,
        opts: TypedLoadOptions,
    ) -> Result<Vec<E>>
    where
        E: DeserializeOwned + for<'f> Facet<'f>,
    {
        Self::load_csv_with_limits(
            path,
            delimiter,
            has_headers,
            opts,
            RemoteLoadLimits::default(),
        )
    }

    /// Load CSV with custom limits for HTTP sources.
    #[cfg(feature = "data")]
    pub fn load_csv_with_limits<E>(
        path: &str,
        delimiter: char,
        has_headers: bool,
        opts: TypedLoadOptions,
        limits: RemoteLoadLimits,
    ) -> Result<Vec<E>>
    where
        E: DeserializeOwned + for<'f> Facet<'f>,
    {
        let examples = projection::load(&opts, |rows| {
            Self::visit_csv_rows(
                path,
                delimiter,
                has_headers,
                limits,
                rows.source_fields().as_ref(),
                |row| rows.push(row),
            )
        })?;
        debug!(examples = examples.len(), "typed csv examples loaded");
        Ok(examples)
    }

    #[cfg(feature = "data")]
    #[tracing::instrument(
        name = "dsrs.data.load_csv_with",
        level = "debug",
        skip(path, opts, mapper),
        fields(
            is_url = is_url(path),
            delimiter,
            has_headers,
            field_map_entries = opts.field_map.len(),
            unknown_fields = ?opts.unknown_fields
        )
    )]
    /// Load rows from CSV and map each row via a custom closure.
    ///
    /// This bypasses schema-driven conversion and gives full control to the caller.
    /// `opts` is accepted for API parity with non-mapper loaders.
    pub fn load_csv_with<E, F>(
        path: &str,
        delimiter: char,
        has_headers: bool,
        opts: TypedLoadOptions,
        mapper: F,
    ) -> Result<Vec<E>>
    where
        F: Fn(&RowRecord) -> Result<E>,
    {
        Self::load_csv_with_mapper_limits(
            path,
            delimiter,
            has_headers,
            opts,
            RemoteLoadLimits::default(),
            mapper,
        )
    }

    /// Map CSV rows with custom limits for HTTP sources.
    #[cfg(feature = "data")]
    pub fn load_csv_with_mapper_limits<E, F>(
        path: &str,
        delimiter: char,
        has_headers: bool,
        _opts: TypedLoadOptions,
        limits: RemoteLoadLimits,
        mapper: F,
    ) -> Result<Vec<E>>
    where
        F: Fn(&RowRecord) -> Result<E>,
    {
        let rows = Self::load_csv_rows(path, delimiter, has_headers, limits)?;
        let examples = Self::rows_with_mapper(rows, mapper)?;
        debug!(
            examples = examples.len(),
            "typed csv examples loaded via mapper"
        );
        Ok(examples)
    }

    #[cfg(feature = "data")]
    #[tracing::instrument(
        name = "dsrs.data.load_parquet",
        level = "debug",
        skip(opts),
        fields(
            field_map_entries = opts.field_map.len(),
            unknown_fields = ?opts.unknown_fields
        )
    )]
    /// Load typed rows from a local Parquet file.
    pub fn load_parquet<E>(path: &str, opts: TypedLoadOptions) -> Result<Vec<E>>
    where
        E: DeserializeOwned + for<'f> Facet<'f>,
    {
        let examples = projection::load(&opts, |rows| {
            Self::visit_parquet_rows(Path::new(path), rows.source_fields().as_ref(), |row| {
                rows.push(row)
            })
        })?;
        debug!(examples = examples.len(), "typed parquet examples loaded");
        Ok(examples)
    }

    #[cfg(feature = "data")]
    #[tracing::instrument(
        name = "dsrs.data.load_parquet_with",
        level = "debug",
        skip(opts, mapper),
        fields(
            field_map_entries = opts.field_map.len(),
            unknown_fields = ?opts.unknown_fields
        )
    )]
    /// Load rows from Parquet and map each row via a custom closure.
    ///
    /// This bypasses schema-driven conversion and gives full control to the caller.
    /// `opts` is accepted for API parity with non-mapper loaders.
    pub fn load_parquet_with<E, F>(path: &str, opts: TypedLoadOptions, mapper: F) -> Result<Vec<E>>
    where
        F: Fn(&RowRecord) -> Result<E>,
    {
        let _ = opts;
        let rows = Self::load_parquet_rows(Path::new(path))?;
        let examples = Self::rows_with_mapper(rows, mapper)?;
        debug!(
            examples = examples.len(),
            "typed parquet examples loaded via mapper"
        );
        Ok(examples)
    }

    #[cfg(feature = "data")]
    #[tracing::instrument(
        name = "dsrs.data.load_hf",
        level = "debug",
        skip(opts),
        fields(
            dataset = dataset_name,
            subset,
            split,
            verbose,
            field_map_entries = opts.field_map.len(),
            unknown_fields = ?opts.unknown_fields
        )
    )]
    /// Load typed rows from a HuggingFace dataset split.
    ///
    /// Supports Parquet, JSON/JSONL, and CSV artifacts discovered in the dataset
    /// repo. `subset` and `split` are substring filters on artifact filenames.
    pub fn load_hf<E>(
        dataset_name: &str,
        subset: &str,
        split: &str,
        verbose: bool,
        opts: TypedLoadOptions,
    ) -> Result<Vec<E>>
    where
        E: DeserializeOwned + for<'f> Facet<'f>,
    {
        let examples = projection::load(&opts, |rows| {
            Self::visit_hf_rows(
                dataset_name,
                subset,
                split,
                verbose,
                rows.source_fields().as_ref(),
                |row| rows.push(row),
            )
        })?;
        debug!(examples = examples.len(), "typed hf examples loaded");
        Ok(examples)
    }

    #[cfg(feature = "data")]
    #[tracing::instrument(
        name = "dsrs.data.load_hf_with",
        level = "debug",
        skip(opts, mapper),
        fields(
            dataset = dataset_name,
            subset,
            split,
            verbose,
            field_map_entries = opts.field_map.len(),
            unknown_fields = ?opts.unknown_fields
        )
    )]
    /// Load rows from HuggingFace and map each row via a custom closure.
    ///
    /// This bypasses schema-driven conversion and gives full control to the caller.
    /// `opts` is accepted for API parity with non-mapper loaders.
    pub fn load_hf_with<E, F>(
        dataset_name: &str,
        subset: &str,
        split: &str,
        verbose: bool,
        opts: TypedLoadOptions,
        mapper: F,
    ) -> Result<Vec<E>>
    where
        F: Fn(&RowRecord) -> Result<E>,
    {
        let _ = opts;
        let mut rows = Vec::new();
        Self::visit_hf_rows(dataset_name, subset, split, verbose, None, |row| {
            rows.push(row)
        })?;
        let examples = Self::rows_with_mapper(rows, mapper)?;
        debug!(
            examples = examples.len(),
            "typed hf examples loaded via mapper"
        );
        Ok(examples)
    }

    #[cfg(feature = "data")]
    #[tracing::instrument(
        name = "dsrs.data.load_hf_from_parquet",
        level = "debug",
        skip(parquet_files, opts),
        fields(
            files = parquet_files.len(),
            field_map_entries = opts.field_map.len(),
            unknown_fields = ?opts.unknown_fields
        )
    )]
    /// Load typed rows from a local set of Parquet files.
    ///
    /// This is primarily used for deterministic/offline testing of HF-like data
    /// ingestion flows without network calls.
    pub fn load_hf_from_parquet<E>(
        parquet_files: Vec<PathBuf>,
        opts: TypedLoadOptions,
    ) -> Result<Vec<E>>
    where
        E: DeserializeOwned + for<'f> Facet<'f>,
    {
        let examples = projection::load(&opts, |rows| {
            Self::visit_parquet_files(&parquet_files, rows.source_fields().as_ref(), |row| {
                rows.push(row)
            })
        })?;
        debug!(
            examples = examples.len(),
            "typed hf parquet examples loaded"
        );
        Ok(examples)
    }

    fn rows_with_mapper<E, F>(rows: Vec<RowRecord>, mapper: F) -> Result<Vec<E>>
    where
        F: Fn(&RowRecord) -> Result<E>,
    {
        rows.into_iter()
            .map(|row| {
                mapper(&row).map_err(|err| DataLoadError::Mapper {
                    row: row.row_index,
                    source: err,
                })
            })
            .map(|result| result.map_err(anyhow::Error::from))
            .collect()
    }

    fn fetch_remote_bytes(
        path: &str,
        limits: RemoteLoadLimits,
    ) -> std::result::Result<(Vec<u8>, Option<String>), DataLoadError> {
        let started = Instant::now();
        let client = reqwest::blocking::Client::builder()
            .build()
            .map_err(|err| DataLoadError::Io(err.into()))?;
        let remaining = limits.timeout.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            return Err(remote_timeout(limits.timeout));
        }
        let mut response = client
            .get(path)
            // The request timeout also covers reads after the headers arrive.
            .timeout(remaining)
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .context("failed to fetch remote dataset")
            .map_err(DataLoadError::Io)?;
        if started.elapsed() >= limits.timeout {
            return Err(remote_timeout(limits.timeout));
        }
        let charset = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| {
                value.split(';').skip(1).find_map(|part| {
                    let (key, value) = part.trim().split_once('=')?;
                    key.trim()
                        .eq_ignore_ascii_case("charset")
                        .then(|| value.trim().trim_matches('"').to_owned())
                })
            });
        let max_bytes = u64::try_from(limits.max_bytes).unwrap_or(u64::MAX);
        if response.content_length().is_some_and(|len| len > max_bytes) {
            return Err(remote_limit("bytes", limits.max_bytes));
        }
        let mut bytes = Vec::new();
        let mut chunk = [0u8; 8192];
        loop {
            if started.elapsed() >= limits.timeout {
                return Err(remote_timeout(limits.timeout));
            }
            let remaining = limits.max_bytes.saturating_sub(bytes.len());
            let read_len = remaining.saturating_add(1).min(chunk.len());
            let count = response.read(&mut chunk[..read_len]).map_err(|err| {
                let timed_out = err
                    .get_ref()
                    .and_then(|source| source.downcast_ref::<reqwest::Error>())
                    .is_some_and(reqwest::Error::is_timeout);
                if timed_out {
                    DataLoadError::Io(anyhow!(err).context(format!(
                        "remote dataset timed out after {:?}",
                        limits.timeout
                    )))
                } else {
                    DataLoadError::Io(err.into())
                }
            })?;
            if started.elapsed() >= limits.timeout {
                return Err(remote_timeout(limits.timeout));
            }
            if count == 0 {
                break;
            }
            if count > remaining {
                return Err(remote_limit("bytes", limits.max_bytes));
            }
            bytes.extend_from_slice(&chunk[..count]);
        }
        Ok((bytes, charset))
    }

    fn fetch_text(
        path: &str,
        limits: RemoteLoadLimits,
    ) -> std::result::Result<String, DataLoadError> {
        if is_url(path) {
            let (bytes, charset) = Self::fetch_remote_bytes(path, limits)?;
            let encoding = charset
                .as_deref()
                .and_then(|label| encoding_rs::Encoding::for_label(label.as_bytes()))
                .unwrap_or(encoding_rs::UTF_8);
            let bytes = if encoding == encoding_rs::UTF_8 && !bytes.starts_with(b"\xef\xbb\xbf") {
                match String::from_utf8(bytes) {
                    Ok(text) => return Ok(text),
                    Err(err) => err.into_bytes(),
                }
            } else {
                bytes
            };
            let (text, _, _) = encoding.decode(&bytes);
            Ok(text.into_owned())
        } else {
            fs::read_to_string(path).map_err(|err| DataLoadError::Io(err.into()))
        }
    }

    fn visit_json_rows(
        path: &str,
        lines: bool,
        limits: RemoteLoadLimits,
        mut visit: impl FnMut(RowRecord),
    ) -> std::result::Result<(), DataLoadError> {
        let remote = is_url(path);
        if lines && !remote {
            return Self::visit_local_jsonl(path, visit);
        }
        let data = Self::fetch_text(path, limits)?;

        if lines {
            let mut rows = 0;
            for (idx, line) in data.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                if rows >= limits.max_rows {
                    return Err(remote_limit("rows", limits.max_rows));
                }
                visit(row_from_jsonl_line(line, idx + 1)?);
                rows += 1;
            }
            debug!(rows, "jsonl rows loaded");
            return Ok(());
        }

        let rows = if data.trim_start().starts_with('[') {
            let limit_hit = Cell::new(false);
            let mut deserializer = serde_json::Deserializer::from_str(&data);
            let (rows, shape_error) = JsonArrayRows {
                max_rows: remote.then_some(limits.max_rows),
                limit_hit: &limit_hit,
                visit,
            }
            .deserialize(&mut deserializer)
            .map_err(|err| {
                if limit_hit.get() {
                    remote_limit("rows", limits.max_rows)
                } else {
                    DataLoadError::Json(anyhow!(err))
                }
            })?;
            deserializer
                .end()
                .map_err(|err| DataLoadError::Json(anyhow!(err)))?;
            if let Some(error) = shape_error {
                return Err(error);
            }
            rows
        } else {
            let value: Value =
                serde_json::from_str(&data).map_err(|err| DataLoadError::Json(anyhow!(err)))?;
            if remote && limits.max_rows == 0 {
                return Err(remote_limit("rows", limits.max_rows));
            }
            visit(row_from_json_value(value, 1)?);
            1
        };

        debug!(rows, "json rows loaded");
        Ok(())
    }

    fn visit_local_jsonl(
        path: &str,
        mut visit: impl FnMut(RowRecord),
    ) -> std::result::Result<(), DataLoadError> {
        let file = fs::File::open(path).map_err(|err| DataLoadError::Io(err.into()))?;
        let mut reader = BufReader::with_capacity(64 * 1024, file);
        let mut line = String::new();
        let mut parsed = Ok(0usize);
        let mut row_index = 0;
        while reader
            .read_line(&mut line)
            .map_err(|err| DataLoadError::Io(err.into()))?
            != 0
        {
            row_index += 1;
            if let Ok(count) = &mut parsed
                && !line.trim().is_empty()
            {
                let text = line.strip_suffix('\n').map_or(line.as_str(), |text| {
                    text.strip_suffix('\r').unwrap_or(text)
                });
                match row_from_jsonl_line(text, row_index) {
                    Ok(row) => {
                        visit(row);
                        *count += 1;
                    }
                    Err(error) => parsed = Err(error),
                }
            }
            // Continue validating input UTF-8 so read errors retain precedence.
            line.clear();
        }
        let rows = parsed?;
        debug!(rows, "jsonl rows loaded");
        Ok(())
    }

    #[cfg(feature = "data")]
    fn load_csv_rows(
        path: &str,
        delimiter: char,
        has_headers: bool,
        limits: RemoteLoadLimits,
    ) -> std::result::Result<Vec<RowRecord>, DataLoadError> {
        let mut rows = Vec::new();
        Self::visit_csv_rows(path, delimiter, has_headers, limits, None, |row| {
            rows.push(row)
        })?;
        Ok(rows)
    }

    #[cfg(feature = "data")]
    fn visit_csv_rows(
        path: &str,
        delimiter: char,
        has_headers: bool,
        limits: RemoteLoadLimits,
        source_fields: Option<&HashSet<&str>>,
        visit: impl FnMut(RowRecord),
    ) -> std::result::Result<(), DataLoadError> {
        if is_url(path) {
            let (bytes, _) = Self::fetch_remote_bytes(path, limits).map_err(|err| match err {
                DataLoadError::Io(source) => DataLoadError::Csv(source),
                other => other,
            })?;

            let cursor = Cursor::new(bytes);
            let mut reader = ReaderBuilder::new()
                .delimiter(delimiter as u8)
                .has_headers(has_headers)
                .from_reader(cursor);
            return Self::visit_csv_records(
                &mut reader,
                has_headers,
                Some(limits.max_rows),
                source_fields,
                visit,
            );
        }

        let mut reader = ReaderBuilder::new()
            .delimiter(delimiter as u8)
            .has_headers(has_headers)
            .from_path(path)
            .map_err(|err| DataLoadError::Csv(err.into()))?;
        Self::visit_csv_records(&mut reader, has_headers, None, source_fields, visit)
    }

    #[cfg(feature = "data")]
    fn visit_csv_records<R: std::io::Read>(
        reader: &mut csv::Reader<R>,
        has_headers: bool,
        max_rows: Option<usize>,
        source_fields: Option<&HashSet<&str>>,
        mut visit: impl FnMut(RowRecord),
    ) -> std::result::Result<(), DataLoadError> {
        let header_names = if has_headers {
            Some(
                reader
                    .headers()
                    .map_err(|err| DataLoadError::Csv(err.into()))?
                    .clone(),
            )
        } else {
            None
        };

        let mut columns = None;
        let mut count = 0;
        let mut record = StringRecord::new();
        loop {
            let read = reader.read_record(&mut record);
            if matches!(&read, Ok(false)) {
                break;
            }
            if let Some(limit) = max_rows
                && count >= limit
            {
                return Err(remote_limit("rows", limit));
            }
            read.map_err(|err| DataLoadError::Csv(err.into()))?;
            count += 1;
            let columns = columns.get_or_insert_with(|| {
                (0..record.len())
                    .filter_map(|index| {
                        let name = header_names
                            .as_ref()
                            .and_then(|headers| headers.get(index))
                            .map(str::to_owned)
                            .unwrap_or_else(|| format!("column_{index}"));
                        source_fields
                            .is_none_or(|fields| fields.contains(name.as_str()))
                            .then_some((index, name))
                    })
                    .collect::<Vec<_>>()
            });
            let values = columns
                .iter()
                .map(|(index, name)| (name.clone(), parse_csv_cell(&record[*index])))
                .collect();
            visit(RowRecord {
                row_index: count,
                values,
            });
        }

        debug!(rows = count, "csv rows loaded");
        Ok(())
    }

    #[cfg(feature = "data")]
    fn load_parquet_rows(path: &Path) -> std::result::Result<Vec<RowRecord>, DataLoadError> {
        let mut rows = Vec::new();
        Self::visit_parquet_rows(path, None, |row| rows.push(row))?;
        Ok(rows)
    }

    #[cfg(feature = "data")]
    fn visit_parquet_rows(
        path: &Path,
        source_fields: Option<&HashSet<&str>>,
        mut visit: impl FnMut(RowRecord),
    ) -> std::result::Result<(), DataLoadError> {
        let file = fs::File::open(path).map_err(|err| DataLoadError::Parquet(err.into()))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|err| DataLoadError::Parquet(err.into()))?;
        let reader = builder
            .build()
            .map_err(|err| DataLoadError::Parquet(err.into()))?;

        let mut emitted = 0;
        let mut row_index = 1usize;

        for batch_result in reader {
            let batch = batch_result.map_err(|err| DataLoadError::Parquet(err.into()))?;
            let schema = batch.schema();
            let mut omitted = Vec::new();
            let columns: Vec<_> = schema
                .fields()
                .iter()
                .zip(batch.columns())
                .filter_map(|(field, column)| {
                    let read = parquet_column_reader(column.as_ref())?;
                    let name = field.name().as_str();
                    if source_fields.is_none_or(|fields| fields.contains(name)) {
                        Some((name, read))
                    } else {
                        omitted.push(column.as_ref());
                        None
                    }
                })
                .collect();

            for local_row in 0..batch.num_rows() {
                let mut values = HashMap::new();

                for &(name, ref read) in &columns {
                    if let Some(value) = read(local_row) {
                        values.insert(name.to_owned(), value);
                    }
                }

                // An unprojected value can still make this source row exist.
                if !values.is_empty() || omitted.iter().any(|column| !column.is_null(local_row)) {
                    visit(RowRecord { row_index, values });
                    emitted += 1;
                }
                row_index += 1;
            }
        }

        debug!(rows = emitted, "parquet rows loaded");
        Ok(())
    }

    #[cfg(feature = "data")]
    fn visit_parquet_files(
        parquet_files: &[PathBuf],
        source_fields: Option<&HashSet<&str>>,
        mut visit: impl FnMut(RowRecord),
    ) -> std::result::Result<(), DataLoadError> {
        let mut next_index = 1usize;

        for file in parquet_files {
            Self::visit_parquet_rows(file, source_fields, |mut row| {
                row.row_index = next_index;
                next_index += 1;
                visit(row);
            })?;
        }

        Ok(())
    }

    #[cfg(feature = "data")]
    fn visit_hf_rows(
        dataset_name: &str,
        subset: &str,
        split: &str,
        verbose: bool,
        source_fields: Option<&HashSet<&str>>,
        visit: impl FnMut(RowRecord),
    ) -> std::result::Result<(), DataLoadError> {
        let api = Api::new().map_err(|err| DataLoadError::Hf(err.into()))?;
        let repo = api.dataset(dataset_name.to_string());
        let metadata = repo.info().map_err(|err| DataLoadError::Hf(err.into()))?;

        let files = metadata.siblings.into_iter().filter_map(|sibling| {
            let file = sibling.rfilename;

            if (!subset.is_empty() && !file.contains(subset))
                || (!split.is_empty() && !file.contains(split))
            {
                return None;
            }

            let format = HfFileFormat::from_name(&file)?;
            Some(
                repo.get(&file)
                    .map(|path| (format, path))
                    .map_err(|err| DataLoadError::Hf(err.into())),
            )
        });
        Self::visit_hf_files(files, verbose, source_fields, visit)
    }

    #[cfg(feature = "data")]
    fn visit_hf_files(
        files: impl IntoIterator<Item = std::result::Result<(HfFileFormat, PathBuf), DataLoadError>>,
        verbose: bool,
        source_fields: Option<&HashSet<&str>>,
        mut visit: impl FnMut(RowRecord),
    ) -> std::result::Result<(), DataLoadError> {
        let mut rows = 0;
        let mut emit = |mut row: RowRecord| {
            rows += 1;
            row.row_index = rows;
            visit(row);
        };
        for file in files {
            let (format, file_path) = file?;
            let path_str = file_path
                .to_str()
                .ok_or_else(|| DataLoadError::Io(anyhow!("invalid UTF-8 file path")))?;

            if verbose {
                println!("Loading file: {path_str}");
            }

            let limits = RemoteLoadLimits::default();
            match format {
                HfFileFormat::Parquet => {
                    Self::visit_parquet_rows(&file_path, source_fields, &mut emit)?
                }
                HfFileFormat::Json { lines } => {
                    Self::visit_json_rows(path_str, lines, limits, &mut emit)?
                }
                HfFileFormat::Csv => {
                    Self::visit_csv_rows(path_str, ',', true, limits, source_fields, &mut emit)?
                }
            }
        }

        if verbose {
            println!("Loaded {rows} rows");
        }

        debug!(rows, "hf rows loaded");
        Ok(())
    }
}

#[cfg(feature = "data")]
enum HfFileFormat {
    Parquet,
    Json { lines: bool },
    Csv,
}

#[cfg(feature = "data")]
impl HfFileFormat {
    fn from_name(name: &str) -> Option<Self> {
        match name.rsplit_once('.')?.1 {
            "parquet" => Some(Self::Parquet),
            "json" => Some(Self::Json { lines: false }),
            "jsonl" => Some(Self::Json { lines: true }),
            "csv" => Some(Self::Csv),
            _ => None,
        }
    }
}

#[cfg(all(test, feature = "data"))]
mod hf_tests {
    use super::*;

    #[test]
    #[cfg_attr(miri, ignore = "mixed shard fixture uses native temporary files")]
    fn local_shards_share_dense_row_numbers_across_formats() {
        let dir = tempfile::tempdir().unwrap();
        let paths = [
            ("split-0.json", r#"[{"value":"one"}]"#),
            ("split-1.jsonl", "\n{\"value\":\"two\"}\n"),
            ("split-2.csv", "value,ignored\nthree,[1]\n"),
        ];
        let files = paths.map(|(name, data)| {
            let path = dir.path().join(name);
            fs::write(&path, data).unwrap();
            Ok((HfFileFormat::from_name(name).unwrap(), path))
        });
        let mut observed = Vec::new();
        let fields = HashSet::from(["value"]);
        DataLoader::visit_hf_files(files, false, Some(&fields), |row| {
            assert!(!row.values.contains_key("ignored"));
            observed.push((row.row_index, row.get::<String>("value").unwrap()));
        })
        .unwrap();
        assert_eq!(
            observed,
            [(1, "one".into()), (2, "two".into()), (3, "three".into())]
        );
    }
}
fn remote_limit(resource: &str, limit: usize) -> DataLoadError {
    DataLoadError::Io(anyhow!(
        "remote dataset exceeded {resource} limit ({limit})"
    ))
}

fn remote_timeout(timeout: Duration) -> DataLoadError {
    DataLoadError::Io(anyhow!("remote dataset timed out after {timeout:?}"))
}

struct JsonArrayRows<'a, F> {
    max_rows: Option<usize>,
    limit_hit: &'a Cell<bool>,
    visit: F,
}

impl<'de, F: FnMut(RowRecord)> DeserializeSeed<'de> for JsonArrayRows<'_, F> {
    type Value = (usize, Option<DataLoadError>);

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_seq(self)
    }
}

impl<'de, F: FnMut(RowRecord)> Visitor<'de> for JsonArrayRows<'_, F> {
    type Value = (usize, Option<DataLoadError>);

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a JSON array")
    }

    fn visit_seq<A>(mut self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut rows = 0;
        let mut shape_error = None;
        while self.max_rows.is_none_or(|limit| rows < limit) {
            let Some(item) = seq.next_element()? else {
                return Ok((rows, shape_error));
            };
            rows += 1;
            if shape_error.is_none() {
                match row_from_json_value(item, rows) {
                    Ok(row) => (self.visit)(row),
                    Err(error) if self.max_rows.is_some() => return Err(A::Error::custom(error)),
                    // For a local file, any later JSON syntax error takes precedence.
                    Err(error) => shape_error = Some(error),
                }
            }
        }
        if seq.next_element::<IgnoredAny>()?.is_some() {
            self.limit_hit.set(true);
            return Err(A::Error::custom("remote row limit exceeded"));
        }
        Ok((rows, shape_error))
    }
}

fn row_from_jsonl_line(
    line: &str,
    row_index: usize,
) -> std::result::Result<RowRecord, DataLoadError> {
    let value = serde_json::from_str(line).map_err(|err| DataLoadError::Json(anyhow!(err)))?;
    row_from_json_value(value, row_index)
}

fn row_from_json_value(
    value: serde_json::Value,
    row_index: usize,
) -> std::result::Result<RowRecord, DataLoadError> {
    let serde_json::Value::Object(object) = value else {
        return Err(DataLoadError::Json(anyhow!(
            "row {row_index}: expected JSON object, got {value}"
        )));
    };

    Ok(RowRecord {
        row_index,
        values: object.into_iter().collect(),
    })
}

#[cfg(feature = "data")]
fn parse_csv_cell(cell: &str) -> serde_json::Value {
    let trimmed = cell.trim();
    match trimmed {
        "" => return Value::String(String::new()),
        "true" => return Value::Bool(true),
        "false" => return Value::Bool(false),
        "null" => return Value::Null,
        _ => {}
    }

    if matches!(
        trimmed.as_bytes()[0],
        b'"' | b'{' | b'[' | b'-' | b'0'..=b'9'
    ) && let Ok(value) = serde_json::from_str(trimmed)
    {
        return value;
    }
    Value::String(cell.to_string())
}

#[cfg(feature = "data")]
type ParquetColumnReader<'a> = Box<dyn Fn(usize) -> Option<Value> + 'a>;

#[cfg(feature = "data")]
fn parquet_column_reader(column: &dyn Array) -> Option<ParquetColumnReader<'_>> {
    let array = column.as_any();
    macro_rules! bind_columns {
        ($($kind:ty),+ $(,)?) => {
            $(
                if let Some(values) = array.downcast_ref::<$kind>() {
                    return Some(Box::new(move |row| {
                        (!values.is_null(row)).then(|| serde_json::json!(values.value(row)))
                    }));
                }
            )+
        };
    }
    bind_columns!(
        StringArray,
        BooleanArray,
        Int64Array,
        Int32Array,
        Int16Array,
        Int8Array,
        UInt64Array,
        UInt32Array,
        UInt16Array,
        UInt8Array,
        Float64Array,
        Float32Array,
    );
    None
}
