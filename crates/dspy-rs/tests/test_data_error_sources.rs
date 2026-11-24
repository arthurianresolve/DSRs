use dspy_rs::{DataLoadError, DataLoader, TypedLoadOptions};
use std::error::Error;

#[derive(Debug, serde::Deserialize, facet::Facet)]
struct Row {
    #[allow(dead_code)]
    value: String,
}

#[test]
fn typed_projection_consumes_a_source_after_its_final_alias() {
    #[derive(Debug, serde::Deserialize, facet::Facet)]
    struct AliasedRow {
        text: String,
        number: i64,
        optional: Option<String>,
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rows.jsonl");
    let load = || {
        DataLoader::load_json::<AliasedRow>(
            path.to_str().unwrap(),
            true,
            TypedLoadOptions {
                field_map: [
                    ("text".to_string(), "source".to_string()),
                    ("number".to_string(), "source".to_string()),
                ]
                .into(),
                unknown_fields: dspy_rs::UnknownFieldPolicy::Error,
            },
        )
    };
    std::fs::write(&path, r#"{"source":" 0042 "}"#).unwrap();
    let rows = load().unwrap();
    assert_eq!(rows[0].text, " 0042 ");
    assert_eq!(rows[0].number, 42);
    assert_eq!(rows[0].optional, None);

    std::fs::write(&path, r#"{"source":" 0042 ","unused":true}"#).unwrap();
    assert!(matches!(
        load().unwrap_err().downcast_ref::<DataLoadError>(),
        Some(DataLoadError::UnknownField { row: 1, field }) if field == "unused"
    ));
}

#[test]
fn remote_http_failures_keep_status_in_the_source_chain() {
    use std::io::{BufRead, BufReader, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/rows.json", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut reader = BufReader::new(&mut stream);
        let mut line = Vec::new();
        loop {
            line.clear();
            let count = reader.read_until(b'\n', &mut line).unwrap();
            assert_ne!(count, 0, "request ended before the headers");
            if line == b"\r\n" {
                break;
            }
        }
        stream
            .write_all(
                b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )
            .unwrap();
    });
    let error = DataLoader::load_json::<Row>(&url, false, TypedLoadOptions::default()).unwrap_err();
    server.join().unwrap();
    assert!(matches!(
        error.downcast_ref::<DataLoadError>(),
        Some(DataLoadError::Io(_))
    ));
    let http = error
        .chain()
        .find_map(|cause| cause.downcast_ref::<reqwest::Error>())
        .unwrap();
    assert_eq!(http.status(), Some(reqwest::StatusCode::TOO_MANY_REQUESTS));
}

#[test]
fn json_and_mapper_failures_keep_the_typed_original_cause() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rows.json");
    std::fs::write(&path, "not json").unwrap();
    let error =
        DataLoader::load_json::<Row>(path.to_str().unwrap(), false, TypedLoadOptions::default())
            .unwrap_err();
    let source = error.downcast_ref::<DataLoadError>().unwrap();
    assert!(matches!(source, DataLoadError::Json(_)));
    assert!(
        source
            .source()
            .unwrap()
            .downcast_ref::<serde_json::Error>()
            .is_some()
    );

    std::fs::write(&path, r#"[{"value":"ok"}]"#).unwrap();
    let error = DataLoader::load_json_with::<Row, _>(
        path.to_str().unwrap(),
        false,
        TypedLoadOptions::default(),
        |_| {
            Err(
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "mapper fixture cause")
                    .into(),
            )
        },
    )
    .unwrap_err();
    let source = error.downcast_ref::<DataLoadError>().unwrap();
    assert!(matches!(source, DataLoadError::Mapper { row: 1, .. }));
    assert_eq!(
        source
            .source()
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .unwrap()
            .kind(),
        std::io::ErrorKind::InvalidInput
    );
}

#[cfg(feature = "data")]
#[test]
fn typed_csv_reports_late_source_errors_before_projection_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rows.csv");
    let load = || {
        DataLoader::load_csv::<Row>(
            path.to_str().unwrap(),
            ',',
            true,
            TypedLoadOptions::default(),
        )
    };
    std::fs::write(&path, b"other\nfirst\ntoo,many\n").unwrap();
    assert!(matches!(
        load().unwrap_err().downcast_ref::<DataLoadError>(),
        Some(DataLoadError::Csv(_))
    ));

    std::fs::write(&path, b"other\nfirst\n").unwrap();
    assert!(matches!(
        load().unwrap_err().downcast_ref::<DataLoadError>(),
        Some(DataLoadError::MissingField { row: 1, .. })
    ));
}

#[test]
fn jsonl_mapper_keeps_line_numbers_and_the_unterminated_final_row() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rows.jsonl");
    std::fs::write(
        &path,
        b"\r\n{\"value\":\"first\"}\r\n \n{\"value\":\"last\"}",
    )
    .unwrap();

    let rows = DataLoader::load_json_with(
        path.to_str().unwrap(),
        true,
        TypedLoadOptions::default(),
        |row| {
            Ok((
                row.row_index,
                row.values["value"].as_str().unwrap().to_owned(),
            ))
        },
    )
    .unwrap();

    assert_eq!(rows, [(2, "first".to_owned()), (4, "last".to_owned())]);
}

#[test]
fn jsonl_read_errors_take_precedence_over_parse_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rows.jsonl");
    let mut bytes = b"not JSON\n".to_vec();
    bytes.resize(64 * 1024, b' ');
    bytes.extend_from_slice(b"\n\xff");
    std::fs::write(&path, bytes).unwrap();

    let error =
        DataLoader::load_json::<Row>(path.to_str().unwrap(), true, TypedLoadOptions::default())
            .unwrap_err();
    let source = error.downcast_ref::<DataLoadError>().unwrap();
    assert!(matches!(source, DataLoadError::Io(_)));
    let io_error = source
        .source()
        .unwrap()
        .downcast_ref::<std::io::Error>()
        .unwrap();
    assert_eq!(io_error.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn typed_jsonl_validates_the_file_before_deserializing_rows_in_order() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CALLS: AtomicUsize = AtomicUsize::new(0);
    #[derive(Debug, serde::Deserialize, facet::Facet)]
    struct Counted {
        #[serde(deserialize_with = "count")]
        value: i64,
    }
    fn count<'de, D: serde::Deserializer<'de>>(d: D) -> Result<i64, D::Error> {
        CALLS.fetch_add(1, Ordering::Relaxed);
        serde::Deserialize::deserialize(d)
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rows.jsonl");
    let load = || {
        DataLoader::load_json::<Counted>(path.to_str().unwrap(), true, TypedLoadOptions::default())
    };
    std::fs::write(&path, b"{\"value\":1}\nnot json").unwrap();
    assert!(matches!(
        load().unwrap_err().downcast_ref::<DataLoadError>(),
        Some(DataLoadError::Json(_))
    ));
    assert_eq!(CALLS.load(Ordering::Relaxed), 0);

    std::fs::write(&path, b"\n{\"value\":\"bad\"}\n{}").unwrap();
    assert!(matches!(
        load().unwrap_err().downcast_ref::<DataLoadError>(),
        Some(DataLoadError::TypeMismatch { row: 2, .. })
    ));
    assert_eq!(CALLS.load(Ordering::Relaxed), 1);
    std::fs::write(&path, b"{\"value\":7,\"unused\":\"extra\"}").unwrap();
    assert_eq!(load().unwrap()[0].value, 7);
}
