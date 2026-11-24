#![cfg(feature = "data")]

use dspy_rs::{DataLoadError, DataLoader, RemoteLoadLimits, TypedLoadOptions};
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, facet::Facet, serde::Deserialize)]
#[facet(crate = facet)]
struct Row {
    question: String,
    answer: String,
}

fn accept_request(listener: &TcpListener) -> TcpStream {
    let timeout = Duration::from_secs(5);
    let deadline = Instant::now() + timeout;
    listener.set_nonblocking(true).unwrap();
    let mut stream = loop {
        match listener.accept() {
            Ok((stream, _)) => break stream,
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                assert!(Instant::now() < deadline, "client did not connect");
                thread::sleep(Duration::from_millis(5));
            }
            Err(error) => panic!("failed to accept the request: {error}"),
        }
    };
    stream.set_nonblocking(false).unwrap();
    stream.set_read_timeout(Some(timeout)).unwrap();
    stream.set_write_timeout(Some(timeout)).unwrap();
    read_request_headers(&mut stream);
    stream
}

fn read_request_headers(stream: &mut TcpStream) {
    let mut reader = BufReader::new(stream);
    let mut line = Vec::new();
    loop {
        line.clear();
        let count = reader.read_until(b'\n', &mut line).unwrap();
        assert_ne!(count, 0, "request ended before the headers");
        if line == b"\r\n" {
            return;
        }
    }
}

fn serve(body: &str, delay: Duration) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = listener.local_addr().unwrap().port();
    let body = body.to_owned();
    let handle = thread::spawn(move || {
        let mut stream = accept_request(&listener);
        thread::sleep(delay);
        let response = format!(
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:X}\r\n{body}\r\n0\r\n\r\n",
            body.len(),
        );
        let _ = stream.write_all(response.as_bytes());
    });
    // The URL-shaped path keeps this local endpoint on the existing HTTP loader path.
    (format!("http://127.0.0.1:{port}/dataset.com"), handle)
}

fn serve_bytes(body: Vec<u8>, content_type: &str) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = listener.local_addr().unwrap().port();
    let content_type = content_type.to_owned();
    let handle = thread::spawn(move || {
        let mut stream = accept_request(&listener);
        let headers = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        stream.write_all(headers.as_bytes()).unwrap();
        stream.write_all(&body).unwrap();
    });
    (format!("http://127.0.0.1:{port}/dataset.com"), handle)
}

#[test]
fn remote_json_enforces_bytes_and_rows_and_accepts_valid_data() {
    let body = r#"[{"question":"Q1","answer":"A1"},{"question":"Q2","answer":"A2"}]"#;
    let mut limits = RemoteLoadLimits {
        max_bytes: body.len() - 1,
        max_rows: 2,
        timeout: Duration::from_secs(2),
    };

    let (url, server) = serve(body, Duration::ZERO);
    let err =
        DataLoader::load_json_with_limits::<Row>(&url, false, TypedLoadOptions::default(), limits)
            .unwrap_err();
    assert!(err.to_string().contains("bytes limit"), "{err}");
    server.join().unwrap();

    limits.max_bytes = body.len();
    limits.max_rows = 1;
    let (url, server) = serve(body, Duration::ZERO);
    let err =
        DataLoader::load_json_with_limits::<Row>(&url, false, TypedLoadOptions::default(), limits)
            .unwrap_err();
    assert!(err.to_string().contains("rows limit"), "{err}");
    server.join().unwrap();

    limits.max_rows = 2;
    let (url, server) = serve(body, Duration::ZERO);
    let rows =
        DataLoader::load_json_with_limits::<Row>(&url, false, TypedLoadOptions::default(), limits)
            .unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[1].answer, "A2");
    server.join().unwrap();
}

#[test]
fn remote_csv_enforces_row_limit() {
    let body = "question,answer\nQ1,A1\nQ2,A2\n";
    let limits = RemoteLoadLimits {
        max_bytes: body.len(),
        max_rows: 1,
        timeout: Duration::from_secs(2),
    };
    let (url, server) = serve(body, Duration::ZERO);
    let err = DataLoader::load_csv_with_limits::<Row>(
        &url,
        ',',
        true,
        TypedLoadOptions::default(),
        limits,
    )
    .unwrap_err();
    assert!(err.to_string().contains("rows limit"), "{err}");
    server.join().unwrap();
}

#[test]
fn remote_json_honors_deadline() {
    let body = r#"{"question":"Q","answer":"A"}"#;
    let (url, server) = serve(body, Duration::from_millis(1_500));
    let limits = RemoteLoadLimits {
        timeout: Duration::from_secs(1),
        ..RemoteLoadLimits::default()
    };
    let err =
        DataLoader::load_json_with_limits::<Row>(&url, false, TypedLoadOptions::default(), limits)
            .unwrap_err();
    let DataLoadError::Io(source) = err.downcast_ref::<DataLoadError>().unwrap() else {
        panic!("unexpected error: {err}");
    };
    let detail = source
        .chain()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(": ");
    assert!(detail.to_lowercase().contains("timed out"), "{detail}");
    server.join().unwrap();
}

#[test]
fn remote_jsonl_mapper_and_csv_share_limits() {
    let jsonl =
        "{\"question\":\"Q1\",\"answer\":\"A1\"}\n{\"question\":\"Q2\",\"answer\":\"A2\"}\n";
    let limits = RemoteLoadLimits {
        max_bytes: jsonl.len(),
        max_rows: 1,
        timeout: Duration::from_secs(2),
    };
    let (url, server) = serve(jsonl, Duration::ZERO);
    let err = DataLoader::load_json_with_mapper_limits(
        &url,
        true,
        TypedLoadOptions::default(),
        limits,
        |row| row.get::<String>("question").map_err(Into::into),
    )
    .unwrap_err();
    assert!(err.to_string().contains("rows limit"), "{err}");
    server.join().unwrap();

    let csv = "question,answer\nQ1,A1\n";
    let (url, server) = serve(csv, Duration::ZERO);
    let err = DataLoader::load_csv_with_limits::<Row>(
        &url,
        ',',
        true,
        TypedLoadOptions::default(),
        RemoteLoadLimits {
            max_bytes: csv.len() - 1,
            ..limits
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("bytes limit"), "{err}");
    server.join().unwrap();
}

#[test]
fn remote_json_preserves_charset_and_bom_decoding() {
    let body = b"{\"question\":\"caf\xe9\",\"answer\":\"yes\"}".to_vec();
    let (url, server) = serve_bytes(body, "application/json; charset=windows-1252");
    let rows = DataLoader::load_json_with_limits::<Row>(
        &url,
        false,
        TypedLoadOptions::default(),
        RemoteLoadLimits::default(),
    )
    .unwrap();
    assert_eq!(rows[0].question, "café");
    server.join().unwrap();

    let body = b"\xef\xbb\xbf{\"question\":\"Q\",\"answer\":\"A\"}".to_vec();
    let (url, server) = serve_bytes(body, "application/json");
    let rows = DataLoader::load_json_with_limits::<Row>(
        &url,
        false,
        TypedLoadOptions::default(),
        RemoteLoadLimits::default(),
    )
    .unwrap();
    assert_eq!(rows[0].answer, "A");
    server.join().unwrap();

    let body = b"{\"question\":\"caf\xe9\",\"answer\":\"yes\"}".to_vec();
    let (url, server) = serve_bytes(body, "application/json");
    let rows = DataLoader::load_json_with_limits::<Row>(
        &url,
        false,
        TypedLoadOptions::default(),
        RemoteLoadLimits::default(),
    )
    .unwrap();
    assert_eq!(rows[0].question, "caf\u{fffd}");
    server.join().unwrap();
}

#[test]
fn remote_json_array_reports_invalid_row() {
    let (url, server) = serve(r#"[{"question":"Q","answer":"A"},null]"#, Duration::ZERO);
    let err = DataLoader::load_json_with_limits::<Row>(
        &url,
        false,
        TypedLoadOptions::default(),
        RemoteLoadLimits::default(),
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("row 2: expected JSON object"),
        "{err}"
    );
    server.join().unwrap();
}

#[test]
fn remote_json_array_stops_at_row_limit() {
    let body = format!("[{}]", vec!["{}"; 10_000].join(","));
    let (url, server) = serve(&body, Duration::ZERO);
    let err = DataLoader::load_json_with_limits::<Row>(
        &url,
        false,
        TypedLoadOptions::default(),
        RemoteLoadLimits {
            max_rows: 1,
            ..RemoteLoadLimits::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("rows limit"), "{err}");
    server.join().unwrap();
}

#[test]
fn remote_json_deadline_includes_stalled_body_after_slow_headers() {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = listener.local_addr().unwrap().port();
    let (done, keep_open) = std::sync::mpsc::channel::<()>();
    let server = thread::spawn(move || {
        let mut stream = accept_request(&listener);
        thread::sleep(Duration::from_millis(1_100));
        let _ = stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 64\r\nConnection: close\r\n\r\n{");
        let _ = keep_open.recv_timeout(Duration::from_secs(5));
    });
    let started = Instant::now();
    let error = DataLoader::load_json_with_limits::<Row>(
        &format!("http://127.0.0.1:{port}/dataset.com"),
        false,
        TypedLoadOptions::default(),
        RemoteLoadLimits {
            timeout: Duration::from_secs(2),
            ..RemoteLoadLimits::default()
        },
    )
    .unwrap_err();
    let elapsed = started.elapsed();
    drop(done);
    server.join().unwrap();

    assert!(error.to_string().contains("timed out"), "{error:#}");
    assert!(elapsed < Duration::from_millis(2_750), "{elapsed:?}");
}

#[test]
fn remote_json_rejects_slow_trickle_after_total_deadline() {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = thread::spawn(move || {
        let mut stream = accept_request(&listener);
        stream
            .write_all(
                b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
            )
            .unwrap();
        for _ in 0..8 {
            if stream.write_all(b"1\r\n{\r\n").is_err() {
                return;
            }
            thread::sleep(Duration::from_millis(175));
        }
        let _ = stream.write_all(b"0\r\n\r\n");
    });
    let err = DataLoader::load_json_with_limits::<Row>(
        &format!("http://127.0.0.1:{port}/dataset.com"),
        false,
        TypedLoadOptions::default(),
        RemoteLoadLimits {
            timeout: Duration::from_secs(1),
            ..RemoteLoadLimits::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("timed out"), "{err}");
    server.join().unwrap();
}
