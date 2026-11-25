//! The load-bearing property of the structural checker: everything the full
//! parser accepts, [`dsrs_syntax::check`] accepts.
//!
//! Runs over local copies of the golden `.dsrs` fixtures maintained next to
//! the full parser (`crates/dspy-rs/tests/fixtures/*.dsrs`). Keep the copies
//! in sync when a fixture changes — `dspy-rs`'s `test_include_program.rs`
//! exercises the same artifacts through `include_program!`, so drift that
//! matters (a fixture the checker would reject) fails that suite too.

#[test]
fn parity_accepts_everything_the_full_parser_accepts() {
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let mut seen = 0usize;
    for entry in std::fs::read_dir(fixtures).expect("fixtures dir readable") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("dsrs") {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("fixture readable");
        dsrs_syntax::check(&src)
            .unwrap_or_else(|e| panic!("syntax checker rejected {}: {e}", path.display()));
        seen += 1;
    }
    assert!(
        seen >= 3,
        "expected the golden .dsrs fixtures, found {seen}"
    );
}
#[test]
fn raw_region_offsets_preserve_crlf_and_byte_columns() {
    use dsrs_syntax::lex::{Lexer, Span, Tok};

    let source = "````\r\n// café\r\n```\r\n````\r\n{\r\n\"文字\": [1]\r\n}\tname";
    let mut lexer = Lexer::new(source);
    let opener = lexer.next_token().unwrap();
    let (body, end) = lexer.scan_code_fence(opener.start).unwrap();
    assert_eq!(body, "// café\r\n```");
    let brace = lexer.resume_at(end).unwrap();
    assert_eq!(brace.tok, Tok::LBrace);
    assert_eq!(brace.span, Span { line: 5, col: 1 });

    let (_, end) = lexer.scan_json(brace.start).unwrap();
    let name = lexer.resume_at(end).unwrap();
    assert_eq!(name.tok, Tok::Ident("name".into()));
    assert_eq!(name.span, Span { line: 7, col: 3 });
    assert_eq!(lexer.resume_at(brace.start).unwrap().span, brace.span);
}
