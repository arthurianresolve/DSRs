/// Returns `true` for an explicit HTTP(S) URL scheme.
pub fn is_url(path: &str) -> bool {
    path.split_once("://").is_some_and(|(scheme, _)| {
        scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https")
    })
}

#[cfg(test)]
mod tests {
    use super::is_url;

    #[test]
    fn routes_only_explicit_http_schemes() {
        assert!(is_url("http://127.0.0.1:8080/dataset"));
        assert!(is_url("HTTP://localhost/dataset"));
        assert!(is_url("https://[::1]/rows"));
        assert!(!is_url("datasets/http://example.org/rows"));
        assert!(!is_url("file:///tmp/data.json"));
    }
}
