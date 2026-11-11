"""Verify that supported-answer labels cannot change native model requests or predictions."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


CRITIQUE = "The answer agrees with the supporting context."


def provider_run(binary, fixture, steps, extra_args=()):
    requests = []

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            index = len(requests)
            requests.append(request)
            if index >= len(steps):
                self.send_error(500)
                return
            field, value = steps[index]
            properties = request["response_format"]["json_schema"]["schema"]["properties"]
            assert list(properties) == [field]
            payload = json.dumps({
                "id": "scoring-isolation",
                "object": "chat.completion",
                "created": 0,
                "model": request["model"],
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps({field: value})},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        environment = dict(
            os.environ,
            DSRS_PROFILE_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1",
            DSRS_PROFILE_MODEL="scoring-isolation",
        )
        environment.pop("DSRS_PROFILE_BARRIER", None)
        command = [
            str(binary), "1", "0", "--live", "--json-output", "--check-quality",
            "--retrieval-recipe", "--fixture", str(fixture), *map(str, extra_args),
        ]
        result = subprocess.run(
            command, env=environment, capture_output=True, text=True, timeout=60
        )
        report = json.loads(result.stdout) if result.stdout.strip() else None
        return result, report, requests
    finally:
        server.shutdown()
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    root = Path(__file__).resolve().parent
    original = json.loads((root / "fixture.json").read_text(encoding="utf-8"))
    office = original["examples"][1]
    strict_annotations = {
        "schema_version": 1,
        "annotation_set_id": "strict-only-scoring-isolation-v1",
        "metric": "context_supported_normalized_exact_match",
        "description": "Strict-only control used to verify scoring isolation.",
        "examples": [{
            "id": office["id"],
            "question": office["question"],
            "context": office["context"],
            "reference_answer": office["answer"],
            "accepted_answers": [office["answer"]],
            "rationale": "Strict-only isolation control.",
        }],
    }
    steps = []
    for answer in ("yes", "ambassador", "Animorphs"):
        steps.extend((("answer", answer), ("critique", CRITIQUE), ("answer", answer)))

    with tempfile.TemporaryDirectory(prefix="dsrs-scoring-isolation-") as directory:
        strict_path = Path(directory) / "strict-annotations.json"
        strict_path.write_text(json.dumps(strict_annotations), encoding="utf-8")
        supported_result, supported, supported_requests = provider_run(
            binary, root / "fixture.json", steps
        )
        strict_result, strict, strict_requests = provider_run(
            binary,
            root / "fixture.json",
            steps,
            ("--supported-answers", strict_path),
        )

    assert supported_result.returncode == strict_result.returncode == 2, (
        supported_result.returncode,
        supported_result.stderr,
        strict_result.returncode,
        strict_result.stderr,
    )
    assert supported_requests == strict_requests
    stable_fields = ("id", "draft", "critique", "revised_answer", "answer", "context")
    assert [tuple(sample[field] for field in stable_fields) for sample in supported["samples"]] == [
        tuple(sample[field] for field in stable_fields) for sample in strict["samples"]
    ]
    assert [sample["exact_match"] for sample in supported["samples"]] == [True, False, True]
    assert [sample["exact_match"] for sample in strict["samples"]] == [True, False, True]
    assert [sample["supported_answer_match"] for sample in supported["samples"]] == [True, True, True]
    assert [sample["supported_answer_match"] for sample in strict["samples"]] == [True, False, True]
    assert supported["quality_status"] == strict["quality_status"] == "failed"
    assert supported["supported_answer_quality_status"] == "passed"
    assert strict["supported_answer_quality_status"] == "failed"
    assert supported["scoring"]["supported_annotation_set"] == (
        "hotpotqa-supported-answers-2026-11-11-v1"
    )
    assert strict["scoring"]["supported_annotation_set"] == (
        "strict-only-scoring-isolation-v1"
    )

    clarified_steps = [
        ("answer", "Chief of Protocol"),
        ("critique", CRITIQUE),
        ("answer", "Chief of Protocol"),
    ]
    clarified_result, clarified, clarified_requests = provider_run(
        binary, root / "clarified-office-fixture.json", clarified_steps
    )
    assert clarified_result.returncode == 0, clarified_result.stderr
    assert len(clarified_requests) == 3
    assert clarified["quality_status"] == clarified["supported_answer_quality_status"] == "passed"
    assert clarified["samples"][0]["exact_match"]
    assert not clarified["samples"][0]["supported_answer_annotation"]

    print(
        "Passed scoring isolation: changing supported-answer annotations left all 9 "
        "parsed request payloads and predictions structurally unchanged; strict remained "
        "2/3, supported changed from 2/3 to 3/3, and the clarified strict gate passed 1/1."
    )


if __name__ == "__main__":
    main()
