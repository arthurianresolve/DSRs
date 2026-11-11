import json
import ast
import re
import threading
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import live
import scoring


class ProfilingTests(unittest.TestCase):
    def quality_runs(self):
        return [{"framework": framework, "instrumented": instrumented,
                 "pipelines": 3, "exact_matches": 3, "supported_answer_matches": 3}
                for framework, instrumented in
                [("dsrs", False), ("dspy", False), ("dspy", False), ("dsrs", False),
                 ("dsrs", True), ("dspy", True)]]

    def test_equal_zero_quality_does_not_pass_the_quality_gate(self):
        runs = self.quality_runs()
        for run in runs:
            run["exact_matches"] = 0
        outcome = live.quality_verdict(runs)
        self.assertEqual(outcome["quality_status"], "failed")
        self.assertEqual(outcome["supported_answer_quality_status"], "passed")
        self.assertFalse(outcome["equal_quality_comparison_eligible"])
        self.assertEqual(outcome["by_framework"]["dsrs"]["timing"],
                         {"exact_matches": 0, "supported_answer_matches": 6, "samples": 6})
        self.assertTrue(all(r["exact_matches"] == 0 for r in runs))

    def test_quality_failure_in_either_framework_or_profile_is_retained(self):
        for index in (0, 1, 4, 5):
            with self.subTest(index=index):
                runs = self.quality_runs()
                runs[index]["exact_matches"] = 2
                outcome = live.quality_verdict(runs)
                self.assertEqual(outcome["quality_status"], "failed")
                self.assertFalse(outcome["equal_quality_comparison_eligible"])
                self.assertEqual(runs[index]["exact_matches"], 2)

    def test_quality_gate_accepts_correct_answers_from_both_frameworks(self):
        outcome = live.quality_verdict(self.quality_runs())
        self.assertEqual(outcome["quality_status"], "passed")
        self.assertEqual(outcome["supported_answer_quality_status"], "passed")
        self.assertTrue(outcome["equal_quality_comparison_eligible"])

    def test_supported_match_never_replaces_the_strict_quality_gate(self):
        runs = self.quality_runs()
        runs[0]["exact_matches"] = 2
        outcome = live.quality_verdict(runs)
        self.assertEqual(outcome["quality_status"], "failed")
        self.assertEqual(outcome["supported_answer_quality_status"], "passed")
        self.assertFalse(outcome["equal_quality_comparison_eligible"])

    def test_missing_or_mismatched_quality_evidence_cannot_pass(self):
        for runs in ([], self.quality_runs()[:1], self.quality_runs()[:-1], self.quality_runs()[1:]):
            with self.subTest(runs=runs):
                with self.assertRaises(ValueError):
                    live.quality_verdict(runs)

    def test_task_instructions_match_across_frameworks(self):
        here = Path(__file__).resolve().parent
        source = (here.parents[2] / "crates/dspy-rs/examples/bench-hotpotqa-reflection.rs").read_text(encoding="utf-8")
        rust = {name: text for text, name in re.findall(r"/// (.+)\nstruct (Answer|Reflect|Revise)", source)}
        python = {node.name: ast.get_docstring(node) for node in ast.parse((here / "bench_dspy.py").read_text(encoding="utf-8")).body
                  if isinstance(node, ast.ClassDef) and node.name in {"Answer", "Reflect", "Revise"}}
        self.assertEqual(set(rust), {"Answer", "Reflect", "Revise"})
        self.assertEqual(rust, python)

    def test_supported_annotations_are_frozen_to_the_reviewed_inputs(self):
        here = Path(__file__).resolve().parent
        annotations = scoring.load_annotations()
        self.assertEqual(annotations.annotation_set_id,
                         "hotpotqa-supported-answers-2026-11-11-v1")
        original = json.loads((here / "fixture.json").read_text(encoding="utf-8"))["examples"][1]
        score = scoring.score_answer(annotations, original, "ambassador")
        self.assertFalse(score["exact_match"])
        self.assertTrue(score["supported_answer_match"])
        self.assertTrue(score["supported_answer_annotation"])
        prospective = json.loads(
            (here / "prospective-fixture.json").read_text(encoding="utf-8")
        )["examples"][0]
        prospective_score = scoring.score_answer(annotations, prospective, "ambassador")
        self.assertFalse(prospective_score["exact_match"])
        self.assertTrue(prospective_score["supported_answer_match"])
        changed = dict(original, question=original["question"] + " changed")
        with self.assertRaisesRegex(ValueError, "does not match the frozen"):
            scoring.score_answer(annotations, changed, "ambassador")
        with tempfile.TemporaryDirectory() as directory:
            incomplete = Path(directory) / "annotations.json"
            incomplete.write_text(json.dumps({
                "schema_version": 1,
                "annotation_set_id": "incomplete-v1",
                "metric": scoring.SUPPORTED_METRIC,
                "description": "missing examples",
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "examples must be a list"):
                scoring.load_annotations(incomplete)

    def test_clarified_office_fixture_changes_only_the_question_contract(self):
        here = Path(__file__).resolve().parent
        original = json.loads((here / "fixture.json").read_text(encoding="utf-8"))["examples"][1]
        clarified = json.loads((here / "clarified-office-fixture.json").read_text(encoding="utf-8"))["examples"][0]
        self.assertEqual(clarified["context"], original["context"])
        self.assertEqual(clarified["answer"], original["answer"])
        self.assertNotEqual(clarified["id"], original["id"])
        self.assertNotEqual(clarified["question"], original["question"])
        self.assertIn("other than ambassador", clarified["question"])

    def test_interference_retries_preserve_attempts_and_incorrect_answers(self):
        data = {"samples": [{"exact_match": False}]}
        env = {"DSRS_PROFILE_BASE_URL": "http://localhost:1234/00-dsrs-timing/v1"}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(live, "wait_idle"), patch.object(live, "run", side_effect=[
                live.CompilerInterference("busy"), data
            ]) as execute:
                name, actual = live.run_controlled([], env, Path(directory), "00-dsrs-timing", None, 1)
            self.assertEqual(name, "00-dsrs-timing-retry-1")
            self.assertIs(actual, data)
            self.assertFalse(actual["samples"][0]["exact_match"])
            self.assertEqual(execute.call_args_list[0].args[3], "00-dsrs-timing")
            self.assertEqual(execute.call_args_list[1].args[1]["DSRS_PROFILE_BASE_URL"],
                             "http://localhost:1234/00-dsrs-timing-retry-1/v1")
            attempts = json.loads((Path(directory) / "00-dsrs-timing.attempts.json").read_text())
            self.assertEqual([a["accepted"] for a in attempts], [False, True])

    def test_provider_failures_are_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(live, "wait_idle"), patch.object(live, "run", side_effect=RuntimeError("provider failed")) as execute:
                with self.assertRaisesRegex(RuntimeError, "provider failed"):
                    live.run_controlled([], {"DSRS_PROFILE_BASE_URL": "http://localhost/x/v1"}, Path(directory), "x", None, 3)
            self.assertEqual(execute.call_count, 1)

    def test_interference_retry_budget_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(live, "wait_idle"), patch.object(live, "run", side_effect=live.CompilerInterference("busy")) as execute:
                with self.assertRaises(live.CompilerInterference):
                    live.run_controlled([], {"DSRS_PROFILE_BASE_URL": "http://localhost/x/v1"}, Path(directory), "x", None, 1)
            self.assertEqual(execute.call_count, 2)
            attempts = json.loads((Path(directory) / "x.attempts.json").read_text())
            self.assertTrue(all(not a["accepted"] for a in attempts))

    def test_cpu_counts_windows_interpreter_children(self):
        class Process:
            def __init__(self, user, system, children=()):
                self.times = SimpleNamespace(user=user, system=system)
                self.members = list(children)
            def cpu_times(self): return self.times
            def children(self, recursive): return self.members
        child = Process(2, 3)
        parent = Process(0.1, 0.2, [child])
        self.assertEqual(live.cpu_ns(parent), 5_300_000_000)

    def test_adapter_totals_use_provider_calls_not_span_count(self):
        record = dict(total_duration=30, load_duration=1, prompt_eval_duration=8,
                      eval_duration=20, prompt_eval_count=3, eval_count=2)
        result = live.summarize({"framework": "dsrs", "instrumented": True,
            "samples": [{"pipeline_ns": 100, "stage_ns": [30, 30, 30],
                "provider_ns": [10, 15, 20], "exact_match": False,
                "supported_answer_match": True,
                "adapter_events": [("dsrs.adapter.format_system", 12),
                    ("dsrs.adapter.format_input", 3), ("dsrs.adapter.parse", 6)]}],
            "provider_records": [record] * 3, "process_interval": {}})
        self.assertEqual(result["adapter_format_ns_mean"], 5)
        self.assertEqual(result["adapter_parse_ns_mean"], 2)
        self.assertEqual(result["client_other_wall_ns_per_pipeline"], 55)
        self.assertEqual(result["exact_matches"], 0)
        self.assertEqual(result["supported_answer_matches"], 1)

    def test_proxy_translates_text_parts_and_preserves_model_timings(self):
        received = []
        class Ollama(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                payload = json.dumps({"message": {"content": '{"answer":"answer"}'}, "total_duration": 30,
                    "load_duration": 1, "prompt_eval_duration": 8, "eval_duration": 20,
                    "prompt_eval_count": 3, "eval_count": 2, "done_reason": "length"}).encode()
                self.send_response(200); self.send_header("Content-Length", str(len(payload)))
                self.end_headers(); self.wfile.write(payload)
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Ollama)
        proxy = ThreadingHTTPServer(("127.0.0.1", 0), live.Proxy)
        live.Proxy.ollama = f"http://127.0.0.1:{upstream.server_port}"
        live.Proxy.records = []
        for server in [upstream, proxy]:
            threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}
            response = live.request_json(f"http://127.0.0.1:{proxy.server_port}/test/v1/chat/completions",
                {"model": "local", "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
                 "temperature": 0, "max_tokens": 128,
                 "response_format": {"type": "json_schema", "json_schema": {"name": "Answer", "schema": schema, "strict": True}}})
            self.assertEqual(received[0]["messages"], [{"role": "user", "content": "hello"}])
            self.assertEqual(received[0]["options"]["num_predict"], 128)
            self.assertEqual(received[0]["format"], schema)
            self.assertEqual(response["usage"]["total_tokens"], 5)
            self.assertEqual(response["choices"][0]["finish_reason"], "length")
            self.assertEqual(live.Proxy.records[0]["done_reason"], "length")
            self.assertEqual(live.Proxy.records[0]["response_content"], '{"answer":"answer"}')
            self.assertTrue(live.Proxy.records[0]["response_json_valid"])
            self.assertEqual(live.Proxy.records[0]["eval_duration"], 20)
            self.assertTrue(live.Proxy.records[0]["ok"])
            import bench_dspy
            lm = bench_dspy.TimedLM("openai/local", api_base=f"http://127.0.0.1:{proxy.server_port}/sdk/v1",
                api_key="local-profile", cache=False, temperature=0, max_tokens=128, num_retries=0)
            actual = bench_dspy.StrictJSONAdapter()(lm, {}, bench_dspy.Answer, [], {"question": "x", "context": "y"})
            self.assertEqual(actual, [{"answer": "answer"}])
            self.assertEqual(len(received), 2)
            self.assertEqual(received[1]["format"]["properties"]["answer"]["type"], "string")
        finally:
            for server in [proxy, upstream]: server.shutdown(); server.server_close()


if __name__ == "__main__":
    unittest.main()
