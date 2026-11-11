import json
import threading
import unittest
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import live


class ProfilingTests(unittest.TestCase):
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
                "adapter_events": [("dsrs.adapter.format_system", 12),
                    ("dsrs.adapter.format_input", 3), ("dsrs.adapter.parse", 6)]}],
            "provider_records": [record] * 3, "process_interval": {}})
        self.assertEqual(result["adapter_format_ns_mean"], 5)
        self.assertEqual(result["adapter_parse_ns_mean"], 2)
        self.assertEqual(result["client_other_wall_ns_per_pipeline"], 55)
        self.assertEqual(result["exact_matches"], 0)

    def test_proxy_translates_text_parts_and_preserves_model_timings(self):
        received = []
        class Ollama(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                payload = json.dumps({"message": {"content": "answer"}, "total_duration": 30,
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
            response = live.request_json(f"http://127.0.0.1:{proxy.server_port}/test/v1/chat/completions",
                {"model": "local", "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
                 "temperature": 0, "max_tokens": 64})
            self.assertEqual(received[0]["messages"], [{"role": "user", "content": "hello"}])
            self.assertEqual(received[0]["options"]["num_predict"], 64)
            self.assertEqual(response["usage"]["total_tokens"], 5)
            self.assertEqual(response["choices"][0]["finish_reason"], "length")
            self.assertEqual(live.Proxy.records[0]["done_reason"], "length")
            self.assertEqual(live.Proxy.records[0]["eval_duration"], 20)
            self.assertTrue(live.Proxy.records[0]["ok"])
        finally:
            for server in [proxy, upstream]: server.shutdown(); server.server_close()


if __name__ == "__main__":
    unittest.main()
