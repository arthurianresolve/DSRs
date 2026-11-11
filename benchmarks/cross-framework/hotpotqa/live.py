"""Profile real HotpotQA reflection calls through a local Ollama timing proxy."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import queue
import statistics
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import psutil


def request_json(url, body=None):
    request = Request(url, None if body is None else json.dumps(body).encode(),
                      {"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=240) as response:
            return json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode(errors='replace')[:2000]}") from error


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Proxy(BaseHTTPRequestHandler):
    records = []
    ollama = ""
    lock = threading.Lock()

    def log_message(self, *args):
        pass

    def do_POST(self):
        start = time.perf_counter_ns()
        record = {"run": self.path.split("/")[1], "started_ns": start}
        with self.lock:
            self.records.append(record)
        try:
            if not self.path.endswith("/v1/chat/completions"):
                raise ValueError("unexpected completion endpoint")
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body.get("stream") or body.get("tools") or body.get("response_format"):
                raise ValueError("this experiment uses non-streaming delimiter predictions without tools")
            messages = []
            for message in body["messages"]:
                content = message["content"]
                if isinstance(content, list):
                    if not all(block.get("type") == "text" for block in content):
                        raise ValueError("this fixture accepts text messages only")
                    content = "".join(block["text"] for block in content)
                if not isinstance(content, str):
                    raise ValueError("message content must be text")
                messages.append({"role": message["role"], "content": content})
            upstream = {"model": body["model"], "messages": messages, "stream": False,
                        "keep_alive": "30m", "options": {"temperature": body["temperature"],
                        "num_predict": body.get("max_tokens") or body.get("max_completion_tokens"),
                        "seed": 0, "num_thread": 4, "num_ctx": 2048}}
            record["request_sha256"] = hashlib.sha256(json.dumps(upstream, sort_keys=True).encode()).hexdigest()
            provider_start = time.perf_counter_ns()
            result = request_json(self.ollama + "/api/chat", upstream)
            record["provider_wall_ns"] = time.perf_counter_ns() - provider_start
            for key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration", "prompt_eval_count", "eval_count"):
                record[key] = result.get(key, 0)
            content = result["message"]["content"]
            payload = {"id": "profile-local", "object": "chat.completion", "created": int(time.time()),
                       "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                       "message": {"role": "assistant", "content": content}}],
                       "usage": {"prompt_tokens": result.get("prompt_eval_count", 0),
                       "completion_tokens": result.get("eval_count", 0),
                       "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)}}
            record["ok"] = True
            data = json.dumps(payload).encode()
            self.send_response(200)
        except Exception as error:
            record.update(ok=False, error=str(error))
            data = json.dumps({"error": {"message": str(error), "type": "profiling_error"}}).encode()
            self.send_response(500)
        record["proxy_wall_ns"] = time.perf_counter_ns() - start
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def cpu_ns(process):
    # Windows venv executables can be redirectors; include the real interpreter.
    total = 0
    for member in [process, *process.children(recursive=True)]:
        times = member.cpu_times()
        total += round((times.user + times.system) * 1e9)
    return total


def read_line(process, label):
    result = queue.Queue()
    threading.Thread(target=lambda: result.put(process.stdout.readline()), daemon=True).start()
    try:
        line = result.get(timeout=600)
    except queue.Empty:
        raise TimeoutError(f"child did not reach {label} within 10 minutes") from None
    if not line:
        raise RuntimeError(f"child exited before {label}; exit={process.poll()}")
    return line.strip()


def compiler_processes(wsl):
    active = [p.info["name"] for p in psutil.process_iter(["name"])
              if (p.info["name"] or "").lower() in {"cargo.exe", "rustc.exe", "cargo", "rustc"}]
    if wsl:
        result = subprocess.run(["wsl", "--distribution", wsl, "--exec", "sh", "-c",
            "pgrep -x cargo; pgrep -x rustc; true"], capture_output=True, text=True, check=True)
        if result.stdout.strip(): active.append("WSL compiler")
    return active


def wait_idle(wsl):
    deadline = time.monotonic() + 900
    while compiler_processes(wsl):
        if time.monotonic() >= deadline:
            raise RuntimeError("compiler activity did not stop within 15 minutes")
        time.sleep(2)


def run(command, env, output, name, wsl):
    with (output / (name + ".stderr.txt")).open("w", encoding="utf-8") as stderr:
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=stderr, env=env, text=True, encoding="utf-8", bufsize=1)
        monitor_stop = threading.Event()
        observer = None
        try:
            if read_line(child, "warmup barrier") != "READY":
                raise RuntimeError("unexpected child warmup output")
            process = psutil.Process(child.pid)
            wait_idle(wsl)
            interference = []
            def monitor():
                while not monitor_stop.wait(1):
                    interference.extend(compiler_processes(wsl))
            observer = threading.Thread(target=monitor, daemon=True)
            observer.start()
            before_cpu = cpu_ns(process)
            before_wall = time.perf_counter_ns()
            with Proxy.lock:
                warmup_count = len([r for r in Proxy.records if r["run"] == name])
            child.stdin.write("continue\n"); child.stdin.flush()
            raw = read_line(child, "result")
            data = json.loads(raw)
            if read_line(child, "end barrier") != "DONE":
                raise RuntimeError("unexpected child completion output")
            wall = time.perf_counter_ns() - before_wall
            cpu = cpu_ns(process) - before_cpu
            monitor_stop.set(); observer.join()
            interference.extend(compiler_processes(wsl))
            child.stdin.write("continue\n"); child.stdin.flush()
            child.wait(timeout=30)
            data["process_interval"] = {"wall_ns": wall, "cpu_ns": cpu,
                "non_cpu_wall_lower_bound_ns": max(0, wall - cpu)}
            with Proxy.lock:
                data["provider_records"] = [r.copy() for r in Proxy.records if r["run"] == name][warmup_count:]
            data["exit_code"] = child.returncode
            data["compiler_interference"] = interference
            (output / (name + ".json")).write_text(json.dumps(data, indent=2), encoding="utf-8")
            if child.returncode:
                raise RuntimeError(f"child exit {child.returncode}")
            if interference:
                raise RuntimeError("compiler activity detected during the measured interval")
            assert len(data["provider_records"]) == len(data["samples"]) * 3
            assert all(r["ok"] for r in data["provider_records"])
            assert all(len(s["provider_ns"]) == 3 for s in data["samples"])
            if data["instrumented"]:
                assert all(sum(key == "dsrs.adapter.parse" for key, _ in s["adapter_events"]) == 3
                           for s in data["samples"]), "each stage must contribute exactly one parse span"
            return data
        except Exception as error:
            (output / (name + ".failure.json")).write_text(json.dumps({"error": str(error), "exit_code": child.poll()}), encoding="utf-8")
            raise
        finally:
            monitor_stop.set()
            if observer is not None: observer.join(timeout=5)
            if child.poll() is None:
                child.kill(); child.wait()


def summarize(data):
    samples, records = data["samples"], data["provider_records"]
    pipeline = sum(s["pipeline_ns"] for s in samples)
    sdk = sum(sum(s["provider_ns"]) for s in samples)
    adapter = [event for s in samples for event in s["adapter_events"]]
    return {"framework": data["framework"], "instrumented": data["instrumented"],
            "pipelines": len(samples), "provider_requests": len(records),
            "pipeline_ns_mean": pipeline / len(samples),
            "stage_ns_mean": [statistics.mean(s["stage_ns"][i] for s in samples) for i in range(3)],
            "provider_sdk_ns_mean": sdk / len(records),
            "client_other_wall_ns_per_pipeline": (pipeline - sdk) / len(samples),
            "adapter_format_ns_mean": sum(n for key, n in adapter if "format" in key) / len(records) if adapter else None,
            "adapter_parse_ns_mean": sum(n for key, n in adapter if "parse" in key) / len(records) if adapter else None,
            "server_phase_ns_mean": {k: statistics.mean(r[k] for r in records) for k in
                ["total_duration", "load_duration", "prompt_eval_duration", "eval_duration"]},
            "tokens_mean": {k: statistics.mean(r[k] for r in records) for k in ["prompt_eval_count", "eval_count"]},
            "exact_matches": sum(s["exact_match"] for s in samples),
            "process_interval": data["process_interval"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ollama", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--port", type=int, default=18084)
    parser.add_argument("--wsl-distro", help="Also gate measurements on compiler activity in this WSL distribution")
    args = parser.parse_args()
    if args.samples < 2 or args.samples % 2:
        parser.error("samples must be even and at least two")
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    if args.out.resolve().is_relative_to(repo):
        parser.error("evidence must be outside the checkout")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        parser.error("commit the harness before measuring")
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "platform": platform.platform(),
                "processor": platform.processor(), "physical_cores": psutil.cpu_count(logical=False),
                "logical_cpus": psutil.cpu_count(), "cpu_frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                "wsl_distro": args.wsl_distro,
                "rustc": subprocess.check_output(["rustc", "-Vv"], text=True), "python": sys.version,
                "binary_sha256": sha(args.rust), "model": args.model,
                "ollama_version": request_json(args.ollama + "/api/version"),
                "models": request_json(args.ollama + "/api/tags"),
                "files": {str(p.relative_to(repo)): sha(p) for p in [here / "live.py", here / "bench_dspy.py", here / "fixture.json", repo / "crates/dspy-rs/examples/bench-hotpotqa-reflection.rs", repo / "Cargo.lock"]},
                "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())),
                "policy": {"cache": False, "history": False, "retries": 0, "temperature": 0, "max_tokens": 64,
                           "seed": 0, "num_thread": 4, "num_ctx": 2048, "keep_alive": "30m", "warmup_passes": 1, "timed_passes": 1,
                           "samples_per_framework": args.samples, "quality": "normalized final-answer exact match, reported without excluding failures"}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    Proxy.ollama = args.ollama
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Proxy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    results = []
    try:
        order = [(f, False) for f in ["dsrs", "dspy", "dspy", "dsrs"] * (args.samples // 2)]
        order += [("dsrs", True), ("dspy", True)]
        for index, (framework, instrumented) in enumerate(order):
            wait_idle(args.wsl_distro)
            name = f"{index:02d}-{framework}-{'profile' if instrumented else 'timing'}"
            command = ([str(args.rust.resolve())] if framework == "dsrs" else [sys.executable, str(here / "bench_dspy.py")]) + ["1", "0", "--live"]
            if instrumented: command.append("--profile-runtime")
            env = dict(os.environ, DSRS_PROFILE_BASE_URL=f"http://127.0.0.1:{args.port}/{name}/v1",
                       DSRS_PROFILE_MODEL=args.model, DSRS_PROFILE_BARRIER="1")
            data = run(command, env, args.out, name, args.wsl_distro)
            results.append({"run": name, **summarize(data)})
            (args.out / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"validated {name}: {len(data['samples'])} pipelines", flush=True)
    finally:
        (args.out / "all-provider-records.json").write_text(json.dumps(Proxy.records, indent=2), encoding="utf-8")
        server.shutdown(); server.server_close()


if __name__ == "__main__":
    main()
