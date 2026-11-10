"""Run paired offline framework workloads; retain raw JSON and environment identity."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust", type=Path, required=True, help="Previously built release executable")
    parser.add_argument("--out", type=Path, required=True, help="New evidence directory, outside the checkout")
    parser.add_argument("--samples", type=int, default=4, help="Even number of runs per framework per phase")
    args = parser.parse_args()
    if args.samples < 2 or args.samples % 2:
        parser.error("samples must be positive and even (ABBA pairs)")
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    rust = args.rust.resolve(strict=True)
    if args.out.resolve().is_relative_to(repo):
        parser.error("keep evidence outside the source checkout")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        parser.error("commit the harness before measuring; the checkout must be clean")
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {"revision": revision, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(), "processor": platform.processor(), "python": sys.version,
        "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
        "cargo": subprocess.check_output(["cargo", "--version"], text=True).strip(),
        "rust_executable_sha256": sha256(rust),
        "files": {p.name: sha256(p) for p in [here / "fixture.json", here / "bench_dspy.py", here / "run.py", repo / "Cargo.lock"]},
        "python_packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())),
        "protocol": {"order": "ABBA repeated", "samples_per_framework": args.samples,
            "warmup": 20, "cpu_iterations": 200, "wait_iterations": 20, "simulated_wait_ms": 5}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    commands = {"dsrs": [str(rust)], "dspy": [sys.executable, str(here / "bench_dspy.py")]}
    records = []
    for phase, iterations, wait in [("cpu", 200, 0), ("simulated-wait", 20, 5), ("adapter-profile", 200, 0)]:
        order = ["dsrs", "dspy", "dspy", "dsrs"] * (args.samples // 2)
        if phase == "adapter-profile":
            order = ["dsrs", "dspy"]
        for index, framework in enumerate(order):
            command = commands[framework] + [str(iterations), str(wait)]
            if phase == "adapter-profile":
                command.append("--profile-adapter")
            result = subprocess.run(command, text=True, encoding="utf-8", capture_output=True, check=False)
            name = f"{phase}-{index:02d}-{framework}"
            (args.out / f"{name}.stderr.txt").write_text(result.stderr, encoding="utf-8")
            (args.out / f"{name}.json").write_text(result.stdout, encoding="utf-8")
            result.check_returncode()
            data = json.loads(result.stdout)
            if phase != "adapter-profile":
                expected = json.loads((here / "fixture.json").read_text(encoding="utf-8"))["examples"]
                assert len(data["samples"]) == len(expected) * iterations
                for i, sample in enumerate(data["samples"]):
                    example = expected[i % len(expected)]
                    assert (sample["id"], sample["answer"]) == (example["id"], example["answer"])
                    assert len(sample["stage_ns"]) == 3
                    assert 0 <= sample["simulated_wait_ns"] <= sum(sample["stage_ns"]) <= sample["pipeline_ns"]
                means = {key: statistics.mean(s[key] for s in data["samples"]) for key in ["pipeline_ns", "simulated_wait_ns"]}
                means["stage_ns"] = [statistics.mean(s["stage_ns"][i] for s in data["samples"]) for i in range(3)]
                records.append({"phase": phase, "framework": framework, **means})
            print(f"validated {name}", flush=True)
    summary = []
    for phase in ["cpu", "simulated-wait"]:
        for framework in commands:
            runs = [r for r in records if (r["phase"], r["framework"]) == (phase, framework)]
            summary.append({"phase": phase, "framework": framework,
                "pipeline_ns_median_of_run_means": statistics.median(r["pipeline_ns"] for r in runs),
                "pipeline_ns_run_means": [r["pipeline_ns"] for r in runs],
                "stage_ns_median_of_run_means": [statistics.median(r["stage_ns"][i] for r in runs) for i in range(3)],
                "simulated_wait_ns_median_of_run_means": statistics.median(r["simulated_wait_ns"] for r in runs)})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
