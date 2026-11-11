"""Audit the follow-up native DSRs run from its retained requests and replies."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
hotpotqa = root.parents[5]
repo = hotpotqa.parents[2]
native = root / "native-live"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalized(text):
    text = "".join(c for c in text.lower() if c.isalnum() or c.isspace())
    return " ".join(word for word in text.split() if word not in {"a", "an", "the"})


territories = read(hotpotqa / "data/territories.json")
names = territories["main"]["en"]["localeDisplayNames"]["territories"]
names = [value for key, value in names.items()
         if re.fullmatch(r"[A-Z]{2}(?:-alt-.+)?", key)
         and key != "ZZ" and len(value) > 3]
qualifier = re.compile(r"\s+(?:of|for)\s+(?:the\s+)?(?:" +
                       "|".join(map(re.escape, names)) + r")\s*\.?$", re.I)


def canonical(question, answer):
    match = qualifier.search(answer) if re.search(r"\b(position|office)\b", question, re.I) else None
    return answer[:match.start()].rstrip() if match and answer[:match.start()].strip() else answer


build = read(root / "build-record.json")
for path, expected in build["source_blob_sha256"].items():
    current = hashlib.sha256((repo / path).read_bytes()).hexdigest()
    assert current in (expected, build["source_sha256"][path]), path
    try:
        committed = subprocess.check_output(
            ["git", "show", f"HEAD:{path}"], cwd=repo, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        committed = b""
    if hashlib.sha256(committed).hexdigest() != expected:
        committed = subprocess.check_output(["git", "show", f":{path}"], cwd=repo)
    assert hashlib.sha256(committed).hexdigest() == expected, path
assert all((root / name).read_text(encoding="utf-8").strip() == "0" for name in
           ("build.exit.txt", "rust-tests.exit.txt", "python-tests.exit.txt"))
assert "11 passed; 0 failed" in (root / "rust-tests.stdout.txt").read_text(encoding="utf-8")
assert "Ran 11 tests" in (root / "python-tests.stderr.txt").read_text(encoding="utf-8")
regression = read(root / "native-regression-final2.json")
assert regression["binary_sha256"] == build["binary_sha256"]
assert [case["exit"] for case in regression["cases"]] == [0, 1, 2]
assert [len(case["requests"]) for case in regression["cases"]] == [3, 5, 3]

summary = read(native / "summary.json")
assert summary["binary_sha256"] == build["binary_sha256"]
assert summary["not_performance_evidence"]
expected = [("baseline", 2, 1, 3, 9), ("original", 2, 2, 3, 9),
            ("fresh", 0, 4, 4, 12), ("clarified", 0, 1, 1, 3),
            ("prospective", 2, 7, 8, 24),
            ("repeat-original-and-clarified", 2, 3, 4, 12),
            ("excluded-only", 1, None, 1, 5)]
assert [(run["name"], run["native_exit"], run.get("exact_matches"),
         run["samples"], run["provider_requests"]) for run in summary["runs"]] == expected
assert read(native / "state.json") == {"status": "completed", "requests": 74}
records = read(native / "requests.json")
receipts = read(native / "transport-receipts.json")
assert len(records) == len(receipts) == build["native_provider_requests"] == 74
assert [record["round"] for record in records] == list(range(1, 75))
for record, receipt in zip(records, receipts, strict=True):
    path = native / f"round-{record['round']}.response.json"
    raw = path.read_bytes()
    assert receipt["round"] == record["round"] and receipt["run"] == record["run"]
    assert receipt["bytes"] == len(raw)
    assert hashlib.sha256(raw).hexdigest() == receipt["sha256"], path
    round_data = json.loads(raw)
    assert round_data["round"] == record["round"]
    request = dict(round_data["request"])
    assert request.pop("seed") == 0 and request == record["request"]
    response = round_data["response"]
    assert response == record["response"]
    assert response["model"] == request["model"] == build["provider_model"]
    assert response["choices"][0]["finish_reason"] == "stop"
    json.loads(response["choices"][0]["message"]["content"])

references = read(root.parent / "relation-recipe/references.json")
original = read(hotpotqa / "fixture.json")["examples"]
assert references[:3] == original
assert read(native / "baseline/fixture.json")["examples"] == original
assert read(native / "original/fixture.json")["examples"] == original
assert read(native / "fresh/fixture.json")["examples"] == references[3:7]
assert read(native / "clarified/fixture.json")["examples"] == [references[7]]
assert read(native / "prospective/fixture.json")["examples"] == read(hotpotqa / "prospective-fixture.json")["examples"]
assert read(native / "repeat-original-and-clarified/fixture.json")["examples"] == original + [references[7]]

for run in summary["runs"]:
    name = run["name"]
    folder = native / name
    fixture = read(folder / "fixture.json")["examples"]
    assert read(folder / "summary.json") == run
    calls = [record for record in records if record["run"] == name]
    assert len(calls) == run["provider_requests"]
    if name == "excluded-only":
        assert len(fixture) == 1 and "other than governor" in fixture[0]["question"]
        assert not (folder / "stdout.json").read_text(encoding="utf-8").strip()
        assert "output validation failed" in (folder / "stderr.txt").read_text(encoding="utf-8")
        assert [list(call["request"]["response_format"]["json_schema"]["schema"]["properties"])
                for call in calls] == [["answer"], ["critique"], ["answer"], ["answer"], ["answer"]]
        continue
    output = read(folder / "stdout.json")
    samples = output["samples"]
    assert len(samples) == len(fixture) == run["samples"]
    assert len(calls) == 3 * len(samples)
    assert output["recipe"] == ("retrieved_context_and_demonstrations" if run["recipe"] else "baseline_reflection")
    assert [sample["answer"] for sample in samples] == run["answers"]
    assert sum(sample["exact_match"] for sample in samples) == run["exact_matches"]
    assert output["quality_status"] == run["quality_status"]
    assert run["native_exit"] == (0 if output["quality_status"] == "passed" else 2)
    for i, (example, sample) in enumerate(zip(fixture, samples, strict=True)):
        stages = calls[3 * i:3 * i + 3]
        fields = [list(stage["request"]["response_format"]["json_schema"]["schema"]["properties"])
                  for stage in stages]
        assert fields == [["answer"], ["critique"], ["answer"]]
        values = [json.loads(stage["response"]["choices"][0]["message"]["content"])
                  for stage in stages]
        for stage in stages:
            prompt = stage["request"]["messages"][-1]["content"]
            assert "[[ ## question ## ]]\n" + example["question"] + "\n\n" in prompt
            assert "[[ ## context ## ]]\n" + sample["context"] + "\n\n" in prompt
        assert "[[ ## answer ## ]]\n" + values[0]["answer"] + "\n\n" in stages[1]["request"]["messages"][-1]["content"]
        revised_input = stages[2]["request"]["messages"][-1]["content"]
        assert "[[ ## draft ## ]]\n" + values[0]["answer"] + "\n\n" in revised_input
        assert "[[ ## critique ## ]]\n" + values[1]["critique"] + "\n\n" in revised_input
        assert sample["id"] == example["id"]
        assert sample["expected_answer"] == example["answer"]
        assert sample["draft"] == values[0]["answer"]
        assert sample["critique"] == values[1]["critique"]
        assert sample["revised_answer"] == values[2]["answer"]
        answer = canonical(example["question"], values[2]["answer"]) if run["recipe"] else values[2]["answer"]
        assert sample["answer"] == answer
        assert sample["exact_match"] == (normalized(answer) == normalized(example["answer"]))

office = read(native / "original/stdout.json")["samples"][1]
assert not office["exact_match"] and normalized(office["answer"]) == "ambassador"
assert "ambassador to Ghana" in office["context"] and "Chief of Protocol" in office["context"]
series = read(native / "original/stdout.json")["samples"][2]
assert series["exact_match"] and "enslaved the Hork-Bajir" in series["context"]
assert "continuity within the series" not in series["context"]
prospective = read(native / "prospective/stdout.json")["samples"][0]
assert not prospective["exact_match"] and normalized(prospective["answer"]) == "ambassador"
assert "ambassador to Velora" in prospective["context"] and "Consul General" in prospective["context"]
assert read(native / "clarified/stdout.json")["samples"][0]["exact_match"]

probe = root / "diagnostics/semantic-prompt"
probe_summary = read(probe / "summary.json")
assert probe_summary["binary_sha256"] != build["binary_sha256"]
assert [(run["name"], run["native_exit"], run["exact_matches"], run["provider_requests"])
        for run in probe_summary["runs"]] == [("original", 2, 1, 9)]
probe_output = read(probe / "original/stdout.json")
assert [sample["answer"] for sample in probe_output["samples"]] == ["no", "ambassador", "Animorphs"]
probe_calls = read(probe / "requests.json")
probe_receipts = read(probe / "transport-receipts.json")
assert len(probe_calls) == len(probe_receipts) == 9
assert "prefer a role explicitly described as served" in probe_calls[0]["request"]["messages"][0]["content"][0]["text"]
for call, receipt in zip(probe_calls, probe_receipts, strict=True):
    raw = (probe / f"round-{call['round']}.response.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt["sha256"]
    payload = json.loads(raw)
    assert payload["response"] == call["response"]

manifest_root = hotpotqa / "evidence/2026-11-11-live"
manifest_path = manifest_root / "artifact-sha256.json"
manifest = read(manifest_path)
actual = {path.relative_to(manifest_root).as_posix() for path in manifest_root.rglob("*")
          if path.is_file() and path != manifest_path}
assert set(manifest) == actual
for path, expected in manifest.items():
    assert hashlib.sha256((manifest_root / path).read_bytes()).hexdigest() == expected, path

print("Passed: 74 checked provider replies; original 2/3 with an ambiguous office answer, fresh 4/4, clarified 1/1, prospective 7/8 with an ambiguous office answer, repeat 3/4, and bounded final rejection.")
