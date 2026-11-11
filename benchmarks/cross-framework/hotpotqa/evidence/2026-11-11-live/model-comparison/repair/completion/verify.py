"""Verify the retained native/provider evidence without a model or native binary."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
hotpotqa = root.parents[4]
repo = hotpotqa.parents[2]


def load(path):
    return json.loads((root / path).read_text(encoding="utf-8"))


def normalized(text):
    text = "".join(c for c in text.lower() if c.isalnum() or c.isspace())
    return " ".join(w for w in text.split() if w not in {"a", "an", "the"})


territories = json.loads((hotpotqa / "data/territories.json").read_text(encoding="utf-8"))
names = territories["main"]["en"]["localeDisplayNames"]["territories"]
names = [v for k, v in names.items() if re.fullmatch(r"[A-Z]{2}(?:-alt-.+)?", k)
         and k != "ZZ" and len(v) > 3]
qualifier = re.compile(r"\s+(?:of|for)\s+(?:the\s+)?(?:" + "|".join(map(re.escape, names)) + r")\s*\.?$", re.I)


def canonical(question, answer):
    match = qualifier.search(answer) if re.search(r"\b(position|office)\b", question, re.I) else None
    return answer[:match.start()].rstrip() if match and answer[:match.start()].strip() else answer


build = load("build-record.json")
for path, expected in build["working_source_sha256"].items():
    data = (repo / path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        data = subprocess.check_output(
            ["git", "show", f"{build['source_commit']}:{path}"], cwd=repo
        )
    assert hashlib.sha256(data).hexdigest() == expected, path
source = json.loads((hotpotqa / "data/source.json").read_text(encoding="utf-8"))
assert hashlib.sha256((hotpotqa / "data/territories.json").read_bytes()).hexdigest() == source["sha256"]
assert hashlib.sha256((hotpotqa / "data/UNICODE-LICENSE.txt").read_bytes()).hexdigest() == source["license_sha256"]

summary = load("relation-recipe/native-live-fixed/summary.json")
assert summary["binary_sha256"] == build["binary_sha256"]
expected = [("baseline", 2, 1, 3, 9), ("original", 0, 3, 3, 9),
            ("fresh", 0, 4, 4, 12), ("clarified", 0, 1, 1, 3),
            ("prospective", 0, 8, 8, 24),
            ("repeat-original-and-clarified", 0, 4, 4, 12),
            ("excluded-only", 1, None, 1, 5)]
assert [(r["name"], r["native_exit"], r.get("exact_matches"), r["samples"], r["provider_requests"])
        for r in summary["runs"]] == expected
requests = load("relation-recipe/native-live-fixed/requests.json")
assert len(requests) == 74
rejected = load("relation-recipe/native-live/requests.json")
assert [r["request"] for r in requests[:13]] == [r["request"] for r in rejected[:13]]
assert [r["round"] for r in requests] == list(range(1, 75))
for record in requests:
    round_file = load(f"relation-recipe/native-live-fixed/round-{record['round']}.response.json")
    assert round_file["round"] == record["round"]
    sent = dict(round_file["request"])
    assert sent.pop("seed") == 0 and sent == record["request"]
    assert round_file["response"] == record["response"]
    response = record["response"]
    assert response["model"] == record["request"]["model"] == "Qwen3-0.6B-Tasks"
    assert response["choices"][0]["finish_reason"] == "stop"
    json.loads(response["choices"][0]["message"]["content"])

for run in summary["runs"]:
    folder = root / "relation-recipe/native-live-fixed" / run["name"]
    assert load(f"relation-recipe/native-live-fixed/{run['name']}/summary.json") == run
    assert len([r for r in requests if r["run"] == run["name"]]) == run["provider_requests"]
    if run["name"] == "excluded-only":
        assert not (folder / "stdout.json").read_text(encoding="utf-8").strip()
        assert "output validation failed" in (folder / "stderr.txt").read_text(encoding="utf-8")
        continue
    report = json.loads((folder / "stdout.json").read_text(encoding="utf-8"))
    assert len(report["samples"]) == run["samples"]
    assert sum(sample["exact_match"] for sample in report["samples"]) == run["exact_matches"]
    assert report["quality_status"] == run["quality_status"]
    assert [sample["answer"] for sample in report["samples"]] == run["answers"]
    assert run["native_exit"] == (0 if report["quality_status"] == "passed" else 2)

for run in summary["runs"][:-1]:
    folder = root / "relation-recipe/native-live-fixed" / run["name"]
    fixture = json.loads((folder / "fixture.json").read_text(encoding="utf-8"))["examples"]
    result = json.loads((folder / "stdout.json").read_text(encoding="utf-8"))
    records = [r for r in requests if r["run"] == run["name"]]
    assert len(records) == 3 * len(fixture)
    for i, (example, sample) in enumerate(zip(fixture, result["samples"], strict=True)):
        stages = records[3*i:3*i+3]
        values = [json.loads(r["response"]["choices"][0]["message"]["content"]) for r in stages]
        for record in stages:
            text = record["request"]["messages"][-1]["content"]
            assert "[[ ## question ## ]]\n" + example["question"] + "\n\n" in text
            assert "[[ ## context ## ]]\n" + sample["context"] + "\n\n" in text
        assert "[[ ## answer ## ]]\n" + values[0]["answer"] + "\n\n" in stages[1]["request"]["messages"][-1]["content"]
        revised_input = stages[2]["request"]["messages"][-1]["content"]
        assert "[[ ## draft ## ]]\n" + values[0]["answer"] + "\n\n" in revised_input
        assert "[[ ## critique ## ]]\n" + values[1]["critique"] + "\n\n" in revised_input
        assert sample["revised_answer"] == values[2]["answer"]
        answer = canonical(example["question"], values[2]["answer"]) if run["recipe"] else values[2]["answer"]
        assert sample["answer"] == answer
        assert sample["exact_match"] == (normalized(answer) == normalized(example["answer"]))

original = json.loads((hotpotqa / "fixture.json").read_text(encoding="utf-8"))["examples"]
assert load("relation-recipe/native-live-fixed/original/fixture.json")["examples"] == original
original_result = load("relation-recipe/native-live-fixed/original/stdout.json")
assert 'companion book to the "Animorphs" series' in original_result["samples"][2]["context"]
assert original_result["samples"][1]["draft"] == "mayor"
assert normalized(original_result["samples"][1]["answer"]) == "chief of protocol"
negative = [r for r in requests if r["run"] == "excluded-only"]
assert [list(r["request"]["response_format"]["json_schema"]["schema"]["properties"]) for r in negative] == [["answer"], ["critique"], ["answer"], ["answer"], ["answer"]]
regression = load("relation-recipe/native-regression.json")
assert regression["binary_sha256"] == build["binary_sha256"]
assert [r["exit"] for r in regression["cases"]] == [0, 1, 2]
assert regression["cases"][0]["requests"] == regression["cases"][2]["requests"]
assert [len(r["requests"]) for r in regression["cases"]] == [3, 5, 3]
assert load("cache-repeatability/results.json")["model_sha256"] == "361cc68159042c36ebff7715dc5a2e4612153e88f3e9c9c234820849d6dc9e1d"
cleanup = load("relation-recipe/cleanup-result.json")
assert cleanup["status"] == "completed" and len(cleanup["directories_removed"]) == 8
assert cleanup["verified_files"] == 98 and all(cleanup["existing_services_running"].values())
manifest_path = hotpotqa / "evidence/2026-11-11-live/artifact-sha256.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest_root = manifest_path.parent
actual = {path.relative_to(manifest_root).as_posix() for path in manifest_root.rglob("*")
          if path.is_file() and path != manifest_path}
assert set(manifest) == actual
for path, expected in manifest.items():
    assert hashlib.sha256((manifest_root / path).read_bytes()).hexdigest() == expected, path
print("Passed: 74 real native requests; original 3/3, earlier synthetic 4/4, clarified 1/1, prospective 8/8, repeat 4/4; bounded final rejection; references isolated.")
