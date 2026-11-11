"""Verify retained real-provider outputs, native results and validation rejection."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
load = lambda path: json.loads((root / path).read_text(encoding="utf-8"))
references = load("context-references.json")
original = json.loads((root.parents[3] / "fixture.json").read_text(encoding="utf-8"))["examples"]
assert references[:3] == original
demos = load("demonstrations.json")
assert not {d["question"] for d in demos} & {r["question"] for r in references}

rows = load("native-compact-results.json")["records"]
assert len(rows) == 24
for example in references:
    stages = [r for r in rows if r["id"] == example["id"]]
    assert [r["stage"] for r in stages] == [0, 1, 2]
    for stage, row in enumerate(stages):
        body, response = row["request"], row["response"]
        assert body["model"] == response["model"] == "Qwen3-0.6B-Tasks"
        assert body["temperature"] == 0 and body["max_tokens"] == 128
        assert response["choices"][0]["finish_reason"] == "stop"
        assert json.loads(response["choices"][0]["message"]["content"]) == row["parsed"]
        text = body["messages"][-1]["content"]
        assert "[[ ## question ## ]]\n" + example["question"] + "\n\n" in text
        context = text.split("[[ ## context ## ]]\n", 1)[1].split("\n\n[[ ## ", 1)[0].rsplit("\n\nReturn a JSON object", 1)[0]
        assert all(line in example["context"].splitlines() for line in context.splitlines())
        for message in body["messages"]:
            if message["role"] == "assistant":
                content = message["content"]
                if isinstance(content, list): content = "".join(p["text"] for p in content)
                assert "\n" not in content
                assert isinstance(json.loads(content), dict)
        if stage:
            field = "answer" if stage == 1 else "draft"
            assert "[[ ## " + field + " ## ]]\n" + stages[0]["parsed"]["answer"] + "\n\n" in text
        if stage == 2:
            assert "[[ ## critique ## ]]\n" + stages[1]["parsed"]["critique"] + "\n\n" in text

native = load("final-native-validation.json")
assert [(r["exact_matches"], r["samples"], r["native_exit"], r["verified_requests"]) for r in native["runs"]] == [(1, 3, 2, 9), (2, 3, 2, 9), (4, 4, 0, 12)]
assert native["invalid_cli_combinations_rejected"] == 3
guarded = load("guarded-live-final/summary.json")
assert guarded["native_exit"] == 1 and guarded["provider_requests"] == 3
attempts = load("guarded-live-final/requests.json")
assert [json.loads(r["response"]["choices"][0]["message"]["content"])["answer"] for r in attempts] == ["ambassador", "mayor", "mayor"]
assert "explicitly excludes" in attempts[1]["request"]["messages"][-1]["content"]
assert "contiguous phrase" in attempts[2]["request"]["messages"][-1]["content"]
assert all("Chief of Protocol" not in r["request"]["messages"][-1]["content"] for r in attempts[1:])
official = load("official-qwen-control-results.json")
assert official["expected_sha256"] == official["model_sha256"]
assert official["temporary_container_removed"] and official["temporary_model_removed"]
print("Passed: 24 real replies and stage inputs; native 1/3 -> 2/3, fresh 4/4; three rejected live retries; no reference changes.")
