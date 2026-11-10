"""Offline DSPy counterpart of bench-hotpotqa-reflection.rs; no provider calls."""
import argparse
import json
from pathlib import Path
import time

import dspy
from dspy.utils.dummies import DummyLM


class Answer(dspy.Signature):
    """Answer the question using the supplied context."""
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField()


class Reflect(dspy.Signature):
    """Critique the answer against the supplied context."""
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.InputField()
    critique: str = dspy.OutputField()


class Revise(dspy.Signature):
    """Revise the answer using the critique and supplied context."""
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    draft: str = dspy.InputField()
    critique: str = dspy.InputField()
    answer: str = dspy.OutputField()


CRITIQUE = "The answer agrees with the supporting context."
WARMUP = 20


def adapter_profile(signature, inputs, field, value, iterations):
    adapter = dspy.ChatAdapter()
    raw = f"[[ ## {field} ## ]]\n{value}\n\n[[ ## completed ## ]]\n"
    format_ns = parse_ns = 0
    for i in range(iterations + WARMUP):
        start = time.perf_counter_ns()
        adapter.format(signature, [], inputs)
        formatted = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        parsed = adapter.parse(signature, raw)
        elapsed = time.perf_counter_ns() - start
        assert parsed[field] == value
        if i >= WARMUP:
            format_ns += formatted
            parse_ns += elapsed
    return {"format_ns_per_call": format_ns / iterations,
            "parse_ns_per_call": parse_ns / iterations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("iterations", type=int, nargs="?", default=200)
    parser.add_argument("wait_ms", type=int, nargs="?", default=0)
    parser.add_argument("--profile-adapter", action="store_true")
    args = parser.parse_args()
    if args.iterations <= 0 or args.wait_ms < 0:
        parser.error("iterations must be positive and wait_ms nonnegative")
    if dspy.__version__ != "3.2.1":
        raise RuntimeError("Install the pinned DSPy 3.2.1 benchmark environment")
    examples = json.loads(Path(__file__).with_name("fixture.json").read_text(encoding="utf-8"))["examples"]
    if args.profile_adapter:
        if args.wait_ms:
            parser.error("adapter replay has no simulated waiting")
        profiles = []
        for e in examples:
            inputs = {"question": e["question"], "context": e["context"]}
            profiles.append({"id": e["id"],
                "answer": adapter_profile(Answer, inputs, "answer", e["answer"], args.iterations),
                "reflect": adapter_profile(Reflect, {**inputs, "answer": e["answer"]}, "critique", CRITIQUE, args.iterations),
                "revise": adapter_profile(Revise, {**inputs, "draft": e["answer"], "critique": CRITIQUE}, "answer", e["answer"], args.iterations)})
        print(json.dumps({"framework": "dspy", "kind": "isolated_adapter_replay",
            "iterations": args.iterations, "warmup": WARMUP, "profiles": profiles}))
        return

    replies = []
    for _ in range(args.iterations + WARMUP):
        for e in examples:
            replies.extend([{"answer": e["answer"]}, {"critique": CRITIQUE}, {"answer": e["answer"]}])
    lm = DummyLM(replies)
    # DummyLM.forward consumes canned responses directly; it performs no cache lookup.
    lm.cache = False
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter(), disable_history=True)
    answer, reflect, revise = dspy.Predict(Answer), dspy.Predict(Reflect), dspy.Predict(Revise)
    samples = []
    for iteration in range(args.iterations + WARMUP):
        for e in examples:
            pipeline_start = time.perf_counter_ns()
            stage_ns = []
            wait_ns = 0

            def call(predict, **inputs):
                nonlocal wait_ns
                start = time.perf_counter_ns()
                if args.wait_ms:
                    time.sleep(args.wait_ms / 1000)
                    wait_ns += time.perf_counter_ns() - start
                result = predict(**inputs)
                stage_ns.append(time.perf_counter_ns() - start)
                return result

            inputs = {"question": e["question"], "context": e["context"]}
            first = call(answer, **inputs)
            assert first.answer == e["answer"]
            feedback = call(reflect, **inputs, answer=first.answer)
            assert feedback.critique == CRITIQUE
            final = call(revise, **inputs, draft=first.answer, critique=feedback.critique)
            assert final.answer == e["answer"]
            if iteration >= WARMUP:
                samples.append({"id": e["id"], "answer": final.answer,
                    "pipeline_ns": time.perf_counter_ns() - pipeline_start,
                    "stage_ns": stage_ns, "simulated_wait_ns": wait_ns})
    print(json.dumps({"framework": "dspy", "kind": "pipeline_timing", "iterations": args.iterations,
        "warmup": WARMUP, "simulated_wait_ms_per_stage": args.wait_ms,
        "cache": False, "history": False, "samples": samples}))


if __name__ == "__main__":
    main()
