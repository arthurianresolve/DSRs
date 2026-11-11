"""DSPy counterpart of bench-hotpotqa-reflection.rs, with offline and live modes."""
import argparse
import json
from pathlib import Path
import time
import os

import dspy
from dspy.utils.dummies import DummyLM
from dspy.adapters.json_adapter import _get_structured_outputs_response_format


class Answer(dspy.Signature):
    """Answer the question using the supplied context. Put a short answer without explanation in the answer output field. Do not copy input fields."""
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField()


class Reflect(dspy.Signature):
    """Critique the answer against the supplied context. Put one brief sentence in the critique output field. Do not copy input fields."""
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.InputField()
    critique: str = dspy.OutputField()


class Revise(dspy.Signature):
    """Revise the answer using the critique and supplied context. Put a short answer without explanation in the answer output field. Do not copy input fields."""
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    draft: str = dspy.InputField()
    critique: str = dspy.InputField()
    answer: str = dspy.OutputField()


CRITIQUE = "The answer agrees with the supporting context."
WARMUP = 20

PROVIDER_NS = []
ADAPTER_EVENTS = []

class TimedLM(dspy.LM):
    def forward(self, *args, **kwargs):
        start = time.perf_counter_ns()
        try:
            return super().forward(*args, **kwargs)
        finally:
            PROVIDER_NS.append(time.perf_counter_ns() - start)

class AdapterTiming:
    def format(self, *args, **kwargs):
        start = time.perf_counter_ns()
        try:
            return super().format(*args, **kwargs)
        finally:
            ADAPTER_EVENTS.append(("dsrs.adapter.format_input", time.perf_counter_ns() - start))
    def parse(self, *args, **kwargs):
        start = time.perf_counter_ns()
        try:
            return super().parse(*args, **kwargs)
        finally:
            ADAPTER_EVENTS.append(("dsrs.adapter.parse", time.perf_counter_ns() - start))

class TimedAdapter(AdapterTiming, dspy.ChatAdapter):
    pass

class StrictJSONAdapter(dspy.JSONAdapter):
    """Use the declared schema-capable endpoint without a second-request fallback."""
    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        lm_kwargs = dict(lm_kwargs, response_format=_get_structured_outputs_response_format(signature))
        return dspy.ChatAdapter.__call__(self, lm, lm_kwargs, signature, demos, inputs)

class TimedJSONAdapter(AdapterTiming, StrictJSONAdapter):
    pass

def barrier(label):
    if os.environ.get("DSRS_PROFILE_BARRIER"):
        print(label, flush=True)
        if input().strip() != "continue":
            raise RuntimeError("controller barrier closed")

def normalized(text):
    text = "".join(c for c in text.lower() if c.isalnum() or c.isspace())
    return " ".join(w for w in text.split() if w not in {"a", "an", "the"})


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
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--profile-runtime", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="One fixture, no warmup; protocol checking only")
    parser.add_argument("--check-protocol", action="store_true", help="All fixtures, no warmup; correctness diagnostics, not timing evidence")
    parser.add_argument("--json-output", action="store_true", help="Use the declared provider-native JSON Schema endpoint")
    args = parser.parse_args()
    if args.json_output and not args.live:
        parser.error("JSON output mode requires a live provider")
    if args.smoke and args.check_protocol:
        parser.error("choose smoke or a full protocol check")
    if args.smoke and not args.live:
        parser.error("smoke mode requires a live provider")
    warmup = 0 if args.smoke or args.check_protocol else 1 if args.live else WARMUP
    if args.live and args.wait_ms:
        parser.error("live profiling must not inject simulated waiting")
    if args.iterations <= 0 or args.wait_ms < 0:
        parser.error("iterations must be positive and wait_ms nonnegative")
    if dspy.__version__ != "3.2.1":
        raise RuntimeError("Install the pinned DSPy 3.2.1 benchmark environment")
    examples = json.loads(Path(__file__).with_name("fixture.json").read_text(encoding="utf-8"))["examples"]
    if args.smoke:
        examples = examples[:1]
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
    for _ in range(args.iterations + warmup):
        for e in examples:
            replies.extend([{"answer": e["answer"]}, {"critique": CRITIQUE}, {"answer": e["answer"]}])
    lm = (TimedLM("openai/" + os.environ.get("DSRS_PROFILE_MODEL", "llama3.2:3b"),
        api_base=os.environ["DSRS_PROFILE_BASE_URL"], api_key="local-profile", cache=False,
        temperature=0, max_tokens=128, num_retries=0) if args.live else DummyLM(replies))
    # DummyLM.forward consumes canned responses directly; it performs no cache lookup.
    lm.cache = False
    if args.json_output:
        adapter = (TimedJSONAdapter if args.profile_runtime else StrictJSONAdapter)()
    else:
        adapter = (TimedAdapter if args.profile_runtime else dspy.ChatAdapter)(use_json_adapter_fallback=False)
    dspy.configure(lm=lm, adapter=adapter, disable_history=True)
    answer, reflect, revise = dspy.Predict(Answer), dspy.Predict(Reflect), dspy.Predict(Revise)
    samples = []
    for iteration in range(args.iterations + warmup):
        if args.live and iteration == warmup:
            barrier("READY")
        for e in examples:
            provider_start, adapter_start = len(PROVIDER_NS), len(ADAPTER_EVENTS)
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
            assert args.live or first.answer == e["answer"]
            feedback = call(reflect, **inputs, answer=first.answer)
            assert args.live or feedback.critique == CRITIQUE
            final = call(revise, **inputs, draft=first.answer, critique=feedback.critique)
            assert args.live or final.answer == e["answer"]
            if iteration >= warmup:
                samples.append({"id": e["id"], "answer": final.answer,
                    "pipeline_ns": time.perf_counter_ns() - pipeline_start,
                    "stage_ns": stage_ns, "simulated_wait_ns": wait_ns,
                    "provider_ns": PROVIDER_NS[provider_start:], "adapter_events": ADAPTER_EVENTS[adapter_start:],
                    "expected_answer": e["answer"], "exact_match": normalized(final.answer) == normalized(e["answer"]),
                    "draft": first.answer, "critique": feedback.critique})
    print(json.dumps({"framework": "dspy", "kind": "protocol_check" if args.check_protocol else "pipeline_timing", "output_format": "json_schema" if args.json_output else "delimiter", "iterations": args.iterations,
        "warmup": warmup, "live": args.live, "smoke": args.smoke, "instrumented": args.profile_runtime, "simulated_wait_ms_per_stage": args.wait_ms,
        "cache": False, "history": False, "samples": samples}))
    if args.live:
        barrier("DONE")


if __name__ == "__main__":
    main()
