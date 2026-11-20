import ast
import builtins
import json


class Submitted(BaseException):
    pass


class Output:
    def __init__(self, limit):
        self.remaining = limit
        self.parts = []
        self.truncated = False

    def write(self, text):
        if text and self.remaining:
            self.parts.append(text[: self.remaining])
        self.truncated |= len(text) > self.remaining
        self.remaining = max(0, self.remaining - len(text))

    def flush(self):
        pass

    def text(self):
        suffix = "\n[output truncated]" if self.truncated else ""
        return "".join(self.parts) + suffix


class Turn:
    def print(self, *values, file=None, **kwargs):
        builtins.print(*values, file=self.output if file is None else file, **kwargs)

    def submit(self, **fields):
        self.submission = json.dumps(fields, allow_nan=False, separators=(",", ":"))
        raise Submitted()


def input_preview(inputs):
    lines = ["Session inputs:"]
    for name, value in inputs.items():
        if isinstance(value, list):
            detail = f"list of {len(value)} items"
        elif isinstance(value, dict):
            detail = f"object with {len(value)} fields"
        elif isinstance(value, str):
            suffix = "…" if len(value) > 160 else ""
            detail = f"{value[:160]!r}{suffix}"
        else:
            detail = repr(value)
        lines.append(f"{name}: {detail}")
    return "\n".join(lines)


class Session:
    def __init__(self, input_json, query, output_limit):
        inputs = json.loads(input_json)
        if not isinstance(inputs, dict):
            raise TypeError("RLM signature inputs must serialize to an object")
        self.preview = input_preview(inputs)
        self.output_limit = output_limit
        self.turn = Turn()

        def query_one(prompt):
            answers = query([prompt])
            if not answers:
                raise RuntimeError("The submodel call budget is exhausted")
            return answers[0]

        self.globals = {
            **inputs,
            "__name__": "__dspy_rlm__",
            "inputs": inputs,
            "print": self.turn.print,
            "llm_query": query_one,
            "llm_query_batched": query,
            "SUBMIT": self.turn.submit,
        }

    def run(self, code):
        turn = self.turn
        turn.output = Output(self.output_limit)
        turn.submission = None
        error = None
        try:
            tree = ast.parse(code, filename="<rlm>", mode="exec")
            tail = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                tail = ast.Expression(tree.body.pop().value)
            exec(compile(tree, "<rlm>", "exec"), self.globals)
            if tail is not None:
                value = eval(compile(tail, "<rlm>", "eval"), self.globals)
                if value is not None:
                    turn.output.write(repr(value))
        except Submitted:
            pass
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"[: self.output_limit]
        return turn.output.text(), error, turn.submission
