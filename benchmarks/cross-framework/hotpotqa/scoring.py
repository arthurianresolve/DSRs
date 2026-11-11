"""Strict HotpotQA and separately labelled context-supported answer scoring."""

import json
from pathlib import Path


SUPPORTED_METRIC = "context_supported_normalized_exact_match"


class SupportedAnswerAnnotations(dict):
    def __init__(self, annotation_set_id, examples):
        super().__init__(examples)
        self.annotation_set_id = annotation_set_id


def normalized(text):
    text = "".join(character for character in text.lower()
                   if character.isalnum() or character.isspace())
    return " ".join(word for word in text.split() if word not in {"a", "an", "the"})


def load_annotations(path=None):
    path = Path(path) if path else Path(__file__).with_name("supported-answer-annotations.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"unsupported answer-annotation schema version {data.get('schema_version')}")
    annotation_set_id = data.get("annotation_set_id", "")
    if not isinstance(annotation_set_id, str) or not annotation_set_id.strip():
        raise ValueError("answer-annotation set id must not be empty")
    if data.get("metric") != SUPPORTED_METRIC:
        raise ValueError(f"unsupported answer-annotation metric {data.get('metric')!r}")
    if not data.get("description", "").strip():
        raise ValueError("answer-annotation description must not be empty")

    if not isinstance(data.get("examples"), list):
        raise ValueError("answer-annotation examples must be a list")
    annotations = {}
    for annotation in data["examples"]:
        identifier = annotation.get("id", "")
        if not identifier or identifier in annotations:
            raise ValueError(f"missing or duplicate answer annotation id {identifier!r}")
        for field in ("question", "context", "reference_answer", "rationale"):
            if not annotation.get(field, "").strip():
                raise ValueError(f"answer annotation {identifier!r} must define {field}")
        accepted = annotation.get("accepted_answers", [])
        normalized_answers = [normalized(answer) for answer in accepted]
        if (not normalized_answers or any(not answer for answer in normalized_answers)
                or len(set(normalized_answers)) != len(normalized_answers)):
            raise ValueError(f"answer annotation {identifier!r} has invalid accepted answers")
        if normalized(annotation["reference_answer"]) not in normalized_answers:
            raise ValueError(f"answer annotation {identifier!r} does not retain its strict reference")
        annotations[identifier] = annotation
    return SupportedAnswerAnnotations(annotation_set_id, annotations)


def score_answer(annotations, example, answer):
    annotation = annotations.get(example["id"])
    if annotation:
        frozen = (annotation["question"], annotation["context"], annotation["reference_answer"])
        current = (example["question"], example["context"], example["answer"])
        if frozen != current:
            raise ValueError(
                f"answer annotation {example['id']!r} does not match the frozen question, context and reference"
            )
        accepted_answers = list(annotation["accepted_answers"])
    else:
        accepted_answers = [example["answer"]]
    normalized_answer = normalized(answer)
    return {
        "exact_match": normalized_answer == normalized(example["answer"]),
        "supported_answer_match": any(
            normalized_answer == normalized(accepted) for accepted in accepted_answers
        ),
        "supported_answer_annotation": annotation is not None,
        "accepted_answers": accepted_answers,
    }
