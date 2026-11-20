"""Use DSRs field rendering and text coercion in a DSPy ChatAdapter."""

from __future__ import annotations

import json
from functools import lru_cache
from types import UnionType
from typing import Annotated, Any, NamedTuple, Union, get_args, get_origin

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.signatures.signature import Signature
from dspy.utils.exceptions import AdapterParseError
from json_repair import loads as repair_json
from pydantic import BaseModel, TypeAdapter, ValidationError

from ._dsrs_dspy import PreparedSignature


class _Output(NamedTuple):
    name: str
    annotation: Any
    validator: TypeAdapter
    plain_text: bool


class _SignaturePlan(NamedTuple):
    native: PreparedSignature
    structure: str
    outputs: tuple[_Output, ...]


@lru_cache(maxsize=256)
def _prepare_signature(signature: type[Signature]) -> _SignaturePlan:
    outputs = []
    schemas = []
    for name, field in signature.output_fields.items():
        annotation = field.rebuild_annotation()
        validator = TypeAdapter(annotation)
        outputs.append(_Output(name, annotation, validator, field.annotation is str))
        schemas.append({"name": name, "schema": validator.json_schema()})
    native = PreparedSignature(
        list(signature.input_fields), json.dumps(schemas, ensure_ascii=False, separators=(",", ":"))
    )
    return _SignaturePlan(native, native.format_field_structure(), tuple(outputs))


def _admits_none(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is Annotated:
        return _admits_none(get_args(annotation)[0])
    return annotation in (Any, object, type(None)) or any(
        choice is None or choice is type(None) for choice in get_args(annotation)
    )


def _drop_null_defaults(value: Any, annotation: Any) -> bool:
    """Let Pydantic apply defaults for nulls emitted for defaultable fields."""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Annotated:
        return _drop_null_defaults(value, args[0])
    if origin in (Union, UnionType):
        non_null = [choice for choice in args if choice is not type(None)]
        if len(non_null) == 1:
            return _drop_null_defaults(value, non_null[0])

    changed = False
    if isinstance(value, list) and origin is list and args:
        for item in value:
            changed |= _drop_null_defaults(item, args[0])
    elif isinstance(value, dict):
        if origin is dict and len(args) == 2:
            for item in value.values():
                changed |= _drop_null_defaults(item, args[1])
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            for name, field in annotation.model_fields.items():
                key = next(
                    (
                        key
                        for key in (field.validation_alias, field.alias, name)
                        if isinstance(key, str) and key in value
                    ),
                    None,
                )
                if key is None:
                    continue
                if value[key] is None and not field.is_required() and not _admits_none(field.annotation):
                    del value[key]
                    changed = True
                else:
                    changed |= _drop_null_defaults(value[key], field.annotation)
    return changed


def _parse_field(native: PreparedSignature, index: int, field: _Output, raw: str) -> Any:
    if field.plain_text:
        return field.validator.validate_python(raw)

    try:
        # Pydantic parses and validates clean JSON in one pass.
        return field.validator.validate_json(raw)
    except ValidationError as error:
        first_error = error.errors(include_url=False, include_context=False, include_input=False)[0]
        if first_error["type"] != "json_invalid":
            try:
                value = json.loads(raw)
            except (ValueError, RecursionError):
                raise error from None
            if _drop_null_defaults(value, field.annotation):
                return field.validator.validate_python(value)
            raise

        if "{" in raw or "[" in raw:
            repaired = repair_json(raw, skip_json_loads=True)
            if isinstance(repaired, (dict, list)):
                _drop_null_defaults(repaired, field.annotation)
                return field.validator.validate_python(repaired)

        coerced = native.coerce_text(index, raw)
        if coerced is not None:
            return field.validator.validate_json(coerced)
        return field.validator.validate_python(raw)


class DSRSBAMLAdapter(ChatAdapter):
    """DSPy chat adapter with DSRs schema rendering and tolerant response parsing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("use_json_adapter_fallback", False)
        super().__init__(*args, **kwargs)

    def format_field_structure(self, signature: type[Signature]) -> str:
        return _prepare_signature(signature).structure

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        plan = _prepare_signature(signature)
        result: dict[str, Any] = {}
        for index, (field, raw) in enumerate(zip(plan.outputs, plan.native.sections(completion))):
            try:
                if raw is None:
                    raise ValueError("missing output field")
                result[field.name] = _parse_field(plan.native, index, field, raw)
            except (ValueError, RecursionError) as error:
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    signature=signature,
                    lm_response=completion,
                    message=f"Output field `{field.name}`: {error}",
                ) from error
        return result
