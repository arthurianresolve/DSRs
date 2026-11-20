import dspy
import pytest
from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import AdapterParseError
from pydantic import BaseModel, Field

import dsrs_dspy.adapter as adapter_module
from dsrs_dspy import DSRSBAMLAdapter


class Address(BaseModel):
    city: str = Field(description="Municipality as written")
    postal_code: str = Field(alias="postalCode")


class Record(BaseModel):
    address: Address
    verified: bool
    note: str | None = None
    label: str = Field(default_factory=lambda values: values["address"].city)


class Extract(dspy.Signature):
    """Extract the record."""

    text: str = dspy.InputField()
    record: Record = dspy.OutputField()
    count: int = dspy.OutputField()


CLEAN = """
[[ ## record ## ]]
{"address":{"city":"München","postalCode":"80331"},"verified":true,"label":"verified"}
[[ ## count ## ]] 1234
[[ ## completed ## ]]
"""

NOISY = """
[[  ## record ##]]
{'address': {'city': 'München', 'postalCode': '80331',}, 'verified': True, 'note': None, 'label': None,}
[[## count ## ]] 1,234
[[ ## completed ## ]]
"""


def test_render_and_parse_reuse_signature(monkeypatch):
    adapter_module._prepare_signature.cache_clear()
    adapter = DSRSBAMLAdapter()

    def unexpected_repair(*args, **kwargs):
        pytest.fail("clean JSON reached the repair path")

    monkeypatch.setattr(adapter_module, "repair_json", unexpected_repair)
    for _ in range(2):
        prompt = adapter.format(Extract, demos=[], inputs={"text": "sample"})
        if isinstance(prompt, list):
            system, user = prompt[0]["content"], prompt[-1]["content"]
        else:
            system, user = prompt.system, prompt.messages[-1].text
        assert "Municipality as written" in system
        assert "postalCode: string" in system
        assert "[[ ## text ## ]]\nsample" in user
        result = adapter.parse(Extract, CLEAN)
        assert result["record"].address.postal_code == "80331"
        assert result["count"] == 1234
    assert adapter_module._prepare_signature.cache_info().misses == 1


@pytest.mark.parametrize("completion", [NOISY, CLEAN.replace('"verified"}', "null}")])
def test_repair_and_pydantic_default_factory(completion):
    result = DSRSBAMLAdapter().parse(Extract, completion)
    assert result["record"] == Record(
        address=Address(city="München", postalCode="80331"),
        verified=True,
        note=None,
        label="München",
    )
    assert result["count"] == 1234


def test_missing_output_does_not_retry_the_model():
    class CountingLM(DummyLM):
        calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            return super().__call__(*args, **kwargs)

    lm = CountingLM([{"count": 42}])
    with pytest.raises(AdapterParseError, match="record"):
        DSRSBAMLAdapter()(lm, {}, Extract, [], {"text": "sample"})
    assert lm.calls == 1


def test_invalid_typed_output_surfaces_as_parse_error():
    with pytest.raises(AdapterParseError, match="count"):
        DSRSBAMLAdapter().parse(Extract, CLEAN.replace("1234", "unknown"))


def test_render_includes_nested_union_model_fields():
    class Decision(BaseModel):
        result: Address | Record

    class Choose(dspy.Signature):
        text: str = dspy.InputField()
        decision: Decision = dspy.OutputField()

    adapter = DSRSBAMLAdapter()
    structure = adapter.format_field_structure(Choose)
    assert "postalCode" in structure
    assert "verified" in structure
    address = Address(city="München", postalCode="80331")
    for choice in (address, Record(address=address, verified=True)):
        payload = choice.model_dump_json(by_alias=True)
        completion = f'[[ ## decision ## ]]\n{{"result":{payload}}}'
        assert adapter.parse(Choose, completion)["decision"].result == choice
