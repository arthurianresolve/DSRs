"""Verify retained fresh-provider replies and native pipeline field propagation."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
fixture_path = root.parents[2] / 'fixture.json'
examples = json.loads(fixture_path.read_text(encoding='utf-8'))['examples']
records = json.loads((root / 'tasks-live/requests.json').read_text(encoding='utf-8'))
summary = json.loads((root / 'tasks-live/summary.json').read_text(encoding='utf-8'))
assert len(records) == 18
verified = 0
for framework in ('dsrs', 'dspy'):
    assert summary['exit_codes'][framework] == 0
    output = json.loads((root / f'tasks-live/{framework}.stdout.json').read_text(encoding='utf-8'))
    rows = sorted((r for r in records if r['framework'] == framework), key=lambda r: r['round'])
    assert len(rows) == 9
    for i, (sample, example) in enumerate(zip(output['samples'], examples, strict=True)):
        assert sample['id'] == example['id'] and sample['expected_answer'] == example['answer']
        for stage, field in enumerate(('draft', 'critique', 'answer')):
            row = rows[i * 3 + stage]
            request, response = row['request'], row['response']
            assert row['round'] == i * 3 + stage + 1
            assert request['model'] == response['model'] == 'Qwen3-0.6B-Tasks'
            assert request['max_tokens'] == 128 and request['temperature'] == 0
            user = next(m['content'] for m in request['messages'] if m['role'] == 'user')
            for key in ('question', 'context'):
                assert f'[[ ## {key} ## ]]\n{example[key]}\n\n' in user
            if stage:
                name = 'answer' if stage == 1 else 'draft'
                assert f'[[ ## {name} ## ]]\n' + sample['draft'] + '\n\n' in user
            if stage == 2:
                assert '[[ ## critique ## ]]\n' + sample['critique'] + '\n\n' in user
            value = json.loads(response['choices'][0]['message']['content'])
            assert value == {('critique' if stage == 1 else 'answer'): sample[field]}
            assert response['choices'][0]['finish_reason'] == 'stop'
            verified += 1
    assert sum(s['exact_match'] for s in output['samples']) == 1
assert summary['not_performance_evidence'] is True
print(f'Passed: {verified} real replies, fixture inputs and stage propagation verified; quality failed (1/3 each).')
