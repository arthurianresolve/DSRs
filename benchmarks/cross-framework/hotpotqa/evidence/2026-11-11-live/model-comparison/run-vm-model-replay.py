"""Run fixed captured requests against the existing private model backend."""
import copy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import urllib.request

root = Path(bundle.get('output_directory', '/tmp/dsrs-model-replay-2026-11-11'))
root.mkdir(exist_ok=False)
assert len(bundle['records']) == 18
container = bundle.get('container', 'localai-llama-1')
ip = subprocess.check_output(['docker', 'inspect', '-f', '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}', container], text=True).strip()
base_url = 'http://' + ip + ':8080'

def request(path, data=None):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(base_url + path, body, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)

def normalized(value):
    value = ''.join(c for c in value.lower() if c.isalnum() or c.isspace())
    return ' '.join(word for word in value.split() if word not in {'a', 'an', 'the'})

models = request('/v1/models')
model = models['data'][0]['id']
assert model == bundle.get('expected_model', 'Qwen3-32B-obliterated-Q4_K_M')
image = subprocess.check_output(['docker', 'inspect', '-f', '{{.Image}}', container], text=True).strip()
arguments = json.loads(subprocess.check_output(['docker', 'inspect', '-f', '{{json .Args}}', container], text=True))
reasoning_flags = [(value, arguments[index + 1] if index + 1 < len(arguments) else '')
                   for index, value in enumerate(arguments) if 'reasoning' in value or 'template-kwargs' in value]
report = {'kind': 'fixed_prompt_model_comparison', 'not_performance_evidence': True,
          'method': 'Replay fixed requests captured from the measured clients, including original llama3.2 draft and critique inputs; not a fresh end-to-end Qwen pipeline.',
          'source_revision': bundle['source_revision'], 'captured_binary_sha256': bundle['binary_sha256'],
          'model': model, 'models_response': models, 'container_image_id': image,
          'kernel': platform.platform(), 'temperature': 0, 'max_tokens': 128, 'seed': 0,
          'provider_options': bundle.get('provider_options', {}),
          'server_reasoning_configuration': reasoning_flags,
          'model_weight_sha256': None, 'model_weight_hash_note': 'Not provided by the backend API; weights were not modified.',
          'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'records': [], 'status': 'running'}

def save():
    temporary = root / 'results.json.tmp'
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(root / 'results.json')

(root / 'prompts.json').write_text(json.dumps(bundle, indent=2), encoding='utf-8')
save()
try:
    for index, captured in enumerate(bundle['records']):
        body = copy.deepcopy(captured['request'])
        body['model'] = model
        body['seed'] = 0
        body.update(bundle.get('provider_options', {}))
        assert body['temperature'] == 0 and body['max_tokens'] == 128
        assert body['messages'] == captured['request']['messages']
        row = {'framework': captured['framework'], 'example_id': captured['example_id'], 'stage': captured['stage'],
               'request': body, 'original_model_response': captured['original_model_response'],
               'messages_sha256': hashlib.sha256(json.dumps(body['messages'], sort_keys=True).encode()).hexdigest(),
               'messages_unchanged': True, 'expected_final_answer': captured['expected_final_answer']}
        report['records'].append(row)
        try:
            response = request('/v1/chat/completions', body)
            row['response'] = response
            message = response['choices'][0]['message']
            row['response_content'] = message.get('content')
            row['finish_reason'] = response['choices'][0]['finish_reason']
            parsed = json.loads(message['content'])
            field = 'critique' if captured['stage'] == 'reflect' else 'answer'
            assert isinstance(parsed[field], str)
            row['parsed'] = parsed
            row['protocol_passed'] = True
            if captured['stage'] == 'revise':
                row['exact_match'] = normalized(parsed['answer']) == normalized(captured['expected_final_answer'])
        except Exception as error:
            row.update(protocol_passed=False, error=str(error))
            raise
        finally:
            save()
        print(f'{index + 1}/18 {captured["framework"]} {captured["stage"]} complete', flush=True)
    report['summary'] = {framework: {'final_answers': [
        {'example_id': r['example_id'], 'answer': r['parsed']['answer'], 'reference': r['expected_final_answer'], 'exact_match': r['exact_match']}
        for r in report['records'] if r['framework'] == framework and r['stage'] == 'revise']} for framework in ('dsrs', 'dspy')}
    for summary in report['summary'].values():
        summary['exact_matches'] = sum(row['exact_match'] for row in summary['final_answers'])
        summary['samples'] = len(summary['final_answers'])
    report['status'] = 'complete'
except Exception as error:
    report.update(status='failed', error=str(error))
    raise
finally:
    report['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    save()
print(json.dumps(report['summary']), flush=True)
