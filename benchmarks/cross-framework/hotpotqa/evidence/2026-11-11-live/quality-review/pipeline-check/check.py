"""Check the measured clients' prompt wiring and parsing using controlled replies."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

repo = Path(r'C:\Users\georg\Documents\DSRs-worktrees\fix-hotpotqa-profiling')
binary = Path(r'C:\Users\georg\Documents\dsrs-build\frozen-profile-bda64b2-f92fad4d\bench-hotpotqa-reflection.exe')
assert hashlib.sha256(binary.read_bytes()).hexdigest() == 'f92fad4d85f0e59e727677ed3e981bd2fdd730f37a7113aaf2861d49d0e85783'
examples = json.loads((repo / 'benchmarks/cross-framework/hotpotqa/fixture.json').read_text(encoding='utf-8'))['examples']
output = Path(r'C:\Users\georg\Documents\dsrs-build\hotpotqa-quality-audit-2026-11-11')
attempt = 2
while output.exists():
    output = output.with_name(f'hotpotqa-quality-audit-2026-11-11-attempt-{attempt}')
    attempt += 1
output.mkdir(exist_ok=False)
requests = []
counts = {'dsrs': 0, 'dspy': 0}

class Provider(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        framework = self.path.split('/')[1]
        index = counts[framework]
        counts[framework] += 1
        example, stage = examples[index // 3], index % 3
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        record = {'framework': framework, 'example_id': example['id'], 'stage': stage, 'request': body}
        requests.append(record)
        try:
            assert self.path == f'/{framework}/v1/chat/completions'
            assert len(body['messages']) == 2
            assert [m['role'] for m in body['messages']] == ['system', 'user']
            user = body['messages'][-1]['content']
            if isinstance(user, list):
                user = ''.join(block['text'] for block in user)
            draft = f"DRAFT-CHECK-{example['id']}"
            critique = f"CRITIQUE-CHECK-{example['id']}"
            expected = {'question': example['question'], 'context': example['context']}
            if stage == 1:
                expected['answer'] = draft
            if stage == 2:
                expected.update(draft=draft, critique=critique)
            for name, value in expected.items():
                assert f'[[ ## {name} ## ]]\n{value}\n\n' in user, f'{framework} stage {stage}: wrong {name}'
            field = 'critique' if stage == 1 else 'answer'
            schema = body['response_format']['json_schema']['schema']
            assert set(schema['properties']) == {field}
            assert body['temperature'] == 0 and body['max_tokens'] == 128
            record.update(validated_input_fields=list(expected), schema_output_field=field, passed=True)
            content = {field: [draft, critique, example['answer']][stage]}
            payload = {'id': 'controlled-test', 'object': 'chat.completion', 'created': 0, 'model': 'controlled-response',
                       'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': json.dumps(content)}}],
                       'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
            status = 200
        except Exception as error:
            record.update(passed=False, error=str(error))
            payload = {'error': {'message': str(error), 'type': 'pipeline_verification_failed'}}
            status = 400
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

server = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
threading.Thread(target=server.serve_forever, daemon=True).start()
results = []
try:
    for framework in ('dsrs', 'dspy'):
        command = [str(binary)] if framework == 'dsrs' else [sys.executable, str(repo / 'benchmarks/cross-framework/hotpotqa/bench_dspy.py')]
        command += ['1', '0', '--live', '--json-output', '--profile-runtime', '--check-protocol']
        environment = dict(os.environ, DSRS_PROFILE_BASE_URL=f'http://127.0.0.1:{server.server_port}/{framework}/v1', DSRS_PROFILE_MODEL='llama3.2:3b')
        environment.pop('DSRS_PROFILE_BARRIER', None)
        child = subprocess.run(command, env=environment, capture_output=True, text=True, encoding='utf-8', timeout=90)
        (output / f'{framework}.stdout.json').write_text(child.stdout, encoding='utf-8')
        (output / f'{framework}.stderr.txt').write_text(child.stderr, encoding='utf-8')
        assert child.returncode == 0, f'{framework} exit {child.returncode}; inspect {output}'
        data = json.loads(child.stdout)
        assert len(data['samples']) == 3 and counts[framework] == 9
        assert all(s['exact_match'] for s in data['samples'])
        assert [s['id'] for s in data['samples']] == [e['id'] for e in examples]
        results.append({'framework': framework, 'exit_code': child.returncode, 'requests': 9, 'exact_matches': 3, 'examples': 3})
        print(f'{framework}: prompt fields, schema, token limit, draft/critique propagation and 3/3 parsed final answers passed', flush=True)
    (output / 'verification.json').write_text(json.dumps({'kind': 'controlled_response_pipeline_check',
        'not_model_quality_evidence': True, 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'results': results}, indent=2), encoding='utf-8')
finally:
    (output / 'requests.json').write_text(json.dumps(requests, indent=2), encoding='utf-8')
    server.shutdown()
    server.server_close()
