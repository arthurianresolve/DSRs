"""Capture the measured clients' requests while replaying their retained model replies."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

repo = Path(r'C:\Users\georg\Documents\DSRs-worktrees\fix-hotpotqa-profiling')
evidence = repo / 'benchmarks/cross-framework/hotpotqa/evidence/2026-11-11-live'
binary = Path(r'C:\Users\georg\Documents\dsrs-build\frozen-profile-bda64b2-f92fad4d\bench-hotpotqa-reflection.exe')
assert hashlib.sha256(binary.read_bytes()).hexdigest() == 'f92fad4d85f0e59e727677ed3e981bd2fdd730f37a7113aaf2861d49d0e85783'
output = Path(r'C:\Users\georg\Documents\dsrs-build\vm107-model-comparison')
output.mkdir(exist_ok=False)
examples = json.loads((repo / 'benchmarks/cross-framework/hotpotqa/fixture.json').read_text(encoding='utf-8'))['examples']
baseline = {framework: json.loads((evidence / 'raw' / name).read_text(encoding='utf-8'))['samples']
            for framework, name in [('dsrs', '00-dsrs-timing.json'), ('dspy', '01-dspy-timing.json')]}
records = []
counts = {'dsrs': 0, 'dspy': 0}

class Replay(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        framework = self.path.split('/')[1]
        index = counts[framework]
        counts[framework] += 1
        example, stage = examples[index // 3], index % 3
        sample = baseline[framework][index // 3]
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        content = [{'answer': sample['draft']}, {'critique': sample['critique']}, {'answer': sample['answer']}][stage]
        user = body['messages'][-1]['content']
        if isinstance(user, list):
            user = ''.join(block['text'] for block in user)
        expected = {'question': example['question'], 'context': example['context']}
        if stage == 1:
            expected['answer'] = sample['draft']
        if stage == 2:
            expected.update(draft=sample['draft'], critique=sample['critique'])
        assert all(f'[[ ## {key} ## ]]\n{value}\n\n' in user for key, value in expected.items())
        records.append({'framework': framework, 'example_id': example['id'],
                        'stage': ['answer', 'reflect', 'revise'][stage], 'request': body,
                        'original_model_response': content,
                        'expected_final_answer': example['answer'] if stage == 2 else None})
        payload = {'id': 'retained-reply', 'object': 'chat.completion', 'created': 0, 'model': 'llama3.2:3b',
                   'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': json.dumps(content)}}],
                   'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

server = ThreadingHTTPServer(('127.0.0.1', 0), Replay)
threading.Thread(target=server.serve_forever, daemon=True).start()
try:
    for framework in ('dsrs', 'dspy'):
        command = [str(binary)] if framework == 'dsrs' else [sys.executable, str(repo / 'benchmarks/cross-framework/hotpotqa/bench_dspy.py')]
        command += ['1', '0', '--live', '--json-output', '--check-protocol']
        environment = dict(os.environ, DSRS_PROFILE_BASE_URL=f'http://127.0.0.1:{server.server_port}/{framework}/v1', DSRS_PROFILE_MODEL='llama3.2:3b')
        environment.pop('DSRS_PROFILE_BARRIER', None)
        child = subprocess.run(command, env=environment, capture_output=True, text=True, encoding='utf-8', timeout=90)
        (output / f'{framework}.stdout.json').write_text(child.stdout, encoding='utf-8')
        (output / f'{framework}.stderr.txt').write_text(child.stderr, encoding='utf-8')
        assert child.returncode == 0, child.stderr
        actual = json.loads(child.stdout)['samples']
        assert counts[framework] == 9
        assert [(s['draft'], s['critique'], s['answer']) for s in actual] == [(s['draft'], s['critique'], s['answer']) for s in baseline[framework]]
        print(f'{framework}: nine baseline requests captured; all retained stage replies reproduced', flush=True)
    assert len(records) == 18
    payload = {'kind': 'fixed_prompt_model_comparison', 'not_performance_evidence': True,
               'fixture': examples, 'source_revision': 'bda64b29974cdd22e4e8004976a7e5c8f4d2263c',
               'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'records': records}
    (output / 'prompts.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
finally:
    server.shutdown()
    server.server_close()
