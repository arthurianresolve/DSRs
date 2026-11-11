"""Run both measured clients end to end through the existing Proxmox shell transport.

The browser transports request/response batches. These runs test answer quality,
not latency: browser handoff and guest-agent overhead are inside SDK elapsed time.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

root = Path(r'C:\Users\georg\Documents\dsrs-build\vm107-tasks-live')
root.mkdir(exist_ok=False)
repo = Path(r'C:\Users\georg\Documents\DSRs-worktrees\fix-hotpotqa-profiling')
binary = Path(r'C:\Users\georg\Documents\dsrs-build\frozen-profile-bda64b2-f92fad4d\bench-hotpotqa-reflection.exe')
assert hashlib.sha256(binary.read_bytes()).hexdigest() == 'f92fad4d85f0e59e727677ed3e981bd2fdd730f37a7113aaf2861d49d0e85783'
lock = threading.Lock()
pending = {}
counts = {'dsrs': 0, 'dspy': 0}
records = []
vm_code = r'''
import base64,hashlib,json,pathlib,subprocess,urllib.request,zlib
root=pathlib.Path('/tmp/dsrs-tasks-live-2026-11-11')
if batch['round']==1: root.mkdir(exist_ok=False)
assert root.is_dir()
ip=subprocess.check_output(['docker','inspect','-f','{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}','localai-tasks-1'],text=True).strip()
responses=[]
for item in batch['requests']:
    body=item['body']; assert body['model']=='Qwen3-0.6B-Tasks' and body['max_tokens']==128 and body['temperature']==0
    body['seed']=0
    request=urllib.request.Request('http://'+ip+':8080/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=120) as response: data=json.load(response)
    responses.append({'framework':item['framework'],'response':data})
raw=json.dumps({'round':batch['round'],'responses':responses}).encode()
(root/('round-'+str(batch['round'])+'.json')).write_bytes(raw)
print('DSRS_PAYLOAD:'+base64.b64encode(zlib.compress(raw,9)).decode()+':END_DSRS DSRS_SHA256:'+hashlib.sha256(raw).hexdigest()+':END_HASH '+'X'*3000)
'''

class Bridge(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        framework = self.path.split('/')[1]
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        with lock:
            counts[framework] += 1
            number = counts[framework]
            record = {'framework': framework, 'round': number, 'request': body}
            records.append(record)
            group = pending.setdefault(number, {})
            group[framework] = body
            if len(group) == 2:
                batch = {'round': number, 'requests': [{'framework': f, 'body': group[f]} for f in ('dsrs', 'dspy')]}
                source = 'batch=' + repr(batch) + '\n' + vm_code
                packed = base64.b64encode(zlib.compress(source.encode(), 9)).decode()
                runner = 'import base64,zlib; exec(zlib.decompress(base64.b64decode(' + repr(packed) + ')))'
                command = 'qm guest exec 107 --timeout 300 -- python3 -u -c ' + shlex.quote(runner)
                (root / f'round-{number}-command.txt').write_text(command, encoding='utf-8')
                print(f'Round {number} ready: both framework requests captured', flush=True)
        result_file = root / f'round-{number}-response.json'
        deadline = time.monotonic() + 240
        while not result_file.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not result_file.exists():
            self.send_error(504, 'Browser transport timed out')
            return
        data = json.loads(result_file.read_text(encoding='utf-8'))
        assert data['round'] == number
        response = next(item['response'] for item in data['responses'] if item['framework'] == framework)
        with lock:
            record['response'] = response
            (root / 'requests.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
        raw = json.dumps(response).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

server = ThreadingHTTPServer(('127.0.0.1', 0), Bridge)
threading.Thread(target=server.serve_forever, daemon=True).start()
children = {}
streams = []
try:
    for framework in ('dsrs', 'dspy'):
        command = [str(binary)] if framework == 'dsrs' else [sys.executable, str(repo / 'benchmarks/cross-framework/hotpotqa/bench_dspy.py')]
        command += ['1', '0', '--live', '--json-output', '--check-protocol']
        environment = dict(os.environ, DSRS_PROFILE_BASE_URL=f'http://127.0.0.1:{server.server_port}/{framework}/v1', DSRS_PROFILE_MODEL='Qwen3-0.6B-Tasks')
        environment.pop('DSRS_PROFILE_BARRIER', None)
        stdout = (root / f'{framework}.stdout.json').open('w', encoding='utf-8')
        stderr = (root / f'{framework}.stderr.txt').open('w', encoding='utf-8')
        streams += [stdout, stderr]
        children[framework] = subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr)
    exits = {framework: child.wait(timeout=2400) for framework, child in children.items()}
    for stream in streams:
        stream.close()
    summary = {'kind': 'actual_framework_pipeline_quality', 'model': 'Qwen3-0.6B-Tasks',
               'not_performance_evidence': True, 'transport': 'Proxmox guest-agent via browser; latency includes UI handoff',
               'exit_codes': exits, 'request_counts': counts, 'frameworks': {}}
    for framework, exit_code in exits.items():
        if exit_code == 0:
            data = json.loads((root / f'{framework}.stdout.json').read_text(encoding='utf-8'))
            assert len(data['samples']) == 3 and counts[framework] == 9
            summary['frameworks'][framework] = {'exact_matches': sum(s['exact_match'] for s in data['samples']),
                'samples': 3, 'answers': [{k: s[k] for k in ('id','draft','critique','answer','expected_answer','exact_match')} for s in data['samples']]}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
finally:
    for child in children.values():
        if child.poll() is None:
            child.kill(); child.wait()
    for stream in streams:
        if not stream.closed:
            stream.close()
    server.shutdown(); server.server_close()
