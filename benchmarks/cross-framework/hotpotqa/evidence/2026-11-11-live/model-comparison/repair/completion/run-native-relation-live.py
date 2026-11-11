"""Drive the frozen DSRs binary through real VM107 provider replies. Quality only."""
import argparse, base64, hashlib, json, os, pathlib, shlex, subprocess, threading, time, zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

base=pathlib.Path(r'C:\Users\georg\Documents\dsrs-build\fewshot-repair')
parser=argparse.ArgumentParser()
parser.add_argument('--out',default='native-live-fixed')
parser.add_argument('--binary',default='native-relation-fixed.exe')
args=parser.parse_args()
root=base/'relation-recipe'/args.out
root.mkdir(exist_ok=False)
binary=base/'relation-recipe'/args.binary
remote_root='/tmp/dsrs-native-relation-fixed-2026-11-11'
refs=json.loads((base/'context-references.json').read_text(encoding='utf-8'))
fresh=json.loads((base/'focused-role/prospective-fixture.json').read_text(encoding='utf-8'))['examples']
negative={'id':'excluded-only-office','question':'What office, other than governor, did Nora hold?',
          'context':'Nora: Nora served as governor.','answer':'No eligible answer is provided.'}
plans=[('baseline',refs[:3],False),('original',refs[:3],True),('fresh',refs[3:7],True),
       ('clarified',[refs[7]],True),('prospective',fresh,True),
       ('repeat-original-and-clarified',refs[:3]+[refs[7]],True),('excluded-only',[negative],True)]
state_path=root/'state.json';records=[];summaries=[];active='setup'
def write_json(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2),encoding='utf-8');temp.replace(path)
class Bridge(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        n=len(records)+1;record={'round':n,'run':active,'request':body};records.append(record)
        code=r'''
import pathlib,json,subprocess,urllib.request,base64,zlib,hashlib
root=pathlib.Path(remote_root)
if n==1:root.mkdir(exist_ok=False)
assert root.is_dir()
ip=subprocess.check_output(['docker','inspect','-f','{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}','localai-tasks-1'],text=True).strip()
body['seed']=0
req=urllib.request.Request('http://'+ip+':8080/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=120) as response:data=json.load(response)
raw=json.dumps({'round':n,'request':body,'response':data}).encode()
(root/('round-'+str(n)+'.json')).write_bytes(raw)
print('DSRS_REPLY:'+base64.b64encode(zlib.compress(raw,9)).decode()+':END_DSRS_REPLY '+hashlib.sha256(raw).hexdigest())
'''
        source='remote_root='+repr(remote_root)+'\nn='+repr(n)+'\nbody='+repr(body)+'\n'+code
        packed=base64.b64encode(zlib.compress(source.encode(),9)).decode()
        runner='import base64,zlib; exec(zlib.decompress(base64.b64decode('+repr(packed)+')))'
        command=root/f'round-{n}.command.txt';response_path=root/f'round-{n}.response.json'
        command.write_text('qm guest exec 107 --timeout 180 -- python3 -u -c '+shlex.quote(runner),encoding='utf-8')
        write_json(state_path,{'status':'request','round':n,'run':active,'command_path':str(command),'response_path':str(response_path)})
        deadline=time.monotonic()+1200
        while not response_path.exists() and time.monotonic()<deadline:time.sleep(.05)
        if not response_path.exists():self.send_error(504);return
        data=json.loads(response_path.read_text(encoding='utf-8'));assert data['round']==n
        sent=dict(data['request']);assert sent.pop('seed')==0 and sent==body
        response=data['response'];assert response['choices'][0]['finish_reason']=='stop'
        record['response']=response;write_json(root/'requests.json',records)
        raw=json.dumps(response).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
server=ThreadingHTTPServer(('127.0.0.1',0),Bridge);threading.Thread(target=server.serve_forever,daemon=True).start()
try:
    env=dict(os.environ,DSRS_PROFILE_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1',DSRS_PROFILE_MODEL='Qwen3-0.6B-Tasks')
    env.pop('DSRS_PROFILE_BARRIER',None)
    for name,rows,recipe in plans:
        active=name;run_dir=root/name;run_dir.mkdir();fixture=run_dir/'fixture.json';write_json(fixture,{'examples':rows})
        first=len(records)
        with (run_dir/'stdout.json').open('w',encoding='utf-8') as stdout,(run_dir/'stderr.txt').open('w',encoding='utf-8') as stderr:
            command=[str(binary),'1','0','--live','--json-output','--check-quality','--fixture',str(fixture)]
            if recipe:command.append('--retrieval-recipe')
            child=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr)
            exit_code=child.wait(timeout=7200)
        summary={'name':name,'native_exit':exit_code,'provider_requests':len(records)-first,'samples':len(rows),'recipe':recipe}
        if (run_dir/'stdout.json').stat().st_size:
            result=json.loads((run_dir/'stdout.json').read_text(encoding='utf-8'))
            summary.update(exact_matches=sum(r['exact_match'] for r in result['samples']),quality_status=result['quality_status'],answers=[r['answer'] for r in result['samples']])
        write_json(run_dir/'summary.json',summary);summaries.append(summary);print(json.dumps(summary),flush=True)
        write_json(root/'summary.json',{'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'not_performance_evidence':True,'runs':summaries})
    write_json(state_path,{'status':'completed','requests':len(records)})
finally:
    server.shutdown();server.server_close()
