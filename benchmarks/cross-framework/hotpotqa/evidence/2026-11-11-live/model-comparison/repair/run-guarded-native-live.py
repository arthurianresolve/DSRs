"""One fresh native DSRs run via browser/guest-agent transport; quality only."""
import base64,json,os,pathlib,shlex,subprocess,threading,time,zlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
root=pathlib.Path(r'C:\Users\georg\Documents\dsrs-build\fewshot-repair\guarded-live-final'); root.mkdir(exist_ok=False)
parent=root.parent
refs=json.loads((parent/'context-references.json').read_text(encoding='utf-8'))
fixture=root/'fixture.json'; fixture.write_text(json.dumps({'examples':[r for r in refs if r['id']=='clarified-office']},indent=2),encoding='utf-8')
records=[]
class Bridge(BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_POST(self):
  body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  n=len(records)+1; row={'round':n,'request':body}; records.append(row)
  code="""
import pathlib,json,subprocess,urllib.request,base64,zlib,hashlib
root=pathlib.Path('/tmp/dsrs-guarded-native-live-final-2026-11-11')
if n==1: root.mkdir(exist_ok=False)
assert root.is_dir()
ip=subprocess.check_output(['docker','inspect','-f','{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}','localai-tasks-1'],text=True).strip()
body['seed']=0
req=urllib.request.Request('http://'+ip+':8080/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=120) as response: data=json.load(response)
raw=json.dumps({'round':n,'request':body,'response':data}).encode()
(root/('round-'+str(n)+'.json')).write_bytes(raw)
print('DSRS_PAYLOAD:'+base64.b64encode(zlib.compress(raw,9)).decode()+':END_DSRS DSRS_SHA256:'+hashlib.sha256(raw).hexdigest()+':END_HASH '+'X'*3000)
"""
  source='n='+repr(n)+'\nbody='+repr(body)+'\n'+code
  packed=base64.b64encode(zlib.compress(source.encode(),9)).decode()
  runner='import base64,zlib; exec(zlib.decompress(base64.b64decode('+repr(packed)+')))'
  (root/f'round-{n}.command.txt').write_text('qm guest exec 107 --timeout 180 -- python3 -u -c '+shlex.quote(runner),encoding='utf-8')
  print(f'Native request {n} ready',flush=True)
  response_path=root/f'round-{n}.response.json'; deadline=time.monotonic()+900
  while not response_path.exists() and time.monotonic()<deadline: time.sleep(.1)
  if not response_path.exists(): self.send_error(504); return
  data=json.loads(response_path.read_text(encoding='utf-8')); assert data['round']==n
  sent=dict(data['request']); sent.pop('seed'); assert sent==body
  row['response']=data['response']; (root/'requests.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
  raw=json.dumps(data['response']).encode(); self.send_response(200); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
server=ThreadingHTTPServer(('127.0.0.1',0),Bridge); threading.Thread(target=server.serve_forever,daemon=True).start()
try:
 env=dict(os.environ,DSRS_PROFILE_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1',DSRS_PROFILE_MODEL='Qwen3-0.6B-Tasks'); env.pop('DSRS_PROFILE_BARRIER',None)
 with (root/'stdout.json').open('w',encoding='utf-8') as stdout,(root/'stderr.txt').open('w',encoding='utf-8') as stderr:
  cmd=[str(parent/'native-final-recipe.exe'),'1','0','--live','--json-output','--check-quality','--retrieval-recipe','--fixture',str(fixture)]
  child=subprocess.Popen(cmd,env=env,stdout=stdout,stderr=stderr); code=child.wait(timeout=1800)
 summary={'native_exit':code,'provider_requests':len(records),'not_performance_evidence':True}
 if code==0: summary['result']=json.loads((root/'stdout.json').read_text(encoding='utf-8'))
 (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary),flush=True)
finally: server.shutdown(); server.server_close()
