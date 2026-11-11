import copy,hashlib,json,os,pathlib,subprocess,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
root=pathlib.Path(r'C:\Users\georg\Documents\dsrs-build\fewshot-repair')
binary=root/'native-recipe.exe'
rows=json.loads((root/'context-repair-results.json').read_text(encoding='utf-8'))['records']
references=json.loads((root/'context-references.json').read_text(encoding='utf-8'))
(root/'native-fixture.json').write_text(json.dumps({'examples':references},indent=2),encoding='utf-8')
records=[]
class Capture(BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_POST(self):
  row=rows[len(records)]; body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  wire_body=copy.deepcopy(body)
  expected={k:v for k,v in row['request'].items() if k!='seed'}
  for message in body['messages']:
   if message['role']=='assistant' and isinstance(message['content'],list):
    message['content']=''.join(part['text'] for part in message['content'])
  assert body==expected,(body,expected)
  assert body['temperature']==0 and body['max_tokens']==128
  records.append({'id':row['id'],'stage':row['stage'],'request':wire_body,'response':row['response']})
  raw=json.dumps(row['response']).encode()
  self.send_response(200); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
server=ThreadingHTTPServer(('127.0.0.1',0),Capture)
threading.Thread(target=server.serve_forever,daemon=True).start()
try:
 env=dict(os.environ,DSRS_PROFILE_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1',DSRS_PROFILE_MODEL='Qwen3-0.6B-Tasks')
 env.pop('DSRS_PROFILE_BARRIER',None)
 cmd=[str(binary),'1','0','--live','--json-output','--check-quality','--retrieval-recipe','--fixture',str(root/'native-fixture.json')]
 result=subprocess.run(cmd,env=env,capture_output=True,text=True,encoding='utf-8',timeout=120)
 (root/'native-capture.stdout.json').write_text(result.stdout,encoding='utf-8')
 (root/'native-capture.stderr.txt').write_text(result.stderr,encoding='utf-8')
 assert result.returncode==2,result.stderr
 assert len(records)==24
 data=json.loads(result.stdout)
 assert [s['answer'] for s in data['samples']]==[r['parsed']['answer'] for r in rows if r['stage']==2]
 (root/'native-requests.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
 (root/'native-capture.json').write_text(json.dumps({'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'command':cmd,'native_exit':result.returncode,'requests':len(records),'quality_status':data['quality_status']},indent=2),encoding='utf-8')
 print('Native capture passed: 24 exact live inputs, all actual provider answers reproduced, failing quality exits 2.')
finally: server.shutdown(); server.server_close()
