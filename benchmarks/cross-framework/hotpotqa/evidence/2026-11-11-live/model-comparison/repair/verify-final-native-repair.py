import hashlib,json,os,pathlib,subprocess,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
root=pathlib.Path(r'C:\Users\georg\Documents\dsrs-build\fewshot-repair')
binary=root/'native-final-recipe.exe'
refs=json.loads((root/'context-references.json').read_text(encoding='utf-8'))
live=json.loads((root/'native-compact-results.json').read_text(encoding='utf-8'))['records']
baseline=[r for r in json.loads((root.parent/'vm107-tasks-live-attempt-2/requests.json').read_text(encoding='utf-8')) if r['framework']=='dsrs']
summary={'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'method':'native replay with exact real-provider request-body verification','not_performance_evidence':True,'runs':[]}
for name,rows,examples,recipe,expected_exit in [
 ('baseline',baseline,refs[:3],False,2),
 ('retrieval-original',live[:9],refs[:3],True,2),
 ('retrieval-fresh',live[9:21],refs[3:7],True,0),
]:
 path=root/('verify-'+name); path.mkdir(exist_ok=False)
 fixture=path/'fixture.json'; fixture.write_text(json.dumps({'examples':examples},indent=2),encoding='utf-8')
 records=[]; mismatches=[]
 class Replay(BaseHTTPRequestHandler):
  def log_message(self,*args): pass
  def do_POST(self):
   row=rows[len(records)]; body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
   expected={k:v for k,v in row['request'].items() if k!='seed'}
   if body!=expected:
    mismatches.append({'expected':expected,'actual':body}); self.send_error(400,'Native request mismatch'); return
   records.append(body); raw=json.dumps(row['response']).encode()
   self.send_response(200); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
 server=ThreadingHTTPServer(('127.0.0.1',0),Replay); threading.Thread(target=server.serve_forever,daemon=True).start()
 try:
  env=dict(os.environ,DSRS_PROFILE_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1',DSRS_PROFILE_MODEL='Qwen3-0.6B-Tasks');env.pop('DSRS_PROFILE_BARRIER',None)
  cmd=[str(binary),'1','0','--live','--json-output','--check-quality','--fixture',str(fixture)]
  if recipe: cmd.append('--retrieval-recipe')
  result=subprocess.run(cmd,env=env,capture_output=True,text=True,encoding='utf-8',timeout=120)
  (path/'stdout.json').write_text(result.stdout,encoding='utf-8'); (path/'stderr.txt').write_text(result.stderr,encoding='utf-8')
  if mismatches: (path/'mismatches.json').write_text(json.dumps(mismatches,indent=2),encoding='utf-8')
  assert not mismatches, name
  assert result.returncode==expected_exit,(name,result.returncode,result.stderr)
  assert len(records)==len(rows)
  data=json.loads(result.stdout); count=sum(s['exact_match'] for s in data['samples'])
  summary['runs'].append({'name':name,'native_exit':result.returncode,'exact_matches':count,'samples':len(examples),'verified_requests':len(records),'quality_status':data['quality_status']})
  print(name,count,'/',len(examples),'native exit',result.returncode,'all wire requests verified',flush=True)
 finally: server.shutdown();server.server_close()
for flags in [['--check-quality'],['--retrieval-recipe'],['--live','--check-quality','--profile-adapter']]:
 result=subprocess.run([str(binary),'1','0',*flags],capture_output=True,text=True,encoding='utf-8')
 assert result.returncode!=0,(flags,result.stdout)
summary['invalid_cli_combinations_rejected']=3
(root/'final-native-validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
