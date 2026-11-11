"""Native CLI regressions: repair a bad draft, reject a bad final, isolate labels."""
import argparse, hashlib, json, os, pathlib, subprocess, tempfile, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def main():
    parser=argparse.ArgumentParser();parser.add_argument('binary',type=pathlib.Path);parser.add_argument('--out',type=pathlib.Path);args=parser.parse_args()
    binary=args.binary.resolve()
    fixture={'examples':[{'id':'repair-draft','question':'What office did Mira hold?',
        'context':'Mira: She served as Consul General of Canada.','answer':'Consul General'}]}
    def run(steps,expected):
        requests=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(body);index=len(requests)-1
                if index>=len(steps):self.send_error(500);return
                field,text=steps[index]
                assert list(body['response_format']['json_schema']['schema']['properties'])==[field]
                raw=json.dumps({'id':'regression','object':'chat.completion','created':0,'model':'regression',
                    'choices':[{'index':0,'message':{'role':'assistant','content':json.dumps({field:text})},'finish_reason':'stop'}],
                    'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            with tempfile.TemporaryDirectory(prefix='dsrs-reflection-regression-') as directory:
                path=pathlib.Path(directory)/'fixture.json';row=json.loads(json.dumps(fixture));row['examples'][0]['answer']=expected;path.write_text(json.dumps(row),encoding='utf-8')
                env=dict(os.environ,DSRS_PROFILE_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1',DSRS_PROFILE_MODEL='regression');env.pop('DSRS_PROFILE_BARRIER',None)
                result=subprocess.run([str(binary),'1','0','--live','--json-output','--check-quality','--retrieval-recipe','--fixture',str(path)],env=env,capture_output=True,text=True,timeout=30)
                return result,requests
        finally:server.shutdown();server.server_close()
    repair=[('answer','mayor'),('critique','The context states Consul General of Canada, not mayor.'),('answer','Consul General of Canada')]
    fixed,requests=run(repair,'Consul General')
    assert fixed.returncode==0,(fixed.returncode,fixed.stderr)
    assert len(requests)==3
    report=json.loads(fixed.stdout);row=report['samples'][0]
    assert row['draft']=='mayor' and row['revised_answer']=='Consul General of Canada' and row['answer']=='Consul General' and row['exact_match']
    rejected,calls=run([('answer','mayor'),('critique','Keep mayor.')]+[('answer','mayor')]*3,'Consul General')
    assert rejected.returncode==1 and len(calls)==5,(rejected.returncode,len(calls),rejected.stderr)
    assert 'contiguous phrase' in rejected.stderr
    changed,changed_requests=run(repair,'Deliberately incorrect evaluation label')
    assert changed.returncode==2 and changed_requests==requests
    assert json.loads(changed.stdout)['samples'][0]['answer']=='Consul General'
    if args.out:
        data={'kind':'native_cli_mock_provider_regressions','binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
            'cases':[{'name':name,'exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'requests':calls}
                for name,result,calls in [('repair',fixed,requests),('reject',rejected,calls),('changed-reference',changed,changed_requests)]]}
        args.out.write_text(json.dumps(data,indent=2),encoding='utf-8')
    print('Passed 3 native regressions: reflection repairs an unsupported draft; invalid final stops after 3 attempts; changing references cannot change requests or answers.')

if __name__=='__main__':main()
