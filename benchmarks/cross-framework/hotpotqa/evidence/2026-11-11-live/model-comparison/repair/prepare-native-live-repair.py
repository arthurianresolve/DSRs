import base64,json,pathlib,shlex,zlib
root=pathlib.Path(r'C:\Users\georg\Documents\dsrs-build\fewshot-repair')
requests=json.loads((root/'native-requests.json').read_text(encoding='utf-8'))
references=json.loads((root/'context-references.json').read_text(encoding='utf-8'))
cases=[]
for i,e in enumerate(references):
 rows=requests[i*3:i*3+3]
 text=rows[0]['request']['messages'][-1]['content']
 context=text.split('[[ ## context ## ]]\n',1)[1].rsplit('\n\nReturn a JSON object',1)[0]
 cases.append({'id':e['id'],'question':e['question'],'context':context,'templates':[r['request'] for r in rows]})
code=r'''
import copy,json,pathlib,subprocess,urllib.request
root=pathlib.Path('/tmp/dsrs-native-compact-recipe-2026-11-11'); root.mkdir(exist_ok=False)
ip=subprocess.check_output(['docker','inspect','-f','{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}','localai-tasks-1'],text=True).strip()
out={'kind':'real_provider_native_request_templates','status':'running','not_performance_evidence':True,'records':[]}
def save(): (root/'results.json').write_text(json.dumps(out),encoding='utf-8')
save()
try:
 for case in cases:
  values={'question':case['question'],'context':case['context']}
  for stage in range(3):
   body=copy.deepcopy(case['templates'][stage]); body['seed']=0
   names=['question','context']+(['answer'] if stage==1 else ['draft','critique'] if stage==2 else [])
   body['messages'][-1]['content']='\n\n'.join('[[ ## '+n+' ## ]]\n'+values[n] for n in names)+'\n\nReturn a JSON object with every output field.'
   req=urllib.request.Request('http://'+ip+':8080/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=120) as response: data=json.load(response)
   row={'id':case['id'],'stage':stage,'request':body,'response':data}; out['records'].append(row); save()
   parsed=json.loads(data['choices'][0]['message']['content']); row['parsed']=parsed
   if stage==0: values.update(answer=parsed['answer'],draft=parsed['answer'])
   if stage==1: values['critique']=parsed['critique']
   save()
 out['status']='completed'
except Exception as e: out.update(status='failed',error=repr(e))
finally: save()
'''
source='cases='+repr(cases)+'\n'+code
(root/'native-live-runner.py').write_text(source,encoding='utf-8')
compile(source,'native-live','exec')
packed=base64.b64encode(zlib.compress(source.encode(),9)).decode()
runner='import base64,zlib; exec(zlib.decompress(base64.b64decode('+repr(packed)+')))'
(root/'native-live-command.txt').write_text('qm guest exec 107 --synchronous 0 -- python3 -u -c '+shlex.quote(runner),encoding='utf-8')
print('Prepared actual native request templates, including native demonstration whitespace; references excluded from provider requests.')
