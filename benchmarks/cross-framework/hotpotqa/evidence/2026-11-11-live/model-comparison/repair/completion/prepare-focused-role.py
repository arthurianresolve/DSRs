import base64,copy,json,pathlib,shlex,zlib
base=pathlib.Path(r'C:\Users\georg\Documents\dsrs-build\fewshot-repair');root=base/'focused-role';root.mkdir(exist_ok=True)
refs=json.loads((base/'context-references.json').read_text(encoding='utf-8'))
native=json.loads((base/'native-requests.json').read_text(encoding='utf-8'))
cases=[]
for i,row in enumerate(refs):
 templates=copy.deepcopy([r['request'] for r in native[i*3:i*3+3]])
 context=templates[0]['messages'][-1]['content'].split('[[ ## context ## ]]\n',1)[1].rsplit('\n\nReturn a JSON object',1)[0]
 cases.append(dict(id=row['id'],question=row['question'],context=context,templates=templates))
code=r'''
import copy,json,pathlib,re,subprocess,urllib.request
root=pathlib.Path('/tmp/dsrs-focused-role-2026-11-11');root.mkdir(exist_ok=False)
ip=subprocess.check_output(['docker','inspect','-f','{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}','localai-tasks-1'],text=True).strip()
out={'status':'running','not_performance_evidence':True,'records':[]}
def save(): (root/'results.json').write_text(json.dumps(out),encoding='utf-8')
def focus(question,context):
 if not re.search(r'\b(position|office|role)\b',question,re.I):return context
 lines=[]
 for line in context.splitlines():
  clauses=re.split(r'(?<=[.!?])\s+|\band (?:also |later |subsequently )?(?=(?:served|serves|became|was appointed)\b)',line)
  relevant=[c for c in clauses if re.search(r'\b(?:served|serves|serving) as\b',c,re.I)]
  if relevant:
   title=line.split(':',1)[0]+': ' if ':' in line else ''
   lines.extend(title+c.strip() for c in relevant)
  elif not re.search(r'\b(?:actress|actor|singer|dancer|born|died|diplomat)\b',line,re.I) or re.search(r'\b(?:played|portrayed|starring)\b',line,re.I):
   lines.append(line)
 return '\n'.join(lines) or context
def user(values):return '\n\n'.join('[[ ## '+n+' ## ]]\n'+v for n,v in values.items())+'\n\nReturn a JSON object with every output field.'
try:
 for case in cases:
  context=focus(case['question'],case['context']);values={'question':case['question'],'context':context}
  for stage in range(3):
   body=copy.deepcopy(case['templates'][stage]);body['seed']=0
   inputs=dict(question=values['question'],context=context)
   if stage==1:inputs['answer']=values['draft']
   if stage==2:inputs.update(draft=values['draft'],critique=values['critique'])
   body['messages'][-1]['content']=user(inputs)
   req=urllib.request.Request('http://'+ip+':8080/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=120) as response:data=json.load(response)
   parsed=json.loads(data['choices'][0]['message']['content']);out['records'].append({'id':case['id'],'stage':stage,'context':context,'request':body,'response':data,'parsed':parsed});save()
   if stage==0:values['draft']=parsed['answer']
   if stage==1:values['critique']=parsed['critique']
 out['status']='completed'
except Exception as exc:out.update(status='failed',error=repr(exc))
finally:save()
'''
source='cases='+repr(cases)+'\n'+code
(root/'runner.py').write_text(source,encoding='utf-8');compile(source,'focused-role','exec')
packed=base64.b64encode(zlib.compress(source.encode(),9)).decode();runner='import base64,zlib; exec(zlib.decompress(base64.b64decode('+repr(packed)+')))'
(root/'command.txt').write_text('qm guest exec 107 --synchronous 0 -- python3 -u -c '+shlex.quote(runner),encoding='utf-8')
print(root)
