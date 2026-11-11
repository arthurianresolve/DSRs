import base64, copy, json, pathlib, shlex, zlib

base = pathlib.Path(r'C:\Users\georg\Documents\dsrs-build\fewshot-repair')
root = base / 'role-demonstrations'
root.mkdir(exist_ok=True)
refs = json.loads((base / 'context-references.json').read_text(encoding='utf-8'))
native = json.loads((base / 'native-requests.json').read_text(encoding='utf-8'))
demos = json.loads((base / 'demonstrations.json').read_text(encoding='utf-8'))
demos[2] = dict(question='What government position did the woman who played Mira in The Island hold?',
    context='The Island: Celia Hart portrayed Mira.\nCelia Hart: She was a diplomat, was appointed ambassador to Lydora, and served as Minister of Education.',
    draft='ambassador',critique='The question asks for a government position. The specifically named office in the context is Minister of Education.',answer='Minister of Education')
demos.append(dict(question='What public office, other than governor, did the performer who played Arlen hold?',
    context='The Harbour: Robin Kell played Arlen.\nRobin Kell: Kell was a governor and later served as Finance Secretary.',
    draft='governor',critique='The question excludes governor. The other office stated in the context is Finance Secretary.',answer='Finance Secretary'))
cases=[]
for i,row in enumerate(refs):
    templates=copy.deepcopy([r['request'] for r in native[i*3:i*3+3]])
    context=templates[0]['messages'][-1]['content'].split('[[ ## context ## ]]\n',1)[1].rsplit('\n\nReturn a JSON object',1)[0]
    cases.append(dict(id=row['id'],question=row['question'],context=context,templates=templates))
code=r'''
import copy,json,math,pathlib,re,subprocess,urllib.request
root=pathlib.Path('/tmp/dsrs-role-demonstrations-2026-11-11'); root.mkdir(exist_ok=False)
ip=subprocess.check_output(['docker','inspect','-f','{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}','localai-tasks-1'],text=True).strip()
out={'status':'running','not_performance_evidence':True,'records':[]}
def save(): (root/'results.json').write_text(json.dumps(out),encoding='utf-8')
stop={'a','an','the','of','and','or','is','are','were','was','what','which','who','in','as','to','have','has','do','did','does','for','by','with','it','its','from','that','this','on','at','be'}
def terms(text): return {w.rstrip('s') for w in re.findall('[a-zA-Z]+',text.lower()) if w not in stop}
def select(question):
    q=terms(question); docs=[terms(d['question']) for d in demos]
    scores=[sum(math.log(1+len(docs)/(1+sum(w in x for x in docs))) for w in q&d) for d in docs]
    indices=sorted((i for i,s in enumerate(scores) if s>0),key=lambda i:(scores[i],i),reverse=True)[:2]
    return [demos[i] for i in sorted(indices)]
def user(values): return '\n\n'.join('[[ ## '+n+' ## ]]\n'+v for n,v in values.items())+'\n\nReturn a JSON object with every output field.'
def choices(question,context):
    if question.lower().split()[0] in ('is','are','was','were','do','does','did','can','has','have'): return ['yes','no']
    excluded=re.search(r'other than (.+?),',question,re.I); spans=set()
    for line in context.splitlines():
        for clause in re.split(r'[.!?;]',line):
            words=list(re.finditer(r'\b[\w-]+\b',clause))
            for start in range(len(words)):
                for end in range(start,min(start+8,len(words))):
                    phrase=clause[words[start].start():words[end].end()]
                    if not excluded or not re.search(r'\b'+re.escape(excluded[1])+r'\b',phrase,re.I): spans.add(phrase)
    return sorted(spans)
try:
 for condition in ['role-demos','role-demos-choices']:
  for case in cases:
   selected=select(case['question']); values={'question':case['question'],'context':case['context']}
   for stage in range(3):
    body=copy.deepcopy(case['templates'][stage]);body['seed']=0
    messages=body['messages'][:1]
    for demo in selected:
     inputs={'question':demo['question'],'context':demo['context']}
     if stage==1: inputs['answer']=demo['draft']
     if stage==2: inputs.update(draft=demo['draft'],critique=demo['critique'])
     field='critique' if stage==1 else 'answer'
     messages.extend([{'role':'user','content':user(inputs)},{'role':'assistant','content':[{'type':'text','text':json.dumps({field:demo[field]},separators=(',',':'))}]}])
    inputs={'question':values['question'],'context':values['context']}
    if stage==1: inputs['answer']=values['draft']
    if stage==2: inputs.update(draft=values['draft'],critique=values['critique'])
    messages.append({'role':'user','content':user(inputs)});body['messages']=messages
    if condition.endswith('choices') and stage!=1:
     body['response_format']['json_schema']['schema']['properties']['answer']['enum']=choices(case['question'],case['context'])
    req=urllib.request.Request('http://'+ip+':8080/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as response:data=json.load(response)
    parsed=json.loads(data['choices'][0]['message']['content'])
    out['records'].append({'condition':condition,'id':case['id'],'stage':stage,'request':body,'response':data,'parsed':parsed});save()
    if stage==0:values['draft']=parsed['answer']
    if stage==1:values['critique']=parsed['critique']
 out['status']='completed'
except Exception as exc:out.update(status='failed',error=repr(exc))
finally:save()
'''
source='cases='+repr(cases)+'\ndemos='+repr(demos)+'\n'+code
(root/'runner.py').write_text(source,encoding='utf-8')
(root/'demonstrations.json').write_text(json.dumps(demos,indent=2),encoding='utf-8')
compile(source,'role-demos','exec')
packed=base64.b64encode(zlib.compress(source.encode(),9)).decode()
runner='import base64,zlib; exec(zlib.decompress(base64.b64decode('+repr(packed)+')))'
(root/'command.txt').write_text('qm guest exec 107 --synchronous 0 -- python3 -u -c '+shlex.quote(runner),encoding='utf-8')
print(root)
