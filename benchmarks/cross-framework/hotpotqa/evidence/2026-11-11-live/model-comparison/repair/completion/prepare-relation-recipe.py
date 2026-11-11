import base64,copy,json,pathlib,shlex,zlib
base=pathlib.Path(r'C:\Users\georg\Documents\dsrs-build\fewshot-repair');root=base/'relation-recipe';root.mkdir(exist_ok=True)
refs=json.loads((base/'context-references.json').read_text(encoding='utf-8'))
refs+=json.loads((base/'focused-role/prospective-fixture.json').read_text(encoding='utf-8'))['examples']
native=json.loads((base/'native-requests.json').read_text(encoding='utf-8'))
demos=json.loads((base/'demonstrations.json').read_text(encoding='utf-8'))
cases=[dict(id=r['id'],question=r['question'],context=r['context']) for r in refs]
# Original cases keep the already retrieved context. Later cases use their supplied short contexts.
for i in range(8):cases[i]['context']=native[i*3]['request']['messages'][-1]['content'].split('[[ ## context ## ]]\n',1)[1].rsplit('\n\nReturn a JSON object',1)[0]
systems=[native[s]['request']['messages'][0] for s in range(3)]
code=r'''
import copy,json,math,pathlib,re,subprocess,urllib.request
root=pathlib.Path('/tmp/dsrs-relation-recipe-2026-11-11');root.mkdir(exist_ok=False)
ip=subprocess.check_output(['docker','inspect','-f','{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}','localai-tasks-1'],text=True).strip()
out={'status':'running','not_performance_evidence':True,'records':[]}
def save():(root/'results.json').write_text(json.dumps(out),encoding='utf-8')
def focus(question,context):
    if not re.search(r'\b(position|office)\b',question,re.I):return context
    holding=re.compile(r'\b(?:served|serves|serving) as\b',re.I)
    appointment=re.compile(r'\b(?:appointed|named|elected|became)\b',re.I)
    split=re.compile(r'\band\s+(?:(?:also|later|subsequently|was|were)\s+)*(?:served|serves|serving|appointed|named|elected|became)\b',re.I)
    excluded=re.search(r'other than (.+?),',question,re.I)
    background=[];facts=[]
    for line in context.splitlines():
        title,sep,text=line.partition(':')
        if not sep:text=line
        label=title+': ' if sep else ''
        for sentence in re.split(r'(?<=[.!?])\s+',text):
            boundaries=[0]+[m.start()+3 for m in split.finditer(sentence)]+[len(sentence)]
            for a,b in zip(boundaries,boundaries[1:]):
                clause=sentence[a:b].strip()
                if clause.endswith(' and'):clause=clause[:-4].rstrip()
                kind=2 if holding.search(clause) else 1 if appointment.search(clause) else 0
                if kind:
                    if not excluded or not re.search(r'\b'+re.escape(excluded[1])+r'\b',clause,re.I):facts.append((kind,label+clause))
                else:background.append(label+clause)
    if not facts:return context
    # Explicit appointment wording in a question must not be replaced by an office-holding fact.
    preferred=1 if appointment.search(question) and any(k==1 for k,_ in facts) else max(k for k,_ in facts)
    return '\n'.join(background+[c for k,c in facts if k==preferred])
stop={'a','an','the','of','and','or','is','are','were','was','what','which','who','in','as','to','have','has','do','did','does','for','by','with','it','its','from','that','this','on','at','be'}
def terms(text):return {w.rstrip('s') for w in re.findall('[a-zA-Z]+',text.lower()) if w not in stop}
def select(question):
    query=terms(question);docs=[terms(d['question']) for d in demos]
    scores=[sum(math.log(1+len(docs)/(1+sum(w in x for x in docs))) for w in query&d) for d in docs]
    indices=sorted((i for i,s in enumerate(scores) if s>0),key=lambda i:(scores[i],i),reverse=True)[:2]
    return [demos[i] for i in sorted(indices)]
def user(values):return '\n\n'.join('[[ ## '+n+' ## ]]\n'+v for n,v in values.items())+'\n\nReturn a JSON object with every output field.'
try:
 for case in cases:
    context=focus(case['question'],case['context']);values={}
    for stage in range(3):
        field='critique' if stage==1 else 'answer';system=copy.deepcopy(systems[stage])
        if stage!=1:system['content'][0]['text']+='\n\nFor an office or position, give only the office title without its country or jurisdiction. Keep words integral to the title.'
        messages=[system]
        for demo in select(case['question']):
            inputs=dict(question=demo['question'],context=demo['context'])
            if stage==1:inputs['answer']=demo['draft']
            if stage==2:inputs.update(draft=demo['draft'],critique=demo['critique'])
            messages.extend([{'role':'user','content':user(inputs)},{'role':'assistant','content':[{'type':'text','text':json.dumps({field:demo[field]},separators=(',',':'))}]}])
        inputs=dict(question=case['question'],context=context)
        if stage==1:inputs['answer']=values['draft']
        if stage==2:inputs.update(draft=values['draft'],critique=values['critique'])
        messages.append({'role':'user','content':user(inputs)})
        body={'model':'Qwen3-0.6B-Tasks','messages':messages,'temperature':0.,'max_tokens':128,'seed':0,'response_format':{'type':'json_schema','json_schema':{'name':'response_schema','strict':True,'schema':{'type':'object','properties':{field:{'type':'string'}},'required':[field],'additionalProperties':False}}}}
        req=urllib.request.Request('http://'+ip+':8080/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=120) as response:data=json.load(response)
        parsed=json.loads(data['choices'][0]['message']['content']);out['records'].append({'id':case['id'],'stage':stage,'context':context,'request':body,'response':data,'parsed':parsed});save()
        if stage==0:values['draft']=parsed['answer']
        if stage==1:values['critique']=parsed['critique']
 out['status']='completed'
except Exception as exc:out.update(status='failed',error=repr(exc))
finally:save()
'''
source='cases='+repr(cases)+'\ndemos='+repr(demos)+'\nsystems='+repr(systems)+'\n'+code
(root/'runner.py').write_text(source,encoding='utf-8');compile(source,'relation-recipe','exec')
(root/'references.json').write_text(json.dumps(refs,indent=2),encoding='utf-8')
packed=base64.b64encode(zlib.compress(source.encode(),9)).decode();runner='import base64,zlib; exec(zlib.decompress(base64.b64decode('+repr(packed)+')))'
(root/'command.txt').write_text('qm guest exec 107 --synchronous 0 -- python3 -u -c '+shlex.quote(runner),encoding='utf-8')
print(root)
