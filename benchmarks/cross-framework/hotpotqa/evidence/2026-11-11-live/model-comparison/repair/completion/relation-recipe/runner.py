cases=[{'id': '5a8b57f25542995d1e6f1371', 'question': 'Were Scott Derrickson and Ed Wood of the same nationality?', 'context': 'Scott Derrickson: Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.\nEd Wood: Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director.'}, {'id': '5a8c7595554299585d9e36b6', 'question': 'What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?', 'context': "Kiss and Tell (1945 film): Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer.\nShirley Temple: Shirley Temple Black (April 23, 1928 – February 10, 2014) was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938.\nShirley Temple:  As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States."}, {'id': '5a85ea095542994775f606a8', 'question': 'What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?', 'context': 'Animorphs: Animorphs is a science fantasy series of young adult books written by Katherine Applegate and her husband Michael Grant, writing together under the name K. A. Applegate, and published by Scholastic.\nAnimorphs:  It is told in first person, with all six main characters taking turns narrating the books through their own perspectives.'}, {'id': 'fresh-same', 'question': 'Were Nira and Oswin of the same nationality?', 'context': 'Nira was a Japanese architect. Oswin was a Japanese composer.'}, {'id': 'fresh-different', 'question': 'Are Pavel and Suri of the same nationality?', 'context': 'Pavel is a Brazilian scientist. Suri is a Swedish scientist.'}, {'id': 'fresh-office', 'question': 'What public office did the performer who played Captain Vale subsequently hold?', 'context': 'Lena Marsh played Captain Vale in The Crossing. Lena Marsh subsequently served as governor.'}, {'id': 'fresh-series', 'question': 'Which series has The Glass Ledger as a companion volume?', 'context': 'The Glass Ledger is a companion volume to the Copper Sky series. Copper Sky is a fantasy novel series.'}, {'id': 'clarified-office', 'question': 'What government position, other than ambassador, was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?', 'context': "Kiss and Tell (1945 film): Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer.\nShirley Temple: Shirley Temple Black (April 23, 1928 – February 10, 2014) was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938.\nShirley Temple:  As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States."}, {'id': 'position-title-counterfactual', 'question': 'What government position was held by the woman who portrayed Iona in the film Harbour Lights?', 'context': 'Harbour Lights: The film starred Elena Brook as Iona.\nElena Brook: Elena was an actress, singer and diplomat.\nElena Brook: She was named ambassador to Velora and also served as Consul General.'}, {'id': 'position-exclusion-counterfactual', 'question': 'What government position, other than envoy, did the woman who played Sera hold?', 'context': 'The Orchard: Amara Field played Sera.\nAmara Field: She was named envoy to Torvia and also served as Treasurer.'}, {'id': 'position-named-only', 'question': 'What government position did the actor who portrayed Halden hold?', 'context': 'Winter Road: Jonas Vale portrayed Halden.\nJonas Vale: He was an actor and was appointed governor.'}, {'id': 'position-excluded-served-role', 'question': 'What office, other than governor, did the performer who played Tamsin hold?', 'context': 'The Ridge: Mira Lorne played Tamsin.\nMira Lorne: She served as governor and was later appointed Education Secretary.'}, {'id': 'acting-role-is-not-office', 'question': 'What role did the actress Dana Keene play in Night Ferry?', 'context': 'Night Ferry: Dana Keene played Captain Orla.\nDana Keene: She later served as mayor.'}, {'id': 'nationality-same-counterfactual', 'question': 'Were Petra and Milan of the same nationality?', 'context': 'Petra: Petra was a Polish architect.\nMilan: Milan was a Polish musician.'}, {'id': 'nationality-different-counterfactual', 'question': 'Are Ravi and Elna of the same nationality?', 'context': 'Ravi: Ravi is an Indian painter.\nElna: Elna is a Norwegian painter.'}, {'id': 'series-counterfactual', 'question': 'What fantasy young adult series, told in first person, has The Ember Codex as a companion book?', 'context': 'The Ember Codex: The Ember Codex is a companion book to the Silverwind series.\nThe Ember Codex: It narrates the lives of distant creatures.\nSilverwind: Silverwind is a fantasy young adult series told in first person.'}]
demos=[{'question': 'Are Ana and Luis of the same nationality?', 'context': 'Ana is Canadian. Luis is Canadian.', 'draft': 'no', 'critique': 'Both people are Canadian, so the answer should be yes.', 'answer': 'yes'}, {'question': 'Do Mara and Leon have the same nationality?', 'context': 'Mara is a Canadian engineer. Leon is a French engineer.', 'draft': 'no', 'critique': 'The answer is correct: Canadian and French are different nationalities.', 'answer': 'no'}, {'question': 'What position did the actor who played Rowan later hold?', 'context': 'In The River, actor Ellis Vale played Rowan. Ellis Vale later served as mayor.', 'draft': 'Ellis Vale', 'critique': 'The question asks for a position rather than a person; the position was mayor.', 'answer': 'mayor'}, {'question': 'Which novel series has the companion book The Moon Archive?', 'context': 'The Moon Archive is a companion book to the Starbound series. Starbound is a science fiction novel series.', 'draft': 'Starbound', 'critique': 'The answer is correct. Starbound is the series; The Moon Archive is a companion book.', 'answer': 'Starbound'}]
systems=[{'role': 'system', 'content': [{'type': 'text', 'text': 'Your input fields are:\n1. `question` (string)\n2. `context` (string)\n\nYour output fields are:\n1. `answer` (string)\n\nReturn compact JSON matching the provider JSON Schema. No markdown.\n\nAnswer the question using the supplied context. Put a short answer without explanation in the answer output field. Do not copy input fields.'}]}, {'role': 'system', 'content': [{'type': 'text', 'text': 'Your input fields are:\n1. `question` (string)\n2. `context` (string)\n3. `answer` (string)\n\nYour output fields are:\n1. `critique` (string)\n\nReturn compact JSON matching the provider JSON Schema. No markdown.\n\nCritique the answer against the supplied context. Put one brief sentence in the critique output field. Do not copy input fields.'}]}, {'role': 'system', 'content': [{'type': 'text', 'text': 'Your input fields are:\n1. `question` (string)\n2. `context` (string)\n3. `draft` (string)\n4. `critique` (string)\n\nYour output fields are:\n1. `answer` (string)\n\nReturn compact JSON matching the provider JSON Schema. No markdown.\n\nRevise the answer using the critique and supplied context. Put a short answer without explanation in the answer output field. Do not copy input fields.'}]}]

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
