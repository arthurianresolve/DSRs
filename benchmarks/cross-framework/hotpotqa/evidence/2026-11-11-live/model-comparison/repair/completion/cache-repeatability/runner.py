body={'model': 'Qwen3-0.6B-Tasks', 'messages': [{'role': 'system', 'content': [{'type': 'text', 'text': 'Your input fields are:\n1. `question` (string)\n2. `context` (string)\n\nYour output fields are:\n1. `answer` (string)\n\nReturn compact JSON matching the provider JSON Schema. No markdown.\n\nAnswer the question using the supplied context. Put a short answer without explanation in the answer output field. Do not copy input fields.\n\nFor an office or position, give only the office title without its country or jurisdiction. Keep words integral to the title.'}]}, {'role': 'user', 'content': '[[ ## question ## ]]\nWhat position did the actor who played Rowan later hold?\n\n[[ ## context ## ]]\nIn The River, actor Ellis Vale played Rowan. Ellis Vale later served as mayor.\n\nReturn a JSON object with every output field.'}, {'role': 'assistant', 'content': [{'type': 'text', 'text': '{"answer":"mayor"}'}]}, {'role': 'user', 'content': "[[ ## question ## ]]\nWhat government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?\n\n[[ ## context ## ]]\nKiss and Tell (1945 film): Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer.\nShirley Temple: Shirley Temple Black (April 23, 1928 – February 10, 2014) was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938.\nShirley Temple: also served as Chief of Protocol of the United States.\n\nReturn a JSON object with every output field."}], 'temperature': 0.0, 'max_tokens': 128, 'response_format': {'type': 'json_schema', 'json_schema': {'name': 'response_schema', 'strict': True, 'schema': {'type': 'object', 'properties': {'answer': {'type': 'string'}}, 'required': ['answer'], 'additionalProperties': False}}}}

import copy,hashlib,json,pathlib,subprocess,urllib.request
root=pathlib.Path('/tmp/dsrs-cache-repeatability-2026-11-11');root.mkdir(exist_ok=False)
container=json.loads(subprocess.check_output(['docker','inspect','localai-tasks-1'],text=True))[0]
ip=next(iter(container['NetworkSettings']['Networks'].values()))['IPAddress']
mount=next(m['Source'] for m in container['Mounts'] if m['Destination']=='/models')
model=pathlib.Path(mount)/'qwen3-0.6b-q8.gguf'
out={'status':'running','not_performance_evidence':True,'records':[],'image':container['Image'],'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest()}
def save():(root/'results.json').write_text(json.dumps(out),encoding='utf-8')
try:
 for cached in [False,True,False,True,False,True]:
    request=copy.deepcopy(body);request.update(seed=0,cache_prompt=cached)
    req=urllib.request.Request('http://'+ip+':8080/v1/chat/completions',json.dumps(request).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as response:data=json.load(response)
    out['records'].append({'cache_prompt':cached,'request':request,'response':data,'parsed':json.loads(data['choices'][0]['message']['content'])});save()
 out['status']='completed'
except Exception as exc:out.update(status='failed',error=repr(exc))
finally:save()
