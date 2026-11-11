import base64,json,pathlib,shlex,zlib
base=pathlib.Path(r'C:\Users\georg\Documents\dsrs-build\fewshot-repair');root=base/'cache-repeatability';root.mkdir(exist_ok=True)
rows=json.loads((base/'relation-recipe/native-live/requests.json').read_text(encoding='utf-8'))
body=next(r['request'] for r in rows if r['round']==13)
code=r'''
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
'''
source='body='+repr(body)+'\n'+code
(root/'runner.py').write_text(source,encoding='utf-8');compile(source,'cache-repeatability','exec')
packed=base64.b64encode(zlib.compress(source.encode(),9)).decode();runner='import base64,zlib; exec(zlib.decompress(base64.b64decode('+repr(packed)+')))'
(root/'command.txt').write_text('qm guest exec 107 --synchronous 0 -- python3 -u -c '+shlex.quote(runner),encoding='utf-8')
print(root)
