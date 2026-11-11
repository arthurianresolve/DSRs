import base64,json,pathlib,shlex,zlib
root=pathlib.Path(r'C:\Users\georg\Documents\dsrs-build\fewshot-repair\title-normalization');root.mkdir(exist_ok=True)
cases=['chief of protocol','Chief of Protocol of the United States','governor','consul general','treasurer','education secretary','Secretary of the Treasury','Clerk of the House of Lords','Minister of Defence of the United Kingdom','Minister of Transport of Eldoria']
code=r'''
import json,pathlib,subprocess,urllib.request
root=pathlib.Path('/tmp/dsrs-title-normalization-2026-11-11');root.mkdir(exist_ok=False)
ip=subprocess.check_output(['docker','inspect','-f','{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}','localai-tasks-1'],text=True).strip()
instruction='Remove a trailing country or jurisdiction from the supplied office title. Keep every word integral to the office itself. Copy a contiguous phrase from the supplied title. Preserve the full title when it has no jurisdiction qualifier.'
demos=[('Minister of Transport of Eldoria','Minister of Transport'),('Secretary of the Treasury','Secretary of the Treasury'),('Clerk of the House of Lords','Clerk of the House of Lords')]
out={'status':'running','not_performance_evidence':True,'records':[]}
def save():(root/'results.json').write_text(json.dumps(out),encoding='utf-8')
def user(title):return '[[ ## title ## ]]\n'+title+'\n\nReturn a JSON object with every output field.'
try:
 for title in cases:
  messages=[{'role':'system','content':[{'type':'text','text':'Your input fields are:\n1. `title` (string)\n\nYour output fields are:\n1. `title` (string)\n\nReturn compact JSON matching the provider JSON Schema. No markdown.\n\n'+instruction}]}]
  for source,target in demos:messages.extend([{'role':'user','content':user(source)},{'role':'assistant','content':[{'type':'text','text':json.dumps({'title':target},separators=(',',':'))}]}])
  messages.append({'role':'user','content':user(title)})
  body={'model':'Qwen3-0.6B-Tasks','messages':messages,'temperature':0.,'max_tokens':128,'seed':0,'response_format':{'type':'json_schema','json_schema':{'name':'response_schema','strict':True,'schema':{'type':'object','properties':{'title':{'type':'string'}},'required':['title'],'additionalProperties':False}}}}
  req=urllib.request.Request('http://'+ip+':8080/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=120) as response:data=json.load(response)
  out['records'].append({'input':title,'request':body,'response':data,'parsed':json.loads(data['choices'][0]['message']['content'])});save()
 out['status']='completed'
except Exception as exc:out.update(status='failed',error=repr(exc))
finally:save()
'''
source='cases='+repr(cases)+'\n'+code
(root/'runner.py').write_text(source,encoding='utf-8');compile(source,'title-normalization','exec')
packed=base64.b64encode(zlib.compress(source.encode(),9)).decode();runner='import base64,zlib; exec(zlib.decompress(base64.b64decode('+repr(packed)+')))'
(root/'command.txt').write_text('qm guest exec 107 --synchronous 0 -- python3 -u -c '+shlex.quote(runner),encoding='utf-8')
print(root)
