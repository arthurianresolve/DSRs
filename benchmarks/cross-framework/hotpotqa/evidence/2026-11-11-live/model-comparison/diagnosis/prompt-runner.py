"""Guest-side quality experiment; no latency claims. bundle contains no gold answers."""
import copy,json,pathlib,subprocess,time,urllib.request
root=pathlib.Path('/tmp/dsrs-quality-diagnosis-2026-11-11')
root.mkdir(exist_ok=False)
ip=subprocess.check_output(['docker','inspect','-f','{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}','localai-tasks-1'],text=True).strip()
out={'model':'Qwen3-0.6B-Tasks','status':'running','not_performance_evidence':True,'records':[]}
def save():
    (root/'results.json').write_text(json.dumps(out),encoding='utf-8')
save()
try:
 for variant in ('baseline','independent_check'):
  for example in bundle['examples']:
   for framework in ('dsrs','dspy'):
    values={'question':example['question'],'context':example['context']}
    for stage in range(3):
     template=bundle['templates'][framework][stage]
     body=copy.deepcopy(template['body'])
     body['model']='Qwen3-0.6B-Tasks'; body['seed']=0
     if variant!='baseline':
      message=body['messages'][0]
      if isinstance(message['content'],list):
       message['content'][0]['text']=message['content'][0]['text'].replace(template['instruction'],bundle['instructions'][stage])
      else: message['content']=message['content'].replace(template['instruction'],bundle['instructions'][stage])
     names=['question','context']+(['answer'] if stage==1 else ['draft','critique'] if stage==2 else [])
     body['messages'][-1]['content']='\n\n'.join('[[ ## '+n+' ## ]]\n'+values[n] for n in names)+'\n\n'+template['suffix']
     req=urllib.request.Request('http://'+ip+':8080/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
     with urllib.request.urlopen(req,timeout=120) as response: data=json.load(response)
     parsed=json.loads(data['choices'][0]['message']['content'])
     out['records'].append({'variant':variant,'id':example['id'],'framework':framework,'stage':stage,'request':body,'response':data,'parsed':parsed})
     if stage==0: values.update(answer=parsed['answer'],draft=parsed['answer'])
     if stage==1: values['critique']=parsed['critique']
     save()
 out['status']='completed'
except Exception as e:
 out['status']='failed'; out['error']=repr(e)
finally: save()
