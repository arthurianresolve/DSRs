import collections,hashlib,json,pathlib
root=pathlib.Path(__file__).resolve().parent
refs={e['id']:e for e in json.loads((root/'references.json').read_text())}
def norm(text):
    text=''.join(c for c in text.lower() if c.isalnum() or c.isspace())
    return ' '.join(w for w in text.split() if w not in {'a','an','the'})
baseline=json.loads((root.parent/'tasks-live/requests.json').read_text())
cases=list(refs)
reports=[]
for name in ('prompt-results.json','sampling-results.json','thinking-results.json',
             'grounded-reflection-results.json','concise-reflection-results.json',
             'full-prompt-results.json','model-control-results.json',
             'checked-reflection-results.json','checked-small-results.json'):
    p=root/name
    if not p.exists(): continue
    data=json.loads(p.read_text(encoding='utf-8'))
    assert data['status']=='completed'
    groups=collections.defaultdict(list)
    for row in data['records']:
        body=row['request']; example=refs[row['id']]
        user=body['messages'][-1]['content']
        assert example['question'] in user and example['context'] in user
        assert body['model']==('Qwen3-32B-obliterated-Q4_K_M' if name in ('model-control-results.json','checked-reflection-results.json') else 'Qwen3-0.6B-Tasks')
        assert row['response']['model']==body['model']
        assert row['response']['choices'][0]['finish_reason']=='stop'
        if 'parsed' in row:
            assert json.loads(row['response']['choices'][0]['message']['content'])==row['parsed']
        if row['stage']:
            prior={r['stage']:r['parsed'] for r in data['records']
                   if r['variant']==row['variant'] and r['framework']==row['framework'] and r['id']==row['id'] and r['stage']<row['stage']}
            field='answer' if row['stage']==1 else 'draft'
            prefix=field.capitalize()+':\n' if row['variant']=='concise' else '[[ ## '+field+' ## ]]\n'
            assert prefix+prior[0]['answer']+'\n\n' in user
            if row['stage']==2:
                prefix='Critique:\n' if row['variant']=='concise' else '[[ ## critique ## ]]\n'
                assert prefix+prior[1]['critique']+'\n\n' in user
        if row['variant']=='baseline' and not row['id'].startswith('synthetic-'):
            old=next(r for r in baseline if r['framework']==row['framework'] and r['round']==cases.index(row['id'])*3+row['stage']+1)
            candidate=dict(body); candidate.pop('seed')
            assert candidate==old['request']
            assert row['response']['choices'][0]['message']['content']==old['response']['choices'][0]['message']['content']
        if row['stage'] in (0,2) and 'parsed' in row:
            key=(row['variant'],row['framework'],'additional' if row['id'].startswith('synthetic-') else 'original',row['stage'])
            groups[key].append({'id':row['id'],'answer':row['parsed']['answer'],'expected':example['answer'],'exact_match':norm(row['parsed']['answer'])==norm(example['answer'])})
    summary={'file':name,'status':data['status'],'requests':len(data['records']),
        'stop_reasons':dict(collections.Counter(r['response']['choices'][0]['finish_reason'] for r in data['records'])),
        'scores':[{'variant':key[0],'framework':key[1],'cases':key[2],'stage':'draft' if key[3]==0 else 'final',
                   'matches':sum(s['exact_match'] for s in rows),'total':len(rows),'answers':rows} for key,rows in groups.items()]}
    reports.append(summary)
    print(name,data['status'],len(data['records']),summary['stop_reasons'])
    for s in summary['scores']:
        if s['stage']=='final': print(s['variant'],s['framework'],s['cases'],str(s['matches'])+'/'+str(s['total']))
assert reports==json.loads((root/'analysis.json').read_text(encoding='utf-8'))
print('Baseline request bodies and all 18 model response texts reproduce the native run exactly; fixture inputs verified.')
