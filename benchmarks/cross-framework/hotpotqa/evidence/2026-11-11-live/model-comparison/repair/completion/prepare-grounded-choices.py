import base64, copy, json, pathlib, shlex, zlib

base = pathlib.Path(r'C:\Users\georg\Documents\dsrs-build\fewshot-repair')
root = base / 'grounded-choices'
root.mkdir(exist_ok=True)
refs = json.loads((base / 'context-references.json').read_text(encoding='utf-8'))
native = json.loads((base / 'native-requests.json').read_text(encoding='utf-8'))
cases = []
for i, row in enumerate(refs):
    request = copy.deepcopy(native[i * 3]['request'])
    context = request['messages'][-1]['content'].split('[[ ## context ## ]]\n', 1)[1].rsplit('\n\nReturn a JSON object', 1)[0]
    cases.append(dict(id=row['id'], question=row['question'], context=context, request=request))

code = r'''
import copy, hashlib, json, pathlib, re, subprocess, urllib.request
root = pathlib.Path('/tmp/dsrs-grounded-choices-2026-11-11')
root.mkdir(exist_ok=False)
ip = subprocess.check_output(['docker','inspect','-f','{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}','localai-tasks-1'], text=True).strip()
out = {'not_performance_evidence':True, 'status':'running', 'records':[]}
def save():
    (root / 'results.json').write_text(json.dumps(out), encoding='utf-8')
def choices(question, context):
    if question.lower().split()[0] in ('is','are','was','were','do','does','did','can','has','have'):
        return ['yes','no']
    excluded = re.search(r'other than (.+?),', question, re.I)
    spans = set()
    for line in context.splitlines():
        for clause in re.split(r'[.!?;]', line):
            words = list(re.finditer(r'\b[\w-]+\b', clause))
            for start in range(len(words)):
                for end in range(start, min(start + 8, len(words))):
                    phrase = clause[words[start].start():words[end].end()]
                    if excluded and re.search(r'\b'+re.escape(excluded[1])+r'\b', phrase, re.I):
                        continue
                    spans.add(phrase)
    return sorted(spans)
try:
    for condition in ['choices', 'choices-precise']:
        for case in cases:
            body = copy.deepcopy(case['request']); body['seed']=0
            allowed = choices(case['question'], case['context'])
            body['response_format']['json_schema']['schema']['properties']['answer']['enum'] = allowed
            if condition == 'choices-precise':
                body['messages'][0]['content'][0]['text'] += '\n\nCopy the shortest precise name or title that answers the question from the context. Prefer a specifically named title over a broad job category. Omit explanations and location qualifiers. Follow any exclusions in the question. For yes/no questions, answer only yes or no.'
            req = urllib.request.Request('http://'+ip+':8080/v1/chat/completions', json.dumps(body).encode(), {'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=120) as response: data=json.load(response)
            out['records'].append({'condition':condition,'id':case['id'],'request':body,'response':data,'parsed':json.loads(data['choices'][0]['message']['content'])})
            save()
    out['status']='completed'
except Exception as exc:
    out.update(status='failed',error=repr(exc))
finally:
    save()
'''
source = 'cases=' + repr(cases) + '\n' + code
(root / 'runner.py').write_text(source, encoding='utf-8')
compile(source, 'grounded-choices', 'exec')
packed = base64.b64encode(zlib.compress(source.encode(), 9)).decode()
runner = 'import base64,zlib; exec(zlib.decompress(base64.b64decode(' + repr(packed) + ')))'
(root / 'command.txt').write_text('qm guest exec 107 --synchronous 0 -- python3 -u -c ' + shlex.quote(runner), encoding='utf-8')
(root / 'references.json').write_text(json.dumps(refs, indent=2), encoding='utf-8')
print(root)
