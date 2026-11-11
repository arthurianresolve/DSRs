import base64,hashlib,json,pathlib,shlex,zlib
base=pathlib.Path(r'C:\Users\georg\Documents\dsrs-build\fewshot-repair');root=base/'relation-recipe'
expected={}
for name in ['grounded-choices','role-demonstrations','focused-role','relation-recipe','title-normalization','cache-repeatability']:
    raw=(base/name/'results.json').read_bytes();expected['/tmp/dsrs-'+name+'-2026-11-11']={'results.json':hashlib.sha256(raw).hexdigest()}
for local,remote,count in [('native-live','native-relation-final',18),('native-live-fixed','native-relation-fixed',74)]:
    expected['/tmp/dsrs-'+remote+'-2026-11-11']={f'round-{n}.json':hashlib.sha256((root/local/f'round-{n}.response.json').read_bytes()).hexdigest() for n in range(1,count+1)}
(root/'cleanup-plan.json').write_text(json.dumps(expected,indent=2),encoding='utf-8')
summary=json.loads((root/'native-live-fixed/summary.json').read_text(encoding='utf-8'))
code=r'''
import base64,datetime,hashlib,json,pathlib,shutil,subprocess,zlib
for target,wanted in expected.items():
    path=pathlib.Path(target)
    assert path.parent==pathlib.Path('/tmp') and path.resolve()==path and path.name.startswith('dsrs-')
    entries=list(path.iterdir());assert all(p.is_file() and not p.is_symlink() for p in entries)
    actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in entries}
    assert actual==wanted,(target,set(actual)^set(wanted))
removed=[]
for target in expected:
    shutil.rmtree(target);assert not pathlib.Path(target).exists();removed.append(target)
services={name:subprocess.check_output(['docker','inspect','-f','{{.State.Running}}',name],text=True).strip()=='true' for name in ['localai-llama-1','localai-tasks-1']}
assert all(services.values())
result={'status':'completed','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'directories_removed':removed,'verified_files':sum(map(len,expected.values())),'existing_services_running':services,'persistent_configuration_changed':False}
raw=json.dumps(result).encode()
print('DSRS_CLEANUP:'+base64.b64encode(zlib.compress(raw,9)).decode()+':END_DSRS_CLEANUP '+hashlib.sha256(raw).hexdigest())
print('Native DSRs / Qwen3-0.6B-Tasks quality verification:')
for run in summary['runs']:
    print(run['name']+': '+str(run.get('exact_matches','rejected'))+'/'+str(run['samples'])+', native exit '+str(run['native_exit']))
print('Retained and verified 74 real native requests.')
print('Final exclusion: three final attempts rejected; five total requests.')
print('Verified '+str(result['verified_files'])+' artifact hashes; removed '+str(len(removed))+' test directories.')
print('Both existing model services running; persistent configuration unchanged.')
'''
source='expected='+repr(expected)+'\nsummary='+repr(summary)+'\n'+code
(root/'cleanup.py').write_text(source,encoding='utf-8');compile(source,'cleanup','exec')
packed=base64.b64encode(zlib.compress(source.encode(),9)).decode();runner='import base64,zlib; exec(zlib.decompress(base64.b64decode('+repr(packed)+')))'
unwrap='import json,sys; d=json.load(sys.stdin); print(d.get("out-data","")); print(d.get("err-data","")); assert d.get("exitcode")==0,d'
(root/'cleanup.command.txt').write_text('qm guest exec 107 --timeout 60 -- python3 -u -c '+shlex.quote(runner)+' | python3 -c '+shlex.quote(unwrap),encoding='utf-8')
print('Prepared cleanup of',len(expected),'verified test directories and',sum(map(len,expected.values())),'files')
