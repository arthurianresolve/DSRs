import hashlib,json,pathlib,struct,subprocess
name='localai-tasks-1'
info=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
args=info['Args']; model=args[args.index('-m')+1]
mount=next(m for m in info['Mounts'] if model.startswith(m['Destination'].rstrip('/')+'/'))
path=pathlib.Path(mount['Source'])/model[len(mount['Destination']):].lstrip('/')
f=path.open('rb')
def number(fmt):
    size=struct.calcsize('<'+fmt); b=f.read(size); assert len(b)==size; return struct.unpack('<'+fmt,b)[0]
def string():
    n=number('Q'); assert n<100_000_000; return f.read(n).decode('utf-8')
formats={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def value(t,keep):
    if t in formats:
        v=number(formats[t]); return v if keep else None
    if t==8:
        if keep: return string()
        f.seek(number('Q'),1); return None
    assert t==9,t
    element=number('I'); count=number('Q')
    if element in formats: f.seek(struct.calcsize('<'+formats[element])*count,1)
    else:
        for _ in range(count): value(element,False)
    return None
assert f.read(4)==b'GGUF'
version=number('I'); assert version in (2,3)
tensors=number('Q'); count=number('Q'); metadata={}
for _ in range(count):
    key=string(); t=number('I'); keep=key.startswith('general.') and t!=9
    v=value(t,keep)
    if keep: metadata[key]=v
f.close()
with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
print(json.dumps({'version':version,'tensors':tensors,'metadata':metadata,'model_sha256':digest},indent=2))
