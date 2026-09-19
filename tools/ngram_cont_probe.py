import json, socket
from pathlib import Path
from qwentok import Tokenizer
tk=Tokenizer()
sock=socket.create_connection(('127.0.0.1',8732),timeout=55)
f=sock.makefile('rb'); records=[]
def gen(prompt,n,drafter,eos=(),suffix=''):
    req=len(records)+1
    sock.sendall((' '.join(map(str,['GEN',req,n,len(eos),*eos,len(prompt),*prompt,drafter]))+' '+suffix+'\n').encode())
    out=[]
    while True:
        v=f.readline().decode().split()
        if not v: raise RuntimeError('disconnected')
        if v[0]=='T':out.append(int(v[2]))
        if v[0]=='D':
            records.append(dict(req=req,drafter=drafter,tokens=out,done=v))
            print(req,drafter,' '.join(v),out,flush=True)
            return out,v
prompt=tk.encode(' '.join(['alpha beta gamma']*80))
for count in [11,12]:
    for d in [0,'NGRAM']:
        out,_=gen(prompt,count,d)
        cont=prompt+out+[198]
        live,done=gen(cont,16,d)
        cold,_=gen(cont,16,0)
        print('CONT_COMPARE',count,d,live==cold,'cached',done[10],flush=True)
sample=tk.encode('<|im_start|>user\nExplain why rainbows have different colors.\n<|im_end|>\n<|im_start|>assistant\n')
x,_=gen(sample,16,0,suffix='SAMPLE 0.8 20 0.95 0 777')
y,d=gen(sample,16,'NGRAM',suffix='SAMPLE 0.8 20 0.95 0 777')
assert x==y and d[7]=='0'
print('SAMPLE PASS')
Path('logs/ngram-cont-probe.json').write_text(json.dumps(records,indent=2))
