"""Run the existing cmp4 and five oracle text snapshots on the loaded engine."""
import socket,json
from pathlib import Path
from qwentok import Tokenizer
tk=Tokenizer(); sock=socket.create_connection(('127.0.0.1',8732),timeout=55)
f=sock.makefile('rb'); results=[]
def run(prompt, n):
    req=len(results)+1
    sock.sendall((' '.join(map(str,['GEN',req,n,0,len(prompt),*prompt,0]))+'\n').encode())
    out=[]
    while True:
        v=f.readline().decode().split()
        if not v: raise RuntimeError('engine disconnected')
        if v[0]=='T': out.append(int(v[2]))
        if v[0]=='D':
            assert v[2]=='length' and len(out)==n,v
            return tk.decode(out),' '.join(v)
fixtures=[]
for prompt,gold in json.loads(Path('tools/server_greedy24.json').read_text()).items():
    fixtures.append((f'cmp4-{len(fixtures)+1}',tk.encode(prompt),24,gold))
for n in [2048,2051,2052,8192,32768]:
    j=json.loads((Path(__file__).resolve().parent.parent / 'data/qsa-oracle'/f'{n}.json').read_text())
    fixtures.append((f'oracle-{n}',j['prompt_ids'],j['gen'],j['response']['choices'][0]['text']))
for name,ids,n,gold in fixtures:
    text,done=run(ids,n)
    record=dict(name=name,match=text==gold,done=done,actual=text,expected=gold)
    results.append(record)
    Path('logs/ngram-snapshots.json').write_text(json.dumps(results,indent=2))
    print(name,'MATCH' if text==gold else 'DIFF',done,flush=True)
print('snapshot matches',sum(r['match'] for r in results),'/',len(results),flush=True)
