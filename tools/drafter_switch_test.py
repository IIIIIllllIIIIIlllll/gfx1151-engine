"""Cold requests switching between serial, MTP and ngram; not a hybrid drafter."""
import json
import socket
from pathlib import Path
from qwentok import Tokenizer

tk = Tokenizer()
ids = tk.encode(' '.join(['alpha beta gamma'] * 80))
records = []
sock = socket.create_connection(('127.0.0.1',8732), timeout=55)
stream = sock.makefile('rb')
for drafter in [0, 1, 'NGRAM', 1, 'NGRAM']:
    req = len(records) + 1
    fields = ['GEN',req,64,0,len(ids),*ids,drafter]
    sock.sendall((' '.join(map(str,fields))+'\n').encode())
    tokens = []
    while True:
        v = stream.readline().decode().split()
        if not v:
            raise RuntimeError('disconnected')
        if v[0] == 'T':
            tokens.append(int(v[2]))
        if v[0] == 'D':
            assert v[2] == 'length' and len(tokens) == 64, v
            records.append(dict(drafter=drafter,tokens=tokens,done=' '.join(v)))
            Path('logs/ngram-drafter-switch.json').write_text(json.dumps(records,indent=2))
            assert tokens == records[0]['tokens'], ('token mismatch',drafter)
            assert int(v[7]) == (3 if drafter == 'NGRAM' else drafter), v
            print('PASS drafter',drafter,' '.join(v),flush=True)
            break
