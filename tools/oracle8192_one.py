#!/usr/bin/env python3
"""One-shot oracle-8192 check against the engine on 8732."""
import json, socket, sys
sys.path.insert(0, 'tools')
from qwentok import Tokenizer
tk = Tokenizer()
j = json.loads(open('data/qsa-oracle/8192.json').read())
ids, n = j['prompt_ids'], j['gen']
gold = j['response']['choices'][0]['text']
sock = socket.create_connection(('127.0.0.1', 8732), timeout=120)
f = sock.makefile('rb')
sock.sendall((' '.join(map(str, ['GEN', 1, n, 0, len(ids), *ids, 0])) + '\n').encode())
out = []
while True:
    v = f.readline().decode().split()
    if not v:
        raise RuntimeError('disconnected')
    if v[0] == 'T':
        out.append(int(v[2]))
    if v[0] == 'D':
        break
text = tk.decode(out)
print('MATCH' if text == gold else 'DIFF')
print('actual  :', repr(text[:120]))
print('expected:', repr(gold[:120]))
