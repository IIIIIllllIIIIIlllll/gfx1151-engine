#!/usr/bin/env python3
"""Reproduce the numbered-case index-90 divergence; compare serial/NGRAM/MTP."""
import socket, sys, json
sys.path.insert(0, 'tools')
from qwentok import Tokenizer

tk = Tokenizer()
sock = socket.create_connection(('127.0.0.1', 8732), timeout=120)
f = sock.makefile('rb')
req = 0

def gen(ids, count, drafter):
    global req
    req += 1
    fields = ['GEN', req, count, 0, len(ids), *ids, drafter]
    sock.sendall((' '.join(map(str, fields)) + '\n').encode())
    out = []
    while True:
        line = f.readline().decode().strip()
        if not line:
            raise RuntimeError('engine disconnected')
        v = line.split()
        if v[0] == 'T':
            out.append(int(v[2]))
        elif v[0] == 'D':
            print(f'req{req} drafter={drafter} ntok={len(out)} :: {line}', flush=True)
            return out

numbered = tk.encode(''.join(f'{i}. The quick brown fox jumps over the lazy dog and then runs back to the green field beside the river. It pauses beneath the tall old tree to watch the leaves drifting slowly in the warm evening breeze.\n' for i in range(1, 9)))
print('prompt tokens:', len(numbered), flush=True)

gold = gen(numbered, 128, 0)
ng = gen(numbered, 128, 'NGRAM')
mtp = gen(numbered, 128, 1)

def firstdiff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))

print('NGRAM vs serial first diff:', firstdiff(gold, ng))
print('MTP   vs serial first diff:', firstdiff(gold, mtp))
print('NGRAM vs MTP    first diff:', firstdiff(ng, mtp))
i = firstdiff(gold, ng)
if i < len(gold):
    print('at', i, 'gold', gold[i], 'ngram', ng[i])
    print('gold around:', gold[max(0, i-6):i+6])
    print('ngrm around:', ng[max(0, i-6):i+6])
    print('gold text:', tk.decode(gold[max(0, i-6):i+6]))
    print('ngrm text:', tk.decode(ng[max(0, i-6):i+6]))
json.dump({'gold': gold, 'ngram': ng, 'mtp': mtp}, open('logs/idx90-repro.json', 'w'))
