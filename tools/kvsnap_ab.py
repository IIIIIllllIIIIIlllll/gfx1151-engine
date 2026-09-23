"""kvsnap A/B: an SSD-snapshot restore must continue bit-exactly like the live state.

Usage (repo root, one engine on 127.0.0.1:8732, kvsnap ON, restart between steps):
  python3 tools/kvsnap_ab.py save [--n 8192|32768]   # empty snapshot dir
  #   -> GEN P (16 tok): engine saves a snapshot of the end-of-turn state
  #   -> GEN Q = P+out+tail on the SAME engine (live continuation) = gold 'live'
  # restart the engine (same GDEC_KVSNAP_DIR), optionally with GDEC_KV_PAGED=1/2
  python3 tools/kvsnap_ab.py run --tag restore-off   # log: "kvsnap: restored"
  python3 tools/kvsnap_ab.py run --tag restore-p2
  python3 tools/kvsnap_ab.py compare                 # every tag vs 'live'

Gold is the live continuation, not a fresh full prefill: restore reproduces
the state bytes of (prefill P + decode), and a fresh one-shot prefill of Q may
legitimately differ in the last bits. The engine must be restarted before each
`run` (otherwise the live/RAM-checkpoint path is used instead of the restore).
State lives in logs/kvsnap-ab.json.
"""
import argparse, json, socket, sys
from pathlib import Path
from qwentok import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / 'logs' / 'kvsnap-ab.json'
GEN_SAVE, GEN_RUN = 16, 32


def gen(ids, n, req=1):
    sock = socket.create_connection(('127.0.0.1', 8732), timeout=600)
    f = sock.makefile('rb')
    sock.sendall((' '.join(map(str, ['GEN', req, n, 0, len(ids), *ids, 0])) + '\n').encode())
    out = []
    while True:
        v = f.readline().decode().split()
        if not v:
            raise RuntimeError('engine disconnected')
        if v[0] == 'T':
            out.append(int(v[2]))
        if v[0] == 'D':
            sock.close()
            if v[2] != 'length' or len(out) != n:
                raise RuntimeError(f'unexpected finish: {v} after {len(out)} tokens')
            return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['save', 'run', 'compare'])
    ap.add_argument('--n', type=int, default=8192, help='oracle prompt: 8192 or 32768')
    ap.add_argument('--tag', default='restore')
    ap.add_argument('--state', default=str(STATE), help='state json (one per A/B series)')
    a = ap.parse_args()
    globals()['STATE'] = Path(a.state)
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    if a.cmd == 'save':
        j = json.loads((ROOT / 'data/qsa-oracle' / f'{a.n}.json').read_text())
        p = j['prompt_ids']
        out = gen(p, GEN_SAVE, 1)
        q = p + out + Tokenizer().encode('\n\nContinue the text above.\n')
        live = gen(q, GEN_RUN, 2)
        st = dict(n=a.n, Q=q, runs={'live': live})
        STATE.parent.mkdir(exist_ok=True)
        STATE.write_text(json.dumps(st))
        print(f'|P|={len(p)} |Q|={len(q)} live={live}', flush=True)
        print('engine log should show "kvsnap: saved" twice; now restart the engine', flush=True)
    elif a.cmd == 'run':
        if 'Q' not in st:
            sys.exit('run `save` first')
        st['runs'][a.tag] = gen(st['Q'], GEN_RUN, 1)
        STATE.write_text(json.dumps(st))
        print(f'{a.tag}: {st["runs"][a.tag]}', flush=True)
        print('confirm the engine log shows "kvsnap: restored" for this request', flush=True)
    else:
        runs = st.get('runs', {})
        if 'live' not in runs:
            sys.exit('run `save` first')
        gold, bad = runs['live'], 0
        for tag, toks in runs.items():
            if tag == 'live':
                continue
            first = next((i for i, (x, y) in enumerate(zip(toks, gold)) if x != y), None)
            bad += toks != gold
            print(tag, 'MATCH' if toks == gold else f'DIFF (first mismatch at token {first})', flush=True)
        print('kvsnap_ab', 'ALL MATCH' if bad == 0 else f'{bad} DIFF', flush=True)
        sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
