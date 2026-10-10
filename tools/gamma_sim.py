#!/usr/bin/env python3
"""gamma_sim.py — replay MTP acceptance traces (QWENOX_SPEC_TRACE=1, tools/gamma_trace.sh)
against γ controllers offline.

  python3 tools/gamma_sim.py [logs/gtr]            # validate + controller table
  SET=x MODES=s MODEL=round python3 tools/gamma_sim.py   # extra set, sampling only, round replay

Trace line (Model::GammaCtl::report): "<tag> gamma-trace: g,a,ms g,a,ms ..."
  g = drafts proposed, a = accepted prefix, ms = wall time since previous round.

Replay model ("round replay"): a γ=7 run gives, per round i, the accepted prefix
a_i ≤ 7. A controller running γ in round i gets min(a_i, γ) accepted and commits
min(a_i, γ)+1 tokens; rounds are consumed in order, so burstiness is kept.
Round cost is c0 + c1·γ ms (least squares over all traced rounds of a mode).
Throughput = Σ commit / Σ cost over all rounds of all traces (pooled, like the
on-GPU checks). Validation: predicted pooled commit/round for fixed γ vs the
measured fixed-γ runs in the same directory (and in logs/grc if present).
"""
import glob, math, os, re, sys
from collections import defaultdict

D = sys.argv[1] if len(sys.argv) > 1 else 'logs/gtr'
tr_re = re.compile(r'gamma-trace:((?: \d+,\d+,[\d.]+)*)')
spec_re = re.compile(r'^spec(?:-sample)?: (\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s \| rounds=(\d+) commit/round=([\d.]+)')
name_re = re.compile(r'([gs])_([ox]\d+)_(\d+)(?:_s(\d+))?\.log$')  # o<offset> real text, x<n> extra set
grc_re = re.compile(r'/o(\d+)_g(\d+)\.log$')  # tools/gamma_real_check.sh (greedy, no trace)


def load(d):
    runs = []
    for f in sorted(glob.glob(os.path.join(d, '*.log'))):
        m = name_re.search(f)
        if m:
            mode, off, gam, seed = m[1], m[2], int(m[3]), m[4]
        elif grc_re.search(f):
            m = grc_re.search(f); mode, off, gam, seed = 'g', 'o' + m[1], int(m[2]), None
        else:
            continue
        txt = open(f, errors='replace').read().splitlines()
        tr = next((tr_re.search(l) for l in txt if tr_re.search(l)), None)
        sp = next((spec_re.match(l) for l in reversed(txt) if spec_re.match(l)), None)
        if not sp:
            continue
        rounds = [tuple(float(x) for x in t.split(',')) for t in tr[1].split()] if tr else []
        runs.append(dict(mode=mode, off=off, gamma=gam, seed=seed,
                         rounds=[(int(g), int(a), ms) for g, a, ms in rounds],
                         tok=int(sp[1]), sec=float(sp[2]), nr=int(sp[4])))
    return runs


def fit_cost(runs):
    xs, ys = [], []
    for r in runs:
        rr = r['rounds'][1:]  # first round includes loop setup
        if not rr:
            continue
        med = sorted(ms for _, _, ms in rr)[len(rr) // 2]
        for g, a, ms in rr:
            if ms < 3 * med:
                xs.append(g); ys.append(ms)
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-9:
        return my, 0.0
    c1 = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - c1 * mx, c1


class Fixed:
    def __init__(self, g): self.cur = g; self.name = f'fixed γ{g}'
    def update(self, g, a): pass


class Ctl:
    """Model::GammaCtl clone (defaults = as shipped in 138de39) + knobs.
    alpha: EMA floor; hyst: switch margin; kmin/gmax: γ range; r: cost slope/intercept;
    extrap: 'last' (p[top-1], as shipped) or 'geo' (continue the ratio of the last two);
    probe: every N rounds propose min(gmax, cur+probe_d) so deeper depths get observed
      (0 = off, as shipped). With probe, observed depths use their own EMA.
    confirm: the same better γ must win this many consecutive rounds before switching."""
    def __init__(self, start=4, alpha=0.1, hyst=0.02, kmin=2, gmax=7, r=0.2, prior=0.8, prior_n=4,
                 extrap='last', probe=0, probe_d=1, confirm=1, name=None):
        self.start, self.alpha, self.hyst, self.kmin, self.gmax, self.r = start, alpha, hyst, kmin, gmax, r
        self.prior, self.prior_n, self.extrap, self.probe, self.probe_d, self.confirm = \
            prior, prior_n, extrap, probe, probe_d, confirm
        self.cur = start
        self.p = [prior] * 8; self.n = [float(prior_n)] * 8; self.obs = [False] * 8
        self.k = 0; self.pend = None; self.pend_n = 0
        self.name = name or (f'α{alpha} h{hyst} min{kmin} {extrap}' + (f' probe{probe}' if probe else '')
                             + (f' conf{confirm}' if confirm > 1 else ''))

    def top(self):
        t = self.cur
        if self.probe:
            while t < 8 and self.obs[t]:
                t += 1
        return t

    def pj(self, k, top):
        if k < top:
            return self.p[k]
        if self.extrap == 'geo' and top >= 2:
            q = min(1.0, self.p[top - 1] / max(1e-3, self.p[top - 2]))
            return max(0.05, self.p[top - 1] * q ** (k - top + 1))
        return self.p[top - 1]

    def score(self, G, top):
        e = pr = 1.0
        for k in range(G):
            pr *= self.pj(k, top); e += pr
        return e / (1 + self.r * G)

    def propose(self):
        self.k += 1
        if self.probe and self.k % self.probe == 0:
            return min(self.gmax, self.cur + self.probe_d)
        return self.cur

    def update(self, g, a):
        if g < self.cur:
            return  # tail round (not in replay) — same guard as C++
        for j in range(min(a + 1, g)):
            self.n[j] += 1; self.obs[j] = True
            self.p[j] += max(self.alpha, 1 / self.n[j]) * ((1 if j < a else 0) - self.p[j])
        top = self.top(); cur = self.cur
        best, sb = cur, self.score(cur, top)
        for G in range(self.kmin, self.gmax + 1):
            s = self.score(G, top)
            if s > sb * (1 + self.hyst) or (best != cur and s > sb):
                best, sb = G, s
        if best == cur:
            self.pend, self.pend_n = None, 0
            return
        if best == self.pend:
            self.pend_n += 1
        else:
            self.pend, self.pend_n = best, 1
        if self.pend_n < self.confirm:
            return
        for j in range(cur, best):
            if top <= j:  # unobserved: seed with the extrapolation (C++: p[cur-1])
                self.p[j] = self.pj(j, top); self.n[j] = self.prior_n
        self.cur = best; self.pend, self.pend_n = None, 0


class Ctl2(Ctl):
    """Two-state variant: per-depth acceptance is tracked separately for rounds that
    start right after a full accept (F) and after a reject (R). Measured on real text
    (09-29): F .90/.85/.82 vs R .77/.70/.70 — deep γ makes full accepts rare, so the
    effective acceptance falls with γ; a single-state model misses this and drifts deep.
    E(γ) = π_F·E_F(γ) + (1−π_F)·E_R(γ), π_F = A_R / (1 − A_F + A_R), A_s = Π_{j<γ} p_s(j)."""
    def __init__(self, **kw):
        kw.setdefault('name', None)
        super().__init__(**kw)
        self.p2 = [[self.prior] * 8 for _ in range(2)]
        self.n2 = [[float(self.prior_n)] * 8 for _ in range(2)]
        self.full = 0
        if kw['name'] is None:
            self.name = '2state ' + self.name

    def es(self, s, G):
        e = pr = 1.0
        for k in range(G):
            pr *= self.p2[s][min(k, self.cur - 1)]; e += pr
        return e, pr

    def score(self, G, top=None):
        eR, aR = self.es(0, G); eF, aF = self.es(1, G)
        pi = aR / max(1e-9, 1 - aF + aR)
        return (pi * eF + (1 - pi) * eR) / (1 + self.r * G)

    def update(self, g, a):
        s = self.full; self.full = int(a >= g)
        if g < self.cur:
            return
        for j in range(min(a + 1, g)):
            self.n2[s][j] += 1
            self.p2[s][j] += max(self.alpha, 1 / self.n2[s][j]) * ((1 if j < a else 0) - self.p2[s][j])
        cur = self.cur
        best, sb = cur, self.score(cur)
        for G in range(self.kmin, self.gmax + 1):
            sc = self.score(G)
            if sc > sb * (1 + self.hyst) or (best != cur and sc > sb):
                best, sb = G, sc
        if best == cur:
            return
        for t in range(2):
            for j in range(cur, best):
                self.p2[t][j] = self.p2[t][cur - 1]; self.n2[t][j] = self.prior_n
        self.cur = best


class Markov:
    """Generator model fitted to one γ7 trace: windows of W rounds, each with its own
    two-state per-depth acceptance p[s][j] (s = previous round fully accepted), shrunk
    toward the mode-wide pooled values with K pseudo-counts. Reproduces the
    selection effect (fixed-γ commit/round from γ7 traces within ~1% of measured γ3/γ4)."""
    W, K = 40, 8.0

    def __init__(self, trace, pooled):
        self.win = []  # (token span, p[2][7])
        for i in range(0, len(trace) - 1, self.W):
            seg = trace[max(1, i):i + self.W + 1]
            prev = trace[max(0, i - 1):i + self.W]
            hit = [[0.0] * 7 for _ in range(2)]; tot = [[0.0] * 7 for _ in range(2)]
            for (g0, a0, _), (g, a, _) in zip(prev, seg):
                s = int(a0 >= g0)
                for j in range(min(a + 1, g)):
                    tot[s][j] += 1; hit[s][j] += j < a
            p = [[(hit[s][j] + self.K * pooled[s][j]) / (tot[s][j] + self.K) for j in range(7)] for s in range(2)]
            self.win.append((sum(a + 1 for _, a, _ in trace[i:i + self.W]), p))

    @staticmethod
    def pooled(traces):
        hit = [[0.0] * 7 for _ in range(2)]; tot = [[0.0] * 7 for _ in range(2)]
        for t in traces:
            for (g0, a0, _), (g, a, _) in zip(t, t[1:]):
                s = int(a0 >= g0)
                for j in range(min(a + 1, g)):
                    tot[s][j] += 1; hit[s][j] += j < a
        return [[hit[s][j] / max(1, tot[s][j]) for j in range(7)] for s in range(2)]


def sim_markov(mk_and_seed, ctl, cost):
    import random
    m, seed = mk_and_seed
    rnd = random.Random(seed)
    c0, c1 = cost
    tok = ms = rounds = 0
    hist = defaultdict(int)
    full = 0
    for span, p in m.win:
        end = tok + span
        while tok < end:
            g = ctl.cur if isinstance(ctl, Fixed) else ctl.propose()
            g = min(g, 7)
            a = 0
            while a < g and rnd.random() < p[full][a]:
                a += 1
            full = int(a == g)
            tok += a + 1; ms += c0 + c1 * g; rounds += 1; hist[g] += 1
            ctl.update(g, a)
    return tok, ms, rounds, hist


def to_positions(trace, seed=1):
    """γ7 round trace → per-position flags x[t] (1 = a draft for t matches the trunk).
    Round (7, a): a ones, then 0 at the corrected token if a < 7. After a full accept
    the bonus position was never drafted: fill with Bernoulli(p1 after a full round)."""
    import random
    rnd = random.Random(seed)
    h = n = 0
    for (g0, a0, _), (g, a, _) in zip(trace, trace[1:]):
        if a0 == g0:
            n += 1; h += a > 0
    q = h / n if n else 0.9
    x = []
    for g, a, _ in trace:
        x += [1] * a
        x.append(0 if a < g else int(rnd.random() < q))
    return x


def sim_pos(x, ctl, cost):
    c0, c1 = cost
    tok = ms = rounds = 0
    hist = defaultdict(int)
    t = 0
    while t + 8 < len(x):
        g = ctl.cur if isinstance(ctl, Fixed) else ctl.propose()
        a = 0
        while a < g and x[t + a]:
            a += 1
        tok += a + 1; ms += c0 + c1 * g; rounds += 1; hist[g] += 1
        ctl.update(g, a)
        t += a + 1
    return tok, ms, rounds, hist


def sim_round(trace, ctl, cost):
    c0, c1 = cost
    tok = ms = rounds = 0
    hist = defaultdict(int)
    for g7, a7, _ in trace:
        g = ctl.cur if isinstance(ctl, Fixed) else ctl.propose()
        g = min(g, g7)
        a = min(a7, g)
        tok += a + 1; ms += c0 + c1 * g; rounds += 1; hist[g] += 1
        ctl.update(g, a)
    return tok, ms, rounds, hist


def pooled(data, mk, cost, simf):
    T = M = R = 0
    H = defaultdict(int)
    per = []
    for d in data:
        tok, ms, r, h = simf(d, mk(), cost)
        T += tok; M += ms; R += r
        for k, v in h.items(): H[k] += v
        per.append(tok / ms * 1000)
    return T / M * 1000, T / R, sum(k * v for k, v in H.items()) / max(1, sum(H.values())), per


def candidates(start, r0):
    c = [lambda g=g: Fixed(g) for g in range(2, 8)]
    c += [lambda: Ctl(start=start, name='shipped (α.1 h.02 min2 r.2)')]
    c += [lambda: Ctl(start=start, r=r0, name=f'shipped, fitted r={r0:.3f}')]
    for kmin in (2, 3):
        for alpha in (0.1, 0.05, 0.03):
            for hyst in (0.02, 0.05):
                for r in (0.2, r0, 0.3):
                    c.append(lambda a=alpha, h=hyst, r=r, k=kmin:
                             Ctl2(start=start, alpha=a, hyst=h, r=r, kmin=k,
                                  name=f'2state min{k} α{a} h{h} r{r:.3f}'))
    if os.environ.get('GRID') != 'full':
        for alpha in (0.1, 0.05):
            for hyst in (0.02, 0.05):
                c.append(lambda a=alpha, h=hyst: Ctl(start=start, alpha=a, hyst=h, r=r0))
        return c
    for alpha in (0.1, 0.05, 0.03):
        for hyst in (0.02, 0.05, 0.1):
            for kmin in (2, 3):
                for extrap in ('last', 'geo'):
                    for probe in (0, 8):
                        for conf in (1, 3):
                            c.append(lambda a=alpha, h=hyst, k=kmin, e=extrap, p=probe, cf=conf:
                                     Ctl(start=start, alpha=a, hyst=h, kmin=k, r=r0, extrap=e, probe=p,
                                         confirm=cf))
    return c


def main():
    sets = os.environ.get('SET', 'o')  # o = real long text, x = extra set (tools/gamma_trace.sh XSET), ox = both
    runs = [r for r in load(D) if r['off'][0] in sets]
    if not any(r['rounds'] for r in runs):
        print(f'no traces in {D}'); sys.exit(1)
    model = os.environ.get('MODEL', 'pos')
    grc = [r for r in load('logs/grc') if 'o' in sets]  # measured fixed γ (greedy, gamma_real_check)
    modes = os.environ.get('MODES', 'gs')  # MODES=s: sampling only
    for mode, label in (('g', 'greedy'), ('s', 'sampling')):
        if mode not in modes:
            continue
        R = [r for r in runs if r['mode'] == mode and r['rounds']]
        if not R:
            continue
        cost = fit_cost(R)
        t7 = [r['rounds'] for r in R if r['gamma'] == 7]
        xs = [to_positions(t, i + 1) for i, t in enumerate(t7)]
        pp = Markov.pooled(t7)
        mks = [(Markov(t, pp), s) for t in t7 for s in range(int(os.environ.get('NSEED', '8')))]
        print('  two-state p_j after full accept: ' + ' '.join(f'{x:.3f}' for x in pp[1])
              + ' | after reject: ' + ' '.join(f'{x:.3f}' for x in pp[0]))
        print(f'\n=== {label}: {len(t7)} γ7 traces, cost ≈ {cost[0]:.1f} + {cost[1]:.2f}·γ ms '
              f'(r = {cost[1] / cost[0]:.3f}), model={model}')
        hit = [0] * 7; tot = [0] * 7
        for t in t7:
            for g, a, _ in t:
                for j in range(min(a + 1, g)):
                    tot[j] += 1; hit[j] += j < a
        print('  γ7 conditional p_j: ' + ' '.join(f'{hit[j] / tot[j]:.3f}' for j in range(7) if tot[j]))
        meas = defaultdict(lambda: [0, 0.0, 0])
        for r in [x for x in runs if x['mode'] == mode] + (grc if mode == 'g' else []):
            if r['gamma'] == 0:
                continue
            m = meas[r['gamma']]; m[0] += r['tok']; m[1] += r['sec']; m[2] += r['nr']
        for g in sorted(meas):
            m = meas[g]
            pr = pooled(xs, lambda: Fixed(g), cost, sim_pos)
            rr = pooled(t7, lambda: Fixed(g), cost, sim_round)
            mm = pooled(mks, lambda: Fixed(g), cost, sim_markov)
            print(f'  validate fixed γ{g}: measured commit/round {m[0] / m[2]:.2f} tok/s {m[0] / m[1]:.1f}'
                  f' | pos-model {pr[1]:.2f} {pr[0]:.1f} | round-model {rr[1]:.2f} {rr[0]:.1f}'
                  f' | markov {mm[1]:.2f} {mm[0]:.1f}')
        ad = [r for r in runs if r['mode'] == mode and r['gamma'] == 0]
        if ad:
            T = sum(r['tok'] for r in ad); S = sum(r['sec'] for r in ad); N = sum(r['nr'] for r in ad)
            print(f'  measured shipped adaptive: {T / S:.1f} tok/s commit/round {T / N:.2f} ({len(ad)} runs)')
        start = 4 if mode == 'g' else 3
        r0 = cost[1] / cost[0]
        data, simf = {'pos': (xs, sim_pos), 'round': (t7, sim_round), 'markov': (mks, sim_markov)}[model]
        rows = []
        for mk in candidates(start, r0):
            tps, cpr, ag, per = pooled(data, mk, cost, simf)
            rows.append((tps, cpr, ag, mk().name, per))
        best_fixed = max((r for r in rows if r[3].startswith('fixed')), key=lambda x: x[0])
        bf = best_fixed[4]
        fx = ' '.join(f'{r[3][7:]}={r[0]:.2f}' for r in rows if r[3].startswith('fixed'))
        print(f'  fixed: {fx}')
        rows.sort(key=lambda x: -x[0])
        print('  top controllers (tok/s, ÷best fixed, commit/round, avg γ, worst trace ÷best fixed):')
        for tps, cpr, ag, nm, per in rows[:20] + [x for x in rows if x[3].startswith('shipped')]:
            worst = min(p / b for p, b in zip(per, bf))
            print(f'   {tps:6.2f} {tps / best_fixed[0]:.3f}  {cpr:.2f}  γ̄{ag:.2f}  worst {worst:.3f}  {nm}')


if __name__ == '__main__':
    main()
