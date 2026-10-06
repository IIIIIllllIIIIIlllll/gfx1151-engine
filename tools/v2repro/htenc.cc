// htenc.cc — hgn v2 "ht" (dtype 16) encoder: rotated-domain f32 targets ->
// 4-bit hashed-trellis code stream in the official tile layout (HGN-V2.md).
//
//   value(L,e) = LUT[ (V >> (28-4e)) & 0xffff ],
//   V = u32(word L) | u32(word L-1 mod 32) << 32          (per 128B block)
//
// The e=0,1,2 windows of a word read nibbles 0..2 of the PREVIOUS word, i.e.
// up to 15 positions back in the block's nibble stream — beyond any exact
// Viterbi state. Beam search over the 256-nibble ring, each entry carrying
// the last 16 nibbles (u64) so EVERY window is exactly evaluable when its
// last nibble lands. Ring consistency is enforced structurally: each sweep
// starts at a rotated offset p0, seeds its 16-nibble history from the
// current words at p0-16..p0-1, and FORCES the last 16 stream nibbles to
// those same values (they are the same ring positions). Beam cost is then
// the exact true cost. After each improving sweep and at the end, a
// coordinate-descent polish runs: every nibble feeds exactly 4 windows, so
// all 16 values are tried per position and the best is kept.
//
// Usage: htenc TARGET.f32 O K OUT.codes [BEAM=128] [SWEEPS=6]
// ht_lut.f32 (65536 f32) must sit next to the executable (written by
// v2_encode.py from the numpy reference so both sides round identically).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

static float LUT[65536];

static void load_lut(const char* self) {
  char path[4096];
  std::snprintf(path, sizeof path, "%s", self);
  char* sl = std::strrchr(path, '/');
  if (sl) sl[1] = 0; else path[0] = 0;
  std::strcat(path, "ht_lut.f32");
  std::ifstream f(path, std::ios::binary);
  if (!f || !f.read((char*)LUT, sizeof LUT)) {
    std::fprintf(stderr, "htenc: cannot read %s\n", path);
    std::exit(1);
  }
}

static inline float lut(uint32_t w16) { return LUT[w16 & 0xffffu]; }

// nibble m positions back (0 = most recent); recent nibble at low bits
static inline uint32_t hb(uint64_t hist, int m) {
  return (uint32_t)(hist >> (4 * m)) & 15u;
}
// last-4 window, oldest at low bits (== w16 bit order)
static inline uint32_t hw4(uint64_t h) {
  return (uint32_t)(((h >> 12) & 15u) | (((h >> 8) & 15u) << 4) |
                    (((h >> 4) & 15u) << 8) | ((h & 15u) << 12));
}

// coordinate-descent polish on one 256-nibble block (words Wd, targets tgt).
// Returns the block's exact squared error after polishing.
static double cd_polish(uint32_t* Wd, const float (*tgt)[8], int max_pass) {
  uint8_t S[256];
  for (int t = 0; t < 256; t++)
    S[t] = (Wd[t >> 3] >> (4 * (t & 7))) & 15u;
  uint16_t wv[32][8];
  float verr[32][8];
  double tot = 0.0;
  for (int L = 0; L < 32; L++) {
    const int pL = (L + 31) & 31;
    for (int e = 0; e < 8; e++) {
      uint32_t w = 0;
      for (int k = 0; k < 4; k++) {
        int pos;
        if (e >= 3) pos = 8 * L + 7 - e + k;
        else if (e == 2) pos = k < 3 ? 8 * L + 5 + k : 8 * pL;
        else if (e == 1) pos = k < 2 ? 8 * L + 6 + k : 8 * pL + (k - 2);
        else pos = k == 0 ? 8 * L + 7 : 8 * pL + (k - 1);
        w |= (uint32_t)S[pos] << (4 * k);
      }
      wv[L][e] = (uint16_t)w;
      const float d = lut(w) - tgt[L][e];
      verr[L][e] = d * d;
      tot += verr[L][e];
    }
  }
  for (int pass = 0; pass < max_pass; pass++) {
    double gain = 0.0;
    for (int p = 0; p < 256; p++) {
      const int L = p >> 3, j = p & 7;
      const int Ln = (L + 1) & 31;
      int8_t av[4][3];
      int na = 0;
      for (int e = 7 - j < 3 ? 3 : 7 - j; e <= 10 - j && e <= 7; e++) {
        av[na][0] = (int8_t)L; av[na][1] = (int8_t)e; av[na][2] = (int8_t)(j - 7 + e);
        na++;
      }
      if (j == 0) {
        av[na][0]=Ln; av[na][1]=2; av[na][2]=3; na++;
        av[na][0]=Ln; av[na][1]=1; av[na][2]=2; na++;
        av[na][0]=Ln; av[na][1]=0; av[na][2]=1; na++;
      } else if (j == 1) {
        av[na][0]=Ln; av[na][1]=1; av[na][2]=3; na++;
        av[na][0]=Ln; av[na][1]=0; av[na][2]=2; na++;
      } else if (j == 2) {
        av[na][0]=Ln; av[na][1]=0; av[na][2]=3; na++;
      } else if (j == 5) {
        av[na][0]=L; av[na][1]=2; av[na][2]=0; na++;
      } else if (j == 6) {
        av[na][0]=L; av[na][1]=2; av[na][2]=1; na++;
        av[na][0]=L; av[na][1]=1; av[na][2]=0; na++;
      } else if (j == 7) {
        av[na][0]=L; av[na][1]=2; av[na][2]=2; na++;
        av[na][0]=L; av[na][1]=1; av[na][2]=1; na++;
        av[na][0]=L; av[na][1]=0; av[na][2]=0; na++;
      }
      float cur = 0.f;
      for (int i = 0; i < na; i++) cur += verr[av[i][0]][av[i][1]];
      float bc = cur;
      uint32_t bn = S[p];
      for (uint32_t n = 0; n < 16; n++) {
        if (n == S[p]) continue;
        float c = 0.f;
        for (int i = 0; i < na; i++) {
          const int k = av[i][2];
          const uint32_t w =
              (wv[av[i][0]][av[i][1]] & ~(15u << (4 * k))) | (n << (4 * k));
          const float d = lut(w) - tgt[av[i][0]][av[i][1]];
          c += d * d;
        }
        if (c < bc) { bc = c; bn = n; }
      }
      if (bn != S[p]) {
        gain += cur - bc;
        S[p] = (uint8_t)bn;
        for (int i = 0; i < na; i++) {
          const int vL = av[i][0], ve = av[i][1], k = av[i][2];
          wv[vL][ve] =
              (uint16_t)((wv[vL][ve] & ~(15u << (4 * k))) | (bn << (4 * k)));
          const float d = lut(wv[vL][ve]) - tgt[vL][ve];
          verr[vL][ve] = d * d;
        }
      }
    }
    if (gain < 1e-6) break;
    tot -= gain;
  }
  for (int t = 0; t < 256; t++)
    Wd[t >> 3] =
        (Wd[t >> 3] & ~(15u << (4 * (t & 7)))) | ((uint32_t)S[t] << (4 * (t & 7)));
  return tot;
}

// exact triple pass: for each word, exhaustively re-optimize its nibbles
// 0,1,2 (4096 combos) with everything else fixed. Those three nibbles feed
// exactly 6 windows: own word's e=7,6,5 and next word's e=2,1,0.
static double triple_pass(uint32_t* Wd, const float (*tgt)[8]) {
  uint8_t S[256];
  for (int t = 0; t < 256; t++)
    S[t] = (Wd[t >> 3] >> (4 * (t & 7))) & 15u;
  double gain_tot = 0.0;
  for (int L = 0; L < 32; L++) {
    const int Ln = (L + 1) & 31;
    const uint8_t* sL = S + 8 * L;    // n0..n7 of word L
    const uint8_t* sN = S + 8 * Ln;   // n0..n7 of word L+1
    // window = 4 nibbles at stream positions, oldest at low bits
    // e7: L0,L1,L2,L3  e6: L1,L2,L3,L4  e5: L2,L3,L4,L5
    // n.e2: N5,N6,N7,L0  n.e1: N6,N7,L0,L1  n.e0: N7,L0,L1,L2
    float cur = 0.f, best = 0.f;
    const float* tL = tgt[L];
    const float* tN = tgt[Ln];
    {
      const uint32_t w7 = sL[0] | (sL[1] << 4) | (sL[2] << 8) | (sL[3] << 12);
      const uint32_t w6 = sL[1] | (sL[2] << 4) | (sL[3] << 8) | (sL[4] << 12);
      const uint32_t w5 = sL[2] | (sL[3] << 4) | (sL[4] << 8) | (sL[5] << 12);
      const uint32_t v2 = sN[5] | (sN[6] << 4) | (sN[7] << 8) | (sL[0] << 12);
      const uint32_t v1 = sN[6] | (sN[7] << 4) | (sL[0] << 8) | (sL[1] << 12);
      const uint32_t v0 = sN[7] | (sL[0] << 4) | (sL[1] << 8) | (sL[2] << 12);
      const float d7 = lut(w7) - tL[7], d6 = lut(w6) - tL[6], d5 = lut(w5) - tL[5];
      const float e2 = lut(v2) - tN[2], e1 = lut(v1) - tN[1], e0 = lut(v0) - tN[0];
      cur = d7*d7 + d6*d6 + d5*d5 + e2*e2 + e1*e1 + e0*e0;
    }
    best = cur;
    uint32_t bn = sL[0] | (sL[1] << 4) | (sL[2] << 8);
    for (uint32_t t3 = 0; t3 < 4096; t3++) {
      const uint32_t n0 = t3 & 15, n1 = (t3 >> 4) & 15, n2 = (t3 >> 8) & 15;
      if (t3 == bn) continue;
      const uint32_t w7 = n0 | (n1 << 4) | (n2 << 8) | (sL[3] << 12);
      const uint32_t w6 = n1 | (n2 << 4) | (sL[3] << 8) | (sL[4] << 12);
      const uint32_t w5 = n2 | (sL[3] << 4) | (sL[4] << 8) | (sL[5] << 12);
      const uint32_t v2 = sN[5] | (sN[6] << 4) | (sN[7] << 8) | (n0 << 12);
      const uint32_t v1 = sN[6] | (sN[7] << 4) | (n0 << 8) | (n1 << 12);
      const uint32_t v0 = sN[7] | (n0 << 4) | (n1 << 8) | (n2 << 12);
      const float d7 = lut(w7) - tL[7], d6 = lut(w6) - tL[6], d5 = lut(w5) - tL[5];
      const float e2 = lut(v2) - tN[2], e1 = lut(v1) - tN[1], e0 = lut(v0) - tN[0];
      const float c = d7*d7 + d6*d6 + d5*d5 + e2*e2 + e1*e1 + e0*e0;
      if (c < best) { best = c; bn = t3; }
    }
    if (bn != (uint32_t)(sL[0] | (sL[1] << 4) | (sL[2] << 8))) {
      gain_tot += cur - best;
      S[8 * L + 0] = bn & 15;
      S[8 * L + 1] = (bn >> 4) & 15;
      S[8 * L + 2] = (bn >> 8) & 15;
    }
  }
  for (int t = 0; t < 256; t++)
    Wd[t >> 3] =
        (Wd[t >> 3] & ~(15u << (4 * (t & 7)))) | ((uint32_t)S[t] << (4 * (t & 7)));
  return gain_tot;
}

// iterated local search: perturb k random nibbles, re-polish, keep if better
static double ils(uint32_t* Wd, const float (*tgt)[8], int iters, int k,
                  uint64_t seed) {
  uint32_t best[32];
  std::memcpy(best, Wd, sizeof best);
  double bc = cd_polish(Wd, tgt, 40);
  std::memcpy(best, Wd, sizeof best);
  uint64_t rs = seed ? seed : 0x9e3779b97f4a7c15ull;
  auto rnd = [&]() {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return rs;
  };
  for (int it = 0; it < iters; it++) {
    uint32_t trial[32];
    std::memcpy(trial, best, sizeof trial);
    for (int i = 0; i < k; i++) {
      const int p = rnd() & 255;
      trial[p >> 3] = (trial[p >> 3] & ~(15u << (4 * (p & 7)))) |
                      ((uint32_t)(rnd() & 15) << (4 * (p & 7)));
    }
    triple_pass(trial, tgt);
    const double tc = cd_polish(trial, tgt, 24);
    if (tc < bc - 1e-9) {
      bc = tc;
      std::memcpy(best, trial, sizeof best);
    }
  }
  std::memcpy(Wd, best, sizeof best);
  return bc;
}

// simulated annealing on nibbles: propose (pos, nibble), accept by
// exp(-delta/T); incremental 4-window delta like cd_polish.
static double sa_refine(uint32_t* Wd, const float (*tgt)[8], int steps,
                        double t0, double t1, uint64_t seed) {
  uint8_t S[256];
  for (int t = 0; t < 256; t++)
    S[t] = (Wd[t >> 3] >> (4 * (t & 7))) & 15u;
  uint16_t wv[32][8];
  float verr[32][8];
  double tot = 0.0;
  for (int L = 0; L < 32; L++) {
    const int pL = (L + 31) & 31;
    for (int e = 0; e < 8; e++) {
      uint32_t w = 0;
      for (int k = 0; k < 4; k++) {
        int pos;
        if (e >= 3) pos = 8 * L + 7 - e + k;
        else if (e == 2) pos = k < 3 ? 8 * L + 5 + k : 8 * pL;
        else if (e == 1) pos = k < 2 ? 8 * L + 6 + k : 8 * pL + (k - 2);
        else pos = k == 0 ? 8 * L + 7 : 8 * pL + (k - 1);
        w |= (uint32_t)S[pos] << (4 * k);
      }
      wv[L][e] = (uint16_t)w;
      const float d = lut(w) - tgt[L][e];
      verr[L][e] = d * d;
      tot += verr[L][e];
    }
  }
  uint64_t rs = seed ? seed : 1;
  auto rnd = [&]() {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return rs;
  };
  const double lr = std::log(t1 / t0);
  for (int st = 0; st < steps; st++) {
    const double T = t0 * std::exp(lr * (double)st / steps);
    const int p = rnd() & 255;
    uint32_t n = rnd() & 15;
    const int L = p >> 3, j = p & 7;
    if (n == S[p]) continue;
    const int Ln = (L + 1) & 31;
    int8_t av[4][3];
    int na = 0;
    for (int e = 7 - j < 3 ? 3 : 7 - j; e <= 10 - j && e <= 7; e++) {
      av[na][0] = (int8_t)L; av[na][1] = (int8_t)e; av[na][2] = (int8_t)(j - 7 + e);
      na++;
    }
    if (j == 0) {
      av[na][0]=Ln; av[na][1]=2; av[na][2]=3; na++;
      av[na][0]=Ln; av[na][1]=1; av[na][2]=2; na++;
      av[na][0]=Ln; av[na][1]=0; av[na][2]=1; na++;
    } else if (j == 1) {
      av[na][0]=Ln; av[na][1]=1; av[na][2]=3; na++;
      av[na][0]=Ln; av[na][1]=0; av[na][2]=2; na++;
    } else if (j == 2) {
      av[na][0]=Ln; av[na][1]=0; av[na][2]=3; na++;
    } else if (j == 5) {
      av[na][0]=L; av[na][1]=2; av[na][2]=0; na++;
    } else if (j == 6) {
      av[na][0]=L; av[na][1]=2; av[na][2]=1; na++;
      av[na][0]=L; av[na][1]=1; av[na][2]=0; na++;
    } else if (j == 7) {
      av[na][0]=L; av[na][1]=2; av[na][2]=2; na++;
      av[na][0]=L; av[na][1]=1; av[na][2]=1; na++;
      av[na][0]=L; av[na][1]=0; av[na][2]=0; na++;
    }
    float cur = 0.f, c = 0.f;
    uint16_t nw[4];
    for (int i = 0; i < na; i++) {
      cur += verr[av[i][0]][av[i][1]];
      const int k = av[i][2];
      nw[i] = (uint16_t)((wv[av[i][0]][av[i][1]] & ~(15u << (4 * k))) |
                         (n << (4 * k)));
      const float d = lut(nw[i]) - tgt[av[i][0]][av[i][1]];
      c += d * d;
    }
    const float delta = c - cur;
    if (delta <= 0.f ||
        (rnd() >> 11) * (1.0 / 9007199254740992.0) < std::exp(-delta / T)) {
      S[p] = (uint8_t)n;
      for (int i = 0; i < na; i++) {
        wv[av[i][0]][av[i][1]] = nw[i];
        const float d = lut(nw[i]) - tgt[av[i][0]][av[i][1]];
        verr[av[i][0]][av[i][1]] = d * d;
      }
      tot += delta;
    }
  }
  for (int t = 0; t < 256; t++)
    Wd[t >> 3] =
        (Wd[t >> 3] & ~(15u << (4 * (t & 7)))) | ((uint32_t)S[t] << (4 * (t & 7)));
  return tot;
}

// ---- guided large-neighborhood moves ------------------------------------
// LUT[w16] depends only on s = bytesum(w16 * 0x83DCD12D): ~1021 distinct
// levels. For a bad window, the ideal s* is known in closed form; try every
// w16 preimage of s* (and neighbours) as a coordinated 4-nibble move,
// scoring the damage to all affected windows exactly.
static std::vector<uint16_t> g_by_sum[1200];
static bool g_sum_ready = false;

static void build_sum_index() {
  if (g_sum_ready) return;
  for (uint32_t w = 0; w < 65536; w++) {
    const uint32_t t = w * 0x83DCD12Du;
    const uint32_t s =
        (t & 0xff) + ((t >> 8) & 0xff) + ((t >> 16) & 0xff) + (t >> 24);
    g_by_sum[s].push_back((uint16_t)w);
  }
  g_sum_ready = true;
}

// window (L,e) -> stream positions of its 4 nibbles (slot k at bits 4k)
static inline void win_pos(int L, int e, int* pos) {
  const int pL = (L + 31) & 31;
  for (int k = 0; k < 4; k++) {
    if (e >= 3) pos[k] = 8 * L + 7 - e + k;
    else if (e == 2) pos[k] = k < 3 ? 8 * L + 5 + k : 8 * pL;
    else if (e == 1) pos[k] = k < 2 ? 8 * L + 6 + k : 8 * pL + (k - 2);
    else pos[k] = k == 0 ? 8 * L + 7 : 8 * pL + (k - 1);
  }
}

// positions affected by changing nibble at p -> fills av[4][3] (vL,ve,slot)
static inline int affected(int p, int8_t av[4][3]) {
  const int L = p >> 3, j = p & 7;
  const int Ln = (L + 1) & 31;
  int na = 0;
  for (int e = 7 - j < 3 ? 3 : 7 - j; e <= 10 - j && e <= 7; e++) {
    av[na][0] = (int8_t)L; av[na][1] = (int8_t)e; av[na][2] = (int8_t)(j - 7 + e);
    na++;
  }
  if (j == 0) {
    av[na][0]=Ln; av[na][1]=2; av[na][2]=3; na++;
    av[na][0]=Ln; av[na][1]=1; av[na][2]=2; na++;
    av[na][0]=Ln; av[na][1]=0; av[na][2]=1; na++;
  } else if (j == 1) {
    av[na][0]=Ln; av[na][1]=1; av[na][2]=3; na++;
    av[na][0]=Ln; av[na][1]=0; av[na][2]=2; na++;
  } else if (j == 2) {
    av[na][0]=Ln; av[na][1]=0; av[na][2]=3; na++;
  } else if (j == 5) {
    av[na][0]=L; av[na][1]=2; av[na][2]=0; na++;
  } else if (j == 6) {
    av[na][0]=L; av[na][1]=2; av[na][2]=1; na++;
    av[na][0]=L; av[na][1]=1; av[na][2]=0; na++;
  } else if (j == 7) {
    av[na][0]=L; av[na][1]=2; av[na][2]=2; na++;
    av[na][0]=L; av[na][1]=1; av[na][2]=1; na++;
    av[na][0]=L; av[na][1]=0; av[na][2]=0; na++;
  }
  return na;
}

// one guided round: for each window (worst first), try coordinated moves
static double guided_pass(uint32_t* Wd, const float (*tgt)[8]) {
  build_sum_index();
  uint8_t S[256];
  for (int t = 0; t < 256; t++)
    S[t] = (Wd[t >> 3] >> (4 * (t & 7))) & 15u;
  uint16_t wv[32][8];
  float verr[32][8];
  for (int L = 0; L < 32; L++)
    for (int e = 0; e < 8; e++) {
      int pos[4];
      win_pos(L, e, pos);
      uint32_t w = 0;
      for (int k = 0; k < 4; k++) w |= (uint32_t)S[pos[k]] << (4 * k);
      wv[L][e] = (uint16_t)w;
      const float d = lut(w) - tgt[L][e];
      verr[L][e] = d * d;
    }
  // window order: worst error first
  int order[256];
  for (int i = 0; i < 256; i++) order[i] = i;
  std::sort(order, order + 256,
            [&](int a, int b) { return verr[a >> 3][a & 7] > verr[b >> 3][b & 7]; });
  double gain = 0.0;
  for (int oi = 0; oi < 256; oi++) {
    const int L = order[oi] >> 3, e = order[oi] & 7;
    const float x = tgt[L][e];
    // ideal sum: x = (1024 + s) * c + d  ->  s = (x - d)/c - 1024
    const double ss = ((double)x - (-10.3828125)) / (1.732421875 / 256.0) - 1024.0;
    const int s0 = (int)std::lround(ss);
    int pos[4];
    win_pos(L, e, pos);
    double best_delta = 0.0;
    uint32_t best_w = wv[L][e];
    for (int ds = -2; ds <= 2; ds++) {
      const int s = s0 + ds;
      if (s < 0 || s >= 1200 || g_by_sum[s].empty()) continue;
      for (const uint16_t wc : g_by_sum[s]) {
        // apply candidate window: 4 nibble substitutions at pos[0..3]
        // gather union of affected windows
        int8_t av[16][3];
        int na = 0;
        for (int k = 0; k < 4; k++) {
          int8_t a1[4][3];
          const int n1 = affected(pos[k], a1);
          for (int i = 0; i < n1; i++) {
            bool dup = false;
            for (int m = 0; m < na; m++)
              if (av[m][0] == a1[i][0] && av[m][1] == a1[i][1]) { dup = true; break; }
            if (!dup) { av[na][0]=a1[i][0]; av[na][1]=a1[i][1]; av[na][2]=a1[i][2]; na++; }
          }
        }
        double cur = 0.0, c = 0.0;
        for (int i = 0; i < na; i++) {
          const int vL = av[i][0], ve = av[i][1];
          cur += verr[vL][ve];
          // rebuild this window's w16 with the candidate substitutions
          int vp[4];
          win_pos(vL, ve, vp);
          uint32_t w = wv[vL][ve];
          for (int k = 0; k < 4; k++)
            for (int m = 0; m < 4; m++)
              if (vp[k] == pos[m])
                w = (w & ~(15u << (4 * k))) | ((uint32_t)((wc >> (4 * m)) & 15) << (4 * k));
          const float d = lut(w) - tgt[vL][ve];
          c += (double)d * d;
        }
        const double delta = c - cur;
        if (delta < best_delta) { best_delta = delta; best_w = wc; }
      }
    }
    if (best_w != wv[L][e]) {
      gain += -best_delta;
      for (int m = 0; m < 4; m++) {
        const int p = pos[m];
        const uint32_t nv = (best_w >> (4 * m)) & 15;
        S[p] = (uint8_t)nv;
        int8_t a1[4][3];
        const int n1 = affected(p, a1);
        for (int i = 0; i < n1; i++) {
          const int vL = a1[i][0], ve = a1[i][1], k = a1[i][2];
          wv[vL][ve] = (uint16_t)((wv[vL][ve] & ~(15u << (4 * k))) | (nv << (4 * k)));
          const float d = lut(wv[vL][ve]) - tgt[vL][ve];
          verr[vL][ve] = d * d;
        }
      }
    }
  }
  for (int t = 0; t < 256; t++)
    Wd[t >> 3] =
        (Wd[t >> 3] & ~(15u << (4 * (t & 7)))) | ((uint32_t)S[t] << (4 * (t & 7)));
  return gain;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: htenc TARGET.f32 O K OUT.codes [BEAM] [SWEEPS]\n");
    return 1;
  }
  load_lut(argv[0]);
  build_sum_index();
  const char* tpath = argv[1];
  const int O = std::atoi(argv[2]), K = std::atoi(argv[3]);
  const char* opath = argv[4];
  const int BEAM = argc > 5 ? std::atoi(argv[5]) : 128;
  const int SWEEPS = argc > 6 ? std::atoi(argv[6]) : 6;
  if (O % 128 || K % 128) { std::fprintf(stderr, "off the 128 grid\n"); return 1; }

  std::ifstream tf(tpath, std::ios::binary);
  std::vector<float> G((size_t)O * K);
  if (!tf.read((char*)G.data(), G.size() * 4)) {
    std::fprintf(stderr, "htenc: short target\n");
    return 1;
  }
  const int ntx = K / 128, nty = O / 128;
  std::vector<uint8_t> codes((size_t)O * K / 2, 0);
  double sse = 0.0;
  uint64_t cnt = 0;
  const bool dbg = std::getenv("HTENC_DEBUG") != nullptr;
  const int ILS = std::getenv("HTENC_ILS") ? std::atoi(std::getenv("HTENC_ILS")) : 0;
  const int ILSK = std::getenv("HTENC_ILS_K") ? std::atoi(std::getenv("HTENC_ILS_K")) : 8;
  const bool TRIPLE = std::getenv("HTENC_TRIPLE") ? std::atoi(std::getenv("HTENC_TRIPLE")) != 0 : true;
  const int BEAM0 = std::getenv("HTENC_BEAM0") ? std::atoi(std::getenv("HTENC_BEAM0")) : BEAM;
  const int NOISE = std::getenv("HTENC_NOISE") ? std::atoi(std::getenv("HTENC_NOISE")) : 0;  // restarts with noisy tie-break
  const int SA = std::getenv("HTENC_SA") ? std::atoi(std::getenv("HTENC_SA")) : 0;  // anneal steps per block
  const double SAT0 = std::getenv("HTENC_SA_T0") ? std::atof(std::getenv("HTENC_SA_T0")) : 0.05;
  const double SAT1 = std::getenv("HTENC_SA_T1") ? std::atof(std::getenv("HTENC_SA_T1")) : 1e-4;
  const int GUIDED = std::getenv("HTENC_GUIDED") ? std::atoi(std::getenv("HTENC_GUIDED")) : 0;  // LNS rounds

  // beam entry: cost + 16-nibble history + parent index
  struct Cand { float c; uint64_t h; uint32_t p; };

#pragma omp parallel
  {
    const int BMAX = BEAM0 > BEAM ? BEAM0 : BEAM;
    std::vector<Cand> beam(BMAX), cand(BMAX * 16);
    std::vector<std::vector<uint32_t>> parents(256);  // per step: parent<<4 | nib
    for (auto& v : parents) v.resize(BMAX);
#pragma omp for schedule(dynamic, 1) reduction(+ : sse) reduction(+ : cnt)
    for (int64_t tile = 0; tile < (int64_t)nty * ntx; tile++) {
      const int ty = tile / ntx, tx = tile % ntx;
      uint8_t* tb = codes.data() + (size_t)tile * 8192;
      for (int g = 0; g < 8; g++) {
        for (int q = 0; q < 8; q++) {
          float tgt[32][8];
          uint32_t Wd[32];
          for (int L = 0; L < 32; L++) {
            const int r = ty * 128 + q * 16 + (L & 15);
            const int c0 = tx * 128 + g * 16 + (L >> 4) * 8;
            const float* gp = &G[(size_t)r * K + c0];
            for (int e = 0; e < 8; e++) tgt[L][e] = gp[e];
            Wd[L] = 0;
          }
          float best_c = 1e30f;
          for (int sweep = 0; sweep < SWEEPS; sweep++) {
            const int CB = sweep == 0 ? BEAM0 : BEAM;  // wide first sweep
            const int p0 = (sweep * 96) & 255;  // rotated frozen window
            // seed history: current nibbles at ring positions p0-16..p0-1,
            // most recent (p0-1) at the low bits
            uint64_t h0 = 0;
            for (int k = 0; k < 16; k++) {
              const int a = (p0 - 16 + k) & 255;
              const uint32_t nib = (Wd[a >> 3] >> (4 * (a & 7))) & 15u;
              h0 |= (uint64_t)nib << (4 * (15 - k));
            }
            beam[0] = {0.f, h0, 0};
            int nb = 1;
            for (int t = 0; t < 256; t++) {
              const int a = (p0 + t) & 255;
              const int L = a >> 3, j = a & 7;
              const float* tg = tgt[L];
              // forced tail: last 16 steps replay the seed (= same ring
              // positions), making the ring exactly self-consistent
              const uint32_t nforce =
                  (Wd[a >> 3] >> (4 * (a & 7))) & 15u;
              const uint32_t nlo = t < 240 ? 0u : nforce;
              const uint32_t nhi = t < 240 ? 16u : nforce + 1;
              int nc = 0;
              for (int b = 0; b < nb; b++) {
                const float bc = beam[b].c;
                const uint64_t bh = beam[b].h;
                for (uint32_t n = nlo; n < nhi; n++) {
                  const uint64_t h = (bh << 4) | n;  // recent at low bits
                  float c = bc;
                  if (j >= 3) {
                    const float d = lut(hw4(h)) - tg[10 - j];
                    c += d * d;
                    if (j == 7) {
                      // e=2: n5,n6,n7 + prev n0 (positions t-2,t-1,t, t-15)
                      float d2 = lut(hb(h, 2) | (hb(h, 1) << 4) | (hb(h, 0) << 8) |
                                     (hb(h, 15) << 12)) - tg[2];
                      float d1 = lut(hb(h, 1) | (hb(h, 0) << 4) | (hb(h, 15) << 8) |
                                     (hb(h, 14) << 12)) - tg[1];
                      float d0 = lut(hb(h, 0) | (hb(h, 15) << 4) | (hb(h, 14) << 8) |
                                     (hb(h, 13) << 12)) - tg[0];
                      c += d2 * d2 + d1 * d1 + d0 * d0;
                    }
                  }
                  cand[nc++] = {c, h, (uint32_t)b};
                }
              }
              const int keep = nc < CB ? nc : CB;
              if (keep < nc)
                std::nth_element(cand.begin(), cand.begin() + keep,
                                 cand.begin() + nc,
                                 [](const Cand& a, const Cand& b) {
                                   return a.c < b.c;
                                 });
              std::sort(cand.begin(), cand.begin() + keep,
                        [](const Cand& a, const Cand& b) { return a.c < b.c; });
              for (int b = 0; b < keep; b++) {
                beam[b] = cand[b];
                parents[t][b] = (cand[b].p << 4) | (hb(cand[b].h, 0));
              }
              nb = keep;
            }
            // best final entry; beam cost is exact (ring forced consistent)
            int bs = 0;
            for (int b = 1; b < nb; b++)
              if (beam[b].c < beam[bs].c) bs = b;
            const float fc = beam[bs].c;
            if (dbg && tile == 0 && g == 0 && q == 0)
              std::fprintf(stderr, "  dbg sweep %d p0 %d: beam_cost %.4f\n", sweep,
                           p0, fc);
            if (fc < best_c) {
              best_c = fc;
              for (int t = 255; t >= 0; t--) {
                const uint32_t pk = parents[t][bs];
                const int a = (p0 + t) & 255;
                Wd[a >> 3] = (Wd[a >> 3] & ~(15u << (4 * (a & 7)))) |
                             ((pk & 15u) << (4 * (a & 7)));
                bs = pk >> 4;
              }
              // polish right away: better seed for the next sweep
              best_c = (float)cd_polish(Wd, tgt, 12);
              if (dbg && tile == 0 && g == 0 && q == 0)
                std::fprintf(stderr, "  dbg sweep %d after cd: %.4f\n", sweep,
                             best_c);
            }
          }
          cd_polish(Wd, tgt, 40);
          if (TRIPLE) {
            triple_pass(Wd, tgt);
            cd_polish(Wd, tgt, 40);
          }
          if (ILS > 0)
            ils(Wd, tgt, ILS, ILSK,
                ((uint64_t)tile * 64 + g * 8 + q + 1) * 0x9e3779b97f4a7c15ull);
          if (SA > 0) {
            sa_refine(Wd, tgt, SA, SAT0, SAT1,
                      ((uint64_t)tile * 64 + g * 8 + q + 7) * 0x2545f4914f6cdd1dull);
            cd_polish(Wd, tgt, 40);
          }
          for (int gr = 0; gr < GUIDED; gr++) {
            const double gg = guided_pass(Wd, tgt);
            cd_polish(Wd, tgt, 40);
            if (dbg && tile == 0 && g == 0 && q == 0)
              std::fprintf(stderr, "  dbg guided round %d gain %.4f\n", gr, gg);
            if (gg < 1e-6) break;
          }
          // emit + true-score
          uint8_t* bp = tb + g * 1024 + q * 128;
          double bsse = 0.0;
          for (int L = 0; L < 32; L++) {
            const uint32_t prev = Wd[(L + 31) & 31];
            const uint64_t V = (uint64_t)Wd[L] | ((uint64_t)prev << 32);
            for (int e = 0; e < 8; e++) {
              const float d = lut((uint32_t)(V >> (28 - 4 * e))) - tgt[L][e];
              sse += (double)d * d;
              bsse += (double)d * d;
              cnt++;
            }
            std::memcpy(bp + 4 * L, &Wd[L], 4);
          }
          if (dbg && tile == 0 && g == 0 && q == 0)
            std::fprintf(stderr, "  dbg best %.4f true block_sse %.4f\n", best_c,
                         bsse);
        }
      }
    }
  }
  std::ofstream of(opath, std::ios::binary);
  of.write((const char*)codes.data(), codes.size());
  of.close();
  std::fprintf(stderr, "htenc: %dx%d mse %.6e rmse %.6f beam %d\n", O, K,
               sse / cnt, std::sqrt(sse / cnt), BEAM);
  return 0;
}
