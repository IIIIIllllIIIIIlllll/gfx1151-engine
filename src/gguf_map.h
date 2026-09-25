// gguf_map.h — build the engine's (hgn-named) non-expert tensors from a
// llama.cpp GGUF (qwen4exp arch, e.g. Unsloth Qwen3.8-Flash-Next UD-Q4_K_XL)
// plus the optional MTP sidecar (mtp-*-Q8_0.gguf: blk.48.* + blk.48.nextn.*).
//
// The result is a set of synthetic hgn::Tensor records (hgn::Checkpoint::
// add_synthetic) under the exact hgn names, so the loader, wview/gemv/gemm
// and the small-tensor uploads keep working unchanged:
//   * Q8_0 matrices  -> dtype 8 q8g32 planar (lossless repack)
//   * F32 matrices   -> dtype 0 bf16 (router gate, ssm_alpha/beta, shexp gate,
//                        HC inject; also the Q8_0 MTP inject)
//   * F32 vectors    -> dtype 1 f32 (norms, conv1d, A_log, dt_bias)
//   * BF16 indexer q_proj(512)+k_proj(128) -> one bf16 index_qk_proj(640)
//   * PLE config KV arrays -> dtype 4 u64
// Routed experts and the PLE n-gram table are NOT built here (experts are
// read in place by 26_kernels_moe_gguf.inc; the table is streamed).
//
// GGUF vs HF/engine transforms (semantics from gufo reference.cpp, MIT;
// verified numerically against hgn by tools/g3_map_check.cpp):
//   * layout: GGUF ne[0] is the torch last dim -> no transposes.
//   * GDN value heads are "tiled" in GGUF (value head g reads key head g%16)
//     while HF / the engine read key head h/3; engine head h = GGUF head
//     (h%3)*16 + h/3. Applies to the v rows of attn_qkv, attn_gate (z),
//     ssm_alpha/beta, ssm_a, ssm_dt.bias, the v channels of ssm_conv1d and
//     the 128-col head blocks of ssm_out's input. ssm_norm (128) is shared.
//   * ssm_a = -exp(A_log)  ->  A_log = log(-ssm_a).
//   * zero-centred RMSNorm weights are stored with +1 folded in (llama.cpp
//     converter); the engine's k_rmsnorm_zc* want the raw w -> w - 1.
//   * MTP eh_proj [2560][5120] = [embedding cols | hidden cols].
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "gguf.h"
#include "hgn.h"

namespace gguf_map {

struct Opts {
  bool vperm = true;  // un-tile GDN value heads
  // +1 folded into these GGUF norm kinds (subtract 1 to get the hgn/HF value)
  bool fold_hc = true, fold_qk = true, fold_idx = true, fold_ssm = false,
       fold_ple = true, fold_mtp = true;
  std::function<bool(int)> layer;  // trunk layer filter (empty = all)
  bool globals = true;             // embed / lm_head / output HC / PLE (non-table)
  bool mtp = true;                 // build mtp.* when an MTP file is given
};

using Map = std::vector<uint32_t>;

// Debug/bisect knob: when set, only hgn names it accepts are written (every
// other tensor keeps its hgn record). Each written tensor is individually
// converted to engine order, so any subset is self-consistent.
inline std::function<bool(const std::string&)> g_keep;
inline void put(hgn::Checkpoint& ck, const hgn::Tensor& d, std::vector<uint8_t>&& buf) {
  if (g_keep && !g_keep(d.name)) return;
  ck.add_synthetic(d, std::move(buf));
}

inline uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

inline void par_for(uint64_t n, const std::function<void(uint64_t, uint64_t)>& f,
                    uint64_t min_per = 4096) {
  unsigned nt = std::thread::hardware_concurrency();
  nt = nt ? std::min(nt, 16u) : 4u;
  if (n < min_per * 2 || nt < 2) { f(0, n); return; }
  nt = (unsigned)std::min<uint64_t>(nt, n / min_per);
  std::vector<std::thread> th;
  for (unsigned i = 0; i < nt; i++)
    th.emplace_back([&, i] { f(n * i / nt, n * (i + 1) / nt); });
  for (auto& t : th) t.join();
}

inline hgn::Tensor desc(const std::string& name, uint32_t dtype,
                        std::initializer_list<uint64_t> dims) {
  hgn::Tensor t;
  t.name = name;
  t.dtype = dtype;
  t.ndims = (uint32_t)dims.size();
  int i = 0;
  for (uint64_t d : dims) t.dims[i++] = d;
  return t;
}

inline void need(bool c, const std::string& what) {
  if (!c) throw std::runtime_error("gguf_map: " + what);
}

// Q8_0 [R][C] -> q8g32 [R'][C'] taking source row rows[r] (identity if empty)
// and source 32-col block blks[b] (identity if empty).
inline void add_q8(hgn::Checkpoint& ck, const std::string& name, const gguf::Tensor& t,
                   std::initializer_list<uint64_t> dims, const Map& rows = {},
                   const Map& blks = {}) {
  need(t.type == gguf::Q8_0, name + " <- " + t.name + ": expected Q8_0, got " +
                                 gguf::type_name(t.type));
  const uint64_t C = t.ne[0], R = t.rows(), nbs = C / 32;
  const uint64_t Ro = rows.empty() ? R : rows.size();
  const uint64_t nbo = blks.empty() ? nbs : blks.size(), Co = nbo * 32;
  hgn::Tensor d = desc(name, 8, dims);
  need(d.numel() == Ro * Co, name + ": shape mismatch vs " + t.name);
  for (uint32_t r : rows) need(r < R, name + ": row map out of range");
  for (uint32_t b : blks) need(b < nbs, name + ": block map out of range");
  std::vector<uint8_t> buf(Ro * Co + Ro * nbo * 2);
  int8_t* q = (int8_t*)buf.data();
  uint16_t* s = (uint16_t*)(buf.data() + Ro * Co);
  par_for(Ro, [&](uint64_t r0, uint64_t r1) {
    for (uint64_t r = r0; r < r1; r++) {
      const uint8_t* src = t.data + (uint64_t)(rows.empty() ? r : rows[r]) * nbs * 34;
      for (uint64_t b = 0; b < nbo; b++) {
        const uint8_t* bk = src + (uint64_t)(blks.empty() ? b : blks[b]) * 34;
        memcpy(&s[r * nbo + b], bk, 2);
        memcpy(q + r * Co + b * 32, bk + 2, 32);
      }
    }
  }, 256);
  put(ck, d, std::move(buf));
}

// any GGUF type -> f32 [R'][C] with source row map
inline std::vector<float> rows_f32(const gguf::Tensor& t, const Map& rows = {}) {
  const uint64_t C = t.ne[0], R = t.rows(), rb = t.row_bytes();
  const uint64_t Ro = rows.empty() ? R : rows.size();
  for (uint32_t r : rows) need(r < R, t.name + ": row map out of range");
  std::vector<float> out(Ro * C);
  par_for(Ro, [&](uint64_t r0, uint64_t r1) {
    for (uint64_t r = r0; r < r1; r++)
      gguf::dequant_row(t.type, t.data + (uint64_t)(rows.empty() ? r : rows[r]) * rb,
                        &out[r * C], C);
  }, 64);
  return out;
}

inline void add_bf16v(hgn::Checkpoint& ck, const std::string& name, const std::vector<float>& f,
                      std::initializer_list<uint64_t> dims) {
  hgn::Tensor d = desc(name, 0, dims);
  need(d.numel() == f.size(), name + ": bf16 shape mismatch");
  std::vector<uint8_t> buf(f.size() * 2);
  uint16_t* p = (uint16_t*)buf.data();
  for (size_t i = 0; i < f.size(); i++) p[i] = f2bf(f[i]);
  put(ck, d, std::move(buf));
}
inline void add_bf16(hgn::Checkpoint& ck, const std::string& name, const gguf::Tensor& t,
                     std::initializer_list<uint64_t> dims, const Map& rows = {}) {
  add_bf16v(ck, name, rows_f32(t, rows), dims);
}
inline void add_f32v(hgn::Checkpoint& ck, const std::string& name, const std::vector<float>& f,
                     std::initializer_list<uint64_t> dims) {
  hgn::Tensor d = desc(name, 1, dims);
  need(d.numel() == f.size(), name + ": f32 shape mismatch");
  std::vector<uint8_t> buf(f.size() * 4);
  memcpy(buf.data(), f.data(), buf.size());
  put(ck, d, std::move(buf));
}
inline std::vector<float> norm_vec(const gguf::Tensor& t, bool fold) {
  std::vector<float> v = rows_f32(t);
  if (fold)
    for (float& x : v) x -= 1.f;
  return v;
}
inline void add_u64(hgn::Checkpoint& ck, const std::string& name, const gguf::File& g,
                    const std::string& key, size_t n) {
  const gguf::Value* v = g.kv(key);
  need(v && v->is_array && v->ai.size() == n, "KV " + key + " missing or wrong length");
  std::vector<uint8_t> buf(n * 8);
  for (size_t i = 0; i < n; i++) {
    uint64_t x = (uint64_t)v->ai[i];
    memcpy(buf.data() + i * 8, &x, 8);
  }
  put(ck, desc(name, 4, {(uint64_t)n}), std::move(buf));
}

// GDN head/row/channel maps (engine index -> GGUF index)
inline uint32_t vhead(const Opts& o, uint32_t h) { return o.vperm ? (h % 3) * 16 + h / 3 : h; }
inline Map map_heads(const Opts& o, uint32_t nh, uint32_t hd, uint32_t base_rows) {
  Map m;
  for (uint32_t r = 0; r < base_rows; r++) m.push_back(r);
  for (uint32_t h = 0; h < nh; h++)
    for (uint32_t d = 0; d < hd; d++) m.push_back(base_rows + vhead(o, h) * hd + d);
  return m;
}

// One transformer block. `g`/`bl`: GGUF file + block index; `P`: hgn prefix
// ("layers.7" / "mtp.layers.0").
inline void build_layer(hgn::Checkpoint& ck, const gguf::File& g, int bl, const std::string& P,
                        bool qsa, const Opts& o) {
  const std::string B = "blk." + std::to_string(bl) + ".";
  auto G = [&](const std::string& s) -> const gguf::Tensor& { return g.at(B + s); };
  const char* hk[2] = {"attn", "mlp"};
  const char* gk[2] = {"attn", "ffn"};
  for (int i = 0; i < 2; i++) {
    const std::string hp = P + "." + hk[i] + "_hyper_connection.";
    const std::string gp = std::string("hc_") + gk[i] + "_";
    add_q8(ck, hp + "input_mix_weight_down.weight", G(gp + "down.weight"), {320, 10240});
    add_q8(ck, hp + "input_mix_weight_up.weight", G(gp + "up.weight"), {10240, 320});
    add_bf16(ck, hp + "block_inject_weight.weight", G(gp + "inject.weight"), {4, 10240});
    add_f32v(ck, hp + "hc_norm.weight", norm_vec(G(gp + "norm.weight"), o.fold_hc), {10240});
  }
  add_bf16(ck, P + ".mlp.gate.weight", G("ffn_gate_inp.weight"), {512, 2560});
  add_bf16(ck, P + ".mlp.shared_expert_gate.weight", G("ffn_gate_inp_shexp.weight"), {1, 2560});
  add_q8(ck, P + ".mlp.shared_expert.gate_proj.weight", G("ffn_gate_shexp.weight"), {640, 2560});
  add_q8(ck, P + ".mlp.shared_expert.up_proj.weight", G("ffn_up_shexp.weight"), {640, 2560});
  add_q8(ck, P + ".mlp.shared_expert.down_proj.weight", G("ffn_down_shexp.weight"), {2560, 640});
  if (qsa) {
    const std::string A = P + ".self_attn.";
    add_q8(ck, A + "q_proj.weight", G("attn_q.weight"), {12288, 2560});
    add_q8(ck, A + "k_proj.weight", G("attn_k.weight"), {512, 2560});
    add_q8(ck, A + "v_proj.weight", G("attn_v.weight"), {512, 2560});
    add_q8(ck, A + "o_proj.weight", G("attn_output.weight"), {2560, 6144});
    add_f32v(ck, A + "q_norm.weight", norm_vec(G("attn_q_norm.weight"), o.fold_qk), {256});
    add_f32v(ck, A + "k_norm.weight", norm_vec(G("attn_k_norm.weight"), o.fold_qk), {256});
    std::vector<float> iq = rows_f32(G("indexer.q_proj.weight"));
    std::vector<float> ik = rows_f32(G("indexer.k_proj.weight"));
    need(iq.size() == 512 * 2560 && ik.size() == 128 * 2560, B + "indexer shapes");
    iq.insert(iq.end(), ik.begin(), ik.end());
    add_bf16v(ck, A + "indexer.index_qk_proj.weight", iq, {640, 2560});
    add_f32v(ck, A + "indexer.q_layernorm.weight", norm_vec(G("indexer.q_norm.weight"), o.fold_idx),
             {128});
    add_f32v(ck, A + "indexer.k_layernorm.weight", norm_vec(G("indexer.k_norm.weight"), o.fold_idx),
             {128});
    return;
  }
  const std::string L = P + ".linear_attn.";
  const Map mqkv = map_heads(o, 48, 128, 4096);  // q(2048) k(2048) | v 48x128
  const Map mz = map_heads(o, 48, 128, 0);
  const Map m48 = map_heads(o, 48, 1, 0);
  Map mout;  // ssm_out input cols: 48 heads x 4 Q8_0 blocks
  for (uint32_t h = 0; h < 48; h++)
    for (uint32_t j = 0; j < 4; j++) mout.push_back(vhead(o, h) * 4 + j);
  add_q8(ck, L + "in_proj_qkv.weight", G("attn_qkv.weight"), {10240, 2560}, mqkv);
  add_q8(ck, L + "in_proj_z.weight", G("attn_gate.weight"), {6144, 2560}, mz);
  add_bf16(ck, L + "in_proj_a.weight", G("ssm_alpha.weight"), {48, 2560}, m48);
  add_bf16(ck, L + "in_proj_b.weight", G("ssm_beta.weight"), {48, 2560}, m48);
  add_q8(ck, L + "out_proj.weight", G("ssm_out.weight"), {2560, 6144}, {}, mout);
  add_f32v(ck, L + "conv1d.weight", rows_f32(G("ssm_conv1d.weight"), mqkv), {10240, 1, 4});
  std::vector<float> a = rows_f32(G("ssm_a")), dt = rows_f32(G("ssm_dt.bias"));
  need(a.size() == 48 && dt.size() == 48, B + "ssm_a/dt shapes");
  std::vector<float> alog(48), dtb(48);
  for (uint32_t h = 0; h < 48; h++) {
    alog[h] = logf(-a[vhead(o, h)]);
    dtb[h] = dt[vhead(o, h)];
  }
  add_f32v(ck, L + "A_log", alog, {48});
  add_f32v(ck, L + "dt_bias", dtb, {48});
  add_f32v(ck, L + "norm.weight", norm_vec(G("ssm_norm.weight"), o.fold_ssm), {128});
}

// Main GGUF (+ optional MTP sidecar) -> synthetic tensors in `ck`.
// Returns the number of tensors written (new or overriding hgn records).
inline size_t build(hgn::Checkpoint& ck, const gguf::File& g, const gguf::File* mtp,
                    const Opts& o = Opts()) {
  const size_t n0 = ck.synthetic_count();
  const std::string arch = g.arch();
  const int nl = (int)g.kv_i(arch + ".block_count", 48);
  need(arch == "qwen4exp", "unexpected arch '" + arch + "'");
  for (int l = 0; l < nl && l < 48; l++) {
    if (o.layer && !o.layer(l)) continue;
    build_layer(ck, g, l, "layers." + std::to_string(l), l % 4 == 3, o);
  }
  if (o.globals) {
    add_q8(ck, "embed_tokens.weight", g.at("token_embd.weight"), {248320, 2560});
    add_q8(ck, "lm_head.weight", g.at("output.weight"), {248320, 2560});
    add_q8(ck, "hyper_connection_mixer.input_mix_weight_down.weight", g.at("output_hc_down.weight"),
           {320, 10240});
    add_q8(ck, "hyper_connection_mixer.input_mix_weight_up.weight", g.at("output_hc_up.weight"),
           {10240, 320});
    add_f32v(ck, "hyper_connection_mixer.hc_norm.weight",
             norm_vec(g.at("output_hc_norm.weight"), o.fold_hc), {10240});
    const std::string E = "layers.1.ple.";
    add_q8(ck, E + "key_proj.weight", g.at("blk.1.ple_key.weight"), {10240, 2560});
    add_q8(ck, E + "value_proj.weight", g.at("blk.1.ple_value.weight"), {2560, 2560});
    add_f32v(ck, E + "conv1d.weight", rows_f32(g.at("blk.1.ple_conv1d.weight")), {10240, 1, 4});
    add_f32v(ck, E + "norm_conv.weight", norm_vec(g.at("blk.1.ple_norm_conv.weight"), o.fold_ple),
             {10240});
    add_f32v(ck, E + "norm_key.weight", norm_vec(g.at("blk.1.ple_norm_key.weight"), o.fold_ple),
             {10240});
    add_f32v(ck, E + "norm_query.weight",
             norm_vec(g.at("blk.1.ple_norm_query.weight"), o.fold_ple), {10240});
    add_u64(ck, E + "ple_embedding.layer_multipliers", g, arch + ".ple.layer_multipliers", 3);
    add_u64(ck, E + "ple_embedding.ngram_heads_offsets", g, arch + ".ple.head_offsets", 16);
    add_u64(ck, E + "ple_embedding.ngram_heads_vocab_sizes", g, arch + ".ple.head_vocab_sizes", 16);
  }
  if (mtp && o.mtp) {
    const gguf::File& m = *mtp;
    const int bl = (int)m.kv_i(m.arch() + ".block_count", 49) - 1;
    const std::string N = "blk." + std::to_string(bl) + ".nextn.";
    const gguf::Tensor& eh = m.at(N + "eh_proj.weight");
    need(eh.ne[0] == 5120 && eh.ne[1] == 2560, "eh_proj shape");
    Map le, lh;
    for (uint32_t b = 0; b < 80; b++) le.push_back(b), lh.push_back(80 + b);
    add_q8(ck, "mtp.fc_embedding.weight", eh, {2560, 2560}, {}, le);
    add_q8(ck, "mtp.fc_hidden.weight", eh, {2560, 2560}, {}, lh);
    add_f32v(ck, "mtp.pre_fc_norm_embedding.weight", norm_vec(m.at(N + "enorm.weight"), o.fold_mtp),
             {2560});
    add_f32v(ck, "mtp.pre_fc_norm_hidden.weight", norm_vec(m.at(N + "hnorm.weight"), o.fold_mtp),
             {10240});
    add_q8(ck, "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
           m.at(N + "hc_head_down.weight"), {320, 10240});
    add_q8(ck, "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
           m.at(N + "hc_head_up.weight"), {10240, 320});
    add_f32v(ck, "mtp.hyper_connection_mixer.hc_norm.weight",
             norm_vec(m.at(N + "hc_head_norm.weight"), o.fold_hc), {10240});
    build_layer(ck, m, bl, "mtp.layers.0", true, o);
  }
  return ck.synthetic_count() - n0;
}

}  // namespace gguf_map
