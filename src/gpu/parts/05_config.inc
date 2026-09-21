// ---------------- config (matches ref.cpp / config.json) --------------------
struct Cfg {
  int d = 2560;
  int layers = 48;
  int branches = 4;
  int gdn_hk = 16, gdn_dk = 128;
  int gdn_hv = 48, gdn_dv = 128;
  int gdn_conv = 4;
  int qsa_hq = 24, qsa_dh = 256;
  int qsa_hkv = 2;
  int rotary_dim = 64;
  int gr_rank = 320;
  int experts = 512, topk = 10, moe_mid = 640;
  float norm_eps = 1e-6f;
  double rope_theta = 1e7;
  int vocab = 248320;
};
static Cfg g_cfg;
static hipStream_t g_str = nullptr;  // capture stream during graph build, else default

static bool is_qsa(int layer) { return layer % 4 == 3; }

