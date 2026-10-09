// launch_panel.inc — start_win.exe 的启动配置面板（由 launch_win.cpp include，单 TU）。
// Win32 原生窗口，零新运行时依赖；不用 .rc/对话框模板（llvm-rc 不可用），
// CreateWindowExW 自建窗口 + 子控件，代码内布局。
//
// 面板分 4 页，顶部 4 个 BS_PUSHLIKE|BS_AUTORADIOBUTTON 切换（不引入 comctl32）：
//   第 1 页 权重：V1（w4b 单文件）/ V2（主体+独立 n-gram）/ GGUF 版本选择，
//          各版本只显示自己相关的文件行；GGUF 仅保存（Windows 启动器只支持 hgn）。
//   第 2 页 上下文与并发：MAX_CONTEXT / PARALLEL / 并发总池（KV_POOL_TOKENS）/
//          KV_PAGED / PREFILL_CHUNK / YaRN 勾选（ROPE_FACTOR 按上限自动推导）。
//          ROPE_ORIGINAL_CTX 不做 UI 行（用户不该改）：校验仍从 g_conf/环境读，
//          也不在 kManagedKeys 里，save_conf 永远不会改写 conf 中的该行。
//   第 3 页 端口与监听：ENGINE_HOST / ENGINE_PORT / API_HOST / API_PORT。
//   第 4 页 显存检查：HIP 可用显存、权重合计、并发总池/工作区/并发估算、
//          合计对照结论，以及其它页的失败汇总；"重新检查"按钮在本页。
// 每页底部有独立的多行只读检查区（[通过]/[失败]/[警告]/[提示] 前缀）。
// 底部按钮行（启动服务 / 仅保存 / 取消）在所有页之外，恒可见。
// "启动服务" 仅在所有页无 [失败] 且所选权重为 V1/V2 时可用。
//
// 依赖 launch_win.cpp 中已定义的：g_root、g_conf、load_conf()、cfg()、
// cfg_optional()、to_w()、msgbox()、file_exists()、dir_exists()、same_file()、
// port_free()、engine_net::address()（launch_win.cpp 已 include engine_net.h）。
//
// 入口：panel_run(allow_start)
//   allow_start=true  ：首次启动 / --setup，按钮为 启动服务 / 仅保存 / 取消
//   allow_start=false ：服务运行中从托盘"设置…"打开，启动按钮禁用（重启生效）
// 返回 kPanelStart（已保存，继续启动）/ kPanelSave（仅保存）/ kPanelCancel /
// kPanelRelang（切换了界面语言，调用方重开面板）。
//
// 界面语言：中英双语，文案处用 TR(中文, English) 就地双语。优先级：
//   service.conf 的 "# start_win: lang=zh|en" 标记 > 环境变量 UI_LANG >
//   GetUserDefaultUILanguage()（中文→zh，其它→en）。
// 面板右下角语言按钮写 lang 标记并返回 kPanelRelang。
// 检查级别前缀本地化为 [通过]/[失败]/[警告]/[提示] 与 [PASS]/[FAIL]/[WARN]/[HINT]，
// 判定用 Lv 枚举（不再字符串比较）。托盘菜单 / 控制台输出暂未翻译。
//
// 单次显示标记：service.conf 里的一行 "# start_win: configured"。首次服务就绪
// （report(kReady)）时由 conf_write_marker() 写入；存在则双击不再弹面板。
// 权重版本选择持久化为另一行注释 "# start_win: weights=v1|v2|gguf"（存在则改写、
// 不存在则追加）；无标记时按 NGRAM_FILE 是否与 MODEL_FILE 同文件推断 V1/V2。

#include <commdlg.h>
#include <set>

namespace {

// ---- service.conf 读写 -------------------------------------------------------

const char* kPanelMarker = "# start_win: configured";
const char* kWeightsMarker = "# start_win: weights=";
const char* kLangMarker = "# start_win: lang=";

// 界面语言开关：panel_run 每次打开时按 lang 标记 > UI_LANG > 系统 UI 语言解析。
// TR 宏就地双语（两个分支必须一个窄串对窄串、一个宽串对宽串）。
bool g_lang_en = false;
#define TR(zh, en) (g_lang_en ? (en) : (zh))

std::string conf_path() { return g_root + "\\service.conf"; }

bool read_text_file(const std::string& path, std::string* out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char buf[8192];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out->append(buf, n);
    fclose(f);
    return true;
}

bool conf_has_marker() {
    std::string text;
    return read_text_file(conf_path(), &text) && text.find(kPanelMarker) != std::string::npos;
}

// 首次服务就绪后调用：写入"面板已配置"标记（幂等）。
void conf_write_marker() {
    if (conf_has_marker()) return;
    FILE* f = fopen(conf_path().c_str(), "ab");
    if (!f) return;  // conf 缺失时面板必然会出现，不用补标记
    const std::string marker_line = std::string("\n") + kPanelMarker + "\n";
    fwrite(marker_line.data(), 1, marker_line.size(), f);
    fclose(f);
}

// 读 "# start_win: <prefix><value>" 行的值；没有返回空串。
std::string conf_marker_value(const char* prefix) {
    std::string text;
    if (!read_text_file(conf_path(), &text)) return "";
    const size_t p = text.find(prefix);
    if (p == std::string::npos) return "";
    size_t b = p + strlen(prefix);
    size_t e = text.find_first_of("\r\n", b);
    std::string v = text.substr(b, e == std::string::npos ? std::string::npos : e - b);
    const size_t a = v.find_first_not_of(" \t");
    const size_t z = v.find_last_not_of(" \t");
    return a == std::string::npos ? "" : v.substr(a, z - a + 1);
}

std::string conf_weights_marker() { return conf_marker_value(kWeightsMarker); }

// 写 "prefix+value" 标记行：已有该前缀行则改写，否则追加到文件末尾。
void conf_write_marker_line(const char* prefix, const char* value) {
    const std::string path = conf_path();
    std::string text;
    read_text_file(path, &text);
    std::string out;
    bool written = false;
    size_t pos = 0;
    while (pos <= text.size()) {
        size_t eol = text.find('\n', pos);
        const bool last = eol == std::string::npos;
        std::string line = text.substr(pos, last ? std::string::npos : eol - pos);
        pos = last ? text.size() + 1 : eol + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!written && line.compare(0, strlen(prefix), prefix) == 0) {
            out += std::string(prefix) + value;
            written = true;
        } else {
            out += line;
        }
        if (!last) out += '\n';
    }
    if (!written) {
        if (!out.empty() && out.back() != '\n') out += '\n';
        out += prefix + std::string(value) + "\n";
    }
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
}

// 界面语言解析：lang 标记 > UI_LANG > 系统 UI 语言（中文 → zh，其它 → en）。
void panel_lang_resolve() {
    const std::string m = conf_marker_value(kLangMarker);
    if (m == "en") return void(g_lang_en = true);
    if (m == "zh") return void(g_lang_en = false);
    if (const char* e = getenv("UI_LANG")) {
        if (_stricmp(e, "en") == 0) return void(g_lang_en = true);
        if (_stricmp(e, "zh") == 0) return void(g_lang_en = false);
    }
    g_lang_en = PRIMARYLANGID(GetUserDefaultUILanguage()) != LANG_CHINESE;
}

// 面板管理的 conf 键（也是环境变量覆盖提示的范围）。
// ROPE_ORIGINAL_CTX 刻意不在其中：面板不做它的 UI 行，save_conf 也不改写它。
const char* kManagedKeys[] = {
    "MODEL_FILE", "NGRAM_FILE", "OVERLAY_FILE", "MTP_FILE", "VISION_FILE",
    "GGUF_FILE", "GGUF_MTP_FILE", "GGUF_VISION_FILE",
    "TOKENIZER_DIR", "MAX_CONTEXT", "PARALLEL", "KV_POOL_TOKENS", "KV_PAGED",
    "PREFILL_CHUNK", "ROPE_FACTOR",
    "KVSNAP_MAX_GB",
    "ENGINE_HOST", "ENGINE_PORT", "API_HOST", "API_PORT",
};
// 可选文件键：写回时用 ${KEY-值}（无冒号，允许显式置空禁用），与 service.conf 现状一致。
const char* kOptionalFileKeys[] = {"OVERLAY_FILE", "MTP_FILE", "VISION_FILE",
                                   "GGUF_MTP_FILE", "GGUF_VISION_FILE"};

bool is_optional_file_key(const std::string& key) {
    for (const char* k : kOptionalFileKeys)
        if (key == k) return true;
    return false;
}

// 环境变量覆盖提示归属的页：0 权重 / 1 上下文 / 2 端口。
int page_of_key(const std::string& k) {
    if (k == "MAX_CONTEXT" || k == "PARALLEL" || k == "KV_POOL_TOKENS" || k == "KV_PAGED" ||
        k == "PREFILL_CHUNK" || k == "ROPE_FACTOR" || k == "KVSNAP_MAX_GB")
        return 1;
    if (k == "ENGINE_HOST" || k == "ENGINE_PORT" || k == "API_HOST" || k == "API_PORT")
        return 2;
    return 0;
}

// 写回 service.conf：注释/空行/未知键原样保留；已知键整行替换为
// KEY="${KEY:-值}"（可选文件键为 KEY="${KEY-值}"），保持 bash 启动器兼容。
// 键缺失时追加到文件末尾。写前备份为 service.conf.bak。
bool save_conf(const std::map<std::string, std::string>& vals, std::string* error) {
    const std::string path = conf_path();
    std::string text;
    read_text_file(path, &text);  // 缺失视为空文件，后面全量追加
    std::map<std::string, bool> written;
    std::string out;
    size_t pos = 0;
    while (pos <= text.size()) {
        size_t eol = text.find('\n', pos);
        const bool last = eol == std::string::npos;
        std::string line = text.substr(pos, last ? std::string::npos : eol - pos);
        pos = last ? text.size() + 1 : eol + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        // 匹配 ^\s*KEY\s*=（KEY 为受管理键）
        size_t i = line.find_first_not_of(" \t");
        bool replaced = false;
        if (i != std::string::npos && line[i] != '#') {
            size_t eq = line.find('=', i);
            if (eq != std::string::npos) {
                std::string key = line.substr(i, eq - i);
                while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) key.pop_back();
                auto it = vals.find(key);
                if (it != vals.end() && !key.empty()) {
                    const char* sep = is_optional_file_key(key) ? "-" : ":-";
                    out += line.substr(0, i) + key + "=\"${" + key + sep + it->second + "}\"";
                    replaced = true;
                    written[key] = true;
                }
            }
        }
        if (!replaced) out += line;
        if (!last) out += '\n';
    }
    for (const auto& kv : vals)
        if (!written[kv.first]) {
            const char* sep = is_optional_file_key(kv.first) ? "-" : ":-";
            out += kv.first + "=\"${" + kv.first + sep + kv.second + "}\"\n";
        }
    CopyFileA(path.c_str(), (g_root + "\\service.conf.bak").c_str(), FALSE);
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        *error = std::string(TR("无法写入 ", "Cannot write ")) + path;
        return false;
    }
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
    return true;
}

// ---- 小工具 -------------------------------------------------------------------

std::string from_w(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()),
                                      nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), &s[0], n,
                        nullptr, nullptr);
    return s;
}

std::string trim_copy(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    const size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// 不退出的数值解析（面板检查用；launch_win.cpp 的 cfg_int/parse_double 会 fail 退出）
bool pn_int(const std::string& s, long* out) {
    if (s.empty()) return false;
    char* end = nullptr;
    const long n = strtol(s.c_str(), &end, 10);
    if (!end || *end) return false;
    *out = n;
    return true;
}

bool pn_double(const std::string& s, double* out) {
    if (s.empty()) return false;
    char* end = nullptr;
    errno = 0;
    const double n = strtod(s.c_str(), &end);
    if (errno || !end || *end || end == s.c_str() || n != n) return false;
    *out = n;
    return true;
}

bool file_size_of(const std::string& path, unsigned long long* size) {
    WIN32_FILE_ATTRIBUTE_DATA d{};
    if (!GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &d)) return false;
    *size = (static_cast<unsigned long long>(d.nFileSizeHigh) << 32) | d.nFileSizeLow;
    return true;
}

// HGN1 头（104B）+ 记录表（160B/条，布局见 src/hgn.h）：只读文件头，
// 返回指定 tensor 的 data_size。用于把磁盘分页的 PLE n-gram 表从权重
// 文件大小里剔除 —— 它按需 gather，不驻留显存（v1 内嵌、v2 独立文件）。
#pragma pack(push, 1)
struct HgnHeader {
    char magic[4];
    uint32_t version, tensor_count, reserved;
    uint64_t records_offset, data_offset, file_size;
    char model_name[64];
};
struct HgnRecord {
    char name[96];
    uint32_t dtype, ndims;
    uint64_t dims[4];
    uint64_t data_offset, data_size, extra;
};
#pragma pack(pop)
static_assert(sizeof(HgnHeader) == 104 && sizeof(HgnRecord) == 160,
              "HGN1 layout must match src/hgn.h");

unsigned long long hgn_tensor_bytes(const std::string& path, const char* name) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return 0;
    HgnHeader h{};
    unsigned long long bytes = 0;
    if (fread(&h, sizeof h, 1, f) == 1 && memcmp(h.magic, "HGN1", 4) == 0 &&
        h.tensor_count <= 4096 &&
        _fseeki64(f, (long long)h.records_offset, SEEK_SET) == 0) {
        for (uint32_t i = 0; i < h.tensor_count; i++) {
            HgnRecord r{};
            if (fread(&r, sizeof r, 1, f) != 1) break;
            if (strncmp(r.name, name, sizeof r.name) == 0) {  // name[96] 定长，可能无 NUL
                bytes = r.data_size;
                break;
            }
        }
    }
    fclose(f);
    return bytes;
}

std::string sniff_magic(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return "";
    char b[4] = {};
    const size_t n = fread(b, 1, 4, f);
    fclose(f);
    return n == 4 ? std::string(b, 4) : "";
}

std::string gib(double bytes) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f", bytes / 1073741824.0);
    return buf;
}

// 绝对路径转项目相对（./...，正斜杠），bash 启动器兼容；不在项目下则原样返回。
std::string make_relative(std::string path) {
    for (auto& c : path)
        if (c == '\\') c = '/';
    std::string root = g_root;
    for (auto& c : root)
        if (c == '\\') c = '/';
    if (path.size() > root.size() + 1 && _strnicmp(path.c_str(), root.c_str(), root.size()) == 0 &&
        path[root.size()] == '/')
        return "." + path.substr(root.size());
    return path;
}

// ---- 面板取值 -----------------------------------------------------------------

enum WeightsVer : int { kVerV1 = 0, kVerV2 = 1, kVerGguf = 2 };

struct PanelVals {
    int weights_ver = kVerV1;
    std::string model, ngram, overlay, mtp, vision, tok;
    bool no_overlay = false, no_mtp = false, no_vision = false;
    std::string gguf, gguf_mtp, gguf_vision;
    bool no_gguf_mtp = false, no_gguf_vision = false;
    std::string max_context, parallel, kv_pool, prefill_chunk;
    bool kv_paged = true;
    std::string rope_factor, rope_original;  // rope_original 无 UI 行，从 conf/环境读
    std::string kvsnap_max_gb;
    std::string engine_host, engine_port, api_host, api_port;
};

PanelVals panel_vals_from_conf() {
    PanelVals v;
    const std::string model_dir = cfg("MODEL_DIR", "models");
    v.model = cfg("MODEL_FILE", model_dir + "\\heretic.hgn");
    v.ngram = cfg("NGRAM_FILE", v.model);
    const bool ngram_follows = v.ngram == v.model;
    if (ngram_follows) v.ngram.clear();  // 空 = 跟随主模型
    v.overlay = cfg_optional("OVERLAY_FILE", "");
    v.no_overlay = v.overlay.empty();
    v.mtp = cfg_optional("MTP_FILE", model_dir + "\\heretic-mtp.hgn");
    v.no_mtp = v.mtp.empty();
    v.vision = cfg_optional("VISION_FILE", model_dir + "\\heretic-vision.hgn");
    v.no_vision = v.vision.empty();
    v.gguf = cfg("GGUF_FILE",
                 model_dir + "\\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf");
    v.gguf_mtp =
        cfg_optional("GGUF_MTP_FILE", model_dir + "\\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf");
    v.no_gguf_mtp = v.gguf_mtp.empty();
    v.gguf_vision = cfg_optional("GGUF_VISION_FILE", model_dir + "\\mmproj-BF16.gguf");
    v.no_gguf_vision = v.gguf_vision.empty();
    v.tok = cfg("TOKENIZER_DIR", model_dir + "\\tokenizer");
    v.max_context = cfg("MAX_CONTEXT", "262144");
    v.parallel = cfg("PARALLEL", "1");
    v.kv_pool = cfg("KV_POOL_TOKENS", "0");
    v.kv_paged = cfg("KV_PAGED", "1") != "0";
    v.prefill_chunk = cfg("PREFILL_CHUNK", "0");
    v.rope_factor = cfg("ROPE_FACTOR", "1");
    v.rope_original = cfg("ROPE_ORIGINAL_CTX", "262144");
    v.kvsnap_max_gb = cfg("KVSNAP_MAX_GB", "20");
    v.engine_host = cfg("ENGINE_HOST", "127.0.0.1");
    v.engine_port = cfg("ENGINE_PORT", "8730");
    v.api_host = cfg("API_HOST", "0.0.0.0");
    v.api_port = cfg("API_PORT", "8731");
    // 版本：标记优先；无标记时 NGRAM_FILE 解析后与 MODEL_FILE 不同 → V2，否则 V1
    const std::string marker = conf_weights_marker();
    if (marker == "v1") v.weights_ver = kVerV1;
    else if (marker == "v2") v.weights_ver = kVerV2;
    else if (marker == "gguf") v.weights_ver = kVerGguf;
    else v.weights_ver = ngram_follows ? kVerV1 : kVerV2;
    return v;
}

// 面板值 → conf 键值。只写所选版本相关的权重键，其余保持 conf 原值不动。
// V1：NGRAM_FILE 写字面量 $MODEL_FILE（bash 默认展开 = 与主模型同文件）。
// V2：OVERLAY_FILE 写空（${KEY-} 无冒号风格，空=禁用）；NGRAM_FILE 为独立文件。
// GGUF：hgn 各键不动，只写 GGUF_* 与公共键。
std::map<std::string, std::string> panel_vals_to_conf(const PanelVals& v) {
    std::map<std::string, std::string> m;
    if (v.weights_ver == kVerGguf) {
        m["GGUF_FILE"] = v.gguf;
        m["GGUF_MTP_FILE"] = v.no_gguf_mtp ? "" : v.gguf_mtp;
        m["GGUF_VISION_FILE"] = v.no_gguf_vision ? "" : v.gguf_vision;
    } else {
        m["MODEL_FILE"] = v.model;
        if (v.weights_ver == kVerV2) {
            m["NGRAM_FILE"] = v.ngram;
            m["OVERLAY_FILE"] = "";
        } else {
            m["NGRAM_FILE"] = "$MODEL_FILE";
            m["OVERLAY_FILE"] = v.no_overlay ? "" : v.overlay;
        }
        m["MTP_FILE"] = v.no_mtp ? "" : v.mtp;
        m["VISION_FILE"] = v.no_vision ? "" : v.vision;
    }
    std::string tok = v.tok;
    while (!tok.empty() && (tok.back() == '\\' || tok.back() == '/')) tok.pop_back();
    m["TOKENIZER_DIR"] = tok;
    m["MAX_CONTEXT"] = v.max_context;
    m["PARALLEL"] = v.parallel;
    m["KV_POOL_TOKENS"] = v.kv_pool;
    m["KV_PAGED"] = v.kv_paged ? "1" : "0";
    m["PREFILL_CHUNK"] = v.prefill_chunk;
    m["ROPE_FACTOR"] = v.rope_factor;
    m["KVSNAP_MAX_GB"] = v.kvsnap_max_gb;
    m["ENGINE_HOST"] = v.engine_host;
    m["ENGINE_PORT"] = v.engine_port;
    m["API_HOST"] = v.api_host;
    m["API_PORT"] = v.api_port;
    return m;
}

// ---- 环境检查（按页拆分） -------------------------------------------------------

struct CheckLines {
    std::string text;  // \r\n 分隔，前缀 [通过]/[失败]/[警告]/[提示]
    std::vector<std::string> fatals;  // 失败行正文（第 4 页汇总用）
    bool fatal = false;
};

// 级别用枚举判定，前缀文本按语言渲染（[失败] / [FAIL] 等）。
enum Lv : int { kLvPass, kLvFail, kLvWarn, kLvHint };

const char* lv_text(Lv lv) {
    switch (lv) {
        case kLvPass: return TR("通过", "PASS");
        case kLvFail: return TR("失败", "FAIL");
        case kLvWarn: return TR("警告", "WARN");
        default: return TR("提示", "HINT");
    }
}

void check_line(CheckLines& c, Lv lv, const std::string& msg) {
    c.text += "[" + std::string(lv_text(lv)) + "] " + msg + "\r\n";
    if (lv == kLvFail) {
        c.fatal = true;
        c.fatals.push_back(msg);
    }
}

void check_weight(CheckLines& c, const char* label, const std::string& path,
                  bool required, bool warn_skip) {
    if (!file_exists(path)) {
        if (warn_skip)
            check_line(c, kLvWarn, std::string(label) + TR("：找不到 ", ": ") + path +
                                       TR("，引擎将跳过", " not found; engine will skip"));
        else if (required)
            check_line(c, kLvFail,
                       std::string(label) + TR("：找不到 ", ": not found: ") + path);
        return;
    }
    unsigned long long size = 0;
    file_size_of(path, &size);
    if (size < 100ull * 1024 * 1024) {
        check_line(c, kLvFail, std::string(label) + TR("：文件异常小（", ": suspiciously small file (") +
                                   std::to_string(size) + TR(" B）", " B): ") + path);
        return;
    }
    // magic 嗅探：.hgn → HGN1（src/hgn.h），.gguf → GGUF；其它扩展名跳过
    const size_t dot = path.find_last_of('.');
    const std::string ext =
        dot == std::string::npos ? "" : path.substr(dot + 1);
    if (_stricmp(ext.c_str(), "hgn") == 0 || _stricmp(ext.c_str(), "gguf") == 0) {
        const std::string want = _stricmp(ext.c_str(), "hgn") == 0 ? "HGN1" : "GGUF";
        const std::string magic = sniff_magic(path);
        if (magic != want) {
            check_line(c, kLvFail,
                       std::string(label) + TR("：不是有效的 .", ": not a valid .") + ext +
                           TR(" 文件（magic 非 ", " file (magic != ") + want + TR("）", "): ") +
                           path);
            return;
        }
        check_line(c, kLvPass,
                   std::string(label) + TR("：", ": ") + path + TR("（", " (") + gib((double)size) +
                       TR(" GiB，", " GiB, ") + want + TR("）", ")"));
    } else {
        check_line(c, kLvWarn, std::string(label) + TR("：", ": ") + path + TR("（", " (") +
                                   gib((double)size) +
                                   TR(" GiB，扩展名未知，未做格式嗅探）",
                                      " GiB, unknown extension; no format sniffing)"));
    }
}

// 第 1 页：权重（按所选版本检查对应的文件集合 + tokenizer）。
void check_weights_page(CheckLines& c, const PanelVals& v) {
    if (v.weights_ver == kVerGguf) {
        check_weight(c, TR("GGUF 主分片", "GGUF main shard"), v.gguf, true, false);
        if (v.gguf.find("-00001-of-") != std::string::npos)
            check_line(c, kLvPass,
                       TR("GGUF 分片命名：填的是第 1 个分片，其余分片须在同一目录",
                          "GGUF shards: the first shard is entered; keep the rest in the same "
                          "directory"));
        else
            check_line(c, kLvWarn,
                       TR("GGUF_FILE 文件名不含 -00001-of-XXXXX 分片命名；多分片模型请填第 1 "
                          "个分片",
                          "GGUF_FILE name lacks the -00001-of-XXXXX pattern; for multi-shard "
                          "models enter the first shard"));
        if (v.no_gguf_mtp)
            check_line(c, kLvPass, TR("GGUF MTP：不使用", "GGUF MTP: not used"));
        else
            check_weight(c, "GGUF MTP", v.gguf_mtp, true, false);
        if (v.no_gguf_vision)
            check_line(c, kLvPass, TR("GGUF 视觉塔：不使用", "GGUF vision tower: not used"));
        else
            check_weight(c, TR("GGUF 视觉塔", "GGUF vision tower"), v.gguf_vision, true, false);
        check_line(c, kLvWarn,
                   TR("Windows 启动器只支持 hgn 权重；GGUF 配置仅保存（Linux 用 bash "
                      "start_gguf.sh 启动）",
                      "The Windows launcher supports hgn weights only; the GGUF config is saved "
                      "only (on Linux use bash start_gguf.sh)"));
    } else {
        check_weight(c, TR("主模型", "Main model"), v.model, true, false);
        if (v.weights_ver == kVerV2) {
            if (v.ngram.empty())
                check_line(c, kLvFail,
                           TR("n-gram 表：V2 权重必须填独立文件",
                              "n-gram table: V2 weights require a separate file"));
            else if (same_file(v.model, v.ngram))
                check_line(c, kLvFail,
                           TR("n-gram 表：V2 必须是与主模型不同的独立文件",
                              "n-gram table: V2 needs a file separate from the main model"));
            else
                check_weight(c, TR("n-gram 表", "n-gram table"), v.ngram, true, false);
            check_line(c, kLvPass,
                       TR("overlay：V2 不使用（保存时置空，引擎本就会跳过）",
                          "overlay: unused with V2 (cleared on save; the engine skips it anyway)"));
        } else {
            check_line(c, kLvPass,
                       TR("n-gram 表：与主模型同文件（V1 内置）",
                          "n-gram table: same file as the main model (built into V1)"));
            if (v.no_overlay)
                check_line(c, kLvPass, TR("overlay：不使用", "overlay: not used"));
            else
                check_weight(c, "overlay", v.overlay, false, true);
        }
        if (v.no_mtp)
            check_line(c, kLvPass, TR("MTP 草稿：不使用", "MTP draft: not used"));
        else
            check_weight(c, TR("MTP 草稿", "MTP draft"), v.mtp, true, false);
        if (v.no_vision)
            check_line(c, kLvPass, TR("视觉塔：不使用", "Vision tower: not used"));
        else
            check_weight(c, TR("视觉塔", "Vision tower"), v.vision, true, false);
    }
    if (!dir_exists(v.tok))
        check_line(c, kLvFail, TR("tokenizer：目录不存在 ", "tokenizer: directory missing: ") + v.tok);
    else if (!file_exists(v.tok + "\\tokenizer.json"))
        check_line(c, kLvFail,
                   TR("tokenizer：找不到 ", "tokenizer: missing ") + v.tok + "\\tokenizer.json");
    else
        check_line(c, kLvPass, TR("tokenizer：", "tokenizer: ") + v.tok + "\\tokenizer.json");
}

// 第 2 页：上下文与并发、YaRN 的数值校验（与启动流程同一套规则）。
void check_context_page(CheckLines& c, const PanelVals& v) {
    // 说明行（不参与判定，恒在列表最上方）
    c.text += TR("· 单请求上限：一条请求的输入+输出最大 token 数，模型原生 262144。\r\n",
                 "· Per-request context: max input+output tokens per request; the model is "
                 "natively 262144.\r\n");
    c.text += TR("· 并发总池：所有并发共享的 KV 缓存总量（类似 vLLM 的 KV cache），单请求上限不能超过它；\r\n",
                 "· KV pool: total KV cache shared by all concurrent streams (like vLLM's KV "
                 "cache); a per-request context must fit within it;\r\n");
    c.text += TR("· 并发合计占用超过总池时新请求会报错。默认跟随单请求上限；Windows 最大 512K。\r\n",
                 "· when streams together exceed the pool, new requests error out. Defaults to "
                 "the per-request cap; max 512K on Windows.\r\n");
    c.text += TR("· YaRN：单请求上限超过原生 262144 时开启，扩展倍数按上限自动推导。\r\n",
                 "· YaRN: enable when the per-request cap exceeds native 262144; the scale "
                 "factor is derived from the cap automatically.\r\n");
    long maxctx = 0, parallel = 1, kv_pool = 0, prefill = 0, rope_orig = 0;
    double factor = 1;
    bool nums_ok = true;
    auto need_int = [&](const char* name, const std::string& s, long lo, long hi, long* out) {
        if (!pn_int(s, out) || *out < lo || *out > hi) {
            check_line(c, kLvFail, std::string(name) + TR(" 必须是 ", " must be an integer ") +
                                       std::to_string(lo) + "-" + std::to_string(hi) +
                                       TR(" 的整数（当前为 \"", " (current: \"") + s +
                                       TR("\"）", "\")"));
            nums_ok = false;
            return false;
        }
        return true;
    };
    need_int("MAX_CONTEXT", v.max_context, 1024, 1 << 20, &maxctx);
    need_int("PARALLEL", v.parallel, 1, 8, &parallel);
    need_int("KV_POOL_TOKENS", v.kv_pool, 0, 1 << 24, &kv_pool);
    need_int("PREFILL_CHUNK", v.prefill_chunk, 0, 1 << 20, &prefill);
    long kvsnap = 20;
    need_int("KVSNAP_MAX_GB", v.kvsnap_max_gb, 0, 1 << 16, &kvsnap);
    // ROPE_ORIGINAL_CTX 无 UI 行，但校验照做（conf 里填坏时启动器同样会拒）
    need_int("ROPE_ORIGINAL_CTX", v.rope_original, 1, 1 << 30, &rope_orig);
    if (!pn_double(v.rope_factor, &factor) || factor < 1) {
        check_line(c, kLvFail,
                   TR("ROPE_FACTOR 必须 >=1（当前为 \"", "ROPE_FACTOR must be >=1 (current: \"") +
                       v.rope_factor + TR("\"）", "\")"));
        nums_ok = false;
    }
    if (!nums_ok) return;
    if (parallel > 1 && !v.kv_paged)
        check_line(c, kLvFail, TR("PARALLEL>1 需要 KV_PAGED=1", "PARALLEL>1 requires KV_PAGED=1"));
    if (factor > 1 && maxctx > factor * rope_orig)
        // 勾选 YaRN 时 factor 按上限自动推导，理论上不会触发；留作兜底
        check_line(c, kLvFail,
                   "MAX_CONTEXT=" + std::to_string(maxctx) +
                       TR(" 超出 YaRN 上限 ROPE_FACTOR×ROPE_ORIGINAL_CTX=",
                          " exceeds the YaRN limit ROPE_FACTOR×ROPE_ORIGINAL_CTX=") +
                       std::to_string((long)(factor * rope_orig)) +
                       TR("；调小 MAX_CONTEXT 或开启 YaRN（并发容量用 KV_POOL_TOKENS 扩）",
                          "; lower MAX_CONTEXT or enable YaRN (grow capacity via "
                          "KV_POOL_TOKENS)"));
    if (factor <= 1 && maxctx > rope_orig)
        check_line(c, kLvWarn, TR("未开启 YaRN 且 MAX_CONTEXT=", "YaRN is off and MAX_CONTEXT=") +
                                   std::to_string(maxctx) + TR(" 超过原生 ", " exceeds native ") +
                                   std::to_string(rope_orig) +
                                   TR("，超出部分的位置编码未验证；建议开启 YaRN",
                                      "; positions beyond it are unverified; consider enabling "
                                      "YaRN"));
    // 单请求不能超过 KV 缓存总量（vLLM 语义；kv_pool=0 = 跟随上限，不触发）
    if (kv_pool > 0 && maxctx > kv_pool)
        check_line(c, kLvFail, TR("单请求上限 MAX_CONTEXT=", "Per-request context MAX_CONTEXT=") +
                                   std::to_string(maxctx) +
                                   TR(" 不能大于并发总池 KV_POOL_TOKENS=",
                                      " exceeds the KV pool KV_POOL_TOKENS=") +
                                   std::to_string(kv_pool) +
                                   TR("；调小上限或调大总池", "; lower the cap or grow the pool"));
    // Windows 设备内存是硬上限 95 GiB 的 arena：512K 池（45,696 B/token ≈ 22.3 GiB）
    // 已是权重 62.1 GiB + prefill 工作区 6.9 GiB 之后放得下的上限。有效值 = max(池, 上限)
    // （0 = 跟随，上限超 512K 时跟随逻辑同样会把池顶超）。
    const long pool_eff = kv_pool > maxctx ? kv_pool : maxctx;
    if (pool_eff > 524288)
        check_line(c, kLvFail,
                   TR("并发总池有效值 ", "Effective KV pool ") + std::to_string(pool_eff) +
                       TR(" 超过 512K（524288 token）：Windows 95 GiB 显存 arena"
                          " 放不下更大的池；调小 MAX_CONTEXT 或 KV_POOL_TOKENS",
                          " exceeds 512K (524288 tokens): the 95 GiB Windows VRAM arena cannot "
                          "fit a larger pool; lower MAX_CONTEXT or KV_POOL_TOKENS"));
    if (!c.fatal)
        check_line(c, kLvPass,
                   TR("数值校验：上下文/并发/YaRN 参数合法",
                      "Numeric checks: context / concurrency / YaRN parameters valid"));
}

// 第 3 页：监听地址与端口。service_running 时占用检查降级为提示。
void check_ports_page(CheckLines& c, const PanelVals& v, bool service_running) {
    // 说明行（不参与判定，恒在列表最上方）
    c.text += TR("· 引擎是内部行协议，由 API 自动桥接，无需直连；浏览器/客户端连 API 端口。\r\n",
                 "· The engine speaks an internal line protocol bridged by the API; browsers/"
                 "clients connect to the API port.\r\n");
    c.text += TR("· 监听地址：127.0.0.1 = 仅本机（引擎推荐），0.0.0.0 = 对外开放。\r\n",
                 "· Bind address: 127.0.0.1 = local only (recommended for the engine), "
                 "0.0.0.0 = open to the network.\r\n");
    sockaddr_in tmp{};
    if (v.engine_host.empty() || !engine_net::address(v.engine_host, 1, &tmp))
        check_line(c, kLvFail,
                   TR("ENGINE_HOST 必须是 IPv4 点分地址，不支持主机名（当前为 \"",
                      "ENGINE_HOST must be an IPv4 dotted address; hostnames are not supported "
                      "(current: \"") +
                       v.engine_host + TR("\"）", "\")"));
    else if (v.engine_host != "127.0.0.1")
        check_line(c, kLvWarn, "ENGINE_HOST=" + v.engine_host +
                                   TR(" 会把引擎行协议暴露到网络，仅确需外部直连时这样设",
                                      " exposes the engine's line protocol to the network; set "
                                      "this only if external direct access is required"));
    if (v.api_host.empty() || !engine_net::address(v.api_host, 1, &tmp))
        check_line(c, kLvFail,
                   TR("API_HOST 必须是 IPv4 点分地址（当前为 \"",
                      "API_HOST must be an IPv4 dotted address (current: \"") +
                       v.api_host + TR("\"）", "\")"));
    long eport = 0, aport = 0;
    bool nums_ok = true;
    auto need_int = [&](const char* name, const std::string& s, long* out) {
        if (!pn_int(s, out) || *out < 1 || *out > 65535) {
            check_line(c, kLvFail, std::string(name) + TR(" 必须是 1-65535 的整数（当前为 \"",
                                                          " must be an integer 1-65535 (current: \"") +
                                       s + TR("\"）", "\")"));
            nums_ok = false;
        }
    };
    need_int("ENGINE_PORT", v.engine_port, &eport);
    need_int("API_PORT", v.api_port, &aport);
    if (nums_ok) {
        if (eport == aport) {
            check_line(c, kLvFail,
                       TR("ENGINE_PORT 与 API_PORT 必须不同", "ENGINE_PORT and API_PORT must differ"));
        } else if (service_running) {
            check_line(c, kLvHint, TR("端口 ", "Ports ") + std::to_string(eport) + "/" +
                                       std::to_string(aport) +
                                       TR("：由当前服务占用（正常）", ": held by the running service (normal)"));
        } else {
            if (!port_free((int)eport))
                check_line(c, kLvFail,
                           TR("端口 ", "Port ") + std::to_string(eport) + TR(" 已被占用", " is in use"));
            if (!port_free((int)aport))
                check_line(c, kLvFail,
                           TR("端口 ", "Port ") + std::to_string(aport) + TR(" 已被占用", " is in use"));
            if (port_free((int)eport) && port_free((int)aport))
                check_line(c, kLvPass, TR("端口 ", "Ports ") + std::to_string(eport) + "/" +
                                           std::to_string(aport) + TR(" 空闲", " are free"));
        }
    }
    if (!c.fatal) check_line(c, kLvPass, TR("监听地址与端口合法", "Bind addresses and ports are valid"));
}

// GGUF 分片（name-00001-of-00004.gguf …）合计大小；单文件/找不到时分片数为 0/1。
unsigned long long gguf_shards_size(const std::string& first, int* count) {
    *count = 0;
    unsigned long long total = 0;
    const size_t of = first.find("-of-");
    size_t dash = std::string::npos;
    if (of != std::string::npos && of > 0) dash = first.rfind('-', of - 1);
    if (dash == std::string::npos) {
        unsigned long long s = 0;
        if (file_size_of(first, &s)) {
            total = s;
            *count = 1;
        }
        return total;
    }
    const std::string pattern = first.substr(0, dash) + "-*" + first.substr(of);
    WIN32_FIND_DATAA fd{};
    HANDLE hf = FindFirstFileA(pattern.c_str(), &fd);
    if (hf == INVALID_HANDLE_VALUE) return 0;
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            total += (static_cast<unsigned long long>(fd.nFileSizeHigh) << 32) | fd.nFileSizeLow;
            ++*count;
        }
    } while (FindNextFileA(hf, &fd));
    FindClose(hf);
    return total;
}

// 第 4 页：显存环境检查完整报告。service_running 时余量结论降级为对照提示。
void check_vram_page(CheckLines& c, const PanelVals& v, bool service_running) {
    // 显存：动态加载 HIP（启动器不链 amdhip64，DLL 缺失只降级不阻塞）
    size_t hip_free = 0, hip_total = 0;
    bool hip_ok = false;
    // 全路径 + LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR：相对路径 "build\..." 跟工作目录
    // 走（快捷方式/其他 cwd 下加载失败），且 amdhip64_7.dll 的依赖 DLL 也在
    // build\ 下，标准搜索顺序找不到——该 flag 把 DLL 自身目录纳入依赖搜索。
    const std::string hip_dll = g_root + "\\build\\amdhip64_7.dll";
    if (HMODULE hip = LoadLibraryExA(hip_dll.c_str(), nullptr,
                                     LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                         LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)) {
        auto p_init = reinterpret_cast<int (*)(unsigned int)>(GetProcAddress(hip, "hipInit"));
        auto p_info =
            reinterpret_cast<int (*)(size_t*, size_t*)>(GetProcAddress(hip, "hipMemGetInfo"));
        if (p_init && p_info && p_init(0) == 0 && p_info(&hip_free, &hip_total) == 0)
            hip_ok = true;
    }
    if (hip_ok)
        check_line(c, kLvHint, TR("HIP 显存：总量 ", "HIP VRAM: total ") + gib((double)hip_total) +
                                   TR(" GiB，当前可用 ", " GiB, currently free ") +
                                   gib((double)hip_free) + " GiB");
    else
        check_line(c, kLvWarn,
                   TR("无法查询显存（build\\amdhip64_7.dll 缺失或驱动异常），"
                      "下面的估算请自行对照",
                      "Cannot query VRAM (build\\amdhip64_7.dll missing or driver problem); "
                      "compare the estimates below manually"));

    // 数值（第 2 页）解析失败时按默认粗估，不阻塞报告
    long maxctx = 262144, parallel = 1, kv_pool = 0, prefill = 0;
    pn_int(v.max_context, &maxctx);
    pn_int(v.parallel, &parallel);
    pn_int(v.kv_pool, &kv_pool);
    pn_int(v.prefill_chunk, &prefill);
    if (maxctx < 1) maxctx = 262144;
    if (parallel < 1) parallel = 1;
    if (kv_pool < 0) kv_pool = 0;
    if (prefill < 0) prefill = 0;
    // GDEC_PREFILL_CHUNK(conf 里的 GDEC_* 会导出到引擎环境)才是真正的 prefill
    // 分段;它比 PREFILL_CHUNK 大时以它为准,工作区估算才贴合引擎实际分配。
    if (const auto it = g_conf.find("GDEC_PREFILL_CHUNK"); it != g_conf.end()) {
        long g = 0;
        if (pn_int(it->second, &g) && g > prefill) prefill = g;
    }

    // 权重文件总大小：按第 1 页所选版本的文件集合，去重
    unsigned long long weights = 0;
    if (v.weights_ver == kVerGguf) {
        int shards = 0;
        weights += gguf_shards_size(v.gguf, &shards);
        if (shards > 0)
            check_line(c, kLvHint, TR("GGUF 权重：", "GGUF weights: ") + std::to_string(shards) +
                                       TR(" 个分片合计 ", " shards totaling ") +
                                       gib((double)weights) + " GiB");
        else
            check_line(c, kLvHint,
                       TR("GGUF 主分片找不到，权重按 0 估算",
                          "GGUF first shard not found; weights estimated as 0"));
        const std::string side[] = {v.no_gguf_mtp ? "" : v.gguf_mtp,
                                    v.no_gguf_vision ? "" : v.gguf_vision};
        unsigned long long extra = 0;
        for (const std::string& f : side) {
            unsigned long long s = 0;
            if (!f.empty() && file_size_of(f, &s)) extra += s;
        }
        if (extra) {
            weights += extra;
            check_line(c, kLvHint, TR("GGUF sidecar（MTP/视觉塔）：", "GGUF sidecars (MTP/vision): ") +
                                       gib((double)extra) + " GiB");
        }
        check_line(c, kLvHint,
                   TR("GGUF 配置在 Windows 不启动，本页估算仅供 Linux 参考",
                      "GGUF configs do not start on Windows; estimates here are for Linux reference"));
    } else {
        // n-gram 表是磁盘分页的（引擎按需 gather，只有小 staging 驻留），
        // 单列且不计入显存估算：v2 是独立文件（直接不算进 weights）；
        // v1 内嵌在主权重文件里，从 HGN 头读出表大小后从驻留口径里减掉。
        unsigned long long ngram_bytes = 0;
        if (v.weights_ver == kVerV2 && !v.ngram.empty() && file_exists(v.ngram) &&
            !same_file(v.ngram, v.model))
            file_size_of(v.ngram, &ngram_bytes);
        const unsigned long long ngram_embedded =
            (v.weights_ver == kVerV1 && file_exists(v.model))
                ? hgn_tensor_bytes(v.model, "layers.1.ple.ngram_embedding.weight")
                : 0;
        const std::string files[] = {v.model,
                                     v.weights_ver == kVerV2 ? "" : (v.no_overlay ? "" : v.overlay),
                                     v.no_mtp ? "" : v.mtp, v.no_vision ? "" : v.vision};
        std::vector<std::string> uniq;
        for (const std::string& f : files) {
            if (f.empty() || !file_exists(f)) continue;
            bool dup = false;
            for (const std::string& u : uniq)
                if (same_file(f, u)) {
                    dup = true;
                    break;
                }
            if (!dup) uniq.push_back(f);
        }
        for (const std::string& u : uniq) {
            unsigned long long s = 0;
            if (file_size_of(u, &s)) weights += s;
        }
        check_line(c, kLvHint, TR("权重文件：", "Weight files: ") + std::to_string(uniq.size()) +
                                   TR(" 个合计 ", " totaling ") + gib((double)weights) + " GiB");
        if (ngram_bytes)
            check_line(c, kLvHint,
                       TR("n-gram 表：", "n-gram table: ") + gib((double)ngram_bytes) +
                           TR(" GiB，磁盘分页按需读取，不驻留显存",
                              " GiB, disk-paged on demand, not VRAM-resident"));
        if (ngram_embedded)
            check_line(c, kLvHint,
                       TR("n-gram 表（内嵌）：", "n-gram table (embedded): ") +
                           gib((double)ngram_embedded) +
                           TR(" GiB，磁盘分页按需读取，不驻留显存",
                              " GiB, disk-paged on demand, not VRAM-resident"));
        // 显存估算按驻留口径：减掉内嵌表（v2 的独立表本就未计入）
        if (ngram_embedded < weights) weights -= ngram_embedded;
    }

    // 显存估算（系数来自 src/gpu/parts/40_model.inc 分配代码核算，2026-10：
    // 45,696 B/token = 12 QSA 层 + 1 MTP 层的 BF16 KV + BTV 转置 V +
    // index f16 twin + d_iscores；512K 实机 ≈ 22.3 GiB 吻合）。
    const double kKvBytesPerToken = 45696.0;
    // 0 = 跟随 MAX_CONTEXT；std::max 与 windows.h 的 max 宏冲突，手写比较
    const double pool_tokens = (double)(kv_pool > maxctx ? kv_pool : maxctx);
    const double kv_bytes = pool_tokens * kKvBytesPerToken;
    const double chunk = prefill > 0 ? (double)prefill : 8192.0;
    const double ws_bytes = chunk / 8192.0 * 6.9 * 1073741824.0;  // 8192≈6.9 GiB 实测
    const double gdn_bytes = parallel * 0.12 * 1073741824.0;      // 每路 GDN 状态
    const double total = (double)weights + kv_bytes + ws_bytes + gdn_bytes;
    check_line(c, kLvHint, TR("并发总池：", "KV pool: ") + std::to_string((long)pool_tokens) +
                               TR(" token × 44.6 KiB ≈ ", " tokens × 44.6 KiB ≈ ") + gib(kv_bytes) +
                               " GiB");
    check_line(c, kLvHint, TR("prefill 工作区：chunk ", "Prefill workspace: chunk ") +
                               std::to_string((long)chunk) + " ≈ " + gib(ws_bytes) + " GiB");
    check_line(c, kLvHint, TR("并发 GDN 状态：", "Parallel GDN state: ") +
                               std::to_string(parallel) + TR(" 路 ≈ ", " streams ≈ ") +
                               gib(gdn_bytes) + " GiB");
    if (!hip_ok) {
        // 查不到显存也要给出估算总量 —— 它不依赖 HIP 查询，只有对比结论需要
        check_line(c, kLvHint, TR("显存估算 ≈ ", "VRAM estimate ≈ ") + gib(total) +
                                   TR(" GiB（显存查询失败，请自行对照）",
                                      " GiB (VRAM query failed; compare manually)"));
    } else if (!service_running) {
        if (total > (double)hip_free)
            check_line(c, kLvFail, TR("预计占用 ", "Estimated use ") + gib(total) +
                                       TR(" GiB 超过当前可用 ", " GiB exceeds the free ") +
                                       gib((double)hip_free) + TR(" GiB，启动会 OOM", " GiB; startup would OOM"));
        else if ((double)hip_free - total < 4.0 * 1073741824.0)
            check_line(c, kLvWarn, TR("预计占用 ", "Estimated use ") + gib(total) +
                                       TR(" GiB，余量不足 4 GiB", " GiB, less than 4 GiB of headroom"));
        else if (total > 95.0 * 1073741824.0)
            check_line(c, kLvWarn, TR("预计占用 ", "Estimated use ") + gib(total) +
                                       TR(" GiB 超过引擎 arena 上限 95 GiB，超出部分回退 hipMalloc",
                                          " GiB exceeds the 95 GiB engine arena cap; the excess "
                                          "falls back to hipMalloc"));
        else
            check_line(c, kLvPass, TR("显存估算 ≈ ", "VRAM estimate ≈ ") + gib(total) +
                                       TR(" / 可用 ", " / free ") + gib((double)hip_free) + " GiB");
    } else if (hip_ok) {
        check_line(c, kLvHint, TR("服务运行中：估算 ≈ ", "Service running: estimate ≈ ") + gib(total) +
                                   TR(" GiB，仅对照总量 ", " GiB, compared against total ") +
                                   gib((double)hip_total) + " GiB");
    }
}

struct AllChecks {
    CheckLines weights, context, ports, vram;
    bool fatal() const { return weights.fatal || context.fatal || ports.fatal || vram.fatal; }
};

AllChecks panel_checks(const PanelVals& v, bool service_running) {
    AllChecks a;
    check_weights_page(a.weights, v);
    check_context_page(a.context, v);
    check_ports_page(a.ports, v, service_running);
    check_vram_page(a.vram, v, service_running);
    // 环境变量覆盖提示，归入对应页
    for (const char* key : kManagedKeys) {
        const char* e = getenv(key);
        if (!e || !*e) continue;
        CheckLines& dst = page_of_key(key) == 1   ? a.context
                          : page_of_key(key) == 2 ? a.ports
                                                  : a.weights;
        check_line(dst, kLvWarn, std::string(key) +
                                     TR(" 正被环境变量覆盖（=\"", " is overridden by an environment variable (=\"") +
                                     e + TR("\"），面板改动不生效", "\"); panel changes will not take effect"));
    }
    return a;
}

// ---- 面板窗口 -----------------------------------------------------------------

// PanelResult（kPanelStart/kPanelSave/kPanelCancel）在 launch_win.cpp 前向声明
// 段中完整定义（report()/on_menu() 在本文件之前就要用到枚举值）。

enum PanelId : int {
    kBtnStart = 100, kBtnSave, kBtnCancel, kBtnRecheck,
    kTab1, kTab2, kTab3, kTab4,
    kWVer1, kWVer2, kWVer3,
    kEdModel, kEdNgram, kEdOverlay, kEdMtp, kEdVision, kEdTok,
    kEdGguf, kEdGgufMtp, kEdGgufVision,
    kBrModel, kBrNgram, kBrOverlay, kBrMtp, kBrVision,
    kBrGguf, kBrGgufMtp, kBrGgufVision,
    kOffOverlay, kOffMtp, kOffVision, kOffGgufMtp, kOffGgufVision,
    kEdMaxctx, kCbParallel, kEdKvPool, kCkKvPaged, kEdPrefill,
    kCkYarn, kEdKvsnap,
    kEdEHost, kEdEPort, kEdAHost, kEdAPort,
    kEdOut1, kEdOut2, kEdOut3, kEdReport, kEdSummary, kBtnLang,
};

struct FileRow {
    HWND label = nullptr, off = nullptr, edit = nullptr, browse = nullptr;
};

struct PanelCtl {
    HWND tabs[4] = {};
    HWND wver[3] = {};
    FileRow row_model, row_ngram, row_overlay, row_mtp, row_vision;
    FileRow row_gguf, row_gguf_mtp, row_gguf_vision, row_tok;
    HWND tok_hint = nullptr, ver_note = nullptr;
    HWND maxctx = nullptr, parallel = nullptr, kv_pool = nullptr, kv_paged = nullptr,
         prefill = nullptr;
    HWND yarn = nullptr;  // "YaRN 长上下文扩展" 勾选框（ROPE_FACTOR 自动推导，无编辑行）
    HWND kvsnap = nullptr;
    HWND ehost = nullptr, eport = nullptr, ahost = nullptr, aport = nullptr;
    HWND out1 = nullptr, out2 = nullptr, out3 = nullptr, report = nullptr, summary = nullptr;
    HWND start = nullptr;
};
PanelCtl g_pc;
HWND g_panel = nullptr;
PanelResult g_panel_result = kPanelCancel;
bool g_panel_done = false;
bool g_panel_running = false;       // 服务运行中打开（启动按钮禁用）
bool g_panel_allow_start = true;    // panel_run 的 allow_start
int g_weights_ver = kVerV1;         // 当前显示的权重版本（布局用）
bool g_pool_touched = false;        // 用户手动改过并发总池框（false = 跟随单请求上限显示）
bool g_pool_guard = false;          // 程序自己在写总池框（防 EN_CHANGE 误置 touched）
AllChecks g_last_checks;
std::vector<HWND> g_page_ctls[4];   // 每页的控件句柄（切页时整体 ShowWindow）

// 布局常量：客户区 644×504，内容右缘 636。
// label 列要装下 "单请求上限 (token)"（高 DPI 下约 185px），故 kLblW=190。
const int kCliW = 644, kCliH = 504;
const int kLblX = 12, kLblW = 190, kEdX = 202;
const int kOffW = 88;                 // "不使用" checkbox（勾选框 + 3 个汉字）
const int kEdPlainW = 354;            // 无 checkbox 文件行的 EDIT 宽（202..556）
const int kBrX = 564, kBrW = 72;      // 浏览按钮
const int kHintX = 328, kHintW = 308; // 第 2 页行尾提示（edit 202+120=322 右侧）
const int kListBottom = 452;          // 各页检查区下缘（底部按钮行在其下）

HWND add_ctrl(HWND parent, const wchar_t* cls, DWORD style, int x, int y, int w, int h,
              const wchar_t* text, int id) {
    HWND c = CreateWindowExW(0, cls, text, WS_CHILD | WS_VISIBLE | style, x, y, w, h, parent,
                             reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                             GetModuleHandleW(nullptr), nullptr);
    SendMessageW(c, WM_SETFONT, reinterpret_cast<WPARAM>(GetStockObject(DEFAULT_GUI_FONT)), TRUE);
    return c;
}

HWND add_label(HWND parent, int x, int y, const wchar_t* text, int w = kLblW) {
    return add_ctrl(parent, L"Static", SS_LEFTNOWORDWRAP, x, y + 3, w, 18, text, 0);
}

// 建控件并登记进某页（page=-1 表示恒可见，不参与切页）
HWND pctl(int page, HWND parent, const wchar_t* cls, DWORD style, int x, int y, int w, int h,
          const wchar_t* text, int id) {
    HWND c = add_ctrl(parent, cls, style, x, y, w, h, text, id);
    if (page >= 0) g_page_ctls[page].push_back(c);
    return c;
}

HWND plabel(int page, HWND parent, int x, int y, const wchar_t* text, int w = kLblW) {
    return pctl(page, parent, L"Static", SS_LEFTNOWORDWRAP, x, y + 3, w, 18, text, 0);
}

void set_edit(HWND h, const std::string& s) { SetWindowTextW(h, to_w(s).c_str()); }

std::string get_edit(HWND h) {
    const int n = GetWindowTextLengthW(h);
    std::wstring w(static_cast<size_t>(n) + 1, L'\0');
    GetWindowTextW(h, &w[0], n + 1);
    w.resize(static_cast<size_t>(n));
    return trim_copy(from_w(w));
}

bool get_check(HWND h) { return SendMessageW(h, BM_GETCHECK, 0, 0) == BST_CHECKED; }

void set_optional_row_state(HWND off, HWND edit, HWND browse) {
    const bool disabled = get_check(off);
    EnableWindow(edit, !disabled);
    if (browse) EnableWindow(browse, !disabled);
}

// YaRN 勾选 → ROPE_FACTOR = max(2, ceil(单请求上限 / ROPE_ORIGINAL_CTX))；
// 不勾 → 1。上限/原生值解析失败时按默认粗估（校验页会单独报 [失败]）。
std::string derive_rope_factor(bool yarn_on, const std::string& max_context,
                               const std::string& rope_original) {
    if (!yarn_on) return "1";
    long mc = 262144, ro = 262144;
    if (!pn_int(max_context, &mc) || mc < 1) mc = 262144;
    if (!pn_int(rope_original, &ro) || ro < 1) ro = 262144;
    long f = (mc + ro - 1) / ro;
    if (f < 2) f = 2;
    return std::to_string(f);
}

PanelVals panel_collect() {
    PanelVals v;
    v.weights_ver = get_check(g_pc.wver[2])   ? kVerGguf
                    : get_check(g_pc.wver[1]) ? kVerV2
                                              : kVerV1;
    v.model = get_edit(g_pc.row_model.edit);
    v.ngram = get_edit(g_pc.row_ngram.edit);
    v.no_overlay = get_check(g_pc.row_overlay.off);
    v.overlay = get_edit(g_pc.row_overlay.edit);
    v.no_mtp = get_check(g_pc.row_mtp.off);
    v.mtp = get_edit(g_pc.row_mtp.edit);
    v.no_vision = get_check(g_pc.row_vision.off);
    v.vision = get_edit(g_pc.row_vision.edit);
    v.gguf = get_edit(g_pc.row_gguf.edit);
    v.no_gguf_mtp = get_check(g_pc.row_gguf_mtp.off);
    v.gguf_mtp = get_edit(g_pc.row_gguf_mtp.edit);
    v.no_gguf_vision = get_check(g_pc.row_gguf_vision.off);
    v.gguf_vision = get_edit(g_pc.row_gguf_vision.edit);
    v.tok = get_edit(g_pc.row_tok.edit);
    v.max_context = get_edit(g_pc.maxctx);
    const int sel = static_cast<int>(SendMessageW(g_pc.parallel, CB_GETCURSEL, 0, 0));
    v.parallel = std::to_string(sel >= 0 ? sel + 1 : 1);
    v.kv_pool = get_edit(g_pc.kv_pool);
    // 并发总池的跟随语义：框值 == 单请求上限（含程序跟随填充）或显式 0/跟随状态 → 存 0
    if (v.kv_pool == v.max_context || v.kv_pool == "0" || !g_pool_touched) v.kv_pool = "0";
    v.kv_paged = get_check(g_pc.kv_paged);
    v.prefill_chunk = get_edit(g_pc.prefill);
    v.rope_original = cfg("ROPE_ORIGINAL_CTX", "262144");  // 无 UI 行，conf/环境原样读
    v.rope_factor = derive_rope_factor(get_check(g_pc.yarn), v.max_context, v.rope_original);
    v.kvsnap_max_gb = get_edit(g_pc.kvsnap);
    v.engine_host = get_edit(g_pc.ehost);
    v.engine_port = get_edit(g_pc.eport);
    v.api_host = get_edit(g_pc.ahost);
    v.api_port = get_edit(g_pc.aport);
    return v;
}

// 第 4 页底部：其它页的失败汇总（GGUF 选中时也在这里说明启动受限原因）。
void refresh_outputs(const PanelVals& v, const AllChecks& c) {
    set_edit(g_pc.out1, c.weights.text);
    set_edit(g_pc.out2, c.context.text);
    set_edit(g_pc.out3, c.ports.text);
    set_edit(g_pc.report, c.vram.text);
    const CheckLines* pages[3] = {&c.weights, &c.context, &c.ports};
    const char* names[3] = {TR("第 1 页", "Page 1"), TR("第 2 页", "Page 2"),
                            TR("第 3 页", "Page 3")};
    size_t n = 0;
    for (const CheckLines* p : pages) n += p->fatals.size();
    std::string s;
    if (n == 0) {
        s = TR("其它页没有失败项。\r\n", "No failures on the other pages.\r\n");
    } else {
        s = TR("其它页还有 ", "The other pages still have ") + std::to_string(n) +
            TR(" 项失败：\r\n", " failure(s):\r\n");
        for (int i = 0; i < 3; ++i)
            for (const std::string& f : pages[i]->fatals)
                s += std::string(names[i]) + TR("：", ": ") + f + "\r\n";
    }
    if (v.weights_ver == kVerGguf)
        s += TR("已选 GGUF 权重：Windows 启动器只支持 hgn，配置仅保存"
                "（Linux 用 bash start_gguf.sh 启动）。\r\n",
                "GGUF weights selected: the Windows launcher supports hgn only; the config is "
                "saved only (on Linux use bash start_gguf.sh).\r\n");
    set_edit(g_pc.summary, s);
}

void panel_recheck() {
    const PanelVals v = panel_collect();
    g_last_checks = panel_checks(v, g_panel_running);
    refresh_outputs(v, g_last_checks);
    // 启动门禁：所有页无 [失败] 且所选权重为 V1/V2
    EnableWindow(g_pc.start,
                 g_panel_allow_start && !g_last_checks.fatal() && v.weights_ver != kVerGguf);
}

bool panel_apply_and_save(std::string* error) {
    const PanelVals v = panel_collect();
    const auto conf_map = panel_vals_to_conf(v);
    if (!save_conf(conf_map, error)) return false;
    conf_write_marker_line(kWeightsMarker, v.weights_ver == kVerV2    ? "v2"
                                           : v.weights_ver == kVerGguf ? "gguf"
                                                                       : "v1");
    // 同步内存中的 g_conf，让接下来的启动流程读到新值
    for (const auto& kv : conf_map) g_conf[kv.first] = kv.second;
    if (v.weights_ver == kVerV1) g_conf["NGRAM_FILE"] = v.model;  // 空 = 跟随主模型
    return true;
}

bool panel_browse(HWND edit) {
    wchar_t buf[MAX_PATH] = L"";
    const std::string cur = get_edit(edit);
    if (!cur.empty()) wcsncpy_s(buf, to_w(cur).c_str(), MAX_PATH - 1);
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = g_panel;
    ofn.lpstrFilter = g_lang_en
                           ? L"hgn weights (*.hgn)\0*.hgn\0GGUF weights (*.gguf)\0*.gguf\0All files (*.*)\0*.*\0"
                           : L"hgn 权重 (*.hgn)\0*.hgn\0GGUF 权重 (*.gguf)\0*.gguf\0所有文件 (*.*)\0*.*\0";
    ofn.lpstrFile = buf;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    ofn.lpstrTitle = TR(L"选择权重文件", L"Select weights file");
    if (!GetOpenFileNameW(&ofn)) return false;
    set_edit(edit, make_relative(from_w(buf)));
    return true;
}

// 把一个文件行排到指定 y（SWP_SHOWWINDOW），或整行隐藏
void place_row(const FileRow& r, int y, bool visible) {
    const UINT flags = SWP_NOZORDER | (visible ? SWP_SHOWWINDOW : SWP_HIDEWINDOW);
    const int ex = r.off ? kEdX + kOffW + 8 : kEdX;
    const int ew = 556 - ex;
    SetWindowPos(r.label, nullptr, kLblX, y + 3, kLblW, 18, flags);
    if (r.off) SetWindowPos(r.off, nullptr, kEdX, y, kOffW, 20, flags);
    SetWindowPos(r.edit, nullptr, ex, y, ew, 22, flags);
    if (r.browse) SetWindowPos(r.browse, nullptr, kBrX, y - 1, kBrW, 24, flags);
}

// 第 1 页布局：按版本只显示相关文件行，顺序重排，隐藏其余
void layout_page1(int ver) {
    g_weights_ver = ver;
    const FileRow* order[9];
    int n = 0;
    if (ver == kVerGguf) {
        order[n++] = &g_pc.row_gguf;
        order[n++] = &g_pc.row_gguf_mtp;
        order[n++] = &g_pc.row_gguf_vision;
        order[n++] = &g_pc.row_tok;
    } else if (ver == kVerV2) {
        order[n++] = &g_pc.row_model;
        order[n++] = &g_pc.row_ngram;
        order[n++] = &g_pc.row_mtp;
        order[n++] = &g_pc.row_vision;
        order[n++] = &g_pc.row_tok;
    } else {
        order[n++] = &g_pc.row_model;
        order[n++] = &g_pc.row_overlay;
        order[n++] = &g_pc.row_mtp;
        order[n++] = &g_pc.row_vision;
        order[n++] = &g_pc.row_tok;
    }
    const FileRow* all[9] = {&g_pc.row_model,  &g_pc.row_ngram,  &g_pc.row_overlay,
                             &g_pc.row_mtp,    &g_pc.row_vision, &g_pc.row_gguf,
                             &g_pc.row_gguf_mtp, &g_pc.row_gguf_vision, &g_pc.row_tok};
    int y = 76;
    for (int i = 0; i < n; ++i) {
        place_row(*order[i], y, true);
        y += 26;
    }
    for (const FileRow* r : all) {
        bool shown = false;
        for (int i = 0; i < n; ++i)
            if (order[i] == r) shown = true;
        if (!shown) place_row(*r, 0, false);
    }
    SetWindowPos(g_pc.tok_hint, nullptr, kEdX, y + 2, 434, 18, SWP_NOZORDER | SWP_SHOWWINDOW);
    SetWindowPos(g_pc.ver_note, nullptr, kEdX, y + 24, 434, 18, SWP_NOZORDER | SWP_SHOWWINDOW);
    SetWindowTextW(g_pc.ver_note, ver == kVerGguf
                                      ? TR(L"GGUF：仅保存配置，启动需 Linux bash start_gguf.sh",
                                           L"GGUF: config saved only; start on Linux via bash start_gguf.sh")
                                      : ver == kVerV2
                                            ? TR(L"V2：主体+独立 n-gram 文件；overlay 无效（保存置空）",
                                                 L"V2: main + separate n-gram file; overlay ignored (cleared on save)")
                                            : TR(L"V1：w4b 单文件，n-gram 内置；overlay/MTP/视觉塔可选",
                                                 L"V1: single w4b file, built-in n-gram; overlay/MTP/vision optional"));
}

void show_page(int idx) {
    for (int i = 0; i < 4; ++i)
        for (HWND c : g_page_ctls[i]) ShowWindow(c, i == idx ? SW_SHOW : SW_HIDE);
    if (idx == 0) layout_page1(g_weights_ver);  // 重新隐藏非当前版本的文件行
}

LRESULT CALLBACK panel_proc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_COMMAND) {
        const UINT id = LOWORD(wp), code = HIWORD(wp);
        // 编辑框输入 / 下拉选择变化 → 即时重检（各页列表独立刷新）
        if (code == EN_CHANGE || code == CBN_SELCHANGE) {
            // 并发总池跟随单请求上限：未手动改过总池框时，上限变了总池框跟着变；
            // 总池框输入非空非 0 → 标记已手动改；输入 0 → 回到跟随并立即回填上限值。
            // g_pool_guard：程序自己 SetWindowText 时不触发这些逻辑。
            if (code == EN_CHANGE && !g_pool_guard) {
                if (id == kEdKvPool) {
                    const std::string s = get_edit(g_pc.kv_pool);
                    if (s == "0") g_pool_touched = false;
                    else if (!s.empty()) g_pool_touched = true;
                }
                if ((id == kEdMaxctx && !g_pool_touched) ||
                    (id == kEdKvPool && get_edit(g_pc.kv_pool) == "0")) {
                    g_pool_guard = true;
                    SetWindowTextW(g_pc.kv_pool, to_w(get_edit(g_pc.maxctx)).c_str());
                    g_pool_guard = false;
                }
            }
            panel_recheck();
            return 0;
        }
        switch (id) {
            case kTab1: case kTab2: case kTab3: case kTab4:
                show_page(static_cast<int>(id - kTab1));
                return 0;
            case kWVer1: case kWVer2: case kWVer3:
                layout_page1(static_cast<int>(id - kWVer1));
                panel_recheck();
                return 0;
            case kBtnStart: {
                panel_recheck();
                const PanelVals v = panel_collect();
                if (v.weights_ver == kVerGguf) {
                    msgbox(TR("已选择 GGUF 权重：Windows 启动器只支持 hgn。\n"
                              "GGUF 配置可以仅保存，Linux 下用 bash start_gguf.sh 启动。",
                              "GGUF weights selected: the Windows launcher supports hgn only.\n"
                              "The GGUF config can be saved only; on Linux start it with "
                              "bash start_gguf.sh."),
                           MB_OK | MB_ICONWARNING);
                    return 0;
                }
                if (g_last_checks.fatal()) {
                    msgbox(TR("环境检查未通过，请修正标[失败]的项后再启动。",
                              "Checks failed: fix the items marked [FAIL] before starting."),
                           MB_OK | MB_ICONWARNING);
                    return 0;
                }
                std::string error;
                if (!panel_apply_and_save(&error)) {
                    msgbox(std::string(TR("保存配置失败：", "Failed to save config: ")) + error,
                           MB_OK | MB_ICONERROR);
                    return 0;
                }
                g_panel_result = kPanelStart;
                g_panel_done = true;
                return 0;
            }
            case kBtnSave: {
                std::string error;
                if (!panel_apply_and_save(&error)) {
                    msgbox(std::string(TR("保存配置失败：", "Failed to save config: ")) + error,
                           MB_OK | MB_ICONERROR);
                    return 0;
                }
                g_panel_result = kPanelSave;
                g_panel_done = true;
                return 0;
            }
            case kBtnCancel:
                g_panel_result = kPanelCancel;
                g_panel_done = true;
                return 0;
            case kBtnLang:
                // 切换语言：写回 lang 标记后返回 kPanelRelang，由调用方重开面板
                conf_write_marker_line(kLangMarker, g_lang_en ? "zh" : "en");
                g_panel_result = kPanelRelang;
                g_panel_done = true;
                return 0;
            case kBtnRecheck: panel_recheck(); return 0;
            case kBrModel:
                if (panel_browse(g_pc.row_model.edit)) panel_recheck();
                return 0;
            case kBrNgram:
                if (panel_browse(g_pc.row_ngram.edit)) panel_recheck();
                return 0;
            case kBrOverlay:
                if (panel_browse(g_pc.row_overlay.edit)) panel_recheck();
                return 0;
            case kBrMtp:
                if (panel_browse(g_pc.row_mtp.edit)) panel_recheck();
                return 0;
            case kBrVision:
                if (panel_browse(g_pc.row_vision.edit)) panel_recheck();
                return 0;
            case kBrGguf:
                if (panel_browse(g_pc.row_gguf.edit)) panel_recheck();
                return 0;
            case kBrGgufMtp:
                if (panel_browse(g_pc.row_gguf_mtp.edit)) panel_recheck();
                return 0;
            case kBrGgufVision:
                if (panel_browse(g_pc.row_gguf_vision.edit)) panel_recheck();
                return 0;
            case kOffOverlay:
                set_optional_row_state(g_pc.row_overlay.off, g_pc.row_overlay.edit,
                                       g_pc.row_overlay.browse);
                panel_recheck();
                return 0;
            case kOffMtp:
                set_optional_row_state(g_pc.row_mtp.off, g_pc.row_mtp.edit, g_pc.row_mtp.browse);
                panel_recheck();
                return 0;
            case kOffVision:
                set_optional_row_state(g_pc.row_vision.off, g_pc.row_vision.edit,
                                       g_pc.row_vision.browse);
                panel_recheck();
                return 0;
            case kOffGgufMtp:
                set_optional_row_state(g_pc.row_gguf_mtp.off, g_pc.row_gguf_mtp.edit,
                                       g_pc.row_gguf_mtp.browse);
                panel_recheck();
                return 0;
            case kOffGgufVision:
                set_optional_row_state(g_pc.row_gguf_vision.off, g_pc.row_gguf_vision.edit,
                                       g_pc.row_gguf_vision.browse);
                panel_recheck();
                return 0;
            case kCkKvPaged: panel_recheck(); return 0;
            case kCkYarn: panel_recheck(); return 0;
            default: break;
        }
    }
    if (msg == WM_CLOSE) {
        g_panel_result = kPanelCancel;
        g_panel_done = true;
        return 0;
    }
    return DefWindowProcW(h, msg, wp, lp);
}

// 第 1 页文件行：[label][x不使用][EDIT][浏览…]（extra_style 给 label，如 WS_GROUP）
FileRow page1_file_row(HWND parent, int y, const wchar_t* label, int id_edit, int id_browse,
                       int id_off, const std::string& value, bool off,
                       DWORD extra_style = 0) {
    FileRow r;
    r.label = pctl(0, parent, L"Static", SS_LEFTNOWORDWRAP | extra_style, kLblX, y + 3, kLblW,
                   18, label, 0);
    int ex = kEdX, ew = kEdPlainW;
    if (id_off) {
        r.off = pctl(0, parent, L"Button", BS_AUTOCHECKBOX, ex, y, kOffW, 20,
                     TR(L"不使用", L"off"), id_off);
        SendMessageW(r.off, BM_SETCHECK, off ? BST_CHECKED : BST_UNCHECKED, 0);
        ex += kOffW + 8;
        ew = 556 - ex;
    }
    r.edit = pctl(0, parent, L"Edit", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, ex, y, ew, 22,
                  L"", id_edit);
    set_edit(r.edit, value);
    if (id_browse)
        r.browse = pctl(0, parent, L"Button", BS_PUSHBUTTON, kBrX, y - 1, kBrW, 24,
                        TR(L"浏览…", L"Browse"), id_browse);
    if (id_off) set_optional_row_state(r.off, r.edit, r.browse);
    return r;
}

PanelResult panel_run(bool allow_start) {
    panel_lang_resolve();  // 每次打开重新解析（语言按钮切换后重开即生效）
    g_panel_running = !allow_start;
    g_panel_allow_start = allow_start;
    g_panel_done = false;
    g_panel_result = kPanelCancel;
    for (auto& v : g_page_ctls) v.clear();
    HINSTANCE hi = GetModuleHandleW(nullptr);
    static bool registered = false;
    if (!registered) {
        WNDCLASSW wc{};
        wc.lpfnWndProc = panel_proc;
        wc.hInstance = hi;
        wc.lpszClassName = L"gfx1151_setup_panel";
        wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_BTNFACE + 1);
        wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));  // IDC_ARROW
        RegisterClassW(&wc);
        registered = true;
    }
    RECT rc{0, 0, kCliW, kCliH};
    AdjustWindowRect(&rc, WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU, FALSE);
    const int win_w = rc.right - rc.left, win_h = rc.bottom - rc.top;
    RECT work{};
    SystemParametersInfoW(SPI_GETWORKAREA, 0, &work, 0);
    const int wx = work.left + (work.right - work.left - win_w) / 2;
    const int wy = work.top + (work.bottom - work.top - win_h) / 2;
    HWND h = CreateWindowExW(0, L"gfx1151_setup_panel",
                             TR(L"gfx1151-engine 启动设置", L"gfx1151-engine Startup Setup"),
                             WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU, wx, wy, win_w, win_h,
                             nullptr, nullptr, hi, nullptr);
    if (!h) {
        msgbox(std::string(TR("无法创建设置窗口，错误码 ", "Cannot create the setup window, error code ")) +
                   std::to_string(GetLastError()),
               MB_OK | MB_ICONERROR);
        return kPanelCancel;
    }
    g_panel = h;

    const PanelVals init = panel_vals_from_conf();
    g_weights_ver = init.weights_ver;

    // ---- 分页切换（恒可见）：BS_PUSHLIKE 单选按钮，不引入 comctl32
    const wchar_t* tab_names[4] = {TR(L"1 权重", L"1 Weights"),
                                   TR(L"2 上下文与并发", L"2 Context & parallel"),
                                   TR(L"3 端口与监听", L"3 Ports & listen"),
                                   TR(L"4 显存检查", L"4 VRAM check")};
    for (int i = 0; i < 4; ++i) {
        g_pc.tabs[i] = add_ctrl(h, L"Button",
                                BS_PUSHLIKE | BS_AUTORADIOBUTTON | WS_TABSTOP |
                                    (i == 0 ? WS_GROUP : 0),
                                12 + i * 158, 8, 150, 24, tab_names[i], kTab1 + i);
    }
    SendMessageW(g_pc.tabs[0], BM_SETCHECK, BST_CHECKED, 0);

    // ---- 第 1 页：权重
    plabel(0, h, kLblX, 44, TR(L"权重版本", L"Weights version"));
    const wchar_t* ver_names[3] = {TR(L"V1 版本", L"V1"), TR(L"V2 版本", L"V2"),
                                   TR(L"GGUF 版本", L"GGUF")};
    for (int i = 0; i < 3; ++i) {
        g_pc.wver[i] = pctl(0, h, L"Button",
                            BS_AUTORADIOBUTTON | WS_TABSTOP | (i == 0 ? WS_GROUP : 0),
                            kEdX + i * 92, 44, i == 2 ? 100 : 86, 20, ver_names[i], kWVer1 + i);
    }
    SendMessageW(g_pc.wver[g_weights_ver], BM_SETCHECK, BST_CHECKED, 0);
    int ry = 76;
    g_pc.row_model = page1_file_row(h, ry, TR(L"主模型", L"Main model"), kEdModel, kBrModel, 0,
                                    init.model, false, WS_GROUP);  // WS_GROUP：关闭上面的单选组
    ry += 26;
    g_pc.row_ngram = page1_file_row(h, ry, TR(L"n-gram 表", L"n-gram table"), kEdNgram, kBrNgram, 0,
                                    init.ngram, false);
    ry += 26;
    g_pc.row_overlay = page1_file_row(h, ry, L"overlay", kEdOverlay, kBrOverlay, kOffOverlay,
                                      init.overlay, init.no_overlay);
    ry += 26;
    g_pc.row_mtp = page1_file_row(h, ry, TR(L"MTP 草稿", L"MTP draft"), kEdMtp, kBrMtp, kOffMtp,
                                  init.mtp, init.no_mtp);
    ry += 26;
    g_pc.row_vision = page1_file_row(h, ry, TR(L"视觉塔", L"Vision tower"), kEdVision, kBrVision,
                                     kOffVision, init.vision, init.no_vision);
    ry += 26;
    g_pc.row_gguf = page1_file_row(h, ry, TR(L"GGUF 主分片", L"GGUF main shard"), kEdGguf, kBrGguf,
                                   0, init.gguf, false);
    ry += 26;
    g_pc.row_gguf_mtp = page1_file_row(h, ry, L"GGUF MTP", kEdGgufMtp, kBrGgufMtp, kOffGgufMtp,
                                       init.gguf_mtp, init.no_gguf_mtp);
    ry += 26;
    g_pc.row_gguf_vision = page1_file_row(h, ry, TR(L"GGUF 视觉塔", L"GGUF vision tower"),
                                          kEdGgufVision, kBrGgufVision, kOffGgufVision,
                                          init.gguf_vision, init.no_gguf_vision);
    ry += 26;
    g_pc.row_tok = page1_file_row(h, ry, L"tokenizer", kEdTok, 0, 0, init.tok, false);
    g_pc.tok_hint = pctl(0, h, L"Static", SS_LEFTNOWORDWRAP, kEdX, 210, 434, 18,
                         TR(L"目录内需包含 tokenizer.json", L"Folder must contain tokenizer.json"),
                         0);
    g_pc.ver_note = pctl(0, h, L"Static", SS_LEFTNOWORDWRAP, kEdX, 232, 434, 18, L"", 0);
    g_pc.out1 = pctl(0, h, L"Edit",
                     WS_BORDER | ES_MULTILINE | ES_READONLY | WS_VSCROLL | ES_AUTOVSCROLL,
                     12, 254, 624, kListBottom - 254, L"", kEdOut1);

    // ---- 第 2 页：上下文与并发、YaRN（行尾无 hint，说明在第 2 页检查列表顶部）
    int y2 = 46;
    plabel(1, h, kLblX, y2, TR(L"单请求上限 (token)", L"Per-request context (tokens)"));
    g_pc.maxctx = pctl(1, h, L"Edit", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, kEdX, y2, 120,
                       22, L"", kEdMaxctx);
    set_edit(g_pc.maxctx, init.max_context);
    y2 += 26;
    plabel(1, h, kLblX, y2, TR(L"并发路数", L"Parallel streams"));
    g_pc.parallel = pctl(1, h, L"ComboBox", CBS_DROPDOWNLIST | WS_VSCROLL | WS_TABSTOP, kEdX,
                         y2, 60, 200, L"", kCbParallel);
    for (int i = 1; i <= 8; ++i)
        SendMessageW(g_pc.parallel, CB_ADDSTRING, 0,
                     reinterpret_cast<LPARAM>(std::to_wstring(i).c_str()));
    long par_init = 1;
    pn_int(init.parallel, &par_init);
    SendMessageW(g_pc.parallel, CB_SETCURSEL, static_cast<WPARAM>(par_init - 1), 0);
    y2 += 26;
    plabel(1, h, kLblX, y2, TR(L"并发总池 (token)", L"KV pool (tokens)"));
    g_pc.kv_pool = pctl(1, h, L"Edit", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, kEdX, y2, 120,
                        22, L"", kEdKvPool);
    // conf=0（跟随）时显示单请求上限的值，g_pool_touched=false；
    // conf 有显式值则显示实值并视为已手动设定
    long pool_init = 0;
    pn_int(init.kv_pool, &pool_init);
    g_pool_touched = pool_init != 0;
    g_pool_guard = true;
    set_edit(g_pc.kv_pool, g_pool_touched ? init.kv_pool : init.max_context);
    g_pool_guard = false;
    y2 += 26;
    g_pc.kv_paged = pctl(1, h, L"Button", BS_AUTOCHECKBOX | WS_TABSTOP, kEdX, y2, 300, 20,
                         TR(L"分页 KV（并发必须开启）", L"Paged KV (required for parallel >1)"),
                         kCkKvPaged);
    SendMessageW(g_pc.kv_paged, BM_SETCHECK, init.kv_paged ? BST_CHECKED : BST_UNCHECKED, 0);
    y2 += 26;
    plabel(1, h, kLblX, y2, TR(L"prefill 分段", L"Prefill chunk"));
    g_pc.prefill = pctl(1, h, L"Edit", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, kEdX, y2, 120,
                        22, L"", kEdPrefill);
    set_edit(g_pc.prefill, init.prefill_chunk);
    plabel(1, h, kHintX, y2, TR(L"0 = 默认 8192（≈6.9 GiB 工作区）",
                                L"0 = default 8192 (≈6.9 GiB workspace)"),
           kHintW);
    y2 += 26;
    {
        // YaRN 勾选框（ROPE_FACTOR 自动推导 = max(2, ceil(上限/原生))，无编辑行）
        double rope_init = 1;
        pn_double(init.rope_factor, &rope_init);
        g_pc.yarn = pctl(1, h, L"Button", BS_AUTOCHECKBOX | WS_TABSTOP, kEdX, y2, 300, 20,
                         TR(L"YaRN 长上下文扩展", L"YaRN long-context scaling"), kCkYarn);
        SendMessageW(g_pc.yarn, BM_SETCHECK, rope_init > 1 ? BST_CHECKED : BST_UNCHECKED, 0);
    }
    y2 += 26;
    plabel(1, h, kLblX, y2, TR(L"KV 快照上限 (GiB)", L"KV snapshot cap (GiB)"));
    g_pc.kvsnap = pctl(1, h, L"Edit", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, kEdX, y2, 120,
                       22, L"", kEdKvsnap);
    set_edit(g_pc.kvsnap, init.kvsnap_max_gb);
    plabel(1, h, kHintX, y2, TR(L"RAM 检查点容量，0 = 关闭", L"RAM checkpoint capacity, 0 = off"),
           kHintW);
    y2 += 26;
    g_pc.out2 = pctl(1, h, L"Edit",
                     WS_BORDER | ES_MULTILINE | ES_READONLY | WS_VSCROLL | ES_AUTOVSCROLL,
                     12, y2 + 10, 624, kListBottom - y2 - 10, L"", kEdOut2);

    // ---- 第 3 页：端口与监听（行尾无 hint，说明在第 3 页检查列表顶部）
    int y3 = 46;
    plabel(2, h, kLblX, y3, TR(L"引擎监听地址", L"Engine bind address"));
    g_pc.ehost = pctl(2, h, L"Edit", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, kEdX, y3, 140,
                      22, L"", kEdEHost);
    set_edit(g_pc.ehost, init.engine_host);
    y3 += 26;
    plabel(2, h, kLblX, y3, TR(L"引擎端口", L"Engine port"));
    g_pc.eport = pctl(2, h, L"Edit", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, kEdX, y3, 90, 22,
                      L"", kEdEPort);
    set_edit(g_pc.eport, init.engine_port);
    y3 += 26;
    plabel(2, h, kLblX, y3, TR(L"API 监听地址", L"API bind address"));
    g_pc.ahost = pctl(2, h, L"Edit", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, kEdX, y3, 140,
                      22, L"", kEdAHost);
    set_edit(g_pc.ahost, init.api_host);
    y3 += 26;
    plabel(2, h, kLblX, y3, TR(L"API 端口", L"API port"));
    g_pc.aport = pctl(2, h, L"Edit", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, kEdX, y3, 90, 22,
                      L"", kEdAPort);
    set_edit(g_pc.aport, init.api_port);
    y3 += 26;
    g_pc.out3 = pctl(2, h, L"Edit",
                     WS_BORDER | ES_MULTILINE | ES_READONLY | WS_VSCROLL | ES_AUTOVSCROLL,
                     12, y3 + 8, 624, kListBottom - y3 - 8, L"", kEdOut3);

    // ---- 第 4 页：显存环境检查
    pctl(3, h, L"Button", BS_PUSHBUTTON, 524, 42, 112, 24, TR(L"重新检查", L"Re-check"),
         kBtnRecheck);
    g_pc.report = pctl(3, h, L"Edit",
                       WS_BORDER | ES_MULTILINE | ES_READONLY | WS_VSCROLL | ES_AUTOVSCROLL,
                       12, 72, 624, 272, L"", kEdReport);
    plabel(3, h, kLblX, 348, TR(L"其它页的失败汇总", L"Failures from other pages"), 200);
    g_pc.summary = pctl(3, h, L"Edit",
                        WS_BORDER | ES_MULTILINE | ES_READONLY | WS_VSCROLL | ES_AUTOVSCROLL,
                        12, 372, 624, kListBottom - 372, L"", kEdSummary);

    // ---- 底部按钮行（恒可见）
    const int by = 462;
    g_pc.start = add_ctrl(h, L"Button", BS_DEFPUSHBUTTON, 157, by, 130, 30,
                          allow_start ? TR(L"启动服务", L"Start service")
                                      : TR(L"启动服务（运行中）", L"Start (running)"),
                          kBtnStart);
    EnableWindow(g_pc.start, allow_start);
    add_ctrl(h, L"Button", BS_PUSHBUTTON, 297, by, 90, 30, TR(L"仅保存", L"Save only"), kBtnSave);
    add_ctrl(h, L"Button", BS_PUSHBUTTON, 397, by, 90, 30, TR(L"取消", L"Cancel"), kBtnCancel);
    if (!allow_start) add_label(h, 12, by + 8, TR(L"运行中：重启生效", L"Restart to apply"), 140);
    // 语言切换：写 lang 标记并返回 kPanelRelang（按钮文字显示目标语言）
    add_ctrl(h, L"Button", BS_PUSHBUTTON, 512, by, 124, 30,
             g_lang_en ? L"切换为中文" : L"Switch to English", kBtnLang);

    show_page(0);
    panel_recheck();
    ShowWindow(h, SW_SHOW);
    UpdateWindow(h);
    SetForegroundWindow(h);

    // 模态循环：不用 PostQuitMessage（托盘打开时 WM_QUIT 会连带杀掉托盘循环）
    MSG m;
    while (!g_panel_done && GetMessageW(&m, nullptr, 0, 0) > 0) {
        if (IsDialogMessageW(h, &m)) continue;
        TranslateMessage(&m);
        DispatchMessageW(&m);
    }
    DestroyWindow(h);
    g_panel = nullptr;
    return g_panel_result;
}

}  // namespace
