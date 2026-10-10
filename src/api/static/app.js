"use strict";
/* qwenox-api 控制台 v1（中英双语）
 * - 四个页面（总览/用量/请求/采样），hash 路由，左侧按钮分页。
 * - 文案双语：静态文本在 index.html 挂 data-i18n 键，动态文案走 t()；
 *   语言存 localStorage，首次按浏览器语言自动选择，侧栏 EN/中文 按钮切换。
 * - 数据源：/health /memory /power /admin/overrides /reqstat/page /reqstat/tail。
 *   请求页无筛选条件时走 /reqstat/page 服务端分页（可翻阅全部历史）；
 *   设了筛选条件回退到 tail 最近 1000 条内存过滤。用量页仍按 tail 聚合，
 *   按天聚合端点落地后再切换（见 usage-note）。
 */

// ---------------- 多语言 ----------------

const I18N = {
  zh: {
    "app.title": "Qwen-Flash-Server 控制台",
    "hint.theme": "切换明暗主题",
    "nav.overview": "总览", "nav.usage": "用量", "nav.requests": "请求",
    "common.loading": "加载中…",
    "common.refresh": "刷新", "common.all": "全部", "common.read_failed": "读取失败",
    "common.close": "关闭",
    "common.requests": "请求数",
    "stamp.updated": "更新", "stamp.failed": "请求失败",
    "ov.status": "服务状态", "ov.model": "模型配置",
    "ov.mem": "显存与内存", "ov.power": "功耗",
    "power.socket": "整机封装", "power.gfx": "GPU 功耗", "power.cpu": "CPU 功耗",
    "power.na": "功耗读数不可用",
    "health.status": "状态", "health.busy_field": "忙闲",
    "health.busy_yes": "推理中", "health.busy_no": "空闲",
    "health.inflight": "在途请求", "health.model": "模型",
    "health.context": "上下文", "health.slots": "并发槽位",
    "health.slots_shared": "（共享池 ",
    "health.rope_off": "未启用", "health.tool": "工具调用",
    "health.tool_on": "解析已启用", "health.tool_off": "透传",
    "health.vision": "视觉输入", "health.vision_on": "接受", "health.vision_off": "拒绝",
    "led.pwr_tip": "电源：控制台页面已加载",
    "led.link_tip": "与 API 的连接：熄灭 = 页面与 API 断开",
    "led.warn_tip": "告警：绿 = 无告警；红 = API 断连或引擎异常退出",
    "led.eng": "引擎",
    "led.eng_tip_idle": "引擎就绪（空闲）",
    "led.eng_tip_busy": "引擎推理中",
    "led.eng_tip_starting": "引擎启动中…",
    "led.eng_tip_stopped": "引擎未启动（到引擎页或托盘菜单启动）",
    "led.eng_tip_exited": "引擎异常退出（见引擎日志）",
    "led.eng_tip_down": "与 API 断开连接",
    "led.eng_external": "（外部进程）",
    "mem.vram": "显存", "mem.resident": "引擎驻留",
    "mem.vram_peak": "引擎显存峰值", "mem.rss": "常驻主机内存 RSS",
    "mem.pinned": "锁页内存", "mem.accessible": "可访问已提交",
    "mem.offline": "引擎未连接（/memory 不可用）",
    "ovr.on": "开", "ovr.off": "关",
    "usage.range": "范围", "usage.d7": "近 7 天", "usage.d14": "近 14 天",
    "usage.d30": "近 30 天",
    "usage.cached": "缓存命中",
    "usage.prompt_tokens": "prompt tokens", "usage.output_tokens": "output tokens",
    "usage.empty": "范围内没有请求记录",
    "usage.note": "v1 数据来自 /reqstat/tail 最近 1000 条记录，更早的历史需要后端按天聚合端点后接入。",
    "req.from": "起", "req.to": "止", "req.size": "每页", "req.clear": "清除筛选",
    "req.time": "时间", "req.cache": "缓存", "req.accept": "接受率",
    "req.prev_page": "上一页", "req.next_page": "下一页",
    "req.note_server": "服务端分页（/reqstat/page），可翻阅全部历史；日期与 finish/drafter 为前端过滤（过滤时仅覆盖最近 1000 条）。",
    "req.note_filtered": "已设筛选：后端不支持按条件检索，仅在 /reqstat/tail 最近 1000 条内过滤分页。",
    "nav.sample": "采样", "nav.engine": "引擎",
    "ovr.title": "采样 / 思考参数（服务端覆盖）",
    "ovr.hint": "跟随客户端：不干预。默认值：只在客户端没传该字段时使用。" +
      "强制：不管客户端传什么都改成这个值（max_tokens 的“强制”是上限，超过才压到此值）。" +
      "对 /v1/chat/completions、/v1/responses、/v1/completions 都生效，思考相关字段对 completions 不起作用。" +
      "被改写的字段会写进响应头 X-Qwenox-Overrides。保存后立即生效，并持久化到服务端文件，重启后仍有效。",
    "ovr.th_param": "参数", "ovr.th_mode": "模式", "ovr.th_value": "值", "ovr.th_note": "说明",
    "ovr.mode_client": "跟随客户端", "ovr.mode_default": "默认值", "ovr.mode_force": "强制",
    "ovr.bool_on": "开 (true)", "ovr.bool_off": "关 (false)",
    "ovr.save": "保存并生效", "ovr.reload": "从服务端重新读取", "ovr.clear": "全部改为跟随客户端",
    "ovr.key_ph": "管理密钥 (QWENOX_API_ADMIN_KEY)",
    "ovr.s_dirty": "有未保存的修改", "ovr.s_loaded": "已读取", "ovr.s_saved": "已保存并生效",
    "ovr.s_applied": "已生效，但未能写入文件",
    "ovr.e_read": "读取失败：{m}", "ovr.e_save": "保存失败：{m}",
    "ovr.e_persist": "持久化失败：{m}（重启后会丢失）", "ovr.e_http": "HTTP {s}",
    "ovr.confirm_reload": "丢弃未保存的修改？",
    "ovr.confirm_clear": "清空服务端覆盖表，所有参数改为跟随客户端？",
    "ovr.meta_persist": "持久化文件：{p}", "ovr.meta_memory": "仅内存（未持久化）",
    "ovr.e_need": "{k} 需要填写数值", "ovr.e_num": "{k} 不是有效数字",
    "ovr.e_int": "{k} 必须是整数", "ovr.e_range": "{k} 超出范围 {lo} ~ {hi}",
    "ovr.f_thinking": "思考开关", "ovr.f_effort": "思考强度", "ovr.f_preserve": "保留历史思考",
    "ovr.n_thinking": "关 = 模板关闭思考，不输出 reasoning_content；客户端不传时为开",
    "ovr.n_effort": "low / medium / xhigh（客户端传 high→xhigh，minimal→low）；仅思考开启时有效",
    "ovr.n_preserve": "多轮时历史 assistant 的思考是否放回提示词；不传时保留",
    "ovr.n_temperature": "0 = 贪心；客户端不传时为 1.0。强制 0 时会去掉请求里的 logprobs",
    "ovr.n_top_p": "客户端不传时为 0.95", "ovr.n_top_k": "0 = 不限；客户端不传时为 20",
    "ovr.n_min_p": "客户端不传时为 0",
    "ovr.n_presence": "-2 ~ 2", "ovr.n_frequency": "-2 ~ 2",
    "ovr.n_max_tokens": "默认值 = 客户端没给输出上限时用它；强制 = 输出上限，客户端给得更大时压到此值",
    "eng.status": "引擎状态", "eng.start": "启动引擎", "eng.stop": "停止引擎",
    "eng.state_stopped": "已停止", "eng.state_starting": "启动中（冷加载可能要几分钟）",
    "eng.state_running": "运行中", "eng.state_exited": "异常退出",
    "eng.pid": "PID", "eng.log": "日志", "eng.external_tag": "外部进程（由其它启动器拉起，此处不能停止）",
    "eng.nocontrol": "当前平台（Linux）不支持网页启停引擎：下方配置保存后由下次启动（start_hgn.sh / start_gguf.sh）生效。",
    "eng.confirm_stop": "停止引擎会中断正在进行的生成。确定停止？",
    "eng.e_start": "启动失败：{m}", "eng.e_stop": "停止失败：{m}",
    "cfg.title": "服务配置（service.conf）",
    "cfg.hint": "保存写入 service.conf（原内容备份为 .bak），下次启动引擎时生效；运行中的引擎与 API 端口不受影响。环境变量覆盖的键已置灰。",
    "cfg.grp_weights": "权重文件", "cfg.grp_ctx": "上下文与并发", "cfg.grp_net": "端口与监听",
    "cfg.weights_ver": "权重版本",
    "cfg.ver_v1": "V1（单文件，n-gram 内嵌主权重）",
    "cfg.ver_v2": "V2（主权重 + 独立 n-gram 文件）",
    "cfg.ver_gguf": "GGUF（仅 Linux 脚本启动）",
    "cfg.save": "保存配置", "cfg.reload": "重新读取",
    "cfg.s_dirty": "有未保存的修改", "cfg.s_loaded": "已读取",
    "cfg.s_saved": "已保存，下次启动引擎时生效",
    "cfg.s_saved_env": "已保存，但这些键被环境变量覆盖，需清除环境变量才生效：{k}",
    "cfg.e_read": "读取失败：{m}", "cfg.e_save": "保存失败：{m}",
    "cfg.env_badge": "env 覆盖", "cfg.confirm_reload": "丢弃未保存的修改？",
    "cfg.pick_ph": "手动输入，或点浏览选择（项目根目录）",
    "cfg.browse": "浏览", "cfg.pick_empty": "此目录下没有可选条目",
    "cfg.pick_here": "选择此目录", "cfg.up": "上一级",
    "cfg.dir_tag": "目录", "cfg.e_files": "文件列表加载失败：{m}",
  },
  en: {
    "app.title": "Qwen-Flash-Server console",
    "hint.theme": "toggle light/dark theme",
    "nav.overview": "Overview", "nav.usage": "Usage", "nav.requests": "Requests",
    "common.loading": "Loading…",
    "common.refresh": "Refresh", "common.all": "All", "common.read_failed": "read failed",
    "common.close": "Close",
    "common.requests": "Requests",
    "stamp.updated": "Updated", "stamp.failed": "request failed",
    "ov.status": "Service status", "ov.model": "Model config",
    "ov.mem": "VRAM & memory", "ov.power": "Power",
    "power.socket": "Package", "power.gfx": "GPU", "power.cpu": "CPU",
    "power.na": "power reading unavailable",
    "health.status": "Status", "health.busy_field": "Busy",
    "health.busy_yes": "generating", "health.busy_no": "idle",
    "health.inflight": "In flight", "health.model": "Model",
    "health.context": "Context", "health.slots": "Slots",
    "health.slots_shared": " (shared pool ",
    "health.rope_off": "off", "health.tool": "Tool calls",
    "health.tool_on": "parsing enabled", "health.tool_off": "passthrough",
    "health.vision": "Vision input", "health.vision_on": "accepted", "health.vision_off": "rejected",
    "led.pwr_tip": "Power: console page loaded",
    "led.link_tip": "Connection to the API: off = page lost contact with the API",
    "led.warn_tip": "Warning: green = all clear; red = API unreachable or engine exited unexpectedly",
    "led.eng": "Engine",
    "led.eng_tip_idle": "Engine ready (idle)",
    "led.eng_tip_busy": "Engine generating",
    "led.eng_tip_starting": "Engine starting…",
    "led.eng_tip_stopped": "Engine not started (start it from the Engine page or tray menu)",
    "led.eng_tip_exited": "Engine exited unexpectedly (see engine log)",
    "led.eng_tip_down": "Lost connection to the API",
    "led.eng_external": " (external)",
    "mem.vram": "VRAM", "mem.resident": "Engine resident",
    "mem.vram_peak": "Engine VRAM peak", "mem.rss": "RSS (host)",
    "mem.pinned": "Pinned memory", "mem.accessible": "Accessible committed",
    "mem.offline": "engine offline (/memory unavailable)",
    "ovr.on": "on", "ovr.off": "off",
    "usage.range": "Range", "usage.d7": "7 days", "usage.d14": "14 days",
    "usage.d30": "30 days",
    "usage.cached": "Cache hit",
    "usage.prompt_tokens": "prompt tokens", "usage.output_tokens": "output tokens",
    "usage.empty": "no requests in range",
    "usage.note": "v1: newest 1000 records via /reqstat/tail; older history needs a backend per-day aggregation endpoint.",
    "req.from": "From", "req.to": "To", "req.size": "Per page", "req.clear": "Clear filters",
    "req.time": "Time", "req.cache": "Cache", "req.accept": "Acceptance",
    "req.prev_page": "Prev", "req.next_page": "Next",
    "req.note_server": "Server-side paging (/reqstat/page) over the full history; date and finish/drafter filters are client-side (newest 1000 records only).",
    "req.note_filtered": "Filters active: the backend has no conditional search, so filtering covers only the newest 1000 records from /reqstat/tail.",
    "nav.sample": "Sampling", "nav.engine": "Engine",
    "ovr.title": "Sampling / thinking parameters (server overrides)",
    "ovr.hint": "Follow client: no interference. Default: used only when the client omitted the field. " +
      "Force: replaces whatever the client sent (for max_tokens it is a cap: larger values clamp to it). " +
      "Applies to /v1/chat/completions, /v1/responses and /v1/completions; thinking fields have no " +
      "effect on completions. Rewritten fields are reported in the X-Qwenox-Overrides response header. " +
      "Saving takes effect immediately and is persisted server-side, surviving restarts.",
    "ovr.th_param": "Parameter", "ovr.th_mode": "Mode", "ovr.th_value": "Value", "ovr.th_note": "Note",
    "ovr.mode_client": "Follow client", "ovr.mode_default": "Default", "ovr.mode_force": "Force",
    "ovr.bool_on": "on (true)", "ovr.bool_off": "off (false)",
    "ovr.save": "Save & apply", "ovr.reload": "Reload from server", "ovr.clear": "All follow client",
    "ovr.key_ph": "Admin key (QWENOX_API_ADMIN_KEY)",
    "ovr.s_dirty": "unsaved changes", "ovr.s_loaded": "loaded", "ovr.s_saved": "saved & applied",
    "ovr.s_applied": "applied, but not written to file",
    "ovr.e_read": "read failed: {m}", "ovr.e_save": "save failed: {m}",
    "ovr.e_persist": "persist failed: {m} (lost on restart)", "ovr.e_http": "HTTP {s}",
    "ovr.confirm_reload": "Discard unsaved changes?",
    "ovr.confirm_clear": "Clear the override table so every field follows the client?",
    "ovr.meta_persist": "persist file: {p}", "ovr.meta_memory": "memory only (not persisted)",
    "ovr.e_need": "{k} needs a value", "ovr.e_num": "{k} is not a valid number",
    "ovr.e_int": "{k} must be an integer", "ovr.e_range": "{k} out of range {lo} ~ {hi}",
    "ovr.f_thinking": "thinking toggle", "ovr.f_effort": "thinking effort",
    "ovr.f_preserve": "preserve history",
    "ovr.n_thinking": "off = template disables thinking, no reasoning_content; on when client omits",
    "ovr.n_effort": "low / medium / xhigh (client high→xhigh, minimal→low); only when thinking is on",
    "ovr.n_preserve": "whether past assistant reasoning goes back into the prompt; kept when omitted",
    "ovr.n_temperature": "0 = greedy; 1.0 when the client omits it. Forcing 0 strips logprobs",
    "ovr.n_top_p": "0.95 when the client omits it", "ovr.n_top_k": "0 = unlimited; 20 when omitted",
    "ovr.n_min_p": "0 when the client omits it",
    "ovr.n_presence": "-2 ~ 2", "ovr.n_frequency": "-2 ~ 2",
    "ovr.n_max_tokens": "default = used when the client gives no output cap; " +
      "force = cap, larger client values clamp to it",
    "eng.status": "Engine status", "eng.start": "Start engine", "eng.stop": "Stop engine",
    "eng.state_stopped": "stopped", "eng.state_starting": "starting (a cold load can take minutes)",
    "eng.state_running": "running", "eng.state_exited": "exited unexpectedly",
    "eng.pid": "PID", "eng.log": "Log",
    "eng.external_tag": "external process (started by another launcher; cannot be stopped here)",
    "eng.nocontrol": "Engine start/stop from the web console is not supported on this platform " +
      "(Linux): saved config takes effect on the next start via start_hgn.sh / start_gguf.sh.",
    "eng.confirm_stop": "Stopping the engine interrupts any ongoing generation. Stop now?",
    "eng.e_start": "start failed: {m}", "eng.e_stop": "stop failed: {m}",
    "cfg.title": "Service configuration (service.conf)",
    "cfg.hint": "Saving writes service.conf (previous content backed up to .bak) and takes effect " +
      "on the next engine start; the running engine and API port are unaffected. " +
      "Keys overridden by environment variables are disabled.",
    "cfg.grp_weights": "Weight files", "cfg.grp_ctx": "Context & concurrency", "cfg.grp_net": "Ports & listen",
    "cfg.weights_ver": "Weights version",
    "cfg.ver_v1": "V1 (single file, n-gram embedded)",
    "cfg.ver_v2": "V2 (main weights + separate n-gram)",
    "cfg.ver_gguf": "GGUF (started via Linux scripts only)",
    "cfg.save": "Save config", "cfg.reload": "Reload",
    "cfg.s_dirty": "unsaved changes", "cfg.s_loaded": "loaded",
    "cfg.s_saved": "saved; takes effect on the next engine start",
    "cfg.s_saved_env": "saved, but these keys are overridden by environment variables: {k}",
    "cfg.e_read": "read failed: {m}", "cfg.e_save": "save failed: {m}",
    "cfg.env_badge": "env override", "cfg.confirm_reload": "Discard unsaved changes?",
    "cfg.pick_ph": "type manually, or browse (project root)",
    "cfg.browse": "Browse", "cfg.pick_empty": "no selectable entries in this folder",
    "cfg.pick_here": "Select this folder", "cfg.up": "up one level",
    "cfg.dir_tag": "dir", "cfg.e_files": "file list load failed: {m}",
  },
};

let LANG = null;
try { LANG = localStorage.getItem("qwenox-lang"); } catch (e) { /* 隐私模式 */ }
if (LANG !== "zh" && LANG !== "en")
  LANG = (navigator.language || "zh").toLowerCase().startsWith("zh") ? "zh" : "en";

function t(key) {
  const v = I18N[LANG][key];
  return v !== undefined ? v : (I18N.zh[key] || key);
}

function applyLang() {
  document.documentElement.lang = LANG === "zh" ? "zh-CN" : "en";
  for (const el of document.querySelectorAll("[data-i18n]"))
    el.textContent = t(el.dataset.i18n);
  for (const el of document.querySelectorAll("[data-i18n-title]"))
    el.title = t(el.dataset.i18nTitle);
  for (const el of document.querySelectorAll("[data-i18n-placeholder]"))
    el.placeholder = t(el.dataset.i18nPlaceholder);
  const btn = document.getElementById("lang-btn");
  btn.querySelector(".hw-switch-label").textContent = LANG === "zh" ? "中文" : "EN";
  btn.classList.toggle("on", LANG === "zh");
  btn.title = LANG === "zh" ? "Switch to English" : "切换到中文";
}

document.getElementById("lang-btn").addEventListener("click", () => {
  LANG = LANG === "zh" ? "en" : "zh";
  try { localStorage.setItem("qwenox-lang", LANG); } catch (e) { /* 同上 */ }
  applyLang();
  // 动态文案（徽章/表格外新增文本/时间戳）随当前页面重渲染，不重新发请求
  if (currentPage === "overview") { refreshHealth(); refreshMemory(); refreshPower(); }
  else if (currentPage === "usage") { if (reqCache.records.length) renderUsage(); }
  else if (currentPage === "requests") renderRequests();
  else if (currentPage === "sample") syncSampleLang();
  else if (currentPage === "engine") renderEngineAll();
});

// ---------------- 主题 ----------------

function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  const btn = document.getElementById("theme-btn");
  btn.querySelector(".hw-switch-label").textContent = t === "dark" ? "☾" : "☀";
  btn.classList.toggle("on", t === "dark");
  try { localStorage.setItem("qwenox-theme", t); } catch (e) { /* 隐私模式 */ }
}
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("qwenox-theme"); } catch (e) { /* 同上 */ }
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(saved || (prefersDark ? "dark" : "light"));
}
document.getElementById("theme-btn").addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  applyTheme(cur === "dark" ? "light" : "dark");
  if (currentPage === "usage") drawUsageChart();  // 图表颜色跟随主题
});

// ---------------- 工具 ----------------

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) {
    let detail = "";
    try {
      const t = await r.text();
      const m = t.match(/"message"\s*:\s*"([^"]*)"/);
      if (m) detail = ": " + m[1];
    } catch (e) { /* 忽略 */ }
    throw new Error(r.status + " " + r.statusText + detail);
  }
  return r.json();
}

const fmtInt = (n) => Number(n).toLocaleString("en-US");
function fmtBytes(b) {
  if (b == null) return "—";
  const u = ["B", "KiB", "MiB", "GiB", "TiB"];
  let i = 0;
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return (i === 0 ? b : b.toFixed(2)) + " " + u[i];
}
const fmtMs = (ms) => (ms >= 10000 ? (ms / 1000).toFixed(1) + " s" : Math.round(ms) + " ms");
const locale = () => (LANG === "zh" ? "zh-CN" : "en-US");
const fmtTime = (ts) => new Date(ts).toLocaleString(locale(), { hour12: false });
const pad2 = (n) => String(n).padStart(2, "0");
function fmtDay(ts) {
  const d = new Date(ts);
  return d.getFullYear() + "-" + pad2(d.getMonth() + 1) + "-" + pad2(d.getDate());
}
// 吞吐：有效 token 数 ÷ 耗时。prefill 口径为 (prompt − 缓存命中)/prefill 时间，
// 与后端 /reqstat/summary 的 prefill_tok_per_s / decode_tok_per_s 一致。
const tokPerS = (tokens, ms) => (ms > 0 ? fmtInt(Math.round((tokens * 1000) / ms)) : "—");
function compactNum(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(1) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(Math.round(n));
}
const stampNow = (el) => {
  el.textContent = t("stamp.updated") + " " +
    new Date().toLocaleTimeString(locale(), { hour12: false });
};
const stampErr = (el, e) => { el.textContent = t("stamp.failed") + " " + e.message; };

// 用 [[dt, dd], ...] 填充 <dl class="kv">
function fillKv(el, rows) {
  el.textContent = "";
  el.classList.remove("placeholder");
  for (const [dt, dd] of rows) {
    const row = document.createElement("div");
    row.className = "kv-row";
    const t = document.createElement("dt");
    t.textContent = dt;
    const d = document.createElement("dd");
    if (dd && dd.nodeType === Node.ELEMENT_NODE) d.appendChild(dd);
    else d.textContent = String(dd);
    row.append(t, d);
    el.appendChild(row);
  }
}
function pill(text, cls) {
  const s = document.createElement("span");
  s.className = "pill " + cls;
  s.textContent = text;
  return s;
}

// ---------------- reqstat 共享缓存 ----------------

const reqCache = { records: [], fetchedAt: 0, error: null };
async function loadReqstat(force) {
  const fresh = Date.now() - reqCache.fetchedAt < 30000;
  if (!force && reqCache.records.length && fresh) return;
  const j = await getJSON("/reqstat/tail?n=1000");
  reqCache.records = (j.records || []).slice().sort((a, b) => b.ts_ms - a.ts_ms);
  reqCache.fetchedAt = Date.now();
  reqCache.error = null;
}

// ---------------- 路由 ----------------

const PAGES = ["overview", "usage", "requests", "sample", "engine"];
let currentPage = "";
const onEnter = { overview: enterOverview, usage: enterUsage, requests: enterRequests,
  sample: enterSample, engine: enterEngine };

function showPage(name) {
  if (!PAGES.includes(name)) name = "overview";
  currentPage = name;
  for (const p of PAGES) {
    document.getElementById("page-" + p).classList.toggle("hidden", p !== name);
  }
  for (const b of document.querySelectorAll(".nav-btn"))
    b.classList.toggle("active", b.dataset.route === name);
  onEnter[name]();
}
document.getElementById("nav").addEventListener("click", (ev) => {
  const b = ev.target.closest(".nav-btn");
  if (b) location.hash = "#/" + b.dataset.route;
});
window.addEventListener("hashchange", () => showPage(location.hash.replace(/^#\/?/, "") || "overview"));

// ---------------- 总览 ----------------

let ovTimer = null;
function enterOverview() {
  refreshHealth();
  refreshMemory();
  refreshPower();
  if (ovTimer) clearInterval(ovTimer);
  let tick = 0;
  ovTimer = setInterval(() => {
    if (currentPage !== "overview" || document.hidden) return;
    refreshHealth();
    refreshPower();
    if (++tick % 5 === 0) refreshMemory();
  }, 2000);
  // 离开页面时停掉轮询
  const stop = () => { if (ovTimer) { clearInterval(ovTimer); ovTimer = null; } };
  const obs = new MutationObserver(() => {
    if (currentPage !== "overview") { stop(); obs.disconnect(); }
  });
  obs.observe(document.getElementById("page-overview"), { attributes: true, attributeFilter: ["class"] });
}

// ENG 灯：引擎真实状态（不再只看有无在途请求——引擎没启动时 idle 绿灯是误导）。
// 绿 = 就绪空闲，橙闪 = 推理中/启动中，灭 = 未启动，红 = 异常退出；
// down = 页面与 API 断开（LINK 灭、WARN 亮，由 CSS :has() 联动）。
function updateEngLed(h) {
  const dot = document.getElementById("busy-dot");
  const engState = h.engine || "stopped";
  const dotCls = engState === "running" ? (h.busy ? "busy" : "idle")
    : engState === "starting" ? "starting"
    : engState === "exited" ? "exited" : "stopped";
  dot.className = "brand-dot " + dotCls;
  dot.title = t("led.eng_tip_" + dotCls) + (h.engine_external ? t("led.eng_external") : "");
}

function engLedDown() {
  const dot = document.getElementById("busy-dot");
  dot.className = "brand-dot down";
  dot.title = t("led.eng_tip_down");
}

// 全局轮询：指示灯在所有页面都要活（总览页自己的 refreshHealth 也会调 updateEngLed）
async function pollEngLed() {
  try { updateEngLed(await getJSON("/health")); }
  catch (e) { engLedDown(); }
}
setInterval(() => { if (!document.hidden) pollEngLed(); }, 3000);

async function refreshHealth() {
  try {
    const h = await getJSON("/health");
    updateEngLed(h);
    const engState = h.engine || "stopped";
    const engPillCls = engState === "running" ? "ok" : engState === "starting" ? "busy"
      : engState === "exited" ? "err" : "";
    const busy = h.busy
      ? pill(t("health.busy_yes"), "busy")
      : pill(t("health.busy_no"), "ok");
    fillKv(document.getElementById("ov-status"), [
      [t("health.status"), pill(h.status === "ok" ? "ok" : h.status, "ok")],
      [t("led.eng"), pill(t("eng.state_" + engState) +
        (h.engine_external ? t("led.eng_external") : ""), engPillCls)],
      [t("health.busy_field"), busy],
      [t("health.inflight"), fmtInt(h.in_flight ?? 0) +
        (h.queued ? (LANG === "zh" ? "（排队 " : " (queued ") + fmtInt(h.queued) + (LANG === "zh" ? "）" : ")") : "")],
    ]);
    const rows = [
      [t("health.model"), h.model || "—"],
      [t("health.context"), fmtInt(h.context)],
      [t("health.slots"), fmtInt(h.slots) + " × " + fmtInt(h.slot_ctx) +
        (h.kv_pool > 0 && h.kv_pool < h.slots * h.slot_ctx
          ? t("health.slots_shared") + fmtInt(h.kv_pool) + (LANG === "zh" ? "）" : ")")
          : "")],
    ];
    if (h.rope_scaling) {
      rows.push(["RoPE", h.rope_scaling.type + " × " + h.rope_scaling.factor]);
    } else rows.push(["RoPE", t("health.rope_off")]);
    if (h.tool_calls) rows.push([t("health.tool"), h.tool_calls.parsed ? t("health.tool_on") : t("health.tool_off")]);
    if (h.vision) rows.push([t("health.vision"), h.vision.accepted ? t("health.vision_on") : t("health.vision_off")]);
    fillKv(document.getElementById("ov-model"), rows);
    stampNow(document.getElementById("overview-stamp"));
  } catch (e) {
    engLedDown();
    stampErr(document.getElementById("overview-stamp"), e);
  }
}

async function refreshMemory() {
  const bars = document.getElementById("mem-bars");
  try {
    const m = await getJSON("/memory");
    const hipUsed = (m.hip_total_bytes ?? 0) - (m.hip_free_bytes ?? 0);
    const mkBar = (label, used, total) => {
      const row = document.createElement("div");
      row.className = "bar-row";
      const l = document.createElement("span"); l.className = "bar-label"; l.textContent = label;
      const tr = document.createElement("div"); tr.className = "bar-track";
      const fl = document.createElement("div"); fl.className = "bar-fill";
      fl.style.width = (total > 0 ? Math.min(100, used / total * 100) : 0).toFixed(1) + "%";
      tr.appendChild(fl);
      const v = document.createElement("span"); v.className = "bar-value";
      v.textContent = fmtBytes(used) + " / " + fmtBytes(total);
      row.append(l, tr, v);
      return row;
    };
    bars.textContent = "";
    bars.append(
      mkBar(t("mem.vram"), hipUsed, m.hip_total_bytes),
      mkBar(t("mem.resident"), m.device_current_bytes, m.hip_total_bytes));
    fillKv(document.getElementById("ov-memory"), [
      [t("mem.vram_peak"), fmtBytes(m.device_peak_bytes)],
      [t("mem.rss"), fmtBytes(m.process_rss_bytes)],
      [t("mem.pinned"), fmtBytes(m.process_locked_bytes)],
      [t("mem.accessible"), fmtBytes(m.gpu_accessible_committed_bytes)],
    ]);
  } catch (e) {
    bars.textContent = t("mem.offline");
    bars.classList.add("muted");
  }
}

async function refreshPower() {
  const el = document.getElementById("ov-power");
  try {
    const p = await getJSON("/power");
    if (!p.available) {
      fillKv(el, [[t("ov.power"), t("power.na")]]);
    } else {
      const rows = [];
      if (p.socket_watts != null) rows.push([t("power.socket"), p.socket_watts + " W"]);
      if (p.gfx_watts != null) rows.push([t("power.gfx"), p.gfx_watts + " W"]);
      if (p.cpu_watts != null) rows.push([t("power.cpu"), p.cpu_watts + " W"]);
      fillKv(el, rows);
    }
  } catch (e) {
    fillKv(el, [[t("ov.power"), t("common.read_failed")]]);
  }
}

// ---------------- 采样（服务端覆盖编辑） ----------------
// 字段定义与后端 kOverrideFields（main.cpp）一致；提交后服务端 validate_overrides
// 仍是最终裁判，前端校验只为即时反馈。"跟随客户端"的字段不进提交表。

const OVR_FIELDS = [
  { k: "enable_thinking", type: "bool", def: true,
    label: "ovr.f_thinking", note: "ovr.n_thinking" },
  { k: "reasoning_effort", type: "effort", def: "xhigh",
    label: "ovr.f_effort", note: "ovr.n_effort" },
  { k: "preserve_thinking", type: "bool", def: true,
    label: "ovr.f_preserve", note: "ovr.n_preserve" },
  { k: "temperature", type: "num", min: 0, max: 2, step: 0.05, def: 1.0,
    note: "ovr.n_temperature" },
  { k: "top_p", type: "num", min: 0, max: 1, step: 0.01, def: 0.95, note: "ovr.n_top_p" },
  { k: "top_k", type: "int", min: 0, max: 2147483647, step: 1, def: 20, note: "ovr.n_top_k" },
  { k: "min_p", type: "num", min: 0, max: 1, step: 0.01, def: 0, note: "ovr.n_min_p" },
  { k: "presence_penalty", type: "num", min: -2, max: 2, step: 0.1, def: 0,
    note: "ovr.n_presence" },
  { k: "frequency_penalty", type: "num", min: -2, max: 2, step: 0.1, def: 0,
    note: "ovr.n_frequency" },
  { k: "max_tokens", type: "int", min: 1, max: 2147483647, step: 1, def: 32768,
    note: "ovr.n_max_tokens" },
];

function fmtMsg(key, vars) {
  let s = t(key);
  if (vars) for (const [k, v] of Object.entries(vars)) s = s.split("{" + k + "}").join(String(v));
  return s;
}
const ovrEl = (id) => document.getElementById(id);
function ovrSetStatus(base) {
  ovrEl("ovr-status").textContent = base + " " +
    new Date().toLocaleTimeString(locale(), { hour12: false });
}
function ovrShowErr(msg) {
  const e = ovrEl("ovr-err");
  e.textContent = msg;
  e.classList.remove("hidden");
}
function ovrHideErr() { ovrEl("ovr-err").classList.add("hidden"); }

let ovrBuilt = false, ovrDirty = false;

function buildSampleTable() {
  const tb = ovrEl("ovr-body");
  tb.textContent = "";
  for (const f of OVR_FIELDS) {
    const tr = document.createElement("tr");
    const param = document.createElement("td");
    param.className = "ovr-param";
    const mode = document.createElement("select");
    for (const [m, key] of [["client", "ovr.mode_client"], ["default", "ovr.mode_default"],
        ["force", "ovr.mode_force"]]) {
      const o = document.createElement("option");
      o.value = m;
      o.textContent = t(key);
      mode.appendChild(o);
    }
    const modeTd = document.createElement("td");
    modeTd.appendChild(mode);
    let val;
    if (f.type === "bool" || f.type === "effort") {
      val = document.createElement("select");
      const opts = f.type === "bool" ? [["true", "ovr.bool_on"], ["false", "ovr.bool_off"]]
        : [["low", null], ["medium", null], ["xhigh", null]];
      for (const [v, key] of opts) {
        const o = document.createElement("option");
        o.value = v;
        o.textContent = key ? t(key) : v;
        val.appendChild(o);
      }
    } else {
      val = document.createElement("input");
      val.type = "number";
      val.min = f.min;
      val.max = f.max;
      val.step = f.step;
    }
    const valTd = document.createElement("td");
    valTd.appendChild(val);
    const note = document.createElement("td");
    note.className = "ovr-note";
    tr.append(param, modeTd, valTd, note);
    tb.appendChild(tr);
    f.row = tr; f.paramTd = param; f.noteTd = note; f.modeEl = mode; f.valEl = val;
    mode.addEventListener("change", () => { syncSampleRow(f); markSampleDirty(); });
    val.addEventListener("input", markSampleDirty);
    val.addEventListener("change", markSampleDirty);
  }
  ovrBuilt = true;
}

// 语言切换后刷新表格里 JS 生成的文案（data-i18n 静态文本由 applyLang 负责）
let ovrLastState = null;
function syncSampleLang() {
  if (!ovrBuilt) return;
  if (ovrLastState)
    ovrEl("ovr-meta").textContent = ovrLastState.persist_path
      ? fmtMsg("ovr.meta_persist", { p: ovrLastState.persist_path })
      : t("ovr.meta_memory");
  for (const f of OVR_FIELDS) {
    f.paramTd.textContent = f.label ? t(f.label) : f.k;
    f.noteTd.textContent = t(f.note);
    const mo = [...f.modeEl.options];
    ["ovr.mode_client", "ovr.mode_default", "ovr.mode_force"]
      .forEach((k, i) => { mo[i].textContent = t(k); });
    if (f.type === "bool") {
      f.valEl.options[0].textContent = t("ovr.bool_on");
      f.valEl.options[1].textContent = t("ovr.bool_off");
    }
  }
}

function syncSampleRow(f) {
  const m = f.modeEl.value;
  f.valEl.disabled = m === "client";
  f.row.classList.toggle("mode-force", m === "force");
  f.row.classList.toggle("mode-default", m === "default");
}

function markSampleDirty() {
  ovrDirty = true;
  ovrEl("ovr-status").textContent = t("ovr.s_dirty");
}

function fillSample(state) {
  ovrLastState = state;
  const table = state.overrides || {};
  for (const f of OVR_FIELDS) {
    const e = table[f.k];
    f.modeEl.value = e ? e.mode : "client";
    f.valEl.value = String(e ? e.value : f.def);
    syncSampleRow(f);
  }
  ovrEl("ovr-key").classList.toggle("hidden", !state.admin_key_required);
  ovrEl("ovr-meta").textContent = state.persist_path
    ? fmtMsg("ovr.meta_persist", { p: state.persist_path })
    : t("ovr.meta_memory");
  ovrDirty = false;
}

async function loadSample() {
  try {
    fillSample(await getJSON("/admin/overrides"));
    ovrHideErr();
    ovrSetStatus(t("ovr.s_loaded"));
  } catch (e) {
    ovrShowErr(fmtMsg("ovr.e_read", { m: e.message }));
  }
}

function getSample() {
  const table = {};
  for (const f of OVR_FIELDS) {
    const m = f.modeEl.value;
    if (m === "client") continue;
    let v;
    if (f.type === "bool") {
      v = f.valEl.value === "true";
    } else if (f.type === "effort") {
      v = f.valEl.value;
    } else {
      const raw = f.valEl.value.trim();
      if (raw === "") throw new Error(fmtMsg("ovr.e_need", { k: f.k }));
      const n = Number(raw);
      if (!isFinite(n)) throw new Error(fmtMsg("ovr.e_num", { k: f.k }));
      if (f.type === "int" && !Number.isInteger(n)) throw new Error(fmtMsg("ovr.e_int", { k: f.k }));
      if (n < f.min || n > f.max)
        throw new Error(fmtMsg("ovr.e_range", { k: f.k, lo: f.min, hi: f.max }));
      v = n;
    }
    table[f.k] = { mode: m, value: v };
  }
  return table;
}

async function postSample(table) {
  const headers = { "Content-Type": "application/json" };
  const key = ovrEl("ovr-key").value.trim();
  if (key) headers["X-Admin-Key"] = key;
  let r, j = null;
  try {
    r = await fetch("/admin/overrides", { method: "POST", headers, body: JSON.stringify(table) });
    try { j = await r.json(); } catch (e) { /* 非 JSON 按状态码报错 */ }
  } catch (e) {
    ovrShowErr(fmtMsg("ovr.e_save", { m: e.message }));
    return;
  }
  if (!r.ok) {
    if (r.status === 401) ovrEl("ovr-key").classList.remove("hidden");
    const m = (j && j.error && (j.error.message || j.error.type)) || fmtMsg("ovr.e_http", { s: r.status });
    ovrShowErr(fmtMsg("ovr.e_save", { m }));
    return;
  }
  fillSample(j);
  ovrHideErr();
  if (j.saved) {
    ovrSetStatus(t("ovr.s_saved"));
  } else {
    ovrSetStatus(t("ovr.s_applied"));
    ovrShowErr(fmtMsg("ovr.e_persist", { m: j.save_error || "?" }));
  }
}

ovrEl("ovr-save").addEventListener("click", () => {
  let table;
  try { table = getSample(); } catch (e) { ovrShowErr(e.message); return; }
  postSample(table);
});
ovrEl("ovr-reload").addEventListener("click", () => {
  if (ovrDirty && !confirm(t("ovr.confirm_reload"))) return;
  loadSample();
});
ovrEl("ovr-clear").addEventListener("click", () => {
  if (!confirm(t("ovr.confirm_clear"))) return;
  postSample({});
});

function enterSample() {
  if (!ovrBuilt) buildSampleTable();
  syncSampleLang();
  loadSample();
}

// ---------------- 用量 ----------------

let usageAgg = [];  // [{day, requests, prompt, cached, gen}]，图表重绘用

async function enterUsage() {
  try {
    await loadReqstat(false);
    renderUsage();
  } catch (e) { stampErr(document.getElementById("usage-stamp"), e); }
}
document.getElementById("usage-refresh").addEventListener("click", async () => {
  try { await loadReqstat(true); renderUsage(); } catch (e) { stampErr(document.getElementById("usage-stamp"), e); }
});
document.getElementById("usage-range").addEventListener("change", renderUsage);

function renderUsage() {
  const days = Number(document.getElementById("usage-range").value);
  const t0 = Date.now() - days * 86400000;
  const byDay = new Map();
  const tot = { requests: 0, prompt: 0, cached: 0, gen: 0 };
  for (const r of reqCache.records) {
    if (r.ts_ms < t0) continue;
    const d = fmtDay(r.ts_ms);
    if (!byDay.has(d)) byDay.set(d, { day: d, requests: 0, prompt: 0, cached: 0, gen: 0 });
    const g = byDay.get(d);
    g.requests++; g.prompt += r.prompt_tokens; g.cached += r.cached_tokens; g.gen += r.output_tokens;
    tot.requests++; tot.prompt += r.prompt_tokens; tot.cached += r.cached_tokens; tot.gen += r.output_tokens;
  }
  usageAgg = [...byDay.values()].sort((a, b) => a.day < b.day ? -1 : 1);

  const totals = document.getElementById("usage-totals");
  totals.textContent = "";
  for (const [label, v] of [[t("common.requests"), tot.requests],
      [t("usage.prompt_tokens"), tot.prompt],
      [t("usage.cached"), tot.cached], [t("usage.output_tokens"), tot.gen]]) {
    const it = document.createElement("div");
    it.className = "total-item";
    const b = document.createElement("b"); b.textContent = fmtInt(v);
    const s = document.createElement("span"); s.textContent = label;
    it.append(b, s);
    totals.appendChild(it);
  }

  drawUsageChart();
  stampNow(document.getElementById("usage-stamp"));
  const note = document.getElementById("usage-note");
  note.textContent = note.title = t("usage.note");
}

function drawUsageChart() {
  const cv = document.getElementById("usage-chart");
  if (document.getElementById("page-usage").classList.contains("hidden")) return;
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight;  // 画布尺寸跟随 .chart-box（撑满屏幕剩余高度）
  if (!W || !H) return;
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext("2d");
  g.scale(dpr, dpr);
  const css = getComputedStyle(document.documentElement);
  const cPrompt = css.getPropertyValue("--chart-prompt").trim();
  const cGen = css.getPropertyValue("--chart-gen").trim();
  const cGrid = css.getPropertyValue("--border").trim();
  const cText = css.getPropertyValue("--text-muted").trim();
  const cTrack = css.getPropertyValue("--bg-raised").trim();
  g.clearRect(0, 0, W, H);
  if (!usageAgg.length) {
    g.fillStyle = cText; g.font = "13px system-ui";
    g.fillText(t("usage.empty"), 20, 30);
    return;
  }
  const ML = 56, MR = 12, MT = 10, MB = 24;
  const pw = W - ML - MR, ph = H - MT - MB;
  const maxV = Math.max(...usageAgg.map((d) => d.prompt + d.gen)) || 1;
  const nice = niceCeil(maxV);
  g.strokeStyle = cGrid; g.fillStyle = cText; g.font = "11px system-ui";
  g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const v = nice * i / 4;
    const y = MT + ph - ph * i / 4;
    g.beginPath(); g.moveTo(ML, y); g.lineTo(W - MR, y); g.stroke();
    g.textAlign = "right"; g.textBaseline = "middle";
    g.fillText(compactNum(v), ML - 6, y);
  }
  const n = usageAgg.length;
  const slot = pw / n;
  const bw = Math.min(46, slot * 0.62);
  const labelEvery = n > 15 ? Math.ceil(n / 10) : 1;
  usageAgg.forEach((d, i) => {
    const x = ML + slot * i + (slot - bw) / 2;
    const hp = (d.prompt / nice) * ph;
    const hg = (d.gen / nice) * ph;
    g.fillStyle = cPrompt; g.fillRect(x, MT + ph - hp, bw, hp);
    g.fillStyle = cGen; g.fillRect(x, MT + ph - hp - hg, bw, hg);
    if (i % labelEvery === 0) {
      g.fillStyle = cText; g.textAlign = "center"; g.textBaseline = "top";
      g.fillText(d.day.slice(5), x + bw / 2, MT + ph + 6);
    }
  });
}
function niceCeil(v) {
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  const m = v / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}
// 图表盒子随窗口/布局伸缩（含切页时从隐藏变可见），观察盒子本身而不是 window
let resizeT = null;
new ResizeObserver(() => {
  clearTimeout(resizeT);
  resizeT = setTimeout(() => { if (currentPage === "usage") drawUsageChart(); }, 100);
}).observe(document.querySelector("#page-usage .chart-box"));

// ---------------- 请求 ----------------
// 无筛选条件时：/reqstat/page 服务端分页，按服务端 total 翻阅全部历史。
// 设了日期/finish/drafter 筛选时：后端不支持按条件检索，回退到
// /reqstat/tail 最近 1000 条做内存过滤分页（note 里说明口径）。

const reqState = { offset: 0, seq: 0 };
const pageSize = () => Number(document.getElementById("req-size").value);
const filtersActive = () =>
  ["req-from", "req-to", "req-finish", "req-drafter"].some(
    (id) => document.getElementById(id).value);

async function enterRequests() {
  reqState.offset = 0;
  await renderRequests();
}

document.getElementById("req-refresh").addEventListener("click", () => renderRequests(true));
for (const id of ["req-from", "req-to", "req-finish", "req-drafter", "req-size"])
  document.getElementById(id).addEventListener("change", () => { reqState.offset = 0; renderRequests(); });
document.getElementById("req-clear").addEventListener("click", () => {
  for (const id of ["req-from", "req-to", "req-finish", "req-drafter"])
    document.getElementById(id).value = "";
  reqState.offset = 0; renderRequests();
});
document.getElementById("req-prev").addEventListener("click", () => {
  reqState.offset = Math.max(0, reqState.offset - pageSize());
  renderRequests();
});
document.getElementById("req-next").addEventListener("click", () => {
  reqState.offset += pageSize();
  renderRequests();
});

function filteredRequests() {
  const from = document.getElementById("req-from").value;
  const to = document.getElementById("req-to").value;
  const fin = document.getElementById("req-finish").value;
  const dr = document.getElementById("req-drafter").value;
  const t0 = from ? new Date(from + "T00:00:00").getTime() : 0;
  const t1 = to ? new Date(to + "T23:59:59.999").getTime() : Infinity;
  return reqCache.records.filter((r) =>
    r.ts_ms >= t0 && r.ts_ms <= t1 &&
    (!fin || r.finish === fin) && (!dr || r.drafter === dr));
}

function fillRequestRows(rows) {
  const tbody = document.querySelector("#req-table tbody");
  tbody.textContent = "";
  for (const r of rows) {
    const tr = document.createElement("tr");
    const cells = [
      fmtTime(r.ts_ms), fmtInt(r.prompt_tokens), fmtInt(r.cached_tokens),
      fmtInt(r.output_tokens), fmtMs(r.ttft_ms), fmtMs(r.prefill_ms), fmtMs(r.decode_ms),
      tokPerS(r.prompt_tokens - r.cached_tokens, r.prefill_ms),
      tokPerS(r.output_tokens, r.decode_ms),
      r.drafter, r.finish,
      r.acceptance == null ? "—" : (r.acceptance * 100).toFixed(1) + "%",
    ];
    cells.forEach((v, i) => {
      const td = document.createElement("td");
      if (i > 0 && i !== 9 && i !== 10) td.className = "r";
      td.textContent = String(v);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
}

async function renderRequests(force) {
  const size = pageSize();
  const stamp = document.getElementById("req-stamp");
  const note = document.getElementById("req-note");
  try {
    let rows = [], total = 0, pages = 1;
    if (filtersActive()) {
      await loadReqstat(force);
      const all = filteredRequests();  // reqCache 已按新→旧排序
      total = all.length;
      pages = Math.max(1, Math.ceil(total / size));
      if (reqState.offset >= total) reqState.offset = (pages - 1) * size;
      rows = all.slice(reqState.offset, reqState.offset + size);
      note.textContent = note.title = t("req.note_filtered");
    } else {
      for (let attempt = 0; attempt < 2; attempt++) {
        const seq = ++reqState.seq;
        const j = await getJSON("/reqstat/page?offset=" + reqState.offset + "&limit=" + size);
        if (seq !== reqState.seq) return;  // 已发出更新的请求，丢弃过期响应
        total = j.total || 0;
        pages = Math.max(1, Math.ceil(total / size));
        const maxOff = (pages - 1) * size;
        if (reqState.offset > maxOff) { reqState.offset = maxOff; continue; }
        rows = (j.records || []).slice().reverse();  // 服务端页内 oldest-first，表格新→旧
        break;
      }
      note.textContent = note.title = t("req.note_server");
    }
    fillRequestRows(rows);
    const page = Math.floor(reqState.offset / size) + 1;
    document.getElementById("req-pageinfo").textContent = LANG === "zh"
      ? "第 " + page + " / " + pages + " 页 · 共 " + fmtInt(total) + " 条"
      : "Page " + page + " / " + pages + " · " + fmtInt(total) + " records";
    document.getElementById("req-prev").disabled = reqState.offset <= 0;
    document.getElementById("req-next").disabled = reqState.offset >= (pages - 1) * size;
    stampNow(stamp);
  } catch (e) { stampErr(stamp, e); }
}

// ---------------- 引擎 / 设置（service.conf + 启停） ----------------
// 数据源：GET/POST /admin/config（20 个受管键，与托盘面板相同）、
// GET /admin/files（工作空间内的受限目录浏览，权重路径键的"浏览"弹窗数据）、
// GET /admin/engine/status、POST /admin/engine/start|stop（仅 Windows）。
// Linux 上隐藏启停按钮、显示提示，只保留配置编辑（保存后下次启动生效）。

const CFG_GROUPS = [
  { title: "cfg.grp_weights", keys: ["MODEL_FILE", "NGRAM_FILE", "OVERLAY_FILE",
      "MTP_FILE", "VISION_FILE", "GGUF_FILE", "GGUF_MTP_FILE", "GGUF_VISION_FILE",
      "TOKENIZER_DIR"] },
  { title: "cfg.grp_ctx", keys: ["MAX_CONTEXT", "PARALLEL", "KV_POOL_TOKENS", "KV_PAGED",
      "PREFILL_CHUNK", "ROPE_FACTOR", "KVSNAP_MAX_GB"] },
  { title: "cfg.grp_net", keys: ["ENGINE_HOST", "ENGINE_PORT", "API_HOST", "API_PORT"] },
];
// 权重组键带"浏览"弹窗（数据来自 /admin/files，仍可手输）：
// TOKENIZER_DIR 选目录，其余选文件。
const CFG_FILE_KEYS = new Set(CFG_GROUPS[0].keys);
// 权重版本 → 该版本相关的权重键（与旧托盘面板语义一致；TOKENIZER_DIR 各版本通用）。
// V1：n-gram 内嵌主权重（保存时 NGRAM_FILE 由服务端固定为 $MODEL_FILE）；
// V2：独立 n-gram 文件（OVERLAY_FILE 由服务端置空）；GGUF：只有 GGUF_* 三键。
const CFG_VER_KEYS = {
  v1: ["MODEL_FILE", "OVERLAY_FILE", "MTP_FILE", "VISION_FILE"],
  v2: ["MODEL_FILE", "NGRAM_FILE", "MTP_FILE", "VISION_FILE"],
  gguf: ["GGUF_FILE", "GGUF_MTP_FILE", "GGUF_VISION_FILE"],
};
const CFG_VER_IDS = ["v1", "v2", "gguf"];

const engEl = (id) => document.getElementById(id);
let cfgBuilt = false, cfgDirty = false, engTimer = null, engLastStatus = null;
let cfgVer = "v1";     // 当前选中的权重版本（GET /admin/config 的 weights_ver）
const cfgInputs = {};  // key -> input/select
const cfgBadges = {};  // key -> env 覆盖徽章
const cfgRows = {};    // key -> 行元素（权重版本切换时隐藏无关行）

function engSetCfgStatus(base) {
  engEl("cfg-status").textContent = base + " " +
    new Date().toLocaleTimeString(locale(), { hour12: false });
}
function engShowErr(id, msg) { const e = engEl(id); e.textContent = msg; e.classList.remove("hidden"); }
function engHideErr(id) { engEl(id).classList.add("hidden"); }

function buildCfgForm() {
  const form = engEl("cfg-form");
  form.textContent = "";
  for (const g of CFG_GROUPS) {
    const h = document.createElement("h3");
    h.className = "cfg-group-title";
    h.dataset.i18n = g.title;
    h.textContent = t(g.title);
    if (g.title === "cfg.grp_weights") {
      // 版本选择行：切换后只显示该版本相关的权重键（见 applyWeightsVer）
      const vrow = document.createElement("div");
      vrow.className = "cfg-row cfg-ver-row";
      const vlab = document.createElement("label");
      vlab.htmlFor = "cfg-weights-ver";
      vlab.textContent = t("cfg.weights_ver");
      const sel = document.createElement("select");
      sel.id = "cfg-weights-ver";
      for (const v of CFG_VER_IDS) {
        const o = document.createElement("option");
        o.value = v;
        o.textContent = t("cfg.ver_" + v);
        sel.appendChild(o);
      }
      sel.value = cfgVer;
      sel.addEventListener("change", () => {
        cfgVer = sel.value;
        applyWeightsVer();
        markCfgDirty();
      });
      vrow.append(vlab, sel);
      form.append(h, vrow);
    } else {
      form.appendChild(h);
    }
    const rows = document.createElement("div");
    rows.className = "cfg-rows";
    for (const key of g.keys) {
      const row = document.createElement("div");
      row.className = "cfg-row";
      const lab = document.createElement("label");
      const badge = document.createElement("span");
      badge.className = "cfg-env-badge hidden";
      badge.textContent = t("cfg.env_badge");
      let inp;
      if (key === "KV_PAGED") {
        inp = document.createElement("select");
        for (const [v, k] of [["1", "ovr.bool_on"], ["0", "ovr.bool_off"]]) {
          const o = document.createElement("option");
          o.value = v;
          o.textContent = t(k);
          inp.appendChild(o);
        }
      } else {
        inp = document.createElement("input");
        inp.type = "text";
        inp.spellcheck = false;
      }
      if (CFG_FILE_KEYS.has(key)) inp.placeholder = t("cfg.pick_ph");
      inp.id = "cfg-" + key;
      lab.htmlFor = inp.id;
      lab.append(key, badge);
      inp.addEventListener("input", markCfgDirty);
      inp.addEventListener("change", markCfgDirty);
      if (CFG_FILE_KEYS.has(key)) {
        // 权重路径键：输入框 + "浏览"按钮（弹窗列项目根目录条目，点选填入）
        const wrap = document.createElement("div");
        wrap.className = "cfg-file-row";
        const browse = document.createElement("button");
        browse.type = "button";
        browse.className = "btn cfg-browse";
        browse.textContent = t("cfg.browse");
        browse.addEventListener("click", () => openFilePicker(key));
        wrap.append(inp, browse);
        row.append(lab, wrap);
      } else {
        row.append(lab, inp);
      }
      rows.appendChild(row);
      cfgInputs[key] = inp;
      cfgBadges[key] = badge;
      cfgRows[key] = row;
    }
    form.appendChild(rows);
    if (g.title === "cfg.grp_weights") applyWeightsVer();
  }
  cfgBuilt = true;
}

function markCfgDirty() {
  cfgDirty = true;
  engEl("cfg-status").textContent = t("cfg.s_dirty");
}

// 按权重版本显示/隐藏权重键行（TOKENIZER_DIR 各版本通用）。
function applyWeightsVer() {
  const vis = new Set([...(CFG_VER_KEYS[cfgVer] || []), "TOKENIZER_DIR"]);
  for (const key of CFG_FILE_KEYS)
    if (cfgRows[key]) cfgRows[key].classList.toggle("hidden", !vis.has(key));
}

// ---------------- 文件选择弹窗 ----------------
// 数据来自 GET /admin/files?path=...：服务端把 path 限制在 conf 根目录（项目根）
// 之内（拒绝绝对路径与 ".."，符号链接解析后越界同样 400）。目录可逐层进入；
// TOKENIZER_DIR 用"选择此目录"按钮选定当前目录，其余权重键点文件即选定。

let fpickKey = null;   // 正在为哪个配置键选文件（null = 弹窗关闭）
let fpickPath = "";    // 当前浏览的相对路径（"" = 项目根）
let fpickData = null;  // 当前目录的 /admin/files 响应

const fpickJoin = (base, name) => (base ? base + "/" + name : name);

async function fpickFetch() {
  fpickData = null;
  renderFilePicker();
  try {
    fpickData = await getJSON("/admin/files" +
      (fpickPath ? "?path=" + encodeURIComponent(fpickPath) : ""));
  } catch (e) {
    if (fpickPath) {
      // 起始目录不存在（如配置指向 ./models 而服务器上没有）：回根目录重来
      fpickPath = "";
      return fpickFetch();
    }
    const list = engEl("fpick-list");
    list.textContent = "";
    const p = document.createElement("p");
    p.className = "note form-err fpick-empty";
    p.textContent = fmtMsg("cfg.e_files", { m: e.message });
    list.appendChild(p);
    return;
  }
  renderFilePicker();
}

function renderFilePicker() {
  const list = engEl("fpick-list");
  list.textContent = "";
  const dirMode = fpickKey === "TOKENIZER_DIR";
  engEl("fpick-here").classList.toggle("hidden", !dirMode);
  engEl("fpick-here").disabled = !fpickPath;
  engEl("fpick-foot").textContent =
    ((fpickData && fpickData.root) || "") + (fpickPath ? "/" + fpickPath : "");
  const addRow = (label, meta, onClick) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "fpick-item";
    const name = document.createElement("span");
    name.className = "fpick-name";
    name.textContent = label;
    const m = document.createElement("span");
    m.className = "fpick-meta";
    m.textContent = meta;
    b.append(name, m);
    b.addEventListener("click", onClick);
    list.appendChild(b);
  };
  if (!fpickData) {
    const p = document.createElement("p");
    p.className = "muted small fpick-empty";
    p.textContent = t("common.loading");
    list.appendChild(p);
    return;
  }
  if (fpickPath) {
    addRow("../", t("cfg.up"), () => {
      fpickPath = fpickPath.slice(0, fpickPath.lastIndexOf("/"));
      fpickFetch();
    });
  }
  const dirs = fpickData.dirs || [];
  const files = fpickData.files || [];
  if (!dirs.length && (dirMode || !files.length)) {
    const p = document.createElement("p");
    p.className = "muted small fpick-empty";
    p.textContent = t("cfg.pick_empty");
    list.appendChild(p);
  }
  for (const e of dirs) {
    addRow(e.name + "/", t("cfg.dir_tag"), () => {
      fpickPath = fpickJoin(fpickPath, e.name);
      fpickFetch();
    });
  }
  if (!dirMode) {
    for (const e of files) {
      addRow(e.name, fmtBytes(e.size) + (e.mtime ? " · " + fmtTime(e.mtime * 1000) : ""),
        () => {
          cfgInputs[fpickKey].value = fpickJoin(fpickPath, e.name);
          markCfgDirty();
          closeFilePicker();
        });
    }
  }
}

function openFilePicker(key) {
  fpickKey = key;
  // 从输入框现有值推断起始目录：规范掉 ./ 与前导/尾随斜杠；
  // 含 ".." 的值不信任，回根目录；目录不存在时由 fpickFetch 回退根目录。
  const v = (cfgInputs[key].value || "").trim().replace(/\\/g, "/");
  let dir = v.includes("/") ? v.slice(0, v.lastIndexOf("/")) : "";
  dir = dir.replace(/^(\.\/+)+/, "").replace(/^\/+/, "").replace(/\/+$/, "");
  fpickPath = dir.includes("..") ? "" : dir;
  engEl("fpick-title").textContent = key;
  engEl("fpick").classList.remove("hidden");
  fpickFetch();
}

function closeFilePicker() {
  fpickKey = null;
  engEl("fpick").classList.add("hidden");
}

engEl("fpick-close").addEventListener("click", closeFilePicker);
engEl("fpick-refresh").addEventListener("click", fpickFetch);
engEl("fpick-here").addEventListener("click", () => {
  if (fpickKey && fpickPath) {
    cfgInputs[fpickKey].value = fpickPath;
    markCfgDirty();
  }
  closeFilePicker();
});
engEl("fpick").addEventListener("click", (e) => {
  if (e.target.id === "fpick") closeFilePicker();  // 点遮罩空白处关闭
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && fpickKey) closeFilePicker();
});

async function loadConfig() {
  try {
    const j = await getJSON("/admin/config");
    for (const key of Object.keys(cfgInputs)) {
      const inp = cfgInputs[key];
      inp.value = (j.values && j.values[key]) || "";
      const envOvr = (j.env_overridden || []).includes(key);
      inp.disabled = envOvr;
      cfgBadges[key].classList.toggle("hidden", !envOvr);
    }
    cfgVer = CFG_VER_KEYS[j.weights_ver] ? j.weights_ver : "v1";
    engEl("cfg-weights-ver").value = cfgVer;
    applyWeightsVer();
    engEl("cfg-key").classList.toggle("hidden", !j.admin_key_required);
    cfgDirty = false;
    engHideErr("cfg-err");
    engSetCfgStatus(t("cfg.s_loaded"));
  } catch (e) {
    engShowErr("cfg-err", fmtMsg("cfg.e_read", { m: e.message }));
  }
}

async function postConfig() {
  const values = {};
  const vis = new Set([...(CFG_VER_KEYS[cfgVer] || []), "TOKENIZER_DIR"]);
  for (const key of Object.keys(cfgInputs)) {
    if (cfgInputs[key].disabled) continue;
    if (CFG_FILE_KEYS.has(key) && !vis.has(key)) continue;  // 非当前版本的权重键不发送
    values[key] = cfgInputs[key].value.trim();
  }
  const headers = { "Content-Type": "application/json" };
  const key = engEl("cfg-key").value.trim();
  if (key) headers["X-Admin-Key"] = key;
  let r, j = null;
  try {
    r = await fetch("/admin/config", { method: "POST", headers,
      body: JSON.stringify({ values, weights_ver: cfgVer }) });
    try { j = await r.json(); } catch (e) { /* 非 JSON 按状态码报错 */ }
  } catch (e) {
    engShowErr("cfg-err", fmtMsg("cfg.e_save", { m: e.message }));
    return;
  }
  if (!r.ok) {
    if (r.status === 401) engEl("cfg-key").classList.remove("hidden");
    const m = (j && j.error && (j.error.message || j.error.type)) ||
      fmtMsg("ovr.e_http", { s: r.status });
    engShowErr("cfg-err", fmtMsg("cfg.e_save", { m }));
    return;
  }
  cfgDirty = false;
  engHideErr("cfg-err");
  if (j.env_overridden && j.env_overridden.length)
    engSetCfgStatus(fmtMsg("cfg.s_saved_env", { k: j.env_overridden.join(", ") }));
  else
    engSetCfgStatus(t("cfg.s_saved"));
}

async function engineAction(action) {
  const headers = {};
  const key = engEl("cfg-key").value.trim();
  if (key) headers["X-Admin-Key"] = key;
  let r, j = null;
  try {
    r = await fetch("/admin/engine/" + action, { method: "POST", headers });
    try { j = await r.json(); } catch (e) { /* 同上 */ }
  } catch (e) {
    engShowErr("eng-err", fmtMsg("eng.e_" + action, { m: e.message }));
    return;
  }
  if (!r.ok) {
    if (r.status === 401) engEl("cfg-key").classList.remove("hidden");
    const m = (j && j.error && (j.error.message || j.error.type)) ||
      fmtMsg("ovr.e_http", { s: r.status });
    engShowErr("eng-err", fmtMsg("eng.e_" + action, { m }));
    return;
  }
  engHideErr("eng-err");
  if (j && j.state) renderEngStatus(j);
  refreshEngineStatus();
}

function renderEngStatus(st) {
  engLastStatus = st;
  const cls = st.state === "running" ? "ok" : st.state === "exited" ? "err"
    : st.state === "starting" ? "busy" : "";
  const stateSpan = document.createElement("span");
  stateSpan.appendChild(pill(t("eng.state_" + st.state), cls));
  if (st.external) {
    const m = document.createElement("span");
    m.className = "muted small";
    m.textContent = " · " + t("eng.external_tag");
    stateSpan.appendChild(m);
  }
  const rows = [[t("eng.status"), stateSpan]];
  if (st.pid) rows.push([t("eng.pid"), String(st.pid)]);
  if (st.state === "exited") rows.push(["exit code", String(st.exit_code)]);
  if (st.log) rows.push([t("eng.log"), st.log]);
  if (st.detail) rows.push(["detail", st.detail]);
  fillKv(engEl("eng-status"), rows);
  const can = !!st.can_control;
  engEl("eng-actions").classList.toggle("hidden", !can);
  engEl("eng-nocontrol").classList.toggle("hidden", can);
  const active = st.state === "running" || st.state === "starting";
  engEl("eng-start").disabled = !can || active;
  engEl("eng-stop").disabled = !can || !active || !!st.external;
}

// 语言切换后重渲染本页的动态文案（徽章/状态 pill/文件下拉标签）
function renderEngineAll() {
  if (!cfgBuilt) return;
  for (const key of Object.keys(cfgBadges)) cfgBadges[key].textContent = t("cfg.env_badge");
  const kv = cfgInputs["KV_PAGED"];
  if (kv) {
    kv.options[0].textContent = t("ovr.bool_on");
    kv.options[1].textContent = t("ovr.bool_off");
  }
  for (const key of CFG_FILE_KEYS)
    if (cfgInputs[key]) cfgInputs[key].placeholder = t("cfg.pick_ph");
  for (const b of document.querySelectorAll(".cfg-browse")) b.textContent = t("cfg.browse");
  const wv = engEl("cfg-weights-ver");
  if (wv) for (const o of wv.options) o.textContent = t("cfg.ver_" + o.value);
  if (fpickKey) renderFilePicker();
  if (engLastStatus) renderEngStatus(engLastStatus);
}

async function refreshEngineStatus() {
  try {
    renderEngStatus(await getJSON("/admin/engine/status"));
    stampNow(engEl("eng-stamp"));
  } catch (e) { stampErr(engEl("eng-stamp"), e); }
}

engEl("eng-start").addEventListener("click", () => engineAction("start"));
engEl("eng-stop").addEventListener("click", () => {
  if (confirm(t("eng.confirm_stop"))) engineAction("stop");
});
engEl("cfg-save").addEventListener("click", postConfig);
engEl("cfg-reload").addEventListener("click", () => {
  if (cfgDirty && !confirm(t("cfg.confirm_reload"))) return;
  loadConfig();
});

function enterEngine() {
  if (!cfgBuilt) buildCfgForm();
  loadConfig();
  refreshEngineStatus();
  if (engTimer) clearInterval(engTimer);
  engTimer = setInterval(() => {
    if (currentPage !== "engine" || document.hidden) return;
    refreshEngineStatus();
  }, 2000);
  // 离开页面时停掉轮询
  const stop = () => { if (engTimer) { clearInterval(engTimer); engTimer = null; } };
  const obs = new MutationObserver(() => {
    if (currentPage !== "engine") { stop(); obs.disconnect(); }
  });
  obs.observe(document.getElementById("page-engine"),
    { attributes: true, attributeFilter: ["class"] });
}

// ---------------- 启动 ----------------

initTheme();
applyLang();
showPage(location.hash.replace(/^#\/?/, "") || "overview");
