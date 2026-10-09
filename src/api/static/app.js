"use strict";
/* gdec-api 控制台 v1（中英双语）
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
    "hint.busy": "引擎忙闲",
    "hint.theme": "切换明暗主题",
    "nav.overview": "总览", "nav.usage": "用量", "nav.requests": "请求",
    "common.loading": "加载中…",
    "common.refresh": "刷新", "common.all": "全部", "common.read_failed": "读取失败",
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
    "mem.vram": "显存", "mem.resident": "引擎驻留",
    "mem.vram_peak": "引擎显存峰值", "mem.rss": "常驻主机内存 RSS",
    "mem.pinned": "锁页内存", "mem.accessible": "可访问已提交",
    "mem.offline": "引擎未连接（/memory 不可用）",
    "ovr.on": "开", "ovr.off": "关",
    "usage.range": "范围", "usage.d7": "近 7 天", "usage.d14": "近 14 天",
    "usage.d30": "近 30 天", "usage.daily": "按天明细", "usage.day": "日期",
    "usage.cached": "缓存命中",
    "usage.prompt_tokens": "prompt tokens", "usage.output_tokens": "output tokens",
    "usage.empty": "范围内没有请求记录",
    "usage.note": "v1 数据来自 /reqstat/tail 最近 1000 条记录，更早的历史需要后端按天聚合端点后接入。",
    "req.from": "起", "req.to": "止", "req.size": "每页", "req.clear": "清除筛选",
    "req.time": "时间", "req.cache": "缓存", "req.accept": "接受率",
    "req.prev_page": "上一页", "req.next_page": "下一页",
    "req.note_server": "服务端分页（/reqstat/page），可翻阅全部历史；日期与 finish/drafter 为前端过滤（过滤时仅覆盖最近 1000 条）。",
    "req.note_filtered": "已设筛选：后端不支持按条件检索，仅在 /reqstat/tail 最近 1000 条内过滤分页。",
    "nav.sample": "采样",
    "ovr.title": "采样 / 思考参数（服务端覆盖）",
    "ovr.hint": "跟随客户端：不干预。默认值：只在客户端没传该字段时使用。" +
      "强制：不管客户端传什么都改成这个值（max_tokens 的“强制”是上限，超过才压到此值）。" +
      "对 /v1/chat/completions、/v1/responses、/v1/completions 都生效，思考相关字段对 completions 不起作用。" +
      "被改写的字段会写进响应头 X-Gdec-Overrides。保存后立即生效，并持久化到服务端文件，重启后仍有效。",
    "ovr.th_param": "参数", "ovr.th_mode": "模式", "ovr.th_value": "值", "ovr.th_note": "说明",
    "ovr.mode_client": "跟随客户端", "ovr.mode_default": "默认值", "ovr.mode_force": "强制",
    "ovr.bool_on": "开 (true)", "ovr.bool_off": "关 (false)",
    "ovr.save": "保存并生效", "ovr.reload": "从服务端重新读取", "ovr.clear": "全部改为跟随客户端",
    "ovr.key_ph": "管理密钥 (GDEC_API_ADMIN_KEY)",
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
  },
  en: {
    "app.title": "Qwen-Flash-Server console",
    "hint.busy": "engine busy/idle",
    "hint.theme": "toggle light/dark theme",
    "nav.overview": "Overview", "nav.usage": "Usage", "nav.requests": "Requests",
    "common.loading": "Loading…",
    "common.refresh": "Refresh", "common.all": "All", "common.read_failed": "read failed",
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
    "mem.vram": "VRAM", "mem.resident": "Engine resident",
    "mem.vram_peak": "Engine VRAM peak", "mem.rss": "RSS (host)",
    "mem.pinned": "Pinned memory", "mem.accessible": "Accessible committed",
    "mem.offline": "engine offline (/memory unavailable)",
    "ovr.on": "on", "ovr.off": "off",
    "usage.range": "Range", "usage.d7": "7 days", "usage.d14": "14 days",
    "usage.d30": "30 days", "usage.daily": "Per day", "usage.day": "Date",
    "usage.cached": "Cache hit",
    "usage.prompt_tokens": "prompt tokens", "usage.output_tokens": "output tokens",
    "usage.empty": "no requests in range",
    "usage.note": "v1: newest 1000 records via /reqstat/tail; older history needs a backend per-day aggregation endpoint.",
    "req.from": "From", "req.to": "To", "req.size": "Per page", "req.clear": "Clear filters",
    "req.time": "Time", "req.cache": "Cache", "req.accept": "Acceptance",
    "req.prev_page": "Prev", "req.next_page": "Next",
    "req.note_server": "Server-side paging (/reqstat/page) over the full history; date and finish/drafter filters are client-side (newest 1000 records only).",
    "req.note_filtered": "Filters active: the backend has no conditional search, so filtering covers only the newest 1000 records from /reqstat/tail.",
    "nav.sample": "Sampling",
    "ovr.title": "Sampling / thinking parameters (server overrides)",
    "ovr.hint": "Follow client: no interference. Default: used only when the client omitted the field. " +
      "Force: replaces whatever the client sent (for max_tokens it is a cap: larger values clamp to it). " +
      "Applies to /v1/chat/completions, /v1/responses and /v1/completions; thinking fields have no " +
      "effect on completions. Rewritten fields are reported in the X-Gdec-Overrides response header. " +
      "Saving takes effect immediately and is persisted server-side, surviving restarts.",
    "ovr.th_param": "Parameter", "ovr.th_mode": "Mode", "ovr.th_value": "Value", "ovr.th_note": "Note",
    "ovr.mode_client": "Follow client", "ovr.mode_default": "Default", "ovr.mode_force": "Force",
    "ovr.bool_on": "on (true)", "ovr.bool_off": "off (false)",
    "ovr.save": "Save & apply", "ovr.reload": "Reload from server", "ovr.clear": "All follow client",
    "ovr.key_ph": "Admin key (GDEC_API_ADMIN_KEY)",
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
  },
};

let LANG = null;
try { LANG = localStorage.getItem("gdec-lang"); } catch (e) { /* 隐私模式 */ }
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
  btn.textContent = LANG === "zh" ? "EN" : "中文";
  btn.title = LANG === "zh" ? "Switch to English" : "切换到中文";
}

document.getElementById("lang-btn").addEventListener("click", () => {
  LANG = LANG === "zh" ? "en" : "zh";
  try { localStorage.setItem("gdec-lang", LANG); } catch (e) { /* 同上 */ }
  applyLang();
  // 动态文案（徽章/表格外新增文本/时间戳）随当前页面重渲染，不重新发请求
  if (currentPage === "overview") { refreshHealth(); refreshMemory(); refreshPower(); }
  else if (currentPage === "usage") { if (reqCache.records.length) renderUsage(); }
  else if (currentPage === "requests") renderRequests();
  else if (currentPage === "sample") syncSampleLang();
});

// ---------------- 主题 ----------------

function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  document.getElementById("theme-btn").textContent = t === "dark" ? "☾" : "☀";
  try { localStorage.setItem("gdec-theme", t); } catch (e) { /* 隐私模式 */ }
}
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("gdec-theme"); } catch (e) { /* 同上 */ }
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

const PAGES = ["overview", "usage", "requests", "sample"];
let currentPage = "";
const onEnter = { overview: enterOverview, usage: enterUsage, requests: enterRequests,
  sample: enterSample };

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

async function refreshHealth() {
  const dot = document.getElementById("busy-dot");
  try {
    const h = await getJSON("/health");
    dot.className = "brand-dot " + (h.busy ? "busy" : "idle");
    const busy = h.busy
      ? pill(t("health.busy_yes"), "busy")
      : pill(t("health.busy_no"), "ok");
    fillKv(document.getElementById("ov-status"), [
      [t("health.status"), pill(h.status === "ok" ? "ok" : h.status, "ok")],
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
    dot.className = "brand-dot down";
    stampErr(document.getElementById("overview-stamp"), e);
  }
}

async function refreshMemory() {
  const bars = document.getElementById("mem-bars");
  const stamp = document.getElementById("mem-stamp");
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
    stampNow(stamp);
  } catch (e) {
    bars.textContent = t("mem.offline");
    bars.classList.add("muted");
    stamp.textContent = "";
  }
}

async function refreshPower() {
  const el = document.getElementById("ov-power");
  const stamp = document.getElementById("power-stamp");
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
    stampNow(stamp);
  } catch (e) {
    fillKv(el, [[t("ov.power"), t("common.read_failed")]]);
    stamp.textContent = "";
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

  const tbody = document.querySelector("#usage-table tbody");
  tbody.textContent = "";
  for (const g of usageAgg.slice().reverse()) {
    const tr = document.createElement("tr");
    for (const [v, right] of [[g.day, false], [fmtInt(g.requests), true], [fmtInt(g.prompt), true],
        [fmtInt(g.cached), true], [fmtInt(g.gen), true]]) {
      const td = document.createElement("td");
      if (right) td.className = "r";
      td.textContent = v;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }

  drawUsageChart();
  stampNow(document.getElementById("usage-stamp"));
  document.getElementById("usage-note").textContent = t("usage.note");
}

function drawUsageChart() {
  const cv = document.getElementById("usage-chart");
  if (document.getElementById("page-usage").classList.contains("hidden")) return;
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = 260;
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
let resizeT = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeT);
  resizeT = setTimeout(() => { if (currentPage === "usage") drawUsageChart(); }, 150);
});

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
      note.textContent = t("req.note_filtered");
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
      note.textContent = t("req.note_server");
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

// ---------------- 启动 ----------------

initTheme();
applyLang();
showPage(location.hash.replace(/^#\/?/, "") || "overview");
