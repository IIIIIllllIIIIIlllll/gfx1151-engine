// dashboard_html.inc — gdec-api 自带的控制台（由 GET / 和 GET /dashboard 返回）。
//
// 单文件：CSS/JS 全部内联，不引用任何外部资源（无 CDN，离线/局域网可用）。
// 页面用同源 fetch 访问：
//   GET /health  每 2 秒   —— API 进程自身的状态，不经过 engine
//   GET /memory  每 10 秒  —— engine 内存记账（占用推理槽位，间隔勿低于 ~1 秒）
//   GET/POST /admin/overrides —— 服务端采样/思考参数覆盖表（打开页面时读一次）
//   POST /v1/chat/completions —— "对话测试"面板，以普通客户端身份流式调用
// /memory 在 engine 不可达时返回 500；页面把错误显示在面板内并保留上次数值，
// 继续按周期重试，不会白屏。

static const char kDashboardHtml[] = R"HTMLDOC(<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>gdec-api 控制台</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --border: #30363d;
    --fg: #e6edf3; --dim: #8b949e;
    --ok: #3fb950; --warn: #d29922; --err: #f85149;
    --c-committed: #58a6ff; --c-rss: #bc8cff;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 14px/1.5 "Segoe UI", system-ui, -apple-system, sans-serif;
  }
  .mono { font-family: Consolas, "Cascadia Mono", "Courier New", monospace; }
  header {
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    padding: 12px 20px; border-bottom: 1px solid var(--border);
    background: var(--panel);
  }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  header nav { display: flex; gap: 10px; font-size: 13px; }
  header nav a { color: var(--dim); text-decoration: none; }
  header nav a:hover { color: var(--fg); }
  #target { color: var(--dim); font-size: 13px; }
  .chip {
    padding: 2px 10px; border-radius: 999px; font-size: 12px;
    border: 1px solid var(--border); color: var(--dim);
  }
  .chip.ok   { color: var(--ok);   border-color: var(--ok); }
  .chip.warn { color: var(--warn); border-color: var(--warn); }
  .chip.err  { color: var(--err);  border-color: var(--err); }
  .btn, #pause {
    background: none; color: var(--dim); cursor: pointer;
    border: 1px solid var(--border); border-radius: 6px; padding: 4px 12px;
    font: inherit;
  }
  #pause { margin-left: auto; }
  .btn:hover, #pause:hover { color: var(--fg); border-color: var(--dim); }
  .btn.primary { color: #fff; background: #238636; border-color: #2ea043; }
  .btn.primary:hover { background: #2ea043; }
  .btn:disabled { opacity: 0.45; cursor: default; }
  main { max-width: 960px; margin: 0 auto; padding: 16px 20px 40px; }
  .panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px; margin-bottom: 16px;
  }
  .panel h2 { font-size: 13px; margin: 0 0 12px; color: var(--dim); font-weight: 600; }
  .panel .meta { float: right; font-weight: 400; font-size: 12px; }
  .hint { color: var(--dim); font-size: 12px; margin: -4px 0 12px; }
  .kv { display: flex; gap: 28px; flex-wrap: wrap; }
  .kv > div > .k { color: var(--dim); font-size: 12px; }
  .kv > div > .v { font-size: 15px; }
  .hero { display: flex; align-items: baseline; gap: 14px; margin-bottom: 14px; }
  .hero .num { font-size: 34px; font-weight: 700; }
  .hero .lab { color: var(--dim); font-size: 13px; }
  .bar {
    height: 10px; border-radius: 5px; background: #21262d;
    overflow: hidden; margin: 4px 0 4px;
  }
  .bar > i { display: block; height: 100%; background: var(--c-committed); }
  .barlab { display: flex; justify-content: space-between; color: var(--dim); font-size: 12px; }
  .grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 12px; margin-top: 14px;
  }
  .cell { background: #0d1117; border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; }
  .cell .k { color: var(--dim); font-size: 12px; }
  .cell .v { font-size: 18px; font-weight: 600; }
  .cell .s { color: var(--dim); font-size: 12px; }
  .stale { opacity: 0.45; }
  .errmsg {
    display: none; margin-top: 12px; padding: 8px 12px; border-radius: 6px;
    background: rgba(248,81,73,0.1); border: 1px solid var(--err); color: var(--err);
    font-size: 13px; word-break: break-all;
  }
  .errmsg.show { display: block; }
  canvas { width: 100%; height: 120px; margin-top: 14px; display: block; }
  .legend { display: flex; gap: 18px; color: var(--dim); font-size: 12px; margin-top: 4px; }
  .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; }
  details { margin-top: 14px; }
  summary { cursor: pointer; color: var(--dim); font-size: 12px; }
  pre {
    background: #0d1117; border: 1px solid var(--border); border-radius: 6px;
    padding: 10px; overflow-x: auto; font-size: 12px; margin: 8px 0 0;
    white-space: pre-wrap; word-break: break-all;
  }
  input, select, textarea {
    background: #0d1117; color: var(--fg); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 8px; font: inherit;
  }
  input:disabled, select:disabled { opacity: 0.4; }
  input[type=number] { width: 110px; }
  input[type=checkbox] { width: auto; }
  textarea { width: 100%; resize: vertical; }
  table.ovr { width: 100%; border-collapse: collapse; }
  table.ovr th { text-align: left; color: var(--dim); font-weight: 400; font-size: 12px; padding: 4px 8px; }
  table.ovr td { padding: 5px 8px; border-top: 1px solid var(--border); vertical-align: middle; }
  table.ovr td.note { color: var(--dim); font-size: 12px; }
  table.ovr tr.on td:first-child { color: var(--warn); }
  table.ovr tr.def td:first-child { color: var(--c-committed); }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 12px; }
  #ovr-status, #chat-status { color: var(--dim); font-size: 12px; }
  .cparams { display: flex; gap: 14px; flex-wrap: wrap; align-items: flex-end; margin-bottom: 10px; }
  .cparams label { display: flex; flex-direction: column; color: var(--dim); font-size: 12px; gap: 2px; }
  .cparams input[type=number] { width: 90px; }
  #c-log { display: flex; flex-direction: column; gap: 10px; margin: 10px 0; }
  .msg { border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; background: #0d1117; }
  .msg .role { color: var(--dim); font-size: 12px; margin-bottom: 4px; }
  .msg.user { border-color: #1f6feb; }
  .msg .body { white-space: pre-wrap; word-break: break-word; }
  .msg .think { white-space: pre-wrap; word-break: break-word; color: var(--dim); font-size: 13px;
                border-left: 2px solid var(--border); padding-left: 10px; margin: 4px 0 8px; }
  .msg .stat { color: var(--dim); font-size: 12px; margin-top: 6px; }
  .msg .ovr-note { color: var(--warn); font-size: 12px; margin-bottom: 4px; }
</style>
</head>
<body>
<header>
  <h1>gdec-api 控制台</h1>
  <span class="mono" id="target"></span>
  <span class="chip" id="conn">连接中…</span>
  <nav><a href="#health">状态</a><a href="#ovr">参数覆盖</a><a href="#chat">对话测试</a><a href="#mem">内存</a></nav>
  <button id="pause" type="button">⏸ 暂停刷新</button>
</header>
<main>
  <section class="panel" id="health">
    <h2>健康状态 <span class="meta mono" id="health-meta"></span></h2>
    <div class="kv">
      <div><div class="k">状态</div><div class="v"><span class="chip" id="h-status">…</span></div></div>
      <div><div class="k">模型</div><div class="v mono" id="h-model">-</div></div>
      <div><div class="k">上下文</div><div class="v mono" id="h-ctx">-</div></div>
      <div><div class="k">进行中 (in_flight)</div><div class="v mono" id="h-inflight">-</div></div>
      <div><div class="k">已持续 (busy_for)</div><div class="v mono" id="h-busyfor">-</div></div>
      <div><div class="k">排队 (queued)</div><div class="v mono" id="h-queued">-</div></div>
      <div><div class="k">服务端覆盖</div><div class="v mono" id="h-ovr">-</div></div>
    </div>
    <div class="errmsg" id="health-err"></div>
  </section>

  <section class="panel" id="ovr">
    <h2>采样 / 思考参数（服务端覆盖） <span class="meta mono" id="ovr-meta"></span></h2>
    <p class="hint">
      <b>跟随客户端</b>：不干预。<b>默认值</b>：只在客户端没传该字段时使用。
      <b>强制</b>：不管客户端传什么都改成这个值（max_tokens 的"强制"是上限，超过才压到此值）。
      对 /v1/chat/completions、/v1/responses、/v1/completions 都生效，思考相关字段对 completions 不起作用。
      被改写的字段会写进响应头 <span class="mono">X-Gdec-Overrides</span>。保存后立即生效，并持久化到服务端文件，重启后仍有效。
    </p>
    <table class="ovr">
      <thead><tr><th>参数</th><th>模式</th><th>值</th><th>说明</th></tr></thead>
      <tbody id="ovr-body"></tbody>
    </table>
    <div class="row">
      <input id="ovr-key" type="password" placeholder="管理密钥 (GDEC_API_ADMIN_KEY)" style="display:none; width:240px">
      <button id="ovr-save" class="btn primary" type="button">保存并生效</button>
      <button id="ovr-reload" class="btn" type="button">从服务端重新读取</button>
      <button id="ovr-clear" class="btn" type="button">全部改为跟随客户端</button>
      <span id="ovr-status"></span>
    </div>
    <div class="errmsg" id="ovr-err"></div>
  </section>

  <section class="panel" id="chat">
    <h2>对话测试 <span class="meta mono" id="chat-meta"></span></h2>
    <p class="hint">
      以普通客户端身份流式调用 /v1/chat/completions。下面是"客户端提交的值"，留空或"不传"表示请求里不带该字段。
      服务端覆盖生效时，回复上方会列出被改写的字段。引擎是单槽位，其他客户端正在生成时，这里会排队等待。
    </p>
    <div class="cparams">
      <label>思考开关<select id="c-think"><option value="">不传</option><option value="true">开</option><option value="false">关</option></select></label>
      <label>思考强度<select id="c-effort"><option value="">不传</option><option>low</option><option>medium</option><option>xhigh</option></select></label>
      <label>temperature<input id="c-temp" type="number" min="0" max="2" step="0.05" placeholder="不传"></label>
      <label>top_p<input id="c-topp" type="number" min="0" max="1" step="0.01" placeholder="不传"></label>
      <label>top_k<input id="c-topk" type="number" min="0" step="1" placeholder="不传"></label>
      <label>min_p<input id="c-minp" type="number" min="0" max="1" step="0.01" placeholder="不传"></label>
      <label>presence<input id="c-pres" type="number" min="-2" max="2" step="0.1" placeholder="不传"></label>
      <label>max_tokens<input id="c-max" type="number" min="1" step="1" placeholder="不传"></label>
      <label>seed<input id="c-seed" type="number" min="0" step="1" placeholder="随机"></label>
    </div>
    <textarea id="c-system" rows="2" placeholder="system 提示词（可选）"></textarea>
    <div id="c-log"></div>
    <textarea id="c-input" rows="3" placeholder="输入消息，Ctrl+Enter 发送"></textarea>
    <div class="row">
      <button id="c-send" class="btn primary" type="button">发送</button>
      <button id="c-stop" class="btn" type="button" disabled>停止</button>
      <button id="c-clear" class="btn" type="button">清空对话</button>
      <span id="chat-status"></span>
    </div>
    <div class="errmsg" id="chat-err"></div>
  </section>

  <section class="panel" id="mem">
    <h2>内存记账 <span class="meta mono" id="mem-meta"></span></h2>
    <div id="mem-data">
      <div class="hero">
        <span class="num mono" id="m-hero" title="gpu_accessible_committed_bytes">-</span>
        <span class="lab">GPU 可访问记账（committed）<br>
          <span id="m-accounting" style="font-size:12px"></span></span>
      </div>
      <div class="barlab"><span>HIP 显存已用</span><span class="mono" id="m-hip-pct"></span></div>
      <div class="bar"><i id="m-hip-bar" style="width:0%"></i></div>
      <div class="barlab"><span id="m-hip-used"></span><span id="m-hip-total"></span></div>
      <div class="grid" id="mem-grid"></div>
      <canvas id="spark"></canvas>
      <div class="legend">
        <span><i style="background:var(--c-committed)"></i>committed</span>
        <span><i style="background:var(--c-rss)"></i>进程 RSS</span>
        <span>最近 10 分钟（每 10 秒一个采样点）</span>
      </div>
    </div>
    <div class="errmsg" id="mem-err"></div>
    <details><summary>原始 JSON（/health 与 /memory）</summary>
      <pre id="raw-health">-</pre>
      <pre id="raw-mem">-</pre>
    </details>
  </section>
</main>
<script>
"use strict";
const HEALTH_MS = 2000;   // /health 轮询间隔
const MEM_MS    = 10000;  // /memory 轮询间隔（查询占用推理槽位，勿低于 ~1 秒）
const TIMEOUT_MS = 5000;
const HIST_CAP = 60;      // 60 点 * 10s = 10 分钟趋势

let paused = false;
const hist = { committed: [], rss: [] };
const $ = id => document.getElementById(id);

function fmtBytes(n) {
  if (n == null || isNaN(n)) return "-";
  const g = 1073741824, m = 1048576, k = 1024;
  if (Math.abs(n) >= g) return (n / g).toFixed(2) + " GiB";
  if (Math.abs(n) >= m) return (n / m).toFixed(1) + " MiB";
  if (Math.abs(n) >= k) return (n / k).toFixed(1) + " KiB";
  return n + " B";
}
function hhmmss() {
  const d = new Date(), p = v => String(v).padStart(2, "0");
  return p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
}
function setChip(el, cls, text) { el.className = "chip " + cls; el.textContent = text; }

// 从 OpenAI 风格错误信封里取消息
function errText(j, status) {
  return (j && j.error && (j.error.message || j.error.type)) || ("HTTP " + status);
}

// fetch 一个端点：返回解析后的 JSON；HTTP 错误抛 Error（附 status）
async function grab(url, opts) {
  const c = new AbortController();
  const t = setTimeout(() => c.abort(), TIMEOUT_MS);
  try {
    const r = await fetch(url, Object.assign({ signal: c.signal }, opts || {}));
    const text = await r.text();
    let j = null;
    try { j = JSON.parse(text); } catch (e) { /* 非 JSON 按状态码报错 */ }
    if (!r.ok) { const e = new Error(errText(j, r.status)); e.status = r.status; throw e; }
    return j;
  } catch (e) {
    if (e.name === "AbortError") throw new Error("请求超时");
    throw e;
  } finally { clearTimeout(t); }
}

function showErr(panelId, err) {
  const el = $(panelId);
  el.textContent = hhmmss() + "  " + err;
  el.classList.add("show");
}
function hideErr(panelId) { $(panelId).classList.remove("show"); }

// ------------------------------------------------------------ 健康状态 --
function ovrSummary(t) {
  const keys = Object.keys(t || {});
  if (!keys.length) return "无（全部跟随客户端）";
  return keys.map(k => k + (t[k].mode === "force" ? "=" : "≈") + JSON.stringify(t[k].value)).join("  ");
}

function renderHealth(j) {
  const s = $("h-status");
  if (j.busy) setChip(s, "warn", "busy");
  else if (j.status === "ok") setChip(s, "ok", "ok");
  else setChip(s, "err", String(j.status));
  $("h-model").textContent = j.model ?? "-";
  $("h-ctx").textContent = j.context ?? "-";
  $("h-inflight").textContent = j.in_flight ?? "-";
  const bf = j.busy_for_s;
  $("h-busyfor").textContent = (bf == null) ? "-" : (bf.toFixed(1) + "s");
  $("h-queued").textContent = j.queued ?? "-";
  $("h-ovr").textContent = ovrSummary(j.server_overrides);
  $("h-ovr").title = "= 强制，≈ 默认值";
  $("raw-health").textContent = JSON.stringify(j, null, 1);
}

// ------------------------------------------------------------ 参数覆盖 --
// type: bool / effort / num / int；def 是切到"默认值/强制"时预填的值
const OVR_FIELDS = [
  { k: "enable_thinking", label: "思考开关", type: "bool", def: true,
    note: "关 = 模板关闭思考，不输出 reasoning_content；客户端不传时为开" },
  { k: "reasoning_effort", label: "思考强度", type: "effort", def: "xhigh",
    note: "low / medium / xhigh（客户端传 high→xhigh，minimal→low）；仅思考开启时有效" },
  { k: "preserve_thinking", label: "保留历史思考", type: "bool", def: true,
    note: "多轮时历史 assistant 的思考是否放回提示词；不传时保留" },
  { k: "temperature", label: "temperature", type: "num", min: 0, max: 2, step: 0.05, def: 1.0,
    note: "0 = 贪心；客户端不传时为 1.0。强制 0 时会去掉请求里的 logprobs" },
  { k: "top_p", label: "top_p", type: "num", min: 0, max: 1, step: 0.01, def: 0.95,
    note: "客户端不传时为 0.95" },
  { k: "top_k", label: "top_k", type: "int", min: 0, max: 2147483647, step: 1, def: 20,
    note: "0 = 不限；客户端不传时为 20" },
  { k: "min_p", label: "min_p", type: "num", min: 0, max: 1, step: 0.01, def: 0,
    note: "客户端不传时为 0" },
  { k: "presence_penalty", label: "presence_penalty", type: "num", min: -2, max: 2, step: 0.1, def: 0,
    note: "-2 ~ 2" },
  { k: "frequency_penalty", label: "frequency_penalty", type: "num", min: -2, max: 2, step: 0.1, def: 0,
    note: "-2 ~ 2" },
  { k: "max_tokens", label: "max_tokens", type: "int", min: 1, max: 2147483647, step: 1, def: 32768,
    note: "默认值 = 客户端没给输出上限时用它；强制 = 输出上限，客户端给得更大时压到此值" },
];

function ovrValueInput(f) {
  if (f.type === "bool") {
    const s = document.createElement("select");
    s.innerHTML = '<option value="true">开 (true)</option><option value="false">关 (false)</option>';
    return s;
  }
  if (f.type === "effort") {
    const s = document.createElement("select");
    s.innerHTML = "<option>low</option><option>medium</option><option>xhigh</option>";
    return s;
  }
  const i = document.createElement("input");
  i.type = "number"; i.min = f.min; i.max = f.max; i.step = f.step;
  return i;
}

function buildOvrTable() {
  const tb = $("ovr-body");
  tb.innerHTML = "";
  for (const f of OVR_FIELDS) {
    const tr = document.createElement("tr");
    tr.id = "ovr-row-" + f.k;
    const name = document.createElement("td");
    name.className = "mono"; name.textContent = f.label;
    if (f.label !== f.k) name.title = f.k;
    const modeTd = document.createElement("td");
    const mode = document.createElement("select");
    mode.innerHTML = '<option value="client">跟随客户端</option><option value="default">默认值</option>' +
                     '<option value="force">强制</option>';
    modeTd.appendChild(mode);
    const valTd = document.createElement("td");
    const val = ovrValueInput(f);
    valTd.appendChild(val);
    const note = document.createElement("td");
    note.className = "note"; note.textContent = f.note;
    tr.append(name, modeTd, valTd, note);
    tb.appendChild(tr);
    f.modeEl = mode; f.valEl = val; f.row = tr;
    mode.addEventListener("change", () => { syncRow(f); markDirty(); });
    val.addEventListener("input", markDirty);
    val.addEventListener("change", markDirty);
  }
}

function setVal(f, v) { f.valEl.value = String(v); }
function getVal(f) {
  const raw = f.valEl.value;
  if (f.type === "bool") return raw === "true";
  if (f.type === "effort") return raw;
  if (raw.trim() === "") throw new Error(f.k + " 需要填写数值");
  const n = Number(raw);
  if (!isFinite(n)) throw new Error(f.k + " 不是有效数字");
  if (f.type === "int" && !Number.isInteger(n)) throw new Error(f.k + " 必须是整数");
  if (n < f.min || n > f.max) throw new Error(f.k + " 超出范围 " + f.min + " ~ " + f.max);
  return n;
}
function syncRow(f) {
  const m = f.modeEl.value;
  f.valEl.disabled = m === "client";
  f.row.className = m === "force" ? "on" : m === "default" ? "def" : "";
}

let ovrDirty = false;
function markDirty() { ovrDirty = true; $("ovr-status").textContent = "有未保存的修改"; }

function fillOvr(state) {
  const t = state.overrides || {};
  for (const f of OVR_FIELDS) {
    const e = t[f.k];
    f.modeEl.value = e ? e.mode : "client";
    setVal(f, e ? e.value : f.def);
    syncRow(f);
  }
  $("ovr-key").style.display = state.admin_key_required ? "" : "none";
  $("ovr-meta").textContent = state.persist_path ? ("持久化文件：" + state.persist_path) : "仅内存（未持久化）";
  ovrDirty = false;
}

async function loadOvr() {
  try {
    fillOvr(await grab("/admin/overrides"));
    hideErr("ovr-err");
    $("ovr-status").textContent = "已读取 " + hhmmss();
  } catch (e) { showErr("ovr-err", "读取失败：" + e.message); }
}

function collectOvr() {
  const t = {};
  for (const f of OVR_FIELDS) {
    const m = f.modeEl.value;
    if (m === "client") continue;
    t[f.k] = { mode: m, value: getVal(f) };
  }
  return t;
}

async function saveOvr(table) {
  const headers = { "Content-Type": "application/json" };
  const key = $("ovr-key").value;
  if (key) headers["X-Admin-Key"] = key;
  try {
    const j = await grab("/admin/overrides", { method: "POST", headers, body: JSON.stringify(table) });
    fillOvr(j);
    hideErr("ovr-err");
    $("ovr-status").textContent = (j.saved ? "已保存并生效 " : "已生效，但未能写入文件 ") + hhmmss();
    if (!j.saved) showErr("ovr-err", "持久化失败：" + (j.save_error || "?") + "（重启后会丢失）");
    tickHealth();
  } catch (e) {
    if (e.status === 401) $("ovr-key").style.display = "";
    showErr("ovr-err", "保存失败：" + e.message);
  }
}

$("ovr-save").addEventListener("click", () => {
  let t;
  try { t = collectOvr(); } catch (e) { showErr("ovr-err", e.message); return; }
  saveOvr(t);
});
$("ovr-reload").addEventListener("click", () => {
  if (ovrDirty && !confirm("丢弃未保存的修改？")) return;
  loadOvr();
});
$("ovr-clear").addEventListener("click", () => {
  if (!confirm("清空服务端覆盖表，所有参数改为跟随客户端？")) return;
  saveOvr({});
});

// ------------------------------------------------------------ 对话测试 --
let convo = [];          // 已完成的轮次：{role, content, reasoning_content?}
let chatAbort = null;

function msgBox(role) {
  const d = document.createElement("div");
  d.className = "msg " + role;
  const r = document.createElement("div");
  r.className = "role"; r.textContent = role;
  d.appendChild(r);
  $("c-log").appendChild(d);
  return d;
}
function addPart(box, cls, text) {
  const e = document.createElement("div");
  e.className = cls; e.textContent = text || "";
  box.appendChild(e);
  return e;
}
function numParam(id) {
  const v = $(id).value.trim();
  return v === "" ? undefined : Number(v);
}

function buildChatBody() {
  const msgs = [];
  const sys = $("c-system").value.trim();
  if (sys) msgs.push({ role: "system", content: sys });
  const body = { messages: msgs.concat(convo), stream: true, stream_options: { include_usage: true } };
  const think = $("c-think").value;
  if (think) body.enable_thinking = think === "true";
  const effort = $("c-effort").value;
  if (effort) body.reasoning_effort = effort;
  const nums = { temperature: "c-temp", top_p: "c-topp", top_k: "c-topk", min_p: "c-minp",
                 presence_penalty: "c-pres", max_tokens: "c-max", seed: "c-seed" };
  for (const [k, id] of Object.entries(nums)) {
    const v = numParam(id);
    if (v !== undefined) body[k] = v;
  }
  return body;
}

function setBusy(b) {
  $("c-send").disabled = b; $("c-stop").disabled = !b; $("c-clear").disabled = b;
}

async function sendChat() {
  const text = $("c-input").value;
  if (!text.trim() || chatAbort) return;
  hideErr("chat-err");
  convo.push({ role: "user", content: text });
  addPart(msgBox("user"), "body", text);
  $("c-input").value = "";
  const body = buildChatBody();
  const box = msgBox("assistant");
  const note = addPart(box, "ovr-note", "");
  const think = addPart(box, "think", "");
  const ans = addPart(box, "body", "");
  const stat = addPart(box, "stat", "等待中…");
  think.style.display = "none";
  let reasoning = "", content = "", usage = null, finish = null;
  const t0 = performance.now();
  let tFirst = 0;
  chatAbort = new AbortController();
  setBusy(true);
  $("chat-status").textContent = "生成中…";
  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body), signal: chatAbort.signal });
    const ov = r.headers.get("X-Gdec-Overrides");
    note.textContent = ov ? ("服务端覆盖了：" + ov) : "";
    if (!r.ok) {
      let j = null; try { j = await r.json(); } catch (e) { /* 忽略 */ }
      throw new Error(errText(j, r.status));
    }
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "", done = false;
    while (!done) {
      const { value, done: eof } = await reader.read();
      if (eof) break;
      buf += dec.decode(value, { stream: true });
      let cut;
      while ((cut = buf.indexOf("\n\n")) >= 0) {
        const frame = buf.slice(0, cut); buf = buf.slice(cut + 2);
        for (const line of frame.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (data === "[DONE]") { done = true; break; }
          let j; try { j = JSON.parse(data); } catch (e) { continue; }
          if (j.error) throw new Error(j.error.message || "stream error");
          if (j.usage) usage = j.usage;
          const ch = j.choices && j.choices[0];
          if (!ch) continue;
          if (ch.finish_reason) finish = ch.finish_reason;
          const d = ch.delta || {};
          if (d.reasoning_content) {
            if (!tFirst) tFirst = performance.now();
            reasoning += d.reasoning_content; think.textContent = reasoning; think.style.display = "";
          }
          if (d.content) {
            if (!tFirst) tFirst = performance.now();
            content += d.content; ans.textContent = content;
          }
          if (tFirst) stat.textContent = "生成中… " + ((performance.now() - t0) / 1000).toFixed(1) + " s";
        }
      }
    }
    $("chat-status").textContent = "完成 " + hhmmss();
  } catch (e) {
    if (e.name === "AbortError") { finish = "已停止"; $("chat-status").textContent = "已停止"; }
    else { showErr("chat-err", e.message); $("chat-status").textContent = "出错"; }
  } finally {
    chatAbort = null;
    setBusy(false);
  }
  const t1 = performance.now();
  const parts = [];
  if (finish) parts.push("结束：" + finish);
  if (tFirst) parts.push("首 token " + ((tFirst - t0) / 1000).toFixed(2) + " s");
  if (usage) {
    parts.push("输出 " + usage.completion_tokens + " tok");
    if (tFirst && usage.completion_tokens > 1)
      parts.push((usage.completion_tokens / ((t1 - tFirst) / 1000)).toFixed(1) + " tok/s");
    const cached = usage.prompt_tokens_details ? usage.prompt_tokens_details.cached_tokens : 0;
    parts.push("prompt " + usage.prompt_tokens + (cached ? "（命中缓存 " + cached + "）" : ""));
  }
  stat.textContent = parts.join("  ·  ");
  if (content || reasoning) {
    const m = { role: "assistant", content };
    if (reasoning) m.reasoning_content = reasoning;
    convo.push(m);
  } else {
    convo.pop();   // 没有任何输出：撤回这条 user，方便改了参数重发
  }
}

$("c-send").addEventListener("click", sendChat);
$("c-stop").addEventListener("click", () => { if (chatAbort) chatAbort.abort(); });
$("c-clear").addEventListener("click", () => { convo = []; $("c-log").innerHTML = ""; hideErr("chat-err"); });
$("c-input").addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); sendChat(); }
});

// ------------------------------------------------------------ 内存记账 --
function memCell(k, v, sub) {
  return '<div class="cell"><div class="k">' + k + '</div><div class="v mono">' + v +
         '</div><div class="s">' + (sub || "") + "</div></div>";
}

function renderMem(j) {
  const f = n => (n == null ? null : Number(n));
  const committed = f(j.gpu_accessible_committed_bytes);
  $("m-hero").textContent = fmtBytes(committed);
  $("m-hero").title =
    "gpu_accessible_committed_bytes = " + (j.gpu_accessible_committed_bytes ?? "-") + " 字节";
  if (j.accounting) $("m-accounting").textContent = j.accounting;

  const total = f(j.hip_total_bytes), free = f(j.hip_free_bytes);
  if (total != null && free != null) {
    const used = total - free, pct = total ? used / total * 100 : 0;
    $("m-hip-bar").style.width = Math.min(100, Math.max(0, pct)).toFixed(1) + "%";
    $("m-hip-pct").textContent = pct.toFixed(1) + "%";
    $("m-hip-used").textContent = "已用 " + fmtBytes(used);
    $("m-hip-total").textContent = "共 " + fmtBytes(total);
  }

  const peak = (c, p) => "峰值 " + fmtBytes(f(p));
  $("mem-grid").innerHTML =
    memCell("设备分配 (HIP device)", fmtBytes(f(j.device_current_bytes)), peak(0, j.device_peak_bytes)) +
    memCell("注册 mmap（专家权重）", fmtBytes(f(j.registered_mmap_bytes)), "mmap + hipHostRegister") +
    memCell("Pinned host", fmtBytes(f(j.pinned_host_current_bytes)), peak(0, j.pinned_host_peak_bytes)) +
    memCell("启动以来 HIP 增量", fmtBytes(f(j.hip_used_since_engine_start_bytes)), "引擎自身驱动") +
    memCell("进程 RSS", fmtBytes(f(j.process_rss_bytes)), "含可回收 mmap 页缓存") +
    memCell("进程已锁定", fmtBytes(f(j.process_locked_bytes)), "");
  $("raw-mem").textContent = JSON.stringify(j, null, 1);

  hist.committed.push(committed); hist.rss.push(f(j.process_rss_bytes));
  if (hist.committed.length > HIST_CAP) { hist.committed.shift(); hist.rss.shift(); }
  drawSpark();
}

function drawSpark() {
  const cv = $("spark");
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth * dpr, H = cv.clientHeight * dpr;
  cv.width = W; cv.height = H;
  const g = cv.getContext("2d");
  g.clearRect(0, 0, W, H);
  const series = [
    { data: hist.committed, color: "#58a6ff" },
    { data: hist.rss, color: "#bc8cff" },
  ];
  let max = 0;
  for (const s of series) for (const v of s.data) if (v != null && v > max) max = v;
  if (max <= 0) return;
  const pad = 6 * dpr, n = HIST_CAP;
  const x = i => pad + (W - 2 * pad) * (i / (n - 1));
  const y = v => H - pad - (H - 2 * pad) * (v / max);
  g.strokeStyle = "#30363d"; g.lineWidth = 1;
  g.strokeRect(pad, pad, W - 2 * pad, H - 2 * pad);
  for (const s of series) {
    g.strokeStyle = s.color; g.lineWidth = 1.5 * dpr; g.beginPath();
    let started = false;
    s.data.forEach((v, i) => {
      if (v == null) return;
      if (!started) { g.moveTo(x(i), y(v)); started = true; }
      else g.lineTo(x(i), y(v));
    });
    g.stroke();
  }
  g.fillStyle = "#8b949e"; g.font = (10 * dpr) + "px Consolas, monospace";
  g.fillText(fmtBytes(max), pad + 4 * dpr, pad + 12 * dpr);
}

async function tickHealth() {
  if (paused) return;
  try {
    const j = await grab("/health");
    renderHealth(j);
    setChip($("conn"), "ok", "已连接");
    hideErr("health-err");
    $("health-meta").textContent = "更新于 " + hhmmss();
  } catch (e) {
    setChip($("conn"), "err", "无法连接");
    showErr("health-err", e.message);
  }
}

async function tickMem() {
  if (paused) return;
  const data = $("mem-data");
  try {
    const j = await grab("/memory");
    renderMem(j);
    data.classList.remove("stale");
    hideErr("mem-err");
    $("mem-meta").textContent = "更新于 " + hhmmss() + "（每 10 秒）";
  } catch (e) {
    data.classList.add("stale");   // 保留上次数值但变暗，提示非实时
    showErr("mem-err", e.message + "（engine 未运行或未连接）");
  }
}

$("pause").addEventListener("click", () => {
  paused = !paused;
  $("pause").textContent = paused ? "▶ 继续刷新" : "⏸ 暂停刷新";
});
$("target").textContent = location.host;

buildOvrTable();
loadOvr();
tickHealth(); tickMem();
setInterval(tickHealth, HEALTH_MS);
setInterval(tickMem, MEM_MS);
window.addEventListener("resize", drawSpark);
</script>
</body>
</html>
)HTMLDOC";
