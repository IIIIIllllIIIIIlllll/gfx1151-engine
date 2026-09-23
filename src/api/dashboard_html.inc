// dashboard_html.inc — gdec-api 自带的性能监控页（由 GET / 和 GET /dashboard 返回）。
//
// 单文件：CSS/JS 全部内联，不引用任何外部资源（无 CDN，离线/局域网可用）。
// 页面用同源 fetch 轮询两个性能 API：
//   GET /health  每 2 秒   —— API 进程自身的状态，不经过 engine
//   GET /memory  每 10 秒  —— engine 内存记账（占用推理槽位，间隔勿低于 ~1 秒）
// /memory 在 engine 不可达时返回 500；页面把错误显示在面板内并保留上次数值，
// 继续按周期重试，不会白屏。

static const char kDashboardHtml[] = R"HTMLDOC(<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>gdec-api 监控</title>
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
  #target { color: var(--dim); font-size: 13px; }
  .chip {
    padding: 2px 10px; border-radius: 999px; font-size: 12px;
    border: 1px solid var(--border); color: var(--dim);
  }
  .chip.ok   { color: var(--ok);   border-color: var(--ok); }
  .chip.warn { color: var(--warn); border-color: var(--warn); }
  .chip.err  { color: var(--err);  border-color: var(--err); }
  #pause {
    margin-left: auto; background: none; color: var(--dim); cursor: pointer;
    border: 1px solid var(--border); border-radius: 6px; padding: 4px 12px;
    font: inherit;
  }
  #pause:hover { color: var(--fg); border-color: var(--dim); }
  main { max-width: 960px; margin: 0 auto; padding: 16px 20px 40px; }
  .panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px; margin-bottom: 16px;
  }
  .panel h2 { font-size: 13px; margin: 0 0 12px; color: var(--dim); font-weight: 600; }
  .panel .meta { float: right; font-weight: 400; font-size: 12px; }
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
</style>
</head>
<body>
<header>
  <h1>gdec-api 监控</h1>
  <span class="mono" id="target"></span>
  <span class="chip" id="conn">连接中…</span>
  <button id="pause" type="button">⏸ 暂停</button>
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
    </div>
    <div class="errmsg" id="health-err"></div>
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

// fetch 一个端点：返回解析后的 JSON；HTTP 错误抛 Error(消息取自 OpenAI 风格 error 信封)
async function grab(url) {
  const c = new AbortController();
  const t = setTimeout(() => c.abort(), TIMEOUT_MS);
  try {
    const r = await fetch(url, { signal: c.signal });
    const text = await r.text();
    let j = null;
    try { j = JSON.parse(text); } catch (e) { /* 非 JSON 按状态码报错 */ }
    if (!r.ok) {
      const msg = (j && j.error && (j.error.message || j.error.type)) || ("HTTP " + r.status);
      throw new Error(msg);
    }
    return j;
  } catch (e) {
    throw new Error(e.name === "AbortError" ? "请求超时" : e.message);
  } finally { clearTimeout(t); }
}

function renderHealth(j) {
  const s = document.getElementById("h-status");
  if (j.busy) setChip(s, "warn", "busy");
  else if (j.status === "ok") setChip(s, "ok", "ok");
  else setChip(s, "err", String(j.status));
  document.getElementById("h-model").textContent = j.model ?? "-";
  document.getElementById("h-ctx").textContent = j.context ?? "-";
  document.getElementById("h-inflight").textContent = j.in_flight ?? "-";
  const bf = j.busy_for_s;
  document.getElementById("h-busyfor").textContent = (bf == null) ? "-" : (bf.toFixed(1) + "s");
  document.getElementById("h-queued").textContent = j.queued ?? "-";
  document.getElementById("raw-health").textContent = JSON.stringify(j, null, 1);
}

function memCell(k, v, sub) {
  return '<div class="cell"><div class="k">' + k + '</div><div class="v mono">' + v +
         '</div><div class="s">' + (sub || "") + "</div></div>";
}

function renderMem(j) {
  const f = n => (n == null ? null : Number(n));
  const committed = f(j.gpu_accessible_committed_bytes);
  document.getElementById("m-hero").textContent = fmtBytes(committed);
  document.getElementById("m-hero").title =
    "gpu_accessible_committed_bytes = " + (j.gpu_accessible_committed_bytes ?? "-") + " 字节";
  if (j.accounting) document.getElementById("m-accounting").textContent = j.accounting;

  const total = f(j.hip_total_bytes), free = f(j.hip_free_bytes);
  if (total != null && free != null) {
    const used = total - free, pct = total ? used / total * 100 : 0;
    document.getElementById("m-hip-bar").style.width = Math.min(100, Math.max(0, pct)).toFixed(1) + "%";
    document.getElementById("m-hip-pct").textContent = pct.toFixed(1) + "%";
    document.getElementById("m-hip-used").textContent = "已用 " + fmtBytes(used);
    document.getElementById("m-hip-total").textContent = "共 " + fmtBytes(total);
  }

  const peak = (c, p) => "峰值 " + fmtBytes(f(p));
  document.getElementById("mem-grid").innerHTML =
    memCell("设备分配 (HIP device)", fmtBytes(f(j.device_current_bytes)), peak(0, j.device_peak_bytes)) +
    memCell("注册 mmap（专家权重）", fmtBytes(f(j.registered_mmap_bytes)), "mmap + hipHostRegister") +
    memCell("Pinned host", fmtBytes(f(j.pinned_host_current_bytes)), peak(0, j.pinned_host_peak_bytes)) +
    memCell("启动以来 HIP 增量", fmtBytes(f(j.hip_used_since_engine_start_bytes)), "引擎自身驱动") +
    memCell("进程 RSS", fmtBytes(f(j.process_rss_bytes)), "含可回收 mmap 页缓存") +
    memCell("进程已锁定", fmtBytes(f(j.process_locked_bytes)), "");
  document.getElementById("raw-mem").textContent = JSON.stringify(j, null, 1);

  hist.committed.push(committed); hist.rss.push(f(j.process_rss_bytes));
  if (hist.committed.length > HIST_CAP) { hist.committed.shift(); hist.rss.shift(); }
  drawSpark();
}

function drawSpark() {
  const cv = document.getElementById("spark");
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

function showErr(panelId, err) {
  const el = document.getElementById(panelId);
  el.textContent = hhmmss() + "  " + err;
  el.classList.add("show");
}
function hideErr(panelId) { document.getElementById(panelId).classList.remove("show"); }

async function tickHealth() {
  if (paused) return;
  try {
    const j = await grab("/health");
    renderHealth(j);
    setChip(document.getElementById("conn"), "ok", "已连接");
    hideErr("health-err");
    document.getElementById("health-meta").textContent = "更新于 " + hhmmss();
  } catch (e) {
    setChip(document.getElementById("conn"), "err", "无法连接");
    showErr("health-err", e.message);
  }
}

async function tickMem() {
  if (paused) return;
  const data = document.getElementById("mem-data");
  try {
    const j = await grab("/memory");
    renderMem(j);
    data.classList.remove("stale");
    hideErr("mem-err");
    document.getElementById("mem-meta").textContent = "更新于 " + hhmmss() + "（每 10 秒）";
  } catch (e) {
    data.classList.add("stale");   // 保留上次数值但变暗，提示非实时
    showErr("mem-err", e.message + "（engine 未运行或未连接）");
  }
}

document.getElementById("pause").addEventListener("click", () => {
  paused = !paused;
  document.getElementById("pause").textContent = paused ? "▶ 继续" : "⏸ 暂停";
});
document.getElementById("target").textContent = location.host;

tickHealth(); tickMem();
setInterval(tickHealth, HEALTH_MS);
setInterval(tickMem, MEM_MS);
window.addEventListener("resize", drawSpark);
</script>
</body>
</html>
)HTMLDOC";
