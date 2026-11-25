"""Self-contained zero-dependency Web Control Console (/_console) for basalt-colab."""
from __future__ import annotations

CONSOLE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>basalt-colab · Control Console</title>
<style>
  :root {
    --bg: #0b0e14; --card: #131822; --border: #232b3a; --text: #e6edf3; --muted: #8b949e;
    --accent: #3fb950; --cyan: #58a6ff; --amber: #d29922; --red: #f85149; --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }
  .wrap { max-width: 1120px; margin: 0 auto; padding: 20px; }
  header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }
  .brand { display: flex; align-items: center; gap: 10px; }
  .brand h1 { margin: 0; font-size: 18px; letter-spacing: -0.02em; }
  .pill { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; background: #1d2433; border: 1px solid var(--border); }
  .pill.ready { color: var(--accent); border-color: #264d32; background: #112418; }
  .pill.starting, .pill.stopping { color: var(--cyan); border-color: #1f3c63; background: #101d30; }
  .pill.off { color: var(--muted); }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
  .actions { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
  button, a.btn { cursor: pointer; font: inherit; font-weight: 600; padding: 7px 14px; border-radius: 8px; border: 1px solid var(--border); background: #1a2130; color: var(--text); text-decoration: none; display: inline-flex; align-items: center; gap: 6px; }
  button:hover, a.btn:hover { background: #232d42; }
  button.primary, a.btn.primary { background: #1e6b34; border-color: #2ea043; color: #fff; }
  button.primary:hover, a.btn.primary:hover { background: #238636; }
  button.danger { background: #3b191d; border-color: #6e282e; color: #ffa198; }
  button.danger:hover { background: #541f25; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 16px; }
  .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 14px; margin-top: 14px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
  .card h2 { margin: 0 0 10px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); display: flex; justify-content: space-between; align-items: center; }
  .kpi { font-size: 24px; font-weight: 700; font-family: var(--mono); }
  .sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .steps { list-style: none; margin: 0; padding: 0; font-family: var(--mono); font-size: 12px; }
  .steps li { padding: 4px 0; display: flex; gap: 10px; border-bottom: 1px solid #1b2230; }
  .steps li:last-child { border-bottom: 0; }
  .ok { color: var(--accent); } .dim { color: var(--muted); } .err { color: var(--red); } .warn { color: var(--amber); }
  textarea, input[type=text], input[type=password] { width: 100%; background: #0c1018; color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-family: var(--mono); font-size: 12px; }
  textarea { min-height: 240px; resize: vertical; line-height: 1.45; }
  pre { margin: 0; background: #0c1018; border: 1px solid var(--border); border-radius: 8px; padding: 10px; max-height: 290px; overflow: auto; font: 12px/1.4 var(--mono); white-space: pre-wrap; word-break: break-word; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; font-family: var(--mono); margin-top: 10px; }
  th, td { text-align: left; padding: 5px 6px; border-bottom: 1px solid #1b2230; }
  th { color: var(--muted); font-weight: 600; }
  .bar-chart { width: 100%; height: 110px; display: block; margin-top: 8px; }
  .auth-box { max-width: 440px; margin: 80px auto; }
  .hidden { display: none !important; }
  .row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  label.chk { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--muted); cursor: pointer; }
  .toast { margin-top: 8px; font-size: 12px; min-height: 18px; }
</style>
</head>
<body>
<div id="authView" class="wrap auth-box hidden">
  <div class="card">
    <h2>basalt-colab · Authentication</h2>
    <p class="sub">Enter your local API key (from <code>basalt-colab status</code> or <code>basalt-colab console</code>). Stored only in browser <code>localStorage</code>.</p>
    <input id="keyInput" type="password" placeholder="sk-ds41-..." autocomplete="off">
    <div class="row" style="margin-top:12px">
      <button class="primary" onclick="saveKey()">Connect</button>
    </div>
    <div id="authErr" class="toast err"></div>
  </div>
</div>

<div id="appView" class="wrap hidden">
  <header>
    <div class="brand">
      <h1>basalt-colab</h1>
      <span id="statePill" class="pill off"><span class="dot"></span><span id="stateText">OFF</span></span>
      <span id="modelMeta" class="sub"></span>
    </div>
    <div class="actions">
      <label class="chk"><input id="waitGpuChk" type="checkbox"> --wait-gpu</label>
      <button id="btnUp" class="primary" onclick="startGpu()">Start G4</button>
      <button id="btnDown" class="danger" onclick="stopGpu()">Stop G4</button>
      <a id="btnWebUi" class="btn hidden" href="/" target="_blank" rel="noopener">Open Basalt Web UI ↗</a>
      <button onclick="logout()" title="Clear saved API key">Sign out</button>
    </div>
  </header>

  <div id="errBanner" class="card hidden" style="margin-top:14px;border-color:#6e282e;background:#241216;">
    <span class="err" id="errText"></span>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Decode Speed</h2>
      <div class="kpi" id="kpiSpeed">—</div>
      <div class="sub" id="kpiHit">VRAM hit — · PCIe —</div>
    </div>
    <div class="card">
      <h2>GPU Activity</h2>
      <div class="kpi" id="kpiActivity">Idle</div>
      <div class="sub" id="kpiIdle">Auto-stop —</div>
    </div>
    <div class="card">
      <h2>Session Cost</h2>
      <div class="kpi" id="kpiSessionCu">0.00 CU</div>
      <div class="sub" id="kpiRan">Not running</div>
    </div>
    <div class="card">
      <h2>Billing Ledger</h2>
      <div class="kpi" id="kpiMonthCu">0.00 CU</div>
      <div class="sub" id="kpiTotals">Today 0.00 · All-time 0.00 CU</div>
    </div>
  </div>

  <div id="bootCard" class="card hidden" style="margin-top:14px;">
    <h2>Boot Timeline <span id="bootStage" class="dim"></span></h2>
    <ul id="bootSteps" class="steps"></ul>
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>CU Cost Curve (Last 14 Days) <span id="ledgerCount" class="dim"></span></h2>
      <svg id="cuChart" class="bar-chart" viewBox="0 0 420 110" preserveAspectRatio="none"></svg>
      <table>
        <thead><tr><th>Started</th><th>Duration</th><th>Est. CU</th><th>Reason</th><th>Req</th></tr></thead>
        <tbody id="recentBody"></tbody>
      </table>
    </div>

    <div class="card">
      <h2>
        <span>config.toml (Live Hot-Reload)</span>
        <span class="row">
          <button onclick="loadConfig()">Refresh</button>
          <button class="primary" onclick="saveConfig()">Save &amp; Apply</button>
        </span>
      </h2>
      <textarea id="cfgText" spellcheck="false"></textarea>
      <div id="cfgToast" class="toast"></div>
    </div>
  </div>

  <div class="card" style="margin-top:14px;">
    <h2>
      <span>Logs</span>
      <span class="row">
        <label class="chk"><input id="vmLogChk" type="checkbox" onchange="loadLogs()"> Include VM logs</label>
        <button onclick="loadLogs()">Refresh</button>
      </span>
    </h2>
    <pre id="logOut">Loading logs...</pre>
  </div>
</div>

<script>
const STORAGE_KEY = "basalt_colab_api_key";
let apiKey = "";
let pollTimer = null;

function initAuth() {
  const hash = location.hash || "";
  const m = hash.match(/key=([^&]+)/);
  if (m) {
    localStorage.setItem(STORAGE_KEY, decodeURIComponent(m[1]));
    history.replaceState(null, "", location.pathname + location.search);
  }
  apiKey = localStorage.getItem(STORAGE_KEY) || "";
  if (!apiKey) { showAuth(); return; }
  verifyAndBoot();
}

function showAuth(msg) {
  document.getElementById("authView").classList.remove("hidden");
  document.getElementById("appView").classList.add("hidden");
  document.getElementById("authErr").textContent = msg || "";
}

async function saveKey() {
  apiKey = document.getElementById("keyInput").value.trim();
  if (!apiKey) return;
  localStorage.setItem(STORAGE_KEY, apiKey);
  await verifyAndBoot();
}

function logout() {
  localStorage.removeItem(STORAGE_KEY);
  apiKey = "";
  if (pollTimer) clearInterval(pollTimer);
  showAuth();
}

async function api(path, opts = {}) {
  const headers = Object.assign({ "Authorization": "Bearer " + apiKey }, opts.headers || {});
  if (opts.body && !headers["Content-Type"]) headers["Content-Type"] = "application/json";
  const r = await fetch(path, Object.assign({}, opts, { headers }));
  if (r.status === 401) { logout(); showAuth("Invalid API key."); throw new Error("unauthorized"); }
  return r;
}

async function verifyAndBoot() {
  try {
    const r = await api("/_control/status");
    const st = await r.json();
    document.getElementById("authView").classList.add("hidden");
    document.getElementById("appView").classList.remove("hidden");
    renderStatus(st);
    await Promise.all([loadConfig(), loadUsage(), loadLogs()]);
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      try {
        const sr = await api("/_control/status");
        renderStatus(await sr.json());
      } catch (_) {}
    }, 1500);
  } catch (e) {
    if (e.message !== "unauthorized") showAuth("Cannot reach proxy: " + e.message);
  }
}

function mmss(s) {
  s = Math.max(0, Math.floor(s || 0));
  return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
}

function renderStatus(st) {
  const base = st.state.startsWith("ready") ? "ready" : st.state;
  const pill = document.getElementById("statePill");
  pill.className = "pill " + base;
  document.getElementById("stateText").textContent = st.state.toUpperCase();
  document.getElementById("modelMeta").textContent = st.model + " · " + (st.engine_summary || "") + " · effort " + st.effort;
  document.getElementById("btnUp").disabled = (base === "ready" || base === "starting" || base === "stopping");
  document.getElementById("btnDown").disabled = (base === "off" || base === "stopping");
  document.getElementById("btnWebUi").classList.toggle("hidden", base !== "ready");

  const errB = document.getElementById("errBanner");
  if (st.last_error && base === "off") {
    errB.classList.remove("hidden");
    document.getElementById("errText").textContent = "Last error: " + st.last_error;
  } else {
    errB.classList.add("hidden");
  }

  const b = st.basalt || {};
  const spd = (b.busy && b.live_tok_s != null) ? b.live_tok_s : b.last_tok_s;
  document.getElementById("kpiSpeed").textContent = spd != null ? spd.toFixed(1) + " tok/s" : "—";
  const hit = b.hit_rate != null ? (b.hit_rate * 100).toFixed(1) + "%" : "—";
  const pcie = b.pcie_share != null ? (b.pcie_share * 100).toFixed(1) + "%" : "—";
  document.getElementById("kpiHit").textContent = "VRAM hit " + hit + " · PCIe " + pcie;

  if (base === "ready") {
    const left = Math.min(st.idle_stop_min * 60 - st.idle_s, st.max_hours * 3600 - (st.ran || 0));
    document.getElementById("kpiActivity").textContent = st.active ? (st.active + " active") : ("Idle " + mmss(st.idle_s));
    document.getElementById("kpiIdle").textContent = "Stops in " + mmss(left) + " · " + st.served + " served";
  } else if (base === "starting") {
    document.getElementById("kpiActivity").textContent = "Starting";
    document.getElementById("kpiIdle").textContent = st.stage || "booting...";
  } else {
    document.getElementById("kpiActivity").textContent = "Off";
    document.getElementById("kpiIdle").textContent = "Starts on request or Start G4";
  }

  document.getElementById("kpiSessionCu").textContent = st.cu != null ? st.cu.toFixed(2) + " CU" : "0.00 CU";
  document.getElementById("kpiRan").textContent = st.ran != null ? ("Running " + mmss(st.ran) + " · ref " + st.basalt_ref) : "0 CU billed while off";

  const bootCard = document.getElementById("bootCard");
  if (base === "starting" || (st.steps && st.steps.length && base !== "off")) {
    bootCard.classList.remove("hidden");
    document.getElementById("bootStage").textContent = base === "starting" ? ("· " + (st.stage || "")) : "· Ready";
    document.getElementById("bootSteps").innerHTML = (st.steps || []).map(x =>
      `<li><span class="dim">${mmss(x.elapsed_s)}</span><span class="ok">✓</span><span>${esc(x.label)}</span></li>`
    ).join("");
  } else {
    bootCard.classList.add("hidden");
  }
}

async function startGpu() {
  const wait_gpu = document.getElementById("waitGpuChk").checked;
  document.getElementById("btnUp").disabled = true;
  try {
    const r = await api("/_control/up", { method: "POST", body: JSON.stringify({ wait_gpu }) });
    const reader = r.body.getReader();
    while (true) {
      const { done } = await reader.read();
      if (done) break;
    }
  } finally {
    await Promise.all([loadUsage(), loadLogs()]);
  }
}

async function stopGpu() {
  document.getElementById("btnDown").disabled = true;
  try {
    const r = await api("/_control/down", { method: "POST", body: JSON.stringify({ exit: false }) });
    const reader = r.body.getReader();
    while (true) {
      const { done } = await reader.read();
      if (done) break;
    }
  } finally {
    await Promise.all([loadUsage(), loadLogs()]);
  }
}

async function loadConfig() {
  const r = await api("/_control/config");
  const d = await r.json();
  document.getElementById("cfgText").value = d.toml || "";
  document.getElementById("cfgToast").textContent = d.path || "";
  document.getElementById("cfgToast").className = "toast dim";
}

async function saveConfig() {
  const toast = document.getElementById("cfgToast");
  try {
    const r = await api("/_control/config", { method: "POST", body: JSON.stringify({ toml: document.getElementById("cfgText").value }) });
    const d = await r.json();
    if (!r.ok) { toast.textContent = "Error: " + (d.error?.message || r.status); toast.className = "toast err"; return; }
    const msg = d.restart_required
      ? "Saved. Hot changes applied live; cold changes (" + d.cold_changed.join(", ") + ") apply on next GPU start."
      : "Saved and applied live.";
    toast.textContent = msg;
    toast.className = d.restart_required ? "toast warn" : "toast ok";
  } catch (e) {
    toast.textContent = "Save failed: " + e.message; toast.className = "toast err";
  }
}

async function loadUsage() {
  const r = await api("/_control/usage?days=14");
  const u = await r.json();
  document.getElementById("kpiMonthCu").textContent = u.month_cu.toFixed(2) + " CU";
  document.getElementById("kpiTotals").textContent = `Today ${u.today_cu.toFixed(2)} · All-time ${u.total_cu.toFixed(2)} CU (${u.total_hours.toFixed(1)}h)`;
  document.getElementById("ledgerCount").textContent = `${u.sessions_count} sessions`;
  const days = u.daily || [];
  const maxCu = Math.max(0.5, ...days.map(d => d.cu));
  const w = 420, h = 110, pad = 14, bw = Math.max(8, Math.floor((w - pad * 2) / Math.max(1, days.length)) - 4);
  const bars = days.map((d, i) => {
    const bh = Math.max(2, Math.round((d.cu / maxCu) * (h - 30)));
    const x = pad + i * ((w - pad * 2) / Math.max(1, days.length));
    const y = h - 18 - bh;
    const lbl = d.date.slice(5);
    return `<g><title>${d.date}: ${d.cu.toFixed(2)} CU (${d.sessions} sessions)</title>`
         + `<rect x="${x.toFixed(1)}" y="${y}" width="${bw}" height="${bh}" rx="3" fill="${d.cu > 0 ? '#3fb950' : '#1f2737'}"/>`
         + (i % 3 === 0 ? `<text x="${(x + bw / 2).toFixed(1)}" y="${h - 4}" text-anchor="middle" fill="#8b949e" font-size="9">${lbl}</text>` : "")
         + `</g>`;
  }).join("");
  document.getElementById("cuChart").innerHTML = bars;
  const rows = (u.recent || []).slice(0, 8).map(e =>
    `<tr><td>${esc((e.started_at || "").replace("T", " ").slice(5, 16))}</td><td>${mmss(e.ran_s)}</td><td>${Number(e.cu_est || 0).toFixed(2)}</td><td>${esc(e.reason || "")}</td><td>${e.served || 0}</td></tr>`
  ).join("");
  document.getElementById("recentBody").innerHTML = rows || `<tr><td colspan="5" class="dim">No sessions recorded yet</td></tr>`;
}

async function loadLogs() {
  const vm = document.getElementById("vmLogChk").checked ? "1" : "0";
  const r = await api("/_control/logs?vm=" + vm);
  const d = await r.json();
  document.getElementById("logOut").textContent = (d.local || "(no local log)") + (d.vm ? ("\n\n--- VM ---\n" + d.vm) : "");
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

initAuth();
</script>
</body>
</html>
"""


def render_console_html() -> bytes:
    return CONSOLE_HTML.encode("utf-8")
