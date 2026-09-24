// Paper trading & backtest tab. Talks to gui/trading.py; long work runs as a
// server-side job this page polls, so closing the tab does not stop it.
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const esc = (v) => (typeof escapeHtml === "function" ? escapeHtml(String(v ?? "")) : String(v ?? ""));
  const toast = (msg, kind) => (typeof showToast === "function" ? showToast(msg, kind) : alert(msg));

  let jobSince = 0;
  let pollTimer = null;
  let accountCurrency = "USD";

  // ---- formatting ---------------------------------------------------------
  function money(v, cur) {
    if (v == null || isNaN(v)) return "—";
    try {
      return new Intl.NumberFormat(undefined, { style: "currency", currency: cur || accountCurrency }).format(v);
    } catch (e) { return Number(v).toFixed(2); }
  }
  function pct(v, signed = true) {
    if (v == null || isNaN(v)) return "—";
    const s = (v * 100).toFixed(2) + "%";
    return signed && v > 0 ? "+" + s : s;
  }
  const tone = (v) => (v > 0 ? "pos" : v < 0 ? "neg" : "");
  const isoDay = (d) => d.toISOString().slice(0, 10);

  async function api(url, body) {
    const opts = body === undefined ? {} : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    };
    const res = await fetch(url, opts);
    let data = {};
    try { data = await res.json(); } catch (e) { /* non-JSON error page */ }
    if (!res.ok || data.ok === false) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  }

  // The provider and models picked on the Configuration tab, as the Analyze
  // form sends them (app.js keeps these as script-level globals).
  function collectLlm() {
    const out = {};
    try {
      const provider = typeof currentProvider !== "undefined" ? currentProvider : null;
      if (!provider) return out;
      out.provider = provider;
      let quick = $("quick-model-select")?.value || "";
      let deep = $("deep-model-select")?.value || "";
      const custom = ["openrouter", "azure", "ollama"].includes(provider);
      if (custom) {
        quick = $("custom-quick-model")?.value || quick;
        deep = $("custom-deep-model")?.value || deep;
      }
      if (quick && quick !== "__custom__") out.quick_model = quick;
      if (deep && deep !== "__custom__") out.deep_model = deep;
      const providers = typeof availableProviders !== "undefined" ? availableProviders : [];
      const known = providers.find((p) => p.key === provider);
      out.backend_url = (custom && $("custom-backend-url")?.value) || (known ? known.url : null);
      out.google_thinking_level = $("google-thinking")?.value || null;
      out.openai_reasoning_effort = $("openai-effort")?.value || null;
      out.anthropic_effort = $("anthropic-effort")?.value || null;
    } catch (e) { /* fall back to the .env defaults on the server */ }
    return out;
  }

  // ---- account ------------------------------------------------------------
  async function loadAccounts(select) {
    const sel = $("paper-account-select");
    const { accounts } = await api("/api/paper/accounts");
    const current = select || sel.value || accounts[0] || "ledger";
    const names = accounts.includes(current) ? accounts : [...accounts, current];
    sel.innerHTML = names.map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
    sel.value = current;
  }

  window.tradingLoadAccount = async function (refresh = false) {
    const account = $("paper-account-select").value || "ledger";
    const body = $("paper-account-body");
    const warn = $("paper-warning");
    if (refresh) body.innerHTML = `<p class="form-hint">Fetching prices…</p>`;
    try {
      const data = await api(`/api/paper/account?account=${encodeURIComponent(account)}${refresh ? "&refresh=1" : ""}`);
      warn.style.display = data.warning ? "" : "none";
      warn.textContent = data.warning || "";
      if (!data.exists) {
        body.innerHTML = `<p class="form-hint">No paper account called “${esc(account)}” yet. Open one below.</p>`;
        $("paper-new-account").open = true;
        return;
      }
      renderAccount(data.summary);
    } catch (e) {
      body.innerHTML = `<p class="form-hint trading-warn">${esc(e.message)}</p>`;
    }
  };

  function renderAccount(s) {
    accountCurrency = s.currency;
    const b = s.benchmark;
    const stats = [
      ["Equity", money(s.equity)],
      ["Cash", money(s.cash)],
      ["Return", `<span class="${tone(s.return)}">${pct(s.return)}</span>`],
      b ? [`vs ${esc(b.ticker)}`, `<span class="${tone(b.excess)}">${pct(b.excess)}</span>`] : null,
    ].filter(Boolean).map(([k, v]) => `<div class="trading-stat"><span class="k">${k}</span><span class="v">${v}</span></div>`).join("");

    const positions = s.positions.length ? `
      <table class="trading-table"><thead><tr><th>Holding</th><th class="num">Units</th><th class="num">Avg cost</th>
      <th class="num">Last</th><th class="num">Value</th><th class="num">Gain</th><th class="num">Weight</th></tr></thead><tbody>
      ${s.positions.map((p) => `<tr><td>${esc(p.ticker)}</td><td class="num">${Number(p.quantity).toLocaleString(undefined, { maximumFractionDigits: 4 })}</td>
        <td class="num">${money(p.average_price)}</td><td class="num">${money(p.last_price)}</td><td class="num">${money(p.value)}</td>
        <td class="num ${tone(p.gain)}">${pct(p.gain)}</td><td class="num">${pct(p.weight, false)}</td></tr>`).join("")}
      </tbody></table>` : `<p class="form-hint">No holdings yet.</p>`;

    const pending = s.pending.length ? `<div class="trading-h">Waiting for the next open</div>
      <table class="trading-table"><tbody>${s.pending.map((o) => `<tr><td>${esc(o.ticker)}</td>
      <td class="trading-rating">${esc(o.rating)}</td><td>from ${esc(o.analysis_date)}</td></tr>`).join("")}</tbody></table>` : "";

    const trades = s.trades.length ? `<div class="trading-h">Recent trades</div>
      <table class="trading-table"><thead><tr><th>Date</th><th>Side</th><th>Ticker</th><th class="num">Units</th>
      <th class="num">Price</th><th>Rating</th><th class="num">Realized</th></tr></thead><tbody>
      ${s.trades.map((t) => `<tr><td>${esc(t.date)}</td><td>${esc(t.side)}</td><td>${esc(t.ticker)}</td>
        <td class="num">${Number(t.quantity).toLocaleString(undefined, { maximumFractionDigits: 4 })}</td>
        <td class="num">${money(t.price)}</td><td>${esc(t.rating)}</td>
        <td class="num ${tone(t.realized)}">${t.realized != null ? money(t.realized) : ""}</td></tr>`).join("")}
      </tbody></table>` : "";

    const decisions = s.decisions.length ? `<div class="trading-h">Recent decisions</div>
      <table class="trading-table"><tbody>${s.decisions.slice(0, 10).map((d) => `<tr><td>${esc(d.ticker)}</td>
      <td class="trading-rating">${esc(d.rating)}</td><td>as of ${esc(d.analysis_date)}</td></tr>`).join("")}</tbody></table>` : "";

    $("paper-account-body").innerHTML = `
      <div class="trading-stats">${stats}</div>
      ${positions}${pending}${trades}${decisions}
      <p class="form-hint">Started ${esc(s.created.slice(0, 10))} with ${money(s.starting_cash)} · largest position
      ${pct(s.rules.max_position, false)} · prices as of ${esc((s.marked_at || s.created).replace("T", " ").slice(0, 16))} UTC</p>`;
  }

  window.tradingCreateAccount = async function () {
    const name = $("paper-new-name").value.trim() || "ledger";
    const body = {
      account: name, cash: parseFloat($("paper-new-cash").value),
      currency: $("paper-new-currency").value, max_position: parseFloat($("paper-new-max").value),
      whole_shares: $("paper-new-whole").checked,
    };
    try {
      await api("/api/paper/init", body);
    } catch (e) {
      if (!String(e.message).includes("already exists") ||
          !confirm(`“${name}” already exists. Replace it and lose its history?`)) {
        toast(e.message, "error");
        return;
      }
      try { await api("/api/paper/init", { ...body, force: true }); }
      catch (e2) { toast(e2.message, "error"); return; }
    }
    toast(`Opened paper account “${name}”`, "success");
    $("paper-new-account").open = false;
    await loadAccounts(name);
    await tradingLoadAccount();
  };

  // ---- screener -----------------------------------------------------------
  window.tradingScreen = async function () {
    const universe = $("paper-screen").value || "us";
    const out = $("paper-screen-results");
    out.innerHTML = `<p class="form-hint">Ranking ${esc(universe)}…</p>`;
    try {
      const data = await api(`/api/paper/screen?universe=${encodeURIComponent(universe)}&top=10`);
      out.innerHTML = data.picks.length ? `
        <div class="trading-h">Top of ${esc(universe)} right now (price only, no AI)</div>
        <table class="trading-table"><thead><tr><th>#</th><th>Ticker</th><th class="num">3-month change</th>
        <th class="num">Above 50-day avg</th></tr></thead><tbody>
        ${data.picks.map((p, i) => `<tr><td>${i + 1}</td><td>${esc(p.ticker)}</td><td class="num pos">${pct(p.momentum)}</td>
          <td class="num">${pct(p.close / p.sma - 1)}</td></tr>`).join("")}</tbody></table>` :
        `<p class="form-hint">Nothing in ${esc(universe)} is both rising and above its 50-day average.</p>`;
    } catch (e) {
      out.innerHTML = `<p class="form-hint trading-warn">${esc(e.message)}</p>`;
    }
  };

  // ---- jobs ---------------------------------------------------------------
  window.tradingRunSession = async function () {
    const llm = collectLlm();
    llm.research_depth = parseInt($("paper-depth").value, 10);
    try {
      await api("/api/paper/run", {
        account: $("paper-account-select").value || "ledger",
        tickers: $("paper-tickers").value,
        screen: $("paper-screen").value || null,
        top: parseInt($("paper-top").value, 10),
        llm,
      });
      startPolling(true);
    } catch (e) { toast(e.message, "error"); }
  };

  window.tradingRunBacktest = async function () {
    const analysts = [...document.querySelectorAll("#bt-analysts input:checked")].map((i) => i.value);
    const llm = collectLlm();
    llm.research_depth = 1;
    try {
      await api("/api/backtest", {
        tickers: $("bt-tickers").value, start: $("bt-start").value, end: $("bt-end").value,
        every: parseInt($("bt-every").value, 10), analysts, llm,
      });
      startPolling(true);
    } catch (e) { toast(e.message, "error"); }
  };

  window.tradingStopJob = async function () {
    try { await api("/api/jobs/stop", {}); } catch (e) { toast(e.message, "error"); }
  };

  function startPolling(fresh) {
    if (fresh) { jobSince = 0; $("job-log").textContent = ""; }
    clearTimeout(pollTimer);
    poll();
  }

  async function poll() {
    let job = null;
    try { ({ job } = await api(`/api/jobs/current?since=${jobSince}`)); } catch (e) { /* retry below */ }
    const running = job && job.status === "running";
    $("job-stop-btn").style.display = running ? "" : "none";
    $("paper-run-btn").disabled = !!running;
    $("bt-run-btn").disabled = !!running;
    if (job) {
      const started = new Date(job.started_at * 1000).toLocaleTimeString();
      $("job-title").textContent = `${job.title}: ${job.status} (started ${started})`;
      if (job.log.length) {
        const pre = $("job-log");
        pre.textContent += (pre.textContent ? "\n" : "") + job.log.join("\n");
        pre.scrollTop = pre.scrollHeight;
      }
      jobSince = job.log_size;
    }
    if (running) {
      pollTimer = setTimeout(poll, 2000);
    } else if (job && job.ended_at && !poll._settled) {
      poll._settled = true;
      if (job.kind === "paper") tradingLoadAccount();
      if (job.kind === "backtest") loadBacktests();
    }
    if (running) poll._settled = false;
  }

  // ---- backtests ----------------------------------------------------------
  async function loadBacktests() {
    const out = $("bt-runs");
    try {
      const { runs } = await api("/api/backtest/runs");
      if (!runs.length) { out.innerHTML = ""; return; }
      out.innerHTML = `<div class="trading-h">Past backtests</div>` + runs.map((r) => r.error ?
        `<p class="form-hint">${esc(r.run_id)}: ${esc(r.error)}</p>` : `
        <details><summary>${esc(r.run_id)}: ${r.resolved} scored, ${r.pending} waiting</summary>
        <table class="trading-table"><thead><tr><th>Rating</th><th class="num">Calls</th><th class="num">Right direction</th>
        <th class="num">Avg vs index</th></tr></thead><tbody>
        ${Object.entries(r.by_rating).map(([rating, sc]) => `<tr><td>${esc(rating)}</td><td class="num">${sc.count}</td>
          <td class="num">${sc.hit_rate == null ? "n/a" : pct(sc.hit_rate, false)}</td>
          <td class="num ${tone(sc.mean_alpha)}">${pct(sc.mean_alpha)}</td></tr>`).join("")}
        </tbody></table><p class="form-hint">Measured over ${esc(r.holding)}. A handful of calls says little; dozens start to mean something.</p></details>`).join("");
    } catch (e) { out.innerHTML = `<p class="form-hint trading-warn">${esc(e.message)}</p>`; }
  }

  function updateEstimate() {
    const tickers = $("bt-tickers").value.split(",").filter((t) => t.trim()).length;
    const start = new Date($("bt-start").value), end = new Date($("bt-end").value);
    const every = parseInt($("bt-every").value, 10);
    if (!tickers || isNaN(start) || isNaN(end) || end < start) { $("bt-estimate").textContent = ""; return; }
    const cells = tickers * (Math.floor((end - start) / 86400000 / every) + 1);
    $("bt-estimate").textContent = `${cells} analyses. On a laptop model allow ~30 min each; a cloud model is faster but costs per run.`;
  }

  // ---- init ---------------------------------------------------------------
  async function init() {
    const today = new Date();
    // Scoring needs five trading days after a date, so end the default grid two weeks back.
    $("bt-end").value = isoDay(new Date(today - 14 * 86400000));
    $("bt-start").value = isoDay(new Date(today - 84 * 86400000));
    $("bt-end").max = $("bt-start").max = isoDay(today);
    ["bt-tickers", "bt-start", "bt-end", "bt-every"].forEach((id) => $(id).addEventListener("input", updateEstimate));
    updateEstimate();
    try { await loadAccounts(); } catch (e) { /* shown by loadAccount */ }
    tradingLoadAccount();
    loadBacktests();
    startPolling(true);
    document.getElementById("nav-trading")?.addEventListener("click", () => {
      tradingLoadAccount();
      loadBacktests();
      clearTimeout(pollTimer);
      poll();
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
