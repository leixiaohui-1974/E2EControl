/**
 * E2EControl – Intelligent Water Network Dashboard
 * Commercial-grade SPA frontend
 */
'use strict';

/* ------------------------------------------------------------------ */
/* State                                                               */
/* ------------------------------------------------------------------ */
const API = '';           // relative – served by Flask
let simId = null;
let pollTimer = null;
let mainChart, topoChart, anomalyChart;
const POOL_COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6'];

/* ------------------------------------------------------------------ */
/* Utilities                                                           */
/* ------------------------------------------------------------------ */
function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function toast(msg, type = 'info', ms = 3500) {
    const c = $('#toast-container');
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = msg;
    c.appendChild(el);
    setTimeout(() => { el.classList.add('out'); setTimeout(() => el.remove(), 300); }, ms);
}

async function api(path, opts = {}) {
    const res = await fetch(`${API}${path}`, {
        headers: { 'Content-Type': 'application/json', ...opts.headers },
        ...opts,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

/* ------------------------------------------------------------------ */
/* Clock                                                               */
/* ------------------------------------------------------------------ */
function tickClock() {
    const el = $('#clock');
    if (el) el.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

/* ------------------------------------------------------------------ */
/* Theme                                                               */
/* ------------------------------------------------------------------ */
function initTheme() {
    const saved = localStorage.getItem('e2e-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    $('#theme-toggle').addEventListener('click', () => {
        const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('e2e-theme', next);
        rebuildCharts();
    });
}

/* ------------------------------------------------------------------ */
/* Navigation                                                          */
/* ------------------------------------------------------------------ */
function initNav() {
    const items = $$('.nav-item');
    const views = $$('.view');
    const title = $('#page-title');

    items.forEach(item => {
        item.addEventListener('click', e => {
            e.preventDefault();
            const target = item.dataset.view;
            items.forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            views.forEach(v => { v.classList.remove('active'); if (v.id === target) v.classList.add('active'); });
            title.textContent = item.querySelector('span').textContent;
            if (target === 'view-history') loadHistory();
            if (target === 'view-config') loadConfig();
        });
    });

    // Sidebar toggle (mobile)
    $('#sidebar-toggle').addEventListener('click', () => {
        $('#sidebar').classList.toggle('open');
    });
}

/* ------------------------------------------------------------------ */
/* Charts                                                              */
/* ------------------------------------------------------------------ */
function chartDefaults() {
    const style = getComputedStyle(document.documentElement);
    const grid = style.getPropertyValue('--chart-grid').trim();
    const txt2 = style.getPropertyValue('--text-2').trim();
    return {
        gridColor: grid || '#2a2d3e',
        tickColor: txt2 || '#8b8fa3',
    };
}

function initCharts() {
    const d = chartDefaults();

    // Main – Water Levels
    mainChart = new Chart($('#mainChart'), {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: { legend: { position: 'top', labels: { color: d.tickColor, usePointStyle: true, pointStyle: 'circle', padding: 16, font: { size: 11 } } } },
            scales: {
                x: { grid: { color: d.gridColor }, ticks: { color: d.tickColor, maxTicksLimit: 12, font: { size: 10 } } },
                y: { grid: { color: d.gridColor }, ticks: { color: d.tickColor, font: { size: 10 } }, title: { display: true, text: 'Level (m)', color: d.tickColor, font: { size: 11 } }, suggestedMin: 0, suggestedMax: 5 },
            },
        },
    });

    // Topology bar
    topoChart = new Chart($('#topologyChart'), {
        type: 'bar',
        data: { labels: ['Pool 1', 'Pool 2', 'Pool 3'], datasets: [{ label: 'Level (m)', data: [0, 0, 0], backgroundColor: POOL_COLORS.slice(0, 3).map(c => c + '99'), borderColor: POOL_COLORS.slice(0, 3), borderWidth: 2, borderRadius: 6 }] },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false }, ticks: { color: d.tickColor, font: { size: 10 } } },
                y: { grid: { color: d.gridColor }, ticks: { color: d.tickColor, font: { size: 10 } }, beginAtZero: true, max: 6 },
            },
        },
    });

    // Anomaly scatter
    anomalyChart = new Chart($('#anomalyChart'), {
        type: 'scatter',
        data: { datasets: [{ label: 'Anomaly Score', data: [], backgroundColor: '#ef444499', borderColor: '#ef4444', pointRadius: 4 }] },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { color: d.tickColor } } },
            scales: {
                x: { grid: { color: d.gridColor }, ticks: { color: d.tickColor }, title: { display: true, text: 'Time Step', color: d.tickColor } },
                y: { grid: { color: d.gridColor }, ticks: { color: d.tickColor }, beginAtZero: true, title: { display: true, text: 'Score', color: d.tickColor } },
            },
        },
    });
}

function rebuildCharts() {
    if (mainChart) mainChart.destroy();
    if (topoChart) topoChart.destroy();
    if (anomalyChart) anomalyChart.destroy();
    initCharts();
}

/* ------------------------------------------------------------------ */
/* Health check                                                        */
/* ------------------------------------------------------------------ */
async function checkHealth() {
    const dot = $('#api-dot');
    const lbl = $('#api-label');
    try {
        const data = await api('/health');
        dot.className = 'indicator-dot online';
        lbl.textContent = 'System Online';
        if (data.version) lbl.textContent += ` (v${data.version})`;
    } catch {
        dot.className = 'indicator-dot offline';
        lbl.textContent = 'Offline';
    }
}

/* ------------------------------------------------------------------ */
/* Scenarios                                                           */
/* ------------------------------------------------------------------ */
async function loadScenarios() {
    try {
        const data = await api('/scenarios');
        const grid = $('#scenario-grid');
        if (!grid || !data.success) return;
        grid.innerHTML = '';
        data.scenarios.forEach(s => {
            const btn = document.createElement('button');
            btn.className = 'scenario-btn';
            btn.textContent = s.name;
            btn.title = s.instruction;
            btn.addEventListener('click', () => {
                const ta = $('#instruction-input');
                if (ta) ta.value = s.instruction;
                toast(`Loaded scenario: ${s.name}`, 'info');
            });
            grid.appendChild(btn);
        });
    } catch { /* ignore */ }
}

/* ------------------------------------------------------------------ */
/* Simulation controls                                                 */
/* ------------------------------------------------------------------ */
let currentInstruction = '';

function initSimControls() {
    const interpretBtn = $('#interpret-btn');
    const runBtn = $('#run-sim-btn');
    const stopBtn = $('#stop-sim-btn');
    const input = $('#instruction-input');
    const resultBox = $('#interpretation-result');

    interpretBtn.addEventListener('click', async () => {
        const text = (input.value || '').trim();
        if (!text) { toast('Please enter an instruction', 'warn'); return; }
        interpretBtn.disabled = true;
        try {
            const data = await api('/interpret', { method: 'POST', body: JSON.stringify({ instruction: text }) });
            if (data.success) {
                resultBox.classList.remove('hidden');
                $('#conf-score').textContent = (data.confidence * 100).toFixed(1) + '%';
                $('#scenario-name').textContent = data.scenario;
                $('#config-preview').textContent = JSON.stringify(data.config, null, 2);
                currentInstruction = text;
                toast('Instruction analyzed', 'success');
            }
        } catch (err) { toast('Analysis failed: ' + err.message, 'error'); }
        finally { interpretBtn.disabled = false; }
    });

    runBtn.addEventListener('click', async () => {
        runBtn.disabled = true; stopBtn.disabled = false;
        $('#sim-progress').classList.remove('hidden');
        $('#progress-fill').style.width = '30%';
        $('#progress-text').textContent = 'Starting simulation...';

        const script = currentInstruction ? [[0, currentInstruction]] : null;
        const sysType = (currentInstruction.toLowerCase().includes('cascade') || currentInstruction.includes('\u7EA7\u8054')) ? 'cascaded' : 'single';

        try {
            const data = await api('/simulation/run', { method: 'POST', body: JSON.stringify({ script, async: true, system_type: sysType }) });
            if (data.success) {
                simId = data.simulation_id;
                startPolling(simId);
                toast('Simulation started', 'success');
            }
        } catch (err) {
            toast('Failed to start: ' + err.message, 'error');
            resetSimUI();
        }
    });

    stopBtn.addEventListener('click', async () => {
        if (!simId) return;
        try {
            await api(`/simulation/${simId}/stop`, { method: 'POST' });
            toast('Stopping simulation...', 'info');
            stopBtn.disabled = true;
        } catch (err) { toast('Stop failed: ' + err.message, 'error'); }
    });
}

/* ------------------------------------------------------------------ */
/* Polling                                                             */
/* ------------------------------------------------------------------ */
function startPolling(id) {
    stopPolling();
    $('#sim-progress').classList.remove('hidden');
    $('#run-sim-btn').disabled = true;
    $('#stop-sim-btn').disabled = false;

    pollTimer = setInterval(async () => {
        try {
            const data = await api(`/simulation/${id}/history`);
            updateCharts(data);
            updateKPIs(data);
            updateTopology(data);

            if (data.status === 'completed') {
                stopPolling();
                $('#progress-fill').style.width = '100%';
                $('#progress-text').textContent = 'Completed';
                toast('Simulation completed', 'success');
            } else if (data.status === 'failed') {
                stopPolling();
                toast('Simulation failed', 'error');
            } else {
                // Estimate progress
                const total = data.time ? data.time.length : 0;
                const pct = Math.min(95, total * 2);
                $('#progress-fill').style.width = pct + '%';
                $('#progress-text').textContent = `Running... step ${total}`;
            }
        } catch (err) {
            console.error('Poll error:', err);
        }
    }, 1000);
}

function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    resetSimUI();
}

function resetSimUI() {
    const p = $('#sim-progress');
    if (p) p.classList.add('hidden');
    const r = $('#run-sim-btn');
    if (r) r.disabled = false;
    const s = $('#stop-sim-btn');
    if (s) s.disabled = true;
}

/* ------------------------------------------------------------------ */
/* Chart updates                                                       */
/* ------------------------------------------------------------------ */
function updateCharts(data) {
    if (!data.levels || !data.levels.length) return;
    const labels = data.time || data.levels[0].map((_, i) => i);

    // Ensure enough datasets
    while (mainChart.data.datasets.length < data.levels.length) {
        const i = mainChart.data.datasets.length;
        mainChart.data.datasets.push({
            label: `Pool ${i + 1}`,
            borderColor: POOL_COLORS[i % POOL_COLORS.length],
            backgroundColor: POOL_COLORS[i % POOL_COLORS.length] + '22',
            data: [], fill: true, tension: 0.35, pointRadius: 0, borderWidth: 2,
        });
    }

    mainChart.data.labels = labels;
    data.levels.forEach((poolData, i) => {
        if (mainChart.data.datasets[i]) mainChart.data.datasets[i].data = poolData;
    });
    mainChart.update('none');

    // Bar chart
    const last = data.levels[0].length - 1;
    const cur = data.levels.map(l => l[last]);
    topoChart.data.labels = cur.map((_, i) => `Pool ${i + 1}`);
    topoChart.data.datasets[0].data = cur;
    topoChart.data.datasets[0].backgroundColor = cur.map((_, i) => (POOL_COLORS[i % POOL_COLORS.length]) + '99');
    topoChart.data.datasets[0].borderColor = cur.map((_, i) => POOL_COLORS[i % POOL_COLORS.length]);
    topoChart.update();

    // Anomaly
    if (data.anomalies && anomalyChart) {
        anomalyChart.data.datasets[0].data = data.anomalies.map(a => ({ x: a.time, y: a.score }));
        anomalyChart.update();
    }

    // Fault log table
    updateFaultLog(data);
}

/* ------------------------------------------------------------------ */
/* KPI updates                                                         */
/* ------------------------------------------------------------------ */
function updateKPIs(data) {
    if (!data.levels || !data.levels.length) return;
    const last = data.levels[0].length - 1;
    const level = data.levels[0][last];
    const flow = data.flows ? (Array.isArray(data.flows[0]) ? data.flows[0][last] : data.flows[last]) : 0;

    setText('#kpi-level', level.toFixed(2));
    setText('#kpi-flow', (typeof flow === 'number' ? flow.toFixed(2) : '--'));
    setText('#kpi-status', data.status === 'running' ? 'Running' : data.status === 'completed' ? 'Idle' : 'Normal');

    // Health & monitoring
    if (data.system_health !== undefined) {
        setText('#kpi-health', (data.system_health * 100).toFixed(0) + '%');
        setText('#mon-health', (data.system_health * 100).toFixed(0) + '%');
    }
    if (data.statistics) {
        setText('#mon-anomalies', data.statistics.total_anomalies || 0);
        setText('#mon-heals', data.statistics.total_healings || 0);
        setText('#kpi-alerts', data.statistics.total_anomalies || 0);
    }
}

function setText(sel, val) {
    const el = $(sel);
    if (el) el.textContent = val;
}

/* ------------------------------------------------------------------ */
/* Topology SVG update                                                 */
/* ------------------------------------------------------------------ */
function updateTopology(data) {
    if (!data.levels || !data.levels.length) return;
    const last = data.levels[0].length - 1;
    const TANK_H = 100;
    const MIN_Y = 4;  // px from top of tank
    const MAX_H = 92;  // max water height in px

    data.levels.forEach((pool, i) => {
        const lvl = pool[last];
        const pct = Math.min(1, Math.max(0, lvl / 5));  // assuming max 5m
        const h = MIN_Y + pct * MAX_H;
        const y = TANK_H - h;
        const water = $(`#topo-water-${i}`);
        if (water) { water.setAttribute('y', y); water.setAttribute('height', h); }
        const lbl = $(`#topo-lbl-${i}`);
        if (lbl) lbl.textContent = lvl.toFixed(2) + 'm';
    });

    // Flow labels
    if (data.flows) {
        for (let i = 0; i < 2; i++) {
            const fl = $(`#topo-flow-${i}`);
            if (fl) {
                const v = Array.isArray(data.flows[0]) ? data.flows[i][last] : data.flows[last];
                fl.textContent = (typeof v === 'number' ? v.toFixed(1) : '--') + ' m\u00B3/s';
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Fault log table                                                     */
/* ------------------------------------------------------------------ */
function updateFaultLog(data) {
    const tbody = document.querySelector('#fault-log-table tbody');
    if (!tbody) return;
    const events = [];
    if (data.faults) data.faults.forEach(f => events.push({ ...f, cat: 'fault' }));
    if (data.healing_events) data.healing_events.forEach(h => events.push({ ...h, cat: 'healing' }));
    if (!events.length) return;
    events.sort((a, b) => (a.time || 0) - (b.time || 0));

    tbody.innerHTML = '';
    events.forEach(e => {
        const tr = document.createElement('tr');
        if (e.cat === 'fault') {
            tr.innerHTML = `<td>${e.time}</td><td><span class="badge badge-red">Fault</span></td><td>${e.component || '-'}</td><td>${e.severity || '-'}</td><td>${e.type || '-'}</td><td>-</td>`;
        } else {
            tr.innerHTML = `<td>${e.time}</td><td><span class="badge badge-green">Heal</span></td><td>-</td><td>-</td><td>Self-Healing</td><td>${e.success ? 'OK' : 'Fail'} (${(e.healing_time || 0).toFixed(1)}s)</td>`;
        }
        tbody.appendChild(tr);
    });
}

/* ------------------------------------------------------------------ */
/* History                                                             */
/* ------------------------------------------------------------------ */
async function loadHistory() {
    const tbody = document.querySelector('#history-table tbody');
    if (!tbody) return;
    tbody.innerHTML = '<tr><td colspan="5" class="table-empty">Loading...</td></tr>';
    try {
        const data = await api('/simulations');
        if (!data.success || !data.simulations.length) {
            tbody.innerHTML = '<tr><td colspan="5" class="table-empty">No simulations found</td></tr>';
            return;
        }
        tbody.innerHTML = '';
        data.simulations.forEach(sim => {
            const tr = document.createElement('tr');
            const statusClass = sim.status === 'completed' ? 'badge-green' : sim.status === 'failed' ? 'badge-red' : 'badge-amber';
            tr.innerHTML = `
                <td style="font-family:monospace;font-size:0.78rem">${String(sim.id).slice(0, 8)}</td>
                <td>${new Date(sim.start_time).toLocaleString()}</td>
                <td>${sim.total_hours || '--'}h</td>
                <td><span class="badge ${statusClass}">${sim.status}</span></td>
                <td><button class="btn btn-sm btn-ghost" onclick="loadSimResult('${sim.id}')">View</button></td>
            `;
            tbody.appendChild(tr);
        });
    } catch {
        tbody.innerHTML = '<tr><td colspan="5" class="table-empty">Failed to load</td></tr>';
    }
}
window.loadSimResult = async function (id) {
    try {
        const data = await api(`/simulation/${id}/history`);
        if (data) {
            updateCharts(data.history || data);
            // Switch to dashboard
            $$('.nav-item')[0].click();
            toast('Loaded simulation ' + String(id).slice(0, 8), 'info');
        }
    } catch (err) { toast('Load failed: ' + err.message, 'error'); }
};

/* ------------------------------------------------------------------ */
/* Config                                                              */
/* ------------------------------------------------------------------ */
async function loadConfig() {
    try {
        const data = await api('/config');
        const el = $('#config-display');
        if (el) el.textContent = JSON.stringify(data.config || data, null, 2);
    } catch {
        const el = $('#config-display');
        if (el) el.textContent = 'Error loading configuration';
    }
}

/* ------------------------------------------------------------------ */
/* Global event handlers                                               */
/* ------------------------------------------------------------------ */
window.triggerEvent = async (type) => {
    if (!simId) { toast('No active simulation', 'warn'); return; }
    const body = { type, data: type === 'flood' ? { magnitude: 20.0 } : { magnitude: 5.0 } };
    try {
        const res = await api(`/simulation/${simId}/event`, { method: 'POST', body: JSON.stringify(body) });
        if (res.success) toast(`${type} event injected`, 'success');
        else toast('Failed: ' + (res.error || 'unknown'), 'error');
    } catch (err) { toast('Error: ' + err.message, 'error'); }
};

window.updateGate = async (idx, value) => {
    setText(`#gate${idx}-val`, parseFloat(value).toFixed(1));
    if (!simId) return;
    try {
        const params = {};
        params[`gate_${idx}`] = parseFloat(value);
        await api(`/simulation/${simId}/control`, { method: 'POST', body: JSON.stringify({ params }) });
    } catch { /* silent */ }
};

window.resetOverride = async () => {
    if (!simId) { toast('No active simulation', 'warn'); return; }
    try {
        await api(`/simulation/${simId}/control/reset`, { method: 'POST' });
        toast('Overrides reset', 'success');
        $$('.slider').forEach(s => { s.value = 0; });
        for (let i = 0; i < 3; i++) setText(`#gate${i}-val`, '0.0');
    } catch (err) { toast('Reset failed', 'error'); }
};

/* ------------------------------------------------------------------ */
/* Language toggle (EN/CN stub)                                        */
/* ------------------------------------------------------------------ */
let lang = 'en';
function initLang() {
    $('#lang-toggle').addEventListener('click', () => {
        lang = lang === 'en' ? 'cn' : 'en';
        $('#lang-toggle').textContent = lang === 'en' ? 'EN' : 'CN';
        toast(lang === 'en' ? 'Language: English' : '\u8BED\u8A00: \u4E2D\u6587', 'info');
    });
}

/* ------------------------------------------------------------------ */
/* Init                                                                */
/* ------------------------------------------------------------------ */
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initNav();
    initCharts();
    initSimControls();
    initLang();
    tickClock();
    setInterval(tickClock, 1000);
    checkHealth();
    setInterval(checkHealth, 15000);
    loadScenarios();
});
