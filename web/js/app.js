// --- Global Variables ---
let globalSimId = null;
let mainChart = null;
let topologyChart = null;
let anomalyChart = null;
let pollInterval = null;
const API_BASE = ''; // Relative path

// --- Command Center Functions (Global) ---
window.triggerEvent = async (type) => {
    if (!globalSimId) {
        alert("No active simulation!");
        return;
    }

    let data = {};
    if (type === 'flood') data = { magnitude: 20.0 };
    if (type === 'drought') data = { magnitude: 5.0 };

    try {
        const res = await fetch(`${API_BASE}/simulation/${globalSimId}/event`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type: type, data: data })
        });
        const result = await res.json();
        if (result.success) {
            console.log(`Event ${type} triggered`);
        } else {
            alert("Failed: " + result.error);
        }
    } catch (err) {
        console.error(err);
    }
};

window.updateGate = async (idx, value) => {
    const label = document.getElementById(`gate${idx}-val`);
    if (label) label.innerText = value;

    if (!globalSimId) return;

    try {
        const params = {};
        params[`gate_${idx}`] = parseFloat(value);

        await fetch(`${API_BASE}/simulation/${globalSimId}/control`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ params })
        });
        console.log(`Gate ${idx} updated to ${value}`);
    } catch (err) {
        console.error(err);
    }
};

window.resetOverride = async () => {
    if (!globalSimId) return;

    try {
        const res = await fetch(`${API_BASE}/simulation/${globalSimId}/control/reset`, {
            method: 'POST'
        });
        const result = await res.json();
        if (result.success) {
            console.log("Overrides reset successfully");
            alert("All manual overrides have been cleared.");
        } else {
            alert("Failed to reset overrides: " + result.error);
        }
    } catch (err) {
        console.error(err);
        alert("Error resetting overrides");
    }
};

window.viewSim = (id) => {
    // Switch to dashboard and load
    const navItems = document.querySelectorAll('.nav-item');
    navItems[0].click(); // Click Dashboard
    // We might need to switch to simulation view if we want to see controls, 
    // but dashboard has the main chart. Let's stick to dashboard or simulation.
    // Actually, let's switch to Simulation view as it has the controls.
    // But wait, the history view 'View' button usually implies seeing the results.
    // Let's just load the results into the main chart.
    loadSimulationResults(id);
};

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    setupNavigation();
    setupSimulationControls();
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
});

function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-link'); // Changed selector to match HTML
    const views = document.querySelectorAll('.view-section');
    const pageTitle = document.getElementById('page-title');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const targetView = item.dataset.view;

            // Update Nav
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');

            // Update View
            views.forEach(view => {
                view.classList.remove('active');
                if (view.id === targetView) {
                    view.classList.add('active');
                }
            });

            // Update Title
            pageTitle.textContent = item.textContent.trim();

            // Load Data if needed
            if (targetView === 'view-history') loadHistory();
            if (targetView === 'view-config') loadConfig();
        });
    });
}

function initCharts() {
    // Main Chart (Water Levels)
    const ctx = document.getElementById('mainChart').getContext('2d');
    mainChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Pool 1 Level',
                    borderColor: '#3498db',
                    data: [],
                    fill: false,
                    tension: 0.4
                },
                {
                    label: 'Pool 2 Level',
                    borderColor: '#2ecc71',
                    data: [],
                    fill: false,
                    tension: 0.4
                },
                {
                    label: 'Pool 3 Level',
                    borderColor: '#e74c3c',
                    data: [],
                    fill: false,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                y: {
                    beginAtZero: false,
                    suggestedMin: 0,
                    suggestedMax: 5
                }
            }
        }
    });

    // Topology Chart
    const topoCtx = document.getElementById('topologyChart').getContext('2d');
    topologyChart = new Chart(topoCtx, {
        type: 'bar',
        data: {
            labels: ['Pool 1', 'Pool 2', 'Pool 3'],
            datasets: [{
                label: 'Water Level (m)',
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(52, 152, 219, 0.7)',
                    'rgba(46, 204, 113, 0.7)',
                    'rgba(231, 76, 60, 0.7)'
                ],
                borderColor: [
                    'rgba(52, 152, 219, 1)',
                    'rgba(46, 204, 113, 1)',
                    'rgba(231, 76, 60, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 6
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });

    // Anomaly Chart (Phase 5)
    const anomalyCanvas = document.getElementById('anomalyChart');
    if (anomalyCanvas) {
        const anomalyCtx = anomalyCanvas.getContext('2d');
        anomalyChart = new Chart(anomalyCtx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Anomaly Score',
                    data: [],
                    backgroundColor: 'rgba(255, 99, 132, 1)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: { display: true, text: 'Time Step' }
                    },
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Score' }
                    }
                }
            }
        });
    }
}

function setupSimulationControls() {
    const interpretBtn = document.getElementById('interpret-btn');
    const instructionInput = document.getElementById('instruction-input');
    const resultBox = document.getElementById('interpretation-result');
    const runSimBtn = document.getElementById('run-sim-btn');
    const stopSimBtn = document.getElementById('stop-sim-btn');
    const simProgress = document.getElementById('sim-progress');

    let currentInstruction = "";

    interpretBtn.addEventListener('click', async () => {
        const text = instructionInput.value.trim();
        if (!text) return;

        interpretBtn.disabled = true;
        interpretBtn.textContent = "Analyzing...";

        try {
            const res = await fetch(`${API_BASE}/interpret`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ instruction: text })
            });
            const data = await res.json();

            if (data.success) {
                resultBox.classList.remove('hidden');
                document.getElementById('conf-score').textContent = (data.confidence * 100).toFixed(1) + '%';
                document.getElementById('scenario-name').textContent = data.scenario;
                document.getElementById('config-preview').textContent = JSON.stringify(data.config, null, 2);
                currentInstruction = text;
            }
        } catch (err) {
            console.error(err);
            alert('Interpretation failed');
        } finally {
            interpretBtn.disabled = false;
            interpretBtn.textContent = "Analyze Instruction";
        }
    });

    runSimBtn.addEventListener('click', async () => {
        runSimBtn.disabled = true;
        stopSimBtn.disabled = false;
        simProgress.classList.remove('hidden');

        const script = currentInstruction ? [[0, currentInstruction]] : null;
        let systemType = 'single';
        if (currentInstruction.toLowerCase().includes('cascade') || currentInstruction.includes('级联')) {
            systemType = 'cascaded';
        }

        try {
            const res = await fetch(`${API_BASE}/simulation/run`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    script: script,
                    async: true,
                    system_type: systemType
                })
            });
            const data = await res.json();

            if (data.success) {
                globalSimId = data.simulation_id;
                startPolling(globalSimId);
            }
        } catch (err) {
            console.error(err);
            resetSimUI();
        }
    });

    stopSimBtn.addEventListener('click', async () => {
        if (!globalSimId) return;

        try {
            const res = await fetch(`${API_BASE}/simulation/${globalSimId}/stop`, { method: 'POST' });
            const data = await res.json();
            if (data.success) {
                alert('Simulation stopping...');
                stopSimBtn.disabled = true;
            } else {
                alert('Failed to stop: ' + data.error);
            }
        } catch (err) {
            console.error(err);
            alert('Error stopping simulation');
        }
    });
}

function startPolling(simId) {
    if (pollInterval) clearInterval(pollInterval);

    document.getElementById('sim-progress').classList.remove('hidden');
    document.getElementById('run-sim-btn').disabled = true;
    document.getElementById('stop-sim-btn').disabled = false;

    pollInterval = setInterval(async () => {
        try {
            // 1. Get History
            const res = await fetch(`${API_BASE}/simulation/${simId}/history`);
            if (res.status === 404) {
                console.warn("Simulation not found (404). Stopping poll.");
                stopPolling();
                return;
            }
            const data = await res.json();

            if (data.status === 'completed' || data.status === 'failed') {
                stopPolling();
                alert(`Simulation ${data.status}!`);
            }

            updateCharts(data);
            updateDashboard(data);

            // 2. Get Health (Phase 5)
            const healthRes = await fetch(`${API_BASE}/simulation/${simId}/health`);
            if (healthRes.ok) {
                const healthData = await healthRes.json();
                updateMonitoring(healthData);
            }

        } catch (err) {
            console.error("Polling error:", err);
        }
    }, 1000);
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
    document.getElementById('sim-progress').classList.add('hidden');
    document.getElementById('run-sim-btn').disabled = false;
    document.getElementById('stop-sim-btn').disabled = true;
}

function updateCharts(data) {
    if (!data.levels || data.levels.length === 0) return;

    const labels = data.time || Array.from({ length: data.levels[0].length }, (_, i) => i);

    // Update Main Chart
    mainChart.data.labels = labels;
    data.levels.forEach((poolLevels, i) => {
        if (mainChart.data.datasets[i]) {
            mainChart.data.datasets[i].data = poolLevels;
        }
    });
    mainChart.update('none');

    // Update Topology Chart
    const lastIdx = data.levels[0].length - 1;
    const currentLevels = data.levels.map(l => l[lastIdx]);
    topologyChart.data.datasets[0].data = currentLevels;
    topologyChart.update();

    // Update Anomaly Chart
    if (data.anomalies && anomalyChart) {
        const scatterData = data.anomalies.map(a => ({
            x: a.time,
            y: a.score
        }));
        anomalyChart.data.datasets[0].data = scatterData;
        anomalyChart.update();
    }

    // Update Fault Log Table
    const tbody = document.querySelector('#fault-log-table tbody');
    if (tbody && data.faults) {
        tbody.innerHTML = '';
        // Combine faults and healing events for a timeline view
        const events = [];
        data.faults.forEach(f => events.push({ ...f, category: 'fault' }));
        if (data.healing_events) {
            data.healing_events.forEach(h => events.push({ ...h, category: 'healing' }));
        }
        events.sort((a, b) => a.time - b.time);

        events.forEach(e => {
            const row = document.createElement('tr');
            if (e.category === 'fault') {
                row.innerHTML = `
                    <td>${e.time}</td>
                    <td><span class="badge bg-danger">Fault</span></td>
                    <td>${e.component}</td>
                    <td>${e.severity}</td>
                    <td>${e.type}</td>
                    <td>-</td>
                `;
            } else {
                row.innerHTML = `
                    <td>${e.time}</td>
                    <td><span class="badge bg-success">Healing</span></td>
                    <td>-</td>
                    <td>-</td>
                    <td>Self-Healing</td>
                    <td>${e.success ? 'Success' : 'Failed'} (${e.healing_time.toFixed(1)}s)</td>
                `;
            }
            tbody.appendChild(row);
        });
    }
}

function updateDashboard(data) {
    if (data.levels && data.levels.length > 0) {
        const lastIdx = data.levels[0].length - 1;
        const level = data.levels[0][lastIdx];
        const flow = data.flows ? data.flows[lastIdx][0] : 0;

        document.getElementById('current-level').textContent = level.toFixed(2) + ' m';
        document.getElementById('current-flow').textContent = flow.toFixed(2) + ' m³/s';
        document.getElementById('target-level').textContent = "3.00";
    }
}

function updateMonitoring(data) {
    if (data.system_health !== undefined) {
        document.getElementById('health-score').textContent = (data.system_health * 100).toFixed(1) + '%';
    }

    if (data.statistics) {
        document.getElementById('anomaly-count').textContent = data.statistics.total_anomalies;
        document.getElementById('healing-count').textContent = data.statistics.total_healings;

        const healingStatus = document.getElementById('healing-status');
        if (data.operation_mode) {
            healingStatus.textContent = data.operation_mode;
        }
    }
}

function resetSimUI() {
    const runSimBtn = document.getElementById('run-sim-btn');
    const stopSimBtn = document.getElementById('stop-sim-btn');
    const simProgress = document.getElementById('sim-progress');

    runSimBtn.disabled = false;
    stopSimBtn.disabled = true;
    simProgress.classList.add('hidden');
}

async function loadSimulationResults(simId) {
    try {
        const res = await fetch(`${API_BASE}/simulation/${simId}/history`);
        const data = await res.json();

        if (data.success) {
            updateCharts(data.history || data);
        }
    } catch (err) {
        console.error(err);
    }
}

async function loadHistory() {
    const tbody = document.querySelector('#history-table tbody');
    if (!tbody) return;

    tbody.innerHTML = '<tr><td colspan="5">Loading...</td></tr>';

    try {
        const res = await fetch(`${API_BASE}/simulations`);
        const data = await res.json();

        if (data.success) {
            tbody.innerHTML = '';
            data.simulations.forEach(sim => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${sim.id}</td>
                    <td>${new Date(sim.start_time).toLocaleString()}</td>
                    <td>${sim.total_hours || '--'}h</td>
                    <td><span class="value ${sim.status === 'completed' ? 'online' : ''}">${sim.status}</span></td>
                    <td>
                        <button class="btn-primary" onclick="window.viewSim(${sim.id})">View</button>
                    </td>
                `;
                tbody.appendChild(row);
            });
        }
    } catch (err) {
        tbody.innerHTML = '<tr><td colspan="5">Error loading history</td></tr>';
    }
}

async function loadConfig() {
    try {
        const res = await fetch(`${API_BASE}/config`);
        const data = await res.json();
        document.getElementById('config-display').textContent = JSON.stringify(data.config, null, 2);
    } catch (err) {
        document.getElementById('config-display').textContent = 'Error loading config';
    }
}

// --- Topology Visualization ---
function resizeCanvas() {
    const topoCanvas = document.getElementById('topologyChart');
    if (!topoCanvas) return;
    const container = topoCanvas.parentElement;
    topoCanvas.width = container.clientWidth;
    topoCanvas.height = container.clientHeight;
}
