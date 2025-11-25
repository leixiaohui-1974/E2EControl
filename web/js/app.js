document.addEventListener('DOMContentLoaded', () => {
    // --- Navigation ---
    const navItems = document.querySelectorAll('.nav-item');
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
                if (view.id === `view-${targetView}`) {
                    view.classList.add('active');
                }
            });

            // Update Title
            pageTitle.textContent = item.textContent.trim();

            // Load Data if needed
            if (targetView === 'history') loadHistory();
            if (targetView === 'settings') loadConfig();
        });
    });

    // --- Dashboard Chart ---
    const ctx = document.getElementById('mainChart').getContext('2d');
    let mainChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Water Level (m)',
                data: [],
                borderColor: '#2563eb',
                tension: 0.4
            }, {
                label: 'Target Level (m)',
                data: [],
                borderColor: '#ef4444',
                borderDash: [5, 5],
                tension: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: {
                intersect: false,
                mode: 'index',
            },
        }
    });

    // --- Simulation Control ---
    const interpretBtn = document.getElementById('interpret-btn');
    const instructionInput = document.getElementById('instruction-input');
    const resultBox = document.getElementById('interpretation-result');
    const runSimBtn = document.getElementById('run-sim-btn');
    const stopSimBtn = document.getElementById('stop-sim-btn');
    const simProgress = document.getElementById('sim-progress');

    let currentInstruction = "";
    let currentSimId = null;
    let pollInterval = null;

    interpretBtn.addEventListener('click', async () => {
        const text = instructionInput.value.trim();
        if (!text) return;

        interpretBtn.disabled = true;
        interpretBtn.textContent = "Analyzing...";

        try {
            const res = await fetch('/interpret', {
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

        // Construct script from current instruction or default
        const script = currentInstruction
            ? [[0, currentInstruction]]
            : null; // Backend will use default if null

        // Check if user wants cascaded (simple toggle for now, or just run cascaded if specific keyword)
        let systemType = 'single';
        if (currentInstruction.toLowerCase().includes('cascade') || currentInstruction.includes('级联')) {
            systemType = 'cascaded';
        }

        try {
            const res = await fetch('/simulation/run', {
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
                currentSimId = data.simulation_id;
                startPolling(currentSimId);
            }
        } catch (err) {
            console.error(err);
            resetSimUI();
        }
    });

    function startPolling(simId) {
        if (pollInterval) clearInterval(pollInterval);

        pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/simulation/${simId}/status`);
                const data = await res.json();

                if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    resetSimUI();
                    loadSimulationResults(simId);
                    alert('Simulation Completed!');
                } else if (data.status === 'failed') {
                    clearInterval(pollInterval);
                    resetSimUI();
                    alert('Simulation Failed: ' + data.error);
                }
            } catch (err) {
                console.error(err);
            }
        }, 1000);
    }

    function resetSimUI() {
        runSimBtn.disabled = false;
        stopSimBtn.disabled = true;
        simProgress.classList.add('hidden');
    }

    async function loadSimulationResults(simId) {
        try {
            const res = await fetch(`/simulation/${simId}/history`);
            const data = await res.json();

            if (data.success) {
                const history = data.history;

                // Check if cascaded (levels is array of arrays)
                if (history.levels && Array.isArray(history.levels[0])) {
                    // Cascaded Visualization
                    const numPools = history.levels[0].length;

                    // Update Chart Datasets
                    const datasets = [];
                    const colors = ['#2563eb', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];

                    for (let i = 0; i < numPools; i++) {
                        datasets.push({
                            label: `Pool ${i} Level (m)`,
                            data: history.levels.map(l => l[i]),
                            borderColor: colors[i % colors.length],
                            tension: 0.4
                        });
                    }

                    mainChart.data.labels = history.time;
                    mainChart.data.datasets = datasets;
                    mainChart.update();

                    // Update Stats (Show Pool 0)
                    const lastIdx = history.levels.length - 1;
                    document.getElementById('current-level').textContent = history.levels[lastIdx][0].toFixed(2) + ' m (P0)';
                    document.getElementById('target-level').textContent = "3.00";
                    document.getElementById('current-flow').textContent = history.flows[lastIdx][0].toFixed(2) + ' m³/s';
                    document.getElementById('current-flow').textContent = history.q_in[history.q_in.length - 1].toFixed(2) + ' m³/s';

                    mainChart.data.datasets = [{
                        label: 'Water Level (m)',
                        data: history.level,
                        borderColor: '#2563eb',
                        tension: 0.4
                    }, {
                        label: 'Target Level (m)',
                        data: history.target_level,
                        borderColor: '#ef4444',
                        borderDash: [5, 5],
                        tension: 0
                    }];
                    mainChart.data.labels = history.time;
                    mainChart.update();
                }
            }
        } catch (err) {
            console.error(err);
        }
    }

    // --- History & Config ---
    async function loadHistory() {
        const tbody = document.getElementById('history-table-body');
        tbody.innerHTML = '<tr><td colspan="5">Loading...</td></tr>';

        try {
            const res = await fetch('/simulations');
            const data = await res.json();

            if (data.success) {
                tbody.innerHTML = '';
                data.simulations.forEach(sim => {
                    const row = `
                        <tr>
                            <td>${sim.id}</td>
                            <td>${new Date(sim.start_time).toLocaleString()}</td>
                            <td>${sim.total_hours || '--'}h</td>
                            <td><span class="value ${sim.status === 'completed' ? 'online' : ''}">${sim.status}</span></td>
                            <td>
                                <button class="btn-primary" onclick="window.viewSim(${sim.id})">View</button>
                            </td>
                        </tr>
                    `;
                    tbody.innerHTML += row;
                });
            }
        } catch (err) {
            tbody.innerHTML = '<tr><td colspan="5">Error loading history</td></tr>';
        }
    }

    async function loadConfig() {
        try {
            const res = await fetch('/config');
            const data = await res.json();
            document.getElementById('config-display').textContent = JSON.stringify(data.config, null, 2);
        } catch (err) {
            document.getElementById('config-display').textContent = 'Error loading config';
        }
    }

    // Expose for onclick
    window.viewSim = (id) => {
        // Switch to dashboard and load
        navItems[0].click();
        loadSimulationResults(id);
    };

    // --- Command Center Functions ---
    window.triggerEvent = async (type) => {
        if (!currentSimId) {
            alert("No active simulation!");
            return;
        }

        let data = {};
        if (type === 'flood') data = { magnitude: 20.0 };
        if (type === 'drought') data = { magnitude: 5.0 };

        try {
            const res = await fetch(`/simulation/${currentSimId}/event`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: type, data: data })
            });
            const result = await res.json();
            if (result.success) {
                // alert(`Event ${type} triggered!`);
                console.log(`Event ${type} triggered`);
            } else {
                alert("Failed: " + result.error);
            }
        } catch (err) {
            console.error(err);
        }
    };

    window.updateGate = async (gateIdx, value) => {
        document.getElementById(`gate${gateIdx}-val`).textContent = value + ' m³/s';

        if (!currentSimId) return;

        try {
            const params = {};
            params[`gate_${gateIdx}`] = parseFloat(value);

            await fetch(`/simulation/${currentSimId}/control`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ params: params })
            });
        } catch (err) {
            console.error(err);
        }
    };

    window.resetOverride = async () => {
        if (!currentSimId) return;
        // Reset by sending empty override or specific command
        // For now, our backend updates overrides, so we need to remove them.
        // But our backend update is a merge. We might need a way to clear.
        // Let's assume sending null or special flag clears it, 
        // OR we just set it back to default range if we knew it.
        // Actually, let's implement a 'clear' param in backend or just set a flag.
        // For this PoC, let's just set 'force_scenario' to null if we used it, 
        // but for gates, the controller will be overridden.
        // We need a way to remove the key from overrides.
        // Let's just reload the page or start new sim for now to clear, 
        // or add a 'clear_overrides' endpoint later.
        alert("Reset not fully implemented in backend yet. Please restart simulation to clear overrides.");
    };
});
