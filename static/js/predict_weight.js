/**
 * predict_weight.js
 * Feature C: Dynamic weight trajectory forecasting with interactive horizon pills (7, 14, 30, 60, 90 days),
 * Chart.js instance updates, loading overlay, and live milestone card refresh without page reload.
 */

document.addEventListener("DOMContentLoaded", function () {
    const ctx = document.getElementById('weightTrajectoryChart');
    if (!ctx) return;

    const data = window.INITIAL_WEIGHT_DATA || {};
    let currentWeight = data.current_weight || 0;
    let isLstmUnlocked = Boolean(data.lstm_unlocked);
    let currentDays = 30;
    const loadingOverlay = document.getElementById("chart_loading_overlay");
    const chartTitle = document.getElementById("chart_title");

    function buildDatasets(activeWeights, lstmWeights, metabolicWeights, lstmActive) {
        const sets = [];
        if (lstmActive) {
            sets.push({
                label: "Hybrid Forecast (Active)",
                data: activeWeights,
                borderColor: "#7c3aed",
                backgroundColor: "rgba(124, 58, 237, 0.08)",
                borderWidth: 3,
                pointRadius: 3,
                pointHoverRadius: 6,
                pointBackgroundColor: "#7c3aed",
                fill: true,
                tension: 0.35
            });

            if (lstmWeights && lstmWeights.length > 0) {
                sets.push({
                    label: "LSTM Deep Learning (Pure Neural)",
                    data: lstmWeights,
                    borderColor: "#c026d3",
                    borderDash: [3, 3],
                    borderWidth: 2,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    pointBackgroundColor: "#c026d3",
                    fill: false,
                    tension: 0.3
                });
            }

            if (metabolicWeights && metabolicWeights.length > 0) {
                sets.push({
                    label: "Metabolic Adaptation Baseline",
                    data: metabolicWeights,
                    borderColor: "#0284c7",
                    borderDash: [5, 5],
                    borderWidth: 2,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    pointBackgroundColor: "#0284c7",
                    fill: false,
                    tension: 0.25
                });
            }
        } else {
            sets.push({
                label: "Metabolic Adaptation Trajectory",
                data: activeWeights,
                borderColor: "#2563eb",
                backgroundColor: "rgba(37, 99, 235, 0.08)",
                borderWidth: 3,
                pointRadius: 3,
                pointHoverRadius: 6,
                pointBackgroundColor: "#2563eb",
                fill: true,
                tension: 0.3
            });
        }
        return sets;
    }

    const initialLabels = (data.trajectory || []).map(pt => `Day ${pt.day} (${(pt.date || '').slice(5)})`);
    const initialActive = (data.trajectory || []).map(pt => pt.weight);
    const initialLstm = (data.lstm_trajectory || []).map(pt => pt.weight);
    const initialMetabolic = (data.metabolic_trajectory || []).map(pt => pt.weight);

    const chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: initialLabels,
            datasets: buildDatasets(initialActive, initialLstm, initialMetabolic, isLstmUnlocked)
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 16,
                        font: {
                            size: 13,
                            weight: '600'
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleFont: { size: 13, weight: 'bold' },
                    bodyFont: { size: 12 },
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: function(context) {
                            const val = context.parsed.y;
                            const diff = (val - currentWeight).toFixed(2);
                            const sign = diff > 0 ? "+" : "";
                            return ` ${context.dataset.label}: ${val.toFixed(2)} kg (${sign}${diff} kg)`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: "rgba(226, 232, 240, 0.6)"
                    },
                    ticks: {
                        maxTicksLimit: 10,
                        font: { size: 11 },
                        color: "#64748b"
                    }
                },
                y: {
                    grid: {
                        color: "rgba(226, 232, 240, 0.8)"
                    },
                    ticks: {
                        callback: function(value) {
                            return value.toFixed(1) + " kg";
                        },
                        font: { size: 11 },
                        color: "#64748b"
                    }
                }
            }
        }
    });

    window.weightChartInstance = chartInstance;

    // Fetch and update forecast dynamically without full-page reload
    async function updateForecast(days) {
        if (loadingOverlay) loadingOverlay.classList.remove("hidden");

        try {
            const response = await fetch(`/api/weight-prediction?days=${days}`, {
                headers: { "Accept": "application/json" }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const json = await response.json();
            if (!json.success || !json.data) {
                throw new Error(json.error || "Unable to fetch prediction data.");
            }

            const d = json.data;
            currentWeight = d.current_weight || currentWeight;
            isLstmUnlocked = Boolean(d.lstm_unlocked);

            // Rebuild chart labels
            const newLabels = (d.trajectory || []).map(pt => `Day ${pt.day} (${(pt.date || '').slice(5)})`);
            const newActive = (d.trajectory || []).map(pt => pt.weight);
            const newLstm = (d.lstm_trajectory || []).map(pt => pt.weight);
            const newMetabolic = (d.metabolic_trajectory || []).map(pt => pt.weight);

            chartInstance.data.labels = newLabels;
            chartInstance.data.datasets = buildDatasets(newActive, newLstm, newMetabolic, isLstmUnlocked);
            chartInstance.update();

            if (chartTitle) {
                chartTitle.textContent = `📈 ${days}-Day Trajectory Curve Comparison`;
            }

            // Dynamically update milestone cards
            if (d.proj_7) {
                const p7w = document.getElementById("proj_7_weight");
                const p7c = document.getElementById("proj_7_change");
                const p7d = document.getElementById("proj_7_date");
                if (p7w) p7w.textContent = Number(d.proj_7.weight).toFixed(1);
                if (p7c) {
                    const sign = d.proj_7.change > 0 ? "+" : "";
                    p7c.textContent = `${sign}${Number(d.proj_7.change).toFixed(2)} kg from now`;
                    p7c.style.color = d.proj_7.change < 0 ? "#10b981" : (d.proj_7.change > 0 ? "#f59e0b" : "#64748b");
                }
                if (p7d) p7d.textContent = d.proj_7.date;
            }

            if (d.proj_14) {
                const p14w = document.getElementById("proj_14_weight");
                const p14c = document.getElementById("proj_14_change");
                const p14d = document.getElementById("proj_14_date");
                if (p14w) p14w.textContent = Number(d.proj_14.weight).toFixed(1);
                if (p14c) {
                    const sign = d.proj_14.change > 0 ? "+" : "";
                    p14c.textContent = `${sign}${Number(d.proj_14.change).toFixed(2)} kg from now`;
                    p14c.style.color = d.proj_14.change < 0 ? "#10b981" : (d.proj_14.change > 0 ? "#f59e0b" : "#64748b");
                }
                if (p14d) p14d.textContent = d.proj_14.date;
            }

            if (d.proj_30) {
                const p30w = document.getElementById("proj_30_weight");
                const p30c = document.getElementById("proj_30_change");
                const p30d = document.getElementById("proj_30_date");
                if (p30w) p30w.textContent = Number(d.proj_30.weight).toFixed(1);
                if (p30c) {
                    const sign = d.proj_30.change > 0 ? "+" : "";
                    p30c.textContent = `${sign}${Number(d.proj_30.change).toFixed(2)} kg from now`;
                    p30c.style.color = d.proj_30.change < 0 ? "#10b981" : (d.proj_30.change > 0 ? "#f59e0b" : "#64748b");
                }
                if (p30d) p30d.textContent = d.proj_30.date;
            }

        } catch (err) {
            console.error("Forecast update error:", err);
            alert("⚠️ Unable to refresh trajectory forecast. Please try again.");
        } finally {
            if (loadingOverlay) loadingOverlay.classList.add("hidden");
        }
    }

    // Horizon pills interaction
    const horizonBtns = document.querySelectorAll("#horizon_pills .filter-pill-btn");
    horizonBtns.forEach(btn => {
        btn.addEventListener("click", function () {
            const days = parseInt(this.dataset.days, 10);
            if (days === currentDays) return;
            horizonBtns.forEach(b => b.classList.remove("active"));
            this.classList.add("active");
            currentDays = days;
            updateForecast(days);
        });
    });

    // Refresh button
    const refreshBtn = document.getElementById("refresh_forecast_btn");
    if (refreshBtn) {
        refreshBtn.addEventListener("click", function () {
            updateForecast(currentDays);
        });
    }
});
