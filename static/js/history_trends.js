/**
 * history_trends.js
 * Feature C: Dynamic history dual-axis Chart.js trend visualization (Calories In, Calories Out, Net, Weight on y1)
 * with quick date range pills (7, 14, 30 days) and async table & metric updates.
 */

document.addEventListener("DOMContentLoaded", function () {
    const ctx = document.getElementById("historyTrendsChart");
    if (!ctx) return;

    const filterForm = document.getElementById("history_filter_form");
    const startDateInput = document.getElementById("start_date");
    const endDateInput = document.getElementById("end_date");
    const chartLoading = document.getElementById("history_chart_loading");

    const initialHistory = window.INITIAL_HISTORY || [];

    function extractChartData(historyRows) {
        const labels = historyRows.map(r => r.date);
        const cIn = historyRows.map(r => r.calories_in);
        const cOut = historyRows.map(r => r.calories_out);
        const net = historyRows.map(r => r.net_calories);
        const weight = historyRows.map(r => r.weight_kg);

        return { labels, cIn, cOut, net, weight };
    }

    const initData = extractChartData(initialHistory);

    const trendsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: initData.labels,
            datasets: [
                {
                    type: 'bar',
                    label: 'Calories In (kcal)',
                    data: initData.cIn,
                    backgroundColor: 'rgba(37, 99, 235, 0.75)',
                    borderColor: '#2563eb',
                    borderWidth: 1,
                    yAxisID: 'y'
                },
                {
                    type: 'bar',
                    label: 'Calories Out (kcal)',
                    data: initData.cOut,
                    backgroundColor: 'rgba(234, 88, 12, 0.75)',
                    borderColor: '#ea580c',
                    borderWidth: 1,
                    yAxisID: 'y'
                },
                {
                    type: 'line',
                    label: 'Net Balance (kcal)',
                    data: initData.net,
                    borderColor: '#10b981',
                    backgroundColor: '#10b981',
                    borderWidth: 2,
                    pointRadius: 3,
                    fill: false,
                    tension: 0.2,
                    yAxisID: 'y'
                },
                {
                    type: 'line',
                    label: 'Actual Recorded Weight (Historical, kg)',
                    data: initData.weight,
                    borderColor: '#7c3aed',
                    backgroundColor: '#7c3aed',
                    borderWidth: 2,
                    borderDash: [4, 4],
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    spanGaps: true,
                    fill: false,
                    tension: 0.15,
                    yAxisID: 'y1'
                }
            ]
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
                        padding: 14,
                        font: { size: 12, weight: '600' }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    padding: 10,
                    cornerRadius: 8,
                    callbacks: {
                        label: function (context) {
                            const val = context.parsed.y;
                            if (val === null || val === undefined) return null;
                            if (context.dataset.yAxisID === 'y1') {
                                return ` ${context.dataset.label}: ${val.toFixed(1)} kg`;
                            }
                            return ` ${context.dataset.label}: ${val.toFixed(1)} kcal`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: "rgba(226, 232, 240, 0.6)" },
                    ticks: { font: { size: 11 }, color: "#64748b" }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: { display: true, text: 'Calories (kcal)', font: { size: 12, weight: 'bold' } },
                    grid: { color: "rgba(226, 232, 240, 0.8)" }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: { display: true, text: 'Body Weight (kg)', font: { size: 12, weight: 'bold' } },
                    grid: { drawOnChartArea: false },
                    ticks: {
                        callback: function (val) {
                            return val.toFixed(1) + " kg";
                        }
                    }
                }
            }
        }
    });

    async function updateHistoryData(start, end) {
        if (chartLoading) chartLoading.classList.remove("hidden");

        try {
            const url = `/api/history?start_date=${encodeURIComponent(start)}&end_date=${encodeURIComponent(end)}`;
            const response = await fetch(url, {
                headers: { "Accept": "application/json" }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            if (!data.success) {
                throw new Error(data.error || "Failed to load history.");
            }

            // Update Chart.js (history returned in chronological ASC order)
            const chartData = extractChartData(data.history || []);
            trendsChart.data.labels = chartData.labels;
            trendsChart.data.datasets[0].data = chartData.cIn;
            trendsChart.data.datasets[1].data = chartData.cOut;
            trendsChart.data.datasets[2].data = chartData.net;
            trendsChart.data.datasets[3].data = chartData.weight;
            trendsChart.update();

            // Update Summary Cards
            const inEl = document.getElementById("history_total_in");
            const outEl = document.getElementById("history_total_out");
            const netEl = document.getElementById("history_total_net");
            const daysEl = document.getElementById("history_active_days");

            if (inEl) inEl.textContent = Number(data.total_in).toFixed(1);
            if (outEl) outEl.textContent = Number(data.total_burned).toFixed(1);
            if (netEl) {
                const netVal = Number(data.total_net);
                netEl.textContent = (netVal > 0 ? "+" : "") + netVal.toFixed(1);
                netEl.style.color = netVal > 0 ? "#b45309" : (netVal < 0 ? "#047857" : "#334155");

                const netStatus = document.getElementById("history_net_status");
                if (netStatus) {
                    netStatus.textContent = netVal > 0 ? "Calorie Surplus" : (netVal < 0 ? "Calorie Deficit" : "Balanced");
                }
            }
            if (daysEl) daysEl.textContent = data.active_days;

            // Update Table
            const tbody = document.getElementById("history_table_tbody");
            const tableContainer = document.getElementById("table_container");
            const emptyMsg = document.getElementById("history_empty_msg");
            const breakdownHeading = document.getElementById("breakdown_heading");

            if (breakdownHeading) {
                breakdownHeading.textContent = `Daily Breakdown (${data.start_date} to ${data.end_date})`;
            }

            if (!data.history || data.history.length === 0) {
                if (tableContainer) tableContainer.style.display = "none";
                if (emptyMsg) {
                    emptyMsg.style.display = "block";
                    emptyMsg.textContent = `No food or exercise entries found between ${data.start_date} and ${data.end_date}.`;
                }
            } else {
                if (tableContainer) tableContainer.style.display = "block";
                if (emptyMsg) emptyMsg.style.display = "none";
                if (tbody) {
                    tbody.textContent = "";

                    // Display in reverse chronological order for table (latest first)
                    const descHistory = [...data.history].reverse();
                    descHistory.forEach(row => {
                        const tr = document.createElement("tr");

                        const tdDate = document.createElement("td");
                        const dateBold = document.createElement("strong");
                        dateBold.textContent = row.date;
                        tdDate.appendChild(dateBold);

                        const tdIn = document.createElement("td");
                        tdIn.style.color = "#2563eb";
                        tdIn.style.fontWeight = "600";
                        tdIn.textContent = `${Number(row.calories_in).toFixed(1)} kcal`;

                        const tdOut = document.createElement("td");
                        tdOut.style.color = "#ea580c";
                        tdOut.style.fontWeight = "600";
                        tdOut.textContent = `${Number(row.calories_out).toFixed(1)} kcal`;

                        const tdNet = document.createElement("td");
                        const netSpan = document.createElement("span");
                        netSpan.style.padding = "3px 8px";
                        netSpan.style.borderRadius = "6px";
                        netSpan.style.fontWeight = "600";
                        netSpan.style.fontSize = "13px";
                        if (row.net_calories > 0) {
                            netSpan.style.background = "#fef3c7";
                            netSpan.style.color = "#92400e";
                            netSpan.textContent = `+${Number(row.net_calories).toFixed(1)} (Surplus)`;
                        } else if (row.net_calories < 0) {
                            netSpan.style.background = "#d1fae5";
                            netSpan.style.color = "#065f46";
                            netSpan.textContent = `${Number(row.net_calories).toFixed(1)} (Deficit)`;
                        } else {
                            netSpan.style.background = "#f1f5f9";
                            netSpan.style.color = "#475569";
                            netSpan.textContent = `0.0 (Balanced)`;
                        }
                        tdNet.appendChild(netSpan);

                        const tdWeight = document.createElement("td");
                        if (row.weight_kg !== null && row.weight_kg !== undefined) {
                            const wBold = document.createElement("strong");
                            wBold.style.color = "#7c3aed";
                            wBold.textContent = `${Number(row.weight_kg).toFixed(1)} kg`;
                            tdWeight.appendChild(wBold);
                        } else {
                            const dashSpan = document.createElement("span");
                            dashSpan.style.color = "#94a3b8";
                            dashSpan.textContent = "--";
                            tdWeight.appendChild(dashSpan);
                        }

                        const tdAction = document.createElement("td");
                        const diaryLink = document.createElement("a");
                        diaryLink.href = `/diary/${encodeURIComponent(row.date)}`;
                        diaryLink.style.color = "#2563eb";
                        diaryLink.style.fontSize = "13px";
                        diaryLink.style.textDecoration = "underline";
                        diaryLink.textContent = "View Diary →";
                        tdAction.appendChild(diaryLink);

                        tr.appendChild(tdDate);
                        tr.appendChild(tdIn);
                        tr.appendChild(tdOut);
                        tr.appendChild(tdNet);
                        tr.appendChild(tdWeight);
                        tr.appendChild(tdAction);

                        tbody.appendChild(tr);
                    });
                }
            }

        } catch (err) {
            console.error("History fetch error:", err);
            alert("⚠️ Unable to fetch history records. Please try again.");
        } finally {
            if (chartLoading) chartLoading.classList.add("hidden");
        }
    }

    // Intercept form submission
    if (filterForm) {
        filterForm.addEventListener("submit", function (e) {
            e.preventDefault();
            const start = startDateInput.value;
            const end = endDateInput.value;
            if (!start || !end) return;

            // Reset active pill
            document.querySelectorAll("#history_quick_pills .filter-pill-btn").forEach(b => b.classList.remove("active"));
            updateHistoryData(start, end);
        });
    }

    // Quick range pills interaction
    const pillBtns = document.querySelectorAll("#history_quick_pills .filter-pill-btn");
    pillBtns.forEach(btn => {
        btn.addEventListener("click", function () {
            const days = parseInt(this.dataset.days, 10);
            pillBtns.forEach(b => b.classList.remove("active"));
            this.classList.add("active");

            const today = new Date();
            const pastDate = new Date();
            pastDate.setDate(today.getDate() - days);

            const formatDate = d => d.toISOString().split("T")[0];
            const endStr = formatDate(today);
            const startStr = formatDate(pastDate);

            if (startDateInput) startDateInput.value = startStr;
            if (endDateInput) endDateInput.value = endStr;

            updateHistoryData(startStr, endStr);
        });
    });
});
