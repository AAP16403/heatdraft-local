/* ═══════════════════════════════════════════════════════════════════
   HeatDraft Inverse Design Dashboard — Frontend Logic
   ═══════════════════════════════════════════════════════════════════ */

"use strict";

// ── State ──────────────────────────────────────────────────────────
let CONFIG = null;           // populated from /api/config
let LAST_RESULTS = null;     // populated after /api/run
let TARGET_RATES = [];       // editable list of target removal rates
let sortCol = null;
let sortAsc = true;
let scatterChart = null;
let rangesChart = null;

// ── DOM Helpers ────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

function show(el)  { el.classList.remove("hidden"); }
function hide(el)  { el.classList.add("hidden"); }

function fmt(n, d = 2) {
    if (n === null || n === undefined) return "—";
    return Number(n).toFixed(d);
}

function fmtInt(n) {
    if (n === null || n === undefined) return "—";
    return Number(n).toLocaleString();
}


// ═══════════════════════════════════════════════════════════════════
//  TOAST NOTIFICATIONS
// ═══════════════════════════════════════════════════════════════════

function showToast(msg, type = "info", duration = 4500) {
    const container = $("#toast-container");
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = msg;
    container.appendChild(toast);
    setTimeout(() => {
        toast.classList.add("fadeout");
        toast.addEventListener("animationend", () => toast.remove());
    }, duration);
}


// ═══════════════════════════════════════════════════════════════════
//  INITIALISATION
// ═══════════════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", async () => {
    try {
        const resp = await fetch("/api/config");
        if (!resp.ok) throw new Error(`Config fetch failed: ${resp.status}`);
        CONFIG = await resp.json();
        initUI();
    } catch (err) {
        showToast(`Failed to load configuration: ${err.message}`, "error");
        console.error(err);
    }
});

function initUI() {
    renderHeaderStats();
    renderPollutantColSelect();
    renderPollutantValues();
    initTargetChips();
    renderControllableCols();
    initAdvancedPanel();
    initSliders();
    bindRunButton();
    bindTabNavigation();
    bindExportCSV();
}


// ═══════════════════════════════════════════════════════════════════
//  HEADER STATS
// ═══════════════════════════════════════════════════════════════════

function renderHeaderStats() {
    const el = $("#header-stats");
    const c = CONFIG;
    el.innerHTML = `
        <div class="stat-pill">Target <span class="stat-value">${c.target_col}</span></div>
        <div class="stat-pill">Train <span class="stat-value">${fmtInt(c.train_rows)}</span></div>
        <div class="stat-pill">Features <span class="stat-value">${c.selected_features}</span></div>
        <div class="stat-pill">High <span class="stat-value">${fmtInt(c.high_rows)}</span></div>
        <div class="stat-pill">Range <span class="stat-value">${fmt(c.target_range[0], 0)}–${fmt(c.target_range[1], 0)}%</span></div>
    `;
}


// ═══════════════════════════════════════════════════════════════════
//  POLLUTANT COLUMN SELECT
// ═══════════════════════════════════════════════════════════════════

function renderPollutantColSelect() {
    const sel = $("#pollutant-col-select");
    sel.innerHTML = "";
    for (const col of CONFIG.categorical_cols) {
        const opt = document.createElement("option");
        opt.value = col;
        opt.textContent = col;
        if (col === CONFIG.defaults.pollutant_col) opt.selected = true;
        sel.appendChild(opt);
    }
    // If the default isn't in categorical_cols, add numeric cols too
    if (!CONFIG.categorical_cols.includes(CONFIG.defaults.pollutant_col)) {
        for (const col of CONFIG.numeric_cols) {
            const opt = document.createElement("option");
            opt.value = col;
            opt.textContent = col + " (numeric)";
            if (col === CONFIG.defaults.pollutant_col) opt.selected = true;
            sel.appendChild(opt);
        }
    }
    sel.addEventListener("change", () => renderPollutantValues());
}


// ═══════════════════════════════════════════════════════════════════
//  POLLUTANT VALUES CHECKBOXES
// ═══════════════════════════════════════════════════════════════════

function renderPollutantValues() {
    const col = $("#pollutant-col-select").value;
    const group = $("#pollutant-values-group");
    group.innerHTML = "";

    const entries = CONFIG.cat_value_counts[col] || [];
    if (entries.length === 0) {
        group.innerHTML = `<span class="field-label-sm" style="padding:8px">No values found for this column</span>`;
        updatePollutantBadge();
        return;
    }

    entries.forEach((item, i) => {
        const label = document.createElement("label");
        label.className = "checkbox-label";
        label.innerHTML = `
            <input type="checkbox" name="pollutant-val" value="${escapeHtml(item.value)}" ${i < 3 ? "checked" : ""}>
            <span class="checkmark"></span>
            ${escapeHtml(item.value)}
            <span class="cb-count">${fmtInt(item.count)}</span>
        `;
        group.appendChild(label);
    });

    updatePollutantBadge();
    group.addEventListener("change", updatePollutantBadge);
}

function updatePollutantBadge() {
    const checked = $$('input[name="pollutant-val"]:checked').length;
    $("#pollutant-count-badge").textContent = checked;
}


// ═══════════════════════════════════════════════════════════════════
//  TARGET REMOVAL RATE CHIPS
// ═══════════════════════════════════════════════════════════════════

function initTargetChips() {
    TARGET_RATES = [...CONFIG.defaults.target_rates];
    renderChips();

    $("#target-add-btn").addEventListener("click", addTargetChip);
    $("#target-add-input").addEventListener("keydown", (e) => {
        if (e.key === "Enter") addTargetChip();
    });
}

function addTargetChip() {
    const input = $("#target-add-input");
    const val = parseFloat(input.value);
    if (isNaN(val) || val < 0 || val > 100) {
        showToast("Target must be a number between 0 and 100", "error");
        return;
    }
    if (TARGET_RATES.includes(val)) {
        showToast("This target already exists", "error");
        return;
    }
    TARGET_RATES.push(val);
    TARGET_RATES.sort((a, b) => a - b);
    input.value = "";
    renderChips();
}

function removeTargetChip(val) {
    TARGET_RATES = TARGET_RATES.filter((t) => t !== val);
    renderChips();
}

function renderChips() {
    const container = $("#target-chips");
    container.innerHTML = "";
    for (const t of TARGET_RATES) {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.innerHTML = `
            ${t}%
            <button class="chip-remove" title="Remove">
                <svg width="10" height="10" viewBox="0 0 10 10"><path d="M2 2l6 6M8 2l-6 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
            </button>
        `;
        chip.querySelector(".chip-remove").addEventListener("click", () => removeTargetChip(t));
        container.appendChild(chip);
    }
}


// ═══════════════════════════════════════════════════════════════════
//  CONTROLLABLE COLUMNS
// ═══════════════════════════════════════════════════════════════════

function renderControllableCols() {
    const group = $("#controllable-cols-group");
    group.innerHTML = "";

    for (const col of CONFIG.numeric_cols) {
        const stats = CONFIG.num_stats[col] || {};
        const label = document.createElement("label");
        label.className = "checkbox-label";
        label.innerHTML = `
            <input type="checkbox" name="ctrl-col" value="${escapeHtml(col)}" checked>
            <span class="checkmark"></span>
            ${escapeHtml(col)}
            <span class="cb-count">${fmt(stats.min, 1)}–${fmt(stats.max, 1)}</span>
        `;
        group.appendChild(label);
    }

    updateControllableBadge();
    group.addEventListener("change", updateControllableBadge);

    // Select All toggle
    const selectAll = $("#ctrl-select-all");
    selectAll.addEventListener("change", () => {
        $$('input[name="ctrl-col"]').forEach((cb) => (cb.checked = selectAll.checked));
        updateControllableBadge();
    });
}

function updateControllableBadge() {
    const checked = $$('input[name="ctrl-col"]:checked').length;
    const total = $$('input[name="ctrl-col"]').length;
    $("#controllable-count-badge").textContent = `${checked}/${total}`;

    // Sync "Select All" state
    const selectAll = $("#ctrl-select-all");
    selectAll.checked = checked === total;
    selectAll.indeterminate = checked > 0 && checked < total;
}


// ═══════════════════════════════════════════════════════════════════
//  ADVANCED PANEL
// ═══════════════════════════════════════════════════════════════════

function initAdvancedPanel() {
    const toggle = $("#advanced-toggle");
    const panel = $("#advanced-panel");
    toggle.addEventListener("click", () => {
        const isOpen = !panel.classList.contains("collapsed");
        if (isOpen) {
            panel.classList.add("collapsed");
            toggle.classList.remove("open");
        } else {
            panel.classList.remove("collapsed");
            toggle.classList.add("open");
        }
    });
}

function initSliders() {
    bindSlider("n-samples-slider", "n-samples-value", (v) => fmtInt(v));
    bindSlider("topk-slider", "topk-value", (v) => v);
    bindSlider("confidence-slider", "confidence-value", (v) => Number(v).toFixed(2));
    bindSlider("risk-weight-slider", "risk-weight-value", (v) => Number(v).toFixed(1));

    // Set defaults
    $("#n-samples-slider").value = CONFIG.defaults.n_samples;
    $("#n-samples-value").textContent = fmtInt(CONFIG.defaults.n_samples);
    $("#topk-slider").value = CONFIG.defaults.topk;
    $("#topk-value").textContent = CONFIG.defaults.topk;
    $("#confidence-slider").value = CONFIG.defaults.confidence;
    $("#confidence-value").textContent = Number(CONFIG.defaults.confidence).toFixed(2);
    $("#risk-weight-slider").value = CONFIG.defaults.risk_weight;
    $("#risk-weight-value").textContent = Number(CONFIG.defaults.risk_weight).toFixed(1);
}

function bindSlider(sliderId, valueId, formatter) {
    const slider = $(`#${sliderId}`);
    const display = $(`#${valueId}`);
    slider.addEventListener("input", () => {
        display.textContent = formatter(slider.value);
    });
}


// ═══════════════════════════════════════════════════════════════════
//  TAB NAVIGATION
// ═══════════════════════════════════════════════════════════════════

function bindTabNavigation() {
    $$(".tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            $$(".tab-btn").forEach((b) => b.classList.remove("active"));
            $$(".tab-panel").forEach((p) => p.classList.add("hidden"));
            btn.classList.add("active");
            const panel = $(`#${btn.dataset.tab}`);
            panel.classList.remove("hidden");
            // Re-animate
            panel.style.animation = "none";
            panel.offsetHeight; // trigger reflow
            panel.style.animation = "";
        });
    });
}


// ═══════════════════════════════════════════════════════════════════
//  RUN BUTTON
// ═══════════════════════════════════════════════════════════════════

function bindRunButton() {
    const btn = $("#run-btn");
    btn.addEventListener("click", runInverseDesign);
}

async function runInverseDesign() {
    const btn = $("#run-btn");

    // Gather parameters
    const pollutantCol = $("#pollutant-col-select").value;
    const pollutantValues = Array.from($$('input[name="pollutant-val"]:checked')).map((cb) => cb.value);
    const controllableCols = Array.from($$('input[name="ctrl-col"]:checked')).map((cb) => cb.value);

    // Validate
    if (pollutantValues.length === 0) {
        showToast("Select at least one pollutant value", "error");
        return;
    }
    if (TARGET_RATES.length === 0) {
        showToast("Add at least one target removal rate", "error");
        return;
    }
    if (controllableCols.length === 0) {
        showToast("Select at least one controllable column", "error");
        return;
    }

    const payload = {
        pollutant_col: pollutantCol,
        pollutant_values: pollutantValues,
        target_rates: TARGET_RATES,
        controllable_cols: controllableCols,
        n_samples: parseInt($("#n-samples-slider").value),
        topk: parseInt($("#topk-slider").value),
        confidence: parseFloat($("#confidence-slider").value),
        risk_weight: parseFloat($("#risk-weight-slider").value),
    };

    // Show loading
    btn.disabled = true;
    btn.classList.add("loading");
    btn.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 18 18" fill="currentColor"><circle cx="9" cy="9" r="7" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="14 28"/></svg>
        Running…
    `;
    show($("#loading-overlay"));

    try {
        const resp = await fetch("/api/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        const data = await resp.json();

        if (!resp.ok) {
            throw new Error(data.error || `Server error ${resp.status}`);
        }

        LAST_RESULTS = data;
        renderResults(data);
        showToast("Inverse design completed successfully!", "success");

    } catch (err) {
        showToast(`Error: ${err.message}`, "error", 8000);
        console.error(err);
    } finally {
        btn.disabled = false;
        btn.classList.remove("loading");
        btn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 18 18" fill="currentColor"><path d="M4 3l11 6-11 6V3z"/></svg>
            Run Inverse Design
        `;
        hide($("#loading-overlay"));
    }
}


// ═══════════════════════════════════════════════════════════════════
//  RENDER RESULTS
// ═══════════════════════════════════════════════════════════════════

function renderResults(data) {
    hide($("#welcome-state"));
    show($("#results-state"));

    renderBestCards(data.summary);
    renderFullTable(data.recommendations);
    renderRangesChart(data.ranges);
    renderScatterChart(data.recommendations);
}


// ── Best Cards ─────────────────────────────────────────────────────

function renderBestCards(summary) {
    const grid = $("#best-cards-grid");
    grid.innerHTML = "";

    const recs = summary.best_recommendations || [];
    if (recs.length === 0) {
        grid.innerHTML = `<p class="panel-desc">No recommendations generated.</p>`;
        return;
    }

    recs.forEach((rec, i) => {
        const card = document.createElement("div");
        card.className = "rec-card";
        card.style.animationDelay = `${i * 80}ms`;

        const predicted = rec.predicted_removal_rate;
        const intPart = Math.floor(predicted);
        const decPart = (predicted - intPart).toFixed(2).substring(1);

        card.innerHTML = `
            <div class="card-header">
                <span class="card-pollutant">${escapeHtml(rec.pollutant_input)}</span>
                <span class="card-target">Target ${fmt(rec.target_removal_rate, 1)}%</span>
            </div>
            <div class="card-metric">
                <span class="metric-big">${intPart}${decPart}</span>
                <span class="metric-unit">% predicted</span>
            </div>
            <div class="metric-label">Predicted Removal Rate</div>
            <div class="card-stats">
                <div class="card-stat">
                    <span class="card-stat-label">Abs Error</span>
                    <span class="card-stat-value">${fmt(rec.abs_error_to_target, 4)}</span>
                </div>
                <div class="card-stat">
                    <span class="card-stat-label">Plausibility</span>
                    <span class="card-stat-value">${fmt(rec.plausibility_score, 4)}</span>
                </div>
                ${rec.low_risk_score !== null ? `
                <div class="card-stat">
                    <span class="card-stat-label">Low Risk</span>
                    <span class="card-stat-value">${fmt(rec.low_risk_score, 4)}</span>
                </div>` : ""}
                ${rec.composite_score !== null ? `
                <div class="card-stat">
                    <span class="card-stat-label">Composite</span>
                    <span class="card-stat-value">${fmt(rec.composite_score, 4)}</span>
                </div>` : ""}
            </div>
        `;
        grid.appendChild(card);
    });
}


// ── Full Table ─────────────────────────────────────────────────────

function renderFullTable(rows) {
    if (!rows || rows.length === 0) {
        $("#results-thead").innerHTML = "";
        $("#results-tbody").innerHTML = `<tr><td colspan="99" style="text-align:center;padding:24px;color:var(--text-muted)">No candidates</td></tr>`;
        return;
    }

    const columns = Object.keys(rows[0]);
    sortCol = null;
    sortAsc = true;

    // Render header
    const thead = $("#results-thead");
    thead.innerHTML = "<tr>" + columns.map((col) =>
        `<th data-col="${escapeHtml(col)}">${escapeHtml(col)}</th>`
    ).join("") + "</tr>";

    // Bind sorting
    thead.querySelectorAll("th").forEach((th) => {
        th.addEventListener("click", () => {
            const col = th.dataset.col;
            if (sortCol === col) {
                sortAsc = !sortAsc;
            } else {
                sortCol = col;
                sortAsc = true;
            }
            // Update sort indicators
            thead.querySelectorAll("th").forEach((t) => t.classList.remove("sort-asc", "sort-desc"));
            th.classList.add(sortAsc ? "sort-asc" : "sort-desc");
            renderTableBody(rows, columns);
        });
    });

    renderTableBody(rows, columns);

    // Filter input
    const filterInput = $("#table-filter");
    filterInput.value = "";
    filterInput.addEventListener("input", () => {
        const query = filterInput.value.toLowerCase().trim();
        const filtered = query
            ? rows.filter((r) => Object.values(r).some((v) => String(v).toLowerCase().includes(query)))
            : rows;
        renderTableBody(filtered, columns);
    });
}

function renderTableBody(rows, columns) {
    let sorted = [...rows];
    if (sortCol) {
        sorted.sort((a, b) => {
            let va = a[sortCol], vb = b[sortCol];
            const na = Number(va), nb = Number(vb);
            if (!isNaN(na) && !isNaN(nb)) {
                return sortAsc ? na - nb : nb - na;
            }
            va = String(va || "");
            vb = String(vb || "");
            return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
        });
    }

    const tbody = $("#results-tbody");
    tbody.innerHTML = sorted.map((row) =>
        "<tr>" + columns.map((col) => {
            const val = row[col];
            const display = typeof val === "number" ? fmt(val, 4) : (val ?? "—");
            return `<td>${escapeHtml(String(display))}</td>`;
        }).join("") + "</tr>"
    ).join("");
}


// ── Ranges Chart ───────────────────────────────────────────────────

function renderRangesChart(ranges) {
    if (!ranges || ranges.length === 0) return;

    // Populate filter dropdowns
    const pollutants = [...new Set(ranges.map((r) => r.pollutant_input))];
    const targets = [...new Set(ranges.map((r) => r.target_removal_rate))].sort((a, b) => a - b);

    const polSel = $("#range-pollutant-filter");
    polSel.innerHTML = pollutants.map((p) => `<option value="${escapeHtml(p)}">${escapeHtml(p)}</option>`).join("");

    const tarSel = $("#range-target-filter");
    tarSel.innerHTML = targets.map((t) => `<option value="${t}">${t}%</option>`).join("");

    const update = () => {
        const pol = polSel.value;
        const tar = parseFloat(tarSel.value);
        const filtered = ranges.filter((r) => r.pollutant_input === pol && r.target_removal_rate === tar);
        drawRangesChart(filtered);
    };

    polSel.addEventListener("change", update);
    tarSel.addEventListener("change", update);
    update();
}

function drawRangesChart(data) {
    const ctx = $("#ranges-chart");

    if (rangesChart) {
        rangesChart.destroy();
        rangesChart = null;
    }

    if (data.length === 0) return;

    const labels = data.map((d) => d.parameter);
    const lows = data.map((d) => d.value_low);
    const medians = data.map((d) => d.value_median);
    const highs = data.map((d) => d.value_high);

    rangesChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Low",
                    data: lows,
                    backgroundColor: "rgba(99, 102, 241, 0.35)",
                    borderColor: "rgba(99, 102, 241, 0.8)",
                    borderWidth: 1,
                    borderRadius: 4,
                },
                {
                    label: "Median",
                    data: medians,
                    backgroundColor: "rgba(20, 184, 166, 0.55)",
                    borderColor: "rgba(20, 184, 166, 0.9)",
                    borderWidth: 1,
                    borderRadius: 4,
                },
                {
                    label: "High",
                    data: highs,
                    backgroundColor: "rgba(245, 158, 11, 0.35)",
                    borderColor: "rgba(245, 158, 11, 0.8)",
                    borderWidth: 1,
                    borderRadius: 4,
                },
            ],
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: "#94a3b8",
                        font: { family: "Inter", size: 12 },
                    },
                },
                tooltip: {
                    backgroundColor: "rgba(15, 23, 42, 0.92)",
                    titleFont: { family: "Inter", size: 13 },
                    bodyFont: { family: "JetBrains Mono", size: 12 },
                    borderColor: "rgba(148, 163, 184, 0.2)",
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                },
            },
            scales: {
                x: {
                    grid: { color: "rgba(148, 163, 184, 0.08)" },
                    ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 11 } },
                },
                y: {
                    grid: { display: false },
                    ticks: { color: "#94a3b8", font: { family: "Inter", size: 12 } },
                },
            },
        },
    });
}


// ── Scatter Chart ──────────────────────────────────────────────────

function renderScatterChart(rows) {
    const ctx = $("#scatter-chart");

    if (scatterChart) {
        scatterChart.destroy();
        scatterChart = null;
    }

    if (!rows || rows.length === 0) return;

    // Group by pollutant
    const groups = {};
    rows.forEach((r) => {
        const pol = r.pollutant_input || "Unknown";
        if (!groups[pol]) groups[pol] = [];
        groups[pol].push(r);
    });

    const palette = [
        "rgba(20, 184, 166, 0.8)",
        "rgba(99, 102, 241, 0.8)",
        "rgba(245, 158, 11, 0.8)",
        "rgba(244, 63, 94, 0.8)",
        "rgba(56, 189, 248, 0.8)",
        "rgba(52, 211, 153, 0.8)",
        "rgba(168, 85, 247, 0.8)",
        "rgba(251, 146, 60, 0.8)",
    ];

    const datasets = Object.entries(groups).map(([pol, items], i) => ({
        label: pol,
        data: items.map((r) => ({
            x: r.predicted_removal_rate,
            y: r.plausibility_score,
            rank: r.rank,
            target: r.target_removal_rate,
            error: r.abs_error_to_target,
        })),
        backgroundColor: palette[i % palette.length],
        borderColor: palette[i % palette.length].replace("0.8", "1"),
        borderWidth: 1,
        pointRadius: items.map((r) => Math.max(4, 14 - (r.rank || 1))),
        pointHoverRadius: 10,
    }));

    scatterChart = new Chart(ctx, {
        type: "scatter",
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: "#94a3b8",
                        font: { family: "Inter", size: 12 },
                        usePointStyle: true,
                        pointStyle: "circle",
                    },
                },
                tooltip: {
                    backgroundColor: "rgba(15, 23, 42, 0.92)",
                    titleFont: { family: "Inter", size: 13 },
                    bodyFont: { family: "JetBrains Mono", size: 12 },
                    borderColor: "rgba(148, 163, 184, 0.2)",
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        title: (items) => {
                            const d = items[0].raw;
                            return `Target: ${fmt(d.target, 1)}%  |  Rank #${d.rank}`;
                        },
                        label: (item) => {
                            const d = item.raw;
                            return [
                                `Predicted: ${fmt(d.x, 3)}%`,
                                `Plausibility: ${fmt(d.y, 4)}`,
                                `Error: ${fmt(d.error, 4)}`,
                            ];
                        },
                    },
                },
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: "Predicted Removal Rate (%)",
                        color: "#94a3b8",
                        font: { family: "Inter", size: 13 },
                    },
                    grid: { color: "rgba(148, 163, 184, 0.08)" },
                    ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 11 } },
                },
                y: {
                    title: {
                        display: true,
                        text: "Plausibility Score",
                        color: "#94a3b8",
                        font: { family: "Inter", size: 13 },
                    },
                    grid: { color: "rgba(148, 163, 184, 0.08)" },
                    ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 11 } },
                },
            },
        },
    });
}


// ═══════════════════════════════════════════════════════════════════
//  CSV EXPORT
// ═══════════════════════════════════════════════════════════════════

function bindExportCSV() {
    $("#export-csv-btn").addEventListener("click", async () => {
        if (!LAST_RESULTS) {
            showToast("Run an inverse design first", "error");
            return;
        }

        try {
            const resp = await fetch("/api/export/csv", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ type: "recommendations" }),
            });

            if (!resp.ok) {
                const data = await resp.json();
                throw new Error(data.error || "Export failed");
            }

            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "inverse_design_recommendations.csv";
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
            showToast("CSV exported!", "success");
        } catch (err) {
            showToast(`Export error: ${err.message}`, "error");
        }
    });
}


// ═══════════════════════════════════════════════════════════════════
//  UTILS
// ═══════════════════════════════════════════════════════════════════

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}
