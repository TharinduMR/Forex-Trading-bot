/**
 * ANTIGRAVITY QUANT // INSTITUTIONAL MULTI-SYMBOL DASHBOARD CONTROLLER
 * Real-time polling, state synchronization, dynamic DOM manipulation, and custom HTML5 Canvas charting.
 * Supports multiple symbols (XAUUSD, EURUSD) with per-symbol data fetching and formatting.
 */

let chartCandles = [];
let chartDataLimit = 120;
let isFetchingHistory = false;
let chartOverlays = {};
let engineState = {};
let selectedTimeframe = "15m";
let activeSymbol = "XAUUSD";  // Currently selected symbol in the dashboard
let symbolProfiles = {};       // Loaded from /api/symbols on init
let currentConfig = {};        // Per-symbol config from status API

// Chart Interactivity State
let chartZoomX = 1.0;
let chartZoomY = 1.0;
let chartPanX = 0;
let chartPanY = 0;
let isDraggingChart = false;
let lastMouseX = 0;
let lastMouseY = 0;

document.addEventListener("DOMContentLoaded", () => {
    initDashboard();
    initChartInteractivity();
    // Poll backend API every 500 milliseconds (0.5 second)
    setInterval(updateDashboard, 500);

    // Bind Fullscreen Button
    const fsBtn = document.getElementById("fullscreen-btn");
    const chartPanel = document.querySelector(".chart-panel");
    if (fsBtn && chartPanel) {
        fsBtn.addEventListener("click", () => {
            if (!document.fullscreenElement) {
                if (chartPanel.requestFullscreen) {
                    chartPanel.requestFullscreen();
                } else if (chartPanel.webkitRequestFullscreen) { /* Safari */
                    chartPanel.webkitRequestFullscreen();
                } else if (chartPanel.msRequestFullscreen) { /* IE11 */
                    chartPanel.msRequestFullscreen();
                }
            } else {
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                } else if (document.webkitExitFullscreen) { /* Safari */
                    document.webkitExitFullscreen();
                } else if (document.msExitFullscreen) { /* IE11 */
                    document.msExitFullscreen();
                }
            }
        });
    }

    // Auto-resize chart on container dimension changes (e.g. going fullscreen)
    const chartContainer = document.querySelector(".chart-container");
    if (chartContainer) {
        let lastWidth = 0, lastHeight = 0;
        const resizeObserver = new ResizeObserver((entries) => {
            for (let entry of entries) {
                const { width, height } = entry.contentRect;
                if (width !== lastWidth || height !== lastHeight) {
                    lastWidth = width;
                    lastHeight = height;
                    if (chartCandles && chartCandles.length > 0) {
                        renderCanvasChart();
                    }
                }
            }
        });
        resizeObserver.observe(chartContainer);
    }

    // Bind Timeframe Selector Buttons
    document.querySelectorAll(".tf-btn").forEach(btn => {
        btn.addEventListener("click", (e) => {
            document.querySelectorAll(".tf-btn").forEach(b => b.classList.remove("active"));
            e.target.classList.add("active");
            selectedTimeframe = e.target.dataset.tf;
            chartDataLimit = 120;
            chartPanX = 0;
            chartZoomX = 1.0;
            const subtitleEl = document.getElementById("chart-subtitle");
            if (subtitleEl) {
                const labelMap = {
                    "1m": "1M INTRADAY CANDLES",
                    "5m": "5M INTRADAY CANDLES",
                    "15m": "15M INTRADAY CANDLES",
                    "30m": "30M INTRADAY CANDLES",
                    "1h": "1H HOURLY CANDLES",
                    "2h": "2H HOURLY CANDLES",
                    "4h": "4H HOURLY CANDLES",
                    "1d": "DAILY CANDLES (1D)",
                    "2d": "2-DAY CANDLES (2D)",
                    "5d": "5-DAY CANDLES (5D)",
                    "7d": "WEEKLY CANDLES (7D)"
                };
                const tfLabel = labelMap[selectedTimeframe] || selectedTimeframe.toUpperCase() + " CANDLES";
                subtitleEl.innerText = `${tfLabel} // REAL-TIME TRIPLE-BARRIER LEVELS`;
            }
            updateDashboard();
        });
    });

    // Bind Symbol Switcher Buttons
    document.querySelectorAll(".sym-btn").forEach(btn => {
        btn.addEventListener("click", (e) => {
            document.querySelectorAll(".sym-btn").forEach(b => b.classList.remove("active"));
            e.target.classList.add("active");
            activeSymbol = e.target.dataset.symbol;
            chartDataLimit = 120;
            chartPanX = 0;
            chartZoomX = 1.0;
            updateSymbolUI();
            updateDashboard();
        });
    });

    // Bind One-Click Trade Buttons
    const btnBuy = document.getElementById("oc-buy-btn");
    const btnSell = document.getElementById("oc-sell-btn");
    const lotInput = document.getElementById("oc-lot");

    if (btnBuy && btnSell && lotInput) {
        const executeOneClick = async (action) => {
            const vol = parseFloat(lotInput.value);
            if (isNaN(vol) || vol <= 0) return alert("Invalid lot size");
            if (!confirm(`Are you sure you want to execute a ${action} order of ${vol} lots on ${activeSymbol}?`)) return;

            try {
                const res = await fetch("/api/trade", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ symbol: activeSymbol, action: action, volume: vol })
                });
                const data = await res.json();
                if (data.success) {
                    updateDashboard(); // Refresh positions immediately
                } else {
                    alert("Order failed: " + data.error);
                }
            } catch (err) {
                alert("Order request failed: " + err);
            }
        };

        btnBuy.addEventListener("click", () => executeOneClick("BUY"));
        btnSell.addEventListener("click", () => executeOneClick("SELL"));
    }
});

function updateSymbolUI() {
    // Update all dynamic titles to reflect the active symbol
    const displayName = (symbolProfiles[activeSymbol] && symbolProfiles[activeSymbol].display_name) || activeSymbol;

    const feedTitle = document.getElementById("live-feed-title");
    if (feedTitle) feedTitle.innerText = `LIVE ${activeSymbol} FEED`;

    const chartTitle = document.getElementById("chart-title");
    if (chartTitle) chartTitle.innerText = `LIVE ${activeSymbol} PRICE & EXECUTION CHART`;
}

let dragMode = "pan";

function initChartInteractivity() {
    const canvas = document.getElementById("priceChart");
    if (!canvas) return;

    canvas.addEventListener("wheel", (e) => {
        e.preventDefault(); // Prevent page scrolling
        const zoomSpeed = 0.05;
        const delta = e.deltaY > 0 ? -1 : 1;

        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // If hovering over axes, zoom only that axis
        if (y > rect.height - 30) {
            chartZoomX += delta * zoomSpeed * chartZoomX;
            chartZoomX = Math.max(0.2, Math.min(chartZoomX, 10.0));
        } else if (x > rect.width - 60) {
            chartZoomY += delta * zoomSpeed * chartZoomY;
            chartZoomY = Math.max(0.1, Math.min(chartZoomY, 10.0));
        } else {
            // General zooming
            if (e.shiftKey) {
                chartZoomX += delta * zoomSpeed * chartZoomX;
                chartZoomX = Math.max(0.2, Math.min(chartZoomX, 10.0));
            } else {
                chartZoomY += delta * zoomSpeed * chartZoomY;
                chartZoomY = Math.max(0.1, Math.min(chartZoomY, 10.0));
            }
        }
        renderCanvasChart();
    });

    canvas.addEventListener("mousedown", (e) => {
        isDraggingChart = true;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;

        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Time Axis Drag -> Horizontal Zoom
        if (y > rect.height - 30) {
            dragMode = "zoomX";
            canvas.style.cursor = "ew-resize";
        }
        // Price Axis Drag -> Vertical Zoom
        else if (x > rect.width - 60) {
            dragMode = "zoomY";
            canvas.style.cursor = "ns-resize";
        } else {
            dragMode = "pan";
            canvas.style.cursor = "grabbing";
        }
    });

    canvas.addEventListener("mousemove", (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Update cursor style on hover if not dragging
        if (!isDraggingChart) {
            if (y > rect.height - 30) {
                canvas.style.cursor = "ew-resize";
            } else if (x > rect.width - 60) {
                canvas.style.cursor = "ns-resize";
            } else {
                canvas.style.cursor = "crosshair";
            }
            return;
        }

        const dx = e.clientX - lastMouseX;
        const dy = e.clientY - lastMouseY;


        if (dragMode === "pan") {
            chartPanX += dx;
            chartPanY += dy;
        } else if (dragMode === "zoomX") {
            chartZoomX += dx * 0.01;
            chartZoomX = Math.max(0.1, Math.min(chartZoomX, 10.0));
        } else if (dragMode === "zoomY") {
            chartZoomY += dy * 0.01;
            chartZoomY = Math.max(0.1, Math.min(chartZoomY, 10.0));
        }

        // Lazy load past data if panned right or zoomed out
        if ((chartPanX > 50 || chartZoomX < 0.5) && !isFetchingHistory && chartDataLimit < 2000) {
            isFetchingHistory = true;
            chartDataLimit += 100;
            // Adjust pan slightly to keep the view smooth while loading
            if (chartPanX > 50) chartPanX -= 20;
            updateDashboard().finally(() => { isFetchingHistory = false; });
        }


        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
        renderCanvasChart();
    });

    const stopDragging = () => {
        isDraggingChart = false;
        canvas.style.cursor = "crosshair";
    };
    canvas.addEventListener("mouseup", stopDragging);
    canvas.addEventListener("mouseleave", stopDragging);

    canvas.addEventListener("dblclick", () => {
        chartZoomX = 1.0;
        chartZoomY = 1.0;
        chartPanX = 0;
        chartPanY = 0;
        renderCanvasChart();
    });
}

async function initDashboard() {
    // Load symbol profiles from backend
    try {
        const res = await fetch("/api/symbols");
        if (res.ok) {
            symbolProfiles = await res.json();
        }
    } catch (e) {
        console.warn("Failed to load symbol profiles:", e);
    }
    await updateDashboard();
}

async function updateDashboard() {
    try {
        const [statusRes, posRes, chartRes, logsRes, histRes] = await Promise.all([
            fetch(`/api/status?symbol=${activeSymbol}&timeframe=${selectedTimeframe}`),
            fetch(`/api/positions`),
            fetch(`/api/chart_data?tf=${selectedTimeframe}&symbol=${activeSymbol}&limit=${chartDataLimit}`),
            fetch("/api/logs"),
            fetch("/api/trade_history")
        ]);

        let statusData = null;
        if (statusRes.ok) {
            statusData = await statusRes.json();
            currentConfig = statusData.config || {};
            renderStatus(statusData);
            engineState = statusData.engine || {};
        }

        if (posRes.ok) {
            const posData = await posRes.json();
            renderPositions(posData);
        }

        if (chartRes.ok) {
            const chartData = await chartRes.json();
            chartCandles = chartData;
            renderCanvasChart();
        }

        if (logsRes.ok) {
            const logsData = await logsRes.json();
            renderLogs(logsData.logs || []);
        }

        if (histRes.ok) {
            const histData = await histRes.json();
            renderHistory(histData, statusData ? statusData.account.realized_pnl : 0.0);
        }

        // Fetch chart overlays (S/R zones, positions, SMC/ICT setups)
        try {
            const overlayRes = await fetch(`/api/chart_overlays?symbol=${activeSymbol}&tf=${selectedTimeframe}`);
            if (overlayRes.ok) {
                chartOverlays = await overlayRes.json();
            }
        } catch (e) {
            console.warn("Failed to fetch chart overlays:", e);
        }
    } catch (err) {
        console.error("Dashboard synchronization error:", err);
        document.getElementById("mt5-status-text").innerText = "API DISCONNECTED";
        document.querySelector(".pulse-dot").className = "pulse-dot disconnected";
    }
}

/**
 * Format a price value according to the current symbol's profile.
 * Gold: $2,650.30 (2 decimals with $ prefix)
 * EURUSD: 1.08450 (5 decimals, no $ prefix)
 */
function formatPrice(val) {
    if (val === undefined || val === null || isNaN(val)) return "--";
    const decimals = currentConfig.price_decimals || 2;
    const format = currentConfig.price_format || "dollar";

    if (format === "dollar") {
        return "$" + val.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
    } else {
        return val.toFixed(decimals);
    }
}

function renderStatus(data) {
    const { account, quote, engine, config } = data;

    // 1. Navbar Status
    const badgeDot = document.querySelector(".pulse-dot");
    const badgeText = document.getElementById("mt5-status-text");
    if (account.connected) {
        badgeDot.className = "pulse-dot connected";
        badgeText.innerText = "MT5 LIVE CONNECTED";
    } else {
        badgeDot.className = "pulse-dot disconnected";
        badgeText.innerText = "MT5 OFFLINE";
    }

    const switchVal = document.getElementById("live-switch-val");
    if (engine.live_trading_enabled) {
        switchVal.innerText = "LIVE TRADING ENABLED";
        switchVal.className = "tag-value live";
    } else {
        switchVal.innerText = "ADVISORY / DRY-RUN";
        switchVal.className = "tag-value advisory";
    }

    document.getElementById("server-clock").innerText = account.server_time || "--:--:--";

    // 2. Card 1: Equity & Balance & PnL
    document.getElementById("metric-equity").innerText = formatCurr(account.equity);
    document.getElementById("metric-balance").innerText = formatCurr(account.balance);
    document.getElementById("account-login-note").innerText = `Login: ${account.login || "--"} | Server: MetaTrader 5`;

    const floatingPnl = account.equity - account.balance;
    const pnlBadge = document.getElementById("floating-pnl-badge");
    pnlBadge.innerText = (floatingPnl >= 0 ? "+" : "") + formatCurr(floatingPnl) + " Open";
    pnlBadge.className = "card-badge " + (floatingPnl >= 0 ? "positive" : "negative");

    const realizedPnl = account.realized_pnl || 0.0;
    const realizedEl = document.getElementById("metric-realized-pnl");
    if (realizedEl) {
        realizedEl.innerText = (realizedPnl >= 0 ? "+" : "") + formatCurr(realizedPnl);
        realizedEl.style.color = realizedPnl >= 0 ? "var(--accent-emerald)" : "var(--accent-crimson)";
    }

    const totalPnl = account.total_pnl || (realizedPnl + floatingPnl);
    const totalEl = document.getElementById("metric-total-pnl");
    if (totalEl) {
        totalEl.innerText = (totalPnl >= 0 ? "+" : "") + formatCurr(totalPnl);
        totalEl.style.color = totalPnl >= 0 ? "var(--accent-emerald)" : "var(--accent-crimson)";
    }

    // 3. Card 2: Margin & Risk
    document.getElementById("metric-free-margin").innerText = formatCurr(account.free_margin);
    const mLevel = account.margin_level > 9000 ? "999.9%+" : account.margin_level.toFixed(1) + "%";
    document.getElementById("metric-margin-level").innerText = mLevel;

    const mBar = document.getElementById("margin-level-bar");
    const mPct = Math.min(100, Math.max(0, (account.margin_level / 500) * 100));
    mBar.style.width = account.margin_level > 9000 ? "100%" : `${mPct}%`;

    // 4. Card 3: Live Market Quote (per-symbol formatting)
    document.getElementById("metric-bid").innerText = quote.bid ? formatPrice(quote.bid) : "--";
    document.getElementById("metric-ask").innerText = quote.ask ? formatPrice(quote.ask) : "--";

    const spreadBadge = document.getElementById("metric-spread-badge");
    spreadBadge.innerText = `SPREAD: ${quote.spread} PTS`;
    if (quote.spread > config.max_spread_points) {
        spreadBadge.className = "card-badge spread-badge danger";
        document.getElementById("spread-warn-note").innerText = `⚠️ SPREAD EXCEEDS LIMIT (${config.max_spread_points} PTS)! TRADES SUPPRESSED.`;
        document.getElementById("spread-warn-note").style.color = "var(--accent-crimson)";
    } else {
        spreadBadge.className = "card-badge spread-badge safe";
        document.getElementById("spread-warn-note").innerText = `Spread Safety Gate: ≤ ${config.max_spread_points} Pts (OK)`;
        document.getElementById("spread-warn-note").style.color = "var(--text-muted)";
    }

    // 5. Card 4 & AI Evaluation Panel
    const metaProb = (engine.meta_prob !== undefined ? engine.meta_prob * 100 : 100.0).toFixed(1);
    document.getElementById("metric-meta-prob").innerText = `${metaProb}%`;
    document.getElementById("meta-prob-bar").style.width = `${metaProb}%`;

    const metaBadge = document.getElementById("meta-decision-badge");
    if (metaProb >= config.meta_threshold) {
        metaBadge.innerText = "GATE APPROVED";
        metaBadge.className = "card-badge ai-badge approved";
    } else {
        metaBadge.innerText = "GATE REJECTED";
        metaBadge.className = "card-badge ai-badge rejected";
    }

    // AI Signal Evaluation Display
    const sigCard = document.getElementById("ai-signal-card");
    const sigText = document.getElementById("ai-signal-text");
    const sigCode = engine.signal_code || 0;

    sigText.innerText = engine.signal || "NO TRADE (0)";
    if (sigCode === 1.0) {
        sigCard.className = "ai-signal-display buy-signal";
        sigText.style.color = "var(--accent-emerald)";
    } else if (sigCode === -1.0) {
        sigCard.className = "ai-signal-display sell-signal";
        sigText.style.color = "var(--accent-crimson)";
    } else {
        sigCard.className = "ai-signal-display flat-signal";
        sigText.style.color = "var(--accent-amber)";
    }

    const confVal = (engine.confidence || 0.0).toFixed(1);
    document.getElementById("ai-primary-conf").innerText = `${confVal}%`;

    // Probability Bars
    const pLong = (engine.prob_long || 0.0).toFixed(1);
    const pFlat = (engine.prob_flat !== undefined ? engine.prob_flat : 100.0).toFixed(1);
    const pShort = (engine.prob_short || 0.0).toFixed(1);

    document.getElementById("prob-val-long").innerText = `${pLong}%`;
    document.getElementById("prob-bar-long").style.width = `${pLong}%`;

    document.getElementById("prob-val-flat").innerText = `${pFlat}%`;
    document.getElementById("prob-bar-flat").style.width = `${pFlat}%`;

    document.getElementById("prob-val-short").innerText = `${pShort}%`;
    document.getElementById("prob-bar-short").style.width = `${pShort}%`;

    // ATR uses per-symbol formatting
    document.getElementById("ai-atr-val").innerText = formatPrice(engine.atr || 0.0);
    document.getElementById("ai-last-action").innerText = engine.action || "Idle";
    document.getElementById("ai-eval-time").innerText = engine.timestamp || "--";
}

function renderPositions(positions) {
    const tbody = document.getElementById("positions-tbody");
    const countBadge = document.getElementById("position-count-badge");

    countBadge.innerText = `${positions.length} POSITIONS`;

    if (!positions || positions.length === 0) {
        tbody.innerHTML = `<tr class="empty-row"><td colspan="10">No active open positions in MT5 account.</td></tr>`;
        return;
    }

    let html = "";
    positions.forEach(p => {
        const isBuy = p.type.includes("BUY");
        const typeClass = isBuy ? "pos-buy" : "pos-sell";
        const profitClass = p.profit >= 0 ? "profit-pos" : "profit-neg";
        const profitStr = (p.profit >= 0 ? "+" : "") + "$" + p.profit.toFixed(2);

        // Determine price formatting for this position's symbol
        const posProfile = symbolProfiles[p.symbol] || {};
        const posDecimals = posProfile.price_decimals || 2;
        const posFormat = posProfile.price_format || "dollar";
        const fmtPosPrice = (v) => {
            if (posFormat === "dollar") return "$" + v.toFixed(posDecimals);
            return v.toFixed(posDecimals);
        };

        html += `
            <tr>
                <td>#${p.ticket}</td>
                <td>${p.time.split(" ")[1]}</td>
                <td class="${typeClass}">${p.type}</td>
                <td>${p.volume.toFixed(2)}</td>
                <td>${fmtPosPrice(p.price_open)}</td>
                <td>${fmtPosPrice(p.price_current)}</td>
                <td>${p.sl > 0 ? fmtPosPrice(p.sl) : "--"}</td>
                <td>${p.tp > 0 ? fmtPosPrice(p.tp) : "--"}</td>
                <td class="${profitClass}">${profitStr}</td>
                <td class="action-cell">
                    <button class="action-btn sltp-btn" onclick="promptSetSLTP(${p.ticket}, ${p.sl}, ${p.tp})">⚙️ SL/TP</button>
                    <button class="action-btn close-btn" onclick="confirmClosePosition(${p.ticket})">✖ CLOSE</button>
                </td>
            </tr>
        `;
    });
    tbody.innerHTML = html;
}

function renderHistory(history, realizedPnl) {
    const tbody = document.getElementById("history-tbody");
    const countBadge = document.getElementById("history-count-badge");
    const pnlBadge = document.getElementById("history-pnl-badge");

    if (countBadge) countBadge.innerText = `${history ? history.length : 0} DEALS`;
    if (pnlBadge) {
        pnlBadge.innerText = `REALIZED PNL: ${(realizedPnl >= 0 ? "+" : "")}$${(realizedPnl || 0.0).toFixed(2)}`;
        pnlBadge.className = "card-badge " + (realizedPnl >= 0 ? "positive" : "negative");
    }

    if (!history || history.length === 0) {
        if (tbody) tbody.innerHTML = `<tr class="empty-row"><td colspan="8">No closed trade deals found in MT5 history.</td></tr>`;
        return;
    }

    let html = "";
    history.forEach(d => {
        const isBuy = d.type.includes("BUY");
        const typeClass = isBuy ? "pos-buy" : "pos-sell";
        const profitClass = d.profit >= 0 ? "profit-pos" : "profit-neg";
        const profitStr = (d.profit >= 0 ? "+" : "") + "$" + d.profit.toFixed(2);

        // Per-symbol price formatting for history
        const histProfile = symbolProfiles[d.symbol] || {};
        const histDecimals = histProfile.price_decimals || 2;
        const histFormat = histProfile.price_format || "dollar";
        const fmtHistPrice = (v) => {
            if (histFormat === "dollar") return "$" + v.toFixed(histDecimals);
            return v.toFixed(histDecimals);
        };

        html += `
            <tr>
                <td>#${d.ticket}</td>
                <td>${d.time}</td>
                <td>${d.symbol}</td>
                <td class="${typeClass}">${d.type}</td>
                <td>${d.volume.toFixed(2)}</td>
                <td>${fmtHistPrice(d.price)}</td>
                <td class="${profitClass}" style="font-weight: 700;">${profitStr}</td>
                <td style="color: var(--text-muted);">${d.comment || "--"}</td>
            </tr>
        `;
    });
    if (tbody) tbody.innerHTML = html;
}

function renderLogs(logs) {
    const term = document.getElementById("terminal-window");
    if (!logs || logs.length === 0) return;

    let html = "";
    logs.forEach(line => {
        let lineClass = "term-line";
        if (line.includes("ORDER EXECUTED")) lineClass += " exec";
        else if (line.includes("Suppressed") || line.includes("Rejected") || line.includes("Failed")) lineClass += " warn";
        else if (line.includes("Candle Processed") || line.includes("STARTING")) lineClass += " highlight";

        html += `<div class="${lineClass}">${escapeHtml(line)}</div>`;
    });

    // Only update and scroll if content changed
    if (term.innerHTML !== html) {
        term.innerHTML = html;
        term.scrollTop = term.scrollHeight;
    }
}

function formatCurr(val) {
    if (val === undefined || val === null || isNaN(val)) return "$0.00";
    return "$" + val.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function escapeHtml(text) {
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
    return text.replace(/[&<>"']/g, m => map[m]);
}

/**
 * HIGH-PERFORMANCE HTML5 CANVAS CANDLESTICK & EXECUTION BARRIER RENDERER
 * Adapts price labels and formatting based on the active symbol profile.
 */
function renderCanvasChart() {
    const canvas = document.getElementById("priceChart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    // Handle high DPI displays and prevent infinite resize loops
    // by freezing the CSS size before modifying the drawing buffer size
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.display = "block";

    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const width = rect.width;
    const height = rect.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    if (!chartCandles || chartCandles.length === 0) {
        ctx.fillStyle = "#4a5a70";
        ctx.font = "14px JetBrains Mono";
        ctx.textAlign = "center";
        ctx.fillText("Waiting for MT5 candle rate stream...", width / 2, height / 2);
        return;
    }

    // Find min and max price
    let minPrice = Infinity;
    let maxPrice = -Infinity;
    chartCandles.forEach(c => {
        if (c.low < minPrice) minPrice = c.low;
        if (c.high > maxPrice) maxPrice = c.high;
    });

    // Include TP and SL in scale if active
    if (engineState.tp_price && engineState.tp_price > 0) {
        minPrice = Math.min(minPrice, engineState.tp_price);
        maxPrice = Math.max(maxPrice, engineState.tp_price);
    }
    if (engineState.sl_price && engineState.sl_price > 0) {
        minPrice = Math.min(minPrice, engineState.sl_price);
        maxPrice = Math.max(maxPrice, engineState.sl_price);
    }

    const padding = (maxPrice - minPrice) * 0.1 || 0.002;
    minPrice -= padding;
    maxPrice += padding;
    const basePriceRange = maxPrice - minPrice;

    // Zoom and Pan transformations for Y axis
    const visiblePriceRange = basePriceRange / chartZoomY;
    const pricePerPixel = visiblePriceRange / (height - 50);
    const priceOffset = chartPanY * pricePerPixel;

    const priceCenter = minPrice + (basePriceRange / 2);
    minPrice = priceCenter - (visiblePriceRange / 2) + priceOffset;
    maxPrice = priceCenter + (visiblePriceRange / 2) + priceOffset;
    const priceRange = visiblePriceRange;

    const baseCandleWidth = Math.max(4, (width - 60) / chartCandles.length);
    const candleWidth = baseCandleWidth * chartZoomX;
    const spacing = candleWidth * 0.2;

    const rightEdge = width - 60;
    const getX = (idx) => rightEdge - ((chartCandles.length - 1 - idx) * candleWidth) + chartPanX;
    const bodyWidth = candleWidth - spacing;

    // Price formatting for chart labels
    const decimals = currentConfig.price_decimals || 2;
    const isForex = (currentConfig.price_format === "decimal");
    const chartPriceLabel = (p) => {
        if (isForex) return p.toFixed(decimals);
        return "$" + p.toFixed(Math.min(decimals, 1));
    };

    // Helper: Map price to Y coordinate
    const priceToY = (price) => height - 30 - ((price - minPrice) / priceRange) * (height - 50);

    // 1. Draw Grid Lines & Price Labels
    ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#8b9bb4";
    ctx.font = "10px JetBrains Mono";
    ctx.textAlign = "right";

    const gridSteps = 5;
    for (let i = 0; i <= gridSteps; i++) {
        const gridPrice = minPrice + (priceRange / gridSteps) * i;
        const y = priceToY(gridPrice);

        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width - 50, y);
        ctx.stroke();

        ctx.fillText(chartPriceLabel(gridPrice), width - 5, y + 3);
    }

    // === CHART OVERLAYS (Behind Candles) ===
    const overlayOffset = 0; // Handled purely by indices now  // Server buffer offset

    // 1.5. Killzone Background Bands (very subtle, behind everything)
    if (chartOverlays.ict_setups) {
        chartOverlays.ict_setups.forEach(setup => {
            if (setup.type === "KILLZONE") {
                const startIdx = setup.start_bar - overlayOffset;
                const endIdx = setup.end_bar - overlayOffset;
                if (endIdx < 0 || startIdx >= chartCandles.length) return;
                const xStart = getX(startIdx);
                const xEnd = getX(endIdx) + candleWidth;
                const kzColor = setup.session === "London" ? "rgba(138, 43, 226, 0.06)" : "rgba(255, 165, 0, 0.06)";
                ctx.fillStyle = kzColor;
                ctx.fillRect(xStart, 20, xEnd - xStart, height - 50);
                // Label at top
                ctx.fillStyle = setup.session === "London" ? "rgba(138, 43, 226, 0.5)" : "rgba(255, 165, 0, 0.5)";
                ctx.font = "bold 9px JetBrains Mono";
                ctx.textAlign = "center";
                ctx.fillText(setup.session + " KZ", (xStart + xEnd) / 2, 14);
            }
        });
    }

    // 1.6. Support Zones (Green semi-transparent rectangles)
    if (chartOverlays.support_zones) {
        chartOverlays.support_zones.forEach(zone => {
            const yTop = priceToY(zone.price_high);
            const yBottom = priceToY(zone.price_low);
            const zoneHeight = yBottom - yTop;
            const alpha = Math.min(0.15, 0.05 * zone.strength);
            ctx.fillStyle = `rgba(0, 255, 136, ${alpha})`;
            ctx.fillRect(0, yTop, width - 50, zoneHeight);
            // Border
            ctx.strokeStyle = `rgba(0, 255, 136, ${alpha + 0.1})`;
            ctx.lineWidth = 0.5;
            ctx.strokeRect(0, yTop, width - 50, zoneHeight);
            // Label
            ctx.fillStyle = `rgba(0, 255, 136, 0.6)`;
            ctx.font = "bold 8px JetBrains Mono";
            ctx.textAlign = "left";
            ctx.fillText("S " + zone.label, 4, yTop + 10);
        });
    }

    // 1.7. Resistance Zones (Red semi-transparent rectangles)
    if (chartOverlays.resistance_zones) {
        chartOverlays.resistance_zones.forEach(zone => {
            const yTop = priceToY(zone.price_high);
            const yBottom = priceToY(zone.price_low);
            const zoneHeight = yBottom - yTop;
            const alpha = Math.min(0.15, 0.05 * zone.strength);
            ctx.fillStyle = `rgba(255, 51, 102, ${alpha})`;
            ctx.fillRect(0, yTop, width - 50, zoneHeight);
            ctx.strokeStyle = `rgba(255, 51, 102, ${alpha + 0.1})`;
            ctx.lineWidth = 0.5;
            ctx.strokeRect(0, yTop, width - 50, zoneHeight);
            ctx.fillStyle = `rgba(255, 51, 102, 0.6)`;
            ctx.font = "bold 8px JetBrains Mono";
            ctx.textAlign = "left";
            ctx.fillText("R " + zone.label, 4, yTop + 10);
        });
    }

    // 1.8. FVG Zones (Fair Value Gaps)
    if (chartOverlays.smc_setups) {
        chartOverlays.smc_setups.forEach(setup => {
            if (setup.type === "FVG") {
                const chartIdx = setup.bar_index - overlayOffset;
                if (chartIdx < 0 || chartIdx >= chartCandles.length) return;
                const x = getX(chartIdx);
                const yTop = priceToY(setup.price_top);
                const yBottom = priceToY(setup.price_bottom);
                const fvgColor = setup.direction === "bullish" ? "rgba(0, 200, 255, 0.12)" : "rgba(255, 140, 0, 0.12)";
                const borderColor = setup.direction === "bullish" ? "rgba(0, 200, 255, 0.3)" : "rgba(255, 140, 0, 0.3)";
                // Extend FVG to the right edge of chart
                ctx.fillStyle = fvgColor;
                ctx.fillRect(x, yTop, width - 50 - x, yBottom - yTop);
                ctx.strokeStyle = borderColor;
                ctx.lineWidth = 0.5;
                ctx.setLineDash([3, 3]);
                ctx.strokeRect(x, yTop, width - 50 - x, yBottom - yTop);
                ctx.setLineDash([]);
                // Label
                ctx.fillStyle = borderColor;
                ctx.font = "bold 8px JetBrains Mono";
                ctx.textAlign = "left";
                ctx.fillText("FVG", x + 2, yTop - 2);
            }
        });
    }

    // 1.9. OTE Zone (Optimal Trade Entry - Fibonacci 62%-79%)
    if (chartOverlays.ict_setups) {
        chartOverlays.ict_setups.forEach(setup => {
            if (setup.type === "OTE_ZONE") {
                const yTop = priceToY(setup.fib_high);
                const yBottom = priceToY(setup.fib_low);
                const oteColor = setup.direction === "bullish" ? "rgba(0, 255, 136, 0.08)" : "rgba(255, 51, 102, 0.08)";
                ctx.fillStyle = oteColor;
                ctx.fillRect(0, yTop, width - 50, yBottom - yTop);
                ctx.strokeStyle = setup.direction === "bullish" ? "rgba(0, 255, 136, 0.25)" : "rgba(255, 51, 102, 0.25)";
                ctx.lineWidth = 0.5;
                ctx.setLineDash([4, 4]);
                ctx.strokeRect(0, yTop, width - 50, yBottom - yTop);
                ctx.setLineDash([]);
                ctx.fillStyle = setup.direction === "bullish" ? "rgba(0, 255, 136, 0.5)" : "rgba(255, 51, 102, 0.5)";
                ctx.font = "bold 9px JetBrains Mono";
                ctx.textAlign = "right";
                ctx.fillText("OTE (0.62-0.79)", width - 55, yTop + 11);
            }
        });
    }

    // 2. Draw Candlesticks
    chartCandles.forEach((c, idx) => {
        const x = getX(idx) + (candleWidth / 2);
        const yOpen = priceToY(c.open);
        const yClose = priceToY(c.close);
        const yHigh = priceToY(c.high);
        const yLow = priceToY(c.low);

        const isGreen = c.close >= c.open;
        ctx.strokeStyle = isGreen ? "#00ff88" : "#ff3366";
        ctx.fillStyle = isGreen ? "#00ff88" : "#ff3366";

        // Wick
        ctx.beginPath();
        ctx.lineWidth = 1.5;
        ctx.moveTo(x, yHigh);
        ctx.lineTo(x, yLow);
        ctx.stroke();

        // Body
        const bodyTop = Math.min(yOpen, yClose);
        const bodyHeight = Math.max(2, Math.abs(yClose - yOpen));
        ctx.fillRect(x - bodyWidth / 2, bodyTop, bodyWidth, bodyHeight);

        // Time labels on X-axis (every ~10 candles)
        if (idx % 12 === 0 || idx === chartCandles.length - 1) {
            ctx.fillStyle = "#4a5a70";
            ctx.textAlign = "center";
            ctx.fillText(c.time, x, height - 8);
        }
    });

    // === CHART OVERLAYS (In Front of Candles) ===

    // 2.5. Position Entry/SL/TP Lines (from open positions)
    if (chartOverlays.positions) {
        chartOverlays.positions.forEach(pos => {
            // Entry line (gold dashed)
            if (pos.entry > 0) {
                const yEntry = priceToY(pos.entry);
                ctx.strokeStyle = "#ffd700";
                ctx.lineWidth = 1.5;
                ctx.setLineDash([8, 4]);
                ctx.beginPath();
                ctx.moveTo(0, yEntry);
                ctx.lineTo(width - 50, yEntry);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = "#ffd700";
                ctx.font = "bold 10px JetBrains Mono";
                ctx.textAlign = "left";
                const posLabel = pos.type === "LONG" ? "▲ LONG" : "▼ SHORT";
                ctx.fillText(posLabel + " Entry: " + formatPrice(pos.entry), 4, yEntry - 4);
            }
            // Position SL line (red solid with label)
            if (pos.sl > 0) {
                const ySl = priceToY(pos.sl);
                ctx.strokeStyle = "rgba(255, 51, 102, 0.8)";
                ctx.lineWidth = 1.5;
                ctx.setLineDash([6, 3]);
                ctx.beginPath();
                ctx.moveTo(0, ySl);
                ctx.lineTo(width - 50, ySl);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = "#ff3366";
                ctx.font = "bold 9px JetBrains Mono";
                ctx.textAlign = "left";
                ctx.fillText("POS SL: " + formatPrice(pos.sl), 4, ySl + 12);
            }
            // Position TP line (green solid with label)
            if (pos.tp > 0) {
                const yTp = priceToY(pos.tp);
                ctx.strokeStyle = "rgba(0, 255, 136, 0.8)";
                ctx.lineWidth = 1.5;
                ctx.setLineDash([6, 3]);
                ctx.beginPath();
                ctx.moveTo(0, yTp);
                ctx.lineTo(width - 50, yTp);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = "#00ff88";
                ctx.font = "bold 9px JetBrains Mono";
                ctx.textAlign = "left";
                ctx.fillText("POS TP: " + formatPrice(pos.tp), 4, yTp - 4);
            }
        });
    }

    // 2.6. CHoCH Markers (diamond shape)
    if (chartOverlays.smc_setups) {
        chartOverlays.smc_setups.forEach(setup => {
            if (setup.type === "CHoCH") {
                const chartIdx = setup.bar_index - overlayOffset;
                if (chartIdx < 0 || chartIdx >= chartCandles.length) return;
                const x = getX(chartIdx) + (candleWidth / 2);
                const y = priceToY(setup.price);
                const size = 6;
                const color = setup.direction === "bullish" ? "#00ff88" : "#ff3366";
                // Diamond shape
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.moveTo(x, y - size);
                ctx.lineTo(x + size, y);
                ctx.lineTo(x, y + size);
                ctx.lineTo(x - size, y);
                ctx.closePath();
                ctx.fill();
                // Label
                ctx.fillStyle = color;
                ctx.font = "bold 8px JetBrains Mono";
                ctx.textAlign = "center";
                ctx.fillText("CHoCH", x, y - size - 3);
            }
        });
    }

    // 2.7. Sweep Markers (triangle shape)
    if (chartOverlays.smc_setups) {
        chartOverlays.smc_setups.forEach(setup => {
            if (setup.type === "SWEEP") {
                const chartIdx = setup.bar_index - overlayOffset;
                if (chartIdx < 0 || chartIdx >= chartCandles.length) return;
                const x = getX(chartIdx) + (candleWidth / 2);
                const y = priceToY(setup.price);
                const size = 5;
                const color = setup.direction === "bullish" ? "#00f0ff" : "#ffd700";
                // Triangle pointing in sweep direction
                ctx.fillStyle = color;
                ctx.beginPath();
                if (setup.direction === "bullish") {
                    // Upward triangle (swept lows, reversal up)
                    ctx.moveTo(x, y + size + 4);
                    ctx.lineTo(x - size, y + size + 4 + size * 2);
                    ctx.lineTo(x + size, y + size + 4 + size * 2);
                } else {
                    // Downward triangle (swept highs, reversal down)
                    ctx.moveTo(x, y - size - 4);
                    ctx.lineTo(x - size, y - size - 4 - size * 2);
                    ctx.lineTo(x + size, y - size - 4 - size * 2);
                }
                ctx.closePath();
                ctx.fill();
                // Label
                ctx.font = "bold 8px JetBrains Mono";
                ctx.textAlign = "center";
                const labelY = setup.direction === "bullish" ? y + size + 4 + size * 2 + 10 : y - size - 4 - size * 2 - 4;
                ctx.fillText("SWEEP", x, labelY);
            }
        });
    }

    // 3. Draw Take Profit (TP) Barrier Line if active
    if (engineState.tp_price && engineState.tp_price > 0) {
        const yTp = priceToY(engineState.tp_price);
        ctx.strokeStyle = "#00ff88";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(0, yTp);
        ctx.lineTo(width - 50, yTp);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = "#00ff88";
        ctx.font = "bold 10px JetBrains Mono";
        ctx.textAlign = "right";
        ctx.fillText("TP: " + formatPrice(engineState.tp_price), width - 55, yTp - 4);
    }

    // 4. Draw Stop Loss (SL) Barrier Line if active
    if (engineState.sl_price && engineState.sl_price > 0) {
        const ySl = priceToY(engineState.sl_price);
        ctx.strokeStyle = "#ff3366";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(0, ySl);
        ctx.lineTo(width - 50, ySl);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = "#ff3366";
        ctx.font = "bold 10px JetBrains Mono";
        ctx.textAlign = "right";
        ctx.fillText("SL: " + formatPrice(engineState.sl_price), width - 55, ySl - 4);
    }

    // 5. Draw Current Candle Close Line (Cyan)
    if (chartCandles.length > 0) {
        const latestPrice = chartCandles[chartCandles.length - 1].close;
        const yCurr = priceToY(latestPrice);

        ctx.strokeStyle = "#00f0ff";
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(0, yCurr);
        ctx.lineTo(width - 50, yCurr);
        ctx.stroke();
        ctx.setLineDash([]);
    }
}

async function promptSetSLTP(ticket, currentSL, currentTP) {
    const slInput = prompt(`Enter new Stop Loss for position #${ticket} (0 to remove):`, currentSL > 0 ? currentSL.toFixed(5) : "");
    if (slInput === null) return;
    const tpInput = prompt(`Enter new Take Profit for position #${ticket} (0 to remove):`, currentTP > 0 ? currentTP.toFixed(5) : "");
    if (tpInput === null) return;

    const sl = parseFloat(slInput) || 0.0;
    const tp = parseFloat(tpInput) || 0.0;

    try {
        const res = await fetch("/api/set_sltp", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticket, sl, tp })
        });
        const data = await res.json();
        if (data.success) {
            alert("✅ " + data.message);
            updateDashboard();
        } else {
            alert("❌ Error: " + data.message);
        }
    } catch (err) {
        alert("❌ Failed to update SL/TP: " + err);
    }
}

async function confirmClosePosition(ticket) {
    if (!confirm(`Are you sure you want to close position #${ticket} at current market price?`)) {
        return;
    }
    try {
        const res = await fetch("/api/close_position", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticket })
        });
        const data = await res.json();
        if (data.success) {
            alert("✅ " + data.message);
            updateDashboard();
        } else {
            alert("❌ Error: " + data.message);
        }
    } catch (err) {
        alert("❌ Failed to close position: " + err);
    }
}


// ==========================================
// AI QUANT ASSISTANT & CHATBOT CONTROLLERS
// ==========================================
let chatHistory = [];

document.addEventListener("DOMContentLoaded", () => {
    const btnAnalyze = document.getElementById("btn-llm-analyze");
    if (btnAnalyze) {
        btnAnalyze.addEventListener("click", runAIAnalysis);
    }

    const btnSend = document.getElementById("btn-llm-send");
    const chatInput = document.getElementById("llm-chat-input");
    if (btnSend && chatInput) {
        btnSend.addEventListener("click", sendChatMessage);
        chatInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") sendChatMessage();
        });
    }
});

function formatChatResponse(text) {
    if (!text) return "";
    let html = text;
    if (typeof marked !== "undefined" && marked.parse) {
        try {
            html = marked.parse(text);
        } catch (e) {
            html = text.replace(/\n/g, '<br>');
        }
    } else {
        html = text.replace(/\n/g, '<br>');
    }
    return html;
}

function compileMathInElement(element) {
    if (typeof renderMathInElement !== "undefined") {
        try {
            renderMathInElement(element, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\(', right: '\\)', display: false },
                    { left: '\\[', right: '\\]', display: true }
                ],
                throwOnError: false
            });
        } catch (e) {
            console.error("KaTeX math compilation error:", e);
        }
    }
}

async function runAIAnalysis() {
    const outputEl = document.getElementById("llm-report-output");

    if (!outputEl) return;
    outputEl.innerHTML = "<em>[AI Quant Analyst] Connecting to NVIDIA GPU Inference (openai/gpt-oss-120b) and gathering 80 institutional features & MT5 positions... Please wait ⏳</em>";

    try {
        const res = await fetch("/api/llm_analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ api_key: "", model_name: "", api_base_url: "" })
        });
        const data = await res.json();
        if (data.success) {
            let text = "### 🏛️ INSTITUTIONAL AI REASONING & POSITION REPORT\n\n";
            text += data.summary + "\n\n";

            if (data.execution_results && data.execution_results.length > 0) {
                text += "#### 🛡️ META-GATE AUTOMATIC POSITION ADJUSTMENTS\n";
                data.execution_results.forEach(msg => {
                    text += `- ${msg}\n`;
                });
            } else if (data.recommendations && data.recommendations.length > 0) {
                text += "#### 💡 RECOMMENDED POSITION ADJUSTMENTS\n```json\n";
                text += JSON.stringify(data.recommendations, null, 2) + "\n```";
            } else {
                text += "#### 🛡️ POSITION VERIFICATION\nAll current SL/TP levels are optimal. No modifications needed.";
            }
            outputEl.innerHTML = formatChatResponse(text);
            compileMathInElement(outputEl);
        } else {
            outputEl.innerHTML = `<span style="color: var(--accent-crimson);">[ERROR] AI Analysis Failed:<br>${data.error || data.message || "Unknown error"}</span>`;
        }
    } catch (err) {
        outputEl.innerHTML = `<span style="color: var(--accent-crimson);">[ERROR] Failed to contact backend:<br>${err}</span>`;
    }
}

async function sendChatMessage() {
    const inputEl = document.getElementById("llm-chat-input");
    const messagesEl = document.getElementById("llm-chat-messages");

    if (!inputEl || !messagesEl) return;
    const userMsg = inputEl.value.trim();
    if (!userMsg) return;

    // Append user message
    const userDiv = document.createElement("div");
    userDiv.className = "chat-msg user";
    userDiv.style.cssText = "background: rgba(255, 215, 0, 0.08); border-left: 3px solid var(--accent-gold); padding: 10px 14px; border-radius: 4px; align-self: flex-end; max-width: 90%; margin-left: auto;";
    userDiv.innerHTML = `<strong style="color: var(--accent-gold); font-size: 11px; display: block; margin-bottom: 4px;">YOU</strong>${userMsg}`;
    messagesEl.appendChild(userDiv);
    inputEl.value = "";
    messagesEl.scrollTop = messagesEl.scrollHeight;

    // Append typing indicator
    const aiDiv = document.createElement("div");
    aiDiv.className = "chat-msg ai";
    aiDiv.style.cssText = "background: rgba(0, 255, 255, 0.08); border-left: 3px solid var(--accent-cyan); padding: 10px 14px; border-radius: 4px; align-self: flex-start; max-width: 90%;";
    aiDiv.innerHTML = `<strong style="color: var(--accent-cyan); font-size: 11px; display: block; margin-bottom: 4px;">ANTIGRAVITY AI QUANT (NVIDIA)</strong><em>Thinking & analyzing live feature matrix... ⏳</em>`;
    messagesEl.appendChild(aiDiv);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    try {
        const res = await fetch("/api/llm_chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: userMsg, history: chatHistory, api_key: "", model_name: "", api_base_url: "" })
        });
        const data = await res.json();
        if (data.success) {
            aiDiv.innerHTML = `<strong style="color: var(--accent-cyan); font-size: 11px; display: block; margin-bottom: 4px;">ANTIGRAVITY AI QUANT</strong><div class="chat-markdown-body">${formatChatResponse(data.content)}</div>`;
            compileMathInElement(aiDiv);
            chatHistory.push({ role: "user", content: userMsg });
            chatHistory.push({ role: "assistant", content: data.content });
        } else {
            aiDiv.innerHTML = `<strong style="color: var(--accent-crimson); font-size: 11px; display: block; margin-bottom: 4px;">ERROR</strong>${data.error || "Failed to generate response."}`;
        }
    } catch (err) {
        aiDiv.innerHTML = `<strong style="color: var(--accent-crimson); font-size: 11px; display: block; margin-bottom: 4px;">ERROR</strong>Network error: ${err}`;
    }
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

// One-Click Trading Handlers
document.getElementById('oc-sell-btn').addEventListener('click', () => { executeTrade('SELL'); });
document.getElementById('oc-buy-btn').addEventListener('click', () => { executeTrade('BUY'); });

async function executeTrade(action) {
    const vol = parseFloat(document.getElementById('oc-lot').value);
    if (isNaN(vol) || vol <= 0) { alert('Invalid lot size'); return; }
    try {
        const response = await fetch('/api/trade', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol: activeSymbol, action: action, volume: vol })
        });
        const result = await response.json();
        if (result.success) {
            console.log(`[Trade Success] ${action} ${vol} on ${activeSymbol}`);
            updateDashboard(); // instantly refresh positions overlay
        } else {
            alert(`Trade Failed: ${result.error}`);
        }
    } catch (e) {
        console.error('Trade request error:', e);
        alert('Trade request error');
    }
}

