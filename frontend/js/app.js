const GRID_SIZE = 256;
const WARN_VEL = 2.8;
const DANGER_VEL = 3.8;

let ws = null;
let currentTemperature = null;
let currentVelocity = null;
let velocityHotspots = [];
let displayMode = "both";
let contourLevels = 20;
let particleDensity = 1500;
let isSimulating = false;
let simInterval = null;
let simStep = 0;

const canvasHeat = document.getElementById("heatmap-canvas");
const canvasVel = document.getElementById("velocity-canvas");
const canvasParticles = document.getElementById("particles-canvas");
const svgChart = document.getElementById("temperature-chart");
const container = document.getElementById("chart-container");

function setupCanvas(canvas, w, h) {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
}

let ctxHeat, ctxVel, ctxP;
let cssW = 0, cssH = 0;

function resizeCanvas() {
    const rect = container.getBoundingClientRect();
    cssW = Math.floor(rect.width);
    cssH = Math.floor(rect.height);
    ctxHeat = setupCanvas(canvasHeat, cssW, cssH);
    ctxVel = setupCanvas(canvasVel, cssW, cssH);
    ctxP = setupCanvas(canvasParticles, cssW, cssH);
    svgChart.setAttribute("viewBox", `0 0 ${GRID_SIZE} ${GRID_SIZE}`);
    if (currentTemperature) drawHeatmap();
    if (currentVelocity) drawVelocityHeat();
    resetParticles();
}
window.addEventListener("resize", resizeCanvas);

function init() {
    resizeCanvas();
    drawColorbars();
    setTimeout(connectWS, 200);
    bindEvents();
    startParticleLoop();
}

function connectWS() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
    ws.onopen = () => {
        setStatus(true);
        console.log("[WS] Connected");
    };
    ws.onclose = () => {
        setStatus(false);
        console.log("[WS] Reconnect in 3s");
        setTimeout(connectWS, 3000);
    };
    ws.onmessage = (ev) => {
        try {
            const d = JSON.parse(ev.data);
            handleData(d);
        } catch (e) { console.error(e); }
    };
}

function setStatus(ok) {
    const el = document.getElementById("connection-status");
    if (ok) {
        el.className = "status connected";
        el.textContent = "● 已连接";
    } else {
        el.className = "status disconnected";
        el.textContent = "● 未连接";
    }
}

function handleData(d) {
    document.getElementById("fps-display").textContent =
        "FPS: " + (d.stats?.fps || 0).toFixed(1);
    document.getElementById("frame-count").textContent =
        "帧数: " + (d.stats?.frame_count || 0);

    if (d.data && d.shape) {
        const [h, w] = d.shape;
        const arr = new Float32Array(d.data);
        currentTemperature = { data: arr, h, w, minT: d.min_temp, maxT: d.max_temp };
        updateTempStats(d.stats, arr, w);
        drawHeatmap();
        drawContours();
    }

    if (d.velocity) {
        const v = d.velocity;
        const [vhr, vwr] = v.sparse_shape;
        const sf = new Float32Array(v.sparse_flat);
        const sparse = { shape: [vhr, vwr], step: v.step, vy: [], vx: [] };
        for (let i = 0; i < vhr; i++) {
            const rowa = [], rowb = [];
            for (let j = 0; j < vwr; j++) {
                const idx = (i * vwr + j) * 2;
                rowa.push(sf[idx]);
                rowb.push(sf[idx + 1]);
            }
            sparse.vy.push(rowa); sparse.vx.push(rowb);
        }
        const mag64 = new Float32Array(v.magnitude_64);
        currentVelocity = {
            sparse, step: v.step, grid_size: v.grid_size,
            minV: v.min_velocity, maxV: v.max_velocity, avgV: v.avg_velocity,
            warning_count: v.warning_count, danger_count: v.danger_count,
            magnitude_64: mag64, mag64_shape: [v.grid_size / 4, v.grid_size / 4],
            total_flow: v.total_flow_m3s,
        };
        velocityHotspots = v.hotspots || [];
        updateVelocityStats();
        drawVelocityHeat();
        handleWarning(d.velocity_warning);
    }
}

function updateTempStats(stats, arr, w) {
    document.getElementById("stat-max").textContent =
        (stats.max_temp ?? 0).toFixed(1) + " °C";
    document.getElementById("stat-min").textContent =
        (stats.min_temp ?? 0).toFixed(1) + " °C";
    document.getElementById("stat-avg").textContent =
        (stats.avg_temp ?? 0).toFixed(1) + " °C";
    const cy = Math.floor(w / 2), cx = Math.floor(w / 2);
    const c = arr[cy * w + cx];
    document.getElementById("stat-center").textContent = c.toFixed(1) + " °C";
    let sum = 0, cnt = 0;
    for (let y = 0; y < w; y++) {
        for (let x = 0; x < w; x++) {
            const dx = x - cx, dy = y - cy;
            const d = Math.sqrt(dx * dx + dy * dy);
            if (d > w * 0.42) { sum += arr[y * w + x]; cnt++; }
        }
    }
    document.getElementById("stat-edge").textContent =
        cnt > 0 ? (sum / cnt).toFixed(1) + " °C" : "-- °C";
    document.getElementById("temp-min").textContent =
        (stats.min_temp ?? 0).toFixed(0) + "°C";
    document.getElementById("temp-max").textContent =
        (stats.max_temp ?? 0).toFixed(0) + "°C";
}

function updateVelocityStats() {
    if (!currentVelocity) return;
    const v = currentVelocity;
    document.getElementById("vel-stat-max").textContent =
        v.maxV.toFixed(2) + " m/s";
    document.getElementById("vel-stat-avg").textContent =
        v.avgV.toFixed(2) + " m/s";
    document.getElementById("vel-stat-min").textContent =
        v.minV.toFixed(2) + " m/s";
    document.getElementById("vel-stat-flow").textContent =
        (v.total_flow || 0).toFixed(2) + " m³/s";
    const elD = document.getElementById("vel-stat-danger");
    const elW = document.getElementById("vel-stat-warn");
    const elS = document.getElementById("vel-stat-status");
    elD.textContent = v.danger_count;
    elW.textContent = v.warning_count;
    const fsEl = document.getElementById("flow-status");
    if (v.danger_count > 0) {
        elS.className = "status-badge danger";
        elS.textContent = "危险";
        fsEl.className = "flow-danger";
        fsEl.textContent = "● 流速危险";
    } else if (v.warning_count > 0) {
        elS.className = "status-badge warning";
        elS.textContent = "预警";
        fsEl.className = "flow-warning";
        fsEl.textContent = "● 流速预警";
    } else {
        elS.className = "status-badge normal";
        elS.textContent = "正常";
        fsEl.className = "flow-normal";
        fsEl.textContent = "● 流速正常";
    }
    document.getElementById("vel-min").textContent = v.minV.toFixed(1);
    document.getElementById("vel-max").textContent = v.maxV.toFixed(1);
}

function handleWarning(vw) {
    const alertEl = document.getElementById("warning-alert-card");
    const overlayEl = document.getElementById("warning-overlay");
    const warnIcon = document.getElementById("warning-icon");
    const warnTitle = document.getElementById("warning-title");
    const warnDesc = document.getElementById("warning-description");
    const warnMaxV = document.getElementById("warn-max-vel");
    const warnD = document.getElementById("warn-danger-areas");
    const warnW = document.getElementById("warn-warning-areas");
    const hsListEl = document.getElementById("hotspot-list");
    const v = currentVelocity;

    if (!vw || !vw.has_warning) {
        alertEl.style.display = "none";
        overlayEl.style.display = "none";
        return;
    }

    alertEl.style.display = "block";
    const isDanger = vw.status === "danger";
    alertEl.className = "warning-alert" + (isDanger ? "" : " warning-level");
    warnIcon.textContent = isDanger ? "🚨" : "⚠️";
    warnTitle.textContent = isDanger ? "严重：局部管道冲刷风险预警" : "预警：局部煤气流速偏高";
    const levelText = isDanger
        ? "检测到严重管道布气，局部区域流速超过危险阈值（>" + DANGER_VEL + " m/s），可能对衬砖造成严重机械冲刷！"
        : "检测到局部区域煤气流速接近预警阈值（>" + WARN_VEL + " m/s），请密切关注发展趋势。";
    warnDesc.textContent = levelText;
    if (v) {
        warnMaxV.textContent = v.maxV.toFixed(2) + " m/s";
        warnD.textContent = v.danger_count;
        warnW.textContent = v.warning_count;
    }
    overlayEl.style.display = isDanger ? "block" : "none";

    hsListEl.innerHTML = "";
    velocityHotspots.forEach(hs => {
        const div = document.createElement("div");
        div.className = "hotspot-item" + (hs.severity === "danger" ? "" : " warning");
        div.innerHTML = `
            <div>
                <div class="hotspot-id">#${hs.id} ${hs.severity === "danger" ? "危险" : "预警"}区</div>
                <div class="hotspot-info">位置 (${(hs.center_x / GRID_SIZE * 100).toFixed(0)}%, ${(hs.center_y / GRID_SIZE * 100).toFixed(0)}%) · ${hs.size_pixels}单元 · 均${hs.avg_velocity.toFixed(1)}m/s</div>
            </div>
            <div class="hotspot-max">${hs.max_velocity.toFixed(2)} m/s</div>
        `;
        hsListEl.appendChild(div);
    });
}

const tempColor = d3.scaleSequential()
    .domain([0, 1])
    .interpolator(d3.interpolateInferno);

function drawHeatmap() {
    if (!currentTemperature || cssW === 0) return;
    const { data, h, w, minT, maxT } = currentTemperature;
    const range = maxT - minT || 1;

    if (displayMode === "particles") {
        ctxHeat.clearRect(0, 0, cssW, cssH);
        return;
    }

    const tmpCnv = document.createElement("canvas");
    tmpCnv.width = w; tmpCnv.height = h;
    const tctx = tmpCnv.getContext("2d");
    const img = tctx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
        const t = Math.max(0, Math.min(1, (data[i] - minT) / range));
        const c = d3.color(tempColor(t));
        const k = i * 4;
        img.data[k] = c.r;
        img.data[k + 1] = c.g;
        img.data[k + 2] = c.b;
        img.data[k + 3] = (displayMode === "particles" || displayMode === "velocity-heat") ? 0 : 220;
    }
    tctx.putImageData(img, 0, 0);
    ctxHeat.clearRect(0, 0, cssW, cssH);
    ctxHeat.imageSmoothingEnabled = true;
    ctxHeat.imageSmoothingQuality = "high";
    ctxHeat.drawImage(tmpCnv, 0, 0, cssW, cssH);
}

function drawVelocityHeat() {
    ctxVel.clearRect(0, 0, cssW, cssH);
    if (!currentVelocity) return;

    if (displayMode !== "velocity-heat" && displayMode !== "full") return;

    const [h64, w64] = currentVelocity.mag64_shape;
    const mag = currentVelocity.magnitude_64;
    const maxV = Math.max(currentVelocity.maxV, WARN_VEL + 2);
    const tmpC = document.createElement("canvas");
    tmpC.width = w64; tmpC.height = h64;
    const tctx = tmpC.getContext("2d");
    const img = tctx.createImageData(w64, h64);
    for (let i = 0; i < w64 * h64; i++) {
        const v = mag[i];
        const k = i * 4;
        const col = velocityColor(v, maxV);
        img.data[k] = col[0]; img.data[k+1] = col[1]; img.data[k+2] = col[2];
        img.data[k + 3] = displayMode === "full" ? 160 : 235;
    }
    tctx.putImageData(img, 0, 0);
    ctxVel.imageSmoothingEnabled = true;
    ctxVel.imageSmoothingQuality = "high";
    ctxVel.drawImage(tmpC, 0, 0, cssW, cssH);
}

function velocityColor(v, maxV) {
    if (v >= DANGER_VEL) {
        const t = Math.min(1, (v - DANGER_VEL) / Math.max(0.1, maxV - DANGER_VEL));
        return [255, Math.floor(50 * (1 - t) + 0), Math.floor(80 * (1 - t) + 0)];
    }
    if (v >= WARN_VEL) {
        const t = (v - WARN_VEL) / (DANGER_VEL - WARN_VEL);
        return [255, Math.floor(165 * (1 - t) + 50), 0];
    }
    if (v >= WARN_VEL * 0.5) {
        const t = (v - WARN_VEL * 0.5) / (WARN_VEL * 0.5);
        return [Math.floor(60 + 195 * t), Math.floor(200 * (1 - t) + 165), Math.floor(60 * (1 - t))];
    }
    const t = v / (WARN_VEL * 0.5);
    return [Math.floor(30 + 30 * t), Math.floor(80 + 120 * t), Math.floor(180 + 20 * t)];
}

function drawContours() {
    svgChart.innerHTML = "";
    if (!currentTemperature) return;
    if (displayMode === "heatmap" || displayMode === "particles" || displayMode === "velocity-heat") return;
    try {
        const { data, h, w, minT, maxT } = currentTemperature;
        const grid = [];
        for (let y = 0; y < h; y++) {
            const row = [];
            for (let x = 0; x < w; x++) row.push(data[y * w + x]);
            grid.push(row);
        }
        const levels = contourLevels;
        const contours = d3.contours().size([w, h]).thresholds(levels)(grid.flat());
        const color = d3.scaleSequential(d3.interpolateTurbo)
            .domain([minT, maxT]);
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        contours.forEach(c => {
            const col = d3.color(color(c.value));
            col.opacity = displayMode === "both" ? 0.85 : 1.0;
            const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
            path.setAttribute("d", d3.geoPath(d3.geoIdentity().scale(1))(c));
            path.setAttribute("fill", displayMode === "both" ? "none" : col);
            path.setAttribute("stroke", col);
            path.setAttribute("stroke-width", displayMode === "both" ? 0.8 : 0.3);
            path.setAttribute("fill-opacity", displayMode === "both" ? 0 : 0.55);
            g.appendChild(path);
        });
        svgChart.appendChild(g);
    } catch (e) {
        console.error("contour err:", e);
    }
}

// ================= Streamline Particles Engine =================
class ParticleSystem {
    constructor() {
        this.particles = [];
        this.trails = 10;
    }
    reset(n) {
        this.particles = [];
        for (let i = 0; i < n; i++) this.particles.push(this.create());
    }
    create() {
        const ang = Math.random() * Math.PI * 2;
        const r = Math.sqrt(Math.random()) * 0.45;
        return {
            x: 0.5 + Math.cos(ang) * r,
            y: 0.5 + Math.sin(ang) * r,
            life: Math.random() * 0.9 + 0.3,
            age: 0,
            speed: Math.random() * 0.4 + 0.2,
            history: [],
        };
    }
    sampleVelocity(nx, ny) {
        if (!currentVelocity) return [0, 0];
        const { sparse, step } = currentVelocity;
        const [hR, wR] = sparse.shape;
        const fx = Math.max(0, Math.min(wR - 1.001, nx * wR));
        const fy = Math.max(0, Math.min(hR - 1.001, ny * hR));
        const x0 = Math.floor(fx), y0 = Math.floor(fy);
        const tx = fx - x0, ty = fy - y0;
        const vx = (1 - ty) * ((1 - tx) * sparse.vx[y0][x0] + tx * sparse.vx[y0][x0 + 1]) +
                   ty * ((1 - tx) * sparse.vx[y0 + 1][x0] + tx * sparse.vx[y0 + 1][x0 + 1]);
        const vy = (1 - ty) * ((1 - tx) * sparse.vy[y0][x0] + tx * sparse.vy[y0][x0 + 1]) +
                   ty * ((1 - tx) * sparse.vy[y0 + 1][x0] + tx * sparse.vy[y0 + 1][x0 + 1]);
        return [vx, vy];
    }
    sampleMagnitude(nx, ny) {
        if (!currentVelocity) return 0;
        const [hr, wr] = currentVelocity.mag64_shape;
        const fx = Math.max(0, Math.min(wr - 1.001, nx * wr));
        const fy = Math.max(0, Math.min(hr - 1.001, ny * hr));
        const x0 = Math.floor(fx), y0 = Math.floor(fy);
        const tx = fx - x0, ty = fy - y0;
        const m = currentVelocity.magnitude_64;
        return (1 - ty) * ((1 - tx) * m[y0 * wr + x0] + tx * m[y0 * wr + x0 + 1]) +
               ty * ((1 - tx) * m[(y0 + 1) * wr + x0] + tx * m[(y0 + 1) * wr + x0 + 1]);
    }
    step(dt) {
        if (!currentVelocity) return;
        const maxV = Math.max(currentVelocity.maxV, 2);
        for (let i = 0; i < this.particles.length; i++) {
            const p = this.particles[i];
            p.history.push([p.x, p.y]);
            if (p.history.length > this.trails) p.history.shift();

            const [vx, vy] = this.sampleVelocity(p.x, p.y);
            const norm = Math.max(0.001, Math.sqrt(vx * vx + vy * vy));
            const magSample = this.sampleMagnitude(p.x, p.y);
            const localSpeed = (magSample / maxV) * 0.012 * p.speed + 0.0015;
            p.x += (vx / norm) * localSpeed;
            p.y += (vy / norm) * localSpeed;
            p.age += dt;

            const cx = p.x - 0.5, cy = p.y - 0.5;
            const d = Math.sqrt(cx * cx + cy * cy);
            if (d > 0.495 || p.age > p.life) {
                Object.assign(p, this.create());
                p.history = [];
            }
        }
    }
    draw(ctx) {
        ctx.clearRect(0, 0, cssW, cssH);
        if (!currentVelocity) return;
        if (displayMode !== "particles" && displayMode !== "full") return;
        const maxV = Math.max(currentVelocity.maxV, 2);
        for (let i = 0; i < this.particles.length; i++) {
            const p = this.particles[i];
            const h = p.history;
            if (h.length < 2) continue;
            for (let s = 1; s < h.length; s++) {
                const [x0, y0] = h[s - 1];
                const [x1, y1] = h[s];
                const nx = (x0 + x1) / 2, ny = (y0 + y1) / 2;
                const mag = this.sampleMagnitude(nx, ny);
                const [r, g, b] = velocityColor(mag, maxV);
                const alpha = (s / h.length) * 0.85;
                const thick = 0.6 + Math.min(3.2, mag / 2.2);
                ctx.beginPath();
                ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
                ctx.lineWidth = thick;
                ctx.lineCap = "round";
                ctx.moveTo(x0 * cssW, y0 * cssH);
                ctx.lineTo(x1 * cssW, y1 * cssH);
                ctx.stroke();
            }
        }
        if (velocityHotspots && velocityHotspots.length > 0) {
            velocityHotspots.forEach(hs => {
                const cx = hs.center_x / GRID_SIZE * cssW;
                const cy = hs.center_y / GRID_SIZE * cssH;
                const rad = Math.sqrt(hs.size_pixels) / 64 * Math.min(cssW, cssH);
                ctx.beginPath();
                const col = hs.severity === "danger" ? "255,71,87" : "255,165,2";
                const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, rad * 1.4);
                grad.addColorStop(0, `rgba(${col},0.35)`);
                grad.addColorStop(1, `rgba(${col},0)`);
                ctx.fillStyle = grad;
                ctx.arc(cx, cy, rad * 1.4, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = `rgba(${col},0.8)`;
                ctx.lineWidth = 2;
                ctx.setLineDash([6, 4]);
                ctx.beginPath();
                ctx.arc(cx, cy, rad, 0, Math.PI * 2);
                ctx.stroke();
                ctx.setLineDash([]);
            });
        }
    }
}

const particles = new ParticleSystem();
let lastTime = performance.now();
function startParticleLoop() {
    function tick() {
        const now = performance.now();
        const dt = (now - lastTime) / 1000;
        lastTime = now;
        particles.step(dt);
        particles.draw(ctxP);
        requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}
function resetParticles() {
    particles.reset(particleDensity);
}

function drawColorbars() {
    const cb = d3.select("#colorbar");
    const w = cb.node().clientWidth || 400;
    const scale = d3.scaleLinear().domain([0, 1]).range([0, w]);
    const axis = d3.axisBottom(scale).ticks(0);
    const defs = cb.append("defs");
    const lg = defs.append("linearGradient").attr("id", "tgrad");
    const stops = 10;
    for (let i = 0; i <= stops; i++) {
        const t = i / stops;
        lg.append("stop").attr("offset", (t * 100) + "%")
          .attr("stop-color", tempColor(t));
    }
    cb.html(function() { return cb.html(); }).attr("width", w).attr("height", 18)
      .append("rect").attr("width", w).attr("height", 18)
      .attr("rx", 3).attr("fill", "url(#tgrad)");

    const vcb = d3.select("#velocity-colorbar");
    const w2 = vcb.node().clientWidth || 400;
    const defs2 = vcb.append("defs");
    const lg2 = defs2.append("linearGradient").attr("id", "vgrad");
    const stops2 = 20;
    const maxV = DANGER_VEL * 1.3;
    for (let i = 0; i <= stops2; i++) {
        const t = i / stops2;
        const v = t * maxV;
        const [r, g, b] = velocityColor(v, maxV);
        lg2.append("stop").attr("offset", (t * 100) + "%")
          .attr("stop-color", `rgb(${r},${g},${b})`);
    }
    vcb.attr("width", w2).attr("height", 18)
       .append("rect").attr("width", w2).attr("height", 18)
       .attr("rx", 3).attr("fill", "url(#vgrad)");
}

// 模拟数据
function genSimTemperature(step, channeling = 0) {
    const W = GRID_SIZE, H = GRID_SIZE;
    const arr = new Float32Array(W * H);
    const cx = W / 2, cy = H / 2;
    let minT = 1e9, maxT = -1e9;
    const chCx = cx + Math.cos(channeling) * W * 0.15;
    const chCy = cy + Math.sin(channeling * 1.3) * H * 0.15;
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const dx = (x - cx) / W, dy = (y - cy) / H;
            const d = Math.sqrt(dx * dx + dy * dy);
            let t = 1500 * Math.exp(-d * d * 4.5)
                + 500 * Math.exp(-d * d * 20)
                + 120 + Math.sin(step * 0.03 + x * 0.015 + y * 0.013) * 25;
            if (channeling !== 0) {
                const cdx = (x - chCx) / W, cdy = (y - chCy) / H;
                const cd = Math.sqrt(cdx * cdx + cdy * cdy);
                const boost = 850 * Math.exp(-cd * cd * 80) * Math.sin(step * 0.08);
                t += boost;
            }
            t += (Math.random() - 0.5) * 12;
            if (t < minT) minT = t;
            if (t > maxT) maxT = t;
            arr[y * W + x] = t;
        }
    }
    currentTemperature = { data: arr, h: H, w: W, minT, maxT };
}

function estimateSimVelocity(channeling = 0) {
    const stepV = 4;
    const hr = GRID_SIZE / stepV, wr = GRID_SIZE / stepV;
    const sparse = { shape: [hr, wr], step: stepV, vy: [], vx: [] };
    const cx = GRID_SIZE / 2, cy = GRID_SIZE / 2;
    const chCx = cx + Math.cos(channeling) * GRID_SIZE * 0.15;
    const chCy = cy + Math.sin(channeling * 1.3) * GRID_SIZE * 0.15;
    const mag64 = new Float32Array(64 * 64);
    let maxV = 0, minV = 1e9, sumV = 0;
    let warnC = 0, dangerC = 0;
    const hotspotMap = new Map();

    for (let i = 0; i < hr; i++) {
        const rv = [], rx = [];
        for (let j = 0; j < wr; j++) {
            const px = j * stepV + stepV / 2;
            const py = i * stepV + stepV / 2;
            let dx = (px - cx) / GRID_SIZE;
            let dy = (py - cy) / GRID_SIZE;
            const r = Math.sqrt(dx * dx + dy * dy);
            if (r < 1e-6) { dx = 0; dy = -0.001; }
            const baseSpeed = 1.5 + Math.exp(-r * r * 6) * 2.5;
            let vx = dx / r * baseSpeed;
            let vy = dy / r * baseSpeed;

            if (channeling !== 0) {
                const ddx = px - chCx, ddy = py - chCy;
                const dist = Math.sqrt(ddx * ddx + ddy * ddy) / GRID_SIZE;
                const chBoost = Math.exp(-dist * dist * 55) * 8;
                if (chBoost > 0.5) {
                    const cn = Math.max(1e-3, Math.sqrt(ddx * ddx + ddy * ddy));
                    vx += ddx / cn * chBoost * 0.03;
                    vy += ddy / cn * chBoost * 0.025;
                }
            }
            vx += (Math.random() - 0.5) * 0.08;
            vy += (Math.random() - 0.5) * 0.08;

            const mag = Math.sqrt(vx * vx + vy * vy);
            const displayMag = baseSpeed + (channeling !== 0 ? Math.exp(-r * r * 6) * 2.5 : 0);
            let effMag = displayMag * (1 + Math.abs(chBoost || 0) * 0.4);
            if (r > 0.45) effMag *= Math.exp(-(r - 0.45) * (r - 0.45) * 40);
            effMag = Math.max(0.2, effMag);

            if (effMag > maxV) maxV = effMag;
            if (effMag < minV) minV = effMag;
            sumV += effMag;
            if (effMag >= WARN_VEL) warnC++;
            if (effMag >= DANGER_VEL) dangerC++;

            rv.push(vy); rx.push(vx);

            const m64i = Math.floor(i / stepV);
            const m64j = Math.floor(j / stepV);
            if (m64i < 64 && m64j < 64) {
                mag64[m64i * 64 + m64j] = Math.max(mag64[m64i * 64 + m64j], effMag);
            }
        }
        sparse.vy.push(rv); sparse.vx.push(rx);
    }
    const cells = hr * wr;
    const avgV = sumV / cells;

    // hotspots
    const hots = [];
    if (channeling !== 0 && maxV > WARN_VEL) {
        hots.push({
            id: 1,
            center_x: chCx, center_y: chCy,
            max_velocity: Math.min(maxV, DANGER_VEL + 2.5),
            avg_velocity: avgV * 1.6,
            size_pixels: Math.floor(80 + Math.random() * 100),
            severity: maxV > DANGER_VEL ? "danger" : "warning",
        });
    }

    currentVelocity = {
        sparse, step: stepV, grid_size: GRID_SIZE,
        minV, maxV, avgV, warning_count: warnC, danger_count: dangerC,
        magnitude_64: mag64, mag64_shape: [64, 64],
        total_flow: 6.5,
    };
    velocityHotspots = hots;
}

function emitSimFrame() {
    simStep++;
    const channelingPhase = (simStep % 300) > 240 && (simStep % 300) < 290
        ? (simStep % 300) * 0.1
        : 0;
    genSimTemperature(simStep, channelingPhase);
    estimateSimVelocity(channelingPhase);

    const stats = {
        frame_count: simStep, fps: 10,
        min_temp: currentTemperature.minT,
        max_temp: currentTemperature.maxT,
        avg_temp: currentTemperature.data.reduce((a, b) => a + b, 0) / currentTemperature.data.length,
    };
    updateTempStats(stats, currentTemperature.data, GRID_SIZE);
    updateVelocityStats();
    drawHeatmap(); drawContours(); drawVelocityHeat();

    handleWarning({
        status: currentVelocity.danger_count > 0 ? "danger" :
                currentVelocity.warning_count > 0 ? "warning" : "normal",
        has_warning: currentVelocity.warning_count > 0 || currentVelocity.danger_count > 0,
    });
}

function bindEvents() {
    document.getElementById("display-mode").addEventListener("change", (e) => {
        displayMode = e.target.value;
        if (displayMode === "heatmap" || displayMode === "both") {
            svgChart.style.display = "block";
        } else if (displayMode === "particles" || displayMode === "velocity-heat") {
            svgChart.style.display = "none";
        } else {
            svgChart.style.display = "block";
        }
        drawHeatmap(); drawContours(); drawVelocityHeat();
    });
    document.getElementById("contour-levels").addEventListener("input", (e) => {
        contourLevels = parseInt(e.target.value);
        document.getElementById("levels-value").textContent = contourLevels;
        drawContours();
    });
    document.getElementById("particle-density").addEventListener("input", (e) => {
        particleDensity = parseInt(e.target.value);
        document.getElementById("density-value").textContent = particleDensity;
        resetParticles();
    });
    document.getElementById("btn-start-sim").addEventListener("click", () => {
        if (isSimulating) return;
        isSimulating = true;
        setStatus(true);
        simInterval = setInterval(emitSimFrame, 100);
        document.getElementById("btn-start-sim").disabled = true;
        document.getElementById("btn-stop-sim").disabled = false;
    });
    document.getElementById("btn-stop-sim").addEventListener("click", () => {
        isSimulating = false;
        clearInterval(simInterval);
        document.getElementById("btn-start-sim").disabled = false;
        document.getElementById("btn-stop-sim").disabled = true;
    });
    document.getElementById("btn-trigger-channeling").addEventListener("click", () => {
        if (!isSimulating) {
            alert("请先启动模拟数据");
            return;
        }
        simStep = Math.floor(simStep / 300) * 300 + 250;
        emitSimFrame();
    });
    document.getElementById("btn-snapshot").addEventListener("click", () => {
        const el = document.getElementById("chart-container");
        const a = document.createElement("canvas");
        a.width = cssW; a.height = cssH;
        const c = a.getContext("2d");
        c.drawImage(canvasHeat, 0, 0);
        c.drawImage(canvasVel, 0, 0);
        c.drawImage(canvasParticles, 0, 0);
        const url = a.toDataURL("image/png");
        const link = document.createElement("a");
        link.download = `bf-gas-${Date.now()}.png`;
        link.href = url;
        link.click();
    });
}

window.addEventListener("DOMContentLoaded", init);
