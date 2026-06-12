class TemperatureVisualizer {
    constructor() {
        this.width = 0;
        this.height = 0;
        this.gridSize = 256;
        this.temperatureData = null;
        this.minTemp = 200;
        this.maxTemp = 1500;
        this.contourLevels = 20;
        this.displayMode = "both";
        this.ws = null;
        this.simulatorRunning = false;
        this.simulatorInterval = null;
        this.simFrameCount = 0;

        this.init();
        this.bindEvents();
        this.connectWebSocket();
    }

    init() {
        const container = d3.select(".chart-container");
        const node = container.node();
        const rect = node.getBoundingClientRect();
        this.width = rect.width;
        this.height = rect.height;

        this.svg = d3.select("#temperature-chart")
            .attr("viewBox", `0 0 ${this.gridSize} ${this.gridSize}`)
            .attr("preserveAspectRatio", "xMidYMid meet");

        this.heatmapGroup = this.svg.append("g").attr("class", "heatmap-group");
        this.contourGroup = this.svg.append("g").attr("class", "contour-group");

        this.colorScale = d3.scaleSequential()
            .domain([0, 1])
            .interpolator(d3.interpolateTurbo);

        this.renderColorbar();
        this.renderInitialPlaceholder();
    }

    renderColorbar() {
        const colorbar = d3.select("#colorbar");
        const gradientId = "temp-gradient";

        let svg = colorbar.selectAll("svg").data([null]);
        const svgEnter = svg.enter().append("svg")
            .attr("width", "100%")
            .attr("height", "100%")
            .style("border-radius", "4px");

        svgEnter.append("defs").append("linearGradient")
            .attr("id", gradientId)
            .attr("x1", "0%")
            .attr("y1", "0%")
            .attr("x2", "100%")
            .attr("y2", "0%");

        svgEnter.append("rect")
            .attr("width", "100%")
            .attr("height", "100%")
            .attr("fill", `url(#${gradientId})`);

        const gradient = d3.select(`#${gradientId}`);
        const stops = [];
        for (let i = 0; i <= 20; i++) {
            const t = i / 20;
            stops.push({ offset: `${t * 100}%`, color: this.colorScale(t) });
        }

        const stopSel = gradient.selectAll("stop").data(stops);
        stopSel.enter().append("stop")
            .merge(stopSel)
            .attr("offset", d => d.offset)
            .attr("stop-color", d => d.color);
    }

    renderInitialPlaceholder() {
        this.heatmapGroup.append("rect")
            .attr("width", this.gridSize)
            .attr("height", this.gridSize)
            .attr("fill", "#2a2a4a");

        this.heatmapGroup.append("text")
            .attr("x", this.gridSize / 2)
            .attr("y", this.gridSize / 2)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle")
            .attr("fill", "#888")
            .attr("font-size", "14")
            .text("等待数据连接...");
    }

    bindEvents() {
        document.getElementById("display-mode").addEventListener("change", (e) => {
            this.displayMode = e.target.value;
            this.updateDisplay();
        });

        document.getElementById("contour-levels").addEventListener("input", (e) => {
            this.contourLevels = parseInt(e.target.value);
            document.getElementById("levels-value").textContent = this.contourLevels;
            if (this.temperatureData) {
                this.renderContours();
            }
        });

        document.getElementById("btn-start-sim").addEventListener("click", () => {
            this.startSimulator();
        });

        document.getElementById("btn-stop-sim").addEventListener("click", () => {
            this.stopSimulator();
        });

        document.getElementById("btn-snapshot").addEventListener("click", () => {
            this.takeSnapshot();
        });

        window.addEventListener("resize", () => {
            const container = d3.select(".chart-container");
            const rect = container.node().getBoundingClientRect();
            this.width = rect.width;
            this.height = rect.height;
        });
    }

    connectWebSocket() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws/stream`;

        try {
            this.ws = new WebSocket(wsUrl);
        } catch (e) {
            this.ws = new WebSocket("ws://localhost:8000/ws/stream");
        }

        this.ws.onopen = () => {
            console.log("[WS] Connected");
            this.updateConnectionStatus(true);
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === "temperature") {
                this.handleTemperatureData(data);
            }
        };

        this.ws.onclose = () => {
            console.log("[WS] Disconnected");
            this.updateConnectionStatus(false);
            setTimeout(() => this.connectWebSocket(), 3000);
        };

        this.ws.onerror = (error) => {
            console.error("[WS] Error:", error);
        };
    }

    handleTemperatureData(data) {
        const shape = data.shape;
        this.gridSize = shape[0];
        this.temperatureData = new Float32Array(data.data);

        this.minTemp = data.min_temp;
        this.maxTemp = data.max_temp;

        this.updateStats(data.stats);
        this.updateColorbarLabels();
        this.updateDisplay();

        document.getElementById("grid-size").textContent = `${this.gridSize} × ${this.gridSize}`;
    }

    updateDisplay() {
        if (this.displayMode === "heatmap" || this.displayMode === "both") {
            this.renderHeatmap();
            this.heatmapGroup.style("display", null);
        } else {
            this.heatmapGroup.style("display", "none");
        }

        if (this.displayMode === "contour" || this.displayMode === "both") {
            this.renderContours();
            this.contourGroup.style("display", null);
        } else {
            this.contourGroup.style("display", "none");
        }
    }

    renderHeatmap() {
        if (!this.temperatureData) return;

        this.heatmapGroup.selectAll("*").remove();

        const canvas = document.createElement("canvas");
        canvas.width = this.gridSize;
        canvas.height = this.gridSize;
        const ctx = canvas.getContext("2d");
        const imageData = ctx.createImageData(this.gridSize, this.gridSize);
        const pixels = imageData.data;

        const range = this.maxTemp - this.minTemp || 1;

        for (let i = 0; i < this.temperatureData.length; i++) {
            const t = (this.temperatureData[i] - this.minTemp) / range;
            const clampedT = Math.max(0, Math.min(1, t));
            const color = d3.color(this.colorScale(clampedT));

            const idx = i * 4;
            pixels[idx] = color.r;
            pixels[idx + 1] = color.g;
            pixels[idx + 2] = color.b;
            pixels[idx + 3] = this.displayMode === "both" ? 200 : 255;
        }

        ctx.putImageData(imageData, 0, 0);

        const imgUrl = canvas.toDataURL();
        this.heatmapGroup.append("image")
            .attr("href", imgUrl)
            .attr("width", this.gridSize)
            .attr("height", this.gridSize);
    }

    renderContours() {
        if (!this.temperatureData) return;

        this.contourGroup.selectAll("*").remove();

        const values = this.temperatureData;
        const n = this.gridSize;

        const grid = new Array(n);
        for (let j = 0; j < n; j++) {
            grid[j] = new Array(n);
            for (let i = 0; i < n; i++) {
                grid[j][i] = values[j * n + i];
            }
        }

        const thresholds = d3.range(this.minTemp, this.maxTemp, (this.maxTemp - this.minTemp) / this.contourLevels);

        const contours = d3.contours()
            .size([n, n])
            .thresholds(thresholds)(values);

        const path = d3.geoPath(d3.geoIdentity().scale(1));

        const range = this.maxTemp - this.minTemp || 1;

        this.contourGroup.selectAll("path")
            .data(contours)
            .enter()
            .append("path")
            .attr("d", d => path(d))
            .attr("fill", "none")
            .attr("stroke", d => {
                const t = (d.value - this.minTemp) / range;
                return this.colorScale(Math.max(0, Math.min(1, t)));
            })
            .attr("stroke-width", d => {
                const t = (d.value - this.minTemp) / range;
                return 0.5 + t * 1.5;
            })
            .attr("stroke-opacity", 0.9)
            .append("title")
            .text(d => `${d.value.toFixed(1)} °C`);
    }

    updateStats(stats) {
        document.getElementById("fps-display").textContent = `FPS: ${stats.fps || 0}`;
        document.getElementById("frame-count").textContent = `帧数: ${stats.frame_count || 0}`;

        if (stats.min_temp !== undefined) {
            document.getElementById("stat-min").textContent = `${stats.min_temp.toFixed(1)} °C`;
            document.getElementById("stat-max").textContent = `${stats.max_temp.toFixed(1)} °C`;
            document.getElementById("stat-avg").textContent = `${stats.avg_temp.toFixed(1)} °C`;

            if (this.temperatureData) {
                const centerIdx = Math.floor(this.gridSize / 2) * this.gridSize + Math.floor(this.gridSize / 2);
                document.getElementById("stat-center").textContent = `${this.temperatureData[centerIdx].toFixed(1)} °C`;

                let edgeSum = 0;
                let edgeCount = 0;
                for (let i = 0; i < this.gridSize; i++) {
                    edgeSum += this.temperatureData[i];
                    edgeSum += this.temperatureData[(this.gridSize - 1) * this.gridSize + i];
                    edgeSum += this.temperatureData[i * this.gridSize];
                    edgeSum += this.temperatureData[i * this.gridSize + this.gridSize - 1];
                    edgeCount += 4;
                }
                document.getElementById("stat-edge").textContent = `${(edgeSum / edgeCount).toFixed(1)} °C`;
            }
        }
    }

    updateColorbarLabels() {
        document.getElementById("temp-min").textContent = `${this.minTemp.toFixed(0)}°C`;
        document.getElementById("temp-max").textContent = `${this.maxTemp.toFixed(0)}°C`;
    }

    updateConnectionStatus(connected) {
        const statusEl = document.getElementById("connection-status");
        if (connected) {
            statusEl.className = "status connected";
            statusEl.textContent = "● 已连接";
        } else {
            statusEl.className = "status disconnected";
            statusEl.textContent = "● 未连接";
        }
    }

    startSimulator() {
        if (this.simulatorRunning) return;
        this.simulatorRunning = true;
        this.simFrameCount = 0;
        document.getElementById("btn-start-sim").disabled = true;
        document.getElementById("btn-stop-sim").disabled = false;

        this.simulatorInterval = setInterval(() => {
            this.generateSimulatedFrame();
        }, 100);
    }

    stopSimulator() {
        this.simulatorRunning = false;
        if (this.simulatorInterval) {
            clearInterval(this.simulatorInterval);
            this.simulatorInterval = null;
        }
        document.getElementById("btn-start-sim").disabled = false;
        document.getElementById("btn-stop-sim").disabled = true;
    }

    generateSimulatedFrame() {
        const n = 256;
        const data = new Float32Array(n * n);
        const cx = n / 2;
        const cy = n / 2;

        this.simFrameCount++;
        const t = this.simFrameCount * 0.05;

        const centerTemp = 1200 + 100 * Math.sin(t);
        const edgeTemp = 400 + 50 * Math.sin(t * 0.7);
        const range = centerTemp - edgeTemp;

        for (let j = 0; j < n; j++) {
            for (let i = 0; i < n; i++) {
                const dx = i - cx;
                const dy = j - cy;
                const dist = Math.sqrt(dx * dx + dy * dy);
                const maxDist = n / 2;

                let baseTemp = centerTemp - range * Math.pow(dist / maxDist, 1.5);

                const noise = 20 * (Math.random() - 0.5);
                baseTemp += noise;

                const ring1 = 30 * Math.sin(dist * 0.08 + t);
                const ring2 = 15 * Math.sin(dist * 0.15 - t * 1.3);
                baseTemp += ring1 + ring2;

                const sectorAngle = Math.atan2(dy, dx);
                const sectorVar = 25 * Math.sin(sectorAngle * 3 + t * 0.5);
                baseTemp += sectorVar;

                data[j * n + i] = Math.max(edgeTemp - 50, Math.min(centerTemp + 50, baseTemp));
            }
        }

        const minTemp = d3.min(data);
        const maxTemp = d3.max(data);

        this.handleTemperatureData({
            type: "temperature",
            shape: [n, n],
            min_temp: minTemp,
            max_temp: maxTemp,
            data: Array.from(data),
            stats: {
                fps: 10,
                frame_count: this.simFrameCount,
                min_temp: minTemp,
                max_temp: maxTemp,
                avg_temp: d3.mean(data),
            }
        });
    }

    takeSnapshot() {
        const svgNode = this.svg.node();
        const svgData = new XMLSerializer().serializeToString(svgNode);
        const svgBlob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });
        const url = URL.createObjectURL(svgBlob);

        const link = document.createElement("a");
        link.href = url;
        link.download = `temperature-snapshot-${Date.now()}.svg`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    window.visualizer = new TemperatureVisualizer();
});
