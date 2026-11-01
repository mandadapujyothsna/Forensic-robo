// RoboForensic Mission Control & Teleoperation Controller

let currentMode = "autonomous";
let telemetryInterval = null;
let lastLogCount = 0;

// DOM Elements
const streamStatusBadge = document.getElementById("streamStatusBadge");
const streamStatusText = document.getElementById("streamStatusText");
const serialStatusBadge = document.getElementById("serialStatusBadge");
const serialStatusText = document.getElementById("serialStatusText");
const topFps = document.getElementById("topFps");
const currentSourceTag = document.getElementById("currentSourceTag");

const zoneLeft = document.getElementById("zoneLeft");
const zoneCenter = document.getElementById("zoneCenter");
const zoneRight = document.getElementById("zoneRight");

const modeAutoBtn = document.getElementById("modeAutoBtn");
const modeManualBtn = document.getElementById("modeManualBtn");

const targetStateText = document.getElementById("targetStateText");
const targetConfText = document.getElementById("targetConfText");
const targetDistText = document.getElementById("targetDistText");
const distBarFill = document.getElementById("distBarFill");
const stopDistLabel = document.getElementById("stopDistLabel");

const actionCodeBadge = document.getElementById("actionCodeBadge");
const actionDescText = document.getElementById("actionDescText");

const terminalLogs = document.getElementById("terminalLogs");
const clearLogsBtn = document.getElementById("clearLogsBtn");

// Controls
const btnFwd = document.getElementById("btnFwd");
const btnLeft = document.getElementById("btnLeft");
const btnRight = document.getElementById("btnRight");
const btnBwd = document.getElementById("btnBwd");
const btnStop = document.getElementById("btnStop");

// Settings Modal
const settingsModal = document.getElementById("settingsModal");
const openSettingsBtn = document.getElementById("openSettingsBtn");
const closeSettingsBtn = document.getElementById("closeSettingsBtn");
const cancelSettingsBtn = document.getElementById("cancelSettingsBtn");
const saveSettingsBtn = document.getElementById("saveSettingsBtn");
const refreshPortsBtn = document.getElementById("refreshPortsBtn");

const streamUrlInput = document.getElementById("streamUrlInput");
const serialPortSelect = document.getElementById("serialPortSelect");
const confThreshInput = document.getElementById("confThreshInput");
const confThreshVal = document.getElementById("confThreshVal");
const stopDistInput = document.getElementById("stopDistInput");
const stopDistVal = document.getElementById("stopDistVal");
const focalLengthInput = document.getElementById("focalLengthInput");
const knownWidthInput = document.getElementById("knownWidthInput");

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  fetchInitialData();
  startTelemetryPolling();
});

function setupEventListeners() {
  // Mode toggles
  modeAutoBtn.addEventListener("click", () => switchMode("autonomous"));
  modeManualBtn.addEventListener("click", () => switchMode("manual"));

  // Manual Control Buttons
  btnFwd.addEventListener("click", () => sendCommand("F"));
  btnLeft.addEventListener("click", () => sendCommand("L"));
  btnRight.addEventListener("click", () => sendCommand("R"));
  btnBwd.addEventListener("click", () => sendCommand("B"));
  btnStop.addEventListener("click", () => sendCommand("S"));

  // Keyboard Hotkeys
  window.addEventListener("keydown", handleKeydown);

  // Settings
  openSettingsBtn.addEventListener("click", openSettings);
  closeSettingsBtn.addEventListener("click", closeSettings);
  cancelSettingsBtn.addEventListener("click", closeSettings);
  saveSettingsBtn.addEventListener("click", saveSettings);
  refreshPortsBtn.addEventListener("click", fetchPorts);

  confThreshInput.addEventListener("input", (e) => {
    confThreshVal.textContent = e.target.value;
  });
  stopDistInput.addEventListener("input", (e) => {
    stopDistVal.textContent = `${e.target.value} cm`;
  });

  clearLogsBtn.addEventListener("click", () => {
    terminalLogs.innerHTML = '<div class="log-line">[LOGS CLEARED]</div>';
  });
}

// Telemetry Poller
function startTelemetryPolling() {
  telemetryInterval = setInterval(async () => {
    try {
      const res = await fetch("/api/telemetry");
      if (!res.ok) return;
      const data = await res.json();
      updateDashboard(data);
    } catch (err) {
      console.warn("Telemetry poll error:", err);
    }
  }, 180);
}

function updateDashboard(data) {
  // Update FPS & Stream Status
  topFps.textContent = data.fps ? data.fps.toFixed(1) : "0.0";
  
  if (data.stream_status.includes("Active")) {
    streamStatusBadge.querySelector(".status-dot").className = "status-dot green";
    streamStatusText.textContent = "STREAM ACTIVE";
  } else {
    streamStatusBadge.querySelector(".status-dot").className = "status-dot red";
    streamStatusText.textContent = "OFFLINE";
  }

  // Serial Port Status
  if (data.serial_connected) {
    serialStatusBadge.querySelector(".status-dot").className = "status-dot green";
    serialStatusText.textContent = `${data.serial_port} CONNECTED`;
  } else {
    serialStatusBadge.querySelector(".status-dot").className = "status-dot red";
    serialStatusText.textContent = `${data.serial_port || 'SERIAL'} DISCONNECTED`;
  }

  currentSourceTag.textContent = `SOURCE: ${data.stream_url}`;

  // Mode Sync
  currentMode = data.mode;
  if (currentMode === "autonomous") {
    modeAutoBtn.classList.add("active");
    modeManualBtn.classList.remove("active");
  } else {
    modeManualBtn.classList.add("active");
    modeAutoBtn.classList.remove("active");
  }

  // Target Status & Confidence
  if (data.target_detected) {
    targetStateText.textContent = "TARGET LOCKED";
    targetStateText.style.color = "var(--neon-green)";
    targetConfText.textContent = `Confidence: ${(data.confidence * 100).toFixed(0)}%`;
  } else {
    targetStateText.textContent = "SEARCHING...";
    targetStateText.style.color = "var(--neon-orange)";
    targetConfText.textContent = "Scanning field of view";
  }

  // Estimated Distance & Proximity Gauge
  if (data.distance_cm !== null && data.distance_cm !== undefined) {
    targetDistText.innerHTML = `${data.distance_cm.toFixed(1)} <span class="unit">cm</span>`;
    
    // Proximity bar fill calculation (0 to 100cm scale)
    const pct = Math.min(100, Math.max(5, (data.distance_cm / 80) * 100));
    distBarFill.style.width = `${pct}%`;

    if (data.distance_cm <= data.stop_distance) {
      targetDistText.style.color = "var(--neon-red)";
      distBarFill.style.background = "var(--neon-red)";
    } else {
      targetDistText.style.color = "#fff";
      distBarFill.style.background = "linear-gradient(90deg, var(--neon-red), var(--neon-orange), var(--neon-green))";
    }
  } else {
    targetDistText.innerHTML = `-- <span class="unit">cm</span>`;
    distBarFill.style.width = "0%";
    targetDistText.style.color = "#fff";
  }

  stopDistLabel.textContent = `Stop Threshold: ${data.stop_distance} cm`;

  // Zone Tracking Radar
  zoneLeft.classList.remove("active");
  zoneCenter.classList.remove("active");
  zoneRight.classList.remove("active");

  if (data.zone === "LEFT") zoneLeft.classList.add("active");
  else if (data.zone === "CENTER") zoneCenter.classList.add("active");
  else if (data.zone === "RIGHT") zoneRight.classList.add("active");

  // Action Code & Description
  actionCodeBadge.textContent = data.command_code;
  actionDescText.textContent = data.command_desc;

  if (data.command_code === "F") {
    actionCodeBadge.style.color = "var(--neon-green)";
    actionCodeBadge.style.borderColor = "var(--neon-green)";
  } else if (data.command_code === "S") {
    actionCodeBadge.style.color = "var(--neon-red)";
    actionCodeBadge.style.borderColor = "var(--neon-red)";
  } else {
    actionCodeBadge.style.color = "var(--neon-cyan)";
    actionCodeBadge.style.borderColor = "var(--neon-cyan)";
  }

  // Update Logs
  if (data.logs && data.logs.length !== lastLogCount) {
    terminalLogs.innerHTML = data.logs.map(l => `<div class="log-line">${l}</div>`).join("");
    terminalLogs.scrollTop = terminalLogs.scrollHeight;
    lastLogCount = data.logs.length;
  }
}

// Mode Switch
async function switchMode(mode) {
  try {
    const res = await fetch("/api/mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode })
    });
    if (res.ok) {
      currentMode = mode;
      if (mode === "autonomous") {
        modeAutoBtn.classList.add("active");
        modeManualBtn.classList.remove("active");
      } else {
        modeManualBtn.classList.add("active");
        modeAutoBtn.classList.remove("active");
      }
    }
  } catch (e) {
    console.error("Failed to switch mode:", e);
  }
}

// Manual Command Transmission
async function sendCommand(cmd) {
  // If in autonomous, temporarily switch or notify
  if (currentMode === "autonomous") {
    await switchMode("manual");
  }

  highlightButton(cmd);

  try {
    await fetch("/api/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command: cmd })
    });
  } catch (err) {
    console.error("Error sending command:", err);
  }
}

function highlightButton(cmd) {
  const map = {
    'F': btnFwd,
    'L': btnLeft,
    'R': btnRight,
    'B': btnBwd,
    'S': btnStop
  };
  const btn = map[cmd];
  if (btn) {
    btn.classList.add("pressed");
    setTimeout(() => btn.classList.remove("pressed"), 200);
  }
}

// Keyboard Navigation Hotkeys
function handleKeydown(e) {
  if (["input", "select", "textarea"].includes(document.activeElement.tagName.toLowerCase())) {
    return;
  }

  const key = e.key.toLowerCase();
  if (key === "w" || key === "arrowup") {
    e.preventDefault();
    sendCommand("F");
  } else if (key === "a" || key === "arrowleft") {
    e.preventDefault();
    sendCommand("L");
  } else if (key === "d" || key === "arrowright") {
    e.preventDefault();
    sendCommand("R");
  } else if (key === "s" || key === "arrowdown") {
    e.preventDefault();
    sendCommand("B");
  } else if (key === " " || key === "escape") {
    e.preventDefault();
    sendCommand("S");
  }
}

// Settings Modal Logic
function openSettings() {
  settingsModal.classList.add("open");
  fetchPorts();
  fetchCalibration();
}

function closeSettings() {
  settingsModal.classList.remove("open");
}

function setStreamPreset(url) {
  streamUrlInput.value = url;
}

async function fetchPorts() {
  try {
    const res = await fetch("/api/ports");
    const ports = await res.json();
    serialPortSelect.innerHTML = ports.map(p => 
      `<option value="${p.port}">${p.port} - ${p.description}</option>`
    ).join("");
    if (ports.length === 0) {
      serialPortSelect.innerHTML = `<option value="">No COM Ports Detected</option>`;
    }
  } catch (e) {
    console.warn("Could not scan ports:", e);
  }
}

async function fetchCalibration() {
  try {
    const res = await fetch("/api/calibration");
    const cal = await res.json();
    focalLengthInput.value = cal.focal_length || 500.0;
    knownWidthInput.value = cal.known_width || 10.0;
  } catch (e) {
    console.warn("Could not fetch calibration:", e);
  }
}

async function fetchInitialData() {
  try {
    const res = await fetch("/api/telemetry");
    const data = await res.json();
    streamUrlInput.value = data.stream_url;
    confThreshInput.value = data.conf_threshold;
    confThreshVal.textContent = data.conf_threshold;
    stopDistInput.value = data.stop_distance;
    stopDistVal.textContent = `${data.stop_distance} cm`;
  } catch (e) {}
}

async function saveSettings() {
  const stream_url = streamUrlInput.value.trim();
  const serial_port = serialPortSelect.value;
  const conf_threshold = parseFloat(confThreshInput.value);
  const stop_distance = parseFloat(stopDistInput.value);

  try {
    await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        stream_url,
        serial_port,
        conf_threshold,
        stop_distance
      })
    });

    // Save calibration
    const focal_length = parseFloat(focalLengthInput.value);
    const known_width = parseFloat(knownWidthInput.value);
    await fetch("/api/calibration", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ focal_length, known_width })
    });

    closeSettings();
  } catch (e) {
    alert("Error saving settings: " + e.message);
  }
}
