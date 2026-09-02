import { setupScene } from "./scene.js";
import { buildTerrainMesh } from "./mesh.js";

// API base URL — uses the same host:port the frontend is served from,
// expecting a reverse proxy or same-origin backend.
// Falls back to localhost:8001 for local dev.
const API_BASE = window.location.origin || "http://localhost:8001";

// Scene object — returns { scene, setTerrainMesh, setHoverCallback, ... }
let sceneObj = null;
let currentMesh = null;
let currentMetadata = null;

// UI elements
const uploadInput = document.getElementById("image-upload");
const processBtn = document.getElementById("process-btn");
const loadingDiv = document.getElementById("loading");
const errorMsg = document.getElementById("error-msg");
const controlsSection = document.getElementById("controls-section");
const demoSelect = document.getElementById("demo-select");
const metaDisplay = document.getElementById("meta-display");
const hoverReadout = document.getElementById("hover-readout");
const pinInfo = document.getElementById("pin-info");

/**
 * Display reconstruction metadata in the UI panel.
 * Distinguishes relative DSM from absolute DSM clearly.
 */
function displayMetadata(metadata) {
  if (!metaDisplay || !metadata) return;

  const mode = metadata.mode || "unknown";
  const isAbsolute = mode === "metric" || mode === "absolute";
  const units = metadata.units || (isAbsolute ? "meters" : "relative");
  const elevMin = metadata.elevation_min;
  const elevMax = metadata.elevation_max;
  const cal = metadata.calibration || {};
  const input = metadata.input || {};

  const modeLabel = isAbsolute ? "ABSOLUTE DSM" : "RELATIVE DSM";
  const modeColor = isAbsolute ? "#66cc66" : "#ffcc00";

  let html = `<div style="border-bottom: 1px solid #555; padding-bottom: 6px; margin-bottom: 6px;">`;
  html += `<span style="color: ${modeColor}; font-weight: bold;">${modeLabel}</span>`;
  html += `</div>`;

  // Elevation range — only show if values are present
  if (elevMin !== undefined && elevMax !== undefined) {
    html += `<div><b>Elevation:</b> ${Number(elevMin).toFixed(1)} — ${Number(elevMax).toFixed(1)} ${isAbsolute ? "m" : "(relative)"}</div>`;
  }

  // Units
  html += `<div><b>Units:</b> ${units}</div>`;

  // Calibration info
  if (cal.applied) {
    html += `<div><b>Calibration:</b> ${cal.source || "unknown"}</div>`;
    if (cal.scale !== undefined && cal.scale !== null) {
      html += `<div><b>Scale:</b> ${Number(cal.scale).toFixed(4)}, <b>Offset:</b> ${Number(cal.offset).toFixed(2)}</div>`;
    }
    if (cal.valid_samples !== undefined) {
      html += `<div><b>Fit samples:</b> ${cal.valid_samples}</div>`;
    }
  } else {
    const reason = cal.reason || "Not calibrated";
    html += `<div><b>Calibration:</b> ${reason}</div>`;
  }

  // Input raster info (only show geospatial fields when present)
  if (input.crs) {
    html += `<div><b>CRS:</b> ${input.crs}${input.epsg ? " (EPSG:" + input.epsg + ")" : ""}</div>`;
  }
  if (input.resolution) {
    html += `<div><b>Resolution:</b> ${Number(input.resolution.x).toFixed(4)} × ${Number(input.resolution.y).toFixed(4)}</div>`;
  }

  metaDisplay.innerHTML = html;
  metaDisplay.style.display = "block";
}

/**
 * Convert a 3D mesh Y-position back to original DSM elevation.
 *
 * The mesh maps: elevation → ((elev - elevMin) / range) * maxHeight
 * So inverse:   y_normalized = y_mesh / maxHeight
 *               elevation = elevMin + y_normalized * range
 *
 * We offset by currentMesh.position.y (typically -10).
 */
function meshYToElevation(point) {
  if (!currentMetadata) return null;
  const elevMin = currentMetadata.elevation_min ?? 0;
  const elevMax = currentMetadata.elevation_max ?? 1;
  const maxHeight = 20;
  const meshYOffset = currentMesh ? currentMesh.position.y : 0;

  // point.y is in world space; adjust for mesh offset
  const localY = point.y - meshYOffset;
  const normalized = Math.max(0, Math.min(1, localY / maxHeight));
  return elevMin + normalized * (elevMax - elevMin);
}

/**
 * Convert a 3D mesh XZ position to a display string.
 * For georeferenced input, derive approximate lat/lon from bounds.
 * For non-georeferenced, show mesh coordinates.
 */
function meshXZToCoords(point) {
  if (!currentMetadata)
    return `X: ${point.x.toFixed(1)}, Z: ${point.z.toFixed(1)}`;

  const input = currentMetadata.input || {};
  const bounds = input.bounds;
  if (bounds && bounds.left !== undefined) {
    // Map mesh XZ to geographic bounds
    // Mesh is 100×100 centered at origin, so range is -50 to +50
    const fracX = (point.x + 50) / 100;
    const fracZ = (point.z + 50) / 100;
    const lon = bounds.left + fracX * (bounds.right - bounds.left);
    const lat =
      bounds.top -
      fracZ * (bounds.top - bounds.top + bounds.top - bounds.bottom);
    // More correct: lat goes from top to bottom
    const latVal = bounds.top - fracZ * (bounds.top - bounds.bottom);
    return `Lon: ${lon.toFixed(6)}, Lat: ${latVal.toFixed(6)}`;
  }

  return `X: ${point.x.toFixed(1)}, Z: ${point.z.toFixed(1)}`;
}

/**
 * Render the terrain mesh from a heightmap + texture.
 */
const renderTerrain = async (heightmapUrl, textureUrl, metadata = null) => {
  try {
    loadingDiv.style.display = "block";
    errorMsg.textContent = "";
    errorMsg.innerHTML = "";
    controlsSection.style.display = "none";

    if (metaDisplay) metaDisplay.style.display = "none";
    if (hoverReadout) hoverReadout.style.display = "none";
    if (pinInfo) pinInfo.style.display = "none";

    // Use backend metadata if available, sensible defaults otherwise
    const elevationMin = metadata?.elevation_min ?? 0;
    const elevationMax = metadata?.elevation_max ?? 1;

    const mesh = await buildTerrainMesh(heightmapUrl, textureUrl, {
      segments: 256,
      width: 100,
      depth: 100,
      maxHeight: 20,
      elevationMin,
      elevationMax,
    });

    if (currentMesh) {
      sceneObj.scene.remove(currentMesh);
      currentMesh.geometry.dispose();
      currentMesh.material.dispose();
    }

    currentMesh = mesh;
    currentMesh.position.y = -10;
    sceneObj.scene.add(currentMesh);

    // Register mesh for raycasting
    sceneObj.setTerrainMesh(mesh);

    controlsSection.style.display = "block";

    // Store metadata for later use
    currentMetadata = metadata;

    // Display metadata in UI
    displayMetadata(metadata);
  } catch (err) {
    errorMsg.textContent = "Failed to load terrain: " + (err.message || err);
    console.error(err);
  } finally {
    loadingDiv.style.display = "none";
  }
};

/**
 * Show a structured error message based on failure type.
 */
function showError(err) {
  const msg = err.message || String(err);
  let html = "";

  if (
    msg.includes("Failed to fetch") ||
    msg.includes("NetworkError") ||
    msg.includes("ERR_CONNECTION_REFUSED")
  ) {
    html = `<b>Connection failed</b> — Is the backend running on port 8001?`;
  } else if (msg.includes("HTTP error! status: 500")) {
    html = `<b>Server error</b> — The depth model may not be installed. Check backend logs.`;
  } else if (msg.includes("HTTP error! status: 400")) {
    html = `<b>Invalid input</b> — ${msg}`;
  } else if (msg.includes("HTTP error! status: 404")) {
    html = `<b>Not found</b> — The requested resource was not found on the server.`;
  } else if (msg.includes("timeout") || msg.includes("Timeout")) {
    html = `<b>Timeout</b> — Processing took too long. Try a smaller image.`;
  } else {
    html = `<b>Failed to process image:</b> ${msg}`;
  }

  errorMsg.innerHTML = html;
}

function init() {
  // Set up the scene — now returns object with scene, setTerrainMesh, etc.
  sceneObj = setupScene(document.getElementById("canvas-container"));

  // Wire up hover readout
  sceneObj.setHoverCallback((point) => {
    if (!hoverReadout || !currentMetadata) return;
    if (point) {
      const elevation = meshYToElevation(point);
      const isAbsolute =
        currentMetadata.mode === "metric" ||
        currentMetadata.mode === "absolute";
      const unitStr = isAbsolute ? "m" : "(relative)";
      const coords = meshXZToCoords(point);
      hoverReadout.textContent = `${coords} | Elev: ${elevation !== null ? elevation.toFixed(1) : "?"} ${unitStr}`;
      hoverReadout.style.display = "block";
    } else {
      hoverReadout.style.display = "none";
    }
  });

  // Wire up double-click pin inspection
  sceneObj.setClickCallback((point) => {
    if (!pinInfo || !currentMetadata) return;
    const elevation = meshYToElevation(point);
    const isAbsolute =
      currentMetadata.mode === "metric" || currentMetadata.mode === "absolute";
    const unitStr = isAbsolute ? "m" : "(relative)";
    const coords = meshXZToCoords(point);
    pinInfo.innerHTML = `<b>Pinned:</b> ${coords} | Elev: ${elevation !== null ? elevation.toFixed(1) : "?"} ${unitStr}`;
    pinInfo.style.display = "block";
  });

  processBtn.addEventListener("click", async () => {
    if (!uploadInput.files || uploadInput.files.length === 0) {
      alert("Please select an image file first.");
      return;
    }

    const file = uploadInput.files[0];
    const formData = new FormData();
    formData.append("image", file);

    processBtn.disabled = true;
    demoSelect.disabled = true;

    try {
      loadingDiv.textContent = "Processing... (this may take a minute)";
      loadingDiv.style.display = "block";

      const response = await fetch(`${API_BASE}/process`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok)
        throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      if (data.error) throw new Error(data.error);

      const metadata = {
        elevation_min: data.elevation_min,
        elevation_max: data.elevation_max,
        mode: data.mode,
        units: data.units,
        calibration: data.calibration,
        input: data.input,
        warnings: data.warnings,
      };

      await renderTerrain(
        API_BASE + data.heightmap_url,
        API_BASE + data.texture_url,
        metadata,
      );
    } catch (err) {
      showError(err);
    } finally {
      processBtn.disabled = false;
      demoSelect.disabled = false;
      loadingDiv.style.display = "none";
    }
  });

  demoSelect.addEventListener("change", async (e) => {
    const val = e.target.value;
    if (!val) return;

    // Demo cache is 8-bit uint8 grayscale, elevation range 0-255
    loadingDiv.textContent = "Loading cached region...";
    await renderTerrain(
      "./demo_cache/demo_forest_heightmap.png",
      "./demo_cache/demo_forest_texture.png",
      {
        elevation_min: 0,
        elevation_max: 255,
        mode: "relative",
        units: "demo (scaled, no metric meaning)",
      },
    );

    // Reset select
    demoSelect.value = "";
  });
}

init();
