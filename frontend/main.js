import { setupScene } from "./scene.js";
import { buildTerrainMesh } from "./mesh.js";

// API base URL — uses the same host:port the frontend is served from,
// expecting a reverse proxy or same-origin backend.
// Falls back to localhost:8001 for local dev.
const API_BASE = window.location.origin || "http://localhost:8001";

let scene = null;
let currentMesh = null;
let currentMetadata = null;

const uploadInput = document.getElementById("image-upload");
const processBtn = document.getElementById("process-btn");
const loadingDiv = document.getElementById("loading");
const errorMsg = document.getElementById("error-msg");
const controlsSection = document.getElementById("controls-section");
const demoSelect = document.getElementById("demo-select");

function init() {
  // Set up the scene immediately (starts blank)
  scene = setupScene(document.getElementById("canvas-container"));

  const renderTerrain = async (heightmapUrl, textureUrl, metadata = null) => {
    try {
      loadingDiv.style.display = "block";
      errorMsg.textContent = "";
      controlsSection.style.display = "none";

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
        scene.remove(currentMesh);
        currentMesh.geometry.dispose();
        currentMesh.material.dispose();
      }

      currentMesh = mesh;
      currentMesh.position.y = -10;
      scene.add(currentMesh);
      controlsSection.style.display = "block";

      // Store metadata for later use (height readout, slope, etc.)
      currentMetadata = metadata;
    } catch (err) {
      errorMsg.textContent = "Failed to load terrain: " + (err.message || err);
      console.error(err);
    } finally {
      loadingDiv.style.display = "none";
    }
  };

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
      errorMsg.textContent = "Failed to process image: " + err.message;
    } finally {
      processBtn.disabled = false;
      demoSelect.disabled = false;
      loadingDiv.style.display = "none";
    }
  });

  demoSelect.addEventListener("change", async (e) => {
    const val = e.target.value;
    if (!val) return;

    // Fast-path loading cached assets directly
    // Demo cache is 8-bit uint8 grayscale, so elevation range is 0-255
    loadingDiv.textContent = "Loading cached region...";
    await renderTerrain(
      "./demo_cache/demo_forest_heightmap.png",
      "./demo_cache/demo_forest_texture.png",
      {
        elevation_min: 0,
        elevation_max: 255,
        mode: "relative",
        units: "demo (scaled)",
      },
    );

    // Reset select
    demoSelect.value = "";
  });
}

init();
