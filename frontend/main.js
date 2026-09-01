import { setupScene } from './scene.js';
import { buildTerrainMesh } from './mesh.js';

let scene = null;
let currentMesh = null;

const uploadInput = document.getElementById('image-upload');
const processBtn = document.getElementById('process-btn');
const loadingDiv = document.getElementById('loading');
const errorMsg = document.getElementById('error-msg');
const controlsSection = document.getElementById('controls-section');
const demoSelect = document.getElementById('demo-select');

function init() {
    // Set up the scene immediately (starts blank)
    scene = setupScene(document.getElementById('canvas-container'));

    const renderTerrain = async (heightmapUrl, textureUrl) => {
        try {
            loadingDiv.style.display = 'block';
            errorMsg.textContent = '';
            controlsSection.style.display = 'none';
            
            const mesh = await buildTerrainMesh(heightmapUrl, textureUrl, {
                segments: 256,
                width: 100,
                depth: 100,
                maxHeight: 20
            });

            if (currentMesh) {
                scene.remove(currentMesh);
                currentMesh.geometry.dispose();
                currentMesh.material.dispose();
            }

            currentMesh = mesh;
            currentMesh.position.y = -10;
            scene.add(currentMesh);
            controlsSection.style.display = 'block';
        } catch (err) {
            errorMsg.textContent = "Failed to load terrain: " + (err.message || err);
            console.error(err);
        } finally {
            loadingDiv.style.display = 'none';
        }
    };

    processBtn.addEventListener('click', async () => {
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
            loadingDiv.style.display = 'block';
            
            const response = await fetch('http://localhost:8000/process', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            if (data.error) throw new Error(data.error);

            const baseUrl = 'http://localhost:8000';
            await renderTerrain(baseUrl + data.heightmap_url, baseUrl + data.texture_url);
        } catch (err) {
            errorMsg.textContent = "Failed to process image: " + err.message;
        } finally {
            processBtn.disabled = false;
            demoSelect.disabled = false;
            loadingDiv.style.display = 'none';
        }
    });

    demoSelect.addEventListener('change', async (e) => {
        const val = e.target.value;
        if (!val) return;
        
        // Fast-path loading cached assets directly
        loadingDiv.textContent = "Loading cached region...";
        await renderTerrain('./demo_cache/demo_forest_heightmap.png', './demo_cache/demo_forest_texture.png');
        
        // Reset select
        demoSelect.value = "";
    });
}

init();
