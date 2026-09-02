import * as THREE from "three";
import { PointerLockControls } from "three/addons/controls/PointerLockControls.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

/**
 * Set up the Three.js scene with dual control modes and terrain inspection.
 *
 * Returns an object (not just scene) so callers can:
 *   - setTerrainMesh(mesh)       — enable raycasting against terrain
 *   - setHoverCallback(fn)       — called with Vector3 point or null
 *   - setClickCallback(fn)       — called with Vector3 point on double-click
 *   - removePin()                — remove the inspection marker
 *   - scene                      — the raw THREE.Scene
 */
export function setupScene(containerElement) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x87ceeb);
  scene.fog = new THREE.Fog(0x87ceeb, 20, 150);

  const camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    1000,
  );
  camera.position.set(0, 10, 30);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.shadowMap.enabled = true;
  containerElement.appendChild(renderer.domElement);

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
  scene.add(ambientLight);

  const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
  dirLight.position.set(50, 100, 50);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.width = 2048;
  dirLight.shadow.mapSize.height = 2048;
  scene.add(dirLight);

  // Controls — dual mode: pointer lock (fly) or orbit (inspect)
  let usePointerLock = true;
  const pointerControls = new PointerLockControls(camera, document.body);
  const orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.enabled = false;

  // Click to lock cursor (pointer lock mode)
  renderer.domElement.addEventListener("click", () => {
    if (usePointerLock && !pointerControls.isLocked) {
      pointerControls.lock();
    }
  });

  // Movement state
  const movementState = {
    forward: false,
    backward: false,
    left: false,
    right: false,
    up: false,
    down: false,
  };
  const velocity = new THREE.Vector3();
  const direction = new THREE.Vector3();
  const speed = 40.0;

  const onKeyDown = (event) => {
    switch (event.code) {
      case "KeyW":
        movementState.forward = true;
        break;
      case "KeyA":
        movementState.left = true;
        break;
      case "KeyS":
        movementState.backward = true;
        break;
      case "KeyD":
        movementState.right = true;
        break;
      case "Space":
        movementState.up = true;
        break;
      case "ShiftLeft":
        movementState.down = true;
        break;
      case "KeyC":
        usePointerLock = !usePointerLock;
        if (usePointerLock) {
          orbitControls.enabled = false;
          pointerControls.lock();
        } else {
          pointerControls.unlock();
          orbitControls.enabled = true;
        }
        break;
    }
  };
  const onKeyUp = (event) => {
    switch (event.code) {
      case "KeyW":
        movementState.forward = false;
        break;
      case "KeyA":
        movementState.left = false;
        break;
      case "KeyS":
        movementState.backward = false;
        break;
      case "KeyD":
        movementState.right = false;
        break;
      case "Space":
        movementState.up = false;
        break;
      case "ShiftLeft":
        movementState.down = false;
        break;
    }
  };
  document.addEventListener("keydown", onKeyDown);
  document.addEventListener("keyup", onKeyUp);

  // Resize handler
  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  // --- Raycasting / terrain inspection state ---
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  let terrainMesh = null;
  let hoverCallback = null;
  let clickCallback = null;
  let pinMarker = null;

  function setTerrainMesh(mesh) {
    terrainMesh = mesh;
  }

  function setHoverCallback(fn) {
    hoverCallback = fn;
  }

  function setClickCallback(fn) {
    clickCallback = fn;
  }

  function removePin() {
    if (pinMarker) {
      scene.remove(pinMarker);
      pinMarker.geometry.dispose();
      pinMarker.material.dispose();
      pinMarker = null;
    }
  }

  // Mouse move — update normalized mouse coords for hover detection
  renderer.domElement.addEventListener("mousemove", (event) => {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
  });

  // Double-click — pin an inspection point (not single click, which is pointer lock)
  renderer.domElement.addEventListener("dblclick", (event) => {
    if (!terrainMesh || !clickCallback) return;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(terrainMesh);
    if (intersects.length > 0) {
      const point = intersects[0].point;
      clickCallback(point);

      // Create or move pin marker
      if (!pinMarker) {
        const geo = new THREE.SphereGeometry(0.5, 16, 16);
        const mat = new THREE.MeshBasicMaterial({ color: 0xff0000 });
        pinMarker = new THREE.Mesh(geo, mat);
        scene.add(pinMarker);
      }
      pinMarker.position.copy(point);
      pinMarker.position.y += 0.5;
    }
  });

  // Animation loop
  let prevTime = performance.now();
  function animate() {
    requestAnimationFrame(animate);
    const time = performance.now();
    const delta = (time - prevTime) / 1000;

    if (usePointerLock && pointerControls.isLocked) {
      velocity.x -= velocity.x * 10.0 * delta;
      velocity.z -= velocity.z * 10.0 * delta;
      velocity.y -= velocity.y * 10.0 * delta;

      direction.z =
        Number(movementState.forward) - Number(movementState.backward);
      direction.x = Number(movementState.right) - Number(movementState.left);
      direction.y = Number(movementState.up) - Number(movementState.down);
      direction.normalize();

      if (movementState.forward || movementState.backward)
        velocity.z -= direction.z * speed * delta;
      if (movementState.left || movementState.right)
        velocity.x -= direction.x * speed * delta;
      if (movementState.up || movementState.down)
        velocity.y += direction.y * speed * delta;

      pointerControls.moveRight(-velocity.x * delta);
      pointerControls.moveForward(-velocity.z * delta);
      camera.position.y += velocity.y * delta;
    } else if (!usePointerLock) {
      orbitControls.update();
    }

    // Hover raycasting — detect terrain under cursor
    if (terrainMesh && hoverCallback) {
      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObject(terrainMesh);
      if (intersects.length > 0) {
        hoverCallback(intersects[0].point);
      } else {
        hoverCallback(null);
      }
    }

    renderer.render(scene, camera);
    prevTime = time;
  }

  animate();

  return {
    scene,
    setTerrainMesh,
    setHoverCallback,
    setClickCallback,
    removePin,
  };
}
