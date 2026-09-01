import * as THREE from 'three';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export function setupScene(containerElement) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x87CEEB); // Sky blue
    scene.fog = new THREE.Fog(0x87CEEB, 20, 150);

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    // Start somewhere above the terrain
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

    // Controls
    let usePointerLock = true;
    const pointerControls = new PointerLockControls(camera, document.body);
    const orbitControls = new OrbitControls(camera, renderer.domElement);
    orbitControls.enabled = false;

    // Click to lock cursor
    renderer.domElement.addEventListener('click', () => {
        if (usePointerLock && !pointerControls.isLocked) {
            pointerControls.lock();
        }
    });

    const movementState = { forward: false, backward: false, left: false, right: false, up: false, down: false };
    const velocity = new THREE.Vector3();
    const direction = new THREE.Vector3();
    const speed = 40.0;

    const onKeyDown = (event) => {
        switch (event.code) {
            case 'KeyW': movementState.forward = true; break;
            case 'KeyA': movementState.left = true; break;
            case 'KeyS': movementState.backward = true; break;
            case 'KeyD': movementState.right = true; break;
            case 'Space': movementState.up = true; break;
            case 'ShiftLeft': movementState.down = true; break;
            case 'KeyC': 
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
            case 'KeyW': movementState.forward = false; break;
            case 'KeyA': movementState.left = false; break;
            case 'KeyS': movementState.backward = false; break;
            case 'KeyD': movementState.right = false; break;
            case 'Space': movementState.up = false; break;
            case 'ShiftLeft': movementState.down = false; break;
        }
    };
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);

    // Resize handler
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });

    let prevTime = performance.now();
    function animate() {
        requestAnimationFrame(animate);
        const time = performance.now();
        const delta = (time - prevTime) / 1000;

        if (usePointerLock && pointerControls.isLocked) {
            velocity.x -= velocity.x * 10.0 * delta;
            velocity.z -= velocity.z * 10.0 * delta;
            velocity.y -= velocity.y * 10.0 * delta;

            direction.z = Number(movementState.forward) - Number(movementState.backward);
            direction.x = Number(movementState.right) - Number(movementState.left);
            direction.y = Number(movementState.up) - Number(movementState.down);
            direction.normalize();

            if (movementState.forward || movementState.backward) velocity.z -= direction.z * speed * delta;
            if (movementState.left || movementState.right) velocity.x -= direction.x * speed * delta;
            if (movementState.up || movementState.down) velocity.y += direction.y * speed * delta;

            pointerControls.moveRight(-velocity.x * delta);
            pointerControls.moveForward(-velocity.z * delta);
            camera.position.y += velocity.y * delta;
        } else if (!usePointerLock) {
            orbitControls.update();
        }

        renderer.render(scene, camera);
        prevTime = time;
    }

    animate();

    return scene;
}
