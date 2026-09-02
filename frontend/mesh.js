import * as THREE from "three";

/**
 * Decode elevation from a 16-bit heightmap PNG using backend metadata.
 *
 * The backend saves heightmaps as 16-bit grayscale PNGs where pixel values
 * are linearly mapped: pixel=0 -> elevation_min, pixel=65535 -> elevation_max.
 *
 * When a 16-bit PNG is rendered to an HTML canvas, browsers typically
 * quantize to 8-bit per channel. We read the R channel (0-255) and
 * recover a 0-1 normalized value, then map back to the original elevation range.
 *
 * @param {number} rValue - Red channel value (0-255)
 * @param {number} elevationMin - Minimum elevation from backend metadata
 * @param {number} elevationMax - Maximum elevation from backend metadata
 * @returns {number} Decoded elevation value
 */
export function decodeElevation(rValue, elevationMin, elevationMax) {
  const normalized = rValue / 255.0;
  return elevationMin + normalized * (elevationMax - elevationMin);
}

export function buildTerrainMesh(heightmapUrl, textureUrl, options = {}) {
  const {
    segments = 256,
    width = 100,
    depth = 100,
    maxHeight = 20,
    elevationMin = 0,
    elevationMax = 1,
  } = options;

  const elevationRange = elevationMax - elevationMin;

  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.src = heightmapUrl;
    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = image.width;
      canvas.height = image.height;
      const context = canvas.getContext("2d");
      context.drawImage(image, 0, 0);

      const imgData = context.getImageData(
        0,
        0,
        canvas.width,
        canvas.height,
      ).data;

      const geometry = new THREE.PlaneGeometry(
        width,
        depth,
        segments,
        segments,
      );
      geometry.rotateX(-Math.PI / 2); // Lay flat

      const vertices = geometry.attributes.position.array;

      // For a PlaneGeometry with WxH segments, there are (W+1)*(H+1) vertices
      // We map each vertex to a pixel in the heightmap
      for (let i = 0, j = 0; i < vertices.length; i += 3, j++) {
        // Calculate which row and col this vertex corresponds to
        const row = Math.floor(j / (segments + 1));
        const col = j % (segments + 1);

        // Map vertex grid to image grid
        const imgX = Math.floor((col / segments) * (canvas.width - 1));
        const imgY = Math.floor((row / segments) * (canvas.height - 1));

        // Read red channel and decode elevation using backend metadata
        const pixelIndex = (imgY * canvas.width + imgX) * 4;
        const elevation = decodeElevation(
          imgData[pixelIndex],
          elevationMin,
          elevationMax,
        );

        // Normalize to [0, maxHeight] for 3D rendering
        if (elevationRange > 0) {
          vertices[i + 1] =
            ((elevation - elevationMin) / elevationRange) * maxHeight;
        } else {
          vertices[i + 1] = 0;
        }
      }

      geometry.computeVertexNormals();

      // Load Texture
      const textureLoader = new THREE.TextureLoader();
      textureLoader.load(
        textureUrl,
        (texture) => {
          const material = new THREE.MeshStandardMaterial({
            map: texture,
            wireframe: false,
            roughness: 0.8,
            metalness: 0.1,
          });

          const mesh = new THREE.Mesh(geometry, material);
          resolve(mesh);
        },
        undefined,
        reject,
      );
    };
    image.onerror = () => {
      reject(new Error(`Failed to load image from ${heightmapUrl}`));
    };
  });
}
