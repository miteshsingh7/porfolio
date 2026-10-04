import * as THREE from 'three';

export function createGraphics({ canvas, stage, state, motion }) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: 'low-power' });
  } catch {
    canvas.hidden = true;
    stage.classList.add('graphics-unavailable');
    return;
  }

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 50);
  camera.position.z = 8;
  const sculpture = new THREE.Group();
  scene.add(sculpture);
  const rows = 48, columns = 36, count = rows * columns;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const lime = new THREE.Color('#B8F500');
  const violet = new THREE.Color('#8B5CF6');
  const targets = Array.from({ length: 7 }, () => new Float32Array(count * 3));
  const tau = Math.PI * 2;

  for (let i = 0; i < count; i++) {
    const row = Math.floor(i / columns), col = i % columns;
    const u = row / rows, v = col / columns, a = u * tau, b = v * tau;
    const put = (shape, x, y, z) => targets[shape].set([x, y, z], i * 3);
    const tube = 0.48 + 0.1 * Math.cos(a * 3);
    put(0, (1.3 + tube * Math.cos(b)) * Math.cos(a), (1.3 + tube * Math.cos(b)) * Math.sin(a), tube * Math.sin(b));
    const latitude = (v - 0.5) * Math.PI;
    put(1, 1.65 * Math.cos(latitude) * Math.cos(a), 1.65 * Math.sin(latitude), 1.65 * Math.cos(latitude) * Math.sin(a));
    const x = i % 12, y = Math.floor(i / 12) % 12, z = Math.floor(i / 144);
    put(2, (x - 5.5) * 0.255, (y - 5.5) * 0.255, (z - 5.5) * 0.255);
    const layer = row % 4, strip = Math.floor(row / 4);
    put(3, (v - 0.5) * 3.25, (layer - 1.5) * 0.65 + 0.1 * Math.sin(b + strip), (strip / 11 - 0.5) * 2.6);
    const helix = u * tau * 2 + Math.floor(col / 18) * Math.PI;
    const radius = 0.8 + (col % 18) / 18 * 0.3;
    put(4, Math.cos(helix) * radius, (u - 0.5) * 3.6, Math.sin(helix) * radius);
    put(5, (u - 0.5) * 3.6, Math.sin(a * 1.5 + b) * 0.6 + Math.cos(a * 3) * 0.18, (v - 0.5) * 2.5);
    const orbit = row % 3, r = 1.35 + Math.floor(row / 3) / 16 * 0.28;
    const cx = Math.cos(b) * r, cy = Math.sin(b) * r;
    put(6, cx, cy * Math.cos(orbit * Math.PI / 3), cy * Math.sin(orbit * Math.PI / 3));
    const color = row % 7 < 2 ? violet : lime;
    colors.set([color.r, color.g, color.b], i * 3);
  }
  positions.set(targets[0]);
  const geometry = new THREE.BufferGeometry();
  const positionAttribute = new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('position', positionAttribute);
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const pointMaterial = new THREE.PointsMaterial({ size: 0.025, vertexColors: true, transparent: true, opacity: 0.95, depthWrite: false });
  const points = new THREE.Points(geometry, pointMaterial);
  points.frustumCulled = false;
  sculpture.add(points);

  const indices = [];
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < columns - 1; col++) {
      const i = row * columns + col;
      if (row % 2 === 0) indices.push(i, i + 1);
      if (col % 6 === 0 && row < rows - 1) indices.push(i, i + columns);
    }
  }
  const wireGeometry = new THREE.BufferGeometry();
  const wirePositions = new Float32Array(indices.length * 3);
  const wireColors = new Float32Array(indices.length * 3);
  const wirePositionAttribute = new THREE.BufferAttribute(wirePositions, 3).setUsage(THREE.DynamicDrawUsage);
  const wireColorAttribute = new THREE.BufferAttribute(wireColors, 3).setUsage(THREE.DynamicDrawUsage);
  wireGeometry.setAttribute('position', wirePositionAttribute);
  wireGeometry.setAttribute('color', wireColorAttribute);
  const wireMaterial = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.2, depthWrite: false });
  const wire = new THREE.LineSegments(wireGeometry, wireMaterial);
  wire.frustumCulled = false;
  sculpture.add(wire);

  let width = 1, height = 1, stageBox, progress = state.s, time = 0, previous = 0, frame;
  let running = false, lost = false;
  const smooth = value => value * value * value * (value * (value * 6 - 15) + 10);
  const mix = (a, b, t) => a + (b - a) * t;
  const clamp = value => Math.min(1, Math.max(0, value));
  const worldHeight = 2 * Math.tan(THREE.MathUtils.degToRad(20)) * 8;
  const resize = () => {
    width = innerWidth; height = innerHeight;
    renderer.setPixelRatio(Math.min(devicePixelRatio, width < 700 ? 1.5 : 2));
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    const rect = stage.getBoundingClientRect();
    stageBox = { left: rect.left, top: rect.top + (width < 1100 ? scrollY : 0), width: rect.width, height: rect.height };
    if ((motion.matches || state.paused) && !document.hidden && !lost) draw(performance.now());
  };

  function draw(now) {
    const dt = Math.min((now - previous) / 1000 || 0, 0.05);
    previous = now;
    const reduced = motion.matches || state.paused;
    if (!reduced) time += dt;
    const damping = reduced ? 1 : 1 - Math.exp(-dt * 5.5);
    progress = mix(progress, state.s, damping);
    const index = Math.min(6, Math.floor(progress));
    const amount = smooth(clamp(progress - index));
    const from = targets[index], to = targets[Math.min(6, index + 1)];
    for (let i = 0; i < positions.length; i++) positions[i] = mix(from[i], to[i], amount);
    positionAttribute.needsUpdate = true;
    // Long links dissolve as the same particles travel into their next form.
    for (let edge = 0; edge < indices.length; edge += 2) {
      const a = indices[edge] * 3, b = indices[edge + 1] * 3;
      const length = Math.hypot(positions[a] - positions[b], positions[a + 1] - positions[b + 1], positions[a + 2] - positions[b + 2]);
      const strength = 1 - smooth(clamp((length - 0.18) / 0.75));
      for (let axis = 0; axis < 3; axis++) {
        wirePositions[edge * 3 + axis] = positions[a + axis];
        wirePositions[(edge + 1) * 3 + axis] = positions[b + axis];
        wireColors[edge * 3 + axis] = colors[a + axis] * strength;
        wireColors[(edge + 1) * 3 + axis] = colors[b + axis] * strength;
      }
    }
    wirePositionAttribute.needsUpdate = true;
    wireColorAttribute.needsUpdate = true;

    const desktop = width >= 1100;
    const departing = desktop ? 0 : smooth(clamp(scrollY / Math.max(1, stageBox.top + stageBox.height * 0.55)));
    const centerX = mix(stageBox.left + stageBox.width / 2, width * 0.82, departing);
    const centerY = desktop ? stageBox.top + stageBox.height / 2 : mix(stageBox.top + stageBox.height / 2 - scrollY, height * 0.38, departing);
    sculpture.position.set((centerX / width - 0.5) * worldHeight * camera.aspect, (0.5 - centerY / height) * worldHeight, 0);
    const fit = Math.min(stageBox.width, stageBox.height - 90) / height * worldHeight / 4.3;
    const ambientScale = worldHeight * Math.min(width, height) * 0.65 / height / 4.3;
    sculpture.scale.setScalar(mix(fit, ambientScale, departing) * (reduced ? 1 : 1 + Math.sin(time * 0.65) * 0.018));
    if (!reduced) {
      sculpture.rotation.x = mix(sculpture.rotation.x, 0.3 + Math.sin(time * 0.12) * 0.12 + state.py * 0.07, damping);
      sculpture.rotation.y = mix(sculpture.rotation.y, -0.5 + Math.sin(time * 0.16) * 0.28 + state.px * 0.14, damping);
      sculpture.rotation.z = -0.18 + Math.sin(time * 0.18) * 0.07;
    }
    const opacity = mix(1, 0.16, departing);
    pointMaterial.opacity = opacity * 0.95;
    wireMaterial.opacity = opacity * (state.hv >= 0 ? 0.29 : 0.2);
    renderer.render(scene, camera);
    canvas.dataset.scene = String(index);
    canvas.dataset.progress = progress.toFixed(3);
    if (running && !reduced) frame = requestAnimationFrame(draw);
  }
  function resume() {
    cancelAnimationFrame(frame);
    running = !document.hidden && !lost;
    previous = performance.now();
    if (running) draw(previous);
  }
  addEventListener('resize', resize);
  addEventListener('graphicsmotionchange', resume);
  addEventListener('scroll', () => { if ((motion.matches || state.paused) && !document.hidden && !lost) draw(performance.now()); }, { passive: true });
  document.addEventListener('visibilitychange', resume);
  motion.addEventListener('change', resume);
  canvas.addEventListener('webglcontextlost', event => { event.preventDefault(); lost = true; running = false; cancelAnimationFrame(frame); });
  canvas.addEventListener('webglcontextrestored', () => { lost = false; resume(); });
  new ResizeObserver(resize).observe(stage);
  document.fonts.ready.then(resize);
  resize();
  resume();
}
