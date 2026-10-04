import * as THREE from 'three';
import { buildDiagram } from './diagrams.js';

export function createGraphics({ canvas, stage, state, motion }) {
  let renderer;
  const stages = [...document.querySelectorAll('[data-diagram]')];
  try {
    renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: 'low-power' });
  } catch {
    canvas.hidden = true;
    stages.forEach(element => element.classList.add('graphics-unavailable'));
    return;
  }
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
  camera.position.z = 10;
  const diagrams = stages.map(element => ({ element, ...buildDiagram(element.dataset.diagram) }));
  diagrams.forEach(({ group }) => { scene.add(group); group.visible = false; });
  const worldHeight = 2 * Math.tan(THREE.MathUtils.degToRad(20)) * 10;
  const clamp = x => Math.max(0, Math.min(1, x));
  const smooth = x => x * x * (3 - 2 * x);
  let width = 1, height = 1, time = 0, previous = 0, frame, lost = false;
  let pointerX = 0, pointerY = 0;

  function resize() {
    width = innerWidth; height = innerHeight;
    renderer.setPixelRatio(Math.min(devicePixelRatio, width < 700 ? 1.5 : 2));
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    resume();
  }
  function draw(now) {
    const dt = Math.min((now - previous) / 1000 || 0, 0.05);
    previous = now;
    const still = motion.matches || state.paused;
    if (!still) time += dt;
    const damping = 1 - Math.exp(-dt * 6);
    pointerX += ((still ? 0 : state.px) - pointerX) * damping;
    pointerY += ((still ? 0 : state.py) - pointerY) * damping;
    renderer.setScissorTest(false);
    renderer.clear();
    renderer.setScissorTest(true);
    diagrams.forEach(({ element, group, update }, index) => {
      const rect = element.getBoundingClientRect();
      if (rect.bottom < 0 || rect.top > height) return;
      const top = rect.top + 48, bottom = rect.bottom - 56;
      const centerY = (top + bottom) / 2;
      const journey = (height / 2 - centerY) / (height * 0.8);
      const exit = smooth(clamp(journey));
      const entry = smooth(clamp(-journey));
      const zoom = still ? 1 : 1 + exit * 0.65 + entry * 0.2;
      const spread = element.dataset.diagram === 'transformer' ? 1 : Math.min(1.8, Math.max(1, rect.width / 650));
      const fit = Math.min(rect.width / (6.6 * spread), (bottom - top) / 6.5) * worldHeight / height;
      group.position.set(((rect.left + rect.width / 2) / width - 0.5) * worldHeight * camera.aspect, (0.5 - centerY / height) * worldHeight, 0);
      group.scale.set(fit * zoom * spread, fit * zoom, fit * zoom);
      group.rotation.set(still ? 0 : pointerY * 0.018, still ? 0 : pointerX * 0.035, 0);
      group.visible = true;
      update(time);
      const opacity = still ? 1 : 1 - exit * 0.55;
      group.traverse(object => {
        if (object.isSprite) {
          if (object.userData.labelWidth === undefined) object.userData.labelWidth = object.scale.x;
          object.scale.x = object.userData.labelWidth / spread;
        }
        if (object.geometry?.type === 'SphereGeometry') object.scale.x = object.scale.y / spread;
        if (!object.material) return;
        if (object.userData.baseOpacity === undefined) object.userData.baseOpacity = object.material.opacity;
        if (object.isSprite || object.isLine || object.isLineSegments) object.material.opacity = object.userData.baseOpacity * opacity;
      });
      const clipTop = Math.max(0, top), clipBottom = Math.min(height, bottom);
      if (clipBottom > clipTop) {
        renderer.setScissor(Math.max(0, rect.left), height - clipBottom, Math.min(rect.width, width), clipBottom - clipTop);
        renderer.render(scene, camera);
      }
      group.visible = false;
      canvas.dataset.scene = element.dataset.diagram;
      canvas.dataset.progress = String(index);
      element.dataset.zoom = zoom.toFixed(3);
    });
    canvas.dataset.scroll = String(Math.round(scrollY));
    renderer.setScissorTest(false);
    if (!still && !document.hidden && !lost) frame = requestAnimationFrame(draw);
  }
  function resume() {
    cancelAnimationFrame(frame);
    if (document.hidden || lost) return;
    previous = performance.now();
    draw(previous);
  }
  addEventListener('resize', resize);
  addEventListener('graphicsmotionchange', resume);
  addEventListener('scroll', () => { if (motion.matches || state.paused) resume(); }, { passive: true });
  document.addEventListener('visibilitychange', resume);
  motion.addEventListener('change', resume);
  canvas.addEventListener('webglcontextlost', event => { event.preventDefault(); lost = true; cancelAnimationFrame(frame); });
  canvas.addEventListener('webglcontextrestored', () => { lost = false; resume(); });
  const observer = new ResizeObserver(resize);
  stages.forEach(element => observer.observe(element));
  document.fonts.ready.then(resize);
  resize();
}
