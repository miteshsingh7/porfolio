import * as THREE from 'three';
import { buildModels, smooth } from './diagrams.js';

export function createGraphics({ canvas, state, motion }) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  } catch {
    canvas.hidden = true;
    return;
  }
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, .1, 100);
  camera.position.z = 7;
  const halfH = Math.tan(THREE.MathUtils.degToRad(22.5)) * 7;

  const models = buildModels();
  const entries = [...document.querySelectorAll('[data-diagram]')]
    .map((el, i) => ({ el, i, model: models[el.dataset.diagram], ready: false, x: 0, y: 0, z: 0, s: 1 }))
    .filter(entry => entry.model);
  entries.forEach(entry => scene.add(entry.model.group));

  const dust = new Float32Array(400 * 3);
  for (let i = 0; i < dust.length; i++) dust[i] = (Math.random() - .5) * (i % 3 === 2 ? 10 : 16);
  const dustGeometry = new THREE.BufferGeometry();
  dustGeometry.setAttribute('position', new THREE.BufferAttribute(dust, 3));
  const dustPoints = new THREE.Points(dustGeometry, new THREE.PointsMaterial({ color: 0xF2F0EB, size: .025, transparent: true, opacity: .3 }));
  scene.add(dustPoints);

  let W = 1, H = 1;
  const resize = () => {
    W = innerWidth;
    H = innerHeight;
    renderer.setSize(W, H, false);
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
  };
  addEventListener('resize', resize);
  resize();

  // Continuous index of the diagram in focus; the hand-off between two diagrams happens across the middle 40% of the scroll gap between them.
  const focus = () => {
    const centers = entries.map(entry => {
      entry.rect = entry.el.getBoundingClientRect();
      return entry.rect.top + entry.rect.height / 2;
    });
    const v = H / 2;
    if (v <= centers[0]) return 0;
    for (let i = 0; i < centers.length - 1; i++) {
      if (v < centers[i + 1]) return i + smooth(((v - centers[i]) / (centers[i + 1] - centers[i]) - .3) / .4);
    }
    return centers.length - 1;
  };

  const clock = new THREE.Clock();
  let speed = 1, time = 0;
  (function frame() {
    const reduce = motion.matches, frozen = reduce || state.paused;
    const dt = frozen ? 0 : Math.min(clock.getDelta(), .05);
    if (frozen) clock.getDelta();
    time += dt;
    speed += ((state.hv >= 0 ? 2.6 : 1) - speed) * .06;
    const f = focus(), aspect = camera.aspect, active = Math.round(f);

    entries.forEach((entry, i) => {
      const { group } = entry.model, d = f - i, visible = 1 - Math.abs(d);
      if (visible <= 0) { group.visible = false; entry.ready = false; return; }
      group.visible = true;

      const r = entry.rect;
      const dockX = ((r.left + r.width / 2) / W * 2 - 1) * halfH * aspect;
      const rawY = -((r.top + r.height / 2) / H * 2 - 1) * halfH;
      const offscreen = Math.min(1, Math.max(0, Math.abs(rawY) - halfH * .55) / (halfH * 1.1));
      const dock = Math.max(-halfH * .5, Math.min(halfH * .5, rawY)) * (1 - Math.abs(d));
      const fit = Math.min((r.width / W) * 2 * halfH * aspect / entry.model.width, (r.height / H) * 2 * halfH / entry.model.height);

      let z = 0, zoom = 1, fade = visible;
      if (!reduce) {
        if (d > 0) { z = smooth(d) * 4.2; zoom = 1 + d * .5; fade = Math.pow(1 - d, 1.6); }
        else if (d < 0) { z = d * 10; fade = 1 + d; }
      }
      const targetScale = fit * zoom;
      if (!entry.ready) { entry.x = dockX; entry.y = dock; entry.z = z; entry.s = targetScale; entry.ready = true; }
      entry.x += (dockX - entry.x) * .12;
      entry.y += (dock - entry.y) * .14;
      entry.z += (z - entry.z) * .14;
      entry.s += (targetScale - entry.s) * .12;
      group.position.set(entry.x, entry.y, entry.z);
      group.scale.setScalar(entry.s);

      const sway = reduce ? -.4 : -.45 + Math.sin(time * .3 + i * 1.7) * .25;
      group.rotation.y += (sway + state.px * .3 + d * .9 - group.rotation.y) * .05;
      group.rotation.x += (.12 + state.py * .1 - group.rotation.x) * .05;

      entry.model.update({ dt, t: time, boost: i === active ? speed : 1 });
      const opacity = fade * (1 - .6 * offscreen);
      entry.model.materials.forEach(material => { material.opacity = material.userData.base * opacity; });
    });

    dustPoints.rotation.y = f * .15;
    camera.position.x += (state.px * .4 - camera.position.x) * .04;
    camera.position.y += (-state.py * .25 - camera.position.y) * .04;
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  })();
}
