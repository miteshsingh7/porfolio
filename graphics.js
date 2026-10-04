import * as THREE from 'three';
import { buildModels } from './models.js';

const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const ease = x => (x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
const smooth = x => x * x * (3 - 2 * x);
const lerp = (a, b, k) => a + (b - a) * k;

export function createGraphics({ canvas, state, motion }) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  } catch {
    canvas.hidden = true;
    return;
  }
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, .1, 100);
  camera.position.z = 7;
  const HALF = Math.tan(THREE.MathUtils.degToRad(22.5)) * 7;

  const slots = Object.entries(buildModels()).map(([key, model]) => {
    const group = new THREE.Group();
    group.add(model.root);
    group.visible = false;
    scene.add(group);
    return { key, model, group, fade: 0 };
  });

  const dustCount = 400, dust = new Float32Array(dustCount * 3);
  for (let i = 0; i < dust.length; i++) dust[i] = (Math.random() - .5) * (i % 3 === 2 ? 10 : 16);
  const dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute('position', new THREE.BufferAttribute(dust, 3));
  const dustPoints = new THREE.Points(dustGeo, new THREE.PointsMaterial({ color: 0xF2F0EB, size: .025, transparent: true, opacity: .3 }));
  scene.add(dustPoints);

  const sections = [...document.querySelectorAll('main section')];
  let W = 1, H = 1;
  const resize = () => { W = innerWidth; H = innerHeight; renderer.setSize(W, H, false); camera.aspect = W / H; camera.updateProjectionMatrix(); };
  addEventListener('resize', resize);
  resize();

  function placement() {
    const mid = H * .45;
    let section = sections[0];
    for (const s of sections) if (s.getBoundingClientRect().top <= mid) section = s;
    const stage = section.querySelector('[data-diagram]');
    const key = stage?.dataset.diagram || section.dataset.model || 'transformer';
    const size = slots.find(s => s.key === key).model.size;
    const hw = HALF * camera.aspect, mobile = W < 900;
    const side = { x: mobile ? 0 : hw * .48, y: 0, s: (mobile ? .6 : .78) * 2 * HALF / size.h, o: mobile ? .12 : .42 };
    if (!stage) return { key, ...side };
    const r = stage.getBoundingClientRect();
    const seen = clamp((Math.min(r.bottom, H) - Math.max(r.top, 0)) / Math.min(r.height, H));
    const v = smooth(mobile ? clamp(seen * 2 - 1) : seen);
    const fit = Math.min((r.height * .9 / H) * 2 * HALF / size.h, (r.width * .92 / W) * 2 * hw / size.w);
    const x = ((r.left + r.width / 2) / W * 2 - 1) * hw, y = -((r.top + r.height / 2) / H * 2 - 1) * HALF;
    return { key, x: lerp(side.x, x, v), y: lerp(side.y, y, v), s: lerp(side.s, fit, v), o: lerp(side.o, 1, v) };
  }

  const anchor = { x: 0, y: 0, s: 1, o: 1 };
  let active = null, t = 0, ry = -.45, rx = .12;
  const clock = new THREE.Clock();

  function frame() {
    const reduce = motion.matches, dt = Math.min(clock.getDelta(), .05), still = reduce || state.paused;
    if (!still) t += dt;
    const target = placement();

    if (target.key !== active) {
      if (active === null || reduce) slots.forEach(s => { s.fade = s.key === target.key ? 1 : 0; });
      if (active === null) Object.assign(anchor, target);
      active = target.key;
      canvas.dataset.scene = active;
    }

    const follow = reduce ? 1 : 1 - Math.exp(-dt * 9);
    ['x', 'y', 's', 'o'].forEach(k => { anchor[k] += (target[k] - anchor[k]) * follow; });
    ry += ((reduce ? -.4 : -.45 + Math.sin(t * .3) * .25) + state.px * .3 - ry) * .05;
    rx += (.12 + state.py * .1 - rx) * .05;

    slots.forEach(slot => {
      const on = slot.key === active;
      slot.fade = clamp(slot.fade + (on ? 1 : -1) * dt * 1.4);
      if (!on && slot.fade <= 0) { slot.group.visible = false; return; }
      slot.group.visible = true;
      const e = ease(slot.fade), off = 1 - e;
      slot.group.position.set(anchor.x, anchor.y, on ? -9 * off : 4.6 * off);
      slot.group.scale.setScalar(anchor.s * (on ? 1 - off * .35 : 1 + off * .25));
      slot.group.rotation.set(rx + (on ? .35 : -.25) * off, ry + (on ? -1.1 : 1.1) * off, (on ? .25 : -.2) * off);
      slot.model.update(t, still ? 0 : dt, state.hv >= 0);
      const alpha = anchor.o * (on ? e : e * e);
      slot.model.mats.forEach(m => { m.opacity = m.userData.b * alpha; });
    });

    dustPoints.rotation.y += still ? 0 : dt * .015;
    camera.position.x += (state.px * .4 - camera.position.x) * .04;
    camera.position.y += (-state.py * .25 - camera.position.y) * .04;
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
