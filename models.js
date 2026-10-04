import * as THREE from 'three';

const LI = new THREE.Color('#B8F500'), VI = new THREE.Color('#8B5CF6'), GR = new THREE.Color('#8A8A8F'), WH = new THREE.Color('#F2F0EB');
const V = (x, y, z = 0) => new THREE.Vector3(x, y, z);
const BX = new THREE.BoxGeometry(1, 1, 1), SP = new THREE.SphereGeometry(.035, 8, 8), PL = new THREE.PlaneGeometry(1, 1), BG = new THREE.BoxGeometry(1, 1, 1);
BG.translate(0, .5, 0);
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const op = (obj, v) => { obj.material.userData.b = v; };
const pathOf = pts => { const p = new THREE.CurvePath(); for (let i = 0; i < pts.length - 1; i++) p.add(new THREE.LineCurve3(pts[i], pts[i + 1])); return p; };

function kit() {
  const root = new THREE.Group(), mats = [];
  const reg = (m, o) => { m.transparent = true; m.depthWrite = false; m.userData.b = o; m.opacity = o; mats.push(m); return m; };
  const add = o => { root.add(o); return o; };
  const LM = (c, o) => reg(new THREE.LineBasicMaterial({ color: c.clone() }), o);
  const MM = (c, o, w) => reg(new THREE.MeshBasicMaterial({ color: c.clone(), wireframe: !!w }), o);
  const line = (p, c, o, loop) => add(new (loop ? THREE.LineLoop : THREE.Line)(new THREE.BufferGeometry().setFromPoints(p), LM(c, o)));
  const rect = (x0, y0, x1, y1, c, o, z = 0) => line([V(x0, y0, z), V(x1, y0, z), V(x1, y1, z), V(x0, y1, z)], c, o, true);
  const ring = (x, y, r, c, o, z = 0) => line(Array.from({ length: 24 }, (_, i) => V(x + Math.cos(i / 24 * 6.283) * r, y + Math.sin(i / 24 * 6.283) * r, z)), c, o, true);
  const mesh = (g, c, o, w) => add(new THREE.Mesh(g, MM(c, o, w)));
  const segs = (pts, c, o) => add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), LM(c, o)));
  const pulses = (n, c = WH, size = .09) => {
    const a = new Float32Array(n * 3).fill(-99), g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(a, 3));
    const p = add(new THREE.Points(g, reg(new THREE.PointsMaterial({ color: c, size }), .95)));
    p.frustumCulled = false;
    return { set(i, v) { a.set([v.x, v.y, v.z], i * 3); }, hide(i) { a.set([0, -99, 0], i * 3); }, done() { g.attributes.position.needsUpdate = true; } };
  };
  const label = (text, x, y, z = 0, color = '#8A8A8F', size = .15) => {
    const cv = document.createElement('canvas'), ctx = cv.getContext('2d'), f = 44, font = `500 ${f}px "JetBrains Mono", ui-monospace, monospace`;
    ctx.font = font;
    cv.width = Math.ceil(ctx.measureText(text).width) + 8; cv.height = Math.ceil(f * 1.3);
    ctx.font = font; ctx.fillStyle = color; ctx.textBaseline = 'middle'; ctx.fillText(text, 4, cv.height / 2);
    const tex = new THREE.CanvasTexture(cv); tex.minFilter = THREE.LinearFilter;
    const s = add(new THREE.Sprite(reg(new THREE.SpriteMaterial({ map: tex }), .85)));
    s.scale.set(size * cv.width / cv.height, size, 1); s.position.set(x, y, z);
    return s;
  };
  return { root, mats, line, rect, ring, mesh, segs, pulses, label };
}

/* Decoder transformer: embeddings -> 2 x [masked multi-head attention + FFN, residuals] -> softmax -> next token fed back */
function transformer() {
  const K = kit(), { root, line, rect, ring, mesh, segs } = K;
  root.position.y = -.1;
  const tx = k => -1.25 + k * .5;
  const toks = [];
  for (let k = 0; k < 6; k++) { const m = mesh(BX, LI, .9, true); m.scale.setScalar(.2); m.position.set(tx(k), -2.4, 0); m.u = { slot: k }; toks.push(m); line([V(tx(k), -2.28), V(tx(k), -2.05)], GR, .4); }
  rect(-1.6, -2.05, 1.6, -1.9, VI, .7); line([V(0, -1.9), V(0, -1.55)], GR, .35);
  const blocks = [];
  for (let b = 0; b < 2; b++) {
    const y0 = -1.55 + b * 1.75, B = { y0, cells: [], nodes: [] };
    rect(-1.6, y0, 1.6, y0 + .85, VI, .6);
    for (let h = 0; h < 4; h++) {
      const z = (h - 1.5) * .1, hx = -1.2 + h * .8;
      rect(hx - .33, y0 + .08, hx + .33, y0 + .78, GR, .3, z);
      for (let r = 0; r < 4; r++) for (let c = 0; c <= r; c++) { const m = mesh(PL, LI, .1); m.scale.setScalar(.13); m.position.set(hx - .225 + c * .15, y0 + .68 - r * .15, z); B.cells.push({ m, sd: Math.random() * 6.3 }); }
    }
    rect(-1.6, y0 + .92, 1.6, y0 + .98, GR, .6);
    const rows = [[4, y0 + 1.06], [8, y0 + 1.24], [4, y0 + 1.42]], R2 = [], pts = [];
    rows.forEach(([n, y]) => { const row = []; for (let i = 0; i < n; i++) { const x = (i - (n - 1) / 2) * (2.4 / (n - 1)), m = mesh(SP, LI, .8); m.position.set(x, y, 0); B.nodes.push(m); row.push(V(x, y)); } R2.push(row); });
    for (let r = 0; r < 2; r++) R2[r].forEach(p => R2[r + 1].forEach(q => pts.push(p, q)));
    B.ffn = segs(pts, LI, .05);
    rect(-1.6, y0 + 1.5, 1.6, y0 + 1.56, GR, .6);
    line([V(-1.6, y0 - .1), V(-1.75, y0 - .1), V(-1.75, y0 + 1.53), V(-1.6, y0 + 1.53)], GR, .5);
    [y0 + .95, y0 + 1.53].forEach(y => ring(-1.75, y, .07, LI, .9));
    blocks.push(B);
  }
  line([V(0, 1.76), V(0, 1.86)], GR, .4); rect(-1.6, 1.86, 1.6, 1.92, VI, .7);
  const bars = [];
  for (let k = 0; k < 6; k++) { const m = mesh(BG, VI, .8, true); m.scale.set(.2, .05, .2); m.position.set(tx(k), 2, 0); bars.push(m); }
  const FB = [V(0, 2.45), V(2.15, 2.45), V(2.15, -2.4), V(1.6, -2.4)], path = pathOf(FB);
  line(FB, GR, .3); line([V(0, 2.0), V(0, 2.45)], GR, .3);
  const P = K.pulses(3);
  let cyc = .35, speed = 1, hm = 0, pk = 2, last = 0;
  return {
    ...K, size: { w: 4.5, h: 5.3 },
    update(t, dt, hov) {
      speed += ((hov ? 2.6 : 1) - speed) * .06; hm += ((hov ? 1 : 0) - hm) * .06;
      cyc += dt * .14 * speed;
      const ph = cyc % 1, cn = Math.floor(cyc);
      if (cn !== last) { last = cn; pk = Math.random() * 6 | 0; toks.forEach(m => { m.u.slot--; if (m.u.slot < 0) { m.u.slot = 5; m.position.x = tx(6); op(m, 0); } }); }
      const py = ph < .78 ? -2.3 + ph / .78 * 4.25 : -99, fb = ph > .82 ? path.getPointAt((ph - .82) / .18) : V(0, -99);
      P.set(0, V(0, py)); P.set(1, V(-1.75, py)); P.set(2, fb); P.done();
      blocks.forEach(B => {
        const a1 = Math.max(0, 1 - Math.abs(py - (B.y0 + .45)) / .7), a2 = Math.max(0, 1 - Math.abs(py - (B.y0 + 1.25)) / .6);
        B.cells.forEach(c => { op(c.m, .08 + a1 * (.15 + .7 * (.5 + .5 * Math.sin(t * 9 + c.sd)))); c.m.material.color.copy(LI).lerp(VI, hm * .7); });
        B.nodes.forEach(n => op(n, .25 + a2 * .75)); op(B.ffn, .04 + a2 * .5);
      });
      bars.forEach((m, i) => { const th = ph > .7 ? (i === pk ? .6 : .08 + .12 * Math.abs(Math.sin(i * 2.1 + cn))) : .05; m.scale.y += (th - m.scale.y) * .1; m.material.color.copy(i === pk && ph > .7 ? LI : VI); });
      toks.forEach(m => { m.position.x += (tx(m.u.slot) - m.position.x) * .08; op(m, m.material.userData.b + (.9 - m.material.userData.b) * .06); m.rotation.y += dt * .6; });
    },
  };
}

/* Feed-forward network in 3D: forward activations sweep input -> output, then gradients flow back */
function neural() {
  const K = kit(), { line, mesh, segs, label } = K;
  const X = [-2.05, -.8, .45, 1.55], spec = [[3, 3], [4, 4], [4, 4], [3, 1]], gap = .62;
  const layers = spec.map(([ny, nz], l) => {
    const nodes = [];
    for (let i = 0; i < ny; i++) for (let j = 0; j < nz; j++) {
      const p = V(X[l], (i - (ny - 1) / 2) * gap, (j - (nz - 1) / 2) * gap), m = mesh(SP, l === 3 ? VI : LI, .5);
      m.scale.setScalar(2.4); m.position.copy(p); nodes.push({ m, p, sd: Math.random() * 6.3 });
    }
    const y = (ny - 1) / 2 * gap + .28, z = (nz - 1) / 2 * gap + .28;
    line([V(X[l], -y, -z), V(X[l], y, -z), V(X[l], y, z), V(X[l], -y, z)], GR, .35, true);
    return nodes;
  });
  const gaps = [0, 1, 2].map(l => {
    const pts = [], edges = [];
    layers[l].forEach(a => layers[l + 1].forEach(b => { pts.push(a.p, b.p); edges.push([a.p, b.p]); }));
    return { seg: segs(pts, LI, .06), edges };
  });
  const loss = mesh(BX, VI, .6, true); loss.scale.setScalar(.34); loss.position.set(2.3, 0, 0);
  layers[3].forEach(n => line([n.p, V(2.13, 0)], GR, .3));
  ['INPUT', 'HIDDEN', 'HIDDEN', 'OUTPUT'].forEach((t, l) => label(t, X[l], l === 0 || l === 3 ? -1.3 : -1.62));
  label('LOSS', 2.3, -.42);
  const fwd = label('FORWARD PASS', 0, 1.6, 0, '#B8F500', .17), bwd = label('BACKPROPAGATION', 0, 1.6, 0, '#8B5CF6', .17);
  const N = 18, P = K.pulses(N), route = [];
  const pick = () => { for (let i = 0; i < N; i++) { const g = i % 3; route[i] = { g, e: gaps[g].edges[Math.random() * gaps[g].edges.length | 0] }; } };
  pick();
  let cyc = .2, last = 0, speed = 1;
  const tmp = V(0, 0);
  return {
    ...K, size: { w: 4.9, h: 4.0 },
    update(t, dt, hov) {
      speed += ((hov ? 2.2 : 1) - speed) * .06;
      cyc += dt * .15 * speed;
      const ph = cyc % 1;
      if (Math.floor(cyc) !== last) { last = Math.floor(cyc); pick(); }
      let w, back = false;
      if (ph < .48) w = ph / .48 * 3; else if (ph < .56) w = 3; else { w = 3 - (ph - .56) / .44 * 3; back = true; }
      layers.forEach((L, l) => {
        const a = Math.max(0, 1 - Math.abs(w - l) * 1.1);
        L.forEach(n => { op(n.m, .28 + a * .72 * (.6 + .4 * Math.sin(t * 6 + n.sd))); n.m.material.color.copy(back ? VI : l === 3 ? VI : LI); n.m.scale.setScalar(2.4 + a * 1.1); });
      });
      gaps.forEach((G, g) => { const a = Math.max(0, 1 - Math.abs(w - (g + .5)) * 1.3); op(G.seg, .05 + a * .32); G.seg.material.color.copy(back ? VI : LI); });
      route.forEach((r, i) => { const s = w - r.g; if (s > 0 && s < 1 && ph < .48 || s > 0 && s < 1 && back) P.set(i, tmp.copy(r.e[0]).lerp(r.e[1], s)); else P.hide(i); });
      P.done();
      op(loss, ph > .44 && ph < .62 ? 1 : .45); loss.rotation.y += dt * .8; loss.rotation.x += dt * .4;
      op(fwd, back ? 0 : .9); op(bwd, back ? .9 : 0);
    },
  };
}

/* Retrieval-augmented generation: ingest -> embed into a vector store -> query retrieves top-k -> LLM generates a grounded answer */
function rag() {
  const K = kit(), { line, rect, mesh, label } = K;
  const DX = -1.5, DY = 1.75;
  for (let k = 0; k < 3; k++) rect(DX - .42, DY - .5, DX + .42, DY + .5, k === 2 ? LI : GR, k === 2 ? .7 : .35, (k - 1) * .2);
  for (let i = 0; i < 5; i++) line([V(DX - .28, DY + .3 - i * .15, .2), V(DX + .28 - (i % 2) * .16, DY + .3 - i * .15, .2)], GR, .55);
  label('DOCUMENTS', DX, DY + .7);
  const VC = V(0, -.55, 0);
  const store = mesh(BX, VI, .3, true); store.scale.set(1.9, 1.5, 1.5); store.position.copy(VC);
  label('VECTOR STORE', 0, -1.52);
  const pts = Array.from({ length: 64 }, () => {
    const p = V(VC.x + (Math.random() - .5) * 1.6, VC.y + (Math.random() - .5) * 1.2, (Math.random() - .5) * 1.2), m = mesh(SP, GR, .5);
    m.scale.setScalar(1.4); m.position.copy(p); return { m, p, glow: 0 };
  });
  const ingest = pathOf([V(DX, DY - .6), V(DX, .9), V(-.35, .9), V(-.35, -.1)]);
  line([V(DX, DY - .6), V(DX, .9), V(-.35, .9), V(-.35, .2)], GR, .25);
  label('CHUNK + EMBED', -.92, 1.08);
  const QP = V(-1.7, -2.15, 0), q = mesh(BX, LI, .9, true); q.scale.setScalar(.26); q.position.copy(QP);
  label('QUERY', QP.x, QP.y - .32);
  line([V(QP.x, QP.y + .15), V(QP.x, -.55), V(-.95, -.55)], GR, .3);
  const LP = V(1.6, .8, 0), llm = [];
  for (let k = 0; k < 4; k++) llm.push(rect(LP.x - .45, LP.y - .55, LP.x + .45, LP.y + .55, VI, .45, (k - 1.5) * .18));
  label('LLM', LP.x, LP.y - .74);
  line([V(.95, -.55), V(LP.x, -.55), V(LP.x, LP.y - .58)], GR, .3); line([V(LP.x, LP.y + .58), V(LP.x, 1.85)], GR, .3);
  const ans = Array.from({ length: 5 }, (_, k) => { const m = mesh(BX, LI, 0, true); m.scale.setScalar(.17); m.position.set(LP.x - .6 + k * .3, 2.05, 0); return m; });
  label('ANSWER', LP.x, 2.4);
  const chunks = Array.from({ length: 4 }, () => { const m = mesh(BX, LI, 0, true); m.scale.setScalar(.15); return m; });
  const links = Array.from({ length: 3 }, () => { const l = line([V(), V()], LI, 0); l.frustumCulled = false; return l; });
  const setLink = (l, a, b) => { const p = l.geometry.attributes.position; p.setXYZ(0, a.x, a.y, a.z); p.setXYZ(1, b.x, b.y, b.z); p.needsUpdate = true; };
  const P = K.pulses(1);
  const qPath = pathOf([V(QP.x, QP.y + .15), V(QP.x, -.55), V(-.6, -.55)]);
  let cyc = .3, last = -1, speed = 1, qPoint = VC.clone(), near = [], targets = [], ctx = [];
  const plan = () => {
    qPoint = V(VC.x + (Math.random() - .5) * .9, VC.y + (Math.random() - .5) * .7, (Math.random() - .5) * .7);
    near = [...pts].sort((a, b) => a.p.distanceTo(qPoint) - b.p.distanceTo(qPoint)).slice(0, 3);
    targets = chunks.map(() => pts[Math.random() * pts.length | 0]);
    ctx = near.map(n => new THREE.QuadraticBezierCurve3(n.p, V((n.p.x + LP.x) / 2, n.p.y + 1.4, n.p.z), LP));
  };
  const tmp = V();
  return {
    ...K, size: { w: 4.5, h: 5.2 },
    update(t, dt, hov) {
      speed += ((hov ? 2.2 : 1) - speed) * .06;
      cyc += dt * .11 * speed;
      const ph = cyc % 1;
      if (Math.floor(cyc) !== last) { last = Math.floor(cyc); plan(); }
      chunks.forEach((m, k) => {
        m.rotation.y += dt * 1.2; m.rotation.x += dt * .7;
        const s = (ph - k * .045) / .2;
        if (ph < .32 && s > 0 && s < 1) { m.position.copy(ingest.getPointAt(s)); if (s > .78) m.position.lerp(targets[k].p, (s - .78) / .22); op(m, .9); }
        else if (ph > .62 && ph < .8 && k < 3) { m.position.copy(ctx[k].getPoint(clamp((ph - .62 - k * .02) / .15))); op(m, .9); }
        else op(m, 0);
        if (ph < .32 && s >= 1 && s < 1.15) targets[k].glow = 1;
      });
      const qs = (ph - .32) / .14;
      if (qs > 0 && qs < 1) P.set(0, qPath.getPointAt(qs).lerp(qPoint, clamp(qs * 1.6 - .6))); else P.hide(0);
      P.done();
      op(q, qs > 0 && qs < 1.2 ? 1 : .55); q.rotation.y += dt * .9;
      const r = clamp((ph - .46) / .08) * (ph < .8 ? 1 : clamp(1 - (ph - .8) / .06));
      links.forEach((l, k) => { setLink(l, qPoint, near[k].p); op(l, r * .85); if (r > .5) near[k].glow = 1; });
      pts.forEach(p => { p.glow = Math.max(0, p.glow - dt * .9); p.m.material.color.copy(GR).lerp(LI, p.glow); op(p.m, .4 + p.glow * .6); p.m.scale.setScalar(1.4 + p.glow * 1.4); });
      const act = clamp((ph - .74) / .06) * clamp(1 - (ph - .96) / .04);
      llm.forEach((l, k) => op(l, .35 + act * (.4 + .25 * Math.sin(t * 7 + k))));
      ans.forEach((m, k) => { const on = ph > .8 + k * .03 && ph < .985; op(m, on ? .9 : 0); m.rotation.y += dt * .6; });
      store.rotation.y = Math.sin(t * .4) * .05;
      tmp.set(0, 0, 0);
    },
  };
}

/* LoRA: frozen W0 in parallel with trainable low-rank A (d -> r) and B (r -> d); h = W0x + BAx. Gradients update only A and B. */
function lora() {
  const K = kit(), { line, rect, ring, mesh, label } = K;
  const tx = k => -1.25 + k * .5;
  const xin = Array.from({ length: 6 }, (_, k) => { const m = mesh(BX, LI, .85, true); m.scale.setScalar(.18); m.position.set(tx(k), -2.4, 0); return m; });
  rect(-1.6, -2.08, 1.6, -1.98, GR, .5); label('x', -1.9, -2.4, 0, '#F2F0EB', .2);
  const WX = -.85, WY = -.25, wc = [];
  rect(WX - .78, WY - .78, WX + .78, WY + .78, GR, .6);
  for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) { const m = mesh(PL, GR, .12); m.scale.setScalar(.2); m.position.set(WX - .6 + j * .24, WY - .6 + i * .24, 0); wc.push({ m, sd: Math.random() * 6.3 }); }
  label('W0  FROZEN', WX - .35, WY - .98);
  const AX = 1.2, Z = .5, ac = [], bc = [];
  line([V(AX - .65, -1.25, Z), V(AX + .65, -1.25, Z), V(AX + .17, -.38, Z), V(AX - .17, -.38, Z)], LI, .7, true);
  line([V(AX - .17, -.02, Z), V(AX + .17, -.02, Z), V(AX + .65, .85, Z), V(AX - .65, .85, Z)], LI, .7, true);
  const cells = (rows, out) => rows.forEach(([y, n]) => { for (let i = 0; i < n; i++) { const m = mesh(PL, LI, .2); m.scale.setScalar(.15); m.position.set(AX + (i - (n - 1) / 2) * .2, y, Z); out.push({ m, y, sd: Math.random() * 6.3 }); } });
  cells([[-1.1, 5], [-.88, 4], [-.67, 3], [-.48, 1]], ac);
  cells([[.08, 1], [.28, 3], [.49, 4], [.7, 5]], bc);
  const rn = [-.07, .07].map(dx => { const m = mesh(SP, LI, .9); m.scale.setScalar(2); m.position.set(AX + dx, -.2, Z); return m; });
  label('A', AX + .88, -.8, Z, '#B8F500', .2); label('B', AX + .88, .42, Z, '#B8F500', .2); label('r', AX + .32, -.2, Z, '#F2F0EB', .16);
  label('TRAINABLE', AX + .95, 1.0, Z, '#B8F500', .13);
  const SX = .2, SY = 1.5;
  ring(SX, SY, .12, LI, .9); line([V(SX - .07, SY), V(SX + .07, SY)], LI, .9); line([V(SX, SY - .07), V(SX, SY + .07)], LI, .9);
  const fwdW = pathOf([V(WX, -1.98), V(WX, WY - .78), V(WX, WY + .78), V(WX, SY), V(SX - .12, SY)]);
  const fwdA = pathOf([V(AX, -1.98, 0), V(AX, -1.25, Z), V(AX, .85, Z), V(AX, SY, 0), V(SX + .12, SY)]);
  const grad = pathOf([V(SX, 1.95), V(SX, SY + .12), V(SX + .12, SY), V(AX, SY, 0), V(AX, .85, Z), V(AX, -1.25, Z)]);
  line([V(WX, -1.98), V(WX, WY - .78)], GR, .4); line([V(WX, WY + .78), V(WX, SY), V(SX - .12, SY)], GR, .4);
  line([V(AX, -1.98, 0), V(AX, -1.25, Z)], GR, .4); line([V(AX, .85, Z), V(AX, SY, 0), V(SX + .12, SY)], GR, .4);
  line([V(SX, SY + .12), V(SX, 1.95)], GR, .4); rect(-1.6, 1.95, 1.6, 2.02, VI, .7);
  const hout = Array.from({ length: 6 }, (_, k) => { const m = mesh(BX, VI, .5, true); m.scale.setScalar(.18); m.position.set(tx(k), 2.3, 0); return m; });
  label('h = W0x + BAx', 0, 2.68, 0, '#F2F0EB', .16);
  const P = K.pulses(2), G = K.pulses(1, VI, .12);
  let cyc = .1, speed = 1;
  return {
    ...K, size: { w: 4.5, h: 5.5 },
    update(t, dt, hov) {
      speed += ((hov ? 2.2 : 1) - speed) * .06;
      cyc += dt * .14 * speed;
      const ph = cyc % 1, fs = ph / .5, bs = (ph - .6) / .4;
      if (fs < 1) { P.set(0, fwdW.getPointAt(fs)); P.set(1, fwdA.getPointAt(fs)); } else { P.hide(0); P.hide(1); }
      if (bs > 0 && bs < 1) G.set(0, grad.getPointAt(bs)); else G.hide(0);
      P.done(); G.done();
      const fy = fs < 1 ? -1.98 + fs * 3.5 : -99, gy = bs > 0 && bs < 1 ? 1.95 - bs * 3.2 : -99;
      wc.forEach(c => { const a = Math.max(0, 1 - Math.abs(fy - c.m.position.y) / .5); op(c.m, .1 + a * .35 * (.6 + .4 * Math.sin(t * 8 + c.sd))); });
      [...ac, ...bc].forEach(c => {
        const a = Math.max(0, 1 - Math.abs(fy - c.y) / .45), g = Math.max(0, 1 - Math.abs(gy - c.y) / .45);
        op(c.m, .16 + Math.max(a, g) * .8 * (.6 + .4 * Math.sin(t * 9 + c.sd)));
        c.m.material.color.copy(LI).lerp(VI, g);
      });
      rn.forEach(m => op(m, .5 + (Math.abs(fy + .2) < .3 || Math.abs(gy + .2) < .3 ? .5 : 0)));
      const out = ph > .48 && ph < .7;
      hout.forEach((m, k) => { m.rotation.y += dt * .6; op(m, out ? .95 : .4); m.material.color.copy(out && k % 2 ? LI : VI); });
      xin.forEach(m => { m.rotation.y += dt * .6; op(m, ph < .1 ? 1 : .7); });
    },
  };
}

export function buildModels() {
  return { transformer: transformer(), neural: neural(), rag: rag(), lora: lora() };
}
