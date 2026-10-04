import * as THREE from 'three';

export const LIME = new THREE.Color('#B8F500');
export const VIOLET = new THREE.Color('#8B5CF6');
export const GREY = new THREE.Color('#8A8A8F');
export const PAPER = new THREE.Color('#F2F0EB');

const V = (x, y, z = 0) => new THREE.Vector3(x, y, z);
const clamp01 = v => Math.min(1, Math.max(0, v));
export const smooth = v => { v = clamp01(v); return v * v * (3 - 2 * v); };

const BOX = new THREE.BoxGeometry(1, 1, 1);
const PLANE = new THREE.PlaneGeometry(1, 1);
const DOT = new THREE.SphereGeometry(.035, 8, 8);
const NODE = new THREE.SphereGeometry(.06, 10, 10);
const BAR = new THREE.BoxGeometry(1, 1, 1);
BAR.translate(0, .5, 0);

// Every material keeps its animated opacity in userData.base so the engine can apply scroll fades on top.
function kit(root) {
  const materials = [];
  const track = (material, base) => {
    material.transparent = true;
    material.depthWrite = false;
    material.userData.base = base;
    materials.push(material);
    return material;
  };
  const lineMat = (color, o) => track(new THREE.LineBasicMaterial({ color }), o);
  const meshMat = (color, o, wireframe = false) => track(new THREE.MeshBasicMaterial({ color, wireframe, side: THREE.DoubleSide }), o);
  const pointMat = (color, o, size) => track(new THREE.PointsMaterial({ color, size }), o);
  const line = (points, color, o, loop = false, parent = root) => {
    const l = new (loop ? THREE.LineLoop : THREE.Line)(new THREE.BufferGeometry().setFromPoints(points), lineMat(color, o));
    parent.add(l);
    return l;
  };
  const rect = (x0, y0, x1, y1, color, o, z = 0, parent = root) =>
    line([V(x0, y0, z), V(x1, y0, z), V(x1, y1, z), V(x0, y1, z)], color, o, true, parent);
  const ring = (x, y, r, color, o, z = 0, parent = root) =>
    line(Array.from({ length: 24 }, (_, i) => V(x + Math.cos(i / 24 * Math.PI * 2) * r, y + Math.sin(i / 24 * Math.PI * 2) * r, z)), color, o, true, parent);
  const mesh = (geometry, material, parent = root) => {
    const m = new THREE.Mesh(geometry, material);
    parent.add(m);
    return m;
  };
  const pulses = (count, color = PAPER, size = .09, parent = root) => {
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) positions[i * 3 + 1] = -99;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const points = new THREE.Points(geometry, pointMat(color, .95, size));
    points.frustumCulled = false;
    parent.add(points);
    return {
      material: points.material,
      set(i, v) { positions[i * 3] = v.x; positions[i * 3 + 1] = v.y; positions[i * 3 + 2] = v.z; },
      hide(i) { positions[i * 3] = 0; positions[i * 3 + 1] = -99; positions[i * 3 + 2] = 0; },
      flush() { geometry.attributes.position.needsUpdate = true; },
    };
  };
  const polyline = (coords, color = GREY, o = .3) => {
    const pts = coords.map(([x, y, z = 0]) => V(x, y, z));
    const path = new THREE.CurvePath();
    for (let i = 0; i < pts.length - 1; i++) path.add(new THREE.LineCurve3(pts[i], pts[i + 1]));
    line(pts, color, o);
    return path;
  };
  const setO = (material, o) => { material.userData.base = o; };
  return { materials, lineMat, meshMat, pointMat, line, rect, ring, mesh, pulses, polyline, setO };
}

/* Decoder transformer: embeddings -> 2 x [masked multi-head attention + FFN, residuals] -> softmax -> next token fed back */
function transformer() {
  const group = new THREE.Group(), inner = new THREE.Group();
  group.add(inner);
  inner.position.y = -.1;
  const k = kit(inner), { line, rect, mesh, meshMat, lineMat, setO } = k;
  const tx = i => -1.25 + i * .5;

  const tokens = [];
  for (let i = 0; i < 6; i++) {
    const m = mesh(BOX, meshMat(LIME, .9, true));
    m.scale.setScalar(.2);
    m.position.set(tx(i), -2.4, 0);
    m.userData.slot = i;
    tokens.push(m);
    line([V(tx(i), -2.28), V(tx(i), -2.05)], GREY, .4);
  }
  rect(-1.6, -2.05, 1.6, -1.9, VIOLET, .7);
  line([V(0, -1.9), V(0, -1.55)], GREY, .35);

  const blocks = [];
  for (let b = 0; b < 2; b++) {
    const y0 = -1.55 + b * 1.75, block = { y0, cells: [], nodes: [] };
    rect(-1.6, y0, 1.6, y0 + .85, VIOLET, .6);
    for (let h = 0; h < 4; h++) {
      const z = (h - 1.5) * .1, hx = -1.2 + h * .8;
      rect(hx - .33, y0 + .08, hx + .33, y0 + .78, GREY, .3, z);
      for (let r = 0; r < 4; r++) for (let c = 0; c <= r; c++) {
        const m = mesh(PLANE, meshMat(LIME, .1));
        m.scale.setScalar(.13);
        m.position.set(hx - .225 + c * .15, y0 + .68 - r * .15, z);
        block.cells.push({ m, seed: Math.random() * 6.3 });
      }
    }
    rect(-1.6, y0 + .92, 1.6, y0 + .98, GREY, .6);
    const rows = [[4, y0 + 1.06], [8, y0 + 1.24], [4, y0 + 1.42]], grid = [], pts = [];
    rows.forEach(([n, y]) => {
      const row = [];
      for (let i = 0; i < n; i++) {
        const x = (i - (n - 1) / 2) * (2.4 / (n - 1));
        const m = mesh(DOT, meshMat(LIME, .8));
        m.position.set(x, y, 0);
        block.nodes.push(m);
        row.push(V(x, y));
      }
      grid.push(row);
    });
    for (let r = 0; r < 2; r++) grid[r].forEach(p => grid[r + 1].forEach(q => pts.push(p, q)));
    block.ffn = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), lineMat(LIME, .05));
    inner.add(block.ffn);
    rect(-1.6, y0 + 1.5, 1.6, y0 + 1.56, GREY, .6);
    line([V(-1.6, y0 - .1), V(-1.75, y0 - .1), V(-1.75, y0 + 1.53), V(-1.6, y0 + 1.53)], GREY, .5);
    [y0 + .95, y0 + 1.53].forEach(y => k.ring(-1.75, y, .07, LIME, .9));
    blocks.push(block);
  }

  line([V(0, 1.76), V(0, 1.86)], GREY, .4);
  rect(-1.6, 1.86, 1.6, 1.92, VIOLET, .7);
  const bars = [];
  for (let i = 0; i < 6; i++) {
    const m = mesh(BAR, meshMat(VIOLET, .8, true));
    m.scale.set(.2, .05, .2);
    m.position.set(tx(i), 2, 0);
    bars.push(m);
  }
  const feedback = [V(0, 2.45), V(2.15, 2.45), V(2.15, -2.4), V(1.6, -2.4)], path = new THREE.CurvePath();
  for (let i = 0; i < 3; i++) path.add(new THREE.LineCurve3(feedback[i], feedback[i + 1]));
  line(feedback, GREY, .3);
  line([V(0, 2), V(0, 2.45)], GREY, .3);
  const pulse = k.pulses(3);

  let cycle = .55, last = 0, pick = 2;
  return {
    group, materials: k.materials, width: 4.6, height: 5.4,
    update({ dt, t, boost }) {
      cycle += dt * .14 * boost;
      const ph = cycle % 1, cn = Math.floor(cycle);
      if (cn !== last) {
        last = cn;
        pick = Math.random() * 6 | 0;
        tokens.forEach(m => {
          m.userData.slot--;
          if (m.userData.slot < 0) { m.userData.slot = 5; m.position.x = tx(6); setO(m.material, 0); }
        });
      }
      const py = ph < .78 ? -2.3 + ph / .78 * 4.25 : -99;
      if (py > -50) { pulse.set(0, V(0, py)); pulse.set(1, V(-1.75, py)); } else { pulse.hide(0); pulse.hide(1); }
      if (ph > .82) pulse.set(2, path.getPointAt((ph - .82) / .18)); else pulse.hide(2);
      pulse.flush();
      const hover = clamp01((boost - 1) / 1.6);
      blocks.forEach(block => {
        const a1 = Math.max(0, 1 - Math.abs(py - (block.y0 + .45)) / .7), a2 = Math.max(0, 1 - Math.abs(py - (block.y0 + 1.25)) / .6);
        block.cells.forEach(c => {
          setO(c.m.material, .08 + a1 * (.15 + .7 * (.5 + .5 * Math.sin(t * 9 + c.seed))));
          c.m.material.color.copy(LIME).lerp(VIOLET, hover * .7);
        });
        block.nodes.forEach(n => setO(n.material, .25 + a2 * .75));
        setO(block.ffn.material, .04 + a2 * .5);
      });
      bars.forEach((m, i) => {
        const h = ph > .7 ? (i === pick ? .6 : .08 + .12 * Math.abs(Math.sin(i * 2.1 + cn))) : .05;
        m.scale.y += (h - m.scale.y) * .1;
        m.material.color.copy(i === pick && ph > .7 ? LIME : VIOLET);
      });
      tokens.forEach(m => {
        m.position.x += (tx(m.userData.slot) - m.position.x) * .08;
        setO(m.material, m.material.userData.base + (.9 - m.material.userData.base) * .06);
        m.rotation.y += dt * .6;
      });
    },
  };
}

/* Deep neural network: 3D layers of neurons; forward activations, then loss gradients flowing back */
function neural() {
  const group = new THREE.Group(), k = kit(group), { setO } = k;
  const dims = [[3, 3], [4, 4], [4, 4], [3, 3], [3, 1]], gap = .44, layers = [];
  dims.forEach(([rows, cols], li) => {
    const x = -2.4 + li * 1.2, nodes = [];
    const hy = (rows - 1) / 2 * gap + .22, hz = (cols - 1) / 2 * gap + .22;
    k.line([V(x, -hy, -hz), V(x, hy, -hz), V(x, hy, hz), V(x, -hy, hz)], GREY, .3, true);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const p = V(x, (r - (rows - 1) / 2) * gap, (c - (cols - 1) / 2) * gap);
      const m = k.mesh(NODE, k.meshMat(LIME, .5));
      m.position.copy(p);
      nodes.push({ m, p });
    }
    layers.push({ nodes });
  });
  const links = [];
  for (let li = 0; li < layers.length - 1; li++) {
    const pts = [];
    layers[li].nodes.forEach(a => layers[li + 1].nodes.forEach(b => pts.push(a.p, b.p)));
    const segments = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), k.lineMat(LIME, .06));
    group.add(segments);
    links.push(segments);
  }
  k.line([V(2.6, 0), V(2.95, 0)], GREY, .35);
  const loss = k.mesh(new THREE.OctahedronGeometry(.3, 0), k.meshMat(VIOLET, .8, true));
  loss.position.set(3.25, 0, 0);

  const count = 40, pulse = k.pulses(count), routes = [], tmp = V(0, 0);
  const reroute = () => {
    routes.length = 0;
    for (let j = 0; j < count; j++) routes.push(layers.map(l => Math.random() * l.nodes.length | 0));
  };
  reroute();

  let cycle = .3, last = 0;
  const n = layers.length - 1;
  return {
    group, materials: k.materials, width: 6.4, height: 3,
    update({ dt, boost }) {
      cycle += dt * .16 * boost;
      const ph = cycle % 1, cn = Math.floor(cycle);
      if (cn !== last) { last = cn; reroute(); }
      const forward = ph < .55;
      const front = forward ? ph / .55 * (n + .6) - .3 : (1 - (ph - .55) / .45) * (n + .6) - .3;
      const color = forward ? LIME : VIOLET;
      layers.forEach((layer, li) => {
        const a = Math.max(0, 1 - Math.abs(front - li) / .9);
        layer.nodes.forEach(({ m }) => {
          setO(m.material, .25 + a * .75);
          m.scale.setScalar(1 + a * .7);
          m.material.color.copy(GREY).lerp(color, .35 + a * .65);
        });
      });
      links.forEach((segments, li) => {
        const a = Math.max(0, 1 - Math.abs(front - (li + .5)) / .8);
        setO(segments.material, .04 + a * .4);
        segments.material.color.copy(color);
      });
      pulse.material.color.copy(forward ? PAPER : VIOLET);
      routes.forEach((route, j) => {
        const f = front - (forward ? 1 : -1) * (j % 8) * .06;
        if (f < 0 || f >= n) { pulse.hide(j); return; }
        const L = Math.floor(f);
        tmp.lerpVectors(layers[L].nodes[route[L]].p, layers[L + 1].nodes[route[L + 1]].p, f - L);
        pulse.set(j, tmp);
      });
      pulse.flush();
      const hit = Math.max(0, 1 - Math.abs(ph - .55) / .08);
      loss.rotation.y += dt * 1.2;
      loss.rotation.x += dt * .7;
      loss.scale.setScalar(1 + hit * .5);
      setO(loss.material, .45 + hit * .55);
      loss.material.color.copy(VIOLET).lerp(PAPER, hit * .5);
    },
  };
}

/* RAG: documents chunked into a rotating vector store, a query retrieves its top-k neighbours, an LLM generates the answer */
function rag() {
  const group = new THREE.Group(), k = kit(group), { setO } = k;

  const docs = new THREE.Group();
  docs.position.set(-2.45, .35, 0);
  group.add(docs);
  for (let i = 3; i >= 0; i--) k.rect(-.42, -.58, .42, .58, i ? GREY : PAPER, i ? .3 : .75, -i * .16, docs);
  for (let i = 0; i < 5; i++) k.line([V(-.28, .36 - i * .17, .01), V(i % 2 ? .12 : .28, .36 - i * .17, .01)], GREY, .55, false, docs);

  const store = new THREE.Group();
  store.position.set(0, .35, 0);
  group.add(store);
  const shell = k.mesh(new THREE.IcosahedronGeometry(1.2, 1), k.meshMat(VIOLET, .16, true), store);
  const cloud = [];
  while (cloud.length < 150) {
    const p = V(Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1);
    if (p.length() < 1) cloud.push(p);
  }
  const cloudPoints = new THREE.Points(new THREE.BufferGeometry().setFromPoints(cloud), k.pointMat(PAPER, .5, .045));
  store.add(cloudPoints);
  const hits = k.pulses(4, LIME, .15, store);
  const retrieval = new Float32Array(8 * 3);
  const retrievalLines = new THREE.LineSegments(new THREE.BufferGeometry(), k.lineMat(LIME, 0));
  retrievalLines.geometry.setAttribute('position', new THREE.BufferAttribute(retrieval, 3));
  retrievalLines.frustumCulled = false;
  group.add(retrievalLines);

  const llm = new THREE.Group();
  llm.position.set(2.45, -.45, 0);
  group.add(llm);
  const layers = [];
  for (let i = 0; i < 4; i++) {
    const m = k.mesh(BOX, k.meshMat(VIOLET, .5, true), llm);
    m.scale.set(1, .1, .7);
    m.position.y = i * .38;
    layers.push(m);
  }
  const answer = [];
  for (let i = 0; i < 5; i++) {
    const m = k.mesh(BAR, k.meshMat(LIME, .85, true), llm);
    m.scale.set(.13, .02, .13);
    m.position.set(-.4 + i * .2, 1.55, 0);
    answer.push(m);
  }

  k.line([V(-2.0, .35), V(-1.25, .35)], GREY, .3);
  k.line([V(-1.3, -2.1), V(-.2, -.85)], GREY, .3);
  k.line([V(1.22, .2), V(1.95, -.45)], GREY, .3);

  const query = k.mesh(BOX, k.meshMat(LIME, .95, true));
  query.scale.setScalar(.2);
  const chunks = [0, 1, 2].map(() => { const m = k.mesh(BOX, k.meshMat(PAPER, 0, true)); m.scale.setScalar(.1); return m; });
  const retrieved = [0, 1, 2, 3].map(() => { const m = k.mesh(BOX, k.meshMat(LIME, 0, true)); m.scale.setScalar(.1); return m; });

  let target = 0, topk = [], chunkTargets = [];
  const pick = () => {
    target = Math.random() * cloud.length | 0;
    topk = cloud.map((p, i) => [p.distanceToSquared(cloud[target]), i]).sort((a, b) => a[0] - b[0]).slice(0, 4).map(x => x[1]);
    chunkTargets = chunks.map(() => Math.random() * cloud.length | 0);
  };
  pick();
  const inGroup = p => p.clone().applyMatrix4(store.matrix);
  const docOut = V(-2.0, .35, 0), queryStart = V(-1.3, -2.1, 0), llmIn = V(1.95, -.45, 0);

  let cycle = .5, last = 0, time = 0;
  return {
    group, materials: k.materials, width: 6.2, height: 4.8,
    update({ dt, boost }) {
      cycle += dt * .12 * boost;
      time += dt;
      const ph = cycle % 1, cn = Math.floor(cycle);
      if (cn !== last) { last = cn; pick(); }
      store.rotation.y += dt * .25;
      store.rotation.x = Math.sin(time * .3) * .15;
      store.updateMatrix();
      shell.rotation.y -= dt * .1;

      chunks.forEach((m, i) => {
        const u = smooth((ph - i * .06) / .2);
        m.position.lerpVectors(docOut, inGroup(cloud[chunkTargets[i]]), u);
        m.rotation.x += dt * 2;
        setO(m.material, u > 0 && u < 1 ? .9 : 0);
      });

      query.position.lerpVectors(queryStart, inGroup(cloud[target]), smooth((ph - .25) / .2));
      query.rotation.y += dt * 1.5;
      setO(query.material, ph < .62 ? .95 : Math.max(0, .95 - (ph - .62) * 8));

      const lit = smooth((ph - .45) / .08) * (1 - smooth((ph - .75) / .08));
      topk.forEach((idx, i) => {
        hits.set(i, cloud[idx]);
        const p = inGroup(cloud[idx]);
        retrieval.set([query.position.x, query.position.y, query.position.z, p.x, p.y, p.z], i * 6);
      });
      hits.flush();
      retrievalLines.geometry.attributes.position.needsUpdate = true;
      setO(hits.material, lit);
      setO(retrievalLines.material, lit * .8);
      setO(cloudPoints.material, .45 + .2 * lit);

      retrieved.forEach((m, i) => {
        const u = smooth((ph - .6 - i * .03) / .16);
        m.position.lerpVectors(inGroup(cloud[topk[i]]), llmIn, u);
        m.rotation.y += dt * 2;
        setO(m.material, u > 0 && u < 1 ? .95 : 0);
      });
      layers.forEach((m, i) => {
        const a = Math.max(0, 1 - Math.abs((ph - .78) / .16 * 4 - i));
        setO(m.material, .35 + a * .6);
        m.material.color.copy(VIOLET).lerp(LIME, a);
      });
      answer.forEach((m, i) => {
        const on = ph > .86 + i * .025;
        m.scale.y += ((on ? .25 + .25 * Math.abs(Math.sin(i * 1.7 + cn)) : .02) - m.scale.y) * .12;
      });
    },
  };
}

/* LoRA: frozen weight matrix W on one path, trainable low-rank adapters A -> B on the other, summed at the output */
function lora() {
  const group = new THREE.Group(), k = kit(group), { setO } = k;
  const cell = .17, size = 8;

  const inputs = [], outputs = [];
  for (let i = 0; i < size; i++) {
    const a = k.mesh(BOX, k.meshMat(LIME, .5, true));
    a.scale.setScalar(.12);
    a.position.set(-2.6, (i - 3.5) * cell, 0);
    inputs.push(a);
    const b = k.mesh(BOX, k.meshMat(PAPER, .4, true));
    b.scale.setScalar(.12);
    b.position.set(2.65, (i - 3.5) * cell, 0);
    outputs.push(b);
  }

  const W = new THREE.Group();
  W.position.set(0, .8, 0);
  group.add(W);
  const half = size / 2 * cell + .06;
  [0, -.12, -.24].forEach((z, i) => k.rect(-half, -half, half, half, GREY, .55 - i * .18, z, W));
  const weights = [];
  for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) {
    const m = k.mesh(PLANE, k.meshMat(GREY, .12), W);
    m.scale.setScalar(cell * .82);
    m.position.set((c - 3.5) * cell, (r - 3.5) * cell, 0);
    weights.push({ m, seed: Math.random() * 6.3 });
  }
  k.rect(-.1, half + .12, .1, half + .28, GREY, .7, 0, W);
  k.line(Array.from({ length: 9 }, (_, i) => V(Math.cos(i / 8 * Math.PI) * .065, half + .28 + Math.sin(i / 8 * Math.PI) * .07)), GREY, .7, false, W);

  const adapter = (x) => {
    const g = new THREE.Group();
    g.position.set(x, -1, 0);
    group.add(g);
    k.rect(-cell - .05, -half, cell + .05, half, LIME, .6, 0, g);
    const cells = [];
    for (let r = 0; r < size; r++) for (let c = 0; c < 2; c++) {
      const m = k.mesh(PLANE, k.meshMat(LIME, .2), g);
      m.scale.setScalar(cell * .82);
      m.position.set((c - .5) * cell, (r - 3.5) * cell, 0);
      cells.push({ m, seed: Math.random() * 6.3 });
    }
    return cells;
  };
  const A = adapter(-.55), B = adapter(.55);
  const rank = [-.1, .1].map(y => { const m = k.mesh(NODE, k.meshMat(LIME, .8)); m.position.set(0, -1 + y, 0); return m; });

  const top = k.polyline([[-2.4, 0], [-1.25, .8], [1.25, .8], [1.72, .1]]);
  const bottom = k.polyline([[-2.4, 0], [-1.25, -1], [1.25, -1], [1.72, -.1]]);
  const out = k.polyline([[1.99, 0], [2.48, 0]]);
  k.ring(1.85, 0, .14, LIME, .9);
  k.line([V(1.78, 0), V(1.92, 0)], LIME, .9);
  k.line([V(1.85, -.07), V(1.85, .07)], LIME, .9);

  const signal = k.pulses(3), gradient = k.pulses(2, VIOLET, .11);
  let cycle = .3, time = 0;
  return {
    group, materials: k.materials, width: 5.8, height: 3.8,
    update({ dt, boost }) {
      cycle += dt * .13 * boost;
      time += dt;
      const ph = cycle % 1;
      const pass = ph / .42;
      if (pass < 1) { signal.set(0, top.getPointAt(pass)); signal.set(1, bottom.getPointAt(pass)); } else { signal.hide(0); signal.hide(1); }
      if (ph > .42 && ph < .52) signal.set(2, out.getPointAt((ph - .42) / .1)); else signal.hide(2);
      signal.flush();
      const back = (ph - .58) / .34;
      if (back > 0 && back < 1) {
        gradient.set(0, bottom.getPointAt(1 - back * .78));
        gradient.set(1, bottom.getPointAt(Math.max(0, 1 - (back - .08) * .78)));
      } else { gradient.hide(0); gradient.hide(1); }
      gradient.flush();

      const start = Math.max(0, 1 - ph / .08);
      inputs.forEach((m, i) => { setO(m.material, .4 + start * .55); m.rotation.y += dt * (.4 + i * .05); });
      const done = Math.max(0, 1 - Math.abs(ph - .54) / .08);
      outputs.forEach((m, i) => { setO(m.material, .35 + done * .6); m.material.color.copy(PAPER).lerp(LIME, done); m.rotation.y -= dt * (.4 + i * .05); });

      const throughW = pass < 1 ? Math.max(0, 1 - Math.abs(pass - .5) / .22) : 0;
      weights.forEach(({ m, seed }) => setO(m.material, .1 + throughW * (.2 + .2 * (.5 + .5 * Math.sin(seed * 5)))));
      const training = back > 0 && back < 1.1 ? 1 : 0;
      const throughA = pass < 1 ? Math.max(0, 1 - Math.abs(pass - .42) / .15) : 0;
      const throughB = pass < 1 ? Math.max(0, 1 - Math.abs(pass - .58) / .15) : 0;
      [[A, throughA], [B, throughB]].forEach(([cells, active]) => cells.forEach(({ m, seed }) => {
        setO(m.material, .18 + active * .5 + training * (.25 + .4 * (.5 + .5 * Math.sin(time * 8 + seed))));
        m.material.color.copy(LIME).lerp(VIOLET, training * .55);
      }));
      rank.forEach(m => { const a = Math.max(throughA, throughB); m.scale.setScalar(1 + a * .8); setO(m.material, .5 + a * .5); });
    },
  };
}

export function buildModels() {
  return { transformer: transformer(), neural: neural(), rag: rag(), lora: lora() };
}
