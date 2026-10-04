import * as THREE from 'three';

const lime = '#B8F500', violet = '#8B5CF6', text = '#F2F0EB';
const point = (x, y, z = 0) => new THREE.Vector3(x, y, z);

export function buildDiagram(type) {
  const group = new THREE.Group();
  const animated = [];
  const sphere = new THREE.SphereGeometry(0.065, 10, 8);
  function line(coords, color = lime, opacity = 0.45) {
    const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false });
    const object = new THREE.Line(new THREE.BufferGeometry().setFromPoints(coords.map(p => point(...p))), material);
    group.add(object);
    return object;
  }
  function label(value, x, y, size = 0.4, color = text) {
    const canvas = document.createElement('canvas');
    canvas.width = 1024; canvas.height = 96;
    const ctx = canvas.getContext('2d');
    ctx.font = '500 48px monospace'; ctx.fillStyle = color;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(value, 512, 48);
    const texture = new THREE.CanvasTexture(canvas);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }));
    sprite.scale.set(size * 1024 / 96, size, 1);
    sprite.position.set(x, y, 0.12); group.add(sprite);
  }
  function box(name, x, y, w = 1.6, h = 0.6, color = lime) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, 0.12), new THREE.MeshBasicMaterial({ color: '#08080A', transparent: true, opacity: 0.94 }));
    mesh.position.set(x, y, 0); group.add(mesh);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7 }));
    edges.position.copy(mesh.position); group.add(edges);
    if (name) label(name, x, y, 0.4, color);
  }
  function node(x, y, radius = 0.12, color = lime) {
    const object = new THREE.Mesh(new THREE.SphereGeometry(radius, 12, 10), new THREE.MeshBasicMaterial({ color, wireframe: true }));
    object.position.set(x, y, 0); group.add(object);
    return object;
  }
  function flow(coords, color = lime, offset = 0, speed = 0.18) {
    line(coords, color, 0.4);
    const curve = new THREE.CurvePath();
    for (let i = 1; i < coords.length; i++) curve.add(new THREE.LineCurve3(point(...coords[i - 1]), point(...coords[i])));
    const pulse = new THREE.Mesh(sphere, new THREE.MeshBasicMaterial({ color }));
    group.add(pulse);
    animated.push(time => pulse.position.copy(curve.getPoint((time * speed + offset) % 1)));
    const end = point(...coords.at(-1)), before = point(...coords.at(-2));
    const dir = end.clone().sub(before).normalize(), side = point(-dir.y, dir.x);
    const base = end.clone().addScaledVector(dir, -0.13);
    line([base.clone().addScaledVector(side, 0.07).toArray(), end.toArray(), base.clone().addScaledVector(side, -0.07).toArray()], color, 0.8);
  }

  if (type === 'transformer') {
    box('INPUT TOKENS', 0, -2.5, 2.6, 0.4);
    box('TOKEN + POSITION EMBEDDING', 0, -1.85, 4.5, 0.45, violet);
    flow([[0, -2.3], [0, -2.08]]);
    flow([[0, -1.62], [0, -1.35]], violet);
    for (let block = 0; block < 2; block++) {
      const y = -1.25 + block * 1.6;
      box('', 0, y + 0.42, 4.5, 1.45, violet);
      label('MASKED MULTI-HEAD ATTENTION', 0, y - 0.1, 0.36);
      for (let head = 0; head < 4; head++) {
        const x = -1.45 + head * 0.96;
        for (let r = 0; r < 3; r++) for (let c = 0; c <= r; c++) {
          const cell = new THREE.Mesh(new THREE.PlaneGeometry(0.13, 0.13), new THREE.MeshBasicMaterial({ color: lime, transparent: true, opacity: 0.4 }));
          cell.position.set(x + c * 0.16, y + 0.6 - r * 0.16, 0.1); group.add(cell);
          animated.push(time => { cell.material.opacity = 0.2 + 0.7 * Math.max(0, Math.sin(time * 2 - head * 0.6 - block)); });
        }
      }
      label('FEED-FORWARD + ADD & NORM', 0, y + 0.94, 0.36);
      const lower = [-1.25, -0.42, 0.42, 1.25], upper = [-1.55, -0.94, -0.31, 0.31, 0.94, 1.55];
      lower.forEach(x => { node(x, y + 0.64, 0.025); upper.forEach(next => line([[x, y + 0.64], [next, y + 0.77]], lime, 0.25)); });
      upper.forEach(x => node(x, y + 0.77, 0.025));
      flow([[-2.25, y - 0.2], [-2.65, y - 0.2], [-2.65, y + 1], [-2.25, y + 1]], violet, block * 0.3);
      if (block === 0) flow([[0, y + 1.15], [0, y + 1.32]]);
    }
    box('LINEAR + SOFTMAX', 0, 2.05, 3.8, 0.4);
    flow([[0, 1.52], [0, 1.85]]);
    box('NEXT TOKEN', 0, 2.7, 2.6, 0.4);
    flow([[0, 2.25], [0, 2.5]]);
    flow([[1.3, 2.7], [2.9, 2.7], [2.9, -2.5], [1.3, -2.5]], violet, 0.4, 0.1);
  } else if (type === 'neural') {
    const layers = [4, 6, 6, 3], xs = [-2.5, -0.85, 0.85, 2.5];
    const positions = layers.map((count, layer) => Array.from({ length: count }, (_, i) => [xs[layer], (i - (count - 1) / 2) * 0.62]));
    positions.forEach((layer, index) => {
      layer.forEach(([x, y]) => {
        const n = node(x, y, 0.16, index === 3 ? violet : lime);
        animated.push(time => { n.scale.setScalar(1 + 0.18 * Math.max(0, Math.sin(time * 2.1 - index * 0.8 - y))); });
      });
      if (index < 3) layer.forEach(a => positions[index + 1].forEach(b => line([a, b], index === 2 ? violet : lime, 0.17)));
    });
    for (let i = 0; i < 6; i++) flow(positions.map(layer => layer[i % layer.length]), i % 2 ? violet : lime, i / 6, 0.16);
    label('INPUT', -2.5, -2.25); label('HIDDEN LAYERS', 0, -2.25); label('OUTPUT', 2.5, -2.25);
    label('FORWARD PASS', 0, 2.2, 0.4, lime);
    flow([[2.5, -2.7], [-2.5, -2.7]], violet, 0.2, 0.13);
    label('LOSS → BACKPROPAGATION → UPDATE', 0, -3.05, 0.36, violet);
  } else if (type === 'rag') {
    box('DOCUMENTS', -1.65, 2.35, 2.4, 0.65);
    box('CHUNKS', 1.65, 2.35, 2.4, 0.65);
    flow([[-0.45, 2.35], [0.45, 2.35]]);
    box('EMBEDDINGS', 1.65, 1.1, 2.4, 0.65, violet);
    flow([[1.65, 2.02], [1.65, 1.43]], violet);
    box('VECTOR STORE', -1.65, 1.1, 2.4, 0.65, violet);
    flow([[0.45, 1.1], [-0.45, 1.1]], violet);
    box('QUERY EMBEDDING', -1.65, -0.15, 2.4, 0.65);
    box('RETRIEVER', 1.65, -0.15, 2.4, 0.65);
    flow([[-0.45, -0.15], [0.45, -0.15]]);
    flow([[-1.65, 0.77], [-1.65, 0.5], [1.65, 0.5], [1.65, 0.18]], violet);
    box('CONTEXT + QUERY', 0, -1.35, 3.5, 0.6);
    flow([[1.65, -0.48], [1.65, -0.8], [0, -0.8], [0, -1.05]]);
    box('LLM', -1.65, -2.55, 2.4, 0.65, violet);
    box('ANSWER', 1.65, -2.55, 2.4, 0.65);
    flow([[0, -1.65], [0, -1.95], [-1.65, -1.95], [-1.65, -2.22]], violet);
    flow([[-0.45, -2.55], [0.45, -2.55]]);
  } else {
    box('INPUT ACTIVATIONS', 0, -2.4, 3.7, 0.6);
    box('FROZEN WEIGHTS W', -1.6, 0, 2.7, 2.4, violet);
    for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) {
      const dot = node(-2.55 + c * 0.38, -0.85 + r * 0.34, 0.035, violet);
      dot.position.z = -0.1;
    }
    box('LoRA A', 1.7, -0.7, 2.1, 0.65);
    box('LoRA B', 1.7, 0.7, 2.1, 0.65);
    flow([[0, -2.1], [0, -1.65], [-1.6, -1.65], [-1.6, -1.2]], violet);
    flow([[0, -1.65], [1.7, -1.65], [1.7, -1.03]]);
    flow([[1.7, -0.37], [1.7, 0.37]]);
    node(0, 1.85, 0.19); label('+', 0, 1.85, 0.28);
    flow([[-1.6, 1.2], [-1.6, 1.85], [-0.2, 1.85]], violet);
    flow([[1.7, 1.03], [1.7, 1.85], [0.2, 1.85]]);
    box('ADAPTED OUTPUT', 0, 2.75, 3.7, 0.55);
    flow([[0, 2.04], [0, 2.48]]);
    label('FROZEN', -1.6, -1.4, 0.36, violet);
    label('TRAINABLE', 1.7, -1.35, 0.36, lime);
  }
  return { group, update: time => animated.forEach(update => update(time)) };
}
