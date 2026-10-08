import { Html, OrbitControls, Segment, Segments } from '@react-three/drei';
import { useFrame, useThree, type ThreeEvent } from '@react-three/fiber';
import { useEffect, useLayoutEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import type { Edge3D, Layout, Vec3 } from '../../../forge/scene3d';
import type { ComponentRef } from '../../../forge/types';

export interface Palette {
  bg: THREE.Color;
  pos: THREE.Color;
  neg: THREE.Color;
  neutral: THREE.Color;
  warn: THREE.Color;
  select: THREE.Color;
  text: string;
}

export interface Hover { kind: 'neuron' | 'edge'; layer: number; index: number; source?: number; x: number; y: number }

export interface CameraGoal { position: Vec3; target: Vec3; key: number }

/** Rendering quality: geometry detail and whether the (aesthetic) glow is drawn. */
export type Quality = 'low' | 'medium' | 'high';
const SPHERE_DETAIL: Record<Quality, [number, number]> = { low: [10, 8], medium: [20, 14], high: [32, 24] };

interface SceneProps {
  layout: Layout;
  sizes: number[];
  names: string[][];
  layerLabels: string[];
  nodeVals: number[][];
  dimmed: boolean[]; // per graph layer (pass not reached)
  ablated: Set<string>; // "layer:index"
  edges: Edge3D[];
  shown: number[];
  emphasised: number[];
  pulses: { edge: number; strength: number; positive: boolean }[];
  pulseDir: 1 | -1;
  palette: Palette;
  selection: ComponentRef | null;
  goal: CameraGoal;
  reduceMotion: boolean;
  onSelect: (ref: ComponentRef) => void;
  onFocusLayer: (layer: number) => void;
  onHover: (h: Hover | null) => void;
  quality: Quality;
  /** Aesthetic layer: glow halos and depth fog. Never changes what a value means. */
  effects: boolean;
  /** [near, far] of the depth fog (camera-distance based). */
  fogRange: [number, number];
}

const tmpObj = new THREE.Object3D();
const tmpColor = new THREE.Color();

/** Signed value → colour: neutral → green (positive) / red (negative), by |v| / max. */
function signColor(out: THREE.Color, v: number, max: number, p: Palette, floor = 0) {
  const t = max > 0 ? Math.min(1, Math.abs(v) / max) : 0;
  const k = floor + (1 - floor) * Math.pow(t, 0.6);
  return out.copy(p.neutral).lerp(v >= 0 ? p.pos : p.neg, k);
}

function layerMax(values: number[]): number {
  let m = 0;
  for (const v of values) if (Math.abs(v) > m) m = Math.abs(v);
  return m;
}

function Nodes({ layout, sizes, nodeVals, dimmed, ablated, palette, onSelect, onHover, focusNeuron, quality }: Pick<SceneProps,
  'layout' | 'sizes' | 'nodeVals' | 'dimmed' | 'ablated' | 'palette' | 'onSelect' | 'onHover' | 'quality'> & { focusNeuron: (l: number, i: number) => void }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const flat = useMemo(() => sizes.flatMap((n, g) => Array.from({ length: n }, (_, i) => [g, i] as const)), [sizes]);
  const count = flat.length;

  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const maxes = nodeVals.map(layerMax);
    flat.forEach(([g, i], k) => {
      const p = layout.positions[g][i];
      tmpObj.position.set(p[0], p[1], p[2]);
      const io = g === 0 || g === sizes.length - 1;
      tmpObj.scale.setScalar(io ? 1.25 : 1);
      tmpObj.updateMatrix();
      mesh.setMatrixAt(k, tmpObj.matrix);
      if (ablated.has(`${g}:${i}`)) tmpColor.copy(palette.neutral).lerp(palette.bg, 0.4);
      else signColor(tmpColor, nodeVals[g]?.[i] ?? 0, maxes[g], palette, 0.08);
      if (dimmed[g]) tmpColor.lerp(palette.bg, 0.72);
      mesh.setColorAt(k, tmpColor);
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [flat, layout, nodeVals, dimmed, ablated, palette, sizes.length]);

  const resolve = (e: ThreeEvent<PointerEvent | MouseEvent>) => (e.instanceId === undefined ? null : flat[e.instanceId]);

  return (
    <instancedMesh key={count} ref={ref} args={[undefined, undefined, count]}
      onClick={(e) => { e.stopPropagation(); const r = resolve(e); if (r) onSelect({ kind: 'neuron', layer: r[0], index: r[1] }); }}
      onDoubleClick={(e) => { e.stopPropagation(); const r = resolve(e); if (r) focusNeuron(r[0], r[1]); }}
      onPointerMove={(e) => { e.stopPropagation(); const r = resolve(e); if (r) onHover({ kind: 'neuron', layer: r[0], index: r[1], x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY }); }}
      onPointerOut={() => onHover(null)}>
      <sphereGeometry args={[0.24, ...SPHERE_DETAIL[quality]]} />
      <meshStandardMaterial roughness={quality === 'high' ? 0.35 : 0.55} metalness={0.05} />
    </instancedMesh>
  );
}

/** Soft halo: brightest facing the camera, fading to nothing at the rim (additive). */
const glowMaterial = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
  vertexShader: `
    varying vec3 vColor;
    varying float vI;
    void main() {
      vColor = instanceColor;
      vec4 mv = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
      vec3 n = normalize(normalMatrix * mat3(instanceMatrix) * normal);
      vI = pow(max(dot(n, normalize(-mv.xyz)), 0.0), 3.0);
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: `
    varying vec3 vColor;
    varying float vI;
    void main() { gl_FragColor = vec4(vColor * vI * 0.9, 1.0); }`,
});

/**
 * Additive halo around each neuron.  Its radius and brightness are
 * proportional to |value| / layer max of the quantity the spheres show, so the
 * glow reads the same real number as the sphere colour; it adds no data.
 */
function Glow({ layout, sizes, nodeVals, dimmed, ablated, palette }: Pick<SceneProps, 'layout' | 'sizes' | 'nodeVals' | 'dimmed' | 'ablated' | 'palette'>) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const flat = useMemo(() => sizes.flatMap((n, g) => Array.from({ length: n }, (_, i) => [g, i] as const)), [sizes]);
  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const maxes = nodeVals.map(layerMax);
    flat.forEach(([g, i], k) => {
      const p = layout.positions[g][i];
      const v = nodeVals[g]?.[i] ?? 0;
      const t = maxes[g] > 0 ? Math.min(1, Math.abs(v) / maxes[g]) : 0;
      const off = ablated.has(`${g}:${i}`) || dimmed[g];
      tmpObj.position.set(p[0], p[1], p[2]);
      tmpObj.scale.setScalar(off ? 0.0001 : 0.35 + 1.25 * t);
      tmpObj.updateMatrix();
      mesh.setMatrixAt(k, tmpObj.matrix);
      tmpColor.copy(v >= 0 ? palette.pos : palette.neg).multiplyScalar(0.15 + 0.85 * t);
      mesh.setColorAt(k, tmpColor);
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [flat, layout, nodeVals, dimmed, ablated, palette]);
  return (
    <instancedMesh key={flat.length} ref={ref} args={[undefined, undefined, flat.length]} raycast={() => null} renderOrder={-1}>
      <sphereGeometry args={[0.6, 24, 16]} />
      <primitive object={glowMaterial} attach="material" />
    </instancedMesh>
  );
}

function EdgeLines({ layout, edges, shown, palette, onSelect, onHover }: Pick<SceneProps, 'layout' | 'edges' | 'shown' | 'palette' | 'onSelect' | 'onHover'>) {
  const geometry = useMemo(() => {
    const pos = new Float32Array(shown.length * 6);
    const col = new Float32Array(shown.length * 6);
    const maxByLayer = new Map<number, number>();
    for (const i of shown) {
      const e = edges[i];
      maxByLayer.set(e.layer, Math.max(maxByLayer.get(e.layer) ?? 0, Math.abs(e.value)));
    }
    const c = new THREE.Color();
    shown.forEach((ei, k) => {
      const e = edges[ei];
      const a = layout.positions[e.layer - 1][e.source];
      const b = layout.positions[e.layer][e.target];
      pos.set([a[0], a[1], a[2], b[0], b[1], b[2]], k * 6);
      const m = maxByLayer.get(e.layer) || 1;
      const t = Math.min(1, Math.abs(e.value) / m);
      if (e.edited) c.copy(palette.warn);
      // Fade towards the background: line alpha per vertex is not available in WebGL lines.
      else c.copy(palette.bg).lerp(e.value >= 0 ? palette.pos : palette.neg, 0.12 + 0.88 * Math.pow(t, 0.8));
      col.set([c.r, c.g, c.b, c.r, c.g, c.b], k * 6);
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.computeBoundingSphere();
    return g;
  }, [layout, edges, shown, palette]);
  useEffect(() => () => geometry.dispose(), [geometry]);

  const edgeAt = (e: ThreeEvent<PointerEvent | MouseEvent>) => (e.index === undefined ? null : edges[shown[Math.floor(e.index / 2)]] ?? null);
  if (!shown.length) return null;
  return (
    <lineSegments geometry={geometry}
      onClick={(ev) => { ev.stopPropagation(); const e = edgeAt(ev); if (e) onSelect({ kind: 'connection', layer: e.layer, source: e.source, target: e.target }); }}
      onPointerMove={(ev) => { ev.stopPropagation(); const e = edgeAt(ev); if (e) onHover({ kind: 'edge', layer: e.layer, index: e.target, source: e.source, x: ev.nativeEvent.offsetX, y: ev.nativeEvent.offsetY }); }}
      onPointerOut={() => onHover(null)}>
      <lineBasicMaterial vertexColors />
    </lineSegments>
  );
}

function Emphasised({ layout, edges, emphasised, palette }: Pick<SceneProps, 'layout' | 'edges' | 'emphasised' | 'palette'>) {
  const max = useMemo(() => layerMax(emphasised.map((i) => edges[i].value)), [edges, emphasised]);
  if (!emphasised.length) return null;
  return (
    <Segments limit={Math.max(16, emphasised.length)} lineWidth={2.6}>
      {emphasised.map((i) => {
        const e = edges[i];
        const c = e.edited ? palette.warn.clone() : signColor(new THREE.Color(), e.value, max, palette, 0.35);
        return <Segment key={i} start={layout.positions[e.layer - 1][e.source]} end={layout.positions[e.layer][e.target]} color={c} />;
      })}
    </Segments>
  );
}

/** Small spheres travelling along the strongest real connections of the active pass step. */
function Pulses({ layout, edges, pulses, pulseDir, palette, reduceMotion }: Pick<SceneProps, 'layout' | 'edges' | 'pulses' | 'pulseDir' | 'palette' | 'reduceMotion'>) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const phase = useRef(0);
  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    pulses.forEach((p, k) => {
      mesh.setColorAt(k, p.positive ? palette.pos : palette.neg);
    });
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [pulses, edges, palette]);
  useFrame((_, dt) => {
    const mesh = ref.current;
    if (!mesh || !pulses.length) return;
    phase.current = reduceMotion ? 0.5 : (phase.current + dt * 0.7) % 1;
    pulses.forEach((p, k) => {
      const e = edges[p.edge];
      const a = layout.positions[e.layer - 1][e.source];
      const b = layout.positions[e.layer][e.target];
      const f = (phase.current + (k % 5) * 0.2) % 1;
      const s = pulseDir === 1 ? f : 1 - f;
      tmpObj.position.set(a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s);
      tmpObj.scale.setScalar(0.45 + 0.75 * p.strength);
      tmpObj.updateMatrix();
      mesh.setMatrixAt(k, tmpObj.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
  });
  if (!pulses.length) return null;
  return (
    <instancedMesh key={pulses.length} ref={ref} args={[undefined, undefined, pulses.length]} raycast={() => null}>
      <sphereGeometry args={[0.11, 10, 8]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

function SelectionMarks({ layout, selection, sizes, palette, ablated }: Pick<SceneProps, 'layout' | 'selection' | 'sizes' | 'palette' | 'ablated'>) {
  const { camera } = useThree();
  const ring = useRef<THREE.Mesh>(null);
  useFrame(() => { if (ring.current) ring.current.quaternion.copy(camera.quaternion); });
  const ablatedPos = [...ablated].map((k) => k.split(':').map(Number)).filter(([g, i]) => layout.positions[g]?.[i]);
  return (
    <>
      {selection?.kind === 'neuron' && layout.positions[selection.layer]?.[selection.index] && (
        <mesh ref={ring} position={layout.positions[selection.layer][selection.index]} raycast={() => null}>
          <ringGeometry args={[0.36, 0.45, 40]} />
          <meshBasicMaterial color={palette.select} side={THREE.DoubleSide} toneMapped={false} />
        </mesh>
      )}
      {selection?.kind === 'layer' && sizes[selection.layer] !== undefined && (
        <mesh position={[layout.layerX[selection.layer], layout.center[1], layout.center[2]]} raycast={() => null}>
          <boxGeometry args={[1.1, layout.extents[selection.layer].y * 2 + 1.1, layout.extents[selection.layer].z * 2 + 1.1]} />
          <meshBasicMaterial color={palette.select} wireframe transparent opacity={0.55} />
        </mesh>
      )}
      {ablatedPos.map(([g, i]) => (
        <mesh key={`${g}:${i}`} position={layout.positions[g][i]} raycast={() => null}>
          <torusGeometry args={[0.32, 0.035, 8, 28]} />
          <meshBasicMaterial color={palette.warn} toneMapped={false} />
        </mesh>
      ))}
    </>
  );
}

function CameraRig({ goal, reduceMotion }: { goal: CameraGoal; reduceMotion: boolean }) {
  const controls = useThree((s) => s.controls) as unknown as OrbitControlsImpl | null;
  const camera = useThree((s) => s.camera);
  const anim = useRef<{ fromP: THREE.Vector3; fromT: THREE.Vector3; toP: THREE.Vector3; toT: THREE.Vector3; t: number } | null>(null);
  useEffect(() => {
    if (!controls) return;
    const toP = new THREE.Vector3(...goal.position);
    const toT = new THREE.Vector3(...goal.target);
    if (reduceMotion) {
      camera.position.copy(toP);
      controls.target.copy(toT);
      controls.update();
      return;
    }
    anim.current = { fromP: camera.position.clone(), fromT: controls.target.clone(), toP, toT, t: 0 };
  }, [goal, controls, camera, reduceMotion]);
  useFrame((_, dt) => {
    const a = anim.current;
    if (!a || !controls) return;
    a.t = Math.min(1, a.t + dt / 0.7);
    const k = 1 - Math.pow(1 - a.t, 3);
    camera.position.lerpVectors(a.fromP, a.toP, k);
    controls.target.lerpVectors(a.fromT, a.toT, k);
    controls.update();
    if (a.t >= 1) anim.current = null;
  });
  return null;
}

export function Scene(props: SceneProps) {
  const { layout, sizes, layerLabels, palette, selection, onSelect, onFocusLayer, goal, reduceMotion, quality, effects, fogRange } = props;
  const paper = palette.bg.r + palette.bg.g + palette.bg.b > 1.5;
  const focusNeuron = (g: number, i: number) => {
    onSelect({ kind: 'neuron', layer: g, index: i });
  };
  return (
    <>
      <color attach="background" args={[palette.bg]} />
      {effects && <fog attach="fog" args={[palette.bg, fogRange[0], fogRange[1]]} />}
      <ambientLight intensity={0.75} />
      <directionalLight position={[6, 10, 8]} intensity={1.4} />
      <directionalLight position={[-8, -4, -6]} intensity={0.35} />

      <EdgeLines layout={layout} edges={props.edges} shown={props.shown} palette={palette} onSelect={onSelect} onHover={props.onHover} />
      <Emphasised layout={layout} edges={props.edges} emphasised={props.emphasised} palette={palette} />
      <Nodes layout={layout} sizes={sizes} nodeVals={props.nodeVals} dimmed={props.dimmed} ablated={props.ablated}
        palette={palette} onSelect={onSelect} onHover={props.onHover} focusNeuron={focusNeuron} quality={quality} />
      {effects && quality !== 'low' && !paper && (
        <Glow layout={layout} sizes={sizes} nodeVals={props.nodeVals} dimmed={props.dimmed} ablated={props.ablated} palette={palette} />
      )}
      <Pulses layout={layout} edges={props.edges} pulses={props.pulses} pulseDir={props.pulseDir} palette={palette} reduceMotion={reduceMotion} />
      <SelectionMarks layout={layout} selection={selection} sizes={sizes} palette={palette} ablated={props.ablated} />

      {layerLabels.map((label, g) => (
        <Html key={g} position={[layout.layerX[g], layout.center[1] + layout.extents[g].y + 1.1, 0]} center zIndexRange={[10, 0]}>
          <button type="button" onClick={() => { onSelect({ kind: 'layer', layer: g }); onFocusLayer(g); }}
            className="whitespace-nowrap text-[11px] font-semibold px-2 py-0.5 rounded-md border"
            style={{ background: 'var(--bg-card)', borderColor: selection?.kind === 'layer' && selection.layer === g ? 'var(--select)' : 'var(--border-soft)', color: palette.text }}
            title={`Select and focus ${label}`} aria-label={`Select and focus ${label}`}>
            {label} <span style={{ color: 'var(--text-faint)', fontWeight: 400 }}>{sizes[g]}</span>
          </button>
        </Html>
      ))}

      <OrbitControls makeDefault enableDamping dampingFactor={0.1} enablePan screenSpacePanning zoomToCursor minDistance={2} maxDistance={220} />
      <CameraRig goal={goal} reduceMotion={reduceMotion} />
    </>
  );
}
