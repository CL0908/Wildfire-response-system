import { BufferAttribute, BufferGeometry, Points, ShaderMaterial, Vector2 } from 'three'
import type { Detection } from './fires'
import { PHASE_BURNING, PHASE_SCORCH } from './fireReplay'
import { cellCentreScene, latLonToScene, sampleHeight } from './terrain'
import type { SceneRect, TerrainMeta } from './terrain'
import type { FireScale } from './scale'

/**
 * The replay's detections as one draw call: a burning one is a glowing ember that fades over
 * its age, and it leaves a dark scorch mark that stays. Colours are written straight to the
 * canvas (sRGB), so Landscape's tone mapping and haze do not touch them and the data layers
 * do not get tone mapping.
 */

export type FireSpot = Pick<Detection, 'lon' | 'lat' | 'frp' | 'row' | 'col'> & Partial<Pick<Detection, 'inside'>>

// The scene-metre values (lift, ember width, pull) and the px clamps come from the scale's
// `fire` values (lib/scale.ts). In the window: Landscape lift 200 clears the trees (candidate dots
// sit at 190), data layers hug the surface at 45; an ember is 150 m wide at size factor 1, about
// 10 px at the default framing on a 1440 px window, and 9 to 22 px wide whatever the zoom.
// The size factor runs from SIZE_MIN to SIZE_MAX.
const SIZE_MIN = 0.6
const SIZE_MAX = 2.1
// Detections outside the park are drawn at this opacity and part-desaturated.
const OUTSIDE_ALPHA = 0.5
const OUTSIDE_COLOUR = 0.6

/** Size factor from sqrt(frp): linear between the faintest and the strongest detection. */
export function sizeFactors(spots: readonly FireSpot[]): Float32Array {
  const out = new Float32Array(spots.length)
  let lo = Infinity
  let hi = -Infinity
  for (const s of spots) {
    const r = Math.sqrt(Math.max(s.frp, 0))
    lo = Math.min(lo, r)
    hi = Math.max(hi, r)
  }
  const span = hi - lo || 1
  spots.forEach((s, i) => {
    const t = (Math.sqrt(Math.max(s.frp, 0)) - lo) / span
    out[i] = SIZE_MIN + (SIZE_MAX - SIZE_MIN) * t
  })
  return out
}

/**
 * Scene x, z of a detection, through the same georeferencing as the terrain mesh. Without mesh
 * bounds the centre of its grid cell is used.
 */
export function spotScene(
  spot: FireSpot,
  meta: TerrainMeta,
  grid: { rows: number; cols: number },
  rect: SceneRect,
): { x: number; z: number } {
  return meta.bounds
    ? latLonToScene(spot.lat, spot.lon, meta.bounds, meta)
    : cellCentreScene(spot.row, spot.col, grid, rect)
}

const VERTEX = /* glsl */ `
uniform float uLift;
uniform float uWorld;
uniform float uView;   // drawing-buffer height in px
uniform float uPR;     // device pixels per CSS px
uniform float uMinPx;
uniform float uMaxPx;
uniform float uMinScorch;
uniform float uPull;
attribute float aSize;
attribute float aInside;
attribute vec2 aState; // phase (0 pending, 1 burning, 2 scorch), age
varying float vPhase;
varying float vAge;
varying float vScorchR;
varying float vInside;

void main() {
  vInside = aInside;
  vPhase = aState.x;
  vAge = aState.y;
  if (vPhase < 0.5) {
    gl_PointSize = 0.0;
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vScorchR = 0.0;
    return;
  }
  vec4 mv = modelViewMatrix * vec4(position + vec3(0.0, uLift, 0.0), 1.0);
  float dist = length(mv.xyz);
  mv.xyz -= normalize(mv.xyz) * uPull;
  gl_Position = projectionMatrix * mv;
  float scale = projectionMatrix[1][1] * 0.5 * uView;
  float base = clamp(uWorld * scale / dist, uMinPx * uPR, uMaxPx * uPR);
  float scorchPx = max(base * aSize * 0.6, uMinScorch * uPR);
  float px = scorchPx;
  if (vPhase < 1.5) {
    float emberPx = base * aSize * mix(1.0, 0.7, vAge);
    px = max(emberPx, scorchPx);
  }
  gl_PointSize = px;
  vScorchR = scorchPx / px;
}
`

const FRAGMENT = /* glsl */ `
varying float vPhase;
varying float vAge;
varying float vScorchR;
varying float vInside;
uniform float uOutsideAlpha;
uniform float uOutsideColour;

void main() {
  float d = length(gl_PointCoord * 2.0 - 1.0);
  if (d > 1.0) discard;

  // Scorch: charcoal-brown disc with a faint warm rim. It fades in as the ember dies.
  float scorchA = vPhase > 1.5 ? 1.0 : smoothstep(0.45, 1.0, vAge);
  float r = d / max(vScorchR, 0.001);
  float disc = 1.0 - smoothstep(0.86, 1.0, r);
  float rim = smoothstep(0.55, 0.85, r) * disc;
  vec3 scorchCol = mix(vec3(0.13, 0.095, 0.075), vec3(0.58, 0.29, 0.13), rim * 0.75);
  float sa = disc * scorchA * 0.92;

  // Ember: yellow-white core, orange body, red-orange edge and a soft halo. Quick rise, slow decay.
  float eA = 0.0;
  vec3 eCol = vec3(1.0, 0.3, 0.06);
  if (vPhase < 1.5) {
    float life = smoothstep(0.0, 0.04, vAge) * (1.0 - smoothstep(0.35, 1.0, vAge));
    float core = 1.0 - smoothstep(0.0, 0.42, d);
    float body = 1.0 - smoothstep(0.4, 0.8, d);
    float halo = pow(1.0 - smoothstep(0.0, 1.0, d), 2.2);
    eCol = mix(vec3(1.0, 0.27, 0.05), vec3(1.0, 0.6, 0.1), body);
    eCol = mix(eCol, vec3(1.0, 0.96, 0.72), core);
    eA = clamp(max(body, halo * 0.7) * life, 0.0, 1.0);
  }

  float a = eA + sa * (1.0 - eA);
  if (a < 0.01) discard;
  vec3 col = (eCol * eA + scorchCol * sa * (1.0 - eA)) / a;
  if (vInside < 0.5) {
    col = mix(vec3(dot(col, vec3(0.299, 0.587, 0.114))), col, uOutsideColour);
    a *= uOutsideAlpha;
  }
  gl_FragColor = vec4(col, a);
}
`

export interface FireMarks {
  points: Points
  /** Per-detection phase and age from the replay (same order as the detections given to build). */
  set(phase: Uint8Array, age: Float32Array): void
  setLandscape(on: boolean): void
  dispose(): void
}

export function buildFireMarks(
  spots: readonly FireSpot[],
  heights: Float32Array,
  meta: TerrainMeta,
  grid: { rows: number; cols: number },
  rect: SceneRect,
  /** The scene's vertical exaggeration, the one the terrain mesh uses. */
  exaggeration: number,
  look: FireScale,
): FireMarks {
  const n = spots.length
  const positions = new Float32Array(n * 3)
  spots.forEach((s, i) => {
    const { x, z } = spotScene(s, meta, grid, rect)
    positions[i * 3] = x
    positions[i * 3 + 1] = sampleHeight(heights, meta, x, z) * exaggeration
    positions[i * 3 + 2] = z
  })
  const state = new Float32Array(n * 2)
  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new BufferAttribute(positions, 3))
  geometry.setAttribute('aSize', new BufferAttribute(sizeFactors(spots), 1))
  geometry.setAttribute('aInside', new BufferAttribute(Float32Array.from(spots, (s) => (s.inside === false ? 0 : 1)), 1))
  const stateAttr = new BufferAttribute(state, 2)
  geometry.setAttribute('aState', stateAttr)

  const view = new Vector2()
  const material = new ShaderMaterial({
    vertexShader: VERTEX,
    fragmentShader: FRAGMENT,
    uniforms: {
      uLift: { value: look.liftData },
      uWorld: { value: look.worldSize },
      uView: { value: 1 },
      uPR: { value: 1 },
      uMinPx: { value: look.minPx },
      uMaxPx: { value: look.maxPx },
      uMinScorch: { value: look.minScorchPx },
      uPull: { value: look.pull },
      uOutsideAlpha: { value: OUTSIDE_ALPHA },
      uOutsideColour: { value: OUTSIDE_COLOUR },
    },
    transparent: true,
    depthWrite: false,
    // No fog or tone mapping chunks are included, so neither applies.
    fog: false,
  })
  const points = new Points(geometry, material)
  points.frustumCulled = false
  points.renderOrder = 1
  points.onBeforeRender = (renderer) => {
    renderer.getDrawingBufferSize(view)
    material.uniforms.uView.value = view.y
    material.uniforms.uPR.value = renderer.getPixelRatio()
  }

  return {
    points,
    set(phase, age) {
      for (let i = 0; i < n; i++) {
        state[i * 2] = phase[i]
        state[i * 2 + 1] = phase[i] === PHASE_BURNING || phase[i] === PHASE_SCORCH ? age[i] : 0
      }
      stateAttr.needsUpdate = true
    },
    setLandscape(on) {
      material.uniforms.uLift.value = on ? look.liftLandscape : look.liftData
    },
    dispose() {
      geometry.dispose()
      material.dispose()
    },
  }
}
