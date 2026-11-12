import {
  BufferGeometry,
  CanvasTexture,
  Color,
  DirectionalLight,
  Group,
  InstancedMesh,
  LinearFilter,
  LinearMipmapLinearFilter,
  Matrix4,
  Mesh,
  MeshLambertMaterial,
  Quaternion,
  OrthographicCamera,
  PlaneGeometry,
  ClampToEdgeWrapping,
  RepeatWrapping,
  ShaderMaterial,
  SRGBColorSpace,
  Vector3,
  BufferAttribute,
} from 'three'
import { LAND_CLASSES, WATER_CODE, pixelAt, type LandcoverData } from './landcover'
import { VERTICAL_EXAGGERATION, sampleHeight, type TerrainMeta } from './terrain'

/**
 * The Landscape layer: ground colours painted from land cover + terrain, a late-morning sun, a
 * daytime sky gradient and strata for the side walls. Everything is procedural and deterministic.
 */

// Sun: compass azimuth measured clockwise from north (135 = south-east), elevation above the horizon.
const SUN_AZIMUTH = (135 * Math.PI) / 180
const SUN_ELEVATION = (52 * Math.PI) / 180

/** Unit vector from the scene centre towards the sun (x = east, y = up, z = south). */
export function sunDirection(): Vector3 {
  const ce = Math.cos(SUN_ELEVATION)
  return new Vector3(Math.sin(SUN_AZIMUTH) * ce, Math.sin(SUN_ELEVATION), -Math.cos(SUN_AZIMUTH) * ce)
}

/** Light and atmosphere values. Intensities follow three's convention: output = albedo * I * cos / PI. */
export const LANDSCAPE_LIGHT = {
  sunColor: '#fff4e0',
  sunIntensity: 3.9,
  skyColor: '#b4d0ee',
  groundColor: '#6b6a45',
  hemiIntensity: 1.8,
  exposure: 1.1,
  fogColor: '#c4d9ec',
  fogDensity: 0.000012,
  /** How much of the sun's light a shadowed point loses (1 = fully dark, hemisphere light only). */
  shadowStrength: 0.75,
}

// ---------------------------------------------------------------- noise

function hash2(ix: number, iy: number, seed: number): number {
  let h = Math.imul(ix, 374761393) ^ Math.imul(iy, 668265263) ^ Math.imul(seed, 1442695041)
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  h ^= h >>> 16
  return (h >>> 0) / 4294967295
}

/** Smooth value noise in 0..1; deterministic for a given seed. */
function vnoise(x: number, y: number, seed: number): number {
  const ix = Math.floor(x)
  const iy = Math.floor(y)
  const fx = x - ix
  const fy = y - iy
  const sx = fx * fx * (3 - 2 * fx)
  const sy = fy * fy * (3 - 2 * fy)
  const a = hash2(ix, iy, seed)
  const b = hash2(ix + 1, iy, seed)
  const c = hash2(ix, iy + 1, seed)
  const d = hash2(ix + 1, iy + 1, seed)
  return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy
}

// ---------------------------------------------------------------- ground

/** Elevation ramp for tree cover (sRGB): warm and light in the lowland, dark and cool on the heights. */
const TREE_LOW: [number, number, number] = [74, 122, 50]
const TREE_MID: [number, number, number] = [40, 104, 58]
const TREE_HIGH: [number, number, number] = [22, 72, 66]
const ELEV_LOW_M = 150
const ELEV_HIGH_M = 900
const SPARSE_TREE: [number, number, number] = [158, 170, 82] // grass showing between the trees
const DRY: [number, number, number] = [140, 146, 78]

const GROUND_SCALE = 2 // texels per terrain pixel along each axis

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

function boxBlur(src: Float32Array, rows: number, cols: number, radius: number): Float32Array {
  const tmp = new Float32Array(src.length)
  const out = new Float32Array(src.length)
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      let sum = 0
      let n = 0
      for (let k = -radius; k <= radius; k++) {
        const cc = c + k
        if (cc >= 0 && cc < cols) {
          sum += src[r * cols + cc]
          n++
        }
      }
      tmp[r * cols + c] = sum / n
    }
  }
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      let sum = 0
      let n = 0
      for (let k = -radius; k <= radius; k++) {
        const rr = r + k
        if (rr >= 0 && rr < rows) {
          sum += tmp[rr * cols + c]
          n++
        }
      }
      out[r * cols + c] = sum / n
    }
  }
  return out
}

/** [1 2 1] / 4 in both directions: softens class edges over about one pixel. */
function smooth121(src: Float32Array, rows: number, cols: number): Float32Array {
  const tmp = new Float32Array(src.length)
  const out = new Float32Array(src.length)
  const at = (a: Float32Array, r: number, c: number) =>
    a[Math.min(rows - 1, Math.max(0, r)) * cols + Math.min(cols - 1, Math.max(0, c))]
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      tmp[r * cols + c] = (at(src, r, c - 1) + 2 * at(src, r, c) + at(src, r, c + 1)) / 4
    }
  }
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      out[r * cols + c] = (at(tmp, r - 1, c) + 2 * at(tmp, r, c) + at(tmp, r + 1, c)) / 4
    }
  }
  return out
}

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v)
const mix = (a: number, b: number, t: number) => a + (b - a) * t

/** Forest colour (sRGB 0..255) at `t` = 0 (low) .. 1 (high) along the elevation ramp. */
function forestRamp(t: number): [number, number, number] {
  const lo = t < 0.5 ? TREE_LOW : TREE_MID
  const hi = t < 0.5 ? TREE_MID : TREE_HIGH
  const u = t < 0.5 ? t * 2 : (t - 0.5) * 2
  return [mix(lo[0], hi[0], u), mix(lo[1], hi[1], u), mix(lo[2], hi[2], u)]
}

export interface GroundTextures {
  color: HTMLCanvasElement
  /** Green channel = roughness (water is glossier). */
  rough: HTMLCanvasElement
}

/**
 * Paints the ground at GROUND_SCALE texels per terrain pixel (canvas top = north). Inputs:
 * class and tree fraction from `lc` (all forest when null), and elevation, slope, ridges and
 * hollows from the heights.
 */
export function paintGround(heights: Float32Array, meta: TerrainMeta, lc: LandcoverData | null): GroundTextures {
  const { rows, cols } = meta
  const n = rows * cols
  const dx = meta.width_m / (cols - 1)
  const dy = meta.height_m / (rows - 1)

  const slope = new Float32Array(n) // degrees
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const r0 = Math.max(0, r - 1)
      const r1 = Math.min(rows - 1, r + 1)
      const c0 = Math.max(0, c - 1)
      const c1 = Math.min(cols - 1, c + 1)
      const gx = (heights[r * cols + c1] - heights[r * cols + c0]) / ((c1 - c0) * dx)
      const gy = (heights[r1 * cols + c] - heights[r0 * cols + c]) / ((r1 - r0) * dy)
      slope[r * cols + c] = (Math.atan(Math.hypot(gx, gy)) * 180) / Math.PI
    }
  }
  const mean = boxBlur(heights, rows, cols, 4)
  const ridge = new Float32Array(n) // metres above the local mean; negative in hollows
  for (let i = 0; i < n; i++) ridge[i] = heights[i] - mean[i]

  const treeFrac = new Float32Array(n)
  const weights = LAND_CLASSES.map(() => new Float32Array(n))
  const treeK = LAND_CLASSES.findIndex((c) => c.code === 10)
  const waterK = LAND_CLASSES.findIndex((c) => c.code === WATER_CODE)
  if (lc) {
    for (let i = 0; i < n; i++) {
      treeFrac[i] = lc.tree[i] / 255
      const k = LAND_CLASSES.findIndex((c) => c.code === lc.classes[i])
      weights[k >= 0 ? k : treeK][i] = 1
    }
  } else {
    treeFrac.fill(0.85)
    weights[treeK].fill(1)
  }
  const soft = weights.map((w, k) => (k === treeK && !lc ? w : smooth121(w, rows, cols)))
  const flat = LAND_CLASSES.map((c) => hexToRgb(c.color))

  const W = cols * GROUND_SCALE
  const H = rows * GROUND_SCALE
  const color = document.createElement('canvas')
  color.width = W
  color.height = H
  const rough = document.createElement('canvas')
  rough.width = W
  rough.height = H
  const cctx = color.getContext('2d')!
  const rctx = rough.getContext('2d')!
  const cImg = cctx.createImageData(W, H)
  const rImg = rctx.createImageData(W, H)

  const K = LAND_CLASSES.length
  const col: [number, number, number] = [0, 0, 0]
  for (let ty = 0; ty < H; ty++) {
    // Texel centre in terrain-pixel coordinates (pixel centres at integers).
    const gy = (ty + 0.5) / GROUND_SCALE - 0.5
    const gyc = Math.min(rows - 1, Math.max(0, gy))
    const r0 = Math.min(rows - 2, Math.floor(gyc))
    const fy = gyc - r0
    for (let tx = 0; tx < W; tx++) {
      const gx = (tx + 0.5) / GROUND_SCALE - 0.5
      const gxc = Math.min(cols - 1, Math.max(0, gx))
      const c0 = Math.min(cols - 2, Math.floor(gxc))
      const fx = gxc - c0
      const i00 = r0 * cols + c0
      const i01 = i00 + 1
      const i10 = i00 + cols
      const i11 = i10 + 1
      const w00 = (1 - fx) * (1 - fy)
      const w01 = fx * (1 - fy)
      const w10 = (1 - fx) * fy
      const w11 = fx * fy
      const bil = (a: Float32Array) => a[i00] * w00 + a[i01] * w01 + a[i10] * w10 + a[i11] * w11

      const h = bil(heights)
      const sl = bil(slope)
      const rg = bil(ridge)
      const tf = bil(treeFrac)
      const t = clamp01((h - ELEV_LOW_M) / (ELEV_HIGH_M - ELEV_LOW_M))

      // Forest colour from elevation, then lighter where the canopy is thin, drier on steep ground and ridges.
      const tree = forestRamp(t)
      const sparse = clamp01((0.9 - tf) / 0.6) * 0.55
      const dry = clamp01((sl - 16) / 26) * 0.28 + clamp01(rg / 30) * 0.18
      for (let k = 0; k < 3; k++) {
        tree[k] = mix(tree[k], SPARSE_TREE[k], sparse)
        tree[k] = mix(tree[k], DRY[k], dry)
      }

      let wSum = 0
      col[0] = col[1] = col[2] = 0
      let wWater = 0
      for (let k = 0; k < K; k++) {
        const wk = bil(soft[k])
        if (wk <= 0) continue
        wSum += wk
        if (k === waterK) wWater = wk
        const src = k === treeK ? tree : flat[k]
        col[0] += src[0] * wk
        col[1] += src[1] * wk
        col[2] += src[2] * wk
      }
      if (wSum > 0) {
        col[0] /= wSum
        col[1] /= wSum
        col[2] /= wSum
      }

      // Fine canopy texture + broader patches (tone and slight hue), calmer on water.
      const calm = 1 - 0.85 * wWater
      const fine = vnoise(gx * 1.1, gy * 1.1, 7) * 0.65 + vnoise(gx * 2.3, gy * 2.3, 19) * 0.35
      const patch = vnoise(gx * 0.2, gy * 0.2, 31) * 0.6 + vnoise(gx * 0.55, gy * 0.55, 43) * 0.4
      const lum =
        (1 + (fine - 0.5) * 0.32 * calm + (patch - 0.5) * 0.26 * calm) * (1 - clamp01(-rg / 40) * 0.16)
      const hue = (patch - 0.5) * 12 * calm
      const o = (ty * W + tx) * 4
      cImg.data[o] = Math.round(Math.min(255, Math.max(0, (col[0] + hue) * lum)))
      cImg.data[o + 1] = Math.round(Math.min(255, Math.max(0, col[1] * lum)))
      cImg.data[o + 2] = Math.round(Math.min(255, Math.max(0, (col[2] - hue * 0.6) * lum)))
      cImg.data[o + 3] = 255
      const rv = Math.round(255 * (1 - 0.68 * clamp01(wWater)))
      rImg.data[o] = rv
      rImg.data[o + 1] = rv
      rImg.data[o + 2] = rv
      rImg.data[o + 3] = 255
    }
  }
  cctx.putImageData(cImg, 0, 0)
  rctx.putImageData(rImg, 0, 0)
  return { color, rough }
}

/** Ground textures as three textures; `channel` 1 expects the `uv1` attribute from `groundUVs`. */
export function groundTextures(
  t: GroundTextures,
  maxAnisotropy: number,
): { map: CanvasTexture; roughnessMap: CanvasTexture } {
  const map = new CanvasTexture(t.color)
  map.colorSpace = SRGBColorSpace
  const roughnessMap = new CanvasTexture(t.rough)
  for (const tex of [map, roughnessMap]) {
    tex.channel = 1
    tex.generateMipmaps = true
    tex.minFilter = LinearMipmapLinearFilter
    tex.magFilter = LinearFilter
    tex.anisotropy = maxAnisotropy
    tex.wrapS = tex.wrapT = ClampToEdgeWrapping
  }
  return { map, roughnessMap }
}

/** Texture coordinates over the whole mesh: vertex (r, c) samples the centre of terrain pixel (r, c). */
export function groundUVs(meta: { rows: number; cols: number }): Float32Array {
  const { rows, cols } = meta
  const out = new Float32Array(rows * cols * 2)
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c
      out[i * 2] = (c + 0.5) / cols
      out[i * 2 + 1] = 1 - (r + 0.5) / rows
    }
  }
  return out
}

// ---------------------------------------------------------------- side walls

/** Horizontal rock strata in warm browns; v runs from the block's base (0) to its highest point (1). */
export function strataTexture(maxAnisotropy: number): CanvasTexture {
  const W = 256
  const H = 512
  const canvas = document.createElement('canvas')
  canvas.width = W
  canvas.height = H
  const ctx = canvas.getContext('2d')!
  const img = ctx.createImageData(W, H)
  const bands = ['#5a3e2b', '#6d4a32', '#7e5839', '#8c6743', '#4b3526', '#98714b', '#68493a', '#765036'].map(hexToRgb)
  // Band boundaries along the height, 5 to 22 rows thick, from a fixed sequence.
  const bounds: { end: number; rgb: [number, number, number] }[] = []
  let y = 0
  for (let i = 0; y < H; i++) {
    y += 5 + Math.floor(hash2(i, 3, 5) * 17)
    bounds.push({ end: y, rgb: bands[Math.floor(hash2(i, 9, 11) * bands.length)] })
  }
  for (let py = 0; py < H; py++) {
    for (let px = 0; px < W; px++) {
      // Slightly wavy boundaries.
      const wy = py + (vnoise(px * 0.03, py * 0.05, 2) - 0.5) * 7
      const band = bounds.find((b) => wy < b.end) ?? bounds[bounds.length - 1]
      const grain = 0.88 + vnoise(px * 0.35, py * 0.9, 13) * 0.24
      const o = (py * W + px) * 4
      // Canvas top is v = 1 (the highest ground), so the lightest, driest bands sit up there.
      const up = 0.9 + (1 - py / H) * 0.18
      img.data[o] = Math.min(255, band.rgb[0] * grain * up)
      img.data[o + 1] = Math.min(255, band.rgb[1] * grain * up)
      img.data[o + 2] = Math.min(255, band.rgb[2] * grain * up)
      img.data[o + 3] = 255
    }
  }
  ctx.putImageData(img, 0, 0)
  const tex = new CanvasTexture(canvas)
  tex.colorSpace = SRGBColorSpace
  tex.wrapS = RepeatWrapping
  tex.wrapT = ClampToEdgeWrapping
  tex.anisotropy = maxAnisotropy
  return tex
}

/** Adds normals and strata UVs to the wall geometry: u along the wall, v = height above `baseY`. */
export function addSkirtStrataUVs(g: BufferGeometry, baseY: number, topY: number): void {
  const pos = g.getAttribute('position')
  const uv = new Float32Array(pos.count * 2)
  for (let i = 0; i < pos.count; i++) {
    uv[i * 2] = (pos.getX(i) + pos.getZ(i)) / 10000
    uv[i * 2 + 1] = (pos.getY(i) - baseY) / (topY - baseY)
  }
  g.setAttribute('uv', new BufferAttribute(uv, 2))
  g.computeVertexNormals()
}

// ---------------------------------------------------------------- sky

const SKY_VERTEX = /* glsl */ `
varying vec2 vNdc;
void main() {
  vNdc = position.xy;
  gl_Position = vec4(position.xy, 0.9999, 1.0);
}`

// The gradient is a function of the view ray's elevation plus a share of how far the camera
// looks down, so the horizon band stays near the top of the picture even in the default
// high-angle view. There are no clouds: the camera never sees much above the horizon, and
// anything laid on the sky would have to be projected and would stretch as the camera moves.
const SKY_FRAGMENT = /* glsl */ `
uniform mat4 uInvProj;
uniform mat4 uCamWorld;
uniform vec3 uSunDir;
uniform float uOffsetDeg;
uniform vec3 uHaze;
uniform vec3 uLow;
uniform vec3 uHorizon;
uniform vec3 uMid;
uniform vec3 uZenith;
uniform vec3 uGlow;
varying vec2 vNdc;

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

void main() {
  vec4 v = uInvProj * vec4(vNdc, 1.0, 1.0);
  vec3 dirView = normalize(v.xyz / v.w);
  vec3 dir = normalize((uCamWorld * vec4(dirView, 0.0)).xyz);
  float e = degrees(asin(clamp(dir.y, -1.0, 1.0)));
  float d = e + uOffsetDeg;

  // Blue deepening towards the zenith, paling to hazy white-blue at the horizon; below it,
  // pale haze easing to a soft blue.
  vec3 c = mix(uHorizon, uMid, smoothstep(0.0, 11.0, d));
  c = mix(c, uZenith, smoothstep(8.0, 62.0, d));
  c = mix(c, uHaze, 1.0 - smoothstep(-4.0, 0.0, d));
  c = mix(c, uLow, 1.0 - smoothstep(-40.0, -4.0, d));

  // Sun glow and disc (the disc rarely shows: the default view looks down at the land).
  float cs = max(dot(dir, uSunDir), 0.0);
  c += uGlow * (pow(cs, 8.0) * 0.16 + pow(cs, 120.0) * 0.55);

  c = mix(c, vec3(1.0, 0.98, 0.93), smoothstep(0.99985, 0.99996, cs));

  c += (hash(gl_FragCoord.xy) + hash(gl_FragCoord.yx + 7.3) - 1.0) / 255.0;
  gl_FragColor = vec4(c, 1.0);
  #include <colorspace_fragment>
}`

export interface Sky {
  mesh: Mesh
  dispose: () => void
}

/** A full-screen backdrop drawn first; it ignores depth and follows the camera's view rays. */
export function buildSky(): Sky {
  const sunDir = sunDirection()
  const lin = (hex: string) => new Color(hex)
  const material = new ShaderMaterial({
    vertexShader: SKY_VERTEX,
    fragmentShader: SKY_FRAGMENT,
    depthTest: false,
    depthWrite: false,
    fog: false,
    uniforms: {
      uInvProj: { value: new Matrix4() },
      uCamWorld: { value: new Matrix4() },
      uSunDir: { value: sunDir },
      uOffsetDeg: { value: 0 },
      uHaze: { value: lin(LANDSCAPE_LIGHT.fogColor) },
      uLow: { value: lin('#8db9e6') },
      uHorizon: { value: lin('#cfe1f0') },
      uMid: { value: lin('#4a8fdc') },
      uZenith: { value: lin('#1d57b5') },
      uGlow: { value: lin('#fff1cf') },
    },
  })
  const mesh = new Mesh(new PlaneGeometry(2, 2), material)
  mesh.frustumCulled = false
  mesh.renderOrder = -1000
  mesh.raycast = () => {}
  const forward = new Vector3()
  mesh.onBeforeRender = (_r, _s, camera) => {
    const u = material.uniforms
    ;(u.uInvProj.value as Matrix4).copy(camera.projectionMatrixInverse)
    ;(u.uCamWorld.value as Matrix4).copy(camera.matrixWorld)
    camera.getWorldDirection(forward)
    const down = (Math.asin(Math.min(1, Math.max(-1, -forward.y))) * 180) / Math.PI
    u.uOffsetDeg.value = Math.max(0, down) * 0.9
  }
  return {
    mesh,
    dispose: () => {
      mesh.geometry.dispose()
      material.dispose()
    },
  }
}

// ---------------------------------------------------------------- shadows

/**
 * Points the sun at the origin and fits its orthographic shadow camera tightly round the terrain
 * block (the box from `baseY` to `topY` under the mesh outline).
 */
export function fitSun(sun: DirectionalLight, halfW: number, halfH: number, baseY: number, topY: number, mapSize: number): void {
  sun.position.copy(sunDirection()).multiplyScalar(30000)
  sun.target.position.set(0, 0, 0)
  sun.updateMatrixWorld()
  sun.target.updateMatrixWorld()
  const cam = new OrthographicCamera(-1, 1, 1, -1, 1, 10)
  cam.position.copy(sun.position)
  cam.lookAt(0, 0, 0)
  cam.updateMatrixWorld()
  const inv = cam.matrixWorldInverse
  let l = Infinity
  let r = -Infinity
  let b = Infinity
  let t = -Infinity
  let nearZ = -Infinity
  let farZ = Infinity
  const p = new Vector3()
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      for (const y of [baseY, topY]) {
        p.set(sx * halfW, y, sz * halfH).applyMatrix4(inv)
        l = Math.min(l, p.x)
        r = Math.max(r, p.x)
        b = Math.min(b, p.y)
        t = Math.max(t, p.y)
        nearZ = Math.max(nearZ, p.z)
        farZ = Math.min(farZ, p.z)
      }
    }
  }
  const pad = 80
  const sc = sun.shadow.camera as OrthographicCamera
  sc.left = l - pad
  sc.right = r + pad
  sc.bottom = b - pad
  sc.top = t + pad
  sc.near = Math.max(1, -nearZ - pad)
  sc.far = -farZ + pad
  sc.updateProjectionMatrix()
  sun.shadow.mapSize.set(mapSize, mapSize)
  sun.shadow.bias = -0.0004
  sun.shadow.normalBias = 9
  sun.shadow.radius = 3.5
  sun.shadow.intensity = LANDSCAPE_LIGHT.shadowStrength
}

// ---------------------------------------------------------------- canopy

// Each instance is one broadleaf tree: an irregular two-lobe crown on a short trunk. Sizes are real
// ground metres; heights are scaled by the vertical exaggeration like the terrain, so a tree keeps
// its proportions against the slopes. Crowns and trunks are two InstancedMeshes sharing one set of
// instance matrices (trunks need their own brown, which a per-instance crown colour would tint).
const CANOPY_SEED = 4171
const CANOPY_SPACING_M = 42
const CANOPY_SPACING_COARSE_M = 60
const CANOPY_MAX = 60000
const CROWN_DIAMETER_M = 46
const TREE_HEIGHT_M = 28
const TREE_MAX_SCENE_M = 110
const TREE_SINK = 0.05 // share of the tree's height below the surface (hides the trunk foot on slopes)
/** WorldCover classes that never carry trees: cropland, built-up, water. */
const NO_CANOPY = new Set([40, 50, WATER_CODE])

// Unit tree: 1 wide (crown diameter) and 1 tall, standing on y = 0.
const ICO_T = (1 + Math.sqrt(5)) / 2
const ICO_VERTS: [number, number, number][] = [
  [-1, ICO_T, 0], [1, ICO_T, 0], [-1, -ICO_T, 0], [1, -ICO_T, 0],
  [0, -1, ICO_T], [0, 1, ICO_T], [0, -1, -ICO_T], [0, 1, -ICO_T],
  [ICO_T, 0, -1], [ICO_T, 0, 1], [-ICO_T, 0, -1], [-ICO_T, 0, 1],
].map(([x, y, z]) => {
  const l = Math.hypot(x, y, z)
  return [x / l, y / l, z / l] as [number, number, number]
})
const ICO_FACES = [
  0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
  3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
]
// [centre x, y, z, radius x, y, z, jitter seed]: a wide flattened main lobe and a smaller one on top.
const CROWN_LOBES: [number, number, number, number, number, number, number][] = [
  [-0.02, 0.6, 0.02, 0.5, 0.29, 0.47, 1],
  [0.2, 0.75, 0.13, 0.29, 0.21, 0.28, 2],
]
const TRUNK_SIDES = 5

/** Crown (20-face lobes, jittered, smooth normals, light on top and dark underneath) and trunk geometries. */
function treeGeometries(): { crown: BufferGeometry; trunk: BufferGeometry } {
  const pos: number[] = []
  const col: number[] = []
  const idx: number[] = []
  for (const [cx, cy, cz, rx, ry, rz, seed] of CROWN_LOBES) {
    const base = pos.length / 3
    ICO_VERTS.forEach(([x, y, z], v) => {
      const j = 0.78 + 0.44 * hash2(v, seed, 911)
      const px = cx + x * rx * j
      const py = cy + y * ry * (0.85 + 0.3 * hash2(v, seed, 913))
      const pz = cz + z * rz * j
      pos.push(px, py, pz)
      const k = 0.5 + 0.5 * clamp01((py - 0.34) / 0.6)
      col.push(k, k, k)
    })
    for (const f of ICO_FACES) idx.push(base + f)
  }
  const crown = new BufferGeometry()
  crown.setAttribute('position', new BufferAttribute(new Float32Array(pos), 3))
  crown.setAttribute('color', new BufferAttribute(new Float32Array(col), 3))
  crown.setIndex(idx)
  crown.computeVertexNormals()

  const tp: number[] = []
  const ti: number[] = []
  const tc: number[] = []
  const y0 = -0.08
  const y1 = 0.62
  for (let i = 0; i < TRUNK_SIDES; i++) {
    const a = (i / TRUNK_SIDES) * Math.PI * 2
    tp.push(Math.cos(a) * 0.04, y0, Math.sin(a) * 0.04, Math.cos(a) * 0.026, y1, Math.sin(a) * 0.026)
    tc.push(0.6, 0.6, 0.6, 1, 1, 1)
  }
  for (let i = 0; i < TRUNK_SIDES; i++) {
    const n = (i + 1) % TRUNK_SIDES
    ti.push(i * 2, n * 2, i * 2 + 1, i * 2 + 1, n * 2, n * 2 + 1)
  }
  const trunk = new BufferGeometry()
  trunk.setAttribute('position', new BufferAttribute(new Float32Array(tp), 3))
  trunk.setAttribute('color', new BufferAttribute(new Float32Array(tc), 3))
  trunk.setIndex(ti)
  trunk.computeVertexNormals()
  return { crown, trunk }
}

export interface Canopy {
  mesh: Group
  count: number
  triangles: number
  dispose: () => void
}

/**
 * Trees on a jittered grid. A candidate is kept with the probability of the local tree-cover
 * fraction (0.85 everywhere when `lc` is null) and never on non-tree classes; thin cover gives
 * smaller trees. Deterministic: the same inputs always give the same forest.
 */
export function buildCanopy(heights: Float32Array, meta: TerrainMeta, lc: LandcoverData | null, coarse: boolean): Canopy {
  const spacing = coarse ? CANOPY_SPACING_COARSE_M : CANOPY_SPACING_M
  const nx = Math.floor(meta.width_m / spacing)
  const nz = Math.floor(meta.height_m / spacing)
  const { crown, trunk } = treeGeometries()
  const crownMaterial = new MeshLambertMaterial({ vertexColors: true })
  const trunkMaterial = new MeshLambertMaterial({ vertexColors: true, color: new Color('#5a4a3a') })
  const capacity = Math.min(CANOPY_MAX, nx * nz)
  const crowns = new InstancedMesh(crown, crownMaterial, capacity)
  const trunks = new InstancedMesh(trunk, trunkMaterial, capacity)
  trunks.instanceMatrix = crowns.instanceMatrix // one matrix buffer for both
  crowns.name = 'landscape-canopy-instances'
  for (const mesh of [crowns, trunks]) {
    mesh.castShadow = false
    mesh.receiveShadow = true
    mesh.frustumCulled = false
    mesh.raycast = () => {} // hover picking stays analytic on the height field
  }
  const group = new Group()
  group.name = 'landscape-trees'
  group.add(trunks, crowns)

  const m = new Matrix4()
  const q = new Quaternion()
  const up = new Vector3(0, 1, 0)
  const at = new Vector3()
  const sc = new Vector3()
  const c = new Color()
  const rgb: [number, number, number] = [0, 0, 0]
  let n = 0
  for (let j = 0; j < nz && n < capacity; j++) {
    for (let i = 0; i < nx && n < capacity; i++) {
      const x = -meta.width_m / 2 + (i + hash2(i, j, CANOPY_SEED)) * spacing
      const z = -meta.height_m / 2 + (j + hash2(i, j, CANOPY_SEED + 1)) * spacing
      const px = pixelAt(meta, x, z)
      const pi = px.row * meta.cols + px.col
      const tf = lc ? lc.tree[pi] / 255 : 0.85
      if (lc && NO_CANOPY.has(lc.classes[pi])) continue
      if (hash2(i, j, CANOPY_SEED + 2) >= tf) continue

      const thin = 0.72 + 0.28 * clamp01(tf * 1.25) // smaller trees where cover is sparse
      const emergent = hash2(i, j, CANOPY_SEED + 7) < 0.04
      const size = (1 + (hash2(i, j, CANOPY_SEED + 3) - 0.5) * 0.5) * thin * (emergent ? 1.15 : 1)
      const tall = (0.85 + hash2(i, j, CANOPY_SEED + 4) * 0.35) * (emergent ? 1.45 : 1)
      const w = CROWN_DIAMETER_M * size
      const hgt = Math.min(TREE_MAX_SCENE_M, TREE_HEIGHT_M * VERTICAL_EXAGGERATION * size * tall) // stays under the candidate dots
      const h = sampleHeight(heights, meta, x, z)
      at.set(x, h * VERTICAL_EXAGGERATION - hgt * TREE_SINK, z)
      q.setFromAxisAngle(up, hash2(i, j, CANOPY_SEED + 5) * Math.PI * 2)
      m.compose(at, q, sc.set(w, hgt, w * (0.88 + 0.24 * hash2(i, j, CANOPY_SEED + 8))))
      crowns.setMatrixAt(n, m)

      // Same forest ramp as the ground, thinned towards grass where tree cover is low; a few crowns
      // run yellower or darker, and every one gets its own value.
      const t = clamp01((h - ELEV_LOW_M) / (ELEV_HIGH_M - ELEV_LOW_M))
      const base = forestRamp(t)
      const sparse = clamp01((0.9 - tf) / 0.6) * 0.45
      const r = hash2(i, j, CANOPY_SEED + 6)
      const yellow = r < 0.12 ? 0.45 : 0
      const dark = r > 0.86 ? 0.7 : 1
      const v = (0.82 + 0.36 * hash2(i, j, CANOPY_SEED + 9)) * dark
      for (let k = 0; k < 3; k++) rgb[k] = mix(mix(base[k], SPARSE_TREE[k], sparse), DRY[k], yellow) * v
      c.setRGB(Math.min(1, rgb[0] / 255), Math.min(1, rgb[1] / 255), Math.min(1, rgb[2] / 255), SRGBColorSpace)
      crowns.setColorAt(n, c)
      n++
    }
  }
  crowns.count = n
  trunks.count = n
  crowns.instanceMatrix.needsUpdate = true
  if (crowns.instanceColor) crowns.instanceColor.needsUpdate = true
  const perTree = ((crown.index?.count ?? 0) + (trunk.index?.count ?? 0)) / 3
  return {
    mesh: group,
    count: n,
    triangles: n * perTree,
    dispose: () => {
      crowns.dispose()
      trunks.dispose()
      crown.dispose()
      trunk.dispose()
      crownMaterial.dispose()
      trunkMaterial.dispose()
    },
  }
}
