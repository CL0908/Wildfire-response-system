import { useEffect, useMemo, useRef, useState } from 'react'
import {
  ACESFilmicToneMapping,
  AmbientLight,
  BackSide,
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  Color,
  CylinderGeometry,
  DirectionalLight,
  DoubleSide,
  FogExp2,
  Group,
  HemisphereLight,
  LineBasicMaterial,
  LinearFilter,
  LinearMipmapLinearFilter,
  LineSegments,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  NoToneMapping,
  PCFSoftShadowMap,
  PerspectiveCamera,
  SRGBColorSpace,
  Scene,
  Sprite,
  SpriteMaterial,
  TOUCH,
  Vector3,
  WebGLRenderer,
} from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import CellReadout from './CellReadout'
import Legend from './Legend'
import LandscapeLegend from './LandscapeLegend'
import { ParkFireKey, ParkKey, ParkLegend } from './ParkLegend'
import ParkReadout, { type ParkHover } from './ParkReadout'
import { drawDrape, drawParkDrape, parkLayerRange } from '../lib/drape'
import { buildFireMarks, type FireMarks } from '../lib/fireMarks'
import type { Replay } from '../lib/fireReplay'
import type { Detection } from '../lib/fires'
import { LAYERS, layerRange, type StageLayer } from '../lib/layers'
import {
  type LandClass,
  type LandcoverData,
  type LandcoverState,
  landClassName,
  loadLandcover,
  matchesTerrain,
  pixelAt,
  presentClasses,
} from '../lib/landcover'
import {
  LANDSCAPE_LIGHT,
  addSkirtStrataUVs,
  bakeShadow,
  buildCanopy,
  buildSky,
  fitSun,
  groundTextures,
  groundUVs,
  paintGround,
  strataTexture,
  sunDirection,
  type Canopy,
} from '../lib/landscape'
import {
  PARK_LAYERS,
  PARK_RULES,
  parkPlacement,
  type ParkData,
  type ParkK,
  type ParkLayer,
  type ParkRule,
} from '../lib/park'
import { PARK_LINE_COLORS, parkBoundaryPositions, parkWindowBand } from '../lib/parkLines'
import { SCALE_NOTE, sceneScale, type Scale, type SceneScale } from '../lib/scale'
import {
  type TerrainState,
  buildIndices,
  buildPositions,
  buildUVs,
  cellCentreScene,
  gridSceneRect,
  loadTerrain,
  rayTerrainHit,
  sampleHeight,
  sceneToCell,
} from '../lib/terrain'
import type { Method, Results } from '../types'

interface Props {
  results: Results
  layer: StageLayer
  method: Method
  /** Fire replay: drawn only while `replayShown`; `replayT` is playback seconds and changes every frame. */
  replayShown?: boolean
  replay?: Replay | null
  replayT?: number
  replayPlaying?: boolean
  /** The detections the replay was built from, in the same order. */
  detections?: readonly Detection[] | null
  /** 'park' shows the whole park without markers, trees or data layers. The view is remounted on change. */
  scale?: Scale
  /** Whole park only: the placements and layers, what the ground shows, and the chosen sensor count and rule. */
  parkData?: ParkData | null
  parkLayer?: ParkLayer
  parkK?: ParkK
  parkRule?: ParkRule
  /** Called once if the 3D view cannot be shown (no WebGL, terrain files missing or unusable). */
  onUnavailable?: () => void
}

type Status = 'loading' | 'ready' | 'error'

const FOV = 35
const MAX_POLAR = (78 * Math.PI) / 180

// Scene metres. Masts are deliberately not to scale (a real 30 m mast would be invisible).
const MAST_HEIGHT = 360
const MAST_RADIUS = 18
const HEAD_SIZE = 250
const DOT_SIZE = 170
const LABEL_SIZE = 200
// Landscape is busy green, so its markers are bigger, taller and outlined to stand out.
const MAST_HEIGHT_L = 520
const MAST_RADIUS_L = 24
const HEAD_SIZE_L = 380
const DOT_SIZE_L = 220
const DOT_LIFT_L = 190 // dot centre above ground; clears the trees (about 100 m at most)
const LABEL_HEIGHT_L = 230 // the number pill's height; its width is twice this
const ACCENT_FALLBACK = '#c2410c'
const SKIRT_FALLBACK = '#243247' // the walls' colour comes from --terrain-skirt
// Wall brightness by the side it faces, so the block reads as solid: north, east, south, west.
const SKIRT_SHADE = [0.95, 0.75, 0.6, 0.85]

// A full turn every 150 s. OrbitControls turns 2 pi / 60 rad per second at speed 1.
const ROTATE_SPEED = 60 / 150
const REDUCED_MOTION = '(prefers-reduced-motion: reduce)'
// Set once the intro rotation has ended for good (user input or a view change).
let introRotationDone = false

// The colour ramp encodes value as lightness, so the light must not change lightness much.
// A lit surface returns (AMBIENT + SUN * max(0, normal . sun)) / PI of the texture colour;
// AMBIENT + SUN stays below PI so no pixel is ever brighter than its legend colour, and flat
// ground (about 0.91) and steep slopes facing away from the sun (about 0.83) stay close to it.
// Relief is carried by the mesh silhouette plus this gentle shading.
const AMBIENT = 2.6
const SUN = 0.54

// Framing: the terrain is fitted into the "safe area" of the stage, which is the canvas minus the
// title block, the control card and the bottom row of buttons and legend (measured from the DOM).
const EDGE_PX = 16 // clearance to the stage edge
const GAP_PX = 12 // clearance to the title, card and bottom row
// The fit is solved at this many compass directions and interpolated between them, so the whole
// opening rotation stays inside the safe area; the extra share covers the interpolation error.
const FIT_AZIMUTHS = 24
const FIT_SAFETY = 1.03
const FIT_INTERIOR_STRIDE = 8
const FIT_RING_STRIDE = 2
const ZOOM_STEP = 1.3
const WHEEL_RATE = 0.0015
const TAP_MAX_MOVE = 10
const TAP_MAX_MS = 600

type Update = (p: Props) => void

/** What the pointer is over in the Landscape layer. */
interface GroundHover {
  /** Analysis-grid cell (row * cols + col) under the pointer, or null outside the grid. */
  cell: number | null
  elevation: number
  landClass: string
  treePct: number | null
}

const MARKER_OUTLINE = '#ffffff'
// Park masts are the Landscape markers scaled up with the scene (k = scene width over 10 km); the
// 0.5 keeps 10 to 40 of them clear of the terrain at the opening view.
const PARK_MAST_SCALE = 0.5
const PARK_LINE_LIFT = 12 // metres per k above the ground, so the lines do not sink into slopes
const PARK_WINDOW_BAND = 32 // metres per k: about 3 px at the opening view of the park

function northLabel(): CanvasTexture {
  const size = 128
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!
  ctx.font = '700 88px system-ui, sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.lineWidth = 12
  ctx.lineJoin = 'round'
  ctx.strokeStyle = '#0b1220'
  ctx.strokeText('N', size / 2, size / 2 + 4)
  ctx.fillStyle = '#f1f5f9'
  ctx.fillText('N', size / 2, size / 2 + 4)
  const texture = new CanvasTexture(canvas)
  texture.colorSpace = SRGBColorSpace
  return texture
}

function circleTexture(fill: string, stroke: string, lineWidth: number): CanvasTexture {
  const size = 64
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!
  ctx.beginPath()
  ctx.arc(size / 2, size / 2, size / 2 - lineWidth / 2 - 1, 0, Math.PI * 2)
  ctx.fillStyle = fill
  ctx.fill()
  ctx.lineWidth = lineWidth
  ctx.strokeStyle = stroke
  ctx.stroke()
  const texture = new CanvasTexture(canvas)
  texture.colorSpace = SRGBColorSpace
  return texture
}

/** Landscape head: accent disc, thick white ring, thin dark outer edge. */
function headTextureLandscape(accent: string): CanvasTexture {
  const size = 96
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!
  const c = size / 2
  ctx.beginPath()
  ctx.arc(c, c, c - 1, 0, Math.PI * 2)
  ctx.fillStyle = '#0b1220'
  ctx.fill()
  ctx.beginPath()
  ctx.arc(c, c, c - 5, 0, Math.PI * 2)
  ctx.fillStyle = '#ffffff'
  ctx.fill()
  ctx.beginPath()
  ctx.arc(c, c, c - 17, 0, Math.PI * 2)
  ctx.fillStyle = accent
  ctx.fill()
  const texture = new CanvasTexture(canvas)
  texture.colorSpace = SRGBColorSpace
  return texture
}

/** Landscape candidate site: white dot with a dark outline. */
function dotTextureLandscape(): CanvasTexture {
  const size = 64
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!
  ctx.beginPath()
  ctx.arc(size / 2, size / 2, size / 2 - 3, 0, Math.PI * 2)
  ctx.fillStyle = '#ffffff'
  ctx.fill()
  ctx.lineWidth = 8
  ctx.strokeStyle = '#0b1220'
  ctx.stroke()
  const texture = new CanvasTexture(canvas)
  texture.colorSpace = SRGBColorSpace
  return texture
}

/** Landscape site number on a dark pill (the canvas is twice as wide as it is tall). */
function siteLabelLandscape(id: number): CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = 128
  canvas.height = 64
  const ctx = canvas.getContext('2d')!
  ctx.beginPath()
  ctx.roundRect(4, 6, 88, 52, 26)
  ctx.fillStyle = 'rgba(11, 18, 32, 0.92)'
  ctx.fill()
  ctx.lineWidth = 2
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)'
  ctx.stroke()
  ctx.font = '700 38px system-ui, sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillStyle = '#ffffff'
  ctx.fillText(String(id), 48, 33)
  const texture = new CanvasTexture(canvas)
  texture.colorSpace = SRGBColorSpace
  return texture
}

function siteLabel(id: number): CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = 128
  canvas.height = 128
  const ctx = canvas.getContext('2d')!
  ctx.font = '700 72px system-ui, sans-serif'
  ctx.textAlign = 'left'
  ctx.textBaseline = 'middle'
  ctx.lineWidth = 12
  ctx.lineJoin = 'round'
  ctx.strokeStyle = '#ffffff'
  ctx.strokeText(String(id), 12, 68)
  ctx.fillStyle = '#1c2430'
  ctx.fillText(String(id), 12, 68)
  const texture = new CanvasTexture(canvas)
  texture.colorSpace = SRGBColorSpace
  return texture
}

/**
 * Four vertical walls from the mesh's edge vertices down to `baseY`. Unlit, with a fixed shade
 * per side. Not used for picking, which tests the height field directly.
 */
function buildSkirt(positions: Float32Array, meta: { rows: number; cols: number }, baseY: number): BufferGeometry {
  const { rows, cols } = meta
  const at = (r: number, c: number) => (r * cols + c) * 3
  // Each edge listed in order along the edge, with the shade of the side it faces.
  const edges: { ids: number[]; shade: number }[] = [
    { ids: Array.from({ length: cols }, (_, c) => at(0, c)), shade: SKIRT_SHADE[0] },
    { ids: Array.from({ length: rows }, (_, r) => at(r, cols - 1)), shade: SKIRT_SHADE[1] },
    { ids: Array.from({ length: cols }, (_, c) => at(rows - 1, c)), shade: SKIRT_SHADE[2] },
    { ids: Array.from({ length: rows }, (_, r) => at(r, 0)), shade: SKIRT_SHADE[3] },
  ]
  const pos: number[] = []
  const shade: number[] = []
  const index: number[] = []
  for (const { ids, shade: k } of edges) {
    const first = pos.length / 3
    for (const i of ids) {
      pos.push(positions[i], positions[i + 1], positions[i + 2], positions[i], baseY, positions[i + 2])
      shade.push(k, k)
    }
    for (let j = 0; j < ids.length - 1; j++) {
      const a = first + j * 2
      index.push(a, a + 1, a + 2, a + 1, a + 3, a + 2)
    }
  }
  const g = new BufferGeometry()
  g.setAttribute('position', new BufferAttribute(new Float32Array(pos), 3))
  g.setAttribute('shade', new BufferAttribute(new Float32Array(shade), 1))
  g.setIndex(index)
  return g
}

interface Fit {
  distance: number
  targetY: number
}

/**
 * Camera distance and target height that put `points` inside the safe area when the camera looks
 * at the terrain from compass direction `azimuth` (0 = from the south). `hx` and `hy` are the safe
 * area's half-extents as shares of the view's half-width and half-height. The target stays on the
 * vertical axis through the terrain centre so rotating still pivots about the middle of the map.
 */
function fitAzimuth(points: Vector3[], aspect: number, azimuth: number, hx: number, hy: number, start: Fit, passes: number, sc: SceneScale): Fit {
  const cam = new PerspectiveCamera(FOV, aspect, sc.near, sc.far)
  const ce = Math.cos(sc.startElevation)
  const dir = new Vector3(Math.sin(azimuth) * ce, Math.sin(sc.startElevation), Math.cos(azimuth) * ce)
  const target = new Vector3(0, start.targetY, 0)
  const p = new Vector3()
  const halfTan = Math.tan((FOV / 2) * (Math.PI / 180))
  let distance = start.distance
  // Moving the target changes the shape only slightly, so a few passes converge.
  for (let pass = 0; pass < passes; pass++) {
    cam.position.copy(target).addScaledVector(dir, distance)
    cam.lookAt(target)
    cam.updateMatrixWorld()
    let minY = Infinity
    let maxY = -Infinity
    let maxX = 0
    for (const v of points) {
      p.copy(v).project(cam)
      minY = Math.min(minY, p.y)
      maxY = Math.max(maxY, p.y)
      maxX = Math.max(maxX, Math.abs(p.x))
    }
    // Raising the target by dy moves the scene down in the view by about dy * cos(elevation).
    target.y += (((minY + maxY) / 2) * distance * halfTan) / Math.cos(sc.startElevation)
    distance *= Math.max(maxX / hx, (maxY - minY) / 2 / hy)
  }
  return { distance, targetY: target.y }
}

function computeFits(points: Vector3[], aspect: number, hx: number, hy: number, startDistance: number, sc: SceneScale): Fit[] {
  const fits: Fit[] = []
  let prev: Fit = { distance: startDistance, targetY: 0 }
  for (let i = 0; i < FIT_AZIMUTHS; i++) {
    prev = fitAzimuth(points, aspect, (i / FIT_AZIMUTHS) * Math.PI * 2, hx, hy, prev, i === 0 ? 16 : 6, sc)
    fits.push(prev)
  }
  return fits.map((f) => ({ distance: f.distance * FIT_SAFETY, targetY: f.targetY }))
}

/** The fit for any compass direction, interpolated between the solved ones. */
function fitAt(fits: Fit[], azimuth: number): Fit {
  const turn = Math.PI * 2
  const t = ((((azimuth % turn) + turn) % turn) / turn) * fits.length
  const i = Math.floor(t) % fits.length
  const a = fits[i]
  const b = fits[(i + 1) % fits.length]
  const k = t - Math.floor(t)
  return { distance: a.distance + (b.distance - a.distance) * k, targetY: a.targetY + (b.targetY - a.targetY) * k }
}

/** The overlay key for the About popover, where the chip has no room for it (phones). */
function OverlayKey({ landscape }: { landscape: boolean }) {
  return (
    <ul className="key" aria-label="Overlay key">
      {!landscape && (
      <li>
        <svg width="16" height="16" aria-hidden="true">
          <defs>
            <pattern id="hatch-key-about" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="6" height="6" fill="#fff" fillOpacity="0.5" />
              <line x1="0" y1="0" x2="0" y2="6" stroke="#4b5563" strokeWidth="1.4" />
            </pattern>
          </defs>
          <rect width="16" height="16" fill="#c9dbd6" />
          <rect width="16" height="16" fill="url(#hatch-key-about)" />
        </svg>
        Not covered
      </li>
      )}
      <li>
        <svg width="16" height="16" aria-hidden="true">
          <circle cx="8" cy="8" r="3" fill="#f9fafb" stroke="#1f2937" strokeWidth="1.2" />
        </svg>
        Candidate site
      </li>
      <li>
        <svg width="16" height="16" aria-hidden="true">
          <circle cx="8" cy="8" r="6" fill="var(--accent)" stroke="#fff" strokeWidth="1.5" />
        </svg>
        Chosen sensor
      </li>
    </ul>
  )
}

function parkAriaLabel(layer: ParkLayer, k: number, rule: ParkRule): string {
  const ground =
    layer === 'landscape'
      ? 'on a sunny day, ground coloured by land cover'
      : `coloured by ${PARK_LAYERS.find((l) => l.key === layer)?.label ?? layer}`
  const ruleName = PARK_RULES.find((r) => r.key === rule)?.label.toLowerCase() ?? rule
  return `3D terrain of the whole park ${ground}, with ${k} sensors placed by ${ruleName}, and the park boundary and the 10 km window outlined. It can be rotated and zoomed.`
}

function ariaLabel(layer: StageLayer, method: Method): string {
  if (layer === 'landscape') {
    return `3D landscape of the study area on a sunny day, ground coloured by land cover, with candidate sites and the sensors placed by ${method.name} at sites ${method.chosen.join(', ')}. It can be rotated and zoomed. The same information is in the 2D map.`
  }
  const layerName = LAYERS.find((l) => l.key === layer)?.label ?? layer
  return `3D terrain of the study area, coloured by ${layerName}, with candidate sites and the sensors placed by ${method.name} at sites ${method.chosen.join(', ')}. It can be rotated and zoomed. The same information is in the 2D map.`
}

function GroundReadout({ hover, results }: { hover: GroundHover | null; results: Results }) {
  const { cols } = results.grid
  return (
    <div className="readout" aria-live="polite">
      {hover === null ? (
        <span className="muted">Hover or tap the ground for details.</span>
      ) : (
        <>
          <div>
            <strong>
              {hover.cell === null
                ? 'Outside the analysis grid'
                : `Row ${Math.floor(hover.cell / cols)}, column ${hover.cell % cols}`}
            </strong>
            {' · '}
            {Math.round(hover.elevation).toLocaleString('en-US')} m
          </div>
          <div className="muted">
            {hover.landClass}
            {hover.treePct !== null ? ` · ${hover.treePct}% tree cover` : ''}
          </div>
        </>
      )}
    </div>
  )
}

type FireUpdate = (p: FireProps) => void
type FireProps = Pick<Props, 'replayShown' | 'replay' | 'replayT' | 'replayPlaying' | 'detections'>

export default function Terrain3D({
  results,
  layer,
  method,
  replayShown = false,
  replay = null,
  replayT = 0,
  replayPlaying = false,
  detections = null,
  scale = 'window',
  parkData = null,
  parkLayer = 'landscape',
  parkK = 20,
  parkRule = 'fire_history',
  onUnavailable,
}: Props) {
  const box = useRef<HTMLDivElement>(null)
  const camera3d = useRef<{ zoom: (factor: number) => void; reset: () => void } | null>(null)
  const [aboutOpen, setAboutOpen] = useState(false)
  const aboutButton = useRef<HTMLButtonElement>(null)
  const aboutPanel = useRef<HTMLDivElement>(null)
  const [status, setStatus] = useState<Status>('loading')
  const [message, setMessage] = useState('')
  const [hoverCell, setHoverCell] = useState<number | null>(null)
  const [hoverGround, setHoverGround] = useState<ParkHover | null>(null)
  // Classes present in the window for the Landscape key; 'none' when land cover could not be loaded.
  const [landKey, setLandKey] = useState<LandClass[] | 'none'>('none')
  const latest = useRef<Props>({ results, layer, method, parkData, parkLayer, parkK, parkRule })
  const update = useRef<Update | null>(null)
  const fireUpdate = useRef<FireUpdate | null>(null)
  const fireLatest = useRef<FireProps>({})
  const unavailable = useRef(onUnavailable)
  const park = scale === 'park'
  // The park shows Landscape until its data has arrived.
  const parkGround: ParkLayer = parkData ? parkLayer : 'landscape'
  const landscape = park ? parkGround === 'landscape' : layer === 'landscape'
  const [lo, hi] = useMemo(
    () => (park || layer === 'landscape' ? [0, 1] : layerRange(results.layers, layer)),
    [results.layers, layer, park],
  )
  const info = LAYERS.find((l) => l.key === layer)
  const covered = useMemo(() => new Set(method.covered_cells), [method])
  const parkPlaced = useMemo(
    () => (parkData ? parkPlacement(parkData, parkK, parkRule) : null),
    [parkData, parkK, parkRule],
  )
  const parkCovered = useMemo(() => (parkPlaced ? new Set(parkPlaced.covered) : null), [parkPlaced])
  const parkRange = useMemo(
    () => (parkData && parkGround !== 'landscape' && parkGround !== 'coverage' ? parkLayerRange(parkData, parkGround) : ([0, 1] as [number, number])),
    [parkData, parkGround],
  )

  // Layer and method changes only repaint the texture and markers; the scene is built once.
  useEffect(() => {
    unavailable.current = onUnavailable
  }, [onUnavailable])

  useEffect(() => {
    if (!aboutOpen) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      setAboutOpen(false)
      aboutButton.current?.focus()
    }
    const onDown = (e: PointerEvent) => {
      const t = e.target as Node
      if (!aboutPanel.current?.contains(t) && !aboutButton.current?.contains(t)) setAboutOpen(false)
    }
    document.addEventListener('keydown', onKey)
    document.addEventListener('pointerdown', onDown)
    return () => {
      document.removeEventListener('keydown', onKey)
      document.removeEventListener('pointerdown', onDown)
    }
  }, [aboutOpen])

  useEffect(() => {
    latest.current = { results, layer, method, parkData, parkLayer, parkK, parkRule }
    update.current?.(latest.current)
  }, [results, layer, method, parkData, parkLayer, parkK, parkRule])

  // The replay clock only moves GPU attributes and asks for one render; nothing is rebuilt.
  useEffect(() => {
    fireLatest.current = { replayShown, replay, replayT, replayPlaying, detections }
    fireUpdate.current?.(fireLatest.current)
  }, [replayShown, replay, replayT, replayPlaying, detections])

  useEffect(() => {
    const host = box.current
    if (!host) return
    let disposed = false
    let cleanup: (() => void) | null = null
    let partial: WebGLRenderer | null = null
    const fail = (text: string) => {
      setMessage(text)
      setStatus('error')
      unavailable.current?.()
    }

    const build = (state: TerrainState, landState: LandcoverState) => {
      if (disposed) return
      if (state.status === 'error') {
        fail(state.message)
        return
      }
      const { meta, heights } = state.terrain
      // Every value that is a fixed length in metres comes from here (see lib/scale.ts).
      const sc = sceneScale(scale, meta.width_m)
      const exag = sc.exaggeration
      // Land cover must sit on exactly this mesh's pixel grid; otherwise Landscape uses elevation only.
      const land: LandcoverData | null =
        landState.status === 'ready' && matchesTerrain(landState.data, meta) ? landState.data : null
      setLandKey(land ? presentClasses(land) : 'none')

      let renderer: WebGLRenderer
      try {
        renderer = new WebGLRenderer({ antialias: true, alpha: true })
      } catch {
        fail('3D view needs WebGL, which is not available in this browser.')
        return
      }
      partial = renderer
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
      // The sun and terrain never move, so the shadow map is drawn once per switch to Landscape.
      renderer.shadowMap.enabled = sc.shadowMap
      renderer.shadowMap.type = PCFSoftShadowMap
      renderer.shadowMap.autoUpdate = false
      const coarse = window.matchMedia('(pointer: coarse)').matches
      const canvas = renderer.domElement
      canvas.setAttribute('role', 'img')
      host.appendChild(canvas)

      const scene = new Scene()
      const geometry = new BufferGeometry()
      geometry.setAttribute(
        'position',
        new BufferAttribute(buildPositions(heights, meta, exag), 3),
      )
      // The grid covers results.bounds, slightly less than the mesh, so UVs come from lat/lon.
      // The park grid covers the whole mesh; the window's grid sits inside it.
      const rect = gridSceneRect(meta, sc.markers ? latest.current.results.bounds : meta.bounds)
      geometry.setAttribute('uv', new BufferAttribute(buildUVs(meta, rect), 2))
      geometry.setIndex(new BufferAttribute(buildIndices(meta.rows, meta.cols), 1))
      geometry.computeVertexNormals()
      let alive = true
      // Data layers: flat-lit drape. Landscape swaps in its own material, lights and sky (setLook).
      const material = new MeshStandardMaterial({ roughness: 1, metalness: 0 })
      geometry.setAttribute('uv1', new BufferAttribute(groundUVs(meta), 2))
      const landMaterial = new MeshStandardMaterial({ roughness: 1, metalness: 0 })
      let groundTex: ReturnType<typeof groundTextures> | null = null
      const terrain = new Mesh(geometry, material)
      terrain.castShadow = true
      terrain.receiveShadow = true
      scene.add(terrain)

      const skirtColor = new Color(
        getComputedStyle(host).getPropertyValue('--terrain-skirt').trim() || SKIRT_FALLBACK,
      )
      const skirtGeometry = buildSkirt(
        geometry.getAttribute('position').array as Float32Array,
        meta,
        (meta.min_m - sc.skirtDepth) * exag,
      )
      const skirtShade = skirtGeometry.getAttribute('shade')
      const skirtColors = new Float32Array(skirtShade.count * 3)
      for (let i = 0; i < skirtShade.count; i++) {
        const k = skirtShade.getX(i)
        skirtColors.set([skirtColor.r * k, skirtColor.g * k, skirtColor.b * k], i * 3)
      }
      skirtGeometry.setAttribute('color', new BufferAttribute(skirtColors, 3))
      skirtGeometry.deleteAttribute('shade')
      const skirtMaterial = new MeshBasicMaterial({ vertexColors: true, side: DoubleSide })
      const skirt = new Mesh<BufferGeometry, MeshBasicMaterial | MeshStandardMaterial>(skirtGeometry, skirtMaterial)
      skirt.raycast = () => {} // never a pick target
      skirt.receiveShadow = true
      scene.add(skirt)
      const wallBaseY = (meta.min_m - sc.skirtDepth) * exag
      addSkirtStrataUVs(skirtGeometry, wallBaseY, meta.max_m * exag, sc.strataRepeat)
      const landSkirtMaterial = new MeshStandardMaterial({ roughness: 1, metalness: 0, side: DoubleSide })
      let strata: CanvasTexture | null = null

      const dataAmbient = new AmbientLight(0xffffff, AMBIENT)
      const dataSun = new DirectionalLight(0xffffff, SUN)
      // From the north-west, the usual hillshade direction; the target is the origin.
      dataSun.position.set(-0.8, 0.55, -0.5).multiplyScalar(20000)
      scene.add(dataAmbient, dataSun)

      // Landscape look. The group is also the hook for the tree stage: add canopy instances to
      // `canopy`, and look up ground with landClassAt / treeFractionAt (lib/landcover.ts).
      const landscapeGroup = new Group()
      landscapeGroup.name = 'landscape'
      landscapeGroup.visible = false
      const canopy = new Group()
      canopy.name = 'landscape-canopy'
      const sky = buildSky()
      const landSun = new DirectionalLight(LANDSCAPE_LIGHT.sunColor, LANDSCAPE_LIGHT.sunIntensity)
      if (sc.shadowMap) {
        landSun.castShadow = true
        fitSun(landSun, meta.width_m / 2, meta.height_m / 2, wallBaseY, meta.max_m * exag, coarse ? 1024 : 2048)
      } else {
        // The park's cast shadows are baked into the ground texture; the sun only lights the surface.
        landSun.position.copy(sunDirection()).multiplyScalar(30000 * sc.k)
      }
      const hemi = new HemisphereLight(LANDSCAPE_LIGHT.skyColor, LANDSCAPE_LIGHT.groundColor, LANDSCAPE_LIGHT.hemiIntensity)
      landscapeGroup.add(sky.mesh, landSun, landSun.target, hemi, canopy)
      scene.add(landscapeGroup)
      const haze = new FogExp2(LANDSCAPE_LIGHT.fogColor, sc.fogDensity)
      let canopyMesh: Canopy | null = null
      let lookIsLandscape = false
      // Built when the detections are first available; rebuilt if they change.
      let fire: FireMarks | null = null
      let fireSource: readonly Detection[] | null = null
      const setLook = (isLandscape: boolean) => {
        if (isLandscape === lookIsLandscape) return
        lookIsLandscape = isLandscape
        landscapeGroup.visible = isLandscape
        dataAmbient.visible = !isLandscape
        dataSun.visible = !isLandscape
        scene.fog = isLandscape ? haze : null
        renderer.toneMapping = isLandscape ? ACESFilmicToneMapping : NoToneMapping
        renderer.toneMappingExposure = isLandscape ? LANDSCAPE_LIGHT.exposure : 1
        fire?.setLandscape(isLandscape)
        terrain.material = isLandscape ? landMaterial : material
        skirt.material = isLandscape ? landSkirtMaterial : skirtMaterial
        if (isLandscape) {
          if (!groundTex) {
            groundTex = groundTextures(
              paintGround(heights, meta, land, sc.groundTexels, sc.shadowMap ? null : bakeShadow(heights, meta, exag)),
              renderer.capabilities.getMaxAnisotropy(),
            )
            landMaterial.map = groundTex.map
            landMaterial.roughnessMap = groundTex.roughnessMap
            landMaterial.needsUpdate = true
          }
          if (!strata) {
            strata = strataTexture(renderer.capabilities.getMaxAnisotropy())
            landSkirtMaterial.map = strata
            landSkirtMaterial.needsUpdate = true
          }
          // Built once, on first use; hiding the group is all a layer switch does.
          if (sc.trees && !canopyMesh) {
            canopyMesh = buildCanopy(heights, meta, land, coarse)
            canopy.add(canopyMesh.mesh)
          }
          renderer.shadowMap.needsUpdate = true
        }
      }

      const labelTexture = northLabel()
      const labelMaterial = new SpriteMaterial({ map: labelTexture, depthTest: false, fog: false, toneMapped: false })
      const label = new Sprite(labelMaterial)
      label.scale.setScalar(sc.northLabelSize)
      label.position.set(0, meta.max_m * exag * 0.5, -meta.height_m / 2 - sc.northLabelOffset)
      label.renderOrder = 1
      scene.add(label)

      const accent = getComputedStyle(host).getPropertyValue('--accent').trim() || ACCENT_FALLBACK
      const dotTexture = circleTexture('#f9fafb', '#1f2937', 10)
      const headTexture = circleTexture(accent, '#ffffff', 10)
      // Markers keep their true colours under Landscape's tone mapping and haze.
      const dotMaterial = new SpriteMaterial({ map: dotTexture, depthWrite: false, fog: false, toneMapped: false })
      const headMaterial = new SpriteMaterial({ map: headTexture, depthWrite: false, fog: false, toneMapped: false })
      const dotTextureL = dotTextureLandscape()
      const headTextureL = headTextureLandscape(accent)
      const dotMaterialL = new SpriteMaterial({ map: dotTextureL, depthWrite: false, fog: false, toneMapped: false })
      const headMaterialL = new SpriteMaterial({ map: headTextureL, depthWrite: false, fog: false, toneMapped: false })
      const mastGeometryL = new CylinderGeometry(MAST_RADIUS_L, MAST_RADIUS_L, MAST_HEIGHT_L, 6)
      const mastGeometry = new CylinderGeometry(MAST_RADIUS, MAST_RADIUS, MAST_HEIGHT, 6)
      const mastMaterial = new MeshBasicMaterial({ color: accent, fog: false, toneMapped: false })
      // A white back-face shell round each mast reads as a light outline against green (Landscape only).
      const outlineGeometry = new CylinderGeometry(MAST_RADIUS_L * 1.7, MAST_RADIUS_L * 1.7, MAST_HEIGHT_L * 1.02, 6)
      const outlineMaterial = new MeshBasicMaterial({ color: MARKER_OUTLINE, side: BackSide, fog: false, toneMapped: false })
      // Whole park: the Landscape mast scaled with the scene, and thin lines for the park boundary and the 10 km window.
      const pk = PARK_MAST_SCALE * sc.k
      const parkMastGeometry = new CylinderGeometry(MAST_RADIUS_L * pk, MAST_RADIUS_L * pk, MAST_HEIGHT_L * pk, 6)
      const parkOutlineGeometry = new CylinderGeometry(MAST_RADIUS_L * pk * 1.7, MAST_RADIUS_L * pk * 1.7, MAST_HEIGHT_L * pk * 1.02, 6)
      const lineMaterial = (color: string) =>
        new LineBasicMaterial({ color, transparent: true, opacity: 1, fog: false, toneMapped: false })
      const boundaryMaterial = lineMaterial(PARK_LINE_COLORS.landscape.boundary)
      // The window outline is a ribbon (a GL line is always 1 px), a little see-through so it stays quiet.
      const windowMaterial = new MeshBasicMaterial({
        color: PARK_LINE_COLORS.landscape.window,
        transparent: true,
        opacity: 0.9,
        side: DoubleSide,
        depthWrite: false,
        fog: false,
        toneMapped: false,
      })
      let parkLines: (LineSegments | Mesh)[] = []
      let parkLinesFor: ParkData | null = null
      const clearParkLines = () => {
        for (const l of parkLines) {
          scene.remove(l)
          l.geometry.dispose()
        }
        parkLines = []
        parkLinesFor = null
      }
      const markers = new Group()
      scene.add(markers)
      let markerLabels: { material: SpriteMaterial; texture: CanvasTexture }[] = []
      let drape: CanvasTexture | null = null

      const halfLabel = sc.northLabelSize * 0.5
      const labelCorners = [-1, 1].flatMap((sx) =>
        [-1, 1].map((sy) => label.position.clone().add(new Vector3(sx * halfLabel, sy * halfLabel, 0))),
      )
      // The wall's bottom corners count for framing too, so the block is not cut off by the view.
      const baseY = wallBaseY
      const fitPoints = [...labelCorners]
      for (const sx of [-1, 1]) {
        for (const sz of [-1, 1]) {
          fitPoints.push(new Vector3((sx * meta.width_m) / 2, baseY, (sz * meta.height_m) / 2))
        }
      }
      // The whole outline, plus a coarse sample of the interior for peaks.
      const vertices = geometry.getAttribute('position').array as Float32Array
      const addVertex = (r: number, c: number) => {
        const i = (r * meta.cols + c) * 3
        fitPoints.push(new Vector3(vertices[i], vertices[i + 1], vertices[i + 2]))
      }
      for (let c = 0; c < meta.cols; c += FIT_RING_STRIDE) {
        addVertex(0, c)
        addVertex(meta.rows - 1, c)
      }
      for (let r = 0; r < meta.rows; r += FIT_RING_STRIDE) {
        addVertex(r, 0)
        addVertex(r, meta.cols - 1)
      }
      for (let r = FIT_INTERIOR_STRIDE; r < meta.rows - 1; r += FIT_INTERIOR_STRIDE) {
        for (let c = FIT_INTERIOR_STRIDE; c < meta.cols - 1; c += FIT_INTERIOR_STRIDE) addVertex(r, c)
      }
      const startDistance =
        (0.62 * Math.max(meta.width_m, meta.height_m)) / Math.tan((FOV / 2) * (Math.PI / 180))
      const safeArea = () => {
        const w = host.clientWidth
        const h = host.clientHeight
        const origin = host.getBoundingClientRect()
        const stage = host.closest('.stage')
        const box = (selector: string) => {
          const r = stage?.querySelector(selector)?.getBoundingClientRect()
          return r ? { left: r.left - origin.left, top: r.top - origin.top, right: r.right - origin.left } : null
        }
        let left = EDGE_PX
        let top = EDGE_PX
        let right = w - EDGE_PX
        let bottom = h - EDGE_PX
        const title = stage?.querySelector('.stage-title')?.getBoundingClientRect()
        // Clear the two main title lines only. The small meta line and the legend chip are
        // overlays the terrain may pass behind, so the map can fill most of the screen.
        if (title) top = Math.max(top, title.top - origin.top + Math.min(title.height, 56) + GAP_PX)
        const card = box('.stage-card:not(.stage-sheet)')
        if (card) right = Math.min(right, card.left - GAP_PX)
        for (const selector of ['.stage-zoom', '.results-link']) {
          const b = box(selector)
          if (b) bottom = Math.min(bottom, b.top - GAP_PX)
        }
        // A stage too small for all that keeps the old behaviour: use most of the canvas.
        if (right - left < w * 0.4) {
          left = EDGE_PX
          right = w - EDGE_PX
        }
        if (bottom - top < h * 0.35) {
          top = EDGE_PX
          bottom = h - EDGE_PX
        }
        return { w, h, left, top, right, bottom }
      }
      let safe = safeArea()
      let fits: Fit[] = []
      const solveFits = () => {
        const { w, h, left, top, right, bottom } = safe
        const aspect = w > 0 && h > 0 ? w / h : 1
        fits = computeFits(
          fitPoints,
          aspect,
          Math.max(0.1, (right - left) / Math.max(w, 1)),
          Math.max(0.1, (bottom - top) / Math.max(h, 1)),
          startDistance,
          sc,
        )
      }
      solveFits()
      // The camera sits at compass direction `azimuth` (0 = south of the terrain looking north),
      // at the distance and target height that frame the terrain in the safe area from there.
      const place = (azimuth: number): Fit => {
        const f = fitAt(fits, azimuth)
        controls.target.set(0, f.targetY, 0)
        camera.position.set(
          Math.sin(azimuth) * Math.cos(sc.startElevation) * f.distance,
          f.targetY + Math.sin(sc.startElevation) * f.distance,
          Math.cos(azimuth) * Math.cos(sc.startElevation) * f.distance,
        )
        return f
      }

      const camera = new PerspectiveCamera(FOV, 1, sc.near, sc.far)
      const render = () => {
        if (alive) renderer.render(scene, camera)
      }

      const clearMarkers = () => {
        markers.clear()
        for (const l of markerLabels) {
          l.material.dispose()
          l.texture.dispose()
        }
        markerLabels = []
      }

      const apply: Update = (props) => {
        // Park: the ground is Landscape until the park data has arrived; the window never has a park layer.
        const parkMode = !sc.markers
        const pd = parkMode ? props.parkData ?? null : null
        const pLayer: ParkLayer = pd ? props.parkLayer ?? 'landscape' : 'landscape'
        const placement = pd ? parkPlacement(pd, props.parkK ?? 20, props.parkRule ?? 'fire_history') : null
        const p = props
        canvas.setAttribute(
          'aria-label',
          parkMode ? parkAriaLabel(pLayer, props.parkK ?? 20, props.parkRule ?? 'fire_history') : ariaLabel(p.layer, p.method),
        )

        let dataCanvas: HTMLCanvasElement | null = null
        if (parkMode) {
          if (pd && placement && pLayer !== 'landscape') dataCanvas = drawParkDrape(pd, pLayer, placement)
        } else if (p.layer !== 'landscape') {
          dataCanvas = drawDrape(p.results, p.layer, p.method)
        }
        setLook(dataCanvas === null)
        if (!dataCanvas) {
          // The data drape is not needed here; free it and let the next data layer rebuild it.
          material.map = null
          drape?.dispose()
          drape = null
        } else {
          const texture = new CanvasTexture(dataCanvas)
          texture.colorSpace = SRGBColorSpace
          texture.generateMipmaps = true
          texture.minFilter = LinearMipmapLinearFilter
          texture.magFilter = LinearFilter
          texture.anisotropy = renderer.capabilities.getMaxAnisotropy()
          const hadMap = material.map !== null
          material.map = texture
          if (!hadMap) material.needsUpdate = true
          drape?.dispose()
          drape = texture
        }

        clearMarkers()
        if (parkMode) {
          if (pd && placement) {
            // The boundary and window follow the ground; built once per data set, recoloured per layer.
            if (parkLinesFor !== pd) {
              clearParkLines()
              const lift = PARK_LINE_LIFT * sc.k
              const mk = (positions: Float32Array, material: LineBasicMaterial | MeshBasicMaterial) => {
                const g = new BufferGeometry()
                g.setAttribute('position', new BufferAttribute(positions, 3))
                const l = material instanceof LineBasicMaterial ? new LineSegments(g, material) : new Mesh(g, material)
                l.renderOrder = 1
                l.raycast = () => {}
                scene.add(l)
                return l
              }
              parkLines = [
                mk(parkBoundaryPositions(pd, heights, meta, rect, exag, lift), boundaryMaterial),
                mk(parkWindowBand(pd, heights, meta, rect, exag, lift, PARK_WINDOW_BAND * sc.k), windowMaterial),
              ]
              parkLinesFor = pd
            }
            const colors = PARK_LINE_COLORS[pLayer === 'landscape' ? 'landscape' : 'data']
            boundaryMaterial.color.set(colors.boundary)
            windowMaterial.color.set(colors.window)
            const headSize = HEAD_SIZE_L * pk
            const mastH = MAST_HEIGHT_L * pk
            for (const [row, col] of placement.sites) {
              const { x, z } = cellCentreScene(row, col, pd.meta, rect)
              const ground = sampleHeight(heights, meta, x, z) * exag
              const mast = new Mesh(parkMastGeometry, mastMaterial)
              mast.position.set(x, ground + mastH / 2, z)
              const shell = new Mesh(parkOutlineGeometry, outlineMaterial)
              shell.position.copy(mast.position)
              const head = new Sprite(headMaterialL)
              head.scale.setScalar(headSize)
              head.position.set(x, ground + mastH, z)
              markers.add(shell, mast, head)
            }
            // Counted from the scene itself, for checks.
            canvas.dataset.masts = String(markers.children.filter((o) => o instanceof Mesh && o.geometry === parkMastGeometry).length)
          } else {
            clearParkLines()
          }
          render()
          return
        }
        const chosen = new Set(p.method.chosen)
        const isL = p.layer === 'landscape'
        const mastH = isL ? MAST_HEIGHT_L : MAST_HEIGHT
        const headSize = isL ? HEAD_SIZE_L : HEAD_SIZE
        for (const site of p.results.sites) {
          const { x, z } = cellCentreScene(site.row, site.col, p.results.grid, rect)
          const ground = sampleHeight(heights, meta, x, z) * exag
          if (!chosen.has(site.id)) {
            const dotSize = isL ? DOT_SIZE_L : DOT_SIZE
            const dot = new Sprite(isL ? dotMaterialL : dotMaterial)
            dot.scale.setScalar(dotSize)
            dot.position.set(x, ground + (isL ? DOT_LIFT_L : DOT_SIZE / 2), z)
            markers.add(dot)
            continue
          }
          const mast = new Mesh(isL ? mastGeometryL : mastGeometry, mastMaterial)
          mast.position.set(x, ground + mastH / 2, z)
          markers.add(mast)
          if (isL) {
            const shell = new Mesh(outlineGeometry, outlineMaterial)
            shell.position.copy(mast.position)
            markers.add(shell)
          }
          const head = new Sprite(isL ? headMaterialL : headMaterial)
          head.scale.setScalar(headSize)
          head.position.set(x, ground + mastH, z)
          markers.add(head)
          const labelTexture = isL ? siteLabelLandscape(site.id) : siteLabel(site.id)
          const labelMaterial = new SpriteMaterial({ map: labelTexture, depthTest: false, fog: false, toneMapped: false })
          const label = new Sprite(labelMaterial)
          if (isL) {
            label.scale.set(LABEL_HEIGHT_L * 2, LABEL_HEIGHT_L, 1)
            // The pill's left edge (4 of 128 px) sits just right of the head.
            label.center.set((4 / 128) - (headSize * 0.5 + 10) / (LABEL_HEIGHT_L * 2), 0.5)
          } else {
            label.scale.setScalar(LABEL_SIZE * 2)
            // Anchor the sprite's left edge just right of the head, whichever way the camera turns.
            label.center.set(-(HEAD_SIZE * 0.5) / (LABEL_SIZE * 2), 0.5)
          }
          label.position.set(x, ground + mastH, z)
          label.renderOrder = 2
          markers.add(label)
          markerLabels.push({ material: labelMaterial, texture: labelTexture })
        }
        render()
      }
      apply(latest.current)
      update.current = apply

      const applyFire: FireUpdate = (p) => {
        const list = p.detections ?? null
        if (list !== fireSource) {
          if (fire) {
            scene.remove(fire.points)
            fire.dispose()
            fire = null
          }
          fireSource = list
          if (list && list.length > 0) {
            // The park's cell-centre fallback (no mesh bounds) needs the park grid, not the window's.
            const fireGrid = sc.markers ? latest.current.results.grid : (latest.current.parkData?.meta ?? latest.current.results.grid)
            fire = buildFireMarks(list, heights, meta, fireGrid, rect, exag, sc.fire)
            fire.setLandscape(lookIsLandscape)
            fire.points.visible = false
            scene.add(fire.points)
          }
        }
        if (!fire) return
        const show = !!p.replayShown && !!p.replay
        fire.points.visible = show
        if (show && p.replay) {
          const state = p.replay.stateAt(p.replayT ?? 0)
          fire.set(state.phase, state.age)
        }
        // Starting playback ends the intro rotation, like any other input.
        if (show && p.replayPlaying) stopRotation()
        render()
      }

      const controls = new OrbitControls(camera, canvas)
      place(0)
      controls.enablePan = false // panning could slide the view under the ground plane
      controls.maxPolarAngle = MAX_POLAR // camera stays above the y = 0 plane (terrain is above 0)
      controls.minDistance = sc.minDistance
      controls.maxDistance = Math.max(...fits.map((f) => f.distance)) * 1.8
      // Touch: one finger rotates, two fingers zoom and rotate. The canvas takes every touch
      // gesture (touch-action: none); browsers cancel the pointers mid-gesture otherwise. The page
      // is scrolled from the Results link or from any control on the stage.
      controls.touches = { ONE: TOUCH.ROTATE, TWO: TOUCH.DOLLY_ROTATE }
      canvas.style.touchAction = 'none'
      const keepTwoFingerGesture = (e: TouchEvent) => {
        if (e.cancelable) e.preventDefault()
      }
      canvas.addEventListener('touchmove', keepTwoFingerGesture, { passive: false })
      controls.update()
      controls.addEventListener('change', render)

      const touchIds = new Set<number>()
      // Intro rotation: a slow orbit about the terrain centre until the first user input. While it
      // runs a requestAnimationFrame loop drives the camera; otherwise rendering is on demand.
      const reducedMotion = window.matchMedia(REDUCED_MOTION)
      let rotateWanted = !introRotationDone && !reducedMotion.matches
      let onScreen = true
      let tabVisible = !document.hidden
      let spin = 0
      let spinLast = 0
      const spinTick = (now: number) => {
        spin = 0
        if (!alive || !rotateWanted || !onScreen || !tabVisible) return
        const dt = Math.min(Math.max((now - spinLast) / 1000, 0), 0.1)
        spinLast = now
        // Re-frame for the current compass direction first, so the corners swinging round stay in view.
        if (pristine) place(controls.getAzimuthalAngle())
        controls.update(dt) // rotates and fires 'change', which renders
        spin = requestAnimationFrame(spinTick)
      }
      const syncRotation = () => {
        controls.autoRotate = rotateWanted
        const run = alive && rotateWanted && onScreen && tabVisible
        if (run && !spin) {
          spinLast = performance.now()
          spin = requestAnimationFrame(spinTick)
        } else if (!run && spin) {
          cancelAnimationFrame(spin)
          spin = 0
        }
      }
      function stopRotation() {
        if (!rotateWanted) return
        rotateWanted = false
        introRotationDone = true
        syncRotation()
      }
      controls.autoRotateSpeed = ROTATE_SPEED
      const visibility = new IntersectionObserver((entries) => {
        onScreen = entries[entries.length - 1].isIntersecting
        syncRotation()
      })
      visibility.observe(host)
      const onVisibility = () => {
        tabVisible = !document.hidden
        syncRotation()
      }
      document.addEventListener('visibilitychange', onVisibility)
      const onMotionPref = () => {
        if (reducedMotion.matches) {
          rotateWanted = false
          syncRotation()
        }
      }
      reducedMotion.addEventListener('change', onMotionPref)
      // Any deliberate input on the stage ends it for good.
      const onStagePointerDown = (e: PointerEvent) => {
        if (e.target === canvas) stopRotation()
      }
      const onStageKey = (e: KeyboardEvent) => {
        const t = e.target as Element | null
        if (t === canvas || t?.closest('.stage-zoom')) stopRotation()
      }
      const stageEl = host.parentElement ?? host
      stageEl.addEventListener('pointerdown', onStagePointerDown, true)
      stageEl.addEventListener('keydown', onStageKey, true)
      controls.addEventListener('start', stopRotation)
      // Safari trackpad pinch arrives as gesture events; touch pinches are left to OrbitControls.
      let gestureScale = 0
      const onGestureStart = (e: Event) => {
        if (e.target !== canvas || touchIds.size > 0) return
        e.preventDefault()
        gestureScale = 1
      }
      const onGestureChange = (e: Event) => {
        if (!gestureScale) return
        e.preventDefault()
        const scale = (e as Event & { scale: number }).scale
        if (!(scale > 0)) return
        dolly(gestureScale / scale)
        gestureScale = scale
      }
      const onGestureEnd = () => {
        gestureScale = 0
      }
      host.addEventListener('gesturestart', onGestureStart)
      host.addEventListener('gesturechange', onGestureChange)
      host.addEventListener('gestureend', onGestureEnd)

      // Until the user moves the camera, a resize re-fits the terrain to the new view.
      let pristine = true
      controls.addEventListener('start', () => {
        pristine = false
      })
      // The re-fit keeps the current compass direction unless `azimuth` says otherwise, so a
      // resize during the intro rotation does not snap the view back.
      const goHome = (azimuth = controls.getAzimuthalAngle()) => {
        place(azimuth)
        controls.maxDistance = Math.max(...fits.map((f) => f.distance)) * 1.8
        controls.update()
        render()
      }
      // Dolly along the view direction with the same limits as the drag and pinch zoom.
      const dolly = (factor: number) => {
        pristine = false
        stopRotation()
        const offset = camera.position.clone().sub(controls.target)
        const d = Math.min(controls.maxDistance, Math.max(controls.minDistance, offset.length() * factor))
        camera.position.copy(controls.target).add(offset.setLength(d))
        controls.update()
      }
      camera3d.current = {
        zoom: (factor) => dolly(1 / factor),
        reset: () => {
          pristine = true
          stopRotation()
          goHome(0)
        },
      }
      // A plain wheel scrolls the page. Only Ctrl or Cmd + wheel (what a trackpad pinch sends) zooms,
      // and OrbitControls' own wheel listener never sees the event.
      const onWheel = (e: WheelEvent) => {
        if (e.target !== canvas) return
        e.stopPropagation()
        if (!e.ctrlKey && !e.metaKey) return
        e.preventDefault()
        const px = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaMode === 2 ? e.deltaY * 400 : e.deltaY
        dolly(Math.exp(Math.max(-300, Math.min(300, px)) * WHEEL_RATE))
      }
      host.addEventListener('wheel', onWheel, { capture: true, passive: false })

      // Cell readout: one analytic ray-terrain test per animation frame, only while the
      // pointer moves or the camera is being moved by the user.
      let raf = 0
      let pending: { x: number; y: number } | null = null
      let lastMouse: { x: number; y: number } | null = null
      const origin = new Vector3()
      const dir = new Vector3()
      const pickHit = (clientX: number, clientY: number): { x: number; z: number } | null => {
        const box = canvas.getBoundingClientRect()
        if (box.width === 0 || box.height === 0) return null
        camera.updateMatrixWorld()
        origin.setFromMatrixPosition(camera.matrixWorld)
        dir
          .set(((clientX - box.left) / box.width) * 2 - 1, -(((clientY - box.top) / box.height) * 2 - 1), 0.5)
          .unproject(camera)
          .sub(origin)
          .normalize()
        return rayTerrainHit(origin, dir, heights, meta, exag, sc.rayStep)
      }
      let lastGroundPixel = -2
      const flushPickAt = (clientX: number, clientY: number) => {
        const hit = pickHit(clientX, clientY)
        const parkMode = !sc.markers
        const grid = parkMode ? latest.current.parkData?.meta ?? null : latest.current.results.grid
        const cell = hit && grid ? sceneToCell(hit.x, hit.z, grid, rect) : null
        setHoverCell(!parkMode && cell && grid ? cell.row * grid.cols + cell.col : null)
        if (!parkMode && latest.current.layer !== 'landscape') return
        if (!hit) {
          lastGroundPixel = -1
          setHoverGround(null)
          return
        }
        // The class under the pointer comes from the fine (terrain-pixel) grid, not the 500 m cell.
        const px = pixelAt(meta, hit.x, hit.z)
        const pixel = px.row * meta.cols + px.col
        const cellIndex = cell && grid ? cell.row * grid.cols + cell.col : null
        // In the park a pixel can straddle two 500 m cells, so the cell is part of the key.
        const key = pixel * 32768 + (cellIndex ?? 0) + 1
        if (key === lastGroundPixel) return
        lastGroundPixel = key
        setHoverGround({
          cell: cellIndex,
          elevation: sampleHeight(heights, meta, hit.x, hit.z),
          landClass: land ? landClassName(land.classes[pixel]) : 'Land cover unavailable',
          treePct: land && land.classes[pixel] === 10 ? Math.round((land.tree[pixel] / 255) * 100) : null,
        })
      }
      const flushPick = () => {
        raf = 0
        if (!alive || !pending) return
        const { x, y } = pending
        pending = null
        flushPickAt(x, y)
      }
      const schedulePick = (x: number, y: number) => {
        pending = { x, y }
        if (!raf) raf = requestAnimationFrame(flushPick)
      }
      let tap: { id: number; x: number; y: number; at: number } | null = null
      const onPointerDown = (e: PointerEvent) => {
        if (e.pointerType !== 'touch') return
        touchIds.add(e.pointerId)
        if (touchIds.size >= 2) stopRotation()
        tap = touchIds.size === 1 ? { id: e.pointerId, x: e.clientX, y: e.clientY, at: performance.now() } : null
      }
      const onPointerMove = (e: PointerEvent) => {
        if (e.pointerType === 'touch') {
          if (tap && Math.hypot(e.clientX - tap.x, e.clientY - tap.y) > TAP_MAX_MOVE) tap = null
          return
        }
        lastMouse = { x: e.clientX, y: e.clientY }
        schedulePick(e.clientX, e.clientY)
      }
      const onPointerUp = (e: PointerEvent) => {
        if (e.pointerType !== 'touch') return
        touchIds.delete(e.pointerId)
        if (tap && tap.id === e.pointerId && performance.now() - tap.at < TAP_MAX_MS) {
          stopRotation()
          schedulePick(tap.x, tap.y)
        }
        tap = null
      }
      const onPointerCancel = (e: PointerEvent) => {
        touchIds.delete(e.pointerId)
        tap = null
      }
      const onPointerLeave = (e: PointerEvent) => {
        if (e.pointerType === 'touch') return
        lastMouse = null
        pending = null
        lastGroundPixel = -2
        setHoverCell(null)
        setHoverGround(null)
      }
      // Wheel zoom and drag rotation move the terrain under a still pointer.
      const repick = () => {
        if (lastMouse) schedulePick(lastMouse.x, lastMouse.y)
      }
      canvas.addEventListener('pointerdown', onPointerDown)
      canvas.addEventListener('pointermove', onPointerMove)
      canvas.addEventListener('pointerup', onPointerUp)
      canvas.addEventListener('pointercancel', onPointerCancel)
      canvas.addEventListener('pointerleave', onPointerLeave)
      controls.addEventListener('change', repick)

      const resize = () => {
        safe = safeArea()
        const { w, h, left, top, right, bottom } = safe
        if (w === 0 || h === 0) return
        renderer.setSize(w, h, false)
        camera.aspect = w / h
        solveFits()
        // Shift the projection so the terrain is centred in the safe area, not in the whole canvas.
        camera.setViewOffset(w, h, w / 2 - (left + right) / 2, h / 2 - (top + bottom) / 2, w, h)
        if (pristine) goHome()
        else render()
      }
      const observer = new ResizeObserver(resize)
      observer.observe(host)
      // The title wraps to a different height as the page is resized or its font loads.
      const titleEl = host.closest('.stage')?.querySelector('.stage-title')
      if (titleEl) observer.observe(titleEl)
      resize()
      syncRotation()
      fireUpdate.current = applyFire
      applyFire(fireLatest.current)
      setStatus('ready')

      cleanup = () => {
        alive = false
        update.current = null
        fireUpdate.current = null
        camera3d.current = null
        host.removeEventListener('wheel', onWheel, { capture: true })
        cancelAnimationFrame(raf)
        cancelAnimationFrame(spin)
        spin = 0
        // Leaving the 3D view (a view change) ends the intro rotation for good.
        introRotationDone = true
        visibility.disconnect()
        document.removeEventListener('visibilitychange', onVisibility)
        reducedMotion.removeEventListener('change', onMotionPref)
        stageEl.removeEventListener('pointerdown', onStagePointerDown, true)
        stageEl.removeEventListener('keydown', onStageKey, true)
        controls.removeEventListener('start', stopRotation)
        host.removeEventListener('gesturestart', onGestureStart)
        host.removeEventListener('gesturechange', onGestureChange)
        host.removeEventListener('gestureend', onGestureEnd)
        observer.disconnect()
        canvas.removeEventListener('touchmove', keepTwoFingerGesture)
        canvas.removeEventListener('pointerdown', onPointerDown)
        canvas.removeEventListener('pointermove', onPointerMove)
        canvas.removeEventListener('pointerup', onPointerUp)
        canvas.removeEventListener('pointercancel', onPointerCancel)
        canvas.removeEventListener('pointerleave', onPointerLeave)
        controls.removeEventListener('change', render)
        controls.removeEventListener('change', repick)
        controls.dispose()
        clearMarkers()
        clearParkLines()
        parkMastGeometry.dispose()
        parkOutlineGeometry.dispose()
        boundaryMaterial.dispose()
        windowMaterial.dispose()
        fire?.dispose()
        drape?.dispose()
        dotTexture.dispose()
        headTexture.dispose()
        dotMaterial.dispose()
        headMaterial.dispose()
        dotTextureL.dispose()
        headTextureL.dispose()
        dotMaterialL.dispose()
        headMaterialL.dispose()
        mastGeometryL.dispose()
        canopyMesh?.dispose()
        mastGeometry.dispose()
        mastMaterial.dispose()
        outlineGeometry.dispose()
        outlineMaterial.dispose()
        geometry.dispose()
        skirtGeometry.dispose()
        skirtMaterial.dispose()
        landSkirtMaterial.dispose()
        strata?.dispose()
        groundTex?.map.dispose()
        groundTex?.roughnessMap.dispose()
        landMaterial.dispose()
        sky.dispose()
        landSun.shadow.map?.dispose()
        material.dispose()
        labelMaterial.dispose()
        labelTexture.dispose()
        // Browsers keep only a handful of WebGL contexts alive; release this one now.
        renderer.forceContextLoss()
        renderer.dispose()
        canvas.remove()
      }
    }

    Promise.all([loadTerrain(scale), loadLandcover(scale)])
      .then(([terrainState, landState]) => build(terrainState, landState))
      .catch(() => {
        if (disposed) return
        if (cleanup) cleanup()
        else if (partial) {
          partial.forceContextLoss()
          partial.dispose()
          partial.domElement.remove()
        }
        cleanup = null
        partial = null
        fail('The 3D view could not be drawn. The 2D map is still available.')
      })

    return () => {
      disposed = true
      cleanup?.()
    }
  }, [scale])

  return (
    <div className="stage-3d">
      <div className="stage-canvas" ref={box} />
      {status === 'loading' && (
        <div className="stage-msg">
          <p role="status">Loading terrain…</p>
        </div>
      )}
      {status === 'error' && (
        <div className="stage-msg">
          <p role="alert" className="error">
            {message}
          </p>
        </div>
      )}
      {status !== 'error' && (
        <>
          <div className="stage-zoom">
            <button
              type="button"
              className="icon-btn glass"
              aria-label="Zoom in"
              onClick={() => camera3d.current?.zoom(ZOOM_STEP)}
            >
              +
            </button>
            <button
              type="button"
              className="icon-btn glass"
              aria-label="Zoom out"
              onClick={() => camera3d.current?.zoom(1 / ZOOM_STEP)}
            >
              −
            </button>
            <button
              type="button"
              className="icon-btn glass"
              aria-label="Reset view"
              onClick={() => camera3d.current?.reset()}
            >
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
                <path
                  d="M4 10a6 6 0 1 0 2-4.5M4 3.5V6.5H7"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          </div>
          <div className="stage-info">
            <div className="stage-chip glass">
              {park ? (
                <>
                  {parkGround === 'landscape' ? (
                    <LandscapeLegend classes={landKey} park />
                  ) : (
                    <ParkLegend layer={parkGround} lo={parkRange[0]} hi={parkRange[1]} />
                  )}
                  <ParkKey layer={parkGround} />
                  {replayShown && <ParkFireKey />}
                </>
              ) : landscape || !info ? (
                <LandscapeLegend classes={landKey} fires={replayShown} />
              ) : (
                <Legend info={info} lo={lo} hi={hi} fires={replayShown} />
              )}
              {park ? (
                <ParkReadout
                  hover={hoverGround}
                  park={parkData}
                  layer={parkGround}
                  placement={parkPlaced}
                  covered={parkCovered}
                />
              ) : landscape ? (
                <GroundReadout hover={hoverGround} results={results} />
              ) : (
                <CellReadout
                  results={results}
                  method={method}
                  covered={covered}
                  cell={hoverCell}
                  idle="Hover or tap a cell for details."
                />
              )}
            </div>
            <div className="about">
              <button
                ref={aboutButton}
                type="button"
                className="chip-btn glass"
                aria-expanded={aboutOpen}
                aria-controls="about-view"
                onClick={() => setAboutOpen((o) => !o)}
              >
                About this view
              </button>
              {aboutOpen && (
                <div id="about-view" ref={aboutPanel} className="about-pop glass" role="group" aria-label="About this view">
                  <div className="about-legend">
                    {park ? <ParkKey layer={parkGround} /> : <OverlayKey landscape={landscape} />}
                  </div>
                  <p className="terrain-hint">Drag to rotate, pinch to zoom. Use Results to move down the page.</p>
                  <p className="mouse-hint">
                    Scroll to move down the page. Hold Ctrl (⌘ on Mac) and scroll, or use the + and − buttons, to
                    zoom. Drag to rotate.
                  </p>
                  <p>
                    Heights exaggerated {SCALE_NOTE[scale].exaggeration}×. Elevation: Copernicus DEM GLO-30,{' '}
                    {SCALE_NOTE[scale].demResolutionM} m.
                  </p>
                  <p>Sensor masts are not to scale.</p>
                  {park && landscape && (
                    <p>
                      Ground colours come from ESA WorldCover 2021 land cover and Copernicus elevation. The shading,
                      sky and sun are illustrative (not the real weather or time of day).
                    </p>
                  )}
                  {park && parkData && <p>{parkData.meta.note}</p>}
                  {landscape && !park && (
                    <p>
                      Ground colours come from ESA WorldCover 2021 land cover and Copernicus elevation. Trees
                      are placed where the data shows tree cover but are illustrative, as are the shading, sky
                      and sun (not the real weather or time of day).
                    </p>
                  )}
                  <p>
                    {park
                      ? landscape
                        ? 'Hover or tap the ground for its row, column, elevation and land cover.'
                        : 'Hover or tap a cell for its row, column, elevation and layer value.'
                      : landscape
                      ? 'Hover or tap the ground for its row, column, elevation and land cover.'
                      : 'Shading shows the terrain\u2019s shape and slightly darkens slopes; hover or tap a cell for its exact value.'}
                  </p>
                  {!park && (
                    <p>
                      Coverage was computed on the {results.grid.cell_size_m} m grid. On this 30 m surface a covered
                      cell can look hidden behind a ridge the coarse model does not see.
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
