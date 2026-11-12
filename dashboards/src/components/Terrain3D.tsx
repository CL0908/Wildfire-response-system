import { useEffect, useMemo, useRef, useState } from 'react'
import {
  AmbientLight,
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  CylinderGeometry,
  DirectionalLight,
  Group,
  LinearFilter,
  LinearMipmapLinearFilter,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
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
import { drawDrape } from '../lib/drape'
import { LAYERS, layerRange } from '../lib/layers'
import {
  type TerrainState,
  VERTICAL_EXAGGERATION,
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
import type { LayerKey, Method, Results } from '../types'

interface Props {
  results: Results
  layer: LayerKey
  method: Method
  /** Called once if the 3D view cannot be shown (no WebGL, terrain files missing or unusable). */
  onUnavailable?: () => void
}

type Status = 'loading' | 'ready' | 'error'

const FOV = 35
const START_ELEVATION = (52 * Math.PI) / 180
const MAX_POLAR = (78 * Math.PI) / 180

// Scene metres. Masts are deliberately not to scale (a real 30 m mast would be invisible).
const MAST_HEIGHT = 360
const MAST_RADIUS = 18
const HEAD_SIZE = 250
const DOT_SIZE = 170
const LABEL_SIZE = 200
const LABEL_SIZE_N = 900 // the "N" sprite's edge length
const ACCENT_FALLBACK = '#c2410c'

// The colour ramp encodes value as lightness, so the light must not change lightness much.
// A lit surface returns (AMBIENT + SUN * max(0, normal . sun)) / PI of the texture colour;
// AMBIENT + SUN stays below PI so no pixel is ever brighter than its legend colour, and flat
// ground (about 0.91) and steep slopes facing away from the sun (about 0.83) stay close to it.
// Relief is carried by the mesh silhouette plus this gentle shading.
const AMBIENT = 2.6
const SUN = 0.54

// Default framing: the terrain should fill this share of the box's half-width / half-height.
const FIT = 0.94
const FIT_STRIDE = 4
const TAP_MAX_MOVE = 10
const TAP_MAX_MS = 600

type Update = (p: Props) => void

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
  ctx.strokeStyle = '#ffffff'
  ctx.strokeText('N', size / 2, size / 2 + 4)
  ctx.fillStyle = '#1c2430'
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
 * Camera distance and target height that make the terrain (every `FIT_STRIDE`th vertex) and
 * the "N" label fill the view with a small margin. The target stays on the vertical axis
 * through the terrain centre so rotating still pivots about the middle of the map.
 */
function fitView(
  positions: Float32Array,
  meta: { rows: number; cols: number },
  labelPoints: Vector3[],
  aspect: number,
  startDistance: number,
): { distance: number; targetY: number } {
  const points = [...labelPoints]
  for (let r = 0; r < meta.rows; r += FIT_STRIDE) {
    for (let c = 0; c < meta.cols; c += FIT_STRIDE) {
      const i = (r * meta.cols + c) * 3
      points.push(new Vector3(positions[i], positions[i + 1], positions[i + 2]))
    }
  }
  const cam = new PerspectiveCamera(FOV, aspect, 200, 120000)
  const dir = new Vector3(0, Math.sin(START_ELEVATION), Math.cos(START_ELEVATION))
  const target = new Vector3()
  const p = new Vector3()
  const halfTan = Math.tan((FOV / 2) * (Math.PI / 180))
  let distance = startDistance
  // Moving the target changes the shape only slightly, so a few passes converge.
  for (let pass = 0; pass < 16; pass++) {
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
    target.y += (((minY + maxY) / 2) * distance * halfTan) / Math.cos(START_ELEVATION)
    distance *= Math.max(maxX, (maxY - minY) / 2) / FIT
  }
  return { distance, targetY: target.y }
}

function ariaLabel(layer: LayerKey, method: Method): string {
  const layerName = LAYERS.find((l) => l.key === layer)?.label ?? layer
  return `3D terrain of the study area, coloured by ${layerName}, with candidate sites and the sensors placed by ${method.name} at sites ${method.chosen.join(', ')}. It can be rotated and zoomed. The same information is in the 2D map.`
}

export default function Terrain3D({ results, layer, method, onUnavailable }: Props) {
  const box = useRef<HTMLDivElement>(null)
  const [status, setStatus] = useState<Status>('loading')
  const [message, setMessage] = useState('')
  const [hoverCell, setHoverCell] = useState<number | null>(null)
  const latest = useRef<Props>({ results, layer, method })
  const update = useRef<Update | null>(null)
  const unavailable = useRef(onUnavailable)
  const [lo, hi] = useMemo(() => layerRange(results.layers, layer), [results.layers, layer])
  const info = LAYERS.find((l) => l.key === layer)!
  const covered = useMemo(() => new Set(method.covered_cells), [method])

  // Layer and method changes only repaint the texture and markers; the scene is built once.
  useEffect(() => {
    unavailable.current = onUnavailable
  }, [onUnavailable])

  useEffect(() => {
    latest.current = { results, layer, method }
    update.current?.(latest.current)
  }, [results, layer, method])

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

    const build = (state: TerrainState) => {
      if (disposed) return
      if (state.status === 'error') {
        fail(state.message)
        return
      }
      const { meta, heights } = state.terrain

      let renderer: WebGLRenderer
      try {
        renderer = new WebGLRenderer({ antialias: true, alpha: true })
      } catch {
        fail('3D view needs WebGL, which is not available in this browser.')
        return
      }
      partial = renderer
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
      const canvas = renderer.domElement
      canvas.setAttribute('role', 'img')
      host.appendChild(canvas)

      const scene = new Scene()
      const geometry = new BufferGeometry()
      geometry.setAttribute(
        'position',
        new BufferAttribute(buildPositions(heights, meta, VERTICAL_EXAGGERATION), 3),
      )
      // The grid covers results.bounds, slightly less than the mesh, so UVs come from lat/lon.
      const rect = gridSceneRect(meta, latest.current.results.bounds)
      geometry.setAttribute('uv', new BufferAttribute(buildUVs(meta, rect), 2))
      geometry.setIndex(new BufferAttribute(buildIndices(meta.rows, meta.cols), 1))
      geometry.computeVertexNormals()
      let alive = true
      const material = new MeshStandardMaterial({ roughness: 1, metalness: 0 })
      const terrain = new Mesh(geometry, material)
      scene.add(terrain)

      scene.add(new AmbientLight(0xffffff, AMBIENT))
      const sun = new DirectionalLight(0xffffff, SUN)
      // From the north-west, the usual hillshade direction; the target is the origin.
      sun.position.set(-0.8, 0.55, -0.5).multiplyScalar(20000)
      scene.add(sun)

      const labelTexture = northLabel()
      const labelMaterial = new SpriteMaterial({ map: labelTexture, depthTest: false })
      const label = new Sprite(labelMaterial)
      label.scale.setScalar(LABEL_SIZE_N)
      label.position.set(0, meta.max_m * VERTICAL_EXAGGERATION * 0.5, -meta.height_m / 2 - 700)
      label.renderOrder = 1
      scene.add(label)

      const accent = getComputedStyle(host).getPropertyValue('--accent').trim() || ACCENT_FALLBACK
      const dotTexture = circleTexture('#f9fafb', '#1f2937', 10)
      const headTexture = circleTexture(accent, '#ffffff', 10)
      const dotMaterial = new SpriteMaterial({ map: dotTexture, depthWrite: false })
      const headMaterial = new SpriteMaterial({ map: headTexture, depthWrite: false })
      const mastGeometry = new CylinderGeometry(MAST_RADIUS, MAST_RADIUS, MAST_HEIGHT, 6)
      const mastMaterial = new MeshBasicMaterial({ color: accent })
      const markers = new Group()
      scene.add(markers)
      let markerLabels: { material: SpriteMaterial; texture: CanvasTexture }[] = []
      let drape: CanvasTexture | null = null

      const camera = new PerspectiveCamera(FOV, 1, 200, 120000)
      const aspect = host.clientWidth > 0 && host.clientHeight > 0 ? host.clientWidth / host.clientHeight : 1
      const halfLabel = (LABEL_SIZE_N * 0.5)
      const labelCorners = [-1, 1].flatMap((sx) =>
        [-1, 1].map((sy) => label.position.clone().add(new Vector3(sx * halfLabel, sy * halfLabel, 0))),
      )
      const startDistance =
        (0.62 * Math.max(meta.width_m, meta.height_m)) / Math.tan((FOV / 2) * (Math.PI / 180))
      const fit = fitView(
        geometry.getAttribute('position').array as Float32Array,
        meta,
        labelCorners,
        aspect,
        startDistance,
      )
      const distance = fit.distance
      // South of the terrain (+Z) looking north, raised by START_ELEVATION.
      camera.position.set(
        0,
        fit.targetY + Math.sin(START_ELEVATION) * distance,
        Math.cos(START_ELEVATION) * distance,
      )

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

      const apply: Update = (p) => {
        canvas.setAttribute('aria-label', ariaLabel(p.layer, p.method))

        const texture = new CanvasTexture(drawDrape(p.results, p.layer, p.method))
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

        clearMarkers()
        const chosen = new Set(p.method.chosen)
        for (const site of p.results.sites) {
          const { x, z } = cellCentreScene(site.row, site.col, p.results.grid, rect)
          const ground = sampleHeight(heights, meta, x, z) * VERTICAL_EXAGGERATION
          if (!chosen.has(site.id)) {
            const dot = new Sprite(dotMaterial)
            dot.scale.setScalar(DOT_SIZE)
            dot.position.set(x, ground + DOT_SIZE / 2, z)
            markers.add(dot)
            continue
          }
          const mast = new Mesh(mastGeometry, mastMaterial)
          mast.position.set(x, ground + MAST_HEIGHT / 2, z)
          markers.add(mast)
          const head = new Sprite(headMaterial)
          head.scale.setScalar(HEAD_SIZE)
          head.position.set(x, ground + MAST_HEIGHT, z)
          markers.add(head)
          const labelTexture = siteLabel(site.id)
          const labelMaterial = new SpriteMaterial({ map: labelTexture, depthTest: false })
          const label = new Sprite(labelMaterial)
          label.scale.setScalar(LABEL_SIZE * 2)
          // Anchor the sprite's left edge just right of the head, whichever way the camera turns.
          label.center.set(-(HEAD_SIZE * 0.5) / (LABEL_SIZE * 2), 0.5)
          label.position.set(x, ground + MAST_HEIGHT, z)
          label.renderOrder = 2
          markers.add(label)
          markerLabels.push({ material: labelMaterial, texture: labelTexture })
        }
        render()
      }
      apply(latest.current)
      update.current = apply

      const controls = new OrbitControls(camera, canvas)
      controls.target.set(0, fit.targetY, 0)
      controls.enablePan = false // panning could slide the view under the ground plane
      controls.maxPolarAngle = MAX_POLAR // camera stays above the y = 0 plane (terrain is above 0)
      controls.minDistance = 2500
      controls.maxDistance = distance * 1.8
      // One finger scrolls the page, two fingers rotate and zoom. OrbitControls sets
      // touch-action: none, which would trap page scrolling on a phone.
      controls.touches = { ONE: -1 as TOUCH, TWO: TOUCH.DOLLY_ROTATE }
      canvas.style.touchAction = 'pan-y'
      const keepTwoFingerGesture = (e: TouchEvent) => {
        if (e.touches.length >= 2 && e.cancelable) e.preventDefault()
      }
      canvas.addEventListener('touchmove', keepTwoFingerGesture, { passive: false })
      controls.update()
      controls.addEventListener('change', render)

      // Cell readout: one analytic ray-terrain test per animation frame, only while the
      // pointer moves or the camera is being moved by the user.
      let raf = 0
      let pending: { x: number; y: number } | null = null
      let lastMouse: { x: number; y: number } | null = null
      const origin = new Vector3()
      const dir = new Vector3()
      const pickCell = (clientX: number, clientY: number): number | null => {
        const box = canvas.getBoundingClientRect()
        if (box.width === 0 || box.height === 0) return null
        camera.updateMatrixWorld()
        origin.setFromMatrixPosition(camera.matrixWorld)
        dir
          .set(((clientX - box.left) / box.width) * 2 - 1, -(((clientY - box.top) / box.height) * 2 - 1), 0.5)
          .unproject(camera)
          .sub(origin)
          .normalize()
        const hit = rayTerrainHit(origin, dir, heights, meta, VERTICAL_EXAGGERATION)
        if (!hit) return null
        const grid = latest.current.results.grid
        const cell = sceneToCell(hit.x, hit.z, grid, rect)
        return cell ? cell.row * grid.cols + cell.col : null
      }
      const flushPick = () => {
        raf = 0
        if (!alive || !pending) return
        const { x, y } = pending
        pending = null
        setHoverCell(pickCell(x, y))
      }
      const schedulePick = (x: number, y: number) => {
        pending = { x, y }
        if (!raf) raf = requestAnimationFrame(flushPick)
      }
      const touchIds = new Set<number>()
      let tap: { id: number; x: number; y: number; at: number } | null = null
      const onPointerDown = (e: PointerEvent) => {
        if (e.pointerType !== 'touch') return
        touchIds.add(e.pointerId)
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
        if (tap && tap.id === e.pointerId && performance.now() - tap.at < TAP_MAX_MS) schedulePick(tap.x, tap.y)
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
        setHoverCell(null)
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
        const { clientWidth: w, clientHeight: h } = host
        if (w === 0 || h === 0) return
        renderer.setSize(w, h, false)
        camera.aspect = w / h
        camera.updateProjectionMatrix()
        render()
      }
      const observer = new ResizeObserver(resize)
      observer.observe(host)
      resize()
      setStatus('ready')

      cleanup = () => {
        alive = false
        update.current = null
        cancelAnimationFrame(raf)
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
        drape?.dispose()
        dotTexture.dispose()
        headTexture.dispose()
        dotMaterial.dispose()
        headMaterial.dispose()
        mastGeometry.dispose()
        mastMaterial.dispose()
        geometry.dispose()
        material.dispose()
        labelMaterial.dispose()
        labelTexture.dispose()
        // Browsers keep only a handful of WebGL contexts alive; release this one now.
        renderer.forceContextLoss()
        renderer.dispose()
        canvas.remove()
      }
    }

    loadTerrain()
      .then(build)
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
  }, [])

  return (
    <div className="terrain3d">
      <div className="terrain-box" ref={box}>
        {status === 'loading' && (
          <p role="status" className="terrain-msg muted">
            Loading terrain…
          </p>
        )}
        {status === 'error' && (
          <p role="alert" className="terrain-msg error">
            {message}
          </p>
        )}
      </div>
      {status !== 'error' && (
        <>
          <CellReadout
            results={results}
            method={method}
            covered={covered}
            cell={hoverCell}
            idle="Hover or tap a cell for details."
          />
          <Legend info={info} lo={lo} hi={hi} />
          <div className="muted terrain-note">
            <p className="terrain-hint">Use two fingers to rotate and zoom.</p>
            <p>
              Heights exaggerated {VERTICAL_EXAGGERATION}×. Elevation: Copernicus DEM GLO-30, 30 m.
            </p>
            <p>Sensor masts are not to scale.</p>
            <p>
              Shading shows the terrain&apos;s shape and slightly darkens slopes; hover or tap a cell for its exact
              value.
            </p>
            <p>
              Coverage was computed on the {results.grid.cell_size_m} m grid. On this 30 m surface a covered cell can
              look hidden behind a ridge the coarse model does not see.
            </p>
          </div>
        </>
      )}
    </div>
  )
}
