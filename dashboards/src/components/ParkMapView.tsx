import { useEffect, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { ParkKey, ParkLegend } from './ParkLegend'
import ParkReadout, { type ParkHover } from './ParkReadout'
import { drawParkDrape, parkLayerRange } from '../lib/drape'
import { PARK_LAYERS, RULE_LABEL, parkPlacement, type ParkData, type ParkK, type ParkLayer, type ParkRule } from '../lib/park'
import { PARK_LINE_COLORS, parkBoundaryEdges } from '../lib/parkLines'

/** A flat map has no Landscape look; App maps that choice to Priority before it gets here. */
export type ParkMapLayer = Exclude<ParkLayer, 'landscape'>

interface Props {
  park: ParkData
  layer: ParkMapLayer
  k: ParkK
  rule: ParkRule
}

const MAX_DPR = 2
const HALO = 'rgba(255, 255, 255, 0.9)'
const ACCENT_FALLBACK = '#c2410c'
const pct = (share: number) => `${Math.round(share * 100)}%`

/** The whole park as one canvas, north up: the chosen layer, the park boundary, the 10 km window and the sensors. */
export default function ParkMapView({ park, layer, k, rule }: Props) {
  const { rows, cols } = park.meta
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [width, setWidth] = useState(0)
  // The cell shown in the readout and outlined on the map: under the pointer, or moved by the arrow keys.
  const [cell, setCell] = useState<number | null>(null)

  const placement = useMemo(() => parkPlacement(park, k, rule), [park, k, rule])
  const covered = useMemo(() => new Set(placement.covered), [placement])
  const edges = useMemo(() => parkBoundaryEdges(park), [park])
  const drape = useMemo(() => drawParkDrape(park, layer, placement), [park, layer, placement])
  const [lo, hi] = useMemo(
    () => (layer === 'coverage' ? ([0, 1] as [number, number]) : parkLayerRange(park, layer)),
    [park, layer],
  )

  useLayoutEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    setWidth(canvas.clientWidth)
    const observer = new ResizeObserver((entries) => setWidth(entries[0].contentRect.width))
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || width <= 0) return
    const dpr = Math.min(MAX_DPR, window.devicePixelRatio || 1)
    const w = Math.round(width * dpr)
    const h = Math.round(((width * rows) / cols) * dpr)
    if (canvas.width !== w) canvas.width = w
    if (canvas.height !== h) canvas.height = h
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    // Cells are flat squares; scaling without smoothing keeps their edges and colours exact.
    ctx.imageSmoothingEnabled = false
    ctx.drawImage(drape, 0, 0, w, h)

    // From here on units are CSS pixels; u is one cell.
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const u = width / cols
    const colors = PARK_LINE_COLORS.data
    const stroke = (halo: number, color: string, line: number) => {
      ctx.lineJoin = 'round'
      ctx.strokeStyle = HALO
      ctx.lineWidth = halo
      ctx.stroke()
      ctx.strokeStyle = color
      ctx.lineWidth = line
      ctx.stroke()
    }

    ctx.beginPath()
    for (const [c0, r0, c1, r1] of edges) {
      ctx.moveTo(c0 * u, r0 * u)
      ctx.lineTo(c1 * u, r1 * u)
    }
    stroke(3.5, colors.boundary, 1.4)

    const win = park.meta.window
    ctx.beginPath()
    ctx.rect(win.col0 * u, win.row0 * u, win.n_cols * u, win.n_rows * u)
    stroke(5, colors.window, 2.2)

    const accent = getComputedStyle(canvas).getPropertyValue('--accent').trim() || ACCENT_FALLBACK
    const radius = Math.min(7, Math.max(4, width / 90))
    let drawn = 0
    for (const [row, col] of placement.sites) {
      ctx.beginPath()
      ctx.arc((col + 0.5) * u, (row + 0.5) * u, radius, 0, Math.PI * 2)
      ctx.fillStyle = accent
      ctx.fill()
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 2
      ctx.stroke()
      ctx.beginPath()
      ctx.arc((col + 0.5) * u, (row + 0.5) * u, radius + 1.5, 0, Math.PI * 2)
      ctx.strokeStyle = 'rgba(17, 24, 39, 0.65)'
      ctx.lineWidth = 1
      ctx.stroke()
      drawn += 1
    }

    if (cell !== null) {
      // At least 8 px, so the outline stays visible when a cell is under 2 px wide.
      const size = Math.max(u, 8)
      const x = (cell % cols) * u + u / 2 - size / 2
      const y = Math.floor(cell / cols) * u + u / 2 - size / 2
      ctx.beginPath()
      ctx.rect(x, y, size, size)
      stroke(4, '#111827', 1.6)
    }

    // Counted from what was drawn, for checks.
    canvas.dataset.masts = String(drawn)
    canvas.dataset.layer = layer
  }, [drape, width, edges, placement, cell, park, layer, rows, cols])

  const cellAt = (e: PointerEvent<HTMLCanvasElement>): number | null => {
    const box = e.currentTarget.getBoundingClientRect()
    const col = Math.floor(((e.clientX - box.left) / box.width) * cols)
    const row = Math.floor(((e.clientY - box.top) / box.height) * rows)
    return col < 0 || col >= cols || row < 0 || row >= rows ? null : row * cols + col
  }

  const onKeyDown = (e: KeyboardEvent<HTMLCanvasElement>) => {
    if (e.key === 'Escape') {
      setCell(null)
      return
    }
    const move = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[e.key]
    if (!move) return
    e.preventDefault()
    const step = e.shiftKey ? 10 : 1
    setCell((prev) => {
      const at = prev ?? Math.floor(rows / 2) * cols + Math.floor(cols / 2)
      const row = Math.min(rows - 1, Math.max(0, Math.floor(at / cols) + move[0] * step))
      const col = Math.min(cols - 1, Math.max(0, (at % cols) + move[1] * step))
      return row * cols + col
    })
  }

  const hover: ParkHover | null =
    cell === null ? null : { cell, elevation: park.layers.elevation_m[cell], landClass: '', treePct: null }
  const layerLabel = PARK_LAYERS.find((l) => l.key === layer)?.label ?? layer
  const m = placement.metrics
  const label =
    `Map of the whole park, ${rows} by ${cols} cells of ${park.meta.cell_m} m, north at the top, coloured by ${layerLabel}. ` +
    `${k} sensors placed by ${RULE_LABEL[rule]} cover ${pct(m.priority_share)} of the modelled priority, ` +
    `${pct(m.area_share)} of the park area and ${pct(m.fire_share)} of past fires. ` +
    'Arrow keys move a cell cursor; shift moves ten cells.'

  return (
    <div className="map">
      <canvas
        ref={canvasRef}
        className="park-canvas"
        style={{ aspectRatio: `${cols} / ${rows}` }}
        role="img"
        tabIndex={0}
        aria-label={label}
        onPointerMove={(e) => setCell(cellAt(e))}
        onPointerDown={(e) => setCell(cellAt(e))}
        onPointerLeave={(e) => {
          // A finger lifting also leaves the canvas; the tapped cell stays.
          if (e.pointerType === 'mouse' && document.activeElement !== e.currentTarget) setCell(null)
        }}
        onFocus={() => setCell((prev) => prev ?? Math.floor(rows / 2) * cols + Math.floor(cols / 2))}
        onBlur={() => setCell(null)}
        onKeyDown={onKeyDown}
      />
      <ParkReadout hover={hover} park={park} layer={layer} placement={placement} covered={covered} />
      <div className="park-keys">
        <ParkLegend layer={layer} lo={lo} hi={hi} />
        <ParkKey layer={layer} />
      </div>
    </div>
  )
}
