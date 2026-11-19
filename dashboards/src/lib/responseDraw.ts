import { CanvasTexture, SRGBColorSpace } from 'three'
import { FAILED, INK, UNREACHED, arrivalColor, crewLetter } from './ranger'
import { breakProgress, breakStatus, crewPosition, type ResponseScene } from './response'

/** Canvas pixels per grid cell in the 3D overlay. */
export const OVERLAY_PX = 40
const RING = '#facc15'
const BURNED_ALPHA_LANDSCAPE = 0.85

/**
 * The fire response drawn flat, as the 3D view's overlay: cells reached by hour t coloured by arrival
 * (the ranger section's scale), the crews' paths and the firebreaks. Canvas top = north, like the data
 * drape. `opaque` paints the unburned cells too (over a data layer); Landscape leaves them see-through.
 */
export function drawResponseOverlay(canvas: HTMLCanvasElement, scene: ResponseScene, t: number, opaque: boolean): void {
  const { rows, cols, horizon } = scene.data
  const S = OVERLAY_PX
  if (canvas.width !== cols * S || canvas.height !== rows * S) {
    canvas.width = cols * S
    canvas.height = rows * S
  }
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  if (opaque) {
    ctx.fillStyle = UNREACHED
    ctx.fillRect(0, 0, canvas.width, canvas.height)
  }
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)'
  ctx.lineWidth = 1
  for (let i = 0; i < scene.arrival.length; i++) {
    const h = scene.arrival[i]
    const x = (i % cols) * S
    const y = Math.floor(i / cols) * S
    if (h <= t) {
      ctx.globalAlpha = opaque ? 1 : BURNED_ALPHA_LANDSCAPE
      ctx.fillStyle = arrivalColor(h, horizon)
      ctx.fillRect(x, y, S, S)
      ctx.globalAlpha = 1
      ctx.strokeRect(x, y, S, S)
    } else if (opaque) {
      ctx.strokeRect(x, y, S, S)
    }
  }

  const px = (v: number) => v * S
  // Crew paths: the whole route thin and dotted, the part walked so far solid, both on a white halo.
  for (const track of scene.tracks) {
    if (t < scene.info.start_h) continue
    const pos = crewPosition(track, t)
    const done = track.stops.filter((s, i) => i > 0 && s.t <= t)
    const walked = [track.points[0], ...done, pos]
    const line = (pts: { x: number; y: number }[], width: number, style: string, dash: number[]) => {
      ctx.beginPath()
      pts.forEach((p, k) => (k === 0 ? ctx.moveTo(px(p.x), px(p.y)) : ctx.lineTo(px(p.x), px(p.y))))
      ctx.lineWidth = width
      ctx.strokeStyle = style
      ctx.setLineDash(dash)
      ctx.lineJoin = 'round'
      ctx.lineCap = 'round'
      ctx.stroke()
      ctx.setLineDash([])
    }
    line(track.points, 11, 'rgba(255, 255, 255, 0.85)', [])
    line(track.points, 3, INK, [8, 8])
    line(walked, 6, INK, [])
  }

  // Firebreaks.
  const inset = 4
  ctx.font = '700 15px system-ui, sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  for (const d of scene.plan.defences) {
    const status = breakStatus(d, t)
    if (status === 'pending') continue
    const x = d.cell[1] * S + inset
    const y = d.cell[0] * S + inset
    const w = S - 2 * inset
    if (status === 'cutting') {
      const f = breakProgress(d, t)
      ctx.fillStyle = 'rgba(15, 23, 42, 0.4)'
      ctx.fillRect(x, y + w * (1 - f), w, w * f)
    }
    if (status === 'failed') {
      ctx.save()
      ctx.beginPath()
      ctx.rect(x, y, w, w)
      ctx.clip()
      ctx.fillStyle = 'rgba(255, 255, 255, 0.6)'
      ctx.fillRect(x, y, w, w)
      ctx.strokeStyle = FAILED
      ctx.lineWidth = 3
      ctx.beginPath()
      for (let k = -w; k < w * 2; k += 10) {
        ctx.moveTo(x + k, y + w)
        ctx.lineTo(x + k + w, y)
      }
      ctx.stroke()
      ctx.restore()
    }
    ctx.strokeStyle = '#ffffff'
    ctx.lineWidth = 9
    ctx.setLineDash([])
    ctx.strokeRect(x, y, w, w)
    ctx.strokeStyle = status === 'failed' ? FAILED : INK
    ctx.lineWidth = status === 'held' ? 6 : 4
    ctx.setLineDash(status === 'held' ? [] : status === 'failed' ? [11, 7] : [3, 6])
    ctx.strokeRect(x, y, w, w)
    ctx.setLineDash([])
    const label = `${crewLetter(d.crew)}${d.order + 1}${status === 'failed' ? '×' : ''}`
    const cx = d.cell[1] * S + S / 2
    const cy = d.cell[0] * S + S - 11
    ctx.fillStyle = status === 'held' ? INK : '#ffffff'
    ctx.fillRect(cx - 14, cy - 8, 28, 16)
    ctx.fillStyle = status === 'held' ? '#ffffff' : status === 'failed' ? FAILED : INK
    ctx.fillText(label, cx, cy + 1)
  }
}

function texture(canvas: HTMLCanvasElement): CanvasTexture {
  const t = new CanvasTexture(canvas)
  t.colorSpace = SRGBColorSpace
  return t
}

function square(size: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  return [canvas, canvas.getContext('2d')!]
}

/** Ignition: accent diamond with a white edge. */
export function ignitionTexture(accent: string): CanvasTexture {
  const [c, ctx] = square(96)
  ctx.beginPath()
  ctx.moveTo(48, 4)
  ctx.lineTo(92, 48)
  ctx.lineTo(48, 92)
  ctx.lineTo(4, 48)
  ctx.closePath()
  ctx.fillStyle = accent
  ctx.fill()
  ctx.lineWidth = 9
  ctx.lineJoin = 'round'
  ctx.strokeStyle = '#ffffff'
  ctx.stroke()
  ctx.lineWidth = 2.5
  ctx.strokeStyle = '#0b1220'
  ctx.stroke()
  return texture(c)
}

/** Crew station: white triangle with a dark edge. */
export function stationTexture(): CanvasTexture {
  const [c, ctx] = square(96)
  ctx.beginPath()
  ctx.moveTo(48, 8)
  ctx.lineTo(90, 84)
  ctx.lineTo(6, 84)
  ctx.closePath()
  ctx.fillStyle = '#ffffff'
  ctx.fill()
  ctx.lineWidth = 9
  ctx.lineJoin = 'round'
  ctx.strokeStyle = INK
  ctx.stroke()
  return texture(c)
}

/** A crew: white disc, dark ring, its letter. */
export function crewTexture(crew: number): CanvasTexture {
  const [c, ctx] = square(96)
  ctx.beginPath()
  ctx.arc(48, 48, 41, 0, Math.PI * 2)
  ctx.fillStyle = '#ffffff'
  ctx.fill()
  ctx.lineWidth = 9
  ctx.strokeStyle = INK
  ctx.stroke()
  ctx.font = '700 52px system-ui, sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillStyle = INK
  ctx.fillText(crewLetter(crew), 48, 52)
  return texture(c)
}

/** The detecting sensor's ring: dark band under a dashed yellow one. */
export function ringTexture(): CanvasTexture {
  const [c, ctx] = square(128)
  ctx.beginPath()
  ctx.arc(64, 64, 54, 0, Math.PI * 2)
  ctx.lineWidth = 16
  ctx.strokeStyle = INK
  ctx.stroke()
  ctx.lineWidth = 8
  ctx.strokeStyle = RING
  ctx.setLineDash([16, 10])
  ctx.stroke()
  return texture(c)
}

/** "Sensor 6 detects at 0:12" on a dark pill. Returns the texture and its width over height. */
export function calloutTexture(text: string): { texture: CanvasTexture; aspect: number } {
  const canvas = document.createElement('canvas')
  const h = 64
  const ctx = canvas.getContext('2d')!
  ctx.font = '700 32px system-ui, sans-serif'
  const w = Math.ceil(ctx.measureText(text).width) + 44
  canvas.width = w
  canvas.height = h
  ctx.font = '700 32px system-ui, sans-serif'
  ctx.beginPath()
  ctx.roundRect(3, 3, w - 6, h - 6, (h - 6) / 2)
  ctx.fillStyle = INK
  ctx.fill()
  ctx.lineWidth = 3
  ctx.strokeStyle = '#ffffff'
  ctx.stroke()
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillStyle = '#ffffff'
  ctx.fillText(text, w / 2, h / 2 + 2)
  return { texture: texture(canvas), aspect: w / h }
}
