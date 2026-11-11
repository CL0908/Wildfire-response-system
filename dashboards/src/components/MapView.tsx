import { useMemo, useState } from 'react'
import { rampColor } from '../lib/color'
import {
  LAYERS,
  NO_FIRE_FILL,
  availableLayers,
  cellFill,
  formatLayerValue,
  layerRange,
  legendT,
} from '../lib/layers'
import type { LayerKey, Method, Results } from '../types'

const CELL = 20
const LEGEND_STEPS = 8

interface Props {
  results: Results
  layer: LayerKey
  method: Method
}

export default function MapView({ results, layer, method }: Props) {
  const { grid, layers, sites, viewsheds } = results
  const { rows, cols } = grid
  const [hoverCell, setHoverCell] = useState<number | null>(null)
  const [activeSite, setActiveSite] = useState<number | null>(null)

  const info = LAYERS.find((l) => l.key === layer)!
  const shownLayers = useMemo(() => availableLayers(layers), [layers])
  const [lo, hi] = useMemo(() => layerRange(layers, layer), [layers, layer])
  const covered = useMemo(() => new Set(method.covered_cells), [method])
  const chosen = useMemo(() => new Set(method.chosen), [method])
  const siteView = useMemo(
    () => (activeSite === null ? null : new Set(viewsheds[String(activeSite)] ?? [])),
    [activeSite, viewsheds],
  )

  const cells = useMemo(() => {
    const out: { idx: number; r: number; c: number; fill: string }[] = []
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        out.push({
          idx: r * cols + c,
          r,
          c,
          fill: cellFill(info, layers[layer]![r][c], lo, hi),
        })
      }
    }
    return out
  }, [layers, layer, info, lo, hi, rows, cols])

  const activeSiteInfo = sites.find((s) => s.id === activeSite) ?? null
  const shownCell =
    activeSiteInfo !== null
      ? activeSiteInfo.row * cols + activeSiteInfo.col
      : hoverCell
  const sr = shownCell === null ? 0 : Math.floor(shownCell / cols)
  const sc = shownCell === null ? 0 : shownCell % cols

  const siteSummary = (id: number) => {
    const cellsOfSite = viewsheds[String(id)] ?? []
    const pr = cellsOfSite.reduce(
      (sum, i) => sum + layers.priority[Math.floor(i / cols)][i % cols],
      0,
    )
    return `Site ${id}: viewshed ${cellsOfSite.length} cells, priority sum ${pr.toFixed(2)}`
  }

  const labelOnLeft = (col: number) => col > cols - 4

  return (
    <div className="map">
      <svg
        className="map-svg"
        viewBox={`0 0 ${cols * CELL} ${rows * CELL}`}
        role="group"
        aria-label={`Map of ${rows} by ${cols} cells coloured by ${info.label}. ${method.name} places sensors at sites ${method.chosen.join(', ')}, covering ${method.covered_cells.length} of ${rows * cols} cells.`}
        onMouseLeave={() => setHoverCell(null)}
      >
        <defs>
          <pattern
            id="hatch"
            width="6"
            height="6"
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <rect width="6" height="6" fill="#fff" fillOpacity="0.5" />
            <line x1="0" y1="0" x2="0" y2="6" stroke="#4b5563" strokeWidth="1.4" />
          </pattern>
        </defs>

        <g shapeRendering="crispEdges" aria-hidden="true">
          {cells.map((c) => (
            <rect
              key={c.idx}
              x={c.c * CELL}
              y={c.r * CELL}
              width={CELL}
              height={CELL}
              fill={c.fill}
              stroke="#fff"
              strokeOpacity="0.5"
              strokeWidth="0.5"
              onMouseEnter={() => setHoverCell(c.idx)}
            />
          ))}
          <g pointerEvents="none">
            {cells
              .filter((c) => !covered.has(c.idx))
              .map((c) => (
                <rect
                  key={c.idx}
                  x={c.c * CELL}
                  y={c.r * CELL}
                  width={CELL}
                  height={CELL}
                  fill="url(#hatch)"
                />
              ))}
            {siteView &&
              cells
                .filter((c) => siteView.has(c.idx))
                .map((c) => (
                  <rect
                    key={c.idx}
                    x={c.c * CELL + 1}
                    y={c.r * CELL + 1}
                    width={CELL - 2}
                    height={CELL - 2}
                    fill="none"
                    stroke="var(--accent)"
                    strokeWidth="1.6"
                  />
                ))}
            {shownCell !== null && (
              <rect
                x={sc * CELL}
                y={sr * CELL}
                width={CELL}
                height={CELL}
                fill="none"
                stroke="#111827"
                strokeWidth="1.5"
              />
            )}
          </g>
        </g>

        <g>
          {sites.map((s) => {
            const isChosen = chosen.has(s.id)
            const cx = s.col * CELL + CELL / 2
            const cy = s.row * CELL + CELL / 2
            const left = labelOnLeft(s.col)
            return (
              <g
                key={s.id}
                tabIndex={0}
                role="img"
                aria-label={`Site ${s.id}, row ${s.row}, column ${s.col}, ${isChosen ? `sensor placed by ${method.name}` : 'candidate, not chosen'}. ${(viewsheds[String(s.id)] ?? []).length} cells in view.`}
                className="site"
                onMouseEnter={() => setActiveSite(s.id)}
                onMouseLeave={() => setActiveSite(null)}
                onFocus={() => setActiveSite(s.id)}
                onBlur={() => setActiveSite(null)}
              >
                <circle cx={cx} cy={cy} r={CELL * 0.7} fill="transparent" />
                {isChosen ? (
                  <>
                    <circle
                      className="marker"
                      cx={cx}
                      cy={cy}
                      r="7"
                      fill="var(--accent)"
                      stroke="#fff"
                      strokeWidth="2"
                    />
                    <text
                      x={cx + (left ? -11 : 11)}
                      y={cy}
                      textAnchor={left ? 'end' : 'start'}
                      dominantBaseline="central"
                      className="site-label"
                      pointerEvents="none"
                    >
                      {s.id}
                    </text>
                  </>
                ) : (
                  <circle
                    className="marker"
                    cx={cx}
                    cy={cy}
                    r="3"
                    fill="#f9fafb"
                    stroke="#1f2937"
                    strokeWidth="1.2"
                  />
                )}
              </g>
            )
          })}
        </g>
      </svg>

      <div className="readout" aria-live="polite">
        {shownCell === null ? (
          <span className="muted">Hover a cell or focus a site for details.</span>
        ) : (
          <>
            <div>
              <strong>
                Row {sr}, column {sc}
              </strong>
              {' · '}
              {covered.has(shownCell) ? 'covered' : 'not covered'} by {method.name}
            </div>
            <div className="muted">
              {shownLayers
                .map((l) => `${l.label} ${formatLayerValue(l, layers[l.key]![sr][sc])}`)
                .join(' · ')}
            </div>
            {activeSiteInfo && <div>{siteSummary(activeSiteInfo.id)}</div>}
          </>
        )}
      </div>

      <div className="legends">
        <div className="legend">
          <span className="legend-title">{info.label}</span>
          {info.scale === 'fires' && (
            <>
              <span className="ramp" aria-hidden="true">
                <i style={{ background: NO_FIRE_FILL }} />
              </span>
              <span className="muted">0</span>
            </>
          )}
          <span className="muted">
            {formatLayerValue(info, info.scale === 'fires' ? Math.min(1, hi) : lo)}
          </span>
          <span className="ramp" aria-hidden="true">
            {Array.from({ length: LEGEND_STEPS }, (_, i) => (
              <i key={i} style={{ background: rampColor(legendT(info, i, LEGEND_STEPS)) }} />
            ))}
          </span>
          <span className="muted">
            {formatLayerValue(info, hi)}
            {info.scale === 'fires' ? ' detections' : ''}
          </span>
        </div>
        <ul className="key" aria-label="Overlay key">
          <li>
            <svg width="16" height="16" aria-hidden="true">
              <defs>
                <pattern
                  id="hatch-key"
                  width="6"
                  height="6"
                  patternUnits="userSpaceOnUse"
                  patternTransform="rotate(45)"
                >
                  <rect width="6" height="6" fill="#fff" fillOpacity="0.5" />
                  <line x1="0" y1="0" x2="0" y2="6" stroke="#4b5563" strokeWidth="1.4" />
                </pattern>
              </defs>
              <rect width="16" height="16" fill="#c9dbd6" />
              <rect width="16" height="16" fill="url(#hatch-key)" />
            </svg>
            Not covered
          </li>
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
      </div>
    </div>
  )
}
