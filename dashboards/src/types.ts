export interface Grid {
  rows: number
  cols: number
  cell_size_m: number
}

export type CoreLayerKey = 'priority' | 'hazard' | 'value' | 'elevation_m'

/** Only present in results.json when the terrain is real. */
export type OptionalLayerKey = 'inside_park' | 'fire_count'

export type LayerKey = CoreLayerKey | OptionalLayerKey

export type Layers = Record<CoreLayerKey, number[][]> &
  Partial<Record<OptionalLayerKey, number[][]>>

export interface Bounds {
  north: number
  south: number
  west: number
  east: number
}

export interface Site {
  id: number
  row: number
  col: number
}

export interface Sampling {
  shots: number
  p_optimal: number
  p_optimal_random: number
  p_feasible: number
  mean_ratio: number
}

export interface Method {
  name: string
  kind: 'classical' | 'quantum (simulated)'
  /** For quantum methods: the best of `sampling.shots` sampled shots. */
  chosen: number[]
  objective: number
  ratio: number
  covered_cells: number[]
  sampling: Sampling | null
}

/** One row of approx_ratio.csv; every value is a string. */
export interface BenchmarkRow {
  instance: string
  ilp_objective: string
  greedy: string
  sa: string
  random_mean: string
  qaoa_mean: string
  xy_qaoa_mean: string
  wsxy_start_mean: string
  wsxy_qaoa_mean: string
  /** Present once rows are averages over several instances of one size. */
  n_instances?: string
  terrain?: string
  /** `<column>_std` holds the standard deviation of a mean column. */
  [column: string]: string | undefined
}

export interface Results {
  generated: string
  terrain: 'real' | 'synthetic'
  /** `sources`, `modelled` and `bounds` are only present when terrain is real. */
  sources?: string[]
  modelled?: string
  bounds?: Bounds
  grid: Grid
  mast_height_m: number
  range_m: number
  n_sensors: number
  total_priority: number
  optimum: number
  layers: Layers
  sites: Site[]
  viewsheds: Record<string, number[]>
  methods: Method[]
  benchmark: BenchmarkRow[]
}
