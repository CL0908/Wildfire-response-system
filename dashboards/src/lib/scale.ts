/** The 3D view's two scales: the 10 km analysis window or the whole park. */
export type Scale = 'window' | 'park'

/** Files under public/ for each scale (without the extension). */
export const SCALE_FILES: Record<Scale, { terrain: string; landcover: string }> = {
  window: { terrain: 'terrain', landcover: 'landcover' },
  park: { terrain: 'park_terrain', landcover: 'park_landcover' },
}

/** The numbers the About note quotes, per scale. */
export const SCALE_NOTE: Record<Scale, { exaggeration: number; demResolutionM: number }> = {
  window: { exaggeration: 2, demResolutionM: 30 },
  // The relief is 1.4% of the park's width against 10% of the window's, so it needs more lift to read.
  park: { exaggeration: 4, demResolutionM: 150 },
}

/** Width the window's constants were tuned for; a scene's extent factor is its width over this. */
const REFERENCE_WIDTH_M = 10000

/** The fire replay's marks: lengths in scene metres and sizes in CSS px (see lib/fireMarks.ts). */
export interface FireScale {
  /** Height of a mark above the ground (which is already exaggerated), over Landscape and over the data layers. */
  liftLandscape: number
  liftData: number
  /** Scene metres an ember is wide at a size factor of 1. */
  worldSize: number
  /** The marks are pulled this far towards the camera so a slope in front does not clip them. */
  pull: number
  /** Ember width in px at a size factor of 1, never below or above these. */
  minPx: number
  maxPx: number
  /** Smallest scorch mark in px. */
  minScorchPx: number
}

/**
 * Every value in the 3D scene that was written as a fixed number of metres, per scale. For the
 * window each value is exactly the one the scene always used (the extent factor `k` is 1); for the
 * park they follow the scene's width.
 */
export interface SceneScale {
  scale: Scale
  /** Scene width over REFERENCE_WIDTH_M (1 for the window). */
  k: number
  exaggeration: number
  /** Camera clip planes and the closest the camera may get to the centre, in scene metres. */
  near: number
  far: number
  minDistance: number
  /** Opening camera elevation above the horizon, radians. */
  startElevation: number
  /** FogExp2 density; haze depends on distance in units of the scene, so it falls as 1 / k. */
  fogDensity: number
  northLabelSize: number
  northLabelOffset: number
  skirtDepth: number
  /** Metres per sample while ray-marching the pointer pick (half the terrain's pixel spacing). */
  rayStep: number
  /** Ground texture texels per terrain pixel along each axis. */
  groundTexels: number
  /** Metres along a wall for one repeat of the strata texture. */
  strataRepeat: number
  /** Resolution of the elevation data, shown in the About note. */
  demResolutionM: number
  /** Individual trees and the sensor markers exist only in the window. */
  trees: boolean
  markers: boolean
  /** A real-time shadow map (window) or cast shadows baked into the ground texture (park). */
  shadowMap: boolean
  fire: FireScale
}

export function sceneScale(scale: Scale, widthM: number): SceneScale {
  if (scale === 'window') {
    return {
      scale,
      k: 1,
      exaggeration: SCALE_NOTE.window.exaggeration,
      near: 200,
      far: 120000,
      minDistance: 2500,
      startElevation: (52 * Math.PI) / 180,
      fogDensity: 0.000012,
      northLabelSize: 900,
      northLabelOffset: 700,
      skirtDepth: 150,
      rayStep: 15,
      groundTexels: 2,
      strataRepeat: 10000,
      demResolutionM: SCALE_NOTE.window.demResolutionM,
      trees: true,
      markers: true,
      shadowMap: true,
      fire: { liftLandscape: 200, liftData: 45, worldSize: 150, pull: 220, minPx: 9, maxPx: 22, minScorchPx: 4.5 },
    }
  }
  const k = widthM / REFERENCE_WIDTH_M
  return {
    scale,
    k,
    exaggeration: SCALE_NOTE.park.exaggeration,
    near: 200 * k,
    far: 120000 * k,
    minDistance: 2500 * k,
    startElevation: (38 * Math.PI) / 180,
    fogDensity: 0.000012 / k,
    northLabelSize: 900 * k,
    northLabelOffset: 700 * k,
    skirtDepth: 150 * k,
    rayStep: 75,
    groundTexels: 1,
    strataRepeat: REFERENCE_WIDTH_M * k,
    demResolutionM: SCALE_NOTE.park.demResolutionM,
    trees: false,
    markers: false,
    shadowMap: false,
    // Metres follow the scene's width. The park is seen from about k times as far, so the same
    // metres would fall below a pixel; the px clamps are smaller than the window's so 16,000
    // detections do not merge into one mass.
    fire: {
      // No trees to clear in the park, so Landscape lifts the marks no higher than the data layers do.
      liftLandscape: 45 * k,
      liftData: 45 * k,
      worldSize: 150 * k,
      pull: 220 * k,
      minPx: 5,
      maxPx: 12,
      minScorchPx: 2,
    },
  }
}
