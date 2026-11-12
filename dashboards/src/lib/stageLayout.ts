import { useMediaQuery } from './useMediaQuery'

/**
 * Below this width the controls sit behind a "Controls" button in a bottom sheet; from it up they
 * are a card beside the terrain. styles.css repeats the same query ("max-width: 999.98px" blocks
 * for the sheet), so change both together.
 */
export const SHEET_QUERY = '(max-width: 999.98px)'

export function useSheetLayout(): boolean {
  return useMediaQuery(SHEET_QUERY)
}
