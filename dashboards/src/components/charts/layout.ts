/** Gutters of one dot plot panel in px: method names, bare left edge, mean labels. */
export const PANEL_PAD = { label: 96, left: 8, right: 48 }

/** Stacks rows from `top`, adding `gap` wherever `group` changes; returns each row's top and the end y. */
export function stackRows<T extends { series: string }>(
  rows: T[],
  top: number,
  rowH: number,
  gap: number,
): { items: (T & { top: number })[]; end: number } {
  const items: (T & { top: number })[] = []
  let y = top
  for (let i = 0; i < rows.length; i++) {
    if (i > 0 && rows[i - 1].series !== rows[i].series) y += gap
    items.push({ ...rows[i], top: y })
    y += rowH
  }
  return { items, end: y }
}
