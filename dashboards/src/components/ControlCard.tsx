import { useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { useSheetLayout } from '../lib/stageLayout'

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
// Places where the card may stop scrolling without cutting a line of text in half.
const BREAKS = '.card-section, .method-row, .stats, .note'

/** Tab stops inside `root`; a radio group counts once, as the browser tabs it. */
function tabStops(root: HTMLElement): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((el) => {
    if (!(el instanceof HTMLInputElement) || el.type !== 'radio' || !el.name) return true
    const group = Array.from(root.querySelectorAll<HTMLInputElement>('input[type="radio"]')).filter(
      (r) => r.name === el.name,
    )
    return (group.find((r) => r.checked) ?? group[0]) === el
  })
}

/** Makes everything outside `el` inert (and so hidden from Tab and screen readers). Returns the undo. */
function inertOutside(el: HTMLElement): () => void {
  const changed: HTMLElement[] = []
  for (let node: HTMLElement = el; node.parentElement; node = node.parentElement) {
    for (const sibling of Array.from(node.parentElement.children)) {
      if (sibling !== node && sibling instanceof HTMLElement && !sibling.inert) {
        sibling.inert = true
        changed.push(sibling)
      }
    }
  }
  return () => {
    for (const s of changed) s.inert = false
  }
}

/**
 * Controls over the stage: a card beside the terrain on wide screens, a bottom sheet behind a
 * button below 1000 px. Where the content is taller than the room, the card scrolls, ends between
 * blocks rather than through a line, and fades at the bottom while there is more to see.
 */
export default function ControlCard({ children }: { children: ReactNode }) {
  const sheetLayout = useSheetLayout()
  const [open, setOpen] = useState(false)
  const [more, setMore] = useState(false)
  const titleId = useId()
  const opener = useRef<HTMLButtonElement>(null)
  const sheet = useRef<HTMLDivElement>(null)
  const card = useRef<HTMLElement>(null)
  const scroller = useRef<HTMLDivElement>(null)
  const sheetOpen = sheetLayout && open

  // A sheet left open when the screen grows past the breakpoint closes, and stays closed if it shrinks again.
  useEffect(() => {
    if (!sheetLayout) setOpen(false) // eslint-disable-line react-hooks/set-state-in-effect
  }, [sheetLayout])

  useEffect(() => {
    const el = sheet.current
    if (!sheetOpen || !el) return
    const undo = inertOutside(el)
    const openerEl = opener.current
    el.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpen(false)
        return
      }
      if (e.key !== 'Tab') return
      const stops = tabStops(el)
      if (stops.length === 0) {
        e.preventDefault()
        return
      }
      const first = stops[0]
      const last = stops[stops.length - 1]
      const active = document.activeElement
      if (e.shiftKey && (active === first || active === el || !el.contains(active))) {
        e.preventDefault()
        last.focus()
      } else if (!e.shiftKey && (active === last || !el.contains(active))) {
        e.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('keydown', onKey)
      undo()
      openerEl?.focus()
    }
  }, [sheetOpen])

  // Stop the scrolling area at a block boundary that fits, then show whether more lies below.
  const measure = () => {
    const sc = scroller.current
    if (!sc) return
    const scrolled = sc.scrollTop
    sc.style.maxHeight = ''
    const room = sc.clientHeight
    if (sc.scrollHeight > room + 1) {
      const base = sc.getBoundingClientRect().top - scrolled
      let stop = 0
      for (const b of Array.from(sc.querySelectorAll(BREAKS))) {
        const y = b.getBoundingClientRect().bottom - base
        if (y <= room + 0.5 && y > stop) stop = y
      }
      if (stop > room * 0.4) sc.style.maxHeight = `${Math.ceil(stop)}px`
    }
    sc.scrollTop = scrolled
    updateMore()
  }
  const updateMore = () => {
    const sc = scroller.current
    if (sc) setMore(sc.scrollTop + sc.clientHeight < sc.scrollHeight - 2)
  }
  useLayoutEffect(measure)
  useEffect(() => {
    const stage = (card.current ?? sheet.current)?.parentElement
    if (!stage) return
    const observer = new ResizeObserver(measure)
    observer.observe(stage)
    return () => observer.disconnect()
  })

  const body = (
    <>
      <div ref={scroller} className="card-scroll" onScroll={updateMore}>
        {children}
      </div>
      <span className="card-fade" data-more={more} aria-hidden="true" />
    </>
  )

  if (!sheetLayout) {
    return (
      <aside ref={card} className="stage-card glass" aria-label="Controls">
        {body}
      </aside>
    )
  }
  return (
    <>
      <button
        ref={opener}
        type="button"
        className="controls-open glass"
        aria-expanded={sheetOpen}
        aria-haspopup="dialog"
        onClick={() => setOpen(true)}
      >
        Controls
      </button>
      {sheetOpen && (
        <div
          ref={sheet}
          className="stage-card stage-sheet glass"
          role="dialog"
          aria-modal="true"
          aria-labelledby={titleId}
          tabIndex={-1}
        >
          <div className="sheet-head">
            <span id={titleId}>Controls</span>
            <button type="button" className="sheet-close" onClick={() => setOpen(false)}>
              Close
            </button>
          </div>
          {body}
        </div>
      )}
    </>
  )
}
