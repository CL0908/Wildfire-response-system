import { formatDate } from '../lib/format'

export default function Footer({ generated }: { generated: string }) {
  return (
    <footer className="site-footer">
      <p>
        Q-Pithak · built at the SEA Quantum Leader Summit 2026 hackathon · results generated{' '}
        {formatDate(generated)} ·{' '}
        <a href="https://github.com/CL0908/Wildfire-response-system">Source code</a>
      </p>
    </footer>
  )
}
