// Small SVG line chart: 81-way position decodability vs layer depth, per arm.
const COLORS = { gpt: '#f2754f', nextlat: '#4fd1c5', mtp: '#b07cff', jtp: '#7c9cff', nextlat_h1: '#5ee0a0' }
const LABEL = { gpt: 'GPT (baseline)', nextlat: 'NextLat', mtp: 'MTP', jtp: 'JTP', nextlat_h1: 'NextLat-h1' }

export default function ByLayerChart({ arms }) {
  const W = 560, H = 260, m = { l: 44, r: 16, t: 16, b: 34 }
  const iw = W - m.l - m.r, ih = H - m.t - m.b
  const maxLayers = Math.max(...Object.values(arms).map(a => a.by_layer_acc.length))
  const x = (li, n) => m.l + (n <= 1 ? iw / 2 : (li / (n - 1)) * iw)
  const y = (acc) => m.t + (1 - acc) * ih
  const yticks = [0, 0.25, 0.5, 0.75, 1.0]
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="position decodability by layer">
      {yticks.map(t => (
        <g key={t}>
          <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} stroke="var(--border)" strokeWidth={1} />
          <text x={m.l - 8} y={y(t) + 4} textAnchor="end" fontSize="11" fill="var(--muted)">{t.toFixed(2)}</text>
        </g>
      ))}
      <text x={m.l + iw / 2} y={H - 6} textAnchor="middle" fontSize="11" fill="var(--muted)">layer depth →</text>
      <text x={14} y={m.t + ih / 2} textAnchor="middle" fontSize="11" fill="var(--muted)" transform={`rotate(-90 14 ${m.t + ih / 2})`}>decode accuracy</text>
      {Object.entries(arms).map(([arm, a]) => {
        const n = a.by_layer_acc.length
        const pts = a.by_layer_acc.map((acc, li) => `${x(li, n)},${y(acc)}`).join(' ')
        return (
          <g key={arm}>
            <polyline points={pts} fill="none" stroke={COLORS[arm] || '#888'} strokeWidth={2.4} />
            {a.by_layer_acc.map((acc, li) => (
              <circle key={li} cx={x(li, n)} cy={y(acc)} r={2.6} fill={COLORS[arm] || '#888'} />
            ))}
          </g>
        )
      })}
      {/* legend */}
      {Object.keys(arms).map((arm, i) => (
        <g key={arm} transform={`translate(${m.l + 6}, ${m.t + 6 + i * 16})`}>
          <rect width="11" height="11" rx="2" fill={COLORS[arm] || '#888'} />
          <text x="16" y="10" fontSize="11" fill="var(--text)">{LABEL[arm] || arm} · {(arms[arm].best_acc * 100).toFixed(0)}% · rank {arms[arm].eff_rank}</text>
        </g>
      ))}
    </svg>
  )
}
