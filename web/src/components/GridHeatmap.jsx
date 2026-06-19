// A grid_size x grid_size heatmap of the probe's decoded belief over board cells.
// belief: sparse list [[cell, prob], ...] (top-k); the rest render as ~empty.
// trueCell is outlined in gold; trap cells get a small marker.

const HUE = {
  gpt: [242, 117, 79],       // warm baseline
  nextlat: [79, 209, 197],   // teal
  mtp: [176, 124, 255],      // purple
  jtp: [124, 156, 255],      // blue
  nextlat_h1: [94, 224, 160], // green
}

export default function GridHeatmap({ belief, trueCell, traps, gridSize, hue }) {
  const [r, g, b] = HUE[hue] || HUE.nextlat
  const probByCell = new Map((belief || []).map(([c, p]) => [c, p]))
  const trapSet = new Set(traps || [])
  const cell = 10
  const gap = 0.7
  const size = gridSize * cell
  const cells = []
  for (let idx = 0; idx < gridSize * gridSize; idx++) {
    const row = Math.floor(idx / gridSize)
    const col = idx % gridSize
    const p = probByCell.get(idx) || 0
    // perceptual-ish ramp so small probabilities still register
    const alpha = p > 0 ? 0.12 + 0.88 * Math.pow(p, 0.7) : 0
    const x = col * cell + gap / 2
    const y = row * cell + gap / 2
    const isTrue = idx === trueCell
    cells.push(
      <g key={idx}>
        <rect
          x={x} y={y} width={cell - gap} height={cell - gap} rx={1.4}
          fill={p > 0 ? `rgba(${r},${g},${b},${alpha})` : 'rgba(255,255,255,0.035)'}
        />
        {trapSet.has(idx) && (
          <circle cx={x + (cell - gap) / 2} cy={y + (cell - gap) / 2} r={0.9} fill="rgba(255,255,255,0.28)" />
        )}
        {isTrue && (
          <rect
            x={x + 0.4} y={y + 0.4} width={cell - gap - 0.8} height={cell - gap - 0.8} rx={1.4}
            fill="none" stroke="var(--true)" strokeWidth={1.1}
          />
        )}
      </g>
    )
  }
  return (
    <svg viewBox={`0 0 ${size} ${size}`} width="100%" style={{ display: 'block', borderRadius: 10, background: 'rgba(0,0,0,0.25)' }}>
      {cells}
    </svg>
  )
}
