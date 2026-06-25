import { useEffect, useMemo, useRef, useState } from 'react'
import { colorFor, labelFor } from '../lib/arms.js'

// "Monitor's-eye view": sweep a playhead across the wandering window into the fork
// (decision point) and watch the probe's real per-position belief over the K secret
// values on a single held-out sequence (data.examples). The hidden secret is a RUNNING
// aggregate S_t, so the true value moves as the walk proceeds — a carry model's belief
// tracks it; a defer model's goes diffuse, then snaps onto the right value at the fork.

const VISIBLE = '#5ee0a0'   // correct read
const BLIND = '#5b6680'     // diffuse / uncommitted
const WRONG = '#ff6b6b'     // confidently wrong

const norm = (conf, chance) => Math.max(0, Math.min(1, (conf - chance) / (1 - chance)))

// Three-state verdict from the probe's categorical readout:
//   correct — top guess == true value, peaked enough to commit (green)
//   wrong   — top guess != true value but still peaked: confidently wrong (red)
//   blind   — diffuse / ≈ uniform, no committed read (gray)
function cellVerdict(probs, trueV, chance, deferredFork) {
  const pmax = Math.max(...probs)
  const argmax = probs.indexOf(pmax)
  if (norm(pmax, chance) < 0.25)
    return { kind: 'blind', color: BLIND, icon: '○', argmax, label: 'blind — no confident read' }
  if (argmax === trueV)
    return {
      kind: 'correct', color: VISIBLE, icon: '✓', argmax,
      label: deferredFork ? '⚠ correct — only at the decision (too late)' : 'reading the secret correctly',
    }
  return { kind: 'wrong', color: WRONG, icon: '✗', argmax, label: 'confidently reading the WRONG value' }
}

function exampleSeries(data, arm, seed, exIdx, posList, K) {
  const sd = data.arms[arm]?.seeds?.[seed]
  const exs = data.examples?.[arm]?.[seed]
  const chance = data.meta.chance || 1 / K
  if (!exs || !exs[exIdx]) return null
  const ex = exs[exIdx]
  const byPos = {}
  ex.steps.forEach(s => { byPos[s.pos] = s })
  const pts = posList.map(p => {
    if (p === 'fork')
      return { pos: 'fork', isFork: true, trueV: ex.fork.true, probs: ex.fork.probs, conf: ex.fork.probs[ex.fork.true], chance }
    const s = byPos[p]
    const probs = s ? s.probs : Array(K).fill(chance)
    const trueV = s ? s.true_t : 0
    return { pos: p, trueV, probs, conf: probs[trueV], chance, isUpdate: s ? s.is_update : 0 }
  })
  const late = pts.filter(p => !p.isFork && typeof p.pos === 'number' && p.pos >= 16)
  const deferred = late.length > 0 && late.every(p => norm(p.conf, p.chance) < 0.25)
  return { sd, pts, deferred }
}

function Sparkline({ pts, idx, color }) {
  const W = 240, H = 40
  const n = pts.length
  const x = i => (n <= 1 ? W / 2 : (i / (n - 1)) * W)
  const y = c => H - 4 - norm(c.conf, c.chance) * (H - 8)
  const path = pts.map((c, i) => `${x(i).toFixed(1)},${y(c).toFixed(1)}`).join(' ')
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} aria-hidden>
      <line x1={0} x2={W} y1={H - 4} y2={H - 4} stroke="var(--border)" />
      <polyline points={path} fill="none" stroke={color} strokeWidth={2} opacity={0.7} />
      <line x1={x(idx)} x2={x(idx)} y1={2} y2={H - 2} stroke="var(--true)" strokeWidth={1.5} />
      <circle cx={x(idx)} cy={y(pts[idx])} r={3.5} fill={color} stroke="#0b0e14" strokeWidth={1} />
    </svg>
  )
}

function MindWindow({ cur, K, st }) {
  const probs = cur.probs || Array(K).fill(1 / K)
  const trueV = cur.trueV ?? 0
  const argmax = st.argmax
  return (
    <div style={{ background: 'var(--panel-2)', border: `1px solid ${st.color}55`, borderRadius: 10, padding: '14px 16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--muted)', marginBottom: 10 }}>
        <span>probe&apos;s belief over the running secret</span>
        <span>true now = <b style={{ color: 'var(--true)' }}>{trueV}</b></span>
      </div>
      <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
        {probs.map((p, k) => {
          const fill = k === trueV ? VISIBLE : (k === argmax && st.kind === 'wrong') ? WRONG : BLIND
          const ring = (k === argmax && st.kind !== 'blind') ? st.color : null
          return (
            <div key={k} style={{
              position: 'relative', width: 56, height: 66, borderRadius: 8, overflow: 'hidden', background: '#0d1117',
              border: k === trueV ? '2px solid var(--true)' : '1px solid var(--border)',
              boxShadow: ring ? `0 0 0 3px ${ring}` : 'none', transition: 'box-shadow .18s ease',
            }}>
              <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: `${p * 100}%`, background: fill, opacity: 0.25 + 0.65 * p, transition: 'height .18s ease, opacity .18s ease, background .18s ease' }} />
              <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: 20, color: 'var(--text)' }}>{k}</div>
              <div style={{ position: 'absolute', bottom: 2, right: 5, fontSize: 10, color: 'var(--muted)', fontVariantNumeric: 'tabular-nums' }}>{(p * 100).toFixed(0)}</div>
            </div>
          )
        })}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 9, marginTop: 12 }}>
        <span style={{
          width: 24, height: 24, borderRadius: '50%', background: st.color, color: '#0b0e14',
          display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800, fontSize: 14,
        }}>{st.icon}</span>
        <span style={{ fontWeight: 700, color: st.color, fontSize: 13 }}>{st.label}</span>
      </div>
    </div>
  )
}

function Lane({ data, arms, lane, setLane, idx, posList, K, exIdx }) {
  const series = useMemo(
    () => exampleSeries(data, lane.arm, lane.seed, exIdx, posList, K),
    [data, lane, posList, K, exIdx],
  )
  if (!series) return <div className="grid-card"><p className="muted">no example data for {lane.arm} seed{lane.seed}</p></div>
  const cur = series.pts[idx]
  const st = cellVerdict(cur.probs || Array(K).fill(1 / K), cur.trueV ?? 0, cur.chance, cur.isFork && series.deferred)
  const color = colorFor(lane.arm)
  const seedList = Object.keys(data.arms[lane.arm]?.seeds || {})

  return (
    <div className="grid-card">
      <div className="gc-head">
        <span style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
          <span className="swatch" style={{ background: color }} />
          <select className="arm-select" value={lane.arm}
            onChange={e => setLane({ ...lane, arm: e.target.value, seed: Object.keys(data.arms[e.target.value].seeds)[0] })}>
            {arms.map(a => <option key={a} value={a}>{labelFor(a)}</option>)}
          </select>
          <select className="arm-select" value={lane.seed}
            onChange={e => setLane({ ...lane, seed: e.target.value })}>
            {seedList.map(s => <option key={s} value={s}>seed {s}</option>)}
          </select>
        </span>
        <span className="gc-read">eff-rank {series.sd?.eff_rank ?? '—'}</span>
      </div>

      <MindWindow cur={cur} K={K} st={st} />

      <div style={{ marginTop: 12 }}>
        <Sparkline pts={series.pts} idx={idx} color={color} />
      </div>
    </div>
  )
}

export default function MindReader({ data }) {
  const arms = Object.keys(data.arms)
  const K = data.meta.K || 4

  const [laneA, setLaneA] = useState({ arm: 'nextlat_h1', seed: Object.keys(data.arms.nextlat_h1?.seeds || { 1234: 1 })[0] })
  const [laneB, setLaneB] = useState({ arm: 'gpt', seed: Object.keys(data.arms.gpt?.seeds || { 1234: 1 })[0] })
  const [idx, setIdx] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [exIdx, setExIdx] = useState(0)
  const [speedMs, setSpeedMs] = useState(900)
  const timer = useRef(null)

  const nEx = data.examples?.[laneA.arm]?.[laneA.seed]?.length || 0
  const safeEx = nEx > 0 ? exIdx % nEx : 0

  const posList = useMemo(() => {
    if (nEx > 0) {
      const ex = data.examples[laneA.arm][laneA.seed][safeEx]
      return [...ex.steps.map(s => s.pos), 'fork']
    }
    return ['fork']
  }, [data, nEx, laneA.arm, laneA.seed, safeEx])

  useEffect(() => { if (idx >= posList.length) setIdx(0) }, [posList.length, idx])
  useEffect(() => {
    if (!playing) return
    timer.current = setInterval(() => setIdx(i => (i + 1) % posList.length), speedMs)
    return () => clearInterval(timer.current)
  }, [playing, posList.length, speedMs])

  const cur = posList[idx]
  const posLabel = cur === 'fork' ? 'FORK · decision point' : `wandering step ${cur} / ${posList[posList.length - 2]}`
  const exTrueFinal = nEx > 0 ? data.examples[laneA.arm][laneA.seed][safeEx].true_final : null

  return (
    <div>
      <div className="controls">
        <button className="btn" onClick={() => setPlaying(p => !p)}>{playing ? '❚❚ pause' : '▶ play'}</button>
        <select value={speedMs} onChange={e => setSpeedMs(Number(e.target.value))} title="playback speed">
          <option value={1500}>0.5× slow</option>
          <option value={900}>1× normal</option>
          <option value={500}>2× fast</option>
        </select>
        <input type="range" min={0} max={posList.length - 1} value={Math.min(idx, posList.length - 1)}
          onChange={e => { setPlaying(false); setIdx(Number(e.target.value)) }} />
        <span className="step-label">{posLabel}</span>
        {nEx > 0 && (
          <button className="btn" onClick={() => { setExIdx(i => (i + 1) % nEx); setIdx(0) }}>
            ↻ example {safeEx + 1}/{nEx}{exTrueFinal != null ? ` · final=${exTrueFinal}` : ''}
          </button>
        )}
      </div>

      <div className="monitor-grids" style={{ marginTop: 18 }}>
        <Lane data={data} arms={arms} lane={laneA} setLane={setLaneA} idx={idx} posList={posList} K={K} exIdx={safeEx} />
        <Lane data={data} arms={arms} lane={laneB} setLane={setLaneB} idx={idx} posList={posList} K={K} exIdx={safeEx} />
      </div>

      <div className="legend">
        <span><span className="swatch" style={{ background: 'transparent', border: '2px solid var(--true)' }} /> true running value (outlined)</span>
        <span><span className="swatch" style={{ background: VISIBLE }} /> correct read ✓</span>
        <span><span className="swatch" style={{ background: WRONG }} /> confidently wrong ✗</span>
        <span><span className="swatch" style={{ background: BLIND }} /> blind / uncommitted ○</span>
      </div>
    </div>
  )
}
