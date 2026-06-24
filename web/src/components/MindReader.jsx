import { useEffect, useMemo, useRef, useState } from 'react'
import { colorFor, labelFor } from '../lib/arms.js'

// "Monitor's-eye view": sweep a playhead across the wandering window into the fork
// (decision point) and watch how readable the hidden RUNNING secret S_t is, per arm/seed.
//
//   'gauge' — monitor-confidence bar = the probe's decode accuracy at that offset
//             (aggregated over many example secrets).
//   'kcell' — a K-cell "mind window": the probe's real per-position belief over the K
//             values on a single held-out sequence (data.examples). The true running value
//             moves as the walk proceeds; a carry model's belief tracks it, a defer model's
//             goes flat. Falls back to a synthesised preview if examples aren't dumped yet.

const VISIBLE = '#5ee0a0'   // correct read
const FADING = '#e0a23f'
const BLIND = '#5b6680'     // diffuse / uncommitted
const WRONG = '#ff6b6b'     // confidently wrong

const frac = x => x - Math.floor(x)
const hash = s => Math.abs([...s].reduce((a, c) => a * 31 + c.charCodeAt(0), 7))
const norm = (conf, chance) => Math.max(0, Math.min(1, (conf - chance) / (1 - chance)))
function stateOf(n) {
  if (n >= 0.5) return { color: VISIBLE, label: 'secret visible to monitor' }
  if (n <= 0.15) return { color: BLIND, label: 'monitor blind' }
  return { color: FADING, label: 'fading…' }
}

// Three-state verdict for the K-cell window, from the probe's categorical readout:
//   correct  — top guess == true value, and the guess is peaked enough to commit (green)
//   wrong    — top guess != true value but still peaked: confidently wrong (red)
//   blind    — diffuse / ≈ uniform, no committed read (gray)
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

// synthesised per-cell distribution — fallback only, when real examples aren't present
function synthDist(conf, chance, K, pos, trueVal) {
  const pTrue = Math.max(chance * 0.9, Math.min(0.985, conf))
  const rest = 1 - pTrue
  const ws = []
  let sum = 0
  for (let k = 0; k < K; k++) {
    if (k === trueVal) { ws.push(null); continue }
    const w = 0.5 + frac(Math.sin((pos + 1) * 7.13 + (k + 1) * 3.7) * 43758.5453)
    ws.push(w); sum += w
  }
  return ws.map((w, k) => (k === trueVal ? pTrue : rest * (w / sum)))
}

// gauge series: per-offset accuracy curve (+ fork) for an arm/seed
function gaugeSeries(armData, seed, metric, posList) {
  const sd = armData?.seeds?.[seed]
  if (!sd) return null
  const byOff = {}
  sd.retention.forEach(r => { byOff[r.off] = r })
  let last = null
  const pts = posList.map(p => {
    if (p === 'fork') return { pos: 'fork', conf: sd.fork, chance: 0.25, isFork: true }
    const r = byOff[p] || last
    last = r
    return { pos: p, conf: r ? r[metric] : 0.25, chance: r ? r.chance : 0.25 }
  })
  const late = pts.filter(p => !p.isFork && typeof p.pos === 'number' && p.pos >= 16)
  const deferred = late.length > 0 && late.every(p => norm(p.conf, p.chance) < 0.25)
  return { sd, pts, deferred }
}

// kcell series: real per-position probs on one example sequence (+ fork)
function exampleSeries(data, arm, seed, exIdx, posList, K) {
  const sd = data.arms[arm]?.seeds?.[seed]
  const exs = data.examples?.[arm]?.[seed]
  const chance = data.meta.chance || 1 / K
  if (exs && exs[exIdx]) {
    const ex = exs[exIdx]
    const byPos = {}
    ex.steps.forEach(s => { byPos[s.pos] = s })
    const pts = posList.map(p => {
      if (p === 'fork') {
        return { pos: 'fork', isFork: true, trueV: ex.fork.true, probs: ex.fork.probs, conf: ex.fork.probs[ex.fork.true], chance }
      }
      const s = byPos[p]
      const probs = s ? s.probs : Array(K).fill(chance)
      const trueV = s ? s.true_t : 0
      return { pos: p, trueV, probs, conf: probs[trueV], chance, isUpdate: s ? s.is_update : 0 }
    })
    const late = pts.filter(p => !p.isFork && typeof p.pos === 'number' && p.pos >= 16)
    const deferred = late.length > 0 && late.every(p => norm(p.conf, p.chance) < 0.25)
    return { sd, pts, deferred, real: true }
  }
  // fallback: synth from the gauge curve
  const g = gaugeSeries(data.arms[arm], seed, 'lin', posList)
  if (!g) return null
  const trueVal = hash(arm + seed) % K
  g.pts = g.pts.map((p, i) => ({ ...p, trueV: trueVal, probs: synthDist(p.conf, p.chance, K, i, trueVal) }))
  return { ...g, real: false }
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

function Gauge({ cur, st }) {
  const n = norm(cur.conf, cur.chance)
  return (
    <div style={{ background: 'var(--panel-2)', border: '1px solid var(--border)', borderRadius: 10, padding: '14px 16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--muted)' }}>
        <span>monitor confidence</span>
        <span style={{ fontVariantNumeric: 'tabular-nums' }}>{(cur.conf * 100).toFixed(0)}%</span>
      </div>
      <div style={{ position: 'relative', height: 16, background: '#0d1117', borderRadius: 8, marginTop: 8, overflow: 'hidden' }}>
        <div style={{ position: 'absolute', inset: 0, width: `${n * 100}%`, background: st.color, transition: 'width .18s ease, background .18s ease' }} />
      </div>
      <div style={{ marginTop: 10, fontSize: 13, fontWeight: 700, color: st.color }}>{st.label}</div>
    </div>
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
          // semantic fill: true value = green; the model's confidently-wrong pick = red; else gray
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
      {/* status light — the at-a-glance verdict */}
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

function Lane({ data, arms, lane, setLane, idx, posList, variant, K, exIdx }) {
  const series = useMemo(() => (
    variant === 'kcell'
      ? exampleSeries(data, lane.arm, lane.seed, exIdx, posList, K)
      : gaugeSeries(data.arms[lane.arm], lane.seed, lane.metric, posList)
  ), [data, lane, posList, variant, K, exIdx])
  if (!series) return <div className="grid-card"><p className="muted">no data for {lane.arm} seed{lane.seed}</p></div>
  const cur = series.pts[idx]
  let st
  if (variant === 'kcell') {
    st = cellVerdict(cur.probs || Array(K).fill(1 / K), cur.trueV ?? 0, cur.chance, cur.isFork && series.deferred)
  } else {
    st = stateOf(norm(cur.conf, cur.chance))
    if (cur.isFork && series.deferred && norm(cur.conf, cur.chance) >= 0.5)
      st = { color: VISIBLE, label: '⚠ visible only at the decision — too late' }
  }
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

      {variant === 'kcell'
        ? <MindWindow cur={cur} K={K} st={st} />
        : <Gauge cur={cur} st={st} />}

      <div style={{ marginTop: 12 }}>
        <Sparkline pts={series.pts} idx={idx} color={color} />
      </div>
    </div>
  )
}

export default function MindReader({ data, variant = 'gauge' }) {
  const arms = Object.keys(data.arms)
  const K = data.meta.K || 4

  const [laneA, setLaneA] = useState({ arm: 'nextlat_h1', seed: Object.keys(data.arms.nextlat_h1?.seeds || { 1234: 1 })[0], metric: 'lin' })
  const [laneB, setLaneB] = useState({ arm: 'gpt', seed: Object.keys(data.arms.gpt?.seeds || { 1234: 1 })[0], metric: 'lin' })
  const [metric, setMetric] = useState('lin')
  const [idx, setIdx] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [exIdx, setExIdx] = useState(0)
  const [speedMs, setSpeedMs] = useState(900)
  const timer = useRef(null)

  // how many example sequences are available (shared across lanes — same sequences per arm)
  const nEx = data.examples?.[laneA.arm]?.[laneA.seed]?.length || 0
  const real = variant === 'kcell' && nEx > 0

  const posList = useMemo(() => {
    if (variant === 'kcell' && nEx > 0) {
      const ex = data.examples[laneA.arm][laneA.seed][exIdx % nEx]
      return [...ex.steps.map(s => s.pos), 'fork']
    }
    const offs = new Set()
    Object.values(data.arms).forEach(a => a.mean_retention.forEach(r => offs.add(r.off)))
    return [...[...offs].sort((a, b) => a - b), 'fork']
  }, [data, variant, nEx, laneA.arm, laneA.seed, exIdx])

  useEffect(() => { setLaneA(l => ({ ...l, metric })); setLaneB(l => ({ ...l, metric })) }, [metric])
  useEffect(() => { if (idx >= posList.length) setIdx(0) }, [posList.length, idx])

  useEffect(() => {
    if (!playing) return
    timer.current = setInterval(() => setIdx(i => (i + 1) % posList.length), speedMs)
    return () => clearInterval(timer.current)
  }, [playing, posList.length, speedMs])

  const cur = posList[idx]
  const posLabel = cur === 'fork' ? 'FORK · decision point' : `wandering step ${cur} / ${posList[posList.length - 2]}`
  const exTrueFinal = real ? data.examples[laneA.arm][laneA.seed][exIdx % nEx].true_final : null

  return (
    <div>
      {variant === 'kcell' && (
        <div style={{ display: 'inline-block', fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#0b0e14', background: real ? VISIBLE : 'var(--true)', borderRadius: 6, padding: '3px 9px', marginBottom: 12 }}>
          {real ? 'Real per-position probe readout' : 'Preview · illustrative (no dump yet)'}
        </div>
      )}
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
        {variant === 'kcell'
          ? (real && (
            <button className="btn" onClick={() => { setExIdx(i => (i + 1) % nEx); setIdx(0) }}>
              ↻ example {(exIdx % nEx) + 1}/{nEx}{exTrueFinal != null ? ` · final=${exTrueFinal}` : ''}
            </button>
          ))
          : (
            <button className="btn" onClick={() => setMetric(m => (m === 'lin' ? 'mlp' : 'lin'))}>
              probe: {metric === 'lin' ? 'linear' : 'MLP (nonlinear)'}
            </button>
          )}
      </div>

      <div className="monitor-grids" style={{ marginTop: 18 }}>
        <Lane data={data} arms={arms} lane={laneA} setLane={setLaneA} idx={idx} posList={posList} variant={variant} K={K} exIdx={exIdx % Math.max(1, nEx)} />
        <Lane data={data} arms={arms} lane={laneB} setLane={setLaneB} idx={idx} posList={posList} variant={variant} K={K} exIdx={exIdx % Math.max(1, nEx)} />
      </div>

      {variant === 'kcell' ? (
        <div className="legend">
          <span><span className="swatch" style={{ background: 'transparent', border: '2px solid var(--true)' }} /> true running value (outlined)</span>
          <span><span className="swatch" style={{ background: VISIBLE }} /> correct read ✓</span>
          <span><span className="swatch" style={{ background: WRONG }} /> confidently wrong ✗</span>
          <span><span className="swatch" style={{ background: BLIND }} /> blind / uncommitted ○</span>
        </div>
      ) : (
        <div className="legend">
          <span><span className="swatch" style={{ background: VISIBLE }} /> secret visible</span>
          <span><span className="swatch" style={{ background: FADING }} /> fading</span>
          <span><span className="swatch" style={{ background: BLIND }} /> monitor blind (≈ chance)</span>
          <span><span className="swatch" style={{ background: 'var(--true)' }} /> playhead</span>
        </div>
      )}
    </div>
  )
}
