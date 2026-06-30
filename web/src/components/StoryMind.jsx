import { useEffect, useMemo, useRef, useState } from 'react'
import { colorFor, labelFor } from '../lib/arms.js'

// Demo 3: read the model's mind on a LANGUAGE surface. A narrative moves a ball around K
// shelves; a linear probe decodes the ball's CURRENT shelf from each frozen model's hidden
// state, per token. NextLat-1step keeps the belief legible throughout (carry); GPT goes blind
// mid-story and only reconstructs it at the query (defer). Precomputed; no model in the browser.

const VISIBLE = '#5ee0a0'   // correct read
const BLIND = '#5b6680'     // diffuse / uncommitted
const WRONG = '#ff6b6b'     // confidently wrong
const norm = (c, ch) => Math.max(0, Math.min(1, (c - ch) / (1 - ch)))
const argmax = p => p.reduce((b, v, i) => (v > p[b] ? i : b), 0)

function verdict(probs, trueV, chance) {
  const am = argmax(probs)
  if (norm(probs[am], chance) < 0.25) return { kind: 'blind', color: BLIND, icon: '○', am, label: 'monitor blind' }
  if (am === trueV) return { kind: 'correct', color: VISIBLE, icon: '✓', am, label: 'reading it correctly' }
  return { kind: 'wrong', color: WRONG, icon: '✗', am, label: 'confidently WRONG' }
}

function pretty(tok) {
  const m = /^shelf(\d+)$/.exec(tok)
  return m ? `shelf ${m[1]}` : tok
}

function MindWindow({ probs, trueV, K, color, st }) {
  return (
    <div style={{ background: 'var(--panel-2)', border: `1px solid ${st.color}55`, borderRadius: 10, padding: '12px 14px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--muted)', marginBottom: 8 }}>
        <span>probe: which shelf is the ball on?</span>
        <span>truth = <b style={{ color: 'var(--true)' }}>shelf {trueV}</b></span>
      </div>
      <div style={{ display: 'flex', gap: 7, justifyContent: 'center' }}>
        {probs.map((p, k) => {
          const fill = k === trueV ? VISIBLE : (k === st.am && st.kind === 'wrong') ? WRONG : BLIND
          const ring = (k === st.am && st.kind !== 'blind') ? st.color : null
          return (
            <div key={k} style={{
              position: 'relative', width: 52, height: 60, borderRadius: 8, overflow: 'hidden', background: '#0d1117',
              border: k === trueV ? '2px solid var(--true)' : '1px solid var(--border)',
              boxShadow: ring ? `0 0 0 3px ${ring}` : 'none', transition: 'box-shadow .15s ease',
            }}>
              <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: `${p * 100}%`, background: fill, opacity: 0.25 + 0.65 * p, transition: 'height .15s ease, opacity .15s ease, background .15s ease' }} />
              <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: 17, color: 'var(--text)' }}>{k}</div>
              <div style={{ position: 'absolute', bottom: 2, right: 4, fontSize: 10, color: 'var(--muted)', fontVariantNumeric: 'tabular-nums' }}>{(p * 100).toFixed(0)}</div>
            </div>
          )
        })}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, marginTop: 10 }}>
        <span style={{ width: 22, height: 22, borderRadius: '50%', background: st.color, color: '#0b0e14', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800, fontSize: 13 }}>{st.icon}</span>
        <span style={{ fontWeight: 700, color: st.color, fontSize: 13 }}>{st.label}</span>
      </div>
    </div>
  )
}

function Lane({ arm, probs, trueV, K, chance, accSoFar }) {
  const st = verdict(probs, trueV, chance)
  return (
    <div className="grid-card">
      <div className="gc-head">
        <span style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
          <span className="swatch" style={{ background: colorFor(arm) }} />
          <b>{labelFor(arm)}</b>
        </span>
        <span className="gc-read">read-accuracy so far: <b className={accSoFar >= 0.8 ? 'hit' : 'miss'}>{(accSoFar * 100).toFixed(0)}%</b></span>
      </div>
      <MindWindow probs={probs} trueV={trueV} K={K} color={colorFor(arm)} st={st} />
    </div>
  )
}

export default function StoryMind({ data }) {
  const K = data.meta.K
  const chance = data.meta.chance
  const [exIdx, setExIdx] = useState(0)
  const [idx, setIdx] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [speedMs, setSpeedMs] = useState(800)
  const timer = useRef(null)

  const ex = data.examples[exIdx]
  // frames: each narrative step, then a final "query" frame from the fork readout
  const frames = useMemo(() => {
    const f = ex.steps.map(s => ({ pos: s.pos, tok: s.tok, isUpdate: s.is_update, trueV: s.true_t, gpt: s.gpt, nl: s.nl, query: false }))
    f.push({ pos: ex.act_pos, tok: 'where is the ball?', trueV: ex.fork.true, gpt: ex.fork.gpt, nl: ex.fork.nl, query: true })
    return f
  }, [ex])

  useEffect(() => { setIdx(0) }, [exIdx])
  useEffect(() => { if (idx >= frames.length) setIdx(0) }, [frames.length, idx])
  useEffect(() => {
    if (!playing) return
    timer.current = setInterval(() => setIdx(i => (i + 1) % frames.length), speedMs)
    return () => clearInterval(timer.current)
  }, [playing, frames.length, speedMs])

  const cur = frames[Math.min(idx, frames.length - 1)]
  // running monitor read-accuracy up to the current step (narrative steps only)
  const upto = ex.steps.slice(0, cur.query ? ex.steps.length : idx + 1)
  const acc = arm => (upto.length ? upto.filter(s => argmax(s[arm]) === s.true_t).length / upto.length : 0)

  return (
    <div>
      <div className="controls">
        <button className="btn" onClick={() => setPlaying(p => !p)}>{playing ? '❚❚ pause' : '▶ play'}</button>
        <select value={speedMs} onChange={e => setSpeedMs(Number(e.target.value))} title="speed">
          <option value={1300}>0.5× slow</option>
          <option value={800}>1× normal</option>
          <option value={450}>2× fast</option>
        </select>
        <input type="range" min={0} max={frames.length - 1} value={Math.min(idx, frames.length - 1)}
          onChange={e => { setPlaying(false); setIdx(Number(e.target.value)) }} />
        <span className="step-label">{cur.query ? 'AT THE QUERY' : `token ${idx + 1}/${frames.length - 1}`}</span>
        <button className="btn" onClick={() => setExIdx(i => (i + 1) % data.examples.length)}>
          ↻ story {exIdx + 1}/{data.examples.length}
        </button>
      </div>

      {/* narrative text with the playhead — flex-wrap so tokens never overflow */}
      <div style={{ background: 'var(--panel-2)', border: '1px solid var(--border)', borderRadius: 12, padding: '14px 16px', margin: '16px 0', display: 'flex', flexWrap: 'wrap', alignItems: 'baseline', gap: '6px 3px', fontSize: 15 }}>
        {ex.tokens.map((t, i) => {
          const isCur = !cur.query && i === cur.pos
          const isQ = cur.query && i >= ex.act_pos - 5
          const active = isCur || isQ
          return (
            <span key={i} style={{
              padding: '2px 5px', borderRadius: 5, whiteSpace: 'nowrap',
              background: active ? 'var(--true)' : 'transparent',
              color: active ? '#0b0e14' : (i > ex.act_pos - 6 ? 'var(--muted)' : 'var(--text)'),
              fontWeight: active ? 700 : 400,
            }}>{pretty(t)}</span>
          )
        })}
      </div>

      <div className="monitor-grids">
        <Lane arm="gpt" probs={cur.gpt} trueV={cur.trueV} K={K} chance={chance} accSoFar={acc('gpt')} />
        <Lane arm="nextlat_h1" probs={cur.nl} trueV={cur.trueV} K={K} chance={chance} accSoFar={acc('nl')} />
      </div>

      <div className="legend">
        <span><span className="swatch" style={{ background: 'transparent', border: '2px solid var(--true)' }} /> true shelf (outlined)</span>
        <span><span className="swatch" style={{ background: VISIBLE }} /> correct read ✓</span>
        <span><span className="swatch" style={{ background: WRONG }} /> confidently wrong ✗</span>
        <span><span className="swatch" style={{ background: BLIND }} /> blind ○</span>
      </div>
    </div>
  )
}
