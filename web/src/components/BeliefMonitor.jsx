import { useState, useEffect, useMemo } from 'react'
import GridHeatmap from './GridHeatmap.jsx'

const LABEL = { gpt: 'GPT', nextlat: 'NextLat', mtp: 'MTP', jtp: 'JTP', nextlat_h1: 'NextLat · 1 step ahead' }
const COLOR = { gpt: '#f2754f', nextlat: '#4fd1c5', mtp: '#b07cff', jtp: '#7c9cff', nextlat_h1: '#5ee0a0' }

const top1 = (s) => (s && s.length ? s[0] : null)

function GridCard({ arm, setArm, armOptions, armMeta, step, gridSize, traps }) {
  const belief = step[arm]
  const t1 = top1(belief)
  const hit = t1 && t1[0] === step.true
  const color = COLOR[arm] || '#fff'
  return (
    <div className="grid-card">
      <div className="gc-head">
        <select className="arm-select" value={arm} onChange={(e) => setArm(e.target.value)}
          style={{ color, borderColor: color + '66' }}>
          {armOptions.map((a) => <option key={a} value={a}>{LABEL[a] || a}</option>)}
        </select>
        <span className="gc-read">
          {t1 ? <>cell <b className={hit ? 'hit' : 'miss'}>{t1[0]} {hit ? '✓' : '✗'}</b> · p={t1[1].toFixed(2)}</> : '·'}
        </span>
      </div>
      <GridHeatmap belief={belief} trueCell={step.true} traps={traps} gridSize={gridSize} hue={arm} />
      {armMeta && (
        <div className="gc-foot">decode {Math.round(armMeta.best_acc * 100)}% · effective rank {armMeta.eff_rank}</div>
      )}
    </div>
  )
}

export default function BeliefMonitor({ data }) {
  const { meta, trajectories, arms } = data
  const armKeys = Object.keys(arms)
  const [ti, setTi] = useState(0)
  const [si, setSi] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [leftArm, setLeftArm] = useState(armKeys.includes('gpt') ? 'gpt' : armKeys[0])
  const [rightArm, setRightArm] = useState(armKeys.includes('nextlat') ? 'nextlat' : armKeys[1] || armKeys[0])

  const traj = trajectories[ti]
  const steps = traj.steps
  const step = steps[Math.min(si, steps.length - 1)]

  useEffect(() => { setSi(0) }, [ti])
  useEffect(() => {
    if (!playing) return
    const id = setInterval(() => setSi((s) => (s + 1) % steps.length), 750)
    return () => clearInterval(id)
  }, [playing, steps.length])

  const acc = useMemo(() => {
    const hit = (a) => steps.filter(s => top1(s[a]) && top1(s[a])[0] === s.true).length / steps.length
    return { left: hit(leftArm), right: hit(rightArm) }
  }, [ti, leftArm, rightArm])

  return (
    <div>
      <div className="monitor-grids">
        <GridCard arm={leftArm} setArm={setLeftArm} armOptions={armKeys} armMeta={arms[leftArm]}
          step={step} gridSize={meta.grid_size} traps={meta.trap_cells} />
        <GridCard arm={rightArm} setArm={setRightArm} armOptions={armKeys} armMeta={arms[rightArm]}
          step={step} gridSize={meta.grid_size} traps={meta.trap_cells} />
      </div>

      <div className="legend">
        <span><span className="swatch" style={{ background: 'var(--true)' }} /> true position</span>
        <span><span className="swatch" style={{ background: 'rgba(255,255,255,0.28)', borderRadius: '50%' }} /> trap cell</span>
        <span>brighter cell = probe is more confident the model is there</span>
        <span style={{ color: 'var(--accent)' }}>tip: swap either panel to any model; every predictive one decodes sharply (that&apos;s the &ldquo;generic&rdquo; part).</span>
      </div>

      <div className="controls">
        <button className="btn" onClick={() => setPlaying(p => !p)}>{playing ? '❚❚ pause' : '▶ play'}</button>
        <span className="step-label">step {Math.min(si, steps.length - 1) + 1} / {steps.length}</span>
        <input type="range" min={0} max={steps.length - 1} value={Math.min(si, steps.length - 1)}
          onChange={(e) => { setPlaying(false); setSi(Number(e.target.value)) }} />
        <select value={ti} onChange={(e) => setTi(Number(e.target.value))}>
          {trajectories.map((_, i) => <option key={i} value={i}>trajectory {i + 1}</option>)}
        </select>
      </div>

      <div className="stats" style={{ marginTop: 22 }}>
        <div className="stat"><div className="v" style={{ color: COLOR[leftArm] }}>{(acc.left * 100).toFixed(0)}%</div><div className="k">{LABEL[leftArm] || leftArm} correct (this trajectory)</div></div>
        <div className="stat"><div className="v" style={{ color: COLOR[rightArm] }}>{(acc.right * 100).toFixed(0)}%</div><div className="k">{LABEL[rightArm] || rightArm} correct (this trajectory)</div></div>
      </div>
    </div>
  )
}
