import { useEffect, useState } from 'react'
import BeliefMonitor from '../components/BeliefMonitor.jsx'
import ByLayerChart from '../components/ByLayerChart.jsx'

export default function Demo() {
  const [data, setData] = useState(null)
  const [err, setErr] = useState(null)

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/demo.json`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(setData)
      .catch(e => setErr(String(e)))
  }, [])

  if (err) return <div className="wrap"><p className="muted" style={{ paddingTop: 60 }}>Failed to load demo data: {err}</p></div>
  if (!data) return <div className="wrap"><p className="muted" style={{ paddingTop: 60 }}>Loading…</p></div>

  const g = data.arms.gpt, n = data.arms.nextlat
  return (
    <div className="wrap">
      <header className="hero">
        <div className="eyebrow">NextLat for AI safety · interactive demo</div>
        <h1>Can you read a model&apos;s <span className="grad">belief state</span>?</h1>
        <p className="lede">
          One of these transformers was trained to predict its own future (NextLat); the other is a
          vanilla next-token model (GPT). Both navigate a grid blindfolded, and we try to read each
          one&apos;s mind with a simple probe. Watch how legible each turns out to be.
        </p>
      </header>

      <div className="panel">
        <div className="section-label">How to read this</div>
        <h2>You&apos;re watching a probe read each model&apos;s mind</h2>
        <div className="how">
          <div className="how-step">
            <span className="n">1</span>
            <p>Two transformers navigate a 9×9 grid. They <b>never see their position</b>, only
            nearby walls and their own moves, so each must <b>track where it is</b> internally.
            That running estimate is its <em>belief state</em>.</p>
          </div>
          <div className="how-step">
            <span className="n">2</span>
            <p>We freeze each model and train a tiny <b>linear probe</b> to decode that internal
            guess: <em>&ldquo;given your hidden state, which cell are you on?&rdquo;</em></p>
          </div>
          <div className="how-step">
            <span className="n">3</span>
            <p>Each grid below is the probe&apos;s decoded guess. <b>Brighter means more confident.</b>
            The <b style={{ color: 'var(--true)' }}>gold outline</b> marks the model&apos;s true cell.</p>
          </div>
        </div>
        <div className="takeaway" style={{ marginTop: 18 }}>
          This is <b>not</b> predicting the future. It&apos;s reading the model&apos;s belief about
          <b> where it is right now</b>. The sharper and more correct a model&apos;s grid, the more
          <b> monitorable</b> its internal state. (Because position is never in the input, this is a
          genuine belief-state task, exactly the regime where the gap shows up.)
        </div>
      </div>

      <div className="panel">
        <div className="section-label">The belief-state monitor</div>
        <h2>Decoding &ldquo;where do you think you are?&rdquo; from frozen hidden states</h2>
        <p className="sub">
          Each grid shows the probe&apos;s decoded probability over the {data.meta.n_cells} cells, step by
          step. Gold outline = the model&apos;s true position. Press play.
        </p>
        <BeliefMonitor data={data} />
        <div className="takeaway">
          Same task, same probe, same training budget. Only the training objective differs.
          <b> NextLat&apos;s belief lands on a single bright cell; GPT&apos;s smears across the board and
          often backs the wrong one.</b> The information is in there either way (the model navigates
          fine); the predictive objective makes it <em>linearly legible</em>.
        </div>
      </div>

      <div className="panel">
        <div className="section-label">It sharpens with depth</div>
        <h2>Position decodability by layer</h2>
        <p className="sub">
          Best-layer linear accuracy at decoding the exact cell (chance ≈ {(1 / data.meta.n_cells * 100).toFixed(1)}%).
          GPT plateaus; <b>every predictive objective</b> (NextLat, MTP, JTP) climbs to ~0.95, so the
          monitorability gain is generic, not unique to NextLat. The number after each label is its
          effective rank (lower is more compact).
        </p>
        <ByLayerChart arms={data.arms} />
        <div className="stats" style={{ marginTop: 18 }}>
          <div className="stat"><div className="v gpt">{(g.best_acc * 100).toFixed(0)}%</div><div className="k">GPT best decode</div></div>
          <div className="stat"><div className="v nextlat">{(n.best_acc * 100).toFixed(0)}%</div><div className="k">predictive arms best decode</div></div>
          <div className="stat"><div className="v nextlat">{n.eff_rank}</div><div className="k">NextLat rank (compact)</div></div>
          <div className="stat"><div className="v">{data.arms.mtp ? data.arms.mtp.eff_rank : '?'}</div><div className="k">MTP rank (diffuse)</div></div>
        </div>
        <div className="takeaway">
          Same decodability, very different compactness. All the predictive objectives reach ~0.95,
          but their effective ranks spread widely (NextLat ≈{n.eff_rank} vs MTP ≈{data.arms.mtp ? data.arms.mtp.eff_rank : '?'}).
          <b> NextLat is the only one that is both highly monitorable and highly compact</b>; that
          compactness is its distinctive property, and whether it buys a monitor anything is the open question.
        </div>
      </div>

      <div className="panel">
        <div className="section-label">What this is (and isn&apos;t)</div>
        <h2>The honest version</h2>
        <p className="muted" style={{ marginTop: 0 }}>
          This is a controlled, small-scale (≈6M-param) existence proof on a synthetic gridworld, not
          a language or agentic claim. The monitorability gain is <b>generic to predict-ahead
          training</b> (multi-token objectives match NextLat); NextLat&apos;s distinctive property is the
          compression you see here. The effect <b>survives a 4× scale-up</b> of the baseline. The open
          question is whether that compactness buys a real monitor anything.
        </p>
      </div>

      <footer>
        Precomputed from frozen checkpoints; no model runs in your browser.<br />
        <span style={{ fontSize: 12 }}>{data.meta.note}</span>
      </footer>
    </div>
  )
}
