import { useEffect, useState } from 'react'
import MindReader from '../components/MindReader.jsx'

export default function Concealworld() {
  const [data, setData] = useState(null)
  const [err, setErr] = useState(null)

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/concealworld.json`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(setData)
      .catch(e => setErr(String(e)))
  }, [])

  if (err) return <div className="wrap"><p className="muted" style={{ paddingTop: 60 }}>Failed to load data: {err}</p></div>
  if (!data) return <div className="wrap"><p className="muted" style={{ paddingTop: 60 }}>Loading…</p></div>

  const order = ['gpt', 'jtp', 'mtp', 'nextlat', 'nextlat_h1']
  const armsSorted = order.filter(a => data.arms[a]).concat(Object.keys(data.arms).filter(a => !order.includes(a)))

  return (
    <div className="wrap">
      <header className="hero">
        <div className="eyebrow">monitorable belief state · concealment</div>
        <h1>Can you catch a model <span className="grad">hiding a secret</span>?</h1>
        <p className="lede">
          A model is given a secret it must <b>use later</b> but <b>hide</b> right now. As it acts
          normally, we probe its hidden state to read the secret out. Some training objectives keep
          the secret in mind the whole time — monitorable. Others let it fade to nothing, then snap it
          back at the last instant. Watch the monitor succeed, or go blind.
        </p>
      </header>

      <div className="panel">
        <div className="section-label">How to read this</div>
        <h2>You are the safety monitor, reading the model&apos;s mind</h2>
        <div className="how">
          <div className="how-step">
            <span className="n">1</span>
            <p>The model tracks a hidden <b>secret</b> (1 of {data.meta.K}) it must act on at a single
            late <em>fork</em>. While it <b>wanders</b>, its behaviour reveals nothing — so the secret
            must live in its internal state, or nowhere.</p>
          </div>
          <div className="how-step">
            <span className="n">2</span>
            <p>At each step we freeze the model and train a probe to decode the secret from its hidden
            state. The bar is the <b>monitor&apos;s confidence</b> — how readable the secret is right
            now (flat at chance = <b>blind</b>).</p>
          </div>
          <div className="how-step">
            <span className="n">3</span>
            <p>A model can <b>keep</b> the secret in mind (readable throughout) or <b>defer</b> — drop
            it and reassemble it at the fork. Defer succeeds at the task but leaves the monitor
            <b> blind until the decision is already happening</b>.</p>
          </div>
        </div>
        <div className="takeaway" style={{ marginTop: 18 }}>
          This is the failure mode that matters for oversight: a model can <b>use</b> information
          without ever holding it where a probe can see it. The training objective decides whether
          your monitor has anything to read.
        </div>
      </div>

      <div className="panel">
        <div className="section-label">The monitor&apos;s-eye view</div>
        <h2>Watch the secret stay — or fade then snap back</h2>
        <p className="sub">
          Press play. The playhead sweeps the wandering window into the fork (the decision point).
          Default matchup: <b>NextLat-h1 keeps the secret</b> vs <b>GPT drops it</b> — swap either arm
          or seed to set up your own, and cycle the <b>↻ example</b> sequences (same sequence shown for
          both models). The secret is a running aggregate, so the gold “true” cell <b>moves</b> as the
          walk proceeds — a carrying model&apos;s belief tracks it; a deferring model&apos;s scatters.
        </p>
        <MindReader data={data} />
        <div className="takeaway">
          GPT&apos;s confidence drains to chance mid-wander and only <b>snaps back at the fork</b> — too
          late for oversight. NextLat-h1 holds the secret the whole way. The information isn&apos;t
          gone from GPT (it acts correctly); it&apos;s just <em>not where a monitor can read it</em>
          until the moment of action.
        </div>
      </div>

      <div className="panel">
        <div className="section-label">It depends on the objective — and the run</div>
        <h2>Who keeps the secret?</h2>
        <p className="sub">
          Per-seed count of runs that keep the secret readable across the late wandering window
          (3 seeds each; “learned” = the run actually solved the commit task).
        </p>
        <div className="stats">
          {armsSorted.map(a => {
            const s = data.arms[a].summary
            const cls = a === 'gpt' ? 'gpt' : a.startsWith('nextlat') ? 'nextlat' : ''
            return (
              <div className="stat" key={a}>
                <div className={`v ${cls}`}>{s.carry}/{s.n_seeds}</div>
                <div className="k">{a === 'nextlat' ? 'NextLat (h8)' : a === 'nextlat_h1' ? 'NextLat-h1' : a.toUpperCase()}
                  {s.learned < s.n_seeds ? ` · ${s.learned}/${s.n_seeds} learned` : ''}</div>
              </div>
            )
          })}
        </div>
        <div className="takeaway">
          Only <b>minimal predict-ahead (NextLat-h1) keeps the secret in all 3 runs.</b> Full-horizon
          NextLat is seed-variable, JTP behaves like GPT, and MTP only learned the task in one run.
          So monitorability here is a <b>lever the objective moves — not a guarantee.</b>
        </div>
      </div>

      <div className="panel">
        <div className="section-label">What this is (and isn&apos;t)</div>
        <h2>The honest version</h2>
        <p className="muted" style={{ marginTop: 0 }}>
          A tiny synthetic task (K={data.meta.K}) where the model <b>clones a concealing teacher</b> —
          it is <b>not adversarially trying to evade the probe</b> (that is the next experiment). The
          effect is real but <b>fragile</b>: seed- and objective-dependent. The cells show the
          probe&apos;s <b>real per-position belief</b> on single held-out sequences (the secret is a
          running aggregate, so its true value moves as the walk proceeds). So the claim is narrow and
          honest: <em>concealment can defeat internals monitoring even without adversarial pressure,
          and the training objective is one lever over whether it does.</em>
        </p>
      </div>

      <footer>
        Precomputed from frozen checkpoints; no model runs in your browser.<br />
        <span style={{ fontSize: 12 }}>{data.meta.note}</span>
      </footer>
    </div>
  )
}
