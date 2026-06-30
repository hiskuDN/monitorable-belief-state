import { useEffect, useState } from 'react'
import StoryMind from '../components/StoryMind.jsx'

export default function Storyworld() {
  const [data, setData] = useState(null)
  const [err, setErr] = useState(null)

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/storyworld.json`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(setData)
      .catch(e => setErr(String(e)))
  }, [])

  if (err) return <div className="wrap"><p className="muted" style={{ paddingTop: 60 }}>Failed to load data: {err}</p></div>
  if (!data) return <div className="wrap"><p className="muted" style={{ paddingTop: 60 }}>Loading…</p></div>

  return (
    <div className="wrap">
      <header className="hero">
        <div className="eyebrow">monitorable belief state · language</div>
        <h1>Can you read a model&apos;s mind <span className="grad">as it reads a story</span>?</h1>
        <p className="lede">
          A short narrative moves a ball around some shelves. Both models can answer &ldquo;where is
          the ball?&rdquo; at the end. But you are a monitor who can only read the model&apos;s hidden
          state mid-story. With one model you can always see where it thinks the ball is; with the
          other you go blind until the moment it answers.
        </p>
      </header>

      <div className="panel">
        <div className="section-label">How to read this</div>
        <h2>You are an interpretability monitor, reading along</h2>
        <div className="how">
          <div className="how-step">
            <span className="n">1</span>
            <p>The ball starts on a shelf, then moves by <b>relative steps</b> (&ldquo;up one&rdquo;,
            &ldquo;down two&rdquo;), with a distractor object and pronouns mixed in. Its current shelf
            is a <b>running total</b>, never a word on the page after the first line.</p>
          </div>
          <div className="how-step">
            <span className="n">2</span>
            <p>At each token we freeze the model and ask a tiny <b>linear probe</b>: <em>&ldquo;which
            shelf does it think the ball is on right now?&rdquo;</em> The cells are the probe&apos;s
            belief; the <b style={{ color: 'var(--true)' }}>gold</b> cell is the truth.</p>
          </div>
          <div className="how-step">
            <span className="n">3</span>
            <p>Both models <b>answer the final question correctly.</b> The difference is <b>when</b>
            the answer is legible inside them: the whole way (monitorable) or only once they
            <b> act</b> (too late for oversight).</p>
          </div>
        </div>
        <div className="takeaway" style={{ marginTop: 18 }}>
          The safety-relevant question is not whether the model knows the answer (it does) but
          whether a monitor can <b>read it early enough to intervene</b>, before the model commits.
        </div>
      </div>

      <div className="panel">
        <div className="section-label">The monitor&apos;s-eye view</div>
        <h2>Watch the belief stay legible, or go dark</h2>
        <p className="sub">
          Press play. The playhead sweeps the story token by token; the two panels are the same
          probe reading <b>GPT</b> vs <b>NextLat (1 step ahead)</b>. Cycle the <b>↻ story</b> button
          for more narratives. &ldquo;Read-accuracy so far&rdquo; is how often the monitor has been
          right up to this point.
        </p>
        <StoryMind data={data} />
        <div className="takeaway">
          GPT&apos;s belief drifts to the wrong shelf (often <b>confidently</b> wrong) as the ball
          moves, and only snaps to the truth <b>at the question</b>. NextLat carries the ball&apos;s
          location the whole way: a monitor reading it is right <b>~95-100%</b> of the time
          throughout, versus <b>~45-60%</b> for GPT. Same task, same correct answer; only the
          <em> training objective</em> differs.
        </div>
      </div>

      <div className="panel">
        <div className="section-label">What this is (and isn&apos;t)</div>
        <h2>The honest version</h2>
        <p className="muted" style={{ marginTop: 0 }}>
          A controlled, small-scale, <b>from-scratch</b> existence proof on a <b>templated</b>
          narrative, not a real pretrained LLM. The belief is engineered as a running total so it
          can&apos;t be read off the surface words; the language contribution is the <b>surface</b>
          (sentences, a distractor, pronouns). The carry advantage is robust at this complexity and
          <b> degrades</b> as the narrative gets harder (more objects, longer), vanishing for every
          model once the task outgrows what a small model can track at all. So the honest claim is
          narrow: <em>on a language surface, a predict-ahead objective keeps a tracked belief
          linearly monitorable mid-task, where a vanilla model defers it to the moment of action.</em>
        </p>
      </div>

      <footer>
        Precomputed from frozen checkpoints; no model runs in your browser.<br />
        <span style={{ fontSize: 12 }}>{data.meta.note}</span>
      </footer>
    </div>
  )
}
