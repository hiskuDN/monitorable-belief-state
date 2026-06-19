import { posts } from '../posts.js'

export default function Blogs() {
  return (
    <div className="wrap">
      <header className="hero" style={{ paddingBottom: 18 }}>
        <div className="eyebrow">writing</div>
        <h1>Blog</h1>
        <p className="lede">Notes on monitorable belief state and predictive training objectives.</p>
      </header>
      <div className="post-list">
        {posts.map((p) => (
          <a key={p.slug} className="post-card" href={`#/blogs/${p.slug}`}>
            <div className="post-date">{p.date}</div>
            <h2>{p.title}</h2>
            <p className="muted" style={{ margin: '0 0 12px' }}>{p.excerpt}</p>
            <span className="post-readmore">Read →</span>
          </a>
        ))}
      </div>
    </div>
  )
}
