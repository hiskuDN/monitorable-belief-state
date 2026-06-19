import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { getPost } from '../posts.js'

const resolveSrc = (src) => {
  if (!src) return src
  if (src.startsWith('http') || src.startsWith('#') || src.startsWith('data:')) return src
  return `${import.meta.env.BASE_URL}${src.replace(/^\//, '')}`
}

export default function BlogPost({ slug }) {
  const post = getPost(slug)
  if (!post) {
    return (
      <div className="wrap">
        <p className="muted" style={{ paddingTop: 60 }}>
          Post not found. <a href="#/blogs">Back to the blog</a>.
        </p>
      </div>
    )
  }
  return (
    <div className="wrap">
      <a className="back-link" href="#/blogs">← Blog</a>
      <article className="prose">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            img: ({ src, alt }) => (
              <figure className="post-figure">
                <img src={resolveSrc(src)} alt={alt} />
                {alt && <figcaption>{alt}</figcaption>}
              </figure>
            ),
            a: ({ href, children }) => {
              const external = href && href.startsWith('http')
              return (
                <a href={href} {...(external ? { target: '_blank', rel: 'noopener noreferrer' } : {})}>
                  {children}
                </a>
              )
            },
          }}
        >
          {post.body}
        </ReactMarkdown>
      </article>
    </div>
  )
}
