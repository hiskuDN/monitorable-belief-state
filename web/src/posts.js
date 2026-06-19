import monitorabilityMd from './content/monitorability.md?raw'

// Blog post registry. Add new posts here; each `body` is a markdown string
// (imported with ?raw) rendered by BlogPost.jsx.
export const posts = [
  {
    slug: 'monitorability-as-a-training-objective',
    title: 'Monitorability as a training objective',
    date: '2026-06-19',
    excerpt:
      'Can you train a model so its internal belief state is easier to read? A small synthetic result, an honest negative, and where it could go for agentic monitoring.',
    body: monitorabilityMd,
  },
]

export const getPost = (slug) => posts.find((p) => p.slug === slug)
