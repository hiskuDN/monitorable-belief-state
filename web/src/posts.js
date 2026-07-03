import monitorabilityMd from './content/monitorability.md?raw'
import concealmentMd from './content/concealment.md?raw'
import storyworldMd from './content/storyworld.md?raw'

// Blog post registry. Add new posts here (newest first); each `body` is a markdown
// string (imported with ?raw) rendered by BlogPost.jsx.
export const posts = [
  {
    slug: 'reading-a-model-mind-as-it-reads-a-story',
    title: 'Reading a model\'s mind as it reads a story',
    date: '2026-07-03',
    excerpt:
      'The first two posts lived in symbolic worlds. This one moves the same test onto language: a little story that shuffles a ball around some shelves. A predict-ahead model keeps the ball\'s location readable the whole way; a plain model drops it mid-story and only reconstructs it at the question, too late to monitor.',
    body: storyworldMd,
  },
  {
    slug: 'catching-a-model-that-hides-a-secret',
    title: 'Catching a model that hides a secret',
    date: '2026-06-26',
    excerpt:
      'Predict-ahead made belief state monitorable, but only where the model had no reason to hide. Here is the harder case: a secret the model uses but masks. A plain model deletes it from its own internals and reconstructs it at the last second; a predict-ahead model keeps it readable.',
    body: concealmentMd,
  },
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
