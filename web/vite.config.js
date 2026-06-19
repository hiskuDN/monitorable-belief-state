import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base: './' makes the build path-relative so it works on GitHub Pages, Netlify,
// Vercel, or a plain file server without reconfiguration.
export default defineConfig({
  plugins: [react()],
  base: './',
})
