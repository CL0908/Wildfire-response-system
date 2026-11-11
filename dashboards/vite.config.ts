import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  // Relative base so the build works under a GitHub Pages project path as well as at a domain root.
  base: './',
  plugins: [react()],
})
