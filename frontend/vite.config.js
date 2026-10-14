import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiProxyTarget = process.env.API_PROXY_TARGET || 'http://127.0.0.1:8000'
export const apiProxyPaths = [
  '/health',
  '/symptoms',
  '/predict',
  '/explain',
  '/evidence',
  '/next-question',
]

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(apiProxyPaths.map((path) => [path, apiProxyTarget])),
  },
})
