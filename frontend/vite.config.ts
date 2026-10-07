/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  test: {
    // Unit tests only; browser tests in e2e/ run with Playwright (npm run test:e2e).
    include: ['src/**/*.test.ts'],
  },
})
