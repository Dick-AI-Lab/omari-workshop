import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base: './' makes the production build portable (any static host, or a sub-path).
export default defineConfig({
  base: './',
  plugins: [react()],
  server: { port: 5173, host: true },
  test: { environment: 'node', include: ['src/**/*.test.ts'] },
} as any);
