import { defineConfig } from "vitest/config";

// Vite entry config (default export required by Vite).
export default defineConfig({
  build: {
    target: "es2022",
    sourcemap: true,
    // three.js alone is ~700 kB minified; split chunks once the real viewer lands (Phase 10).
    chunkSizeWarningLimit: 1000,
  },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
});
