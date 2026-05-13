import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
export default defineConfig({
    plugins: [react()],
    css: {
        modules: {
            localsConvention: 'camelCaseOnly'
        }
    },
    server: {
        host: '0.0.0.0',
        port: 5173,
        proxy: {
            '/web-console': {
                target: 'http://127.0.0.1:8888',
                changeOrigin: true
            }
        }
    }
});
