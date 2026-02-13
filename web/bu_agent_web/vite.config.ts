import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  css: {
    preprocessorOptions: {
      less: {
        // 支持 Less 全局变量和混入
        javascriptEnabled: true,
        // 可以在这里修改主题变量
        modifyVars: {
          // '@primary-color': '#1677ff',
        },
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    // API 代理配置 - 开发环境通过 Vite 代理转发到后端
    // 生产环境通过 nginx 转发
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
