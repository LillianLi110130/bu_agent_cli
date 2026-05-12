import React from 'react';
import ReactDOM from 'react-dom/client';
import { ConfigProvider, theme } from 'antd';

import 'antd/dist/reset.css';
import './main.less';
import App from './App';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <ConfigProvider
      theme={{
        algorithm: theme.defaultAlgorithm,
        token: {
          colorPrimary: '#0f766e',
          colorInfo: '#0f766e',
          borderRadius: 18,
          colorBgLayout: '#f5f2ea',
          colorBgContainer: '#ffffff',
          colorText: '#171717',
          colorTextSecondary: '#5f6368',
          fontFamily:
            '"Aptos", "Segoe UI", "PingFang SC", "Microsoft YaHei", -apple-system, BlinkMacSystemFont, sans-serif'
        }
      }}
    >
      <App />
    </ConfigProvider>
  </React.StrictMode>
);
