/**
 * App 根组件
 * 使用 ConfigProvider 配置 Ant Design 主题
 * 使用 RouterProvider 集成路由
 */
import { ConfigProvider, theme } from 'antd';
import { RouterProvider } from 'react-router-dom';
import router from './router';

const customTheme = {
  token: {
    colorPrimary: '#1677ff',
    borderRadius: 8,
    fontSize: 14,
  },
  algorithm: theme.defaultAlgorithm,
};

function App() {
  return (
    <ConfigProvider theme={customTheme}>
      <RouterProvider router={router} />
    </ConfigProvider>
  );
}

export default App;
