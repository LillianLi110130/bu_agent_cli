/**
 * 路由配置
 * 使用 react-router-dom v7 实现单页应用路由
 */
import { createBrowserRouter, Navigate } from 'react-router-dom';
import Login from './pages/Login';
import Chat from './pages/Chat';
import Error from './pages/Error';

// 路由配置
const router = createBrowserRouter([
  {
    path: '/',
    element: <Navigate to="/chat" replace />,
  },
  {
    path: '/login',
    element: <Login />,
  },
  {
    path: '/chat',
    element: <Chat />,
  },
  {
    path: '*',
    element: <Error />,
  },
]);

export default router;
