/**
 * Error 错误页面
 * 显示错误类型（404等）和导航链接
 */
import { useNavigate, useRouteError } from 'react-router-dom';
import { Button, Result } from 'antd';
import styles from './index.module.less';

const Error = () => {
  const navigate = useNavigate();
  const error = useRouteError() as { status?: number; statusText?: string };

  // 获取错误状态码
  const statusCode = error?.status || 404;
  const statusText = error?.statusText || '页面不存在';

  // 返回首页
  const handleBackHome = () => {
    navigate('/chat');
  };

  // 返回登录页
  const handleBackLogin = () => {
    navigate('/login');
  };

  return (
    <div className={styles.errorPage}>
      <Result
        status={statusCode as any}
        title={statusCode}
        subTitle={statusText}
        extra={
          <div className={styles.errorActions}>
            <Button type="primary" onClick={handleBackHome}>
              返回首页
            </Button>
            <Button onClick={handleBackLogin}>
              返回登录
            </Button>
          </div>
        }
      />
    </div>
  );
};

export default Error;
