/**
 * Chat 聊天页面
 * 主对话页面，包含侧边栏和主内容区
 */
import { Layout } from "antd";
import styles from "./index.module.less";

const { Sider, Content } = Layout;

const Chat = () => {
  return (
    <Layout className={styles.chatPage}>
      {/* 侧边栏 - 会话列表 */}
      <Sider width={260} className={styles.chatSidebar} theme="light">
        <div className={styles.sidebarHeader}>
          <h2>对话</h2>
        </div>
        <div className={styles.sidebarContent}>
          {/* 会话列表组件将在后续阶段实现 */}
          <p className={styles.placeholderText}>会话列表（占位）</p>
        </div>
      </Sider>

      {/* 主内容区 - 消息列表和输入框 */}
      <Layout className={styles.chatMain}>
        <Content className={styles.chatContent}>
          {/* 顶部标题栏 */}
          <div className={styles.chatHeader}>
            <h1>AI 助手</h1>
          </div>

          {/* 消息列表区域 */}
          <div className={styles.chatMessages}>
            {/* 消息列表组件将在后续阶段实现 */}
            <p className={styles.placeholderText}>消息列表区域（占位）</p>
          </div>

          {/* 输入区域 */}
          <div className={styles.chatInput}>
            {/* 输入框组件将在后续阶段实现 */}
            <p className={styles.placeholderText}>输入框区域（占位）</p>
          </div>
        </Content>
      </Layout>
    </Layout>
  );
};

export default Chat;
