import { Badge, Button, Input, Typography } from 'antd';

import type { WebSessionItem } from '../types';
import styles from './ChatSidebar.module.less';

const { Paragraph, Text, Title } = Typography;

type ChatSidebarProps = {
  sessions: WebSessionItem[];
  activeSessionId: string;
  workerId: string;
  isOnline?: boolean;
  searchValue: string;
  onSearchChange: (value: string) => void;
  onSelectSession: (sessionId: string) => void;
  onCreateSession: () => void;
};

function formatSidebarTime(value: string) {
  const date = new Date(value);
  const now = new Date();
  const sameDay = date.toDateString() === now.toDateString();
  return sameDay
    ? date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

function resolveBadgeColor(isOnline?: boolean) {
  if (isOnline === true) {
    return '#12b76a';
  }
  if (isOnline === false) {
    return '#b42318';
  }
  return '#98a2b3';
}

function resolveWorkerLabel(isOnline?: boolean) {
  if (isOnline === true) {
    return '在线';
  }
  if (isOnline === false) {
    return '离线';
  }
  return '未连接';
}

export function ChatSidebar({
  sessions,
  activeSessionId,
  workerId,
  isOnline,
  searchValue,
  onSearchChange,
  onSelectSession,
  onCreateSession
}: ChatSidebarProps) {
  return (
    <aside className={styles.sidebar}>
      <div className={styles.brandBlock}>
        <div>
          <Text className={styles.eyebrow}>BU Agent</Text>
          <Title className={styles.brandTitle} level={4}>
            Web 控制台
          </Title>
        </div>
        <Badge
          className={styles.workerBadge}
          color={resolveBadgeColor(isOnline)}
          text={<span className={styles.workerText}>{workerId} · {resolveWorkerLabel(isOnline)}</span>}
        />
      </div>

      <Button block className={styles.newChatButton} size="large" type="primary" onClick={onCreateSession}>
        新建会话
      </Button>

      <Input
        allowClear
        className={styles.searchInput}
        placeholder="搜索左侧会话"
        value={searchValue}
        onChange={(event) => onSearchChange(event.target.value)}
      />

      <div className={styles.sessionList}>
        {sessions.map((session) => {
          const active = session.id === activeSessionId;
          return (
            <button
              key={session.id}
              className={active ? styles.sessionCardActive : styles.sessionCard}
              type="button"
              onClick={() => onSelectSession(session.id)}
            >
              <div className={styles.sessionHeader}>
                <Text className={styles.sessionTitle}>{session.title}</Text>
                <Text className={styles.sessionTime}>{formatSidebarTime(session.updatedAt)}</Text>
              </div>
              <Paragraph className={styles.sessionSnippet} ellipsis={{ rows: 2 }}>
                {session.lastMessage || '这是一个浏览器侧的新会话容器，发送第一条消息后会同步到服务端。'}
              </Paragraph>
            </button>
          );
        })}
      </div>

      <div className={styles.footnote}>
        左侧会话用于整理 Web 页面展示，本地 agent 上下文仍然取决于当前连接的 worker。
      </div>
    </aside>
  );
}
