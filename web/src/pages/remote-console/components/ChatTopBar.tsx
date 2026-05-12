import { Button, Tag, Typography } from 'antd';

import { SUBMIT_STATUS_LABEL } from '../constants';
import type { SubmitStatus } from '../types';
import styles from './ChatTopBar.module.less';

const { Paragraph, Text, Title } = Typography;

type ChatTopBarProps = {
  sessionTitle: string;
  workerId: string;
  status: SubmitStatus;
  isOnline?: boolean;
  onCreateSession: () => void;
};

function resolveStatusColor(status: SubmitStatus) {
  if (status === 'failed') {
    return 'error';
  }
  if (status === 'processing' || status === 'submitted') {
    return 'warning';
  }
  if (status === 'completed') {
    return 'success';
  }
  return 'default';
}

function resolveOnlineLabel(isOnline?: boolean) {
  if (isOnline === true) {
    return '在线';
  }
  if (isOnline === false) {
    return '离线';
  }
  return '未连接';
}

export function ChatTopBar({
  sessionTitle,
  workerId,
  status,
  isOnline,
  onCreateSession
}: ChatTopBarProps) {
  return (
    <header className={styles.topBar}>
      <div className={styles.brandLine}>
        <Text className={styles.eyebrow}>BU Agent</Text>
        <Tag
          className={styles.workerTag}
          color={isOnline === true ? 'success' : isOnline === false ? 'error' : 'default'}
        >
          {workerId} · {resolveOnlineLabel(isOnline)}
        </Tag>
      </div>

      <div className={styles.mainRow}>
        <div className={styles.titleBlock}>
          <Title className={styles.title} level={2}>
            {sessionTitle}
          </Title>
          <Paragraph className={styles.subtitle}>
            {status === 'idle' ? '开始新的对话' : `当前状态：${SUBMIT_STATUS_LABEL[status]}`}
          </Paragraph>
        </div>

        <div className={styles.actions}>
          <Tag className={styles.statusTag} color={resolveStatusColor(status)}>
            {SUBMIT_STATUS_LABEL[status]}
          </Tag>
          <Button className={styles.newChatButton} onClick={onCreateSession}>
            新对话
          </Button>
        </div>
      </div>
    </header>
  );
}
