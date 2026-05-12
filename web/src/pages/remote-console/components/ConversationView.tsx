import { Button, Skeleton, Typography } from 'antd';
import { useEffect, useRef } from 'react';

import type { ConversationMessage, SubmitStatus } from '../types';
import styles from './ConversationView.module.less';

const { Paragraph, Text, Title } = Typography;

type ConversationViewProps = {
  messages: ConversationMessage[];
  status: SubmitStatus;
  isLoading: boolean;
  suggestions: string[];
  onSuggestionClick: (content: string) => void;
};

function formatTimestamp(value: string) {
  return new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function renderMessageBody(message: ConversationMessage) {
  if (message.role === 'assistant') {
    return (
      <div className={styles.assistantCard}>
        <Text className={styles.assistantLabel}>
          {message.status === 'processing' ? '正在回复' : '助手'}
        </Text>
        <Paragraph className={styles.messageCopy}>{message.content}</Paragraph>
        <Text className={styles.messageTime}>{formatTimestamp(message.createdAt)}</Text>
      </div>
    );
  }

  if (message.role === 'user') {
    return (
      <div className={styles.userCard}>
        <Paragraph className={styles.messageCopy}>{message.content}</Paragraph>
        <Text className={styles.messageTimeMuted}>{formatTimestamp(message.createdAt)}</Text>
      </div>
    );
  }

  if (message.role === 'error') {
    return (
      <div className={styles.errorCard}>
        <Text className={styles.systemEyebrow}>请求失败</Text>
        <Paragraph className={styles.systemCopy}>{message.content}</Paragraph>
      </div>
    );
  }

  return (
    <div className={styles.systemCard}>
      <Text className={styles.systemEyebrow}>
        {message.status === 'processing' ? '处理中' : '系统提示'}
      </Text>
      <Paragraph className={styles.systemCopy}>{message.content}</Paragraph>
    </div>
  );
}

export function ConversationView({
  messages,
  status,
  isLoading,
  suggestions,
  onSuggestionClick
}: ConversationViewProps) {
  const scrollerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const viewport = scrollerRef.current;
    if (!viewport) {
      return;
    }
    viewport.scrollTo({ top: viewport.scrollHeight, behavior: 'smooth' });
  }, [messages, status]);

  if (isLoading) {
    return (
      <section className={styles.shell}>
        <div className={styles.centerColumn}>
          <div className={styles.loadingStack}>
            <Skeleton active paragraph={{ rows: 3 }} />
            <Skeleton active paragraph={{ rows: 4 }} />
          </div>
        </div>
      </section>
    );
  }

  if (messages.length === 0) {
    return (
      <section className={styles.shell}>
        <div className={styles.centerColumn}>
          <div className={styles.emptyState}>
            <Text className={styles.emptyEyebrow}>本地终端对话</Text>
            <Title className={styles.emptyTitle} level={2}>
              你想聊点什么？
            </Title>
            <Paragraph className={styles.emptyBody}>
              可以直接输入问题，或者先试试下面这些示例。
            </Paragraph>
            <div className={styles.suggestionGrid}>
              {suggestions.map((suggestion) => (
                <Button
                  key={suggestion}
                  className={styles.suggestionButton}
                  onClick={() => onSuggestionClick(suggestion)}
                >
                  {suggestion}
                </Button>
              ))}
            </div>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section ref={scrollerRef} className={styles.shell}>
      <div className={styles.centerColumn}>
        {messages.map((message) => (
          <div
            key={message.id}
            className={
              message.role === 'user'
                ? styles.messageRowUser
                : message.role === 'assistant'
                  ? styles.messageRowAssistant
                  : styles.messageRowSystem
            }
          >
            {renderMessageBody(message)}
          </div>
        ))}
      </div>
    </section>
  );
}
