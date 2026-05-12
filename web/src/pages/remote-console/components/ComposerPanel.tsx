import { Button, Input, Typography } from 'antd';
import type { KeyboardEvent } from 'react';

import { SUBMIT_STATUS_LABEL } from '../constants';
import type { SubmitStatus } from '../types';
import styles from './ComposerPanel.module.less';

const { TextArea } = Input;
const { Text } = Typography;

type ComposerPanelProps = {
  value: string;
  status: SubmitStatus;
  disabled: boolean;
  loading: boolean;
  canStopStream: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onStopStream: () => void;
};

function buildHelperCopy(status: SubmitStatus) {
  if (status === 'submitted' || status === 'processing') {
    return '本地终端正在处理中，请稍候。';
  }
  if (status === 'failed') {
    return '发送失败，请调整后重试。';
  }
  return 'Enter 发送，Shift+Enter 换行';
}

export function ComposerPanel({
  value,
  status,
  disabled,
  loading,
  canStopStream,
  onChange,
  onSubmit,
  onStopStream
}: ComposerPanelProps) {
  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      onSubmit();
    }
  };

  return (
    <div className={styles.dock}>
      <div className={styles.centerColumn}>
        <div className={styles.composer}>
          <TextArea
            autoSize={{ minRows: 3, maxRows: 8 }}
            bordered={false}
            className={styles.textarea}
            disabled={disabled}
            placeholder="直接输入你想让本地终端处理的问题..."
            value={value}
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={handleKeyDown}
          />

          <div className={styles.footer}>
            <Text className={styles.helperText}>{buildHelperCopy(status)}</Text>
            <div className={styles.actions}>
              <Text className={styles.statusText}>{SUBMIT_STATUS_LABEL[status]}</Text>
              {canStopStream ? (
                <Button className={styles.stopButton} onClick={onStopStream}>
                  停止接收
                </Button>
              ) : null}
              <Button
                className={styles.sendButton}
                disabled={disabled || !value.trim()}
                loading={loading}
                type="primary"
                onClick={onSubmit}
              >
                发送
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
