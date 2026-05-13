import { Alert, Badge, Button, Typography } from 'antd';

import { ComposerPanel } from './components/ComposerPanel';
import { ConversationView } from './components/ConversationView';
import { useRemoteConsole } from './hooks/useRemoteConsole';
import styles from './index.module.less';

const { Text } = Typography;

function resolveOnlineStatus(isOnline?: boolean) {
  if (isOnline === true) {
    return {
      badgeStatus: 'success' as const,
      label: '本地终端在线'
    };
  }

  if (isOnline === false) {
    return {
      badgeStatus: 'error' as const,
      label: '本地终端离线'
    };
  }

  return {
    badgeStatus: 'default' as const,
    label: '正在检测连接状态'
  };
}

export function RemoteConsolePage() {
  const controller = useRemoteConsole();
  const onlineStatus = resolveOnlineStatus(controller.workerSummary.isOnline);

  return (
    <div className={styles.page}>
      <div className={styles.glowPrimary} />
      <div className={styles.glowSecondary} />

      <main className={styles.chatStage}>
        <div className={styles.statusSlot}>
          <div className={styles.statusActions}>
            <div className={styles.connectionBadge}>
              <Badge status={onlineStatus.badgeStatus} />
              <Text className={styles.connectionText}>{onlineStatus.label}</Text>
            </div>
            {controller.canLaunchLocalCrab ? (
              <Button className={styles.launchButton} onClick={controller.launchLocalCrab}>
                启动本地 Crab
              </Button>
            ) : null}
          </div>
        </div>

        {controller.viewState.lastError ? (
          <Alert
            banner
            className={styles.alertBanner}
            closable
            message={controller.viewState.lastError}
            type="error"
          />
        ) : null}

        <ConversationView
          isLoading={false}
          messages={controller.messages}
          onSuggestionClick={controller.submitCurrentDraft}
          status={controller.viewState.submitStatus}
          suggestions={controller.emptyStateSuggestions}
        />

        <ComposerPanel
          canStopStream={controller.canStopStream}
          disabled={controller.isComposerDisabled}
          loading={controller.isSubmitting}
          onChange={controller.setDraft}
          onStopStream={controller.stopCurrentStream}
          onSubmit={() => controller.submitCurrentDraft()}
          status={controller.viewState.submitStatus}
          value={controller.draft}
        />
      </main>
    </div>
  );
}
