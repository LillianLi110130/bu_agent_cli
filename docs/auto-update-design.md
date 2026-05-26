# Crab CLI 自动更新详细设计

本文档定义 Crab CLI 第一版自动更新实现。更新基于对象存储完成：客户端启动前读取远端 `manifest`，发现新版本后下载 portable 安装包，校验后调用新包里的部署脚本完成升级。运行中的 Crab 只提供检查能力，不直接安装更新。

## 目标

- 启动 `crab` 前检查新版本。
- 对象存储保存更新清单和安装包。
- 支持 Linux x64 和 Windows x64 portable 包。
- 更新前让用户确认，不做静默安装。
- 下载后校验 `sha256`。
- 更新失败时保留旧版本可启动。
- 运行中只支持 `/update check` 和 `/update status`，不支持 `/update install`。

## 职责边界

```text
scripts/release/*        构建安装包、生成 sha256、根据 release_notes.json 生成 manifest
对象存储                 保存 manifest 和 portable 安装包
launcher / crab shim     启动 crab 前调用 updater
agent_core.updater       启动前检查、下载、校验、解压、调用 deploy；运行中只检查状态
deploy.sh / deploy.bat   安装或更新已经解压好的 bundle
```

更新检查逻辑不放进 `scripts/release/*`。这些脚本只负责发布产物。运行时检查由 launcher 和 Python updater 完成。

## 对象存储结构

对象存储固定使用以下结构：

```text
crab-cli/
  channels/
    stable.json
  releases/
    0.8.0/
      manifest.json
      tg-agent-linux-x64-v0.8.0-portable.tar.gz
      tg-agent-linux-x64-v0.8.0-portable.tar.gz.sha256
      tg-agent-windows-x64-v0.8.0-portable.zip
      tg-agent-windows-x64-v0.8.0-portable.zip.sha256
```

`channels/stable.json` 是客户端启动前检查更新时读取的通道 manifest。`releases/<version>/manifest.json` 是版本 manifest，保存该版本完整发布信息。`releases/<version>/*` 是不可变发布产物，发布后不覆盖。如果某个版本有问题，发布新版本并更新 `stable.json`。

## Manifest 格式

### 通道 Manifest

`channels/stable.json` 使用以下格式：

```json
{
  "schema": 1,
  "channel": "stable",
  "latest": "0.8.0",
  "published_at": "2026-05-26",
  "releases": {
    "linux-x64": {
      "url": "https://oss.example.com/crab-cli/releases/0.8.0/tg-agent-linux-x64-v0.8.0-portable.tar.gz",
      "sha256": "replace-with-linux-sha256",
      "size": 123456789
    },
    "windows-x64": {
      "url": "https://oss.example.com/crab-cli/releases/0.8.0/tg-agent-windows-x64-v0.8.0-portable.zip",
      "sha256": "replace-with-windows-sha256",
      "size": 123456789
    }
  },
  "notes": [
    "新增启动前自动更新检查",
    "优化 worker 连接稳定性"
  ]
}
```

`stable.json` 保留完整 `releases` 信息。这样启动前检查只需要请求一次 `stable.json`，即可完成版本比较、展示 notes、下载对应平台安装包。

### 版本 Manifest

`releases/<version>/manifest.json` 使用以下格式：

```json
{
  "schema": 1,
  "version": "0.8.0",
  "published_at": "2026-05-26",
  "releases": {
    "linux-x64": {
      "file": "tg-agent-linux-x64-v0.8.0-portable.tar.gz",
      "url": "https://oss.example.com/crab-cli/releases/0.8.0/tg-agent-linux-x64-v0.8.0-portable.tar.gz",
      "sha256": "replace-with-linux-sha256",
      "size": 123456789
    },
    "windows-x64": {
      "file": "tg-agent-windows-x64-v0.8.0-portable.zip",
      "url": "https://oss.example.com/crab-cli/releases/0.8.0/tg-agent-windows-x64-v0.8.0-portable.zip",
      "sha256": "replace-with-windows-sha256",
      "size": 123456789
    }
  },
  "notes": [
    "新增启动前自动更新检查",
    "优化 worker 连接稳定性"
  ]
}
```

`stable.json` 的 `latest`、`published_at`、`releases` 和 `notes` 必须来自当前稳定版本的 `releases/<version>/manifest.json`。

字段含义：

| 字段 | 含义 |
| --- | --- |
| `schema` | manifest 格式版本 |
| `channel` | 更新通道，第一版固定为 `stable` |
| `latest` | 当前稳定版本 |
| `published_at` | 发布日期，格式为 `YYYY-MM-DD` |
| `releases` | 各平台安装包信息 |
| `url` | 安装包下载地址 |
| `sha256` | 安装包校验值 |
| `size` | 安装包字节大小 |
| `notes` | 远端新版本更新说明 |

`notes` 不写死在客户端代码里。人工维护的单一源头是 `config/release_notes.json`。欢迎页直接读取该文件，展示当前已安装版本的更新内容。打包发布时，脚本读取该文件，并把其中的 `notes` 写入 `stable.json`。自动更新提示只从远端 `stable.json` 读取新版本的 `notes` 并展示。

`config/release_notes.json` 只保存当前版本信息：

```json
{
  "version": "0.8.0",
  "published_at": "2026-05-26",
  "notes": [
    "新增启动前自动更新检查",
    "优化 worker 连接稳定性"
  ]
}
```

欢迎页读取该文件时必须校验 `version` 等于当前 `get_cli_version()`。版本不一致时只显示版本号，不显示 notes。

生成 `stable.json` 时，脚本必须校验 `config/release_notes.json` 的 `version` 等于 `pyproject.toml` 的版本号，然后把 `notes` 原样写入 manifest：

```json
"notes": [
  "新增启动前自动更新检查",
  "优化 worker 连接稳定性"
]
```

平台 key 第一版固定为：

```text
linux-x64
windows-x64
```

## 本地文件

自动更新文件固定放在：

```text
~/.tg_agent/update_state.json
~/.tg_agent/updates/downloads/
~/.tg_agent/updates/staging/
~/.tg_agent/updates/logs/
~/.tg_agent/updates/backups/
```

目录用途：

| 目录 | 用途 |
| --- | --- |
| `downloads/` | 保存下载到本地的 portable 压缩包 |
| `staging/` | 保存解压后的待安装新版本，用于执行 `deploy.sh` / `deploy.bat` |
| `logs/` | 保存更新和部署过程日志，用于失败排查 |
| `backups/` | 保存更新前的旧版本备份，用于失败回滚 |

`update_state.json` 记录检查和安装状态：

```json
{
  "last_check_at": "2026-05-26T10:30:00+08:00",
  "last_seen_version": "0.8.0",
  "auto_check_enabled": true,
  "last_install": {
    "version": "0.8.0",
    "status": "success",
    "finished_at": "2026-05-26T10:35:00+08:00"
  }
}
```

## 启动前检查

启动前检查接入两个入口：

1. portable bundle 的 `crab-launcher.sh`、`crab-launcher.ps1`
2. 安装后生成的 `~/.tg_agent/bin/crab`

启动流程固定为：

```text
用户启动 crab
  -> launcher/shim 调用 updater
  -> updater 检查 stable.json
  -> 没有更新则继续启动 crab
  -> 有更新则询问用户
  -> 用户确认后下载、校验、解压
  -> updater 生成 pending 部署脚本并退出
  -> launcher 执行 pending 部署脚本
  -> 安装完成后启动新版本 crab
```

启动前检查每次启动都会执行一次。检查失败不阻断启动。用户可以用环境变量跳过：

```bash
CRAB_SKIP_UPDATE_CHECK=1 crab
```

launcher 只负责调用 updater 和执行 updater 生成的 pending 部署脚本，不写下载和安装细节。`check-before-launch` 负责检查、下载、校验、解压，并生成 pending 部署脚本；真正安装由 launcher 在 Crab 主进程启动前执行 pending 部署脚本完成。

Linux launcher 调用方式：

```bash
if [[ "${CRAB_SKIP_UPDATE_CHECK:-}" != "1" ]]; then
  set +e
  "${venv_python}" -m agent_core.updater check-before-launch
  update_status=$?
  set -e
  if [[ "${update_status}" -eq 20 ]]; then
    "${install_root}/updates/pending_update.sh"
  elif [[ "${update_status}" -ne 0 ]]; then
    exit "${update_status}"
  fi
fi

exec "${venv_python}" -u "${entry_shim}" "$@"
```

Windows launcher 调用方式：

```powershell
if ($env:CRAB_SKIP_UPDATE_CHECK -ne "1") {
    & $venvPython -m agent_core.updater check-before-launch
    $updateExitCode = $LASTEXITCODE
    if ($updateExitCode -eq 20) {
        $pendingUpdate = Join-Path $installRoot "updates\pending_update.ps1"
        powershell -NoProfile -ExecutionPolicy Bypass -File $pendingUpdate
        $pendingExitCode = $LASTEXITCODE
        if ($pendingExitCode -ne 0) {
            exit $pendingExitCode
        }
    }
    elseif ($updateExitCode -ne 0) {
        exit $updateExitCode
    }
}

& $venvPython -u $entryShim @CliArgs
```

## Updater 模块

新增模块：

```text
agent_core/updater.py
cli/update_handler.py
```

`agent_core/updater.py` 提供命令行入口：

```bash
python -m agent_core.updater check-before-launch
python -m agent_core.updater check
python -m agent_core.updater status
```

`cli/update_handler.py` 提供 slash 命令：

```text
/update check
/update status
```

第一版不提供 `/update install`。运行中的 Crab 不直接下载和安装新版本，只提示用户关闭所有 Crab 后重新启动，通过启动前更新流程完成安装。

`check-before-launch` 行为：

```text
1. 读取当前版本
2. 下载 stable.json
3. 识别当前平台
4. 比较 latest 和当前版本
5. 无更新则退出
6. 有更新则显示版本和 notes
7. 用户确认后下载、校验、解压
8. 生成 `~/.tg_agent/updates/pending_update.sh` 或 `pending_update.ps1`
9. updater 退出后，launcher 执行 pending 部署脚本
10. 更新完成后启动新版本 crab
```

有更新时固定使用以下交互：

```text
发现 Crab CLI 新版本

当前版本: 0.7.0
最新版本: 0.8.0
发布时间: 2026-05-26

更新内容:
- 新增启动前自动更新检查
- 优化 worker 连接稳定性

请选择:
1. 立即更新
2. 跳过本次，继续启动

请输入选项 [1/2]:
```

选项行为：

| 选项 | 行为 |
| --- | --- |
| `1` | 执行 `install`，更新完成后继续启动 `crab` |
| `2` | 本次不更新，继续启动 `crab` |
| 直接回车 | 等同于选项 `2` |
| 非法输入 | 重新提示一次；第二次仍非法则按选项 `2` 处理 |

如果当前不是交互式终端，只打印一行提示并继续启动：

```text
发现 Crab CLI 新版本 0.8.0，当前版本 0.7.0。请关闭所有 Crab 后重新启动以更新。
```

`/update check` 行为：

```text
1. 下载 stable.json
2. 比较 latest 和当前版本
3. 无更新则提示当前已是最新版本
4. 有更新则展示版本、发布时间和 notes
5. 提示用户关闭所有 Crab 后重新启动以更新
6. 写入 update_state.json
```

`/update status` 行为：

```text
1. 读取当前版本
2. 读取 update_state.json
3. 展示当前版本
4. 展示上次检查时间
5. 展示上一次成功更新时间
```

## Deploy 更新模式

`deploy.sh`、`deploy.bat` 和 `win_deploy.ps1` 增加更新模式参数：

```text
--update
```

更新模式执行：

```text
1. 检查 bundle 结构
2. 备份当前 ~/.tg_agent/.venv 到 ~/.tg_agent/updates/backups/
3. 使用新 bundle 的 python-runtime 和 wheelhouse 安装新版本
4. 刷新 ~/.tg_agent/bin/crab 和 crab-entry.py
5. 执行 crab --help 验证
6. 成功则保留新版本
7. 失败则恢复旧版本
```

`--update` 模式固定不修改 shell profile，并且固定在更新前备份当前安装。首次安装仍由普通 `deploy.sh` / `deploy.bat` 负责。

## 并发锁

下载和安装必须持有锁：

```text
~/.tg_agent/update.lock
```

launcher 启动时如果发现 `update.lock` 已存在，说明另一个更新流程正在运行。此时不检查 manifest，也不启动 Crab，直接提示并退出：

```text
Crab 正在更新中，请稍后重新启动。
```

检查 manifest 不需要持有锁。下载和安装必须先获取 `update.lock`。

## 发布流程

发布新版本按以下顺序执行：

```text
1. 更新 pyproject.toml 版本号
2. 更新 config/release_notes.json
3. 构建 Linux portable bundle
4. 构建 Windows portable bundle
5. 执行 verify 脚本
6. 计算两个安装包的 sha256
7. 读取 config/release_notes.json 并生成 releases/<version>/manifest.json
8. 根据 releases/<version>/manifest.json 生成 channels/stable.json
9. 手动上传安装包、.sha256 和 manifest.json 到对象存储 releases/<version>/
10. 手动上传 stable.json 到对象存储 channels/stable.json
```

打包脚本需要补充两个能力：

```text
1. 生成 .sha256
2. 校验 config/release_notes.json 和 pyproject.toml 版本一致
3. 从 config/release_notes.json 生成 releases/<version>/manifest.json
4. 从 releases/<version>/manifest.json 生成 stable.json
```

对象存储上传第一版手动完成，`scripts/release/*` 不实现上传功能。

## 验收标准

- `stable.json` 中版本高于当前版本时，启动前能提示更新。
- 用户跳过时，Crab 正常启动。
- 用户在启动前确认更新时，安装包会下载到 `~/.tg_agent/updates/downloads/`。
- `sha256` 不一致时拒绝安装。
- 更新成功后 `crab --help` 可用。
- 更新失败后旧版本仍可启动。
- `CRAB_SKIP_UPDATE_CHECK=1` 时不检查更新。
- 运行中 `/update check` 发现新版本时，只提示关闭所有 Crab 后重新启动，不执行安装。
