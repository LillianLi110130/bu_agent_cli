# Wheel 打包与安装说明

当前项目的推荐分发方式已经收敛为 `wheel`。

也就是说，发布侧只需要构建 `.whl` 包，使用侧只需要：

1. 安装 Python
2. `pip install xxx.whl`
3. 直接运行 `tg-agent`

CLI 会在首次启动时自动初始化用户级默认配置，因此不再要求用户手动设置命令行环境变量。

## 打包方式

在仓库根目录执行：

```bash
python -m build --wheel
```

执行完成后，产物会生成在 `dist/` 目录下，例如：

```text
dist/
  tg_agent_cli-0.1.0-py3-none-any.whl
```

如果当前环境还没有安装 `build`，先执行：

```bash
python -m pip install build
```

## 安装方式

用户拿到 `.whl` 后，直接安装：

```bash
pip install tg_agent_cli-0.1.0-py3-none-any.whl
```

安装完成后直接运行：

```bash
tg-agent
```

或先查看帮助：

```bash
tg-agent --help
```

## 首次启动时会自动做什么

`tg-agent` 第一次启动时，会自动在用户目录下创建：

- `~/.tg_agent/.env`
- `~/.tg_agent/tg_crab_worker.json`

其中默认生成的 `~/.tg_agent/.env` 内容如下：

```dotenv
# tg-agent runtime configuration
# Generated automatically on first CLI launch.
# Edit this file if you want to override the default model or API endpoint.
OPENAI_API_KEY=
LLM_MODEL=GLM-4.7
LLM_BASE_URL=https://open.bigmodel.cn/api/coding/paas/v4
```

这意味着：

- 用户不需要手动 `export` 或 `set` 环境变量
- 用户不需要自己创建 `.env`
- 用户安装完 `whl` 后可以直接执行 `tg-agent`

## 配置优先级

CLI 启动时会按以下顺序读取配置：

1. 当前进程已经存在的环境变量
2. 当前工作目录下的 `.env`
3. `~/.tg_agent/.env`

所以：

- 如果用户已经在 shell 中设置了变量，仍然会优先使用 shell 中的值
- 如果没有设置，CLI 会自动回退到首次生成的 `~/.tg_agent/.env`

## Worker 配置文件

`tg_crab_worker.json` 的查找顺序如下：

1. 当前工作目录下的 `tg_crab_worker.json`
2. `~/.tg_agent/tg_crab_worker.json`
3. 包内默认文件

首次启动时，如果 `~/.tg_agent/tg_crab_worker.json` 不存在，会自动从包内默认配置复制过去。

## 需要注意的边界

### 1. 用户机器仍然需要 Python

`wheel` 方案不再依赖 `exe`，但前提仍然是用户机器上已经有可用的 Python 和 `pip`。

### 2. 真实密钥可能仍然需要补充

虽然现在不需要用户手动设置命令行环境变量，但如果实际运行依赖真实的 `OPENAI_API_KEY`，用户后续仍然需要把它写入：

```text
~/.tg_agent/.env
```

### 3. Server 不受这套自动初始化影响

自动创建 `.env` 和用户级配置的逻辑只在 CLI 入口生效。

Server 端仍然应该由你自己的启动逻辑显式加载对应环境配置文件，例如 `.env.dev`、`.env.uat`、`.env.prod`。
