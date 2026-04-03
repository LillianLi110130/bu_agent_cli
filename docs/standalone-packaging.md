# 独立安装包打包与使用说明

本文档说明如何把本项目打包成不依赖用户本机 Python 的独立可执行程序，并分别介绍 Windows 和 Linux 下的打包、安装、运行、配置和常见问题。

## 目标

当前项目已经支持通过 PyInstaller 打包为独立可执行文件：

- Windows 产物：`tg-agent.exe`
- Linux 产物：`tg-agent`

最终用户可以：

- 直接运行可执行文件
- 或执行安装脚本，把程序安装到用户目录后再通过命令使用

安装脚本现在是零交互模式：

- 安装时不会停下来询问 `OPENAI_API_KEY`、`LLM_MODEL`、`LLM_BASE_URL`
- 如果用户目录下不存在 `~/.tg_agent/.env`，脚本会自动创建一个默认模板
- 用户后续按需修改该文件即可

## 相关文件

与独立打包相关的主要文件如下：

- `scripts/release.ps1`
  - Windows 本地一键发布脚本
- `scripts/build_standalone.ps1`
  - Windows 独立打包脚本
- `scripts/build_standalone.sh`
  - Linux 独立打包脚本
- `scripts/install-tg-agent.ps1`
  - Windows 安装脚本
- `scripts/install-tg-agent.sh`
  - Linux 安装脚本
- `packaging/tg-agent.spec`
  - PyInstaller 打包配置

## 打包前准备

### 1. 在目标系统上打包

Windows 包请在 Windows 上打，Linux 包请在 Linux 上打。

不要依赖跨平台打包。也就是说：

- 不要在 Windows 上生成 Linux 可执行文件
- 不要在 Linux 上生成 Windows 可执行文件

### 2. 确保打包解释器中已经安装 PyInstaller

例如：

```powershell
python -m pip install pyinstaller
```

Linux：

```bash
python3 -m pip install pyinstaller
```

### 3. 关于默认 Python 选择

Windows 下的发布脚本和打包脚本现在会自动寻找可用的 Python，优先顺序如下：

1. 显式传入的 `-Python`
2. 当前激活的 `venv`
3. 当前激活的 `conda`
4. 环境变量 `PYTHON`
5. PATH 中的 `python`
6. Conda 环境列表中可执行 `PyInstaller` 的解释器

因此大多数情况下可以直接执行脚本，不需要手动传 `-Python`。

## Windows 打包

### 方式一：本地一键发布

这是最推荐的方式，会自动完成：

- 构建独立可执行文件
- 整理发布目录
- 生成 zip 包

执行命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release.ps1
```

如果需要覆盖版本号：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release.ps1 -Version 0.1.1
```

如果已经构建过 exe，只想重新整理发布目录和 zip：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release.ps1 -SkipBuild
```

打包完成后会生成：

```text
dist/release/
  tg-agent-windows-x64-v0.1.0/
  tg-agent-windows-x64-v0.1.0.zip
```

zip 包内通常包含：

- `tg-agent.exe`
- `install-tg-agent.ps1`
- `tg_crab_worker.json`
- `README.txt`

### 方式二：仅构建独立运行目录

如果你只想先产出可执行文件和安装脚本，不需要 zip 包：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_standalone.ps1
```

完成后目录为：

```text
dist/standalone/windows/package/
  tg-agent.exe
  install-tg-agent.ps1
  tg_crab_worker.json
```

## Linux 打包

Linux 目前提供的是独立打包脚本：

```bash
bash ./scripts/build_standalone.sh
```

完成后目录为：

```text
dist/standalone/linux/package/
  tg-agent
  install-tg-agent.sh
  tg_crab_worker.json
```

如果后续需要，也可以再补 Linux 版 `release.sh`，用于直接输出带版本号的压缩包。

## Windows 使用方式

### 方式一：直接运行，不安装

进入打包产物目录后，直接运行：

```powershell
.\tg-agent.exe
```

查看帮助：

```powershell
.\tg-agent.exe --help
```

这种方式不需要安装，适合临时试用或快速验证。

### 方式二：执行安装脚本

如果希望用户安装后直接通过命令运行，使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\install-tg-agent.ps1
```

安装脚本会自动做这些事：

- 将 `tg-agent.exe` 复制到 `~/.tg_agent/bin/`
- 如果不存在 `~/.tg_agent/.env`，自动创建默认模板
- 如果不存在 `~/.tg_agent/tg_crab_worker.json`，自动复制默认配置
- 将 `~/.tg_agent/bin` 添加到用户 PATH

安装过程不会停下来要求用户输入环境变量。

安装完成后，重新打开终端，再执行：

```powershell
tg-agent
```

## Linux 使用方式

### 方式一：直接运行，不安装

```bash
./tg-agent
```

查看帮助：

```bash
./tg-agent --help
```

### 方式二：执行安装脚本

```bash
bash ./install-tg-agent.sh
```

安装脚本会自动做这些事：

- 将 `tg-agent` 复制到 `~/.tg_agent/bin/`
- 如果不存在 `~/.tg_agent/.env`，自动创建默认模板
- 如果不存在 `~/.tg_agent/tg_crab_worker.json`，自动复制默认配置
- 将 `~/.tg_agent/bin` 追加到 `~/.profile`

安装过程不会停下来要求用户输入环境变量。

安装完成后，重新打开 shell，或者手工执行：

```bash
export PATH="$HOME/.tg_agent/bin:$PATH"
```

然后运行：

```bash
tg-agent
```

## 默认生成的 `.env` 模板

安装脚本首次创建的 `~/.tg_agent/.env` 内容如下：

```dotenv
# tg-agent runtime configuration
# Fill in OPENAI_API_KEY after install if your shell or workspace does not already provide it.
OPENAI_API_KEY=
LLM_MODEL=GLM-4.7
LLM_BASE_URL=https://open.bigmodel.cn/api/coding/paas/v4
```

这意味着：

- 安装时不需要手工输入任何值
- 默认模型和 Base URL 会自动写入
- `OPENAI_API_KEY` 可以后续再编辑
- 如果用户已经在 shell 环境变量或项目 `.env` 里设置了密钥，也可以不改这个文件

## 配置文件位置

### 运行时配置优先级

运行时环境变量和 `.env` 的优先级如下：

1. 当前 shell 中已经存在的环境变量
2. 当前工作目录下的 `.env`
3. `~/.tg_agent/.env`

也就是说：

- 如果系统环境变量里已经设置了 `OPENAI_API_KEY`，程序会优先使用它
- 如果系统里没设置，则会看当前工作目录 `.env`
- 再没有，就看 `~/.tg_agent/.env`

### Worker 配置优先级

`tg_crab_worker.json` 的查找顺序如下：

1. 当前工作目录下的 `tg_crab_worker.json`
2. `~/.tg_agent/tg_crab_worker.json`
3. 打包时内置的默认 `tg_crab_worker.json`

## 安装后建议配置

如果用户目录里的 `~/.tg_agent/.env` 还是默认模板，通常只需要补上：

```dotenv
OPENAI_API_KEY=你的密钥
```

如有需要，也可以修改：

```dotenv
LLM_MODEL=GLM-4.7
LLM_BASE_URL=https://open.bigmodel.cn/api/coding/paas/v4
```

这些内容既可以写在：

- 当前项目目录的 `.env`
- 也可以写在 `~/.tg_agent/.env`

如果希望“全局安装后任何项目都能直接用”，建议写到 `~/.tg_agent/.env`。

## 常用命令

### 查看帮助

Windows：

```powershell
tg-agent --help
```

Linux：

```bash
tg-agent --help
```

### 指定工作目录

Windows：

```powershell
tg-agent --root-dir D:\your-project
```

Linux：

```bash
tg-agent --root-dir /path/to/your-project
```

### 关闭 IM 模式，仅本地 CLI 运行

Windows：

```powershell
tg-agent --no-im-enable --no-local-bridge
```

Linux：

```bash
tg-agent --no-im-enable --no-local-bridge
```

## 验证建议

建议至少做下面两步验证。

### Windows

```powershell
.\tg-agent.exe --help
.\tg-agent.exe --no-im-enable --no-local-bridge
```

### Linux

```bash
./tg-agent --help
./tg-agent --no-im-enable --no-local-bridge
```

如果是安装后的命令验证，则改为：

Windows：

```powershell
tg-agent --help
```

Linux：

```bash
tg-agent --help
```

## 常见问题

### 1. 为什么脚本选到了错误的 Python

如果终端里没有真正激活目标环境，脚本可能会先命中 PATH 中的默认 `python`。

虽然现在脚本会继续尝试其他 Conda 环境中的解释器，但最稳妥的方式仍然是先激活环境：

```powershell
conda activate 你的环境名
powershell -ExecutionPolicy Bypass -File .\scripts\release.ps1
```

### 2. 为什么 exe 能 `--help`，但自动化环境里启动 CLI 报没有控制台

这是因为 `prompt_toolkit` 需要真实的交互式控制台。在某些自动化进程、CI、IDE 后台执行器里，没有标准 Windows console buffer，可能会报 `NoConsoleScreenBufferError`。

这不一定代表 exe 本身不能正常使用。请优先在真实终端中验证。

### 3. 为什么打包日志里还有 Pydantic 或 DLL 警告

当前打包配置已经补齐了关键 DLL，但日志里仍可能出现一些警告。是否真正影响使用，应以以下结果为准：

- `tg-agent.exe --help`
- 真实终端中启动 CLI

## 建议的对外交付方式

如果要发给其他用户，建议直接提供：

- Windows：`tg-agent-windows-x64-v<version>.zip`
- Linux：对应平台压缩包

并告诉用户：

1. 解压
2. 运行安装脚本
3. 如有需要，编辑 `~/.tg_agent/.env`
4. 重新打开终端
5. 执行 `tg-agent`

这样体验最接近普通桌面软件安装包，同时又不需要用户自己安装 Python。
