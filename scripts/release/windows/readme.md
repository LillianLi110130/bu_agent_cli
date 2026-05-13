# TG-Agent Windows Portable 打包与使用手册

本文档面向两类读者：

1. 打包维护者：负责在 Windows 机器上构建 `tg-agent` portable bundle。
2. 最终用户：拿到 portable bundle 后，在自己的 Windows 电脑上完成安装和启动。

当前方案不是 `exe` 单文件方案，也不是要求用户预装 Python 的 `wheel` 方案，而是：

- 包内自带 `python-runtime/`
- 包内自带离线 `wheelhouse/`
- 通过 `deploy.bat` / `win_deploy.ps1` 完成安装
- 通过 `tg-agent-launcher.bat` 完成启动

---

## 1. 产物说明

标准 portable bundle 目录大致如下：

```text
tg-agent-windows-x64-v0.1.0-portable/
  deploy.bat
  win_deploy.ps1
  tg-agent-launcher.bat
  crab.ico
  README.txt
  python-runtime/
  wheelhouse/
  app/
```

各目录/脚本职责如下：

- `deploy.bat`
  一次性安装入口。双击或命令行运行都可以。
- `win_deploy.ps1`
  真正执行安装逻辑的 PowerShell 脚本。
- `tg-agent-launcher.bat`
  日常启动入口。
- `crab.ico`
  桌面快捷方式使用的自定义图标。没有提供时，快捷方式回退到系统默认图标。
- `python-runtime/`
  随包分发的 Python 运行时。
- `wheelhouse/`
  离线依赖包目录，包含项目 wheel 和依赖 wheels。
- `app/`
  项目代码和默认配置载荷。

---

## 2. 方案特点

这个方案的目标是：

- 用户电脑不需要预装 Python
- 用户安装时默认不需要联网下载依赖
- 不打成自定义业务 `exe`
- 保留 `bat + PowerShell` 的安装和启动方式

但要注意：

- 打包机需要有一套可用的标准 CPython
- `wheelhouse` 必须和打包用的 Python 主版本、次版本一致
- 如果你改了项目依赖，必须重新构建 `wheelhouse`

---

## 3. 打包前准备

### 3.1 打包机要求

建议满足以下条件：

- Windows
- 一套标准 CPython 3.10.x
- 可用的 `pip`
- 仓库代码已更新到目标版本

不建议：

- 直接用 conda runtime 做 portable runtime
- 混用不同 Python 版本的 wheelhouse

### 3.2 为什么要求标准 CPython

当前 `build_windows_portable.ps1` 会把 Python runtime 一起裁进产物里。

它依赖的是：

- `python.exe`
- `sys.base_prefix`
- 标准 CPython 安装目录结构

如果你传入的是 conda Python，脚本会默认拒绝，因为 conda runtime 直接搬运在 Windows 上容易出路径和可重定位问题。

### 3.3 如何确认自己的 Python

例如：

```powershell
D:\python\python.exe -c "import sys; print(sys.executable); print(sys.version); print(sys.base_prefix)"
```

推荐你显式传 `-PythonExecutable`，不要完全依赖脚本自动发现。

---

## 4. 打包步骤

建议按两步走：

1. 先构建 wheelhouse
2. 再构建 portable bundle

这样更稳，也更容易排查问题。

如果你希望桌面快捷方式带自定义图标，先把 `.ico` 文件放到：

```text
scripts/release/windows/assets/crab.ico
```

也可以在构建 portable bundle 时显式传：

```powershell
-ShortcutIcon D:\path\to\your\crab.ico
```

### 4.1 第一步：构建 wheelhouse

脚本位置：

- [build_wheelhouse.ps1](/d:/llm_project/bu_agent_cli/scripts/release/windows/build_wheelhouse.ps1)

推荐命令：

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/release/windows/build_wheelhouse.ps1 -PythonExecutable D:/python/python.exe -Clean
```

说明：

- `-PythonExecutable`
  指定用哪个 Python 构建 wheelhouse。建议显式传。
- `-Clean`
  先清空旧 wheelhouse，避免把别的 Python 版本残留 wheel 混进去。

默认输出目录：

```text
dist/package/windows/wheelhouse
```

如果只想构建项目自己的 wheel，不带依赖，可以使用：

```powershell
-ProjectOnly
```

但日常打 portable 包通常不建议只打项目 wheel，因为最终用户安装时需要完整依赖。

### 4.2 第二步：构建 portable bundle

脚本位置：

- [build_windows_portable.ps1](/d:/llm_project/bu_agent_cli/scripts/release/windows/build_windows_portable.ps1)

推荐命令：

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/release/windows/build_windows_portable.ps1 -PythonExecutable D:/python/python.exe -SourceWheelhouse ./dist/package/windows/wheelhouse
```

说明：

- `-PythonExecutable`
  指定 portable bundle 内置 runtime 的来源 Python。
- `-SourceWheelhouse`
  指定已经构建好的离线 wheelhouse。
- `-ShortcutIcon`
  可选，显式指定桌面快捷方式图标的 `.ico` 文件。若不传，脚本会尝试使用 `scripts/release/windows/assets/crab.ico`。

默认输出目录：

```text
dist/release/tg-agent-windows-x64-v<version>-portable/
```

如果不想额外生成外层 zip，可以加：

```powershell
-SkipZip
```

如果旧的输出目录被占用，可以改一个新的输出根目录：

```powershell
-OutputRoot .\dist\release2
```

---

## 5. 打包脚本会默认用什么 Python

如果你不传 `-PythonExecutable`，当前脚本会按下面顺序尝试：

1. `VIRTUAL_ENV\Scripts\python.exe`
2. `PYTHON` 环境变量
3. `PATH` 里的 `python`
4. `py -3`

这意味着：

- 如果你当前激活了 Poetry venv，通常会优先拿到 Poetry venv 里的 `python.exe`
- 但我仍然建议显式传 `-PythonExecutable`

原因很简单：

- 更可控
- 更不容易误拿到 conda Python
- 更不容易混入错误版本 wheelhouse

---

## 6. 打包后建议验证

验证脚本位置：

- [verify_windows_portable.ps1](/d:/llm_project/bu_agent_cli/scripts/release/windows/verify_windows_portable.ps1)

基本验证：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\windows\verify_windows_portable.ps1
```

如果要跑 smoke：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\windows\verify_windows_portable.ps1 -RunSmoke
```

---

## 7. 最终用户如何使用

### 7.1 第一次使用

用户拿到 bundle 后：

1. 解压整个目录
2. 双击或运行 `deploy.bat`
3. 安装完成后，再运行 `tg-agent-launcher.bat`

### 7.2 `deploy.bat` 做了什么

`deploy.bat` 本身只是一个入口，真正的安装逻辑在 `win_deploy.ps1` 里。

它会：

1. 找到包里的 `python-runtime\python.exe`
2. 在用户目录下创建：

```text
%USERPROFILE%\.tg_agent\.venv
```

3. 给这个 venv 补 `pip`
4. 从 bundle 内的 `wheelhouse\` 离线安装 `tg-agent`
5. 生成：

```text
%USERPROFILE%\.tg_agent\bin\tg-agent-entry.py
```

6. 如果用户目录里还没有以下文件，就复制默认文件：

```text
%USERPROFILE%\.tg_agent\.env
%USERPROFILE%\.tg_agent\tg_crab_worker.json
```

7. 默认会在桌面创建快捷方式：

```text
TG-Agent Portable.lnk
```

如果 bundle 根目录里存在 `crab.ico`，这个快捷方式会使用它作为图标；否则使用系统默认图标。

### 7.3 `tg-agent-launcher.bat` 做了什么

它是日常启动入口，会：

1. 找：

```text
%USERPROFILE%\.tg_agent\.venv\Scripts\python.exe
%USERPROFILE%\.tg_agent\bin\tg-agent-entry.py
```

2. 决定本次启动的工作目录 `WORKSPACE`
3. 切到该目录后执行 `tg-agent`

它不会重新安装依赖，也不会重新创建 venv。它的职责就是“启动”。

---

## 8. 怎么指定工作目录

这是这套方案里最容易搞混的地方。

### 8.1 最推荐的方式：从终端进入目标目录后启动

例如：

```powershell
cd D:\my-project
D:\path\to\tg-agent-windows-x64-v0.1.0-portable\tg-agent-launcher.bat
```

这时工作目录就是：

```text
D:\my-project
```

也就是说：

- 你从哪个目录启动
- 哪个目录就是当前 workspace

### 8.2 直接双击 launcher 会怎样

如果你直接从 bundle 目录双击 `tg-agent-launcher.bat`：

- 当前目录通常就是 bundle 自己所在目录
- 脚本会检测到这一点
- 为了避免把运行态写进安装目录，它会回落到桌面目录：

```text
%USERPROFILE%\Desktop
```

### 8.3 桌面快捷方式为什么会默认落到桌面

因为 `deploy.bat` 创建桌面快捷方式时，把快捷方式的 `WorkingDirectory` 设成了桌面。

所以：

- 双击桌面快捷方式时
- launcher 看到当前目录是桌面
- 就会以桌面作为 workspace 启动

### 8.4 如果想固定某个工作目录怎么办

可以自己再写一个小 bat：

```bat
@echo off
cd /d "D:\my-workspace"
call "D:\path\to\tg-agent-windows-x64-v0.1.0-portable\tg-agent-launcher.bat"
```

也可以在命令行里每次先 `cd` 到目标目录。

当前版本还没有给 `tg-agent-launcher.bat` 单独加 `--workspace` 这种包装参数，它主要依赖当前工作目录。

---

## 9. 用户目录里会生成什么

安装完成后，通常会在用户目录下出现：

```text
%USERPROFILE%\.tg_agent\
  .env
  tg_crab_worker.json
  .venv\
  bin\
```

其中：

- `.env`
  用户级全局配置
- `tg_crab_worker.json`
  用户级默认 worker 配置
- `.venv`
  安装出来的 Python 虚拟环境
- `bin\tg-agent-entry.py`
  启动 shim

这几个目录/文件属于用户数据，不属于 portable bundle 安装目录。

---

## 10. 配置文件怎么改

首次安装后，如果你需要补 API Key 或修改模型配置，主要改这里：

```text
%USERPROFILE%\.tg_agent\.env
```

例如：

```dotenv
OPENAI_API_KEY=your-key
LLM_MODEL=GLM-4.7
LLM_BASE_URL=https://open.bigmodel.cn/api/coding/paas/v4
```

如果你在某个具体 workspace 下还放了自己的 `.env`，那么运行时通常会优先使用 workspace 的 `.env` 覆盖用户级默认值。

---

## 11. 常见问题

### 11.1 打包时报 `Missing hatchling`

说明当前构建 Python 里没有安装 `hatchling`。

修复：

```powershell
D:\python\python.exe -m pip install hatchling
```

### 11.2 安装时报 `No matching distribution found`

通常是 wheelhouse 和 Python 版本不匹配。

例如：

- bundle 里是 Python 3.10 runtime
- 但 wheelhouse 里混入了 `cp314` wheels

修复思路：

1. 清空旧 wheelhouse
2. 用同一个 Python 版本重新构建 wheelhouse
3. 再重新打包

### 11.3 启动时报 `No module named ...`

说明 wheelhouse 不完整，或者项目依赖刚改过但你没重建 wheelhouse。

修复思路：

1. 重新执行 `build_wheelhouse.ps1`
2. 重新执行 `build_windows_portable.ps1`
3. 删除用户机旧的：

```text
%USERPROFILE%\.tg_agent\.venv
```

4. 再重新跑 `deploy.bat`

### 11.4 打包时报输出目录被占用

这是 Windows 常见文件句柄问题。通常是旧 bundle 目录还被别的进程占着。

建议：

- 关闭资源管理器中打开的 bundle 目录
- 关闭在该目录中的 `cmd` / `PowerShell`
- 关闭正在运行的 launcher
- 或者换一个新的 `-OutputRoot`

### 11.5 双击脚本窗口一闪而过

当前版本已经做了改进：

- `deploy.bat` 成功和失败都会停住
- `tg-agent-launcher.bat` 失败会停住
- 从 bundle 目录直接双击 launcher 时，成功退出也会停住

如果仍然看不到信息，建议从命令行运行。

---

## 12. 推荐的标准操作流程

### 12.1 打包维护者

推荐固定按这两条命令执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\windows\build_wheelhouse.ps1 -PythonExecutable D:\python\python.exe -Clean
powershell -ExecutionPolicy Bypass -File .\scripts\release\windows\build_windows_portable.ps1 -PythonExecutable D:\python\python.exe -SourceWheelhouse .\dist\package\windows\wheelhouse
```

### 12.2 最终用户

推荐固定按这个顺序使用：

1. 解压 bundle
2. 运行 `deploy.bat`
3. 打开终端并进入你的项目目录
4. 运行 `tg-agent-launcher.bat`

例如：

```powershell
cd D:\my-project
D:\tools\tg-agent-windows-x64-v0.1.0-portable\tg-agent-launcher.bat
```

---

## 13. 当前方案的边界

这套 portable 方案已经能满足：

- 无需用户预装 Python
- 用户安装时默认离线
- 保留 `bat + PowerShell`
- 不做业务 `exe`

但它目前仍然有几个边界：

- workspace 选择仍然主要依赖当前工作目录
- 还没有做图形化“选择工作目录”界面
- 桌面快捷方式默认以桌面作为工作目录
- 如果项目依赖变化，必须重新打包 wheelhouse

---

## 14. 相关脚本

主要脚本位置：

- [build_wheelhouse.ps1](/d:/llm_project/bu_agent_cli/scripts/release/windows/build_wheelhouse.ps1)
- [build_windows_portable.ps1](/d:/llm_project/bu_agent_cli/scripts/release/windows/build_windows_portable.ps1)
- [verify_windows_portable.ps1](/d:/llm_project/bu_agent_cli/scripts/release/windows/verify_windows_portable.ps1)
- [render_launcher.ps1](/d:/llm_project/bu_agent_cli/scripts/release/windows/render_launcher.ps1)

设计文档：

- [TG_AGENT_WINDOWS_INSTALL_DESIGN.md](</d:/llm_project/bu_agent_cli/.doc/打包/TG_AGENT_WINDOWS_INSTALL_DESIGN.md>)
