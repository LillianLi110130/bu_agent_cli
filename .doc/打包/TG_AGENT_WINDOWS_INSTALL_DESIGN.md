# TG-Agent Windows 打包安装方案设计

## 1. 背景

当前 `tg-agent-cli` 的主分发方式仍然是 `wheel` 安装，用户侧前提是：

1. 本机已具备可用的 Python 环境
2. 能执行 `pip install`
3. 能自行处理运行时依赖、环境变量与启动方式

这套方式适合研发环境，但对公司内部普通 Windows 用户并不友好。

参考 `.doc/打包` 中另一套 agent CLI 的实现方式，可以看到其核心思路不是“发一个 Python 包”，而是“发一个可离线部署的完整运行包”，包括：

1. 预封装 Python 运行时
2. 部署脚本负责解压、修复路径、生成启动器
3. 启动器负责设置环境变量、初始化用户目录并拉起应用
4. 可选再通过安装器封装成一键安装体验

本设计用于定义：如果 `tg-agent-cli` 参考这一路线完成 Windows 分发，推荐采用什么方案，以及需要对现有代码做哪些调整。

---

## 2. 设计目标

### 2.1 目标

1. 支持 Windows 用户在无 Python 预装的情况下完成部署和启动
2. 支持尽量离线的交付与安装
3. 支持桌面快捷方式启动
4. 支持升级时保留用户级配置与认证状态
5. 支持现有 CLI、Worker、IM bridge、Ralph 等运行能力
6. 尽量复用当前仓库已有的运行时初始化逻辑

### 2.2 非目标

1. 本期不追求跨平台统一安装器
2. 本期不优先追求极致小体积
3. 本期不要求立即替换现有 `wheel` 分发方式
4. 本期不要求一步到位做 MSI 企业安装包

---

## 3. 现状分析

### 3.1 当前仓库已有基础

当前项目已经具备一部分“独立分发”所需能力：

1. 能区分源码运行与 frozen 运行
   - `agent_core/runtime_paths.py`
2. 能从应用目录读取打包内置 `.env`
   - `packaged_env_path()`
3. CLI 启动时会自动创建用户级运行时配置
   - `ensure_cli_runtime_state()`
4. frozen 模式下，Worker 子进程可复用同一个可执行入口
   - `claude_code.py` 中 `_INTERNAL_WORKER_FLAG`

这意味着我们不是从零开始，已经具备“安装目录 + 用户目录 + 单可执行入口”的一部分兼容能力。

### 3.2 当前不足

如果直接照搬安装包思路，当前代码还存在几个明显问题：

1. 用户数据目录治理不彻底
   - 部分状态写入 `~/.tg_agent`
   - 但也有部分逻辑依赖当前工作目录下的 `.tg_agent`
2. 用户配置会被覆盖
   - `ensure_cli_runtime_state()` 当前会重写用户 `.env`
   - 用户手工改过的配置在升级或重新启动后有丢失风险
3. 认证与桥接状态仍可能跟随工作目录
   - token、IM bridge 队列、Ralph 进程状态等仍有工作目录耦合
4. 当前打包文档主线仍然是 `wheel`
   - 尚未形成完整的 Windows 安装版产物链路

这些问题如果不先处理，安装到 `Program Files` 或共享目录后，容易出现权限、数据覆盖和升级异常。

---

## 4. 参考方案分析

参考 `.doc/打包/win_deploy.ps1`，对方方案可抽象为以下流程：

1. 交付一个包含 Python 运行时、Node.js、配置目录、缓存依赖和脚本的分发目录
2. 部署脚本执行 `conda-unpack` 修复搬迁后的运行时路径
3. 对 Windows 上 `conda-pack` 的已知问题做补丁修复
4. 动态生成 `launcher.bat`
5. `launcher.bat` 设置环境变量、证书路径、默认端口
6. 首次启动时自动初始化用户目录
7. 最终启动业务应用
8. 创建桌面快捷方式，用户只接触快捷方式

其优点：

1. 对用户机器要求低
2. 可离线交付
3. 启动体验统一
4. 升级与运行时版本可控

其代价：

1. 安装包体积较大
2. 打包链路复杂度提升
3. 需要维护运行时修复和安装脚本
4. Windows 特定问题需要专项处理

---

## 5. 方案选型

### 5.1 备选方案

#### 方案 A：继续仅提供 wheel

优点：

1. 简单
2. 现有链路基本可复用
3. 维护成本最低

缺点：

1. 用户门槛高
2. 对 Python/pip 环境依赖强
3. 不适合大规模内部桌面分发

#### 方案 B：提供 Windows Portable Bundle

形态：

1. 打包 Python 运行时
2. 提供 `deploy.bat` / `win_deploy.ps1`
3. 提供 `launcher.bat`
4. 用户双击部署，再双击启动

优点：

1. 实现复杂度可控
2. 易于调试
3. 不强依赖安装器工具链
4. 最适合作为第一阶段落地方案

缺点：

1. 仍需要用户执行一次部署脚本
2. “安装感”不如标准 installer

#### 方案 C：Portable Bundle + NSIS 安装器

形态：

1. 底层产物仍是 Portable Bundle
2. 再用 NSIS 封装为安装包

优点：

1. 用户体验最好
2. 更符合 Windows 桌面软件分发习惯
3. 支持快捷方式、卸载入口、安装目录选择

缺点：

1. 工具链更多
2. 初期调试成本更高
3. 如果底层目录治理没做好，安装器只会放大问题

### 5.2 推荐结论

推荐采用分阶段方案：

1. 第一阶段落地方案 B：`Windows Portable Bundle`
2. 第二阶段在方案 B 稳定后，增加方案 C：`NSIS Installer`
3. 保留现有 `wheel` 作为研发和高级用户分发通道

原因：

1. 方案 B 已足够接近参考项目的核心能力
2. 方案 B 能先解决“无 Python 运行环境也能启动”的主要问题
3. 方案 B 还能逼出目录治理、配置持久化等底层问题
4. 在底层稳定前直接做 NSIS，收益不如先把运行模型理顺

---

## 6. 总体架构

### 6.1 分发产物结构

建议第一阶段产物目录如下：

```text
tg-agent-win-x64-v<version>/
  deploy.bat
  win_deploy.ps1
  tg-agent-launcher.bat
  tg-agent.ico
  tg-agent-env.zip
  tg-agent-runtime/
    python.exe
    Scripts/
    Lib/site-packages/...
  app/
    agent_core/
    cli/
    tools/
    tg_mem/
    skills/
    plugins/
    config/
    template/
    claude_code.py
    .env
    tg_crab_worker.json
  cache/
    conda_unpack_wheels/
```

说明：

1. `tg-agent-runtime/` 存放 `conda-pack` 后解压的 Python 运行时
2. `app/` 存放项目源码与静态资源
3. `cache/conda_unpack_wheels/` 用于 Windows 特定修复
4. `tg-agent-launcher.bat` 是用户实际启动入口

### 6.2 目录职责划分

建议明确三类目录：

1. 安装目录
   - 只读
   - 放运行时、程序文件、默认配置、静态资源
2. 用户配置目录
   - 建议默认 `%USERPROFILE%\\.tg_agent`
   - 放 `.env`、`tg_crab_worker.json`、token 等
3. 用户工作目录
   - 由用户在桌面启动时自行选择，或由当前命令行工作目录决定
   - 放 IM bridge、Ralph、工作流产物、项目态状态

原则：

1. 安装目录不写用户状态
2. 升级安装目录时不影响用户目录
3. 所有会变化的数据都放用户目录或工作目录

---

## 7. 启动流程设计

### 7.1 首次部署流程

`deploy.bat` 负责：

1. 调用 `powershell -ExecutionPolicy Bypass -File .\\win_deploy.ps1`

`win_deploy.ps1` 负责：

1. 解压或校验 `tg-agent-runtime/`
2. 执行 `conda-unpack`
3. 若存在 Windows 兼容性问题，使用本地 wheels 做修复
4. 生成 `tg-agent-launcher.bat`
5. 创建桌面快捷方式
6. 可选创建开始菜单快捷方式

### 7.2 启动器流程

`tg-agent-launcher.bat` 建议负责：

1. 切换到安装目录
2. 设置 Python 运行时 PATH
3. 定位固定的用户级全局目录 `~/.tg_agent`
4. 设置工作目录
5. 设置 SSL 证书路径
6. 首次执行时完成用户目录初始化
7. 最终启动 CLI

建议入口形态：

```bat
"%RUNTIME_DIR%\python.exe" -u "%APP_DIR%\claude_code.py" --root-dir "%TG_AGENT_WORKSPACE%"
```

如果后续做成 PyInstaller 单可执行，也可切换为：

```bat
"%~dp0tg-agent.exe" --root-dir "%TG_AGENT_WORKSPACE%"
```

### 7.3 用户首次启动行为

首次启动时应自动完成：

1. 创建 `%USERPROFILE%\\.tg_agent`
2. 生成默认 `.env`
3. 生成或复制默认 `tg_crab_worker.json`
4. 若需要鉴权，则在首次进入相关流程时触发认证

不建议：

1. 每次启动都覆盖用户 `.env`
2. 每次启动都覆盖用户 worker 配置

---

## 8. 代码改造设计

### 8.1 运行时配置初始化

当前 `ensure_cli_runtime_state()` 的问题是会直接覆盖用户配置。

建议改为：

1. 仅在文件不存在时生成
2. 对默认模板做版本化迁移
3. 必要时仅补充缺失字段，不覆盖用户已有值

建议新增能力：

1. `ensure_runtime_env_file()`
2. `ensure_runtime_worker_config()`
3. `merge_missing_env_keys()`
4. `migrate_runtime_state(version)`

### 8.2 用户状态目录统一

建议新增统一路径函数，例如：

1. `tg_agent_home()`
2. `tg_agent_workspace_root()`
3. `tg_agent_runtime_state_dir()`
4. `tg_agent_logs_dir()`

然后调整以下模块：

1. `cli/worker/auth.py`
   - token 默认写入用户级状态目录
   - 不再依赖 `Path.cwd()`
2. `cli/im_bridge/store.py`
   - 默认根目录保持为当前 workspace/root_dir
3. `cli/ralph_process_manager.py`
   - 进程状态和日志保持在当前 workspace/root_dir
4. 其他所有 `.tg_agent` 路径拼接逻辑
   - 优先走统一路径函数

### 8.3 frozen / packaged 模式兼容

当前已有 `is_frozen_app()` 和 `application_root()`，建议继续复用，不另起一套机制。

建议约定：

1. `application_root()` 只指向安装目录或 bundle 根目录
2. `tg_agent_home()` 只指向用户级持久目录
3. `--root-dir` 由桌面启动器或用户显式指定，不隐含默认工作区目录

### 8.4 启动参数默认值

建议在安装版中默认加上：

1. `--root-dir %TG_AGENT_WORKSPACE%`
   - 其中 `%TG_AGENT_WORKSPACE%` 应由桌面启动程序让用户选择或记忆上次目录

可选：

1. 如果 IM worker 默认应启用，则由启动器显式传参
2. 若某些模式仅面向桌面用户，可在启动器中裁剪默认参数

---

## 9. 打包构建设计

### 9.1 第一阶段构建链路

建议新增以下脚本：

1. `scripts/release/build_windows_portable.ps1`
2. `scripts/release/render_launcher.ps1`
3. `scripts/release/verify_windows_portable.ps1`

构建步骤：

1. 使用指定 conda 环境安装项目依赖
2. 用 `conda-pack` 生成 Python 环境包
3. 解压到 `dist/release/tg-agent-win-x64-v<version>/tg-agent-runtime`
4. 拷贝 `agent_core/`、`cli/`、`tools/`、`tg_mem/`、`skills/`、`plugins/`、`config/`、`template/`
5. 拷贝 `claude_code.py`、`.env`、`tg_crab_worker.json`
6. 收集 Windows 修复所需 wheels
7. 生成 `deploy.bat` 与 `win_deploy.ps1`
8. 生成 `tg-agent-launcher.bat`
9. 对产物执行验证脚本

### 9.2 第二阶段安装器链路

待 portable 方案稳定后新增：

1. `scripts/release/build_windows_installer.ps1`
2. `installer/tg_agent_desktop.nsi`

NSIS 只负责：

1. 安装文件到目标目录
2. 创建桌面/开始菜单快捷方式
3. 写入卸载信息
4. 调用部署后初始化脚本

不建议把业务逻辑大量写进 NSIS 脚本。

---

## 10. 配置与数据策略

### 10.1 默认配置来源

默认配置仍建议保留三层优先级：

1. 进程环境变量
2. 工作目录 `.env`
3. 用户目录 `.env`
4. 安装目录内置 `.env`

其中：

1. 安装目录 `.env` 仅作为默认模板来源
2. 用户目录 `.env` 是用户长期持久化配置

### 10.2 用户配置更新策略

建议：

1. 新版本新增配置项时，只补充缺失键
2. 不回写用户已修改的键
3. 为模板加 `runtime_state_version`
4. 升级时记录迁移日志

### 10.3 认证信息

建议：

1. token 永久放在用户目录
2. 不放在安装目录
3. 不放在临时工作目录

这样升级安装包不会影响登录状态。

---

## 11. 风险与应对

### 11.1 conda-pack Windows 兼容性

风险：

1. 解压后路径替换可能破坏部分包

应对：

1. 复用参考项目做法
2. 建立受影响包白名单
3. 提前缓存 wheels
4. 安装后做 import 自检

### 11.2 安装目录写权限问题

风险：

1. 安装在 `Program Files` 后无法写入状态

应对：

1. 一律把状态写到用户目录
2. 启动器固定使用用户目录下的 `~/.tg_agent`，并在启动时显式选择工作目录

### 11.3 升级覆盖用户配置

风险：

1. 用户修改 `.env` 后被启动时重写

应对：

1. 改造初始化逻辑为“补齐缺失项，不覆盖用户值”

### 11.4 产物体积较大

风险：

1. 预打包 Python 环境会使发布包增大

应对：

1. 先接受体积换取稳定性
2. 后续再考虑裁剪依赖或迁移为 PyInstaller/Nuitka 等方案

---

## 12. 实施计划

### Phase 1：目录治理与配置保护

目标：

1. 统一用户数据目录
2. 禁止启动时覆盖用户配置
3. 理顺 token、bridge、Ralph 等状态路径

交付物：

1. 运行时路径抽象
2. 配置初始化改造
3. 回归测试

### Phase 2：Windows Portable Bundle

目标：

1. 产出可离线部署目录
2. 一键部署脚本可用
3. 桌面快捷方式可用

交付物：

1. 构建脚本
2. 部署脚本
3. 启动器
4. 验证脚本

### Phase 3：NSIS 安装器

目标：

1. 提供标准安装体验

交付物：

1. NSIS 脚本
2. 安装/卸载验证流程

---

## 13. 推荐落地结论

综合当前代码基础和参考项目经验，推荐的最终路线是：

1. 保留 `wheel` 分发，继续服务开发环境
2. 新增 `Windows Portable Bundle`，作为内部桌面用户主分发方式
3. 在 portable 方案稳定后，再封装为 `NSIS Installer`

其中最关键的前置工作不是安装器本身，而是：

1. 统一用户状态目录
2. 把安装目录与用户目录彻底分离
3. 修复启动时覆盖用户配置的问题

只有这三件事先做好，后续无论是 `conda-pack + launcher`，还是 `NSIS`，都能稳定落地。

---

## 14. 后续建议

建议下一步按以下顺序推进：

1. 先改运行时目录与配置初始化逻辑
2. 再补 `scripts/release/build_windows_portable.ps1`
3. 再补 `deploy.bat` / `win_deploy.ps1` / `tg-agent-launcher.bat`
4. 最后再做 NSIS

如果进入实现阶段，建议优先修改这些位置：

1. `agent_core/runtime_paths.py`
2. `claude_code.py`
3. `cli/worker/auth.py`
4. `cli/im_bridge/store.py`
5. `cli/ralph_process_manager.py`
6. `docs/standalone-packaging.md`
