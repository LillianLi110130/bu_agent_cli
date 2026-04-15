# Linux Portable 打包说明

这套脚本的目标是让 Linux 用户获得和 Windows portable 包尽量一致的体验：

- 用户机器不需要预装 Python
- 发布产物内自带 `python-runtime/`
- 首次执行 `deploy.sh` 后，会在 `~/.tg_agent/.venv` 完成离线安装
- 之后既可以运行 `./tg-agent-launcher.sh`，也可以在新 shell 里直接输入 `tg-agent`

## 目录

- [build_wheelhouse.sh](./build_wheelhouse.sh)：构建 Linux wheelhouse
- [build_linux_portable.sh](./build_linux_portable.sh)：组装 Linux portable bundle
- [verify_linux_portable.sh](./verify_linux_portable.sh)：校验 bundle 结构并可做 smoke test

## 前置要求

需要在 Linux 环境里执行这些脚本。不要用 Windows 产物直接改脚本来复用。

至少准备这两样：

1. 一个用于构建 wheel 的 Linux Python
2. 一份可分发的 Linux Python runtime

推荐做法：

- 构建 wheel 用你自己的 Linux Python 3.10+ 环境
- `python-runtime/` 用 `python-build-standalone` 之类的可分发 runtime

原因：

- Windows 的 `python.exe` / `.dll` 不能在 Linux 跑
- Windows 的 `win_amd64` wheels 也不能在 Linux 安装
- Linux 要重新产出自己的 runtime 和 wheelhouse

## 推荐流程

### 1. 准备 Linux runtime

把你要分发的 Linux Python runtime 解压到一个目录，例如：

```bash
/opt/python-runtime/cpython-3.12.10-linux-x86_64
```

这个目录下最终要能找到其中一种：

```text
bin/python3
bin/python
python/bin/python3
python/bin/python
```

### 2. 构建 Linux wheelhouse

```bash
./scripts/release/linux/build_wheelhouse.sh \
  --python-executable /usr/bin/python3 \
  --clean
```

默认输出到：

```text
dist/package/linux/wheelhouse
```

如果你只想先打项目自身 wheel：

```bash
./scripts/release/linux/build_wheelhouse.sh \
  --python-executable /usr/bin/python3 \
  --clean \
  --project-only
```

### 3. 组装 portable bundle

```bash
./scripts/release/linux/build_linux_portable.sh \
  --python-executable /usr/bin/python3 \
  --python-runtime-dir /opt/python-runtime/cpython-3.12.10-linux-x86_64 \
  --source-wheelhouse ./dist/package/linux/wheelhouse
```

默认输出到：

```text
dist/release/tg-agent-linux-x64-v<version>-portable
dist/release/tg-agent-linux-x64-v<version>-portable.tar.gz
```

常用参数：

- `--version TEXT`
  覆盖 bundle 名字里的版本号
- `--output-root PATH`
  指定输出根目录
- `--skip-tar`
  只生成目录，不打 `tar.gz`
- `--skip-project-wheel-build`
  使用现成 wheelhouse 时，不再额外重打项目 wheel

## 验证打包结果

只校验结构：

```bash
./scripts/release/linux/verify_linux_portable.sh
```

校验结构并执行一次隔离 smoke test：

```bash
./scripts/release/linux/verify_linux_portable.sh --run-smoke
```

smoke test 会：

- 在 bundle 下创建临时 HOME
- 运行 `deploy.sh --skip-profile-update`
- 再执行 `tg-agent-launcher.sh --help`

## 用户侧安装体验

用户拿到产物后，大致这样使用：

```bash
tar -xzf tg-agent-linux-x64-v<version>-portable.tar.gz
cd tg-agent-linux-x64-v<version>-portable
./deploy.sh
./tg-agent-launcher.sh
```

`deploy.sh` 会做这些事：

- 用 bundle 内的 `python-runtime/` 创建 `~/.tg_agent/.venv`
- 从 `wheelhouse/` 离线安装项目和依赖
- 生成 `~/.tg_agent/bin/tg-agent`
- 把 `~/.tg_agent/bin` 追加到 `~/.profile`
  如果存在，也会顺手追加到 `~/.bashrc` / `~/.zshrc`
- 如果机器上已经有旧的 `tg-agent` 命令，会给出冲突提醒和卸载提示

安装完成后，用户可以：

- 继续运行 bundle 里的 `./tg-agent-launcher.sh`
- 或重新打开一个 shell，直接运行：

```bash
tg-agent
```

## 一个完整示例

```bash
./scripts/release/linux/build_wheelhouse.sh \
  --python-executable /usr/bin/python3 \
  --clean

./scripts/release/linux/build_linux_portable.sh \
  --python-executable /usr/bin/python3 \
  --python-runtime-dir /opt/python-runtime/cpython-3.12.10-linux-x86_64 \
  --source-wheelhouse ./dist/package/linux/wheelhouse

./scripts/release/linux/verify_linux_portable.sh --run-smoke
```

## 注意事项

- 这些脚本要在 Linux 上跑，不建议在 Windows 上直接执行
- 最好在较老的 Linux 基线环境里构建 wheelhouse，兼容性会更好
- 如果 runtime 来自 conda，默认会被拒绝
  除非你显式传 `--allow-conda-python-runtime`
- 如果不传 `--python-runtime-dir`，脚本会退回去尝试复制 `--python-home` / `sys.base_prefix`
  这适合本地实验，不是最稳的分发方案

真正要发给用户时，推荐始终显式提供：

```bash
--python-runtime-dir <portable-linux-runtime>
```
