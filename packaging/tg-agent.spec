# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_submodules

project_root = Path(SPECPATH).resolve().parent
conda_bin = Path(sys.prefix) / "Library" / "bin"

datas = [
    (str(project_root / "agent_core" / "prompts"), "agent_core/prompts"),
    (str(project_root / "skills"), "skills"),
    (str(project_root / "plugins"), "plugins"),
    (str(project_root / "template"), "template"),
    (str(project_root / ".devagent"), ".devagent"),
    (str(project_root / "config"), "config"),
    (str(project_root / "gateway"), "gateway"),
    (str(project_root / "tg_crab_worker.json"), "."),
]

binary_names = [
    "ffi-8.dll",
    "libcrypto-3-x64.dll",
    "libexpat.dll",
    "libmpdec-4.dll",
    "libssl-3-x64.dll",
]
binaries = []
for binary_name in binary_names:
    binary_path = conda_bin / binary_name
    if binary_path.exists():
        binaries.append((str(binary_path), "."))

hiddenimports = sorted(
    set(
        collect_submodules("cli.worker")
        + [
            "agent_core.gateway.main",
        ]
    )
)

a = Analysis(
    [str(project_root / "claude_code.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="tg-agent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
