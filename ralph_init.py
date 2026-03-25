#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""初始化 Ralph spec 模板和内置 .devagent 配置。"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


class RalphInitializer:
    """负责把 Ralph 模板复制到工作区。"""

    def __init__(self) -> None:
        self.script_dir = Path(__file__).parent.absolute()
        self.template_dir = self.script_dir / "template"
        self.devagent_dir = self.script_dir / ".devagent"

    def check_template_exists(self) -> bool:
        """检查 spec 模板目录是否存在且非空。"""
        if not self.template_dir.exists():
            print(f"[ERROR] 模板目录不存在: {self.template_dir}")
            return False

        if not any(self.template_dir.iterdir()):
            print(f"[ERROR] 模板目录为空: {self.template_dir}")
            return False

        return True

    def get_template_structure(self) -> dict:
        """返回模板目录的嵌套结构。"""

        def scan_directory(path: Path, parent_key: str = "") -> dict:
            structure: dict[str, dict] = {}

            for item in sorted(path.iterdir(), key=lambda candidate: (candidate.is_file(), candidate.name)):
                key = f"{parent_key}/{item.name}".lstrip("/")
                if item.is_dir():
                    structure[key] = {
                        "type": "directory",
                        "path": item,
                        "children": scan_directory(item, key),
                    }
                else:
                    structure[key] = {
                        "type": "file",
                        "path": item,
                    }
            return structure

        return scan_directory(self.template_dir)

    def initialize_agent_setting(self, target_dir: str, agent_type: str, force: bool = False) -> bool:
        """把内置 agent 配置合并到目标目录。

        已有用户文件会保留，只在文件/目录类型冲突时才报错。
        """
        _ = force

        if agent_type != "devagent":
            print(f"[ERROR] 不支持的 agent 配置类型: {agent_type}")
            return False

        if not self.devagent_dir.exists():
            print(f"[ERROR] .devagent 目录不存在: {self.devagent_dir}")
            return False

        target_path = Path(target_dir).absolute()
        try:
            target_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"[ERROR] 创建目标目录失败: {exc}")
            return False

        if not self._check_devagent_conflicts(self.devagent_dir, target_path):
            print("[ERROR] 合并 .devagent 时发现不兼容的路径冲突。")
            return False

        print(f"[INFO] 开始将 .devagent 配置合并到: {target_path}")
        print(f"[INFO] 配置来源: {self.devagent_dir}")

        success = True
        copied_files = 0
        created_dirs = 0
        skipped_files = 0

        def copy_devagent(src_path: Path, dst_path: Path) -> None:
            nonlocal success, copied_files, created_dirs, skipped_files

            relative_path = src_path.relative_to(self.devagent_dir)

            try:
                if src_path.is_dir():
                    existed_before = dst_path.exists()
                    dst_path.mkdir(parents=True, exist_ok=True)
                    if relative_path != Path(".") and not existed_before:
                        created_dirs += 1
                        print(f"[OK] 已创建目录: {relative_path}")

                    for item in sorted(src_path.iterdir(), key=lambda candidate: (candidate.is_file(), candidate.name)):
                        copy_devagent(item, dst_path / item.name)
                    return

                dst_path.parent.mkdir(parents=True, exist_ok=True)
                if dst_path.exists():
                    skipped_files += 1
                    print(f"[SKIP] 保留已有文件: {relative_path}")
                    return

                shutil.copy2(src_path, dst_path)
                copied_files += 1
                print(f"[OK] 已复制文件: {relative_path}")
            except Exception as exc:
                print(f"[ERROR] 复制 {relative_path} 失败: {exc}")
                success = False

        copy_devagent(self.devagent_dir, target_path / ".devagent")

        if success:
            print("\n[SUCCESS] Agent 配置初始化完成。")
            print(f"新建目录数: {created_dirs}")
            print(f"复制文件数: {copied_files}")
            print(f"保留已有文件数: {skipped_files}")
            print(f"目标目录: {target_path}")
            print(f"配置类型: {agent_type}")
        else:
            print("\n[ERROR] Agent 配置初始化失败。")

        return success

    def initialize_spec(self, target_dir: str, spec_name: str, force: bool = False) -> bool:
        """根据模板初始化 Ralph spec 目录。"""
        if not self.check_template_exists():
            return False

        target_base_path = Path(target_dir).absolute()
        if target_dir == ".":
            target_path = target_base_path / "docs" / "spec" / spec_name
        else:
            target_path = target_base_path / spec_name

        try:
            target_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"[ERROR] 创建 spec 目录失败: {exc}")
            return False

        if not force and any(target_path.iterdir()):
            print(f"[ERROR] Spec 目录非空: {target_path}")
            print("如需继续初始化，请使用 --force。")
            return False

        template_structure = self.get_template_structure()
        if not template_structure:
            print("[ERROR] 模板目录为空。")
            return False

        print(f"[INFO] 开始初始化 spec '{spec_name}'，目标路径: {target_path}")
        print(f"[INFO] 模板来源: {self.template_dir}")

        success = True
        copied_files = 0
        created_dirs = 0

        def copy_template(src_path: Path, dst_path: Path, relative_path: str) -> None:
            nonlocal success, copied_files, created_dirs

            try:
                if src_path.is_dir():
                    existed_before = dst_path.exists()
                    dst_path.mkdir(parents=True, exist_ok=True)
                    if not existed_before:
                        created_dirs += 1
                        print(f"[OK] 已创建目录: {relative_path}")

                    for item in sorted(src_path.iterdir(), key=lambda candidate: (candidate.is_file(), candidate.name)):
                        child_relative_path = f"{relative_path}/{item.name}".lstrip("/")
                        copy_template(item, dst_path / item.name, child_relative_path)
                    return

                shutil.copy2(src_path, dst_path)
                copied_files += 1
                print(f"[OK] 已复制文件: {relative_path}")
            except Exception as exc:
                print(f"[ERROR] 复制 {relative_path} 失败: {exc}")
                success = False

        for item_name, item_info in template_structure.items():
            copy_template(item_info["path"], target_path / item_name, item_name)

        self._process_spec_name_parameters(target_path, spec_name)

        if success:
            print("\n[SUCCESS] Spec 初始化完成。")
            print(f"新建目录数: {created_dirs}")
            print(f"复制文件数: {copied_files}")
            print(f"Spec 目录: {target_path}")
            print(f"Spec 名称: {spec_name}")
            print("\n[目录结构]")
            self._display_project_structure(target_path)
        else:
            print("\n[ERROR] Spec 初始化失败。")

        return success

    def _process_spec_name_parameters(self, target_path: Path, spec_name: str) -> None:
        """为后续模板变量替换预留扩展点。"""
        _ = target_path
        _ = spec_name

    def _check_devagent_conflicts(self, src_dir: Path, target_dir: Path) -> bool:
        """在合并 .devagent 前检查文件/目录类型冲突。"""
        conflict_found = False

        def check_conflicts(src_path: Path, dst_path: Path) -> None:
            nonlocal conflict_found

            if conflict_found:
                return

            relative_path = src_path.relative_to(src_dir)

            if src_path.is_dir():
                if dst_path.exists() and not dst_path.is_dir():
                    print(f"[ERROR] 路径类型冲突: {relative_path}（期望目录）")
                    conflict_found = True
                    return

                for item in src_path.iterdir():
                    check_conflicts(item, dst_path / item.name)
                return

            if dst_path.exists() and dst_path.is_dir():
                print(f"[ERROR] 路径类型冲突: {relative_path}（期望文件）")
                conflict_found = True

        check_conflicts(src_dir, target_dir / ".devagent")
        return not conflict_found

    def _display_project_structure(self, path: Path, prefix: str = "") -> None:
        """打印目录树结构。"""
        if path.is_file():
            return

        items = sorted(path.iterdir(), key=lambda candidate: (candidate.is_file(), candidate.name))
        for index, item in enumerate(items):
            is_last = index == len(items) - 1
            branch = "`-- " if is_last else "|-- "
            print(f"{prefix}{branch}{item.name}{'/' if item.is_dir() else ''}")
            if item.is_dir():
                child_prefix = prefix + ("    " if is_last else "|   ")
                self._display_project_structure(item, child_prefix)


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        description="初始化 Ralph spec 模板或内置 agent 配置。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  ralph_init --spec-name my_spec
  ralph_init --spec-name my_spec --target-dir /path/to/specs
  ralph_init --spec-name my_spec --force
  ralph_init --agent-setting devagent
  ralph_init --list-templates
        """,
    )

    parser.add_argument("--spec-name", help="要初始化的 spec 名称。")
    parser.add_argument(
        "--agent-setting",
        choices=["devagent"],
        help="要初始化的内置 agent 配置包。",
    )
    parser.add_argument(
        "--target-dir",
        default=".",
        help="目标目录。spec 默认会初始化到 ./docs/spec/<spec_name>。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许在非空 spec 目录中继续初始化。",
    )
    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="列出内置 spec 模板结构。",
    )

    args = parser.parse_args()
    initializer = RalphInitializer()

    if args.list_templates:
        if not initializer.check_template_exists():
            return 1

        print("=== 模板结构 ===")
        structure = initializer.get_template_structure()

        def print_structure(struct: dict, indent: int = 0) -> None:
            for name, info in struct.items():
                prefix = "  " * indent
                if info["type"] == "directory":
                    print(f"{prefix}[DIR] {name}/")
                    print_structure(info.get("children", {}), indent + 1)
                else:
                    print(f"{prefix}[FILE] {name}")

        print_structure(structure)
        return 0

    if not args.spec_name and not args.agent_setting:
        print("[ERROR] --spec-name 和 --agent-setting 至少需要提供一个。")
        parser.print_help()
        return 1

    if args.agent_setting:
        success = initializer.initialize_agent_setting(
            target_dir=args.target_dir,
            agent_type=args.agent_setting,
            force=args.force,
        )
    else:
        success = initializer.initialize_spec(
            target_dir=args.target_dir,
            spec_name=args.spec_name,
            force=args.force,
        )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
