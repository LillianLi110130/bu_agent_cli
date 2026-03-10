#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ralph_init.py - Ralph模板初始化脚本
功能: 将template目录下的文件初始化到指定的规格目录下
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

class RalphInitializer:
    """Ralph规格模板初始化器"""

    def __init__(self):
        self.script_dir = Path(__file__).parent.absolute()
        self.template_dir = self.script_dir / "template"
        self.devagent_dir = self.script_dir / ".devagent"

    def check_template_exists(self) -> bool:
        """检查模板目录是否存在"""
        if not self.template_dir.exists():
            print(f"[ERROR] 模板目录不存在: {self.template_dir}")
            return False

        if not any(self.template_dir.iterdir()):
            print(f"[ERROR] 模板目录为空: {self.template_dir}")
            return False

        return True

    def get_template_structure(self) -> dict:
        """获取模板目录结构"""

        def scan_directory(path, parent_key=""):
            """递归扫描目录"""
            structure = {}

            for item in path.iterdir():
                if item.is_dir():
                    key = f"{parent_key}/{item.name}".lstrip('/')
                    structure[key] = {
                        'type': 'directory',
                        'path': item,
                        'children': scan_directory(item, key)
                    }
                else:
                    key = f"{parent_key}/{item.name}".lstrip('/')
                    structure[key] = {
                        'type': 'file',
                        'path': item
                    }
            return structure

        return scan_directory(self.template_dir)

    def initialize_agent_setting(self, target_dir: str, agent_type: str, force: bool = False) -> bool:
        """初始化agent设置"""

        if agent_type != 'devagent':
            print(f"[ERROR] 不支持的agent设置类型: {agent_type}")
            return False

        # 检查.devagent目录是否存在
        if not self.devagent_dir.exists():
            print(f"[ERROR] .devagent目录不存在: {self.devagent_dir}")
            return False

        # 获取目标目录
        target_path = Path(target_dir).absolute()

        # 检查目标目录是否存在
        if not target_path.exists():
            try:
                target_path.mkdir(parents=True, exist_ok=True)
                print(f"[INFO] 创建目标目录: {target_path}")
            except Exception as e:
                print(f"[ERROR] 创建目标目录失败: {e}")
                return False

        # 检查文件冲突
        if not self._check_devagent_conflicts(self.devagent_dir, target_path):
            print("[ERROR] 存在文件冲突, 操作终止")
            return False

        print(f"[INFO] 开始复制.devagent配置到目录: {target_path}")
        print(f"[INFO] 配置来源:", self.devagent_dir)

        # 复制.devagent目录
        success = True
        copied_files = 0
        copied_dirs = 0

        def copy_devagent(src_path, dst_path):
            """复制.devagent文件或目录"""
            nonlocal copied_files, copied_dirs, success

            relative_path = src_path.relative_to(self.devagent_dir)

            try:
                if src_path.is_dir():
                    # 创建目标目录
                    dst_path.mkdir(parents=True, exist_ok=True)
                    if dst_path != target_path:  # 不统计根目录
                        copied_dirs += 1
                        print(f"[OK] 创建目录: {relative_path}")

                    # 递归复制子目录和文件
                    for item in src_path.iterdir():
                        copy_devagent(item, dst_path / item.name)

                else:
                    # 复制文件
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                    copied_files += 1
                    print(f"[OK] 复制文件: {relative_path}")

            except Exception as e:
                print(f"[ERROR] 复制 {relative_path} 失败: {e}")
                success = False

        # 开始复制.devagent目录
        copy_devagent(self.devagent_dir, target_path / ".devagent")

        # 输出结果
        if success:
            print(f"\n[SUCCESS] Agent设置初始化完成! ")
            print(f"创建目录: {copied_dirs} 个")
            print(f"复制文件: {copied_files} 个")
            print(f"配置位置: {target_path}")
            print(f"配置类型: {agent_type}")

        else:
            print("\n[ERROR] Agent设置初始化过程中出现错误")

        return success

    def initialize_spec(self, target_dir: str, spec_name: str, force: bool = False) -> bool:
        """初始化规格模板"""

        # 检查模板目录
        if not self.check_template_exists():
            return False

        # 构建目标路径:
        # 如果target_dir是当前目录('.'), 则调整为当前目录/docs/spec/spec-name
        # 否则调整为指定目录/spec_name
        target_base_path = Path(target_dir).absolute()
        if target_dir == '.':
            # 默认路径: 当前目录 + /docs + /spec + /spec-name
            target_path = target_base_path / "docs" / "spec" / spec_name
        else:
            # 指定目录: 指定目录 + /spec_name
            target_path = target_base_path / spec_name

        # 如果目标目录不存在, 创建它
        if not target_path.exists():
            try:
                target_path.mkdir(parents=True, exist_ok=True)
                print(f"[INFO] 创建规格目录: {target_path}")
            except Exception as e:
                print(f"[ERROR] 创建规格目录失败: {e}")
                return False

        # 检查目标目录是否为空
        if not force and any(target_path.iterdir()):
            print(f"[ERROR] 规格目录不为空: {target_path}")
            print("使用 --force 参数强制初始化")
            return False

        # 获取模板结构
        template_structure = self.get_template_structure()

        if not template_structure:
            print("[ERROR] 模板目录结构为空")
            return False

        print(f"[INFO] 开始初始化规格 '{spec_name}' 到目录: {target_path}")
        print(f"[INFO] 模板来源:", self.template_dir)

        # 复制模板文件
        success = True
        copied_files = 0
        copied_dirs = 0

        def copy_template(src_path, dst_path, relative_path):
            """复制模板文件或目录"""
            nonlocal copied_files, copied_dirs, success

            try:
                if src_path.is_dir():
                    # 创建目标目录
                    dst_path.mkdir(parents=True, exist_ok=True)
                    copied_dirs += 1
                    print(f"[OK] 创建目录: {relative_path}")

                    # 递归复制子目录和文件
                    for item in src_path.iterdir():
                        new_relative_path = f"{relative_path}/{item.name}".lstrip('/')
                        copy_template(item, dst_path / item.name, new_relative_path)

                else:
                    # 复制文件
                    shutil.copy2(src_path, dst_path)
                    copied_files += 1
                    print(f"[OK] 复制文件: {relative_path}")

            except Exception as e:
                print(f"[ERROR] 复制 {relative_path} 失败: {e}")
                success = False

        # 开始复制模板
        for item_name, item_info in template_structure.items():
            src_path = item_info['path']
            dst_path = target_path / item_name
            copy_template(src_path, dst_path, item_name)

        # 处理规格名称参数化 (如果需要)
        self._process_spec_name_parameters(target_path, spec_name)

        # 输出结果
        if success:
            print(f"\n[SUCCESS] 规格初始化完成! ")
            print(f"创建目录: {copied_dirs} 个")
            print(f"复制文件: {copied_files} 个")
            print(f"规格位置: {target_path}")
            print(f"规格名称: {spec_name}")

            # 显示规格结构
            print("\n[规格结构]:")
            self._display_project_structure(target_path)

        else:
            print("\n[ERROR] 规格初始化过程中出现错误")

        return success

    def _process_spec_name_parameters(self, target_path: Path, spec_name: str):
        """处理规格名称参数化 (预留功能, 用于未来扩展) """
        # 当前版本中, 模板文件不包含参数化内容
        # 这个函数为未来扩展预留, 可以在模板文件中使用占位符, 然后在这里替换
        pass

    def _check_devagent_conflicts(self, src_dir: Path, target_dir: Path) -> bool:
        """检查.devagent目录复制时的文件冲突"""
        conflict_found = False

        def check_conflicts(src_path, dst_path):
            """递归检查冲突"""
            nonlocal conflict_found

            if conflict_found:
                return

            relative_path = src_path.relative_to(src_dir)

            if src_path.is_dir():
                # 检查目录
                if dst_path.exists() and not dst_path.is_dir():
                    print(f"[ERROR] 文件冲突: {relative_path} (目标位置已存在同名文件)")
                    conflict_found = True
                    return

                # 递归检查子目录
                for item in src_path.iterdir():
                    check_conflicts(item, dst_path / item.name)

            else:
                # 检查文件
                if dst_path.exists():
                    print(f"[ERROR] 文件冲突: {relative_path} (目标位置已存在同名文件)")
                    conflict_found = True
                    return

        # 检查根目录冲突
        if (target_dir / ".devagent").exists():
            print("[ERROR] 文件冲突: .devagent (目标位置已存在同名目录)")
            return False

        # 开始检查冲突
        check_conflicts(src_dir, target_dir / ".devagent")

        return not conflict_found

    def _display_project_structure(self, path: Path, prefix="", is_last=True):
        """显示项目结构树"""
        if path.is_file():
            return

        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))

        for i, item in enumerate(items):
            is_last_item = i == len(items) - 1

            if item.is_dir():
                print(f"{prefix}{'└── ' if is_last_item else '├── '}{item.name}/")
                new_prefix = prefix + ("    " if is_last_item else "│   ")
                self._display_project_structure(item, new_prefix, is_last_item)
            else:
                print(f"{prefix}{'└── ' if is_last_item else '├── '}{item.name}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Ralph规格模板初始化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 初始化到当前目录/docs/spec/spec-name, 规格名称为my_spec
  ralph_init --spec-name my_spec

  # 初始化到指定目录/spec_name
  ralph_init --spec-name my_spec --target-dir /path/to/specs

  # 强制初始化到非空目录
  ralph_init --spec-name my_spec --force

  # 复制.devagent目录到当前项目
  ralph_init --agent-setting devagent
        """
    )

    parser.add_argument(
        '--spec-name',
        help='规格名称 (必需参数, 除非使用--agent-setting或--list-templates)'
    )

    parser.add_argument(
        '--agent-setting',
        choices=['devagent'],
        help='agent设置类型 (当前仅支持devagent)'
    )

    parser.add_argument(
        '--target-dir',
        default='.',
        help='目标目录 (默认: 当前目录/docs/spec/spec-name, 指定目录: 指定目录/spec_name)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='强制初始化到非空目录'
    )

    parser.add_argument(
        '--list-templates',
        action='store_true',
        help='显示可用的模板结构'
    )

    args = parser.parse_args()

    initializer = RalphInitializer()

    if args.list_templates:
        # 显示模板结构
        if not initializer.check_template_exists():
            return 1

        print("=== 可用模板结构 ===")
        structure = initializer.get_template_structure()

        def print_structure(struct, indent=0):
            for name, info in struct.items():
                prefix = "  " * indent
                if info['type'] == 'directory':
                    print(f"{prefix}[DIR] {name}/")
                    print_structure(info.get('children', {}), indent + 1)
                else:
                    print(f"{prefix}[FILE] {name}")

        print_structure(structure)
        return 0

    # 检查必需参数 (除了list-templates)
    if not args.list_templates and not args.spec_name and not args.agent_setting:
        print("[ERROR] 必须提供规格名称参数 --spec-name 或 --agent-setting 参数")
        parser.print_help()
        return 1

    # 执行初始化
    if args.agent_setting:
        success = initializer.initialize_agent_setting(
            target_dir=args.target_dir,
            agent_type=args.agent_setting,
            force=args.force
        )
    else:
        success = initializer.initialize_spec(
            target_dir=args.target_dir,
            spec_name=args.spec_name,
            force=args.force
        )

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())