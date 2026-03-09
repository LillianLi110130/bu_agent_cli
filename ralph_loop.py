#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ralph_loop.py - DevAgent Task Processing Loop (Python版本)
根据plan.json配置文件自动处理任务队列

功能:
1. 读取任务配置文件 (plan.json)
2. 筛选未完成的任务并按优先级排序
3. 根据任务状态执行不同的处理逻辑
4. 更新任务状态和记录执行结果
5. 循环处理直到所有任务完成
"""

import os
import json
import time
import subprocess
import shutil
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

class Config:
    """配置常量类"""

    # 默认配置值
    MAX_RETRY_ATTEMPTS = 10
    LOG_DIR = "./docs/spec/[plan_name]/logs"
    PLAN_FILE = "./docs/spec/[plan_name]/plan/plan.json"
    MAIN_LOG_FILE = "implement.log"
    LAST_OUTPUT_LINES = 20
    LOOP_DELAY = 1
    LOG_LEVEL = "INFO"

    # Git分支配置
    MAIN_BRANCH = "main"  # 主分支名称
    WORK_BRANCH = "devagent-work"  # 工作分支名称
    ENABLE_GIT = False  # 是否启用Git分支管理

    # 静默模式配置
    SILENT_MODE = False  # 是否启用静默模式

    @classmethod
    def update_from_args(cls, args):
        """根据命令行参数更新配置"""
        if args.plan_file:
            cls.PLAN_FILE = args.plan_file
            # 如果通过plan_name设置了plan_file,则自动设置对应的log_dir
            if args.plan_name and not args.log_dir:
                # 从plan_file路径推断log_dir路径
                plan_file_dir = os.path.dirname(cls.PLAN_FILE)
                plan_dir = os.path.dirname(plan_file_dir)  # 获取plan目录的父目录
                cls.LOG_DIR = os.path.join(plan_dir, "logs")
        if args.log_dir:
            cls.LOG_DIR = args.log_dir
        if args.max_retry:
            cls.MAX_RETRY_ATTEMPTS = args.max_retry
        if args.delay:
            cls.LOOP_DELAY = args.delay
        if args.log_level:
            cls.LOG_LEVEL = args.log_level.upper()
        if args.main_branch:
            cls.MAIN_BRANCH = args.main_branch
        if args.work_branch:
            cls.WORK_BRANCH = args.work_branch
        if args.enable_git:
            cls.ENABLE_GIT = True
        if args.silent:
            cls.SILENT_MODE = True

        # 记录配置信息
        Logger.log_info(f"配置文件路径: {cls.PLAN_FILE}")
        Logger.log_info(f"日志目录: {cls.LOG_DIR}")


class Task:
    """任务数据类"""

    def __init__(self, task_data: Dict[str, Any]):
        self.task_name = task_data.get('task_name', '')
        self.priority = task_data.get('priority', 999)  # 默认低优先级
        self.status = task_data.get('status', 'TODO')
        self.implment_times = task_data.get('implment_times', 0)
        self.task_file = task_data.get('task_file', '')
        self.complete_time = task_data.get('complete_time', '')
        self.last_output = task_data.get('last_output', '')
        self.log_file = task_data.get('log_file', '')

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'task_name': self.task_name,
            'priority': self.priority,
            'status': self.status,
            'implment_times': self.implment_times,
            'task_file': self.task_file,
            'complete_time': self.complete_time,
            'last_output': self.last_output,
            'log_file': self.log_file
        }


class ConfigReader:
    """模块1: 配置文件读取与解析"""

    @staticmethod
    def read_plan_file() -> List[Dict[str, Any]]:
        """
        读取plan.json文件并解析为任务数组

        Returns:
            List[Dict[str, Any]]: 任务列表
        """
        if os.path.exists(Config.PLAN_FILE):
            try:
                with open(Config.PLAN_FILE, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        return json.loads(content)
            except (json.JSONDecodeError, IOError) as e:
                Logger.log_error(f"读取配置文件失败: {e}")

        # 文件不存在或读取失败时返回空列表
        return []

    @staticmethod
    def write_plan_file(tasks: List[Dict[str, Any]]) -> bool:
        """
        将任务列表写回plan.json文件

        Args:
            tasks: 任务列表

        Returns:
            bool: 是否成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(Config.PLAN_FILE), exist_ok=True)

            # 使用临时文件避免写入中断
            temp_file = Config.PLAN_FILE + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(tasks, f, indent=2, ensure_ascii=False)

            # 原子性替换
            shutil.move(temp_file, Config.PLAN_FILE)
            return True
        except (IOError, OSError) as e:
            Logger.log_error(f"写入配置文件失败: {e}")
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False


class TaskFilterSorter:
    """模块2: 任务筛选与排序"""

    @staticmethod
    def filter_and_sort_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        筛选status不为DONE的任务,按priority排序

        Args:
            tasks: 原始任务列表

        Returns:
            List[Dict[str, Any]]: 筛选排序后的任务列表
        """
        try:
            # 筛选非DONE状态的任务
            filtered_tasks = [
                task for task in tasks
                if task.get('status') != 'DONE'
            ]

            # 按优先级排序 (数值越小优先级越高)
            sorted_tasks = sorted(
                filtered_tasks,
                key=lambda x: x.get('priority', 999)
            )

            return sorted_tasks
        except Exception as e:
            Logger.log_error(f"任务筛选排序失败: {e}")
            return []


class TodoTaskHandler:
    """模块3: TODO任务处理"""

    def __init__(self, git_manager = None):
        self.git_manager = git_manager

    def handle_todo_task(self, task: Dict[str, Any]) -> tuple[bool, str]:
        """
        处理状态为TODO的任务

        Args:
            task: 任务数据

        Returns:
            tuple[bool, str]: (是否成功, 日志文件路径)
        """
        try:
            task_name = task.get('task_name', '')
            task_file = task.get('task_file', '')
            branch = task.get('branch', '')

            # 验证task_name
            if not task_name:
                Logger.log_error("任务task_name为空")
                return False, ""

            # 构建日志文件路径
            log_file = os.path.join(Config.LOG_DIR, f"{task_name}_implment.log")

            # 确保日志目录存在
            if not Logger.ensure_log_directory():
                Logger.log_error(f"无法创建日志目录: {Config.LOG_DIR}")
                return False, ""

            # 创建日志文件
            if not Logger.create_log_file(log_file):
                Logger.log_error(f"无法创建日志文件: {log_file}")
                return False, ""

            Logger.log_info(f"处理TODO任务: {task_name}")
            Logger.log_info(f"执行文件: {task_file}")
            Logger.log_info(f"日志文件: {log_file}")

            # 检查task_file是否存在
            if not os.path.exists(task_file):
                error_msg = f"文件不存在: {task_file}"
                Logger.log_error(error_msg)
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(error_msg)
                return False, log_file

            # 如果启用了Git分支管理，创建子工作分支
            if self.git_manager and Config.ENABLE_GIT:
                if not self.git_manager.create_sub_work_branch(branch):
                    Logger.log_error("创建子工作分支失败,跳过任务执行")
                    return False, log_file

            # 执行devagent命令
            success = TodoTaskHandler._execute_devagent(task, task_file, log_file)

            return success, log_file

        except Exception as e:
            Logger.log_error(f"处理TODO任务时发生异常: {e}")
            # 如果启用了Git分支管理，清理子工作分支
            if self.git_manager and Config.ENABLE_GIT:
                self.git_manager.cleanup_sub_work_branch()
            return False, ""

    @staticmethod
    def _execute_devagent(task: Dict[str, Any], task_file: str, log_file: str) -> bool:
        """
        执行devagent命令

        Args:
            task: 任务数据
            task_file: 任务文件路径
            log_file: 日志文件路径

        Returns:
            bool: 是否执行成功
        """
        try:
            # 检查devagent命令是否可用
            if shutil.which('devagent') is None:
                error_msg = "devagent命令未找到"
                Logger.log_error(error_msg)
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(error_msg)
                return False

            # 读取implement.md模板文件
            implement_template_path = ".devagent/commands/ralph/implement.md"
            if not os.path.exists(implement_template_path):
                error_msg = f"implement.md模板文件不存在: {implement_template_path}"
                Logger.log_error(error_msg)
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(error_msg)
                return False

            with open(implement_template_path, 'r', encoding='utf-8') as f:
                implement_content = f.read()

            # 更新implement.md中的占位符信息
            implement_content = TodoTaskHandler._update_implement_placeholders(
                implement_content, task, task_file, log_file
            )

            # 记录开始执行时间
            start_time = time.time()
            Logger.log_info(f"开始执行devagent命令, 使用implement.md模板")
            Logger.log_debug(f"输入内容长度: {len(implement_content)} 字符")

            # 执行devagent命令, 使用shell=True确保更好的兼容性
            process = subprocess.Popen(
                'devagent --yolo',
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )

            # 写入输入并获取输出
            stdout, stderr = process.communicate(input=implement_content)

            # 记录执行时间
            execution_time = time.time() - start_time

            # 写入日志文件, 包含执行信息
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== DevAgent执行日志 ===\n")
                f.write(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"输入模板: {implement_template_path}\n")
                f.write(f"任务文件: {task_file}\n")
                f.write(f"执行耗时: {Logger.format_duration(execution_time)}\n")
                f.write(f"返回码: {process.returncode}\n")
                f.write(f"=== 标准输出 ===\n")
                f.write(stdout if stdout else "(无输出)")

                if stderr:
                    f.write(f"\n=== 标准错误 ===\n")
                    f.write(stderr)

            # 记录执行结果
            if process.returncode == 0:
                Logger.log_info(f"devagent命令执行成功 (耗时: {Logger.format_duration(execution_time)})")
                Logger.log_debug(f"输出长度: {len(stdout)} 字符")
                return True
            else:
                Logger.log_error(f"devagent命令执行失败 (返回码: {process.returncode}, 耗时: {Logger.format_duration(execution_time)})")
                if stderr:
                    Logger.log_error(f"错误输出: {stderr.strip()}")
                if stdout:
                    Logger.log_debug(f"标准输出: {stdout.strip()}")
                return False

        except Exception as e:
            error_msg = f"执行devagent命令失败: {e}"
            Logger.log_error(error_msg)
            try:
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== 执行异常 ===\n")
                    f.write(f"异常时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"异常信息: {error_msg}\n")
            except:
                pass
            return False

    @staticmethod
    def _update_implement_placeholders(implement_content: str, task: Dict[str, Any], task_file: str, log_file: str) -> str:
        """
        更新implement.md中的占位符信息

        Args:
            implement_content: implement.md模板内容
            task: 任务数据
            task_file: 任务文件路径
            log_file: 日志文件路径

        Returns:
            str: 更新后的内容
        """
        try:
            task_name = task.get('task_name', '')

            # 获取基础路径信息
            plan_file_dir = os.path.dirname(Config.PLAN_FILE)
            plan_dir = os.path.dirname(plan_file_dir)  # plan目录的父目录
            implement_dir = os.path.join(plan_dir, "implement")

            # 更新占位符
            implement_content = implement_content.replace("{#implement_dir}", implement_dir)
            implement_content = implement_content.replace("{#plan_dir}", plan_dir)
            implement_content = implement_content.replace("{#task_file}", task_file)
            implement_content = implement_content.replace("{#task_name}", task_name)

            Logger.log_debug(f"更新implement.md占位符: task_name={task_name}, implement_dir={implement_dir}")
            return implement_content

        except Exception as e:
            Logger.log_error(f"更新implement.md占位符失败: {e}")
            return implement_content  # 返回原始内容


class FailedTaskHandler:
    """模块4: FAILED任务处理"""

    def __init__(self, git_manager = None):
        self.git_manager = git_manager

    def handle_failed_task(self, task: Dict[str, Any]) -> tuple[bool, str]:
        """
        处理状态为FAILED的任务

        Args:
            task: 任务数据

        Returns:
            tuple[bool, str]: (是否成功, 日志文件路径)
        """
        try:
            task_name = task.get('task_name', '')
            task_file = task.get('task_file', '')
            count = task.get('implment_times', 0)
            branch = task.get('branch', '')

            # 验证task_name
            if not task_name:
                Logger.log_error("任务task_name为空")
                return False, ""

            # 检查重试次数
            if count >= Config.MAX_RETRY_ATTEMPTS:
                error_msg = f"任务 {task_name} 已达到最大重试次数 ({Config.MAX_RETRY_ATTEMPTS})"
                Logger.log_error(error_msg)
                return False, ""

            # 确定新的重试次数
            new_count = count + 1
            log_file = os.path.join(Config.LOG_DIR, f"{task_name}_implment_{new_count}.log")

            # 确保日志目录存在
            if not Logger.ensure_log_directory():
                Logger.log_error(f"无法创建日志目录: {Config.LOG_DIR}")
                return False, ""

            # 创建日志文件
            if not Logger.create_log_file(log_file):
                Logger.log_error(f"无法创建日志文件: {log_file}")
                return False, ""

            Logger.log_info(f"重试FAILED任务: {task_name} (第{new_count}次尝试)")
            Logger.log_info(f"执行文件: {task_file}")
            Logger.log_info(f"日志文件: {log_file}")

            # 检查task_file是否存在
            if not os.path.exists(task_file):
                error_msg = f"文件不存在: {task_file}"
                Logger.log_error(error_msg)
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(error_msg)
                return False, log_file

            # 如果启用了Git分支管理，创建子工作分支
            if self.git_manager and Config.ENABLE_GIT:
                if not self.git_manager.create_sub_work_branch(f"{branch}_retry{new_count}"):
                    Logger.log_error("创建子工作分支失败，跳过任务执行")
                    return False, log_file

            # 执行devagent命令
            success = FailedTaskHandler._execute_devagent(task, task_file, log_file)

            return success, log_file

        except Exception as e:
            Logger.log_error(f"处理FAILED任务时发生异常: {e}")
            # 如果启用了Git分支管理，清理子工作分支
            if self.git_manager and Config.ENABLE_GIT:
                self.git_manager.cleanup_sub_work_branch()
            return False, ""

    @staticmethod
    def _execute_devagent(task: Dict[str, Any], task_file: str, log_file: str) -> bool:
        """
        执行devagent命令 (复用TODO任务的处理逻辑)
        """
        return TodoTaskHandler._execute_devagent(task, task_file, log_file)


class TaskStatusUpdater:
    """模块5: 任务状态更新"""

    def __init__(self, git_manager = None):
        self.git_manager = git_manager

    def update_task_status(self, task: Dict[str, Any], success: bool, log_file: str) -> bool:
        """
        根据执行结果更新任务状态

        Args:
            task: 原始任务数据
            success: 是否执行成功
            log_file: 日志文件路径

        Returns:
            bool: 是否更新成功
        """
        try:
            # 读取当前任务列表
            tasks = ConfigReader.read_plan_file()

            # 获取last_output内容
            last_output = TaskStatusUpdater._get_last_output(log_file)

            # 更新时间戳
            complete_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 更新任务状态
            updated = False
            for t in tasks:
                if t.get('task_name') == task.get('task_name'):
                    t['implment_times'] = task.get('implment_times', 0) + 1
                    t['complete_time'] = complete_time
                    t['last_output'] = last_output
                    t['log_file'] = log_file
                    t['status'] = 'DONE' if success else 'FAILED'
                    updated = True
                    break

            if not updated:
                Logger.log_warning(f"未找到任务: {task.get('task_name')}")
                return False

            # 如果启用了Git分支管理且任务成功，合并子工作分支
            if success and self.git_manager and Config.ENABLE_GIT:
                merge_success = self.git_manager.merge_sub_work_branch()
                if not merge_success:
                    Logger.log_error("合并子工作分支失败")
                    # 合并失败时清理子工作分支
                    self.git_manager.cleanup_sub_work_branch()
                    return False
            elif not success and self.git_manager and Config.ENABLE_GIT:
                # 任务失败时清理子工作分支
                self.git_manager.cleanup_sub_work_branch()

            # 写回文件
            return ConfigReader.write_plan_file(tasks)

        except Exception as e:
            Logger.log_error(f"更新任务状态失败: {e}")
            # 发生异常时清理子工作分支
            if self.git_manager and Config.ENABLE_GIT:
                self.git_manager.cleanup_sub_work_branch()
            return False

    @staticmethod
    def _get_last_output(log_file: str) -> str:
        """
        获取日志文件最后几行内容

        Args:
            log_file: 日志文件路径

        Returns:
            str: 最后几行内容
        """
        if not log_file or not os.path.exists(log_file):
            return ""

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 获取最后n行
            last_lines = lines[-Config.LAST_OUTPUT_LINES:]
            return ''.join(last_lines).strip()
        except Exception as e:
            Logger.log_error(f"读取日志文件失败: {e}")
            return ""


class Logger:
    """模块6: 日志管理"""

    # 日志级别
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"

    # 当前日志级别，从配置读取
    @classmethod
    def _get_log_level(cls):
        return getattr(Config, 'LOG_LEVEL', cls.INFO)

    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        将秒数格式化为 时:分:秒 格式

        Args:
            seconds: 秒数

        Returns:
            str: 格式化后的时间字符串
        """
        try:
            seconds = float(seconds)
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60

            if hours > 0:
                return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
            elif minutes > 0:
                return f"{minutes:02d}:{secs:06.3f}"
            else:
                return f"{secs:.3f}秒"
        except (ValueError, TypeError):
            return f"{seconds}秒"

    @staticmethod
    def ensure_log_directory() -> bool:
        """确保日志目录存在"""
        try:
            os.makedirs(Config.LOG_DIR, exist_ok=True)
            return True
        except OSError as e:
            print(f"ERROR: 无法创建日志目录 {Config.LOG_DIR}: {e}")
            return False

    @staticmethod
    def create_log_file(log_file: str) -> bool:
        """创建日志文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # 创建文件
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('')  # 创建空文件
            return True
        except OSError as e:
            print(f"ERROR: 无法创建日志文件 {log_file}: {e}")
            return False

    @staticmethod
    def _should_log(level: str) -> bool:
        """检查是否应该记录该级别的日志"""
        level_priority = {
            Logger.DEBUG: 1,
            Logger.INFO: 2,
            Logger.WARN: 3,
            Logger.ERROR: 4
        }

        current_log_level = Logger._get_log_level()
        current_priority = level_priority.get(current_log_level, 2)
        message_priority = level_priority.get(level, 2)

        return message_priority >= current_priority

    @staticmethod
    def log_message(level: str, message: str):
        """记录日志消息"""
        if not Logger._should_log(level):
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}]: {message}"

        # 静默模式处理: 只显示进度信息, 其他日志只写入文件
        if Config.SILENT_MODE:
            # 在静默模式下, 只输出进度相关的信息
            if level == Logger.INFO and any(keyword in message.lower() for keyword in 
                                            ['处理任务', '任务完成', '进度', '开始处理', '所有任务已完成', '第', '轮处理']):
                # 简化进度信息显示
                if '第' in message and '轮处理' in message:
                    # 显示轮次信息
                    print(f"\r{message}", end='', flush=True)
                elif '处理任务' in message or '任务完成' in message:
                    # 显示任务处理状态
                    print(f"\r{message}", end='', flush=True)
                elif '所有任务已完成' in message:
                    print(f"\n{message}")
                else:
                    # 其他进度信息正常显示
                    print(message)
            elif level == Logger.ERROR:
                # 错误信息在静默模式下也显示
                print(f"\nERROR: {message}", file=sys.stderr)
        else:
            # 非静默模式: 正常输出所有日志
            if level == Logger.ERROR:
                print(log_entry, file=sys.stderr)
            else:
                print(log_entry)

        # 写入主日志文件
        main_log_file = os.path.join(Config.LOG_DIR, Config.MAIN_LOG_FILE)
        try:
            with open(main_log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
        except OSError as e:
            print(f"ERROR: 无法写入日志文件 {main_log_file}: {e}")

    @staticmethod
    def log_debug(message: str):
        """记录调试日志"""
        Logger.log_message(Logger.DEBUG, message)

    @staticmethod
    def log_info(message: str):
        """记录信息日志"""
        Logger.log_message(Logger.INFO, message)

    @staticmethod
    def log_warn(message: str):
        """记录警告日志"""
        Logger.log_message(Logger.WARN, message)

    @staticmethod
    def log_error(message: str):
        """记录错误日志"""
        Logger.log_message(Logger.ERROR, message)


class ProgressDisplay:
    """进度显示类"""

    def __init__(self):
        self.total_tasks = 0
        self.completed_tasks = 0
        self.current_task = ""
        self.start_time = None

    def initialize(self, total_tasks: int):
        """初始化进度显示"""
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.current_task = ""
        self.start_time = time.time()

        if Config.SILENT_MODE and total_tasks > 0:
            print(f"总任务数: {total_tasks}")

    def update_task(self, task_name: str):
        """更新当前任务"""
        self.current_task = task_name
        if Config.SILENT_MODE:
            print(f"\r正在处理: {task_name}", end='', flush=True)

    def complete_task(self, success: bool = True):
        """标记任务完成"""
        self.completed_tasks += 1

        if Config.SILENT_MODE and self.total_tasks > 0:
            progress = (self.completed_tasks / self.total_tasks) * 100
            status = "√" if success else "X"
            print(f"\r{status} 进度: {progress:.1f}% ({self.completed_tasks}/{self.total_tasks})", end='', flush=True)

    def finish(self):
        """完成所有任务, 显示最终结果"""
        if Config.SILENT_MODE:
            if self.total_tasks > 0:
                elapsed_time = time.time() - self.start_time
                print(f"\n所有任务已完成! 总耗时: {Logger.format_duration(elapsed_time)}")
            else:
                print("\n没有任务需要处理")


class GitBranchManager:
    """Git分支管理类"""

    def __init__(self):
        self.current_branch = None
        self.sub_work_branch = None
        self.original_branch = None

    def check_git_available(self) -> bool:
        """检查Git是否可用"""
        try:
            result = subprocess.run(
                ['git', '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            Logger.log_info(f"Git可用: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            Logger.log_error("Git不可用，请确保Git已安装并配置到PATH环境变量中")
            return False

    def get_current_branch(self) -> str:
        """获取当前分支"""
        try:
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            Logger.log_error("获取当前分支失败")
            return ""

    def ensure_work_branch_exists(self) -> bool:
        """确保工作分支存在，如果不存在则基于主分支创建"""
        try:
            # 检查工作分支是否存在
            result = subprocess.run(
                ['git', 'branch', '--list', Config.WORK_BRANCH],
                capture_output=True,
                text=True,
                check=True
            )

            if Config.WORK_BRANCH in result.stdout:
                # 工作分支已存在，切换到工作分支
                Logger.log_info(f"工作分支 {Config.WORK_BRANCH} 已存在")
                return self.switch_to_branch(Config.WORK_BRANCH)
            else:
                # 工作分支不存在，基于主分支创建
                Logger.log_info(f"创建工作分支 {Config.WORK_BRANCH} 基于 {Config.MAIN_BRANCH}")
                
                # 切换到主分支
                if not self.switch_to_branch(Config.MAIN_BRANCH):
                    return False

                # 创建并切换到工作分支
                subprocess.run(
                    ['git', 'checkout', '-b', Config.WORK_BRANCH],
                    check=True
                )
                Logger.log_info(f"成功创建并切换到工作分支 {Config.WORK_BRANCH}")
                return True

        except subprocess.CalledProcessError as e:
            Logger.log_error(f"确保工作分支存在失败: {e}")
            return False

    def switch_to_branch(self, branch_name: str) -> bool:
        """切换到指定分支"""
        try:
            subprocess.run(
                ['git', 'checkout', branch_name],
                check=True
            )
            Logger.log_info(f"切换到分支 {branch_name}")
            return True
        except subprocess.CalledProcessError as e:
            Logger.log_error(f"切换到分支 {branch_name} 失败: {e}")
            return False

    def create_sub_work_branch(self, task_name: str) -> bool:
        """基于工作分支创建子工作分支"""
        try:
            # 生成子工作分支名称
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            branch_name = f"{Config.WORK_BRANCH}-{task_name}-{timestamp}"

            # 保存当前分支
            self.original_branch = self.get_current_branch()

            # 确保在工作分支上
            if not self.switch_to_branch(Config.WORK_BRANCH):
                return False

            # 创建子工作分支
            subprocess.run(
                ['git', 'checkout', '-b', branch_name],
                check=True
            )

            self.sub_work_branch = branch_name
            Logger.log_info(f"创建子工作分支 {branch_name}")
            return True

        except subprocess.CalledProcessError as e:
            Logger.log_error(f"创建子工作分支失败: {e}")
            return False

    def merge_sub_work_branch(self) -> bool:
        """将子工作分支合并回工作分支"""
        if not self.sub_work_branch:
            Logger.log_error("没有活动的子工作分支")
            return False

        try:
            # 切换到工作分支
            if not self.switch_to_branch(Config.WORK_BRANCH):
                return False

            # 合并子工作分支
            subprocess.run(
                ['git', 'merge', self.sub_work_branch, '--no-ff', '-m', f"Merge {self.sub_work_branch}"],
                check=True
            )

            Logger.log_info(f"成功将子工作分支 {self.sub_work_branch} 合并回工作分支")

            # 删除子工作分支
            subprocess.run(
                ['git', 'branch', '-d', self.sub_work_branch],
                check=True
            )
            Logger.log_info(f"删除子工作分支 {self.sub_work_branch}")

            self.sub_work_branch = None
            return True

        except subprocess.CalledProcessError as e:
            Logger.log_error(f"合并子工作分支失败: {e}")
            return False

    def cleanup_sub_work_branch(self) -> bool:
        """清理子工作分支 (在任务失败时调用) """
        if not self.sub_work_branch:
            return True

        try:
            # 切换到工作分支
            self.switch_to_branch(Config.WORK_BRANCH)

            # 强制删除子工作分支
            subprocess.run(
                ['git', 'branch', '-D', self.sub_work_branch],
                check=True
            )

            Logger.log_info(f"清理子工作分支 {self.sub_work_branch}")
            self.sub_work_branch = None
            return True

        except subprocess.CalledProcessError as e:
            Logger.log_error(f"清理子工作分支失败: {e}")
            return False

    def initialize_git_workflow(self) -> bool:
        """初始化Git工作流"""
        if not Config.ENABLE_GIT:
            Logger.log_info("Git分支管理未启用")
            return True

        if not self.check_git_available():
            Logger.log_error("Git不可用，跳过Git分支管理")
            return False

        return self.ensure_work_branch_exists()


class DependencyChecker:
    """依赖检查器"""

    @staticmethod
    def check_dependencies() -> bool:
        """检查必要的依赖工具是否可用"""
        # 检查devagent命令
        if shutil.which('devagent') is None:
            Logger.log_error("devagent命令未找到")
            Logger.log_error("请确保devagent已正确安装并配置到PATH环境变量中")
            return False

        Logger.log_info("所有依赖检查通过")
        return True


class RalphLoop:
    """主循环处理类"""

    def __init__(self):
        self.iteration = 0
        self.git_manager = GitBranchManager()
        self.todo_handler = TodoTaskHandler(self.git_manager)
        self.failed_handler = FailedTaskHandler(self.git_manager)
        self.status_updater = TaskStatusUpdater(self.git_manager)
        self.progress_display = ProgressDisplay()

    def dry_run(self) -> int:
        """干运行模式: 显示任务计划但不实际执行"""
        print("=== 干运行模式: 任务执行计划 ===")

        # 读取任务列表
        tasks = ConfigReader.read_plan_file()

        if not tasks:
            print("没有发现任何任务")
            return 0

        # 筛选和排序任务
        pending_tasks = TaskFilterSorter.filter_and_sort_tasks(tasks)

        print(f"总任务数: {len(tasks)}")
        print(f"待处理任务数: {len(pending_tasks)}")
        print()

        if pending_tasks:
            print("=== 待处理任务列表 (按优先级排序) ===")
            for i, task in enumerate(pending_tasks, 1):
                task_name = task.get('task_name', '未知任务')
                priority = task.get('priority', 999)
                status = task.get('status', 'UNKNOWN')
                implment_times = task.get('implment_times', 0)
                task_file = task.get('task_file', '')

                print(f"{i}. {task_name}")
                print(f"    优先级: {priority}")
                print(f"    状态: {status}")
                print(f"    执行次数: {implment_times}")
                print(f"    执行文件: {task_file}")

                if status == 'FAILED' and implment_times >= Config.MAX_RETRY_ATTEMPTS:
                    print(f"    ⚠ 已达到最大重试次数 ({Config.MAX_RETRY_ATTEMPTS})")

                print()

        # 显示已完成的任务
        done_tasks = [task for task in tasks if task.get('status') == 'DONE']
        if done_tasks:
            print(f"=== 已完成任务 ({len(done_tasks)}个) ===")
            for task in done_tasks:
                task_name = task.get('task_name', '未知任务')
                complete_time = task.get('complete_time', '')
                print(f"  √ {task_name} - 完成时间: {complete_time}")

        return 0

    def main_loop(self) -> bool:
        """主循环逻辑"""
        Logger.log_info("开始处理任务队列...")

        # 初始化Git工作流
        if Config.ENABLE_GIT:
            if not self.git_manager.initialize_git_workflow():
                Logger.log_error("Git工作流初始化失败，但将继续处理任务")

        # 初始化进度显示
        initial_tasks = ConfigReader.read_plan_file()
        initial_pending_tasks = TaskFilterSorter.filter_and_sort_tasks(initial_tasks)
        self.progress_display.initialize(len(initial_pending_tasks))

        while True:
            try:
                self.iteration += 1
                Logger.log_info(f"=== 第{self.iteration}轮处理 ===")

                # 读取任务列表
                tasks = ConfigReader.read_plan_file()

                # 筛选和排序任务
                pending_tasks = TaskFilterSorter.filter_and_sort_tasks(tasks)

                # 检查是否有待处理任务
                if not pending_tasks:
                    Logger.log_info("所有任务已完成!")
                    self.progress_display.finish()
                    break

                Logger.log_info(f"发现 {len(pending_tasks)} 个待处理任务")

                # 获取第一个任务
                current_task = pending_tasks[0]
                task_name = current_task.get('task_name', '未知任务')
                status = current_task.get('status', 'UNKNOWN')
                implment_times = current_task.get('implment_times', 0)

                Logger.log_info(f"处理任务: {task_name} (状态: {status}, 尝试次数: {implment_times})")

                # 更新进度显示
                self.progress_display.update_task(task_name)

                # 检查是否超过最大重试次数
                if status == 'FAILED' and implment_times >= Config.MAX_RETRY_ATTEMPTS:
                    Logger.log_error(f"任务 {task_name} 已达到最大重试次数 ({Config.MAX_RETRY_ATTEMPTS}), 跳过处理")
                    self.progress_display.complete_task(False)
                    # 跳过此任务, 继续处理下一个 -> 结束任务
                    break

                # 根据状态处理任务
                success = False
                log_file = ""

                if status == 'TODO':
                    success, log_file = self.todo_handler.handle_todo_task(current_task)
                elif status == 'FAILED':
                    success, log_file = self.failed_handler.handle_failed_task(current_task)
                else:
                    Logger.log_error(f"未知任务状态: {status}")
                    continue

                # 更新任务状态
                if self.status_updater.update_task_status(current_task, success, log_file):
                    if success:
                        Logger.log_info(f"任务 {task_name} 执行成功")
                        self.progress_display.complete_task(True)
                    else:
                        Logger.log_error(f"任务 {task_name} 执行失败")
                        self.progress_display.complete_task(False)
                else:
                    Logger.log_error(f"更新任务 {task_name} 状态失败")
                    self.progress_display.complete_task(False)

                # 短暂延迟，避免过快循环
                time.sleep(Config.LOOP_DELAY)

            except KeyboardInterrupt:
                Logger.log_info("用户中断执行")
                # 中断时清理Git分支
                if Config.ENABLE_GIT:
                    self.git_manager.cleanup_sub_work_branch()
                break
            except Exception as e:
                Logger.log_error(f"处理任务时发生异常: {e}")
                # 异常时清理Git分支
                if Config.ENABLE_GIT:
                    self.git_manager.cleanup_sub_work_branch()
                # 短暂延迟后继续处理下一个任务
                time.sleep(Config.LOOP_DELAY)
                continue

        return True


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='DevAgent任务处理循环')
    parser.add_argument('plan_name', nargs='?', type=str, help='任务配置名称，对应 docs/spec/[plan_name]/plan/plan.json 文件')
    parser.add_argument('--plan-file', type=str, help='任务配置文件路径 (如果指定此参数，则忽略plan_name参数)')
    parser.add_argument('--log-dir', type=str, help='日志文件目录')
    parser.add_argument('--max-retry', type=int, help='最大重试次数')
    parser.add_argument('--delay', type=float, help='循环延迟时间 (秒)')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARN', 'ERROR'], help='日志级别')
    parser.add_argument('--dry-run', action='store_true', help='干运行模式，不实际执行任务')
    parser.add_argument('--enable-git', action='store_true', help='启用Git分支管理')
    parser.add_argument('--main-branch', type=str, help='主分支名称 (默认: main)')
    parser.add_argument('--work-branch', type=str, help='工作分支名称 (默认: devagent-work)')
    parser.add_argument('--silent', action='store_true', help='静默模式，终端只显示进度')

    args = parser.parse_args()

    # 处理plan_name参数: 如果指定了plan_name且没有指定--plan-file，则构建配置文件路径
    if args.plan_name and not args.plan_file:
        # 构建配置文件路径: 当前目录/docs/spec/plan_name/plan/plan.json
        current_dir = os.getcwd()
        plan_file_path = os.path.join(current_dir, "docs", "spec", args.plan_name, "plan", "plan.json")
        args.plan_file = plan_file_path

        # 检查文件是否存在
        if not os.path.exists(plan_file_path):
            Logger.log_error(f"配置文件不存在: {plan_file_path}")
            Logger.log_error("请确保指定的plan_name对应的配置文件存在")
            return 1
    elif not args.plan_name and not args.plan_file:
        # 如果没有指定任何plan相关参数，显示帮助信息
        parser.print_help()
        Logger.log_error("必须指定plan_name或--plan-file参数")
        return 1

    # 根据命令行参数更新配置
    Config.update_from_args(args)

    # 静默模式下显示简洁信息，非静默模式显示完整信息
    if Config.SILENT_MODE:
        print("DevAgent任务处理循环 - 静默模式")
        if args.dry_run:
            print("模式: 干运行")
        print()
    else:
        print("=== DevAgent任务处理循环 (Python版本) ===")
        print(f"配置文件: {Config.PLAN_FILE}")
        print(f"日志目录: {Config.LOG_DIR}")
        print(f"最大重试次数: {Config.MAX_RETRY_ATTEMPTS}")
        print(f"日志级别: {Config.LOG_LEVEL}")
        print(f"Git分支管理: {'启用' if Config.ENABLE_GIT else '禁用'}")
        if Config.ENABLE_GIT:
            print(f"主分支: {Config.MAIN_BRANCH}")
            print(f"工作分支: {Config.WORK_BRANCH}")
        if args.dry_run:
            print("模式: 干运行 (不实际执行任务)")
        print()

    # 检查依赖
    if not DependencyChecker.check_dependencies():
        return 1

    # 检查配置文件是否存在
    if not os.path.exists(Config.PLAN_FILE):
        Logger.log_warning(f"配置文件 {Config.PLAN_FILE} 不存在, 创建空配置")
        if not ConfigReader.write_plan_file([]):
            Logger.log_error("创建空配置文件失败")
            return 1

    # 运行主循环
    loop = RalphLoop()
    try:
        if args.dry_run:
            # 干运行模式: 只显示任务计划，不实际执行
            return loop.dry_run()
        else:
            success = loop.main_loop()
            return 0 if success else 1
    except KeyboardInterrupt:
        Logger.log_info("用户中断执行")
        return 0
    except Exception as e:
        Logger.log_error(f"主循环执行失败: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())