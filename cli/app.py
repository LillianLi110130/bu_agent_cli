"""CLI Application for Claude Code.

Contains the main ClaudeCodeCLI class and loading indicator.
Pure UI logic - receives pre-configured Agent and context.
"""

import json
import os
import sys
import threading
import time
import hashlib
from pathlib import Path
from typing import Any, Callable

from bu_agent_sdk import Agent
from bu_agent_sdk.agent import (
    FinalResponseEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from bu_agent_sdk.agent.registry import AgentRegistry
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.llm.messages import (
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    SystemMessage,
    UserMessage,
)
from bu_agent_sdk.plugin import PluginManager
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import ThreadedCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers import BashLexer
from rich.console import Console
from rich.panel import Panel

from cli.slash_commands import (
    SlashCommand,
    SlashCommandCompleter,
    SlashCommandRegistry,
    is_slash_command,
    parse_slash_command,
)
from cli.at_commands import (
    AtCommandCompleter,
    AtCommandRegistry,
    expand_at_command,
    is_at_command,
    parse_at_command,
)
from cli.image_input import (
    IMAGE_USAGE,
    ImageInputError,
    is_image_command,
    parse_image_command,
)
from cli.plugins_handler import PluginSlashHandler
from cli.ralph_commands import RalphSlashHandler
from tools import SandboxContext, SecurityError

ModelPreset = dict[str, str | bool]
UserInputPayload = str | list[ContentPartTextParam | ContentPartImageParam]


# =============================================================================
# Loading Indicator
# =============================================================================


class _LoadingIndicator:
    """A simple loading indicator using direct stdout with ANSI codes."""

    def __init__(self, message: str = "Thinking"):
        self.message = message
        self._stop_event = threading.Event()
        self._thread = None
        self._frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def _show_frame(self, frame: int):
        """Show a single frame."""
        sys.stdout.write(f"\r\033[36m{self._frames[frame]}\033[0m {self.message}...")
        sys.stdout.flush()

    def _clear(self):
        """Clear the loading line."""
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def start(self):
        """Start the loading animation in a separate thread."""
        self._stop_event.clear()
        self._show_frame(0)

        def _run():
            frame = 1
            while not self._stop_event.is_set():
                self._show_frame(frame % len(self._frames))
                frame += 1
                time.sleep(0.08)
            self._clear()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the loading animation."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        self._clear()
        self._clear()
        sys.stdout.write("\n")
        sys.stdout.flush()


# =============================================================================
# CLI Application
# =============================================================================


class ClaudeCodeCLI:
    """Interactive CLI for Claude Code assistant.

    Pure UI class - displays agent events and handles user input.
    """

    # Color scheme
    COLOR_TOOL_CALL = "bright_blue"
    COLOR_TOOL_RESULT = "green"
    COLOR_ERROR = "red"
    COLOR_THINKING = "dim cyan"
    COLOR_FINAL = "bold green"
    IMAGE_DETAIL_MAX_CHARS = 2200
    IMAGE_MEMORY_MAX_CHARS = 600

    def __init__(
        self,
        agent: Agent,
        context: SandboxContext,
        *,
        slash_registry: SlashCommandRegistry | None = None,
        at_registry: AtCommandRegistry | None = None,
        agent_registry: AgentRegistry | None = None,
        plugin_manager: PluginManager | None = None,
        system_prompt_builder: Callable[[], str] | None = None,
    ):
        """Initialize CLI with pre-configured agent and context.

        Args:
            agent: Configured Agent instance
            context: SandboxContext for the session
        """
        self._console = Console()
        self._agent = agent
        self._ctx = context
        self._step_number = 0
        self._loading: _LoadingIndicator | None = None
        self._slash_registry = slash_registry or SlashCommandRegistry()
        self._at_registry = at_registry or AtCommandRegistry(
            Path(__file__).resolve().parent.parent / "bu_agent_sdk" / "skills"
        )
        self._agent_registry = agent_registry
        self._plugin_manager = plugin_manager
        self._system_prompt_builder = system_prompt_builder
        self._model_presets_path = (
            Path(__file__).resolve().parent.parent
            / "config"
            / "model_presets.json"
        )
        self._default_model_preset: str | None = None
        self._auto_vision_preset: str | None = None
        self._image_summary_preset: str | None = None
        self._model_presets = self._load_model_presets()
        self._sticky_preset: str | None = self._resolve_current_preset_name()
        self._auto_switched: bool = False
        self._auto_from_preset: str | None = None
        self._model_pick_active = False
        self._model_pick_order: list[str] = []
        self._agents_md_hash: str | None = None
        self._agents_md_content: str | None = None
        self._ralph_handler: RalphSlashHandler | None = None

        if context.subagent_manager:
            context.subagent_manager.set_result_callback(self._on_task_completed)

    def _load_model_presets(self) -> dict[str, ModelPreset]:
        """Load model presets from config/model_presets.json."""
        if not self._model_presets_path.exists():
            return {}

        try:
            raw = self._model_presets_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as e:
            self._console.print(
                f"[yellow]Failed to load model presets: {e}[/yellow]"
            )
            return {}

        if not isinstance(data, dict):
            self._console.print(
                "[yellow]model_presets.json must be a JSON object.[/yellow]"
            )
            return {}

        default_name = data.get("default")
        if isinstance(default_name, str) and default_name.strip():
            self._default_model_preset = default_name.strip()
        
        auto_vision_name = data.get("auto_vision_preset")
        if isinstance(auto_vision_name, str) and auto_vision_name.strip():
            self._auto_vision_preset = auto_vision_name.strip()

        image_summary_name = data.get("image_summary_preset")
        if isinstance(image_summary_name, str) and image_summary_name.strip():
            self._image_summary_preset = image_summary_name.strip()

        preset_data = data.get("presets")
        if not isinstance(preset_data, dict):
            return {}

        presets: dict[str, ModelPreset] = {}
        for name, config in preset_data.items():
            if not isinstance(name, str) or not isinstance(config, dict):
                continue

            model = config.get("model")
            if not isinstance(model, str) or not model.strip():
                continue

            cleaned: ModelPreset = {"model": model.strip()}

            base_url = config.get("base_url")
            if isinstance(base_url, str) and base_url.strip():
                cleaned["base_url"] = base_url.strip()

            api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
            if isinstance(api_key_env, str) and api_key_env.strip():
                cleaned["api_key_env"] = api_key_env.strip()
            else:
                cleaned["api_key_env"] = "OPENAI_API_KEY"

            cleaned["vision"] = bool(config.get("vision", False))
            presets[name.strip()] = cleaned

        return presets

    def _clear_auto_switch_state(self) -> None:
        self._auto_switched = False
        self._auto_from_preset = None

    def _preset_supports_vision(self, preset_name: str | None) -> bool:
        if not preset_name:
            return False
        preset = self._model_presets.get(preset_name)
        if not preset:
            return False
        return bool(preset.get("vision", False))

    def _resolve_vision_preset_name(self) -> str | None:
        if (
            self._auto_vision_preset
            and self._auto_vision_preset in self._model_presets
            and self._preset_supports_vision(self._auto_vision_preset)
        ):
            return self._auto_vision_preset
        for name in self._model_presets.keys():
            if self._preset_supports_vision(name):
                return name
        return None

    def _resolve_image_summary_preset_name(self) -> str | None:
        if (
            self._image_summary_preset
            and self._image_summary_preset in self._model_presets
            and self._preset_supports_vision(self._image_summary_preset)
        ):
            return self._image_summary_preset
        return None
    
    def _resolve_image_summary_llm(self) -> tuple[Any | None, str | None, str | None]:
        summary_preset = self._resolve_image_summary_preset_name()
        current_preset = (
            self._resolve_exact_current_preset_name()
            or self._resolve_current_preset_name()
        )
        if summary_preset:
            if (
                current_preset
                and current_preset == summary_preset
                and self._preset_supports_vision(current_preset)
            ):
                return self._agent.llm, current_preset, None
            target_preset = summary_preset
        elif current_preset and self._preset_supports_vision(current_preset):
            target_preset = current_preset
        else:
            target_preset = self._resolve_vision_preset_name()

        if not target_preset:
            return None, None, "No vision preset configured."

        preset = self._model_presets.get(target_preset)
        if not preset:
            return None, None, f"Vision preset '{target_preset}' not found."

        api_key_env = str(preset.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.getenv(api_key_env)
        if not api_key:
            return None, None, f"Missing API key env var: {api_key_env}"

        model = str(preset["model"])
        base_url_raw = preset.get("base_url")
        base_url = str(base_url_raw) if isinstance(base_url_raw, str) else None

        try:
            return (
                ChatOpenAI(model=model, api_key=api_key, base_url=base_url),
                target_preset,
                None,
            )
        except Exception as e:
            return None, None, f"Failed to initialize vision summarizer: {e}"

    def _normalize_image_summary(self, text: str, max_chars: int) -> str:
        normalized = " ".join(text.split())
        if not normalized:
            return "未提取到可用图像信息。"
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 3].rstrip() + "..."

    def _normalize_image_detail(self, text: str) -> str:
        return self._normalize_image_summary(text, self.IMAGE_DETAIL_MAX_CHARS)

    async def _extract_image_detail(
        self,
        llm: Any,
        image_part: ContentPartImageParam,
        user_text_hint: str,
    ) -> str:
        max_chars = self.IMAGE_DETAIL_MAX_CHARS
        system_prompt = (
            "你是图像信息提取器。请从截图中做高保真提取，重点支持代码和报错场景。"
            f"请输出中文纯文本，不使用Markdown，不超过{max_chars}字。"
            "按以下顺序输出：截图类型、可见文本原样摘录、关键代码片段、错误信息与调用栈、"
            "文件名和行号、与用户问题相关结论、不确定项。"
            "看不清的内容必须明确写“不可辨识”，禁止臆测。"
        )
        user_prompt = f"请对这张图做详细信息提取。用户原问题: {user_text_hint or '（无）'}"
        high_detail_part = ContentPartImageParam(
            image_url=ImageURL(
                url=str(image_part.image_url.url),
                detail="high",
                media_type=image_part.image_url.media_type,
            )
        )
        response = await llm.ainvoke(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(
                    content=[
                        ContentPartTextParam(text=user_prompt),
                        high_detail_part,
                    ]
                ),
            ],
            tools=None,
            tool_choice=None,
        )
        return self._normalize_image_detail(response.content or "")
    
    async def _compress_image_detail(self, llm: Any, detail_text: str) -> str:
        max_chars = self.IMAGE_MEMORY_MAX_CHARS
        system_prompt = (
            "你是信息压缩器。把详细图像记忆压缩为后续文本模型可用的短记忆。"
            f"输出中文纯文本，不使用Markdown，不超过{max_chars}字。"
            "必须保留：关键错误关键词、文件名/行号、关键结论、主要不确定项。"
        )
        user_prompt = f"请压缩以下图像详细记忆：\n{detail_text}"
        response = await llm.ainvoke(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt),
            ],
            tools=None,
            tool_choice=None,
        )
        return self._normalize_image_summary(response.content or "", max_chars)
    
    async def _prepare_text_model_image_memory(self, *, manual: bool) -> None:
        messages = self._agent._context.get_messages()
        total_images = 0
        for msg in messages:
            if not isinstance(msg, UserMessage):
                continue
            if not isinstance(msg.content, list):
                continue
            for part in msg.content:
                if getattr(part, "type", None) == "image_url":
                    total_images += 1

        if total_images == 0:
            return
        
        summary_llm, summary_label, summary_error = self._resolve_image_summary_llm()
        if summary_llm is not None and summary_label is not None:
            self._console.print(
                f"[dim]Converting {total_images} historical image(s) into text memory via {summary_label}...[/dim]"
            )
        else:
            self._console.print(
                f"[yellow]Vision summarizer unavailable ({summary_error}); using fallback image memory.[/yellow]"
            )

        summarized_count = 0
        fallback_count = 0

        for msg in messages:
            if not isinstance(msg, UserMessage):
                continue
            if not isinstance(msg.content, list):
                continue

            user_text_hint = "\n".join(
                part.text
                for part in msg.content
                if getattr(part, "type", None) == "text"
            ).strip()

            new_parts: list[ContentPartTextParam | ContentPartImageParam] = []
            changed = False

            for part in msg.content:
                if getattr(part, "type", None) != "image_url":
                    new_parts.append(part)
                    continue

                memory_text: str
                if summary_llm is not None:
                    try:
                        detail_text = await self._extract_image_detail(
                            summary_llm,
                            part,
                            user_text_hint=user_text_hint,
                        )
                        summary_text = await self._compress_image_detail(
                            summary_llm, detail_text
                        )
                        memory_text = f"[ImageSummary] {summary_text}"
                        summarized_count += 1
                    except Exception as e:
                        err_preview = " ".join(str(e).split())[:120]
                        memory_text = "[ImageSummary] " + self._normalize_image_summary(
                            f"视觉摘要失败（{err_preview}），切换到文本模型时已移除原始图像数据。",
                            self.IMAGE_MEMORY_MAX_CHARS,
                        )
                        fallback_count += 1
                else:
                    memory_text = "[ImageSummary] " + self._normalize_image_summary(
                        "未配置可用视觉摘要模型，切换到文本模型时已移除原始图像数据。",
                        self.IMAGE_MEMORY_MAX_CHARS,
                    )
                    fallback_count += 1
                new_parts.append(
                    ContentPartTextParam(
                        text=memory_text
                    )
                )
                changed = True

            if changed:
                msg.content = new_parts

        self._agent._context.rebuild_role_index()

        stripped = self._agent._context.strip_user_image_inputs()
        if stripped:
            self._console.print(
                f"[yellow]Removed {stripped} residual image part(s) after memory conversion.[/yellow]"
            )

        status_style = "yellow" if manual else "dim"
        self._console.print(
            f"[{status_style}]Converted {summarized_count}/{total_images} image(s) into text memory.[/{status_style}]"
        )
        if fallback_count:
            self._console.print(
                f"[{status_style}]Fallback memory used for {fallback_count} image(s).[/"
                f"{status_style}]"
            )
    
    async def _apply_auto_model_policy(self, has_image: bool) -> bool:
        """Apply automatic model switching for image/text turns."""
        if not self._model_presets:
            return True

        if self._sticky_preset is None:
            self._sticky_preset = self._resolve_current_preset_name()

        sticky_is_vision = self._preset_supports_vision(self._sticky_preset)
        current_preset = (
            self._resolve_exact_current_preset_name()
            or self._resolve_current_preset_name()
        )

        if has_image:
            # If user intentionally stays on a vision model, never auto-switch.
            if sticky_is_vision:
                return True

            if current_preset and self._preset_supports_vision(current_preset):
                if not self._auto_switched:
                    self._auto_switched = True
                    self._auto_from_preset = self._sticky_preset
                return True

            vision_preset = self._resolve_vision_preset_name()
            if not vision_preset:
                self._console.print(
                    "[red]Image input requires a vision preset. Configure `vision: true` and `auto_vision_preset` in config/model_presets.json.[/red]"
                )
                return False

            self._console.print(
                f"[dim]Auto switch to vision preset: {vision_preset}[/dim]"
            )
            switched = await self._switch_model_preset(vision_preset, manual=False)
            if not switched:
                self._console.print(
                    "[red]Failed to switch to vision model. Image request aborted.[/red]"
                )
                return False

            self._auto_switched = True
            self._auto_from_preset = self._sticky_preset
            return True

        # Text-only turn.
        if not self._auto_switched:
            return True
        if sticky_is_vision:
            return True

        target_preset = self._auto_from_preset or self._sticky_preset
        if not target_preset:
            self._clear_auto_switch_state()
            return True
            
        self._console.print(
            f"[dim]Auto switch back to text preset: {target_preset}[/dim]"
        )
        switched = await self._switch_model_preset(target_preset, manual=False)
        if switched:
            self._clear_auto_switch_state()
        else:
            self._console.print(
                "[yellow]Auto switch-back failed; continue with current model.[/yellow]"
            )
        return True

    def _maybe_inject_agents_md(self) -> None:
        """Inject AGENTS.md into context once per content hash."""
        config_path = self._ctx.working_dir / "AGENTS.md"
        if not config_path.exists():
            return
        # Ensure system prompt is present before injecting AGENTS.md
        if not self._agent._context and self._agent.system_prompt:
            self._agent._context.add_message(
                SystemMessage(content=self._agent.system_prompt)
            )
        content = config_path.read_text(encoding="utf-8").strip()
        if not content:
            return
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if self._agents_md_hash == content_hash and self._agents_md_content:
            # If context was cleared, re-inject even if content unchanged
            for msg in self._agent._context.get_messages():
                if msg.role == "developer" and getattr(msg, "content", "") == self._agents_md_content:
                    return
        # Remove previously injected AGENTS.md content (if any)
        if self._agents_md_content:
            messages = self._agent._context.get_messages()
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if msg.role == "developer" and getattr(msg, "content", "") == self._agents_md_content:
                    self._agent._context.remove_message_at(i)
        self._agents_md_hash = content_hash
        self._agents_md_content = content
        self._agent._context.inject_message(UserMessage(content=content), pinned=True)

    def _build_project_snapshot(self) -> str:
        """Build a lightweight snapshot of the project for summarization."""
        root = self._ctx.working_dir

        def is_ignored_dir(name: str) -> bool:
            return name in {
                ".git",
                ".venv",
                "venv",
                "node_modules",
                "__pycache__",
                ".pytest_cache",
                ".mypy_cache",
                "dist",
                "build",
                ".idea",
            }

        # Tree (depth 4)
        tree_lines: list[str] = []
        for current, dirs, files in os.walk(root):
            rel = os.path.relpath(current, root)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > 4:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if not is_ignored_dir(d)]
            indent = "  " * depth
            tree_lines.append(f"{indent}{os.path.basename(current)}/")
            for f in sorted(files):
                tree_lines.append(f"{indent}  {f}")

        # Important files (top-level)
        important = [
            "README.md",
            "pyproject.toml",
            "package.json",
            "requirements.txt",
        ]
        file_snippets: list[str] = []
        for name in important:
            path = root / name
            if path.exists():
                content = path.read_text(encoding="utf-8")[:4000]
                file_snippets.append(f"## {name}\n{content}")

        # Files (depth-limited, read all with truncation)
        file_snippets_all: list[str] = []
        for current, dirs, files in os.walk(root):
            rel = os.path.relpath(current, root)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > 4:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if not is_ignored_dir(d)]
            for f in sorted(files):
                path = Path(current) / f
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")[:2000]
                except Exception:
                    continue
                file_snippets_all.append(f"## {path.relative_to(root)}\n{content}")

        snapshot = "\n".join(
            [
                f"Project root: {root}",
                "",
                "Tree (depth 4):",
                "\n".join(tree_lines),
                "",
                "Key files:",
                "\n\n".join(file_snippets) if file_snippets else "(none found)",
                "",
                "Files (samples):",
                "\n\n".join(file_snippets_all) if file_snippets_all else "(none found)",
            ]
        )
        return snapshot

    def _print_current_model(self):
        """Print the current model configuration."""
        model = str(self._agent.llm.model)
        base_url = getattr(self._agent.llm, "base_url", None)
        base_url_display = str(base_url) if base_url else "(default)"
        preset_name = self._resolve_current_preset_name()
        preset_line = preset_name or "(unmatched)"
        if preset_name and self._preset_supports_vision(preset_name):
            preset_line += " [vision]"
        self._console.print(f"Current model: [cyan]{model}[/cyan]")
        self._console.print(f"Preset: [dim]{preset_line}[/dim]")
        self._console.print(f"Base URL: [dim]{base_url_display}[/dim]")
        self._console.print(
            f"Context messages: [dim]{len(self._agent.messages)}[/dim]"
        )

    def _print_model_presets(self):
        """Print configured model presets."""
        if not self._model_presets:
            self._console.print(
                f"[yellow]No model presets found at {self._model_presets_path}[/yellow]"
            )
            return

        self._console.print("[bold cyan]Model presets:[/bold cyan]")
        for name, preset in self._model_presets.items():
            model = str(preset["model"])
            base_url = str(preset.get("base_url", "(inherit current)"))
            api_key_env = str(preset.get("api_key_env", "OPENAI_API_KEY"))
            vision_marker = " vision" if self._preset_supports_vision(name) else ""
            marker = (
                " [green](default)[/green]"
                if name == self._default_model_preset
                else ""
            )
            if name == self._auto_vision_preset:
                marker += " [magenta](auto-vision)[/magenta]"
            if name == self._image_summary_preset:
                marker += " [blue](image-summary)[/blue]"
            self._console.print(
                f"  [cyan]{name}[/cyan]{marker} -> {model} "
                f"[dim]{vision_marker}[/dim] "
                f"[dim](base_url: {base_url}, key: {api_key_env})[/dim]"
            )

    def _resolve_current_preset_name(self) -> str | None:
        """Best-effort preset match for current model/base URL."""
        exact_match = self._resolve_exact_current_preset_name()
        if exact_match is not None:
            return exact_match

        if self._default_model_preset in self._model_presets:
            return self._default_model_preset

        return next(iter(self._model_presets.keys()), None)

    def _resolve_exact_current_preset_name(self) -> str | None:
        """Exact preset match for current model/base URL, without fallback."""
        if not self._model_presets:
            return None

        current_model = str(self._agent.llm.model)
        current_base_url = getattr(self._agent.llm, "base_url", None)
        current_base_url_str = str(current_base_url) if current_base_url else None

        for name, preset in self._model_presets.items():
            if preset.get("model") != current_model:
                continue
            preset_base_url = preset.get("base_url")
            if preset_base_url is None or preset_base_url == current_base_url_str:
                return name

        return None

    def _start_model_pick_mode(self):
        """Show numbered model presets and enter pick mode."""
        if not self._model_presets:
            self._console.print("[yellow]No model presets configured.[/yellow]")
            self._model_pick_active = False
            self._model_pick_order = []
            return

        self._model_pick_order = list(self._model_presets.keys())
        self._model_pick_active = True
        current_preset = self._resolve_current_preset_name()

        self._console.print()
        self._console.print("[bold cyan]Select a model preset:[/bold cyan]")
        for idx, name in enumerate(self._model_pick_order, 1):
            preset = self._model_presets[name]
            model = str(preset["model"])
            markers: list[str] = []
            if name == current_preset:
                markers.append("current")
            if name == self._default_model_preset:
                markers.append("default")
            if self._preset_supports_vision(name):
                markers.append("vision")
            if name == self._auto_vision_preset:
                markers.append("auto-vision")
            if name == self._image_summary_preset:
                markers.append("image-summary")
            marker_text = f" [dim]({', '.join(markers)})[/dim]" if markers else ""
            self._console.print(f"  {idx}. [cyan]{name}[/cyan] -> {model}{marker_text}")
        self._console.print(
            "[dim]Type the number and press Enter to switch, or 'q' to cancel.[/dim]"
        )

    async def _handle_model_pick_input(self, user_input: str) -> bool:
        """Handle one line of input while in numbered model-pick mode."""
        if not self._model_pick_active:
            return False

        value = user_input.strip()
        if not value:
            self._console.print("[dim]Enter a number, or 'q' to cancel.[/dim]")
            return True

        if value.lower() in {"q", "quit", "cancel", "exit"}:
            self._model_pick_active = False
            self._model_pick_order = []
            self._console.print("[yellow]Model selection cancelled.[/yellow]")
            return True

        if not value.isdigit():
            self._console.print("[red]Invalid selection. Please enter a number.[/red]")
            return True

        index = int(value)
        if index < 1 or index > len(self._model_pick_order):
            self._console.print(
                f"[red]Selection out of range. Choose 1-{len(self._model_pick_order)}.[/red]"
            )
            return True

        preset_name = self._model_pick_order[index - 1]
        self._model_pick_active = False
        self._model_pick_order = []
        await self._switch_model_preset(preset_name)
        return True

    async def _switch_model_preset(self, preset_name: str, *, manual: bool = True) -> bool:
        """Switch to a configured model preset without clearing conversation context."""
        preset = self._model_presets.get(preset_name)
        if not preset:
            self._console.print(f"[red]Unknown preset: {preset_name}[/red]")
            self._console.print("[dim]Use /model list to see available presets.[/dim]")
            return False

        current_preset = self._resolve_exact_current_preset_name()
        current_is_vision = self._preset_supports_vision(current_preset)
        target_is_vision = bool(preset.get("vision", False))

        if current_preset == preset_name:
            if manual:
                self._sticky_preset = preset_name
                self._clear_auto_switch_state()
            self._console.print(
                f"[dim]Already using preset [cyan]{preset_name}[/cyan].[/dim]"
            )
            return True

        model = str(preset["model"])
        api_key_env = str(preset.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.getenv(api_key_env)
        if not api_key:
            self._console.print(
                f"[red]Missing API key env var: {api_key_env}. Switch aborted.[/red]"
            )
            return False

        context_snapshot: list[Any] | None = None
        if current_is_vision and not target_is_vision:
            context_snapshot = [
                message.model_copy(deep=True)
                for message in self._agent._context.get_messages()
            ]
            try:
                await self._prepare_text_model_image_memory(manual=manual)
            except Exception as e:
                self._agent._context.replace_messages(context_snapshot)
                self._console.print(
                    f"[red]Failed to prepare image memory before switch: {e}[/red]"
                )
                return False

        try:
            preflight = await self._agent.preflight_model_switch(model)
        except Exception as e:
            if context_snapshot is not None:
                self._agent._context.replace_messages(context_snapshot)
            self._console.print(f"[red]Model switch preflight failed: {e}[/red]")
            return False
        if not preflight.ok:
            if context_snapshot is not None:
                self._agent._context.replace_messages(context_snapshot)
            self._console.print(
                f"[red]Model switch preflight failed: {preflight.reason or 'context is too large'}[/red]"
            )
            self._console.print(
                f"[dim]Estimated tokens: {preflight.estimated_tokens}, "
                f"threshold: {preflight.threshold}, "
                f"utilization: {preflight.threshold_utilization:.0%}[/dim]"
            )
            return False

        old_llm = self._agent.llm
        old_model = str(old_llm.model)
        old_base_url = getattr(old_llm, "base_url", None)
        new_base_url_raw = preset.get("base_url")
        new_base_url = str(new_base_url_raw) if isinstance(new_base_url_raw, str) else None
        if new_base_url is None and old_base_url is not None:
            new_base_url = str(old_base_url)

        try:
            self._agent.llm = ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=new_base_url,
            )
        except Exception as e:
            self._agent.llm = old_llm
            if context_snapshot is not None:
                self._agent._context.replace_messages(context_snapshot)
            self._console.print(f"[red]Failed to switch model: {e}[/red]")
            return False

        if preflight.compacted:
            self._console.print(
                "[yellow]Context was compacted before switching to fit target model.[/yellow]"
            )

        self._console.print(
            f"[green]Model switched:[/] [dim]{old_model}[/dim] -> [cyan]{model}[/cyan]"
        )
        self._console.print(
            f"[dim]Context preserved ({len(self._agent.messages)} messages).[/dim]"
        )
        if manual:
            self._sticky_preset = preset_name
            self._clear_auto_switch_state()
        return True

    def _start_loading(self, message: str = "Thinking") -> _LoadingIndicator:
        """Start a loading animation."""
        loading = _LoadingIndicator(message)
        loading.start()
        time.sleep(0.02)
        return loading

    def _stop_loading(self, loading: _LoadingIndicator | None):
        """Stop the loading animation."""
        if loading:
            loading.stop()

    def _print_welcome(self):
        """Print welcome message."""
        self._console.print(
            Panel(
                f"[bold cyan]Claude Code CLI[/bold cyan]\n\n"
                f"Type your message and press Enter to send.\n"
                f"Press [cyan]/[/cyan] to see available commands.\n"
                f"Press [cyan]@[/cyan] + [cyan]Tab[/cyan] to see available skills.\n"
                f"Use [cyan]@\"<path>\"<message>[/cyan] or "
                f"[cyan]@'<path>'<message>[/cyan] for image input.\n"
                f"Press Ctrl+D or type [cyan]/exit[/cyan] to quit.\n",
                title="[bold blue]Welcome[/bold blue]",
                border_style="bright_blue",
            )
        )

        # Show sandbox info
        self._console.print()
        self._console.print(f"[dim]Working directory:[/] {self._ctx.working_dir}")
        self._console.print(f"[dim]Model:[/] {self._agent.llm.model}")
        self._console.print(f"[dim]Tools:[/] bash, read, write, edit, glob, grep, todos")
        self._console.print(f"[dim]Slash Commands:[/] Press [cyan]/[/cyan] + [cyan]Tab[/cyan] to see all")
        self._console.print(f"[dim]Skill Commands:[/] Press [cyan]@[/cyan] + [cyan]Tab[/cyan] to see all")
        if self._model_presets:
            self._console.print(
                f"[dim]Model presets:[/] {', '.join(self._model_presets.keys())}"
            )
        self._console.print()

    def _print_help(self):
        """Print help information."""
        help_text = """
[bold cyan]Available Commands:[/bold cyan]

  [blue]help[/blue]          - Show this help message
  [blue]exit[/blue]          - Exit the CLI
  [blue]pwd[/blue]           - Print current working directory
  [blue]ls [path][/blue]     - List files in directory (AI can do this too)

[bold cyan]Available Tools (for AI):[/bold cyan]

  [blue]bash <cmd>[/blue]    - Run shell commands
  [blue]read <file>[/blue]   - Read file contents
  [blue]write <file>[/blue]  - Write content to file
  [blue]edit <file>[/blue]   - Edit file (replace text)
  [blue]glob <pattern>[/blue]- Find files by pattern
  [blue]grep <pattern>[/blue] - Search file contents
  [blue]todos[/blue]         - Manage todo list

[bold cyan]Tips:[/boldcyan]

  - Just type your request naturally, e.g., "List all Python files"
  - Send image input with [blue]@"<path>"<message>[/blue] or [blue]@'<path>'<message>[/blue]
  - The AI will use tools automatically to help you
"""
        self._console.print(Panel(help_text, border_style="dim"))

    def _print_slash_help(self):
        """Print slash command help information."""
        self._console.print()
        self._console.print("[bold cyan]Slash Commands:[/bold cyan]")
        self._console.print("[dim]Press / to see available commands, Tab to autocomplete[/dim]")
        self._console.print("[bold cyan]Skill Commands (@):[/bold cyan]")
        self._console.print("[dim]Use @<skill-name> to invoke skills, Tab to autocomplete[/dim]")
        self._console.print(
            "[dim]Use @\"<path>\"<message> or @'<path>'<message> for image input[/dim]"
        )
        self._console.print()

        categories = self._slash_registry.get_by_category()
        for category, commands in sorted(categories.items()):
            self._console.print(f"[bold blue]{category}:[/bold blue]")
            for cmd in commands:
                self._console.print(f"  [cyan]/{cmd.name}[/cyan] - {cmd.description}")
        self._console.print()

    def _print_slash_command_detail(self, command_name: str):
        """Print detailed help for a specific slash command.

        Args:
            command_name: Name of the command (without /)
        """
        cmd = self._slash_registry.get(command_name)
        if not cmd:
            self._console.print(f"[red]Unknown command: /{command_name}[/red]")
            return

        self._console.print()
        self._console.print(Panel(
            f"[bold cyan]/{cmd.name}[/bold cyan]\n\n"
            f"[dim]{cmd.description}[/dim]\n\n"
            f"[bold]Usage:[/bold] {cmd.usage}\n\n"
            + (f"[bold]Examples:[/bold]\n" + "\n".join(f"  • {ex}" for ex in cmd.examples) if cmd.examples else ""),
            title="[bold blue]Command Details[/bold blue]",
            border_style="bright_blue",
        ))
        self._console.print()

    async def _handle_slash_command(self, text: str) -> bool:
        """Handle a slash command.

        Args:
            text: The slash command text

        Returns:
            True if the command was handled, False otherwise
        """
        command_name, args = parse_slash_command(text)

        # Handle help command
        if command_name in ("help", "h"):
            if args and args[0].startswith("/"):
                # Show details for a specific command
                self._print_slash_command_detail(args[0][1:])
            elif args:
                self._print_slash_command_detail(args[0])
            else:
                self._print_slash_help()
            return True

        # Handle exit/quit commands
        if command_name in ("exit", "quit", "q"):
            self._console.print("[yellow]Goodbye![/yellow]")
            raise EOFError()

        # Handle pwd command
        if command_name == "pwd":
            self._console.print(f"{self._ctx.working_dir}")
            return True

        # Handle clear command
        if command_name == "clear" or command_name == "cls":
            os.system("cls" if os.name == "nt" else "clear")
            return True

        # Handle model command
        if command_name == "model":
            if not args:
                self._start_model_pick_mode()
                return True

            if len(args) > 1:
                self._console.print("[red]Usage: /model [show|list|<preset>][/red]")
                return True

            subcommand = args[0].lower()
            if subcommand == "show":
                self._print_current_model()
                return True
            if subcommand == "list":
                self._print_model_presets()
                return True

            # Convenience: /model <preset>
            await self._switch_model_preset(args[0])
            return True

        # Handle reset command
        if command_name == "reset":
            self._agent.clear_history()
            self._console.print("[yellow]Conversation context reset.[/yellow]")
            return True

        if command_name == "init":
            # Generate docs/PROJECT.md with an AI summary of the project
            out_path = self._ctx.working_dir / "AGENTS.md"

            snapshot = self._build_project_snapshot()
            system = SystemMessage(
                content=(
                    "The user just ran `/init`.\n"
                    "Generate AGENTS.md based on the project snapshot.\n"
                    "Keep it concise and useful for onboarding.\n"
                    "内容要求中文"
                )
            )
            user = UserMessage(
                content=(
                    "Based on the project snapshot below, write AGENTS.md with:\n"
                    "1) Overview\n2) Project Structure\n3) How It Works\n"
                    "4) Constraints/Assumptions\n\n"
                    f"{snapshot}"
                )
            )

            response = await self._agent.llm.ainvoke(
                messages=[system, user],
                tools=None,
                tool_choice=None,
            )
            content = response.content or ""
            # Strip any hidden thinking blocks from the output
            if content:
                import re
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r"<analysis>.*?</analysis>", "", content, flags=re.DOTALL | re.IGNORECASE)
            out_path.write_text(content, encoding="utf-8")
            self._console.print("[yellow]Generated AGENTS.md[/yellow]")
            return True

        if command_name == "tasks":
            if not self._ctx.subagent_manager:
                self._console.print("[yellow]Subagent manager not initialized.[/yellow]")
                return True
            tasks_info = self._ctx.subagent_manager.list_all_tasks()
            self._console.print("[bold cyan]Background Tasks:[/bold cyan]")
            self._console.print(tasks_info)
            return True

        if command_name == "task":
            if not self._ctx.subagent_manager:
                self._console.print("[yellow]Subagent manager not initialized.[/yellow]")
                return True
            if not args:
                self._console.print("[red]Usage: /task <task_id>[/red]")
                return True
            task_id = args[0]
            task_info = self._ctx.subagent_manager.get_task_status(task_id)
            if task_info is None:
                self._console.print(f"[red]Task '{task_id}' not found.[/red]")
                return True
            self._console.print("[bold cyan]Task Details:[/bold cyan]")
            self._console.print(task_info)
            return True

        if command_name == "task_cancel":
            if not self._ctx.subagent_manager:
                self._console.print("[yellow]Subagent manager not initialized.[/yellow]")
                return True
            if not args:
                self._console.print("[red]Usage: /task_cancel <task_id>[/red]")
                return True
            task_id = args[0]
            result = await self._ctx.subagent_manager.cancel_task(task_id)
            self._console.print(result)
            return True

        # Handle history command
        if command_name == "history":
            self._console.print("[dim]Command history not implemented yet.[/dim]")
            return True

        # Handle skills command - list available @ skills
        if command_name == "skills":
            self._print_available_skills()
            return True

        # Handle allow command - add directory to sandbox
        if command_name == "allow":
            if not args:
                self._console.print("[red]Usage: /allow <path>[/red]")
                self._console.print("[dim]Example: /allow /path/to/project[/dim]")
            else:
                path_str = " ".join(args)
                try:
                    added_path = self._ctx.add_allowed_dir(path_str)
                    self._console.print(f"[green]Added to allowed directories:[/] {added_path}")
                except SecurityError as e:
                    self._console.print(f"[red]{e}[/red]")
            return True

        # Handle allowed command - list allowed directories
        if command_name == "allowed":
            self._console.print()
            self._console.print("[bold cyan]Allowed Directories:[/bold cyan]")
            for i, allowed_dir in enumerate(self._ctx.allowed_dirs, 1):
                # 标记当前工作目录
                marker = " [dim](current)[/]" if str(allowed_dir.resolve()) == str(self._ctx.working_dir.resolve()) else ""
                self._console.print(f"  {i}. {allowed_dir}{marker}")
            self._console.print()
            return True

        # Handle agents command - manage agent configurations
        if command_name == "agents":
            from cli.agents_handler import AgentSlashHandler

            handler = AgentSlashHandler(
                registry=self._agent_registry,
                console=self._console,
            )
            return await handler.handle(args)

        if command_name == "plugins":
            if self._plugin_manager is None:
                self._console.print("[yellow]Plugin manager not configured.[/yellow]")
                return True

            handler = PluginSlashHandler(
                manager=self._plugin_manager,
                console=self._console,
            )
            result = await handler.handle(args)
            if result.reloaded:
                self._refresh_system_prompt()
            return result.handled

        if command_name == "ralph":
            if self._ralph_handler is None:
                self._ralph_handler = RalphSlashHandler(
                    workspace_root=self._ctx.working_dir,
                    console=self._console,
                )
            return await self._ralph_handler.handle(args)

        if self._plugin_manager is not None:
            plugin_command = self._plugin_manager.get_command(command_name)
            if plugin_command is not None:
                await self._run_agent(plugin_command.render_prompt(args), has_image=False)
                return True

        # Unknown command
        self._console.print(f"[red]Unknown command: /{command_name}[/red]")
        self._console.print(f"[dim]Type /help for available commands.[/dim]")
        return True

    def _refresh_system_prompt(self) -> None:
        """Rebuild the agent system prompt after plugin registry changes."""
        if self._system_prompt_builder is None:
            return
        self._agent.system_prompt = self._system_prompt_builder()
        self._agent.clear_history()
        self._console.print("[yellow]Conversation context reset after plugin reload.[/yellow]")

    def _print_available_skills(self):
        """Print all available @ skills grouped by category."""
        self._console.print()
        self._console.print("[bold cyan]Available Skills (@):[/bold cyan]")
        self._console.print(
            "[dim]Use @<skill-name> to load a skill before your message[/dim]"
        )
        self._console.print(
            "[dim]Image input: @\"<path>\"<message> or @'<path>'<message>[/dim]"
        )
        self._console.print()

        categories = self._at_registry.get_by_category()
        if not categories:
            self._console.print("[yellow]No skills found.[/yellow]")
            self._console.print()
            return

        for category, commands in sorted(categories.items()):
            self._console.print(f"[bold blue]{category}:[/bold blue]")
            for cmd in commands:
                self._console.print(f"  [cyan]@{cmd.name}[/cyan] - {cmd.description}")
        self._console.print()

    async def _handle_at_command(self, text: str) -> bool:
        """Handle an @ command for skill invocation."""
        skill_name, message = parse_at_command(text)
        if not skill_name:
            self._console.print("[yellow]Invalid @ command.[/yellow]")
            self._console.print("[dim]Type @ and press Tab to see available skills.[/dim]")
            return True

        skill = self._at_registry.get(skill_name)
        if not skill:
            self._console.print(f"[yellow]Skill not found: @{skill_name}[/yellow]")
            self._console.print("[dim]Use /skills to list available skills.[/dim]")
            return True

        self._console.print(f"[cyan]Using @{skill.name}...[/cyan]")
        try:
            expanded_message = expand_at_command(skill, message)
        except (IOError, ValueError) as e:
            self._console.print(f"[red]Failed to load skill: {e}[/red]")
            return True

        await self._run_agent(expanded_message, has_image=False)
        return True

    async def _run_agent(self, user_input: UserInputPayload, has_image: bool = False):
        """Run the agent with user input and display events."""
        if not await self._apply_auto_model_policy(has_image=has_image):
            return
        self._step_number = 0
        self._console.print()

        # Inject AGENTS.md (if present) before each user query
        self._maybe_inject_agents_md()

        # Start loading animation
        self._loading = self._start_loading("Thinking")

        try:
            async for event in self._agent.query_stream(user_input):
                if isinstance(event, ToolCallEvent):
                    self._stop_loading(self._loading)
                    self._loading = None

                    self._step_number += 1
                    args_str = str(event.args)[:100]
                    if len(str(event.args)) > 100:
                        args_str += "..."
                    self._console.print()
                    self._console.print(
                        f"[{self.COLOR_TOOL_CALL}]Step {self._step_number}:[/] "
                        f"[bold]{event.tool}[/]"
                    )
                    self._console.print(f"  [dim]Args: {args_str}[/]")

                    # Start "Executing" loading while tool runs
                    self._loading = self._start_loading("Executing")

                elif isinstance(event, ToolResultEvent):
                    self._stop_loading(self._loading)
                    self._loading = None

                    result = str(event.result)
                    if event.is_error:
                        self._console.print(f"  [{self.COLOR_ERROR}]Error: {result[:200]}[/]")
                    else:
                        if len(result) > 300:
                            self._console.print(
                                f"  [{self.COLOR_TOOL_RESULT}]Result: {result[:300]}...[/]"
                            )
                        else:
                            self._console.print(f"  [{self.COLOR_TOOL_RESULT}]Result: {result}[/]")

                    # Restart "Thinking" loading for next LLM call
                    self._loading = self._start_loading("Thinking")

                elif isinstance(event, ThinkingEvent):
                    self._stop_loading(self._loading)
                    self._loading = None
                    self._console.print(f"[{self.COLOR_THINKING}]Thinking: {event.content}[/]")

                elif isinstance(event, TextEvent):
                    self._stop_loading(self._loading)
                    self._loading = None
                    self._console.print(event.content, end="")

                elif isinstance(event, FinalResponseEvent):
                    self._stop_loading(self._loading)
                    self._loading = None
                    self._console.print()
                    self._console.print()

        except Exception as e:
            self._stop_loading(self._loading)
            self._loading = None
            self._console.print(f"[{self.COLOR_ERROR}]Error: {e}[/]")

        # Ensure loading is stopped
        self._stop_loading(self._loading)
        self._loading = None
        self._console.print()

    async def _on_task_completed(self, result: Any):
        """Handle background task completion notification.

        Args:
            result: TaskResult from SubagentManager
        """
        from bu_agent_sdk.agent.subagent_manager import TaskResult
        from rich.panel import Panel

        if not isinstance(result, TaskResult):
            return

        # Display completion notification
        status_emoji = "✅" if result.status == "completed" else "❌"
        status_color = "green" if result.status == "completed" else "red"

        if result.status == "completed":
            self._console.print()
            self._console.print(
                Panel(
                    f"[{status_color}]{status_emoji} Task Completed:[/{status_color}]\n"
                    f"[bold]Subagent:[/] {result.subagent_name}\n"
                    f"[bold]Task ID:[/] {result.task_id}\n"
                    f"[bold]Execution Time:[/] {result.execution_time_ms:.0f}ms\n"
                    f"[bold]Tools Used:[/] {', '.join(result.tools_used) if result.tools_used else 'None'}\n"
                    f"[bold]Result:[/] {result.final_response[:500]}..."
                    if len(result.final_response) > 500 else "...",
                    title="[bold blue]Background Task Notification[/bold blue]",
                    border_style=status_color,
                )
            )
        elif result.status == "failed":
            self._console.print()
            self._console.print(
                Panel(
                    f"[{status_color}]{status_emoji} Task Failed:[/{status_color}]\n"
                    f"[bold]Subagent:[/] {result.subagent_name}\n"
                    f"[bold]Task ID:[/] {result.task_id}\n"
                    f"[bold]Error:[/] {result.error}",
                    title="[bold blue]Background Task Notification[/bold blue]",
                    border_style=status_color,
                )
            )
        elif result.status == "cancelled":
            self._console.print()
            self._console.print(
                Panel(
                    f"[yellow]⏹️  Task Cancelled:[/yellow]\n"
                    f"[bold]Subagent:[/] {result.subagent_name}\n"
                    f"[bold]Task ID:[/] {result.task_id}",
                    title="[bold blue]Background Task Notification[/bold blue]",
                    border_style="yellow",
                )
            )

    async def run(self):
        """Run the interactive CLI."""
        from prompt_toolkit import PromptSession
        from prompt_toolkit.styles import Style

        # Print welcome
        self._print_welcome()

        # Create key bindings
        kb = KeyBindings()

        @kb.add("c-d")
        def _exit(event):  # noqa: D401
            event.app.exit(exception=EOFError)

        @kb.add("enter")
        def _enter(event):  # noqa: D401
            """Accept @ completion with Enter instead of submitting immediately."""
            buffer = event.current_buffer
            complete_state = buffer.complete_state

            # When selecting @skill completion, Enter should apply completion
            # and keep input in the prompt for the user to continue typing.
            if complete_state and buffer.text.lstrip().startswith("@"):
                completion = complete_state.current_completion
                if completion is None and complete_state.completions:
                    completion = complete_state.completions[0]

                if completion is not None:
                    buffer.apply_completion(completion)
                    skill_name, message = parse_at_command(buffer.text)
                    if skill_name and not message and not buffer.text.endswith(" "):
                        buffer.insert_text(" ")
                    return

            buffer.validate_and_handle()

        # Mark as intentionally used
        _ = _exit
        _ = _enter

        # Create slash command completer
        slash_completer = SlashCommandCompleter(self._slash_registry)
        # Create at command completer
        at_completer = AtCommandCompleter(self._at_registry)
        # Use merged completer to handle both / and @
        from prompt_toolkit.completion import merge_completers
        merged_completer = merge_completers([slash_completer, at_completer])
        threaded_completer = ThreadedCompleter(merged_completer)

        # Define style for better visual feedback
        style = Style.from_dict({
            "completion-menu.completion": "bg:#008888 #ffffff",
            "completion-menu.completion.current": "bg:#ffffff #000000",
            "completion-menu.meta.completion": "bg:#00aaaa #000000",
            "completion-menu.meta.current": "bg:#00ffff #000000",
            "completion-menu": "bg:#008888 #ffffff",
        })

        # Create prompt session with completer
        session = PromptSession(
            message=lambda: HTML("<ansiblue>>> </ansiblue>"),
            key_bindings=kb,
            completer=threaded_completer,
            complete_while_typing=True,
            auto_suggest=AutoSuggestFromHistory(),
            style=style,
            enable_history_search=True,
        )

        while True:
            try:
                user_input = await session.prompt_async()
            except EOFError:
                self._console.print("\n[yellow]Goodbye![/yellow]")
                break
            except KeyboardInterrupt:
                continue

            user_input = user_input.strip()
            if not user_input:
                continue

            # Handle numbered model picker mode
            if self._model_pick_active:
                if await self._handle_model_pick_input(user_input):
                    continue
            
            # Handle quoted @ image command
            if is_image_command(user_input):
                try:
                    parsed = parse_image_command(user_input, self._ctx)
                except ImageInputError as e:
                    self._console.print(f"[red]{e}[/red]")
                    self._console.print(f"[dim]{IMAGE_USAGE}[/dim]")
                    continue
                await self._run_agent(parsed.content_parts, has_image=True)
                continue

            # Handle @ commands (skill invocation)
            if is_at_command(user_input):
                try:
                    if await self._handle_at_command(user_input):
                        continue
                except EOFError:
                    break
                continue

            # Handle slash commands
            if is_slash_command(user_input):
                try:
                    if await self._handle_slash_command(user_input):
                        continue
                except EOFError:
                    break
                continue

            # Handle legacy built-in commands (without slash)
            if user_input.lower() in ["exit", "quit"]:
                self._console.print("[yellow]Goodbye![/yellow]")
                break

            if user_input.lower() == "help":
                self._print_help()
                continue

            if user_input.lower() == "pwd":
                self._console.print(f"Current directory: {self._ctx.working_dir}")
                continue

            # Run agent
            await self._run_agent(user_input, has_image=False)
