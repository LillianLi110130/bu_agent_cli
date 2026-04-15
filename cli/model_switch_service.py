"""Shared model-switching service for CLI commands and runtime hooks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from agent_core import Agent
from agent_core.llm import ChatOpenAI
from agent_core.llm.messages import (
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    SystemMessage,
    UserMessage,
)
from config.model_config import ModelPreset


@dataclass
class ModelAutoState:
    sticky_preset: str | None = None
    auto_switched: bool = False
    auto_from_preset: str | None = None


class ModelSwitchService:
    """Encapsulates preset resolution and context-preserving model switching."""

    IMAGE_DETAIL_MAX_CHARS = 2200
    IMAGE_MEMORY_MAX_CHARS = 600

    def __init__(
        self,
        *,
        agent: Agent,
        model_presets: dict[str, ModelPreset],
        default_model_preset: str | None,
        auto_vision_preset: str | None,
        image_summary_preset: str | None,
        console: Any | None = None,
    ) -> None:
        self._agent = agent
        self._model_presets = model_presets
        self._default_model_preset = default_model_preset
        self._auto_vision_preset = auto_vision_preset
        self._image_summary_preset = image_summary_preset
        self._console = console

    def _print(self, message: str) -> None:
        if self._console is not None:
            self._console.print(message)

    def clear_auto_switch_state(self, state: ModelAutoState) -> None:
        state.auto_switched = False
        state.auto_from_preset = None

    def preset_supports_vision(self, preset_name: str | None) -> bool:
        if not preset_name:
            return False
        preset = self._model_presets.get(preset_name)
        if not preset:
            return False
        return bool(preset.get("vision", False))

    def resolve_vision_preset_name(self) -> str | None:
        if (
            self._auto_vision_preset
            and self._auto_vision_preset in self._model_presets
            and self.preset_supports_vision(self._auto_vision_preset)
        ):
            return self._auto_vision_preset
        for name in self._model_presets.keys():
            if self.preset_supports_vision(name):
                return name
        return None

    def resolve_image_summary_preset_name(self) -> str | None:
        if (
            self._image_summary_preset
            and self._image_summary_preset in self._model_presets
            and self.preset_supports_vision(self._image_summary_preset)
        ):
            return self._image_summary_preset
        return None

    def resolve_current_preset_name(self) -> str | None:
        exact_match = self.resolve_exact_current_preset_name()
        if exact_match is not None:
            return exact_match

        if self._default_model_preset in self._model_presets:
            return self._default_model_preset

        return next(iter(self._model_presets.keys()), None)

    def resolve_exact_current_preset_name(self) -> str | None:
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

    def _resolve_image_summary_llm(self) -> tuple[Any | None, str | None, str | None]:
        summary_preset = self.resolve_image_summary_preset_name()
        current_preset = (
            self.resolve_exact_current_preset_name() or self.resolve_current_preset_name()
        )
        if summary_preset:
            if (
                current_preset
                and current_preset == summary_preset
                and self.preset_supports_vision(current_preset)
            ):
                return self._agent.llm, current_preset, None
            target_preset = summary_preset
        elif current_preset and self.preset_supports_vision(current_preset):
            target_preset = current_preset
        else:
            target_preset = self.resolve_vision_preset_name()

        if not target_preset:
            return None, None, "未配置视觉预设。"

        preset = self._model_presets.get(target_preset)
        if not preset:
            return None, None, f"未找到视觉预设：{target_preset}"

        api_key_env = str(preset.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.getenv(api_key_env)
        if not api_key:
            return None, None, f"缺少 API Key 环境变量：{api_key_env}"

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
            return None, None, f"初始化视觉摘要器失败：{e}"

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

    def _iter_user_messages_with_parts(self) -> list[UserMessage]:
        return [
            msg
            for msg in self._agent._context.get_messages()
            if isinstance(msg, UserMessage) and isinstance(msg.content, list)
        ]

    @staticmethod
    def _count_message_images(message: UserMessage) -> int:
        if not isinstance(message.content, list):
            return 0
        return sum(1 for part in message.content if getattr(part, "type", None) == "image_url")

    @staticmethod
    def _extract_user_text_hint(message: UserMessage) -> str:
        if not isinstance(message.content, list):
            return ""
        return "\n".join(
            part.text for part in message.content if getattr(part, "type", None) == "text"
        ).strip()

    def _build_fallback_image_memory(self, reason: str) -> str:
        return "[ImageSummary] " + self._normalize_image_summary(
            reason,
            self.IMAGE_MEMORY_MAX_CHARS,
        )

    async def _build_image_memory_text(
        self,
        *,
        summary_llm: Any | None,
        image_part: ContentPartImageParam,
        user_text_hint: str,
    ) -> tuple[str, bool]:
        if summary_llm is None:
            return (
                self._build_fallback_image_memory(
                    "未配置可用视觉摘要模型，切换到文本模型时已移除原始图像数据。"
                ),
                True,
            )

        try:
            detail_text = await self._extract_image_detail(
                summary_llm,
                image_part,
                user_text_hint=user_text_hint,
            )
            summary_text = await self._compress_image_detail(summary_llm, detail_text)
        except Exception as e:
            err_preview = " ".join(str(e).split())[:120]
            return (
                self._build_fallback_image_memory(
                    f"视觉摘要失败（{err_preview}），切换到文本模型时已移除原始图像数据。"
                ),
                True,
            )

        artifact_path = self._agent._context.persist_image_detail_artifact(
            detail_text,
            source_hint=user_text_hint,
        )
        if artifact_path:
            summary_text = f"{summary_text}\n详细视觉提取见 {artifact_path}"

        return f"[ImageSummary] {summary_text}", False

    async def _rewrite_message_image_memory(
        self,
        message: UserMessage,
        *,
        summary_llm: Any | None,
    ) -> tuple[int, int, bool]:
        if not isinstance(message.content, list):
            return 0, 0, False

        user_text_hint = self._extract_user_text_hint(message)
        new_parts: list[ContentPartTextParam | ContentPartImageParam] = []
        summarized_count = 0
        fallback_count = 0
        changed = False

        for part in message.content:
            if getattr(part, "type", None) != "image_url":
                new_parts.append(part)
                continue

            memory_text, used_fallback = await self._build_image_memory_text(
                summary_llm=summary_llm,
                image_part=part,
                user_text_hint=user_text_hint,
            )
            new_parts.append(ContentPartTextParam(text=memory_text))
            changed = True
            if used_fallback:
                fallback_count += 1
            else:
                summarized_count += 1

        if changed:
            message.content = new_parts
        return summarized_count, fallback_count, changed

    def _print_image_memory_preparation_start(
        self,
        *,
        total_images: int,
        summary_llm: Any | None,
        summary_label: str | None,
        summary_error: str | None,
    ) -> None:
        if summary_llm is not None and summary_label is not None:
            self._print(
                f"[dim]正在通过 {summary_label} 将 {total_images} 张历史图片转换为文本记忆...[/dim]"
            )
            return
        self._print(f"[yellow]视觉摘要器不可用（{summary_error}）；将使用回退图片记忆。[/yellow]")

    def _print_image_memory_preparation_result(
        self,
        *,
        total_images: int,
        summarized_count: int,
        fallback_count: int,
        manual: bool,
    ) -> None:
        status_style = "yellow" if manual else "dim"
        self._print(
            f"[{status_style}]已将 {summarized_count}/{total_images} 张图片转换为文本记忆。[/{status_style}]"
        )
        if fallback_count:
            self._print(
                f"[{status_style}]其中 {fallback_count} 张图片使用了回退记忆。[/" f"{status_style}]"
            )

    async def prepare_text_model_image_memory(self, *, manual: bool) -> None:
        user_messages = self._iter_user_messages_with_parts()
        total_images = sum(self._count_message_images(message) for message in user_messages)
        if total_images == 0:
            return

        summary_llm, summary_label, summary_error = self._resolve_image_summary_llm()
        self._print_image_memory_preparation_start(
            total_images=total_images,
            summary_llm=summary_llm,
            summary_label=summary_label,
            summary_error=summary_error,
        )

        summarized_count = 0
        fallback_count = 0

        for message in user_messages:
            message_summarized, message_fallback, changed = await self._rewrite_message_image_memory(
                message,
                summary_llm=summary_llm,
            )
            summarized_count += message_summarized
            fallback_count += message_fallback
            if changed:
                self._agent._context.rebuild_role_index()
                self._agent._context.invalidate_budget_baseline()

        stripped = self._agent._context.strip_user_image_inputs()
        if stripped:
            self._print(f"[yellow]记忆转换后已移除 {stripped} 个残留图片片段。[/yellow]")

        self._print_image_memory_preparation_result(
            total_images=total_images,
            summarized_count=summarized_count,
            fallback_count=fallback_count,
            manual=manual,
        )

    def _resolve_switch_preset(self, preset_name: str) -> ModelPreset | None:
        preset = self._model_presets.get(preset_name)
        if preset is not None:
            return preset

        self._print(f"[red]未知预设：{preset_name}[/red]")
        self._print("[dim]使用 /model list 查看可用预设。[/dim]")
        return None

    def _snapshot_context_messages(self) -> list[Any]:
        return [message.model_copy(deep=True) for message in self._agent._context.get_messages()]

    def _restore_context_snapshot(self, context_snapshot: list[Any] | None) -> None:
        if context_snapshot is not None:
            self._agent._context.replace_messages(context_snapshot)

    def _should_prepare_text_image_memory(
        self,
        *,
        current_preset: str | None,
        target_preset: ModelPreset,
    ) -> bool:
        current_is_vision = self.preset_supports_vision(current_preset)
        target_is_vision = bool(target_preset.get("vision", False))
        return current_is_vision and not target_is_vision

    async def _prepare_switch_context(
        self,
        *,
        current_preset: str | None,
        target_preset: ModelPreset,
        manual: bool,
    ) -> tuple[bool, list[Any] | None]:
        if not self._should_prepare_text_image_memory(
            current_preset=current_preset,
            target_preset=target_preset,
        ):
            return True, None

        context_snapshot = self._snapshot_context_messages()
        try:
            await self.prepare_text_model_image_memory(manual=manual)
        except Exception as e:
            self._restore_context_snapshot(context_snapshot)
            self._print(f"[red]切换前准备图片记忆失败：{e}[/red]")
            return False, None
        return True, context_snapshot

    async def _run_switch_preflight(
        self,
        *,
        model: str,
        context_snapshot: list[Any] | None,
    ) -> tuple[bool, Any | None]:
        try:
            preflight = await self._agent.preflight_model_switch(model)
        except Exception as e:
            self._restore_context_snapshot(context_snapshot)
            self._print(f"[red]模型切换预检查失败：{e}[/red]")
            return False, None

        if preflight.ok:
            return True, preflight

        self._restore_context_snapshot(context_snapshot)
        self._print(f"[red]模型切换预检查失败：{preflight.reason or '上下文过大'}[/red]")
        self._print(
            f"[dim]预估 tokens：{preflight.estimated_tokens}，"
            f"阈值：{preflight.threshold}，"
            f"使用率：{preflight.threshold_utilization:.0%}[/dim]"
        )
        return False, None

    def _resolve_switch_target_base_url(self, preset: ModelPreset, old_llm: Any) -> str | None:
        new_base_url_raw = preset.get("base_url")
        new_base_url = str(new_base_url_raw) if isinstance(new_base_url_raw, str) else None
        old_base_url = getattr(old_llm, "base_url", None)
        if new_base_url is None and old_base_url is not None:
            return str(old_base_url)
        return new_base_url

    def _apply_manual_switch_state(
        self,
        *,
        preset_name: str,
        manual: bool,
        auto_state: ModelAutoState | None,
    ) -> None:
        if manual and auto_state is not None:
            auto_state.sticky_preset = preset_name
            self.clear_auto_switch_state(auto_state)

    def _print_switch_success(
        self,
        *,
        old_model: str,
        model: str,
        preflight: Any,
    ) -> None:
        if preflight.compacted:
            self._print("[yellow]为适配目标模型，已在切换前压缩上下文。[/yellow]")
        self._print(f"[green]模型已切换：[/] [dim]{old_model}[/dim] -> [cyan]{model}[/cyan]")
        self._print(f"[dim]已保留上下文（{len(self._agent.messages)} 条消息）。[/dim]")
        if (
            getattr(preflight, "token_estimate_source", None) == "local_full"
            and preflight.estimated_tokens > 0
        ):
            self._print("[dim]切换模型后使用本地估算，下一次模型返回 usage 后校准。[/dim]")

    async def switch_model_preset(
        self,
        preset_name: str,
        *,
        manual: bool = True,
        auto_state: ModelAutoState | None = None,
    ) -> bool:
        preset = self._resolve_switch_preset(preset_name)
        if preset is None:
            return False

        current_preset = self.resolve_exact_current_preset_name()
        if current_preset == preset_name:
            self._apply_manual_switch_state(
                preset_name=preset_name,
                manual=manual,
                auto_state=auto_state,
            )
            self._print(f"[dim]当前已在使用预设 [cyan]{preset_name}[/cyan]。[/dim]")
            return True

        model = str(preset["model"])
        api_key_env = str(preset.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.getenv(api_key_env)
        if not api_key:
            self._print(f"[red]缺少 API Key 环境变量：{api_key_env}。已中止切换。[/red]")
            return False

        prepared, context_snapshot = await self._prepare_switch_context(
            current_preset=current_preset,
            target_preset=preset,
            manual=manual,
        )
        if not prepared:
            return False

        preflight_ok, preflight = await self._run_switch_preflight(
            model=model,
            context_snapshot=context_snapshot,
        )
        if not preflight_ok or preflight is None:
            return False

        old_llm = self._agent.llm
        old_model = str(old_llm.model)
        new_base_url = self._resolve_switch_target_base_url(preset, old_llm)

        try:
            self._agent.set_llm(
                ChatOpenAI(
                    model=model,
                    api_key=api_key,
                    base_url=new_base_url,
                )
            )
        except Exception as e:
            self._agent.set_llm(old_llm)
            self._restore_context_snapshot(context_snapshot)
            self._print(f"[red]切换模型失败：{e}[/red]")
            return False

        self._print_switch_success(old_model=old_model, model=model, preflight=preflight)
        self._apply_manual_switch_state(
            preset_name=preset_name,
            manual=manual,
            auto_state=auto_state,
        )
        return True

    def _initialize_sticky_preset(self, auto_state: ModelAutoState) -> None:
        if auto_state.sticky_preset is None:
            auto_state.sticky_preset = self.resolve_current_preset_name()

    def _resolve_turn_state(
        self,
        auto_state: ModelAutoState,
    ) -> tuple[bool, str | None]:
        sticky_is_vision = self.preset_supports_vision(auto_state.sticky_preset)
        current_preset = (
            self.resolve_exact_current_preset_name() or self.resolve_current_preset_name()
        )
        return sticky_is_vision, current_preset

    async def _ensure_model_for_image_turn(
        self,
        *,
        auto_state: ModelAutoState,
        sticky_is_vision: bool,
        current_preset: str | None,
    ) -> bool:
        if sticky_is_vision:
            return True

        if current_preset and self.preset_supports_vision(current_preset):
            if not auto_state.auto_switched:
                auto_state.auto_switched = True
                auto_state.auto_from_preset = auto_state.sticky_preset
            return True

        vision_preset = self.resolve_vision_preset_name()
        if not vision_preset:
            self._print(
                "[red]图片输入需要视觉预设。请在 config/model_presets.json 中配置 `vision: true` 和 `auto_vision_preset`。[/red]"
            )
            return False

        self._print(f"[dim]自动切换到视觉预设：{vision_preset}[/dim]")
        switched = await self.switch_model_preset(
            vision_preset,
            manual=False,
            auto_state=auto_state,
        )
        if not switched:
            self._print("[red]切换到视觉模型失败，图片请求已中止。[/red]")
            return False

        auto_state.auto_switched = True
        auto_state.auto_from_preset = auto_state.sticky_preset
        return True

    async def _ensure_model_for_text_turn(
        self,
        *,
        auto_state: ModelAutoState,
        sticky_is_vision: bool,
    ) -> None:
        if not auto_state.auto_switched or sticky_is_vision:
            return

        target_preset = auto_state.auto_from_preset or auto_state.sticky_preset
        if not target_preset:
            self.clear_auto_switch_state(auto_state)
            return

        self._print(f"[dim]自动切回文本预设：{target_preset}[/dim]")
        switched = await self.switch_model_preset(
            target_preset,
            manual=False,
            auto_state=auto_state,
        )
        if switched:
            self.clear_auto_switch_state(auto_state)
            return

        self._print("[yellow]自动切回失败，将继续使用当前模型。[/yellow]")

    async def ensure_model_for_turn(
        self,
        *,
        has_image: bool,
        auto_state: ModelAutoState,
    ) -> bool:
        if not self._model_presets:
            return True

        self._initialize_sticky_preset(auto_state)
        sticky_is_vision, current_preset = self._resolve_turn_state(auto_state)
        if has_image:
            return await self._ensure_model_for_image_turn(
                auto_state=auto_state,
                sticky_is_vision=sticky_is_vision,
                current_preset=current_preset,
            )
        await self._ensure_model_for_text_turn(
            auto_state=auto_state,
            sticky_is_vision=sticky_is_vision,
        )
        return True
