"""Shared model-switching service for CLI commands and runtime hooks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from bu_agent_sdk import Agent
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.llm.messages import (
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    SystemMessage,
    UserMessage,
)

ModelPreset = dict[str, str | bool]


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
        current_preset = self.resolve_exact_current_preset_name() or self.resolve_current_preset_name()
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

    async def prepare_text_model_image_memory(self, *, manual: bool) -> None:
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
            self._print(
                f"[dim]Converting {total_images} historical image(s) into text memory via {summary_label}...[/dim]"
            )
        else:
            self._print(
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
                            summary_llm,
                            detail_text,
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

                new_parts.append(ContentPartTextParam(text=memory_text))
                changed = True

            if changed:
                msg.content = new_parts

        self._agent._context.rebuild_role_index()

        stripped = self._agent._context.strip_user_image_inputs()
        if stripped:
            self._print(
                f"[yellow]Removed {stripped} residual image part(s) after memory conversion.[/yellow]"
            )

        status_style = "yellow" if manual else "dim"
        self._print(
            f"[{status_style}]Converted {summarized_count}/{total_images} image(s) into text memory.[/{status_style}]"
        )
        if fallback_count:
            self._print(
                f"[{status_style}]Fallback memory used for {fallback_count} image(s).[/"
                f"{status_style}]"
            )

    async def switch_model_preset(
        self,
        preset_name: str,
        *,
        manual: bool = True,
        auto_state: ModelAutoState | None = None,
    ) -> bool:
        preset = self._model_presets.get(preset_name)
        if not preset:
            self._print(f"[red]Unknown preset: {preset_name}[/red]")
            self._print("[dim]Use /model list to see available presets.[/dim]")
            return False

        current_preset = self.resolve_exact_current_preset_name()
        current_is_vision = self.preset_supports_vision(current_preset)
        target_is_vision = bool(preset.get("vision", False))

        if current_preset == preset_name:
            if manual and auto_state is not None:
                auto_state.sticky_preset = preset_name
                self.clear_auto_switch_state(auto_state)
            self._print(f"[dim]Already using preset [cyan]{preset_name}[/cyan].[/dim]")
            return True

        model = str(preset["model"])
        api_key_env = str(preset.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.getenv(api_key_env)
        if not api_key:
            self._print(f"[red]Missing API key env var: {api_key_env}. Switch aborted.[/red]")
            return False

        context_snapshot: list[Any] | None = None
        if current_is_vision and not target_is_vision:
            context_snapshot = [
                message.model_copy(deep=True)
                for message in self._agent._context.get_messages()
            ]
            try:
                await self.prepare_text_model_image_memory(manual=manual)
            except Exception as e:
                self._agent._context.replace_messages(context_snapshot)
                self._print(
                    f"[red]Failed to prepare image memory before switch: {e}[/red]"
                )
                return False

        try:
            preflight = await self._agent.preflight_model_switch(model)
        except Exception as e:
            if context_snapshot is not None:
                self._agent._context.replace_messages(context_snapshot)
            self._print(f"[red]Model switch preflight failed: {e}[/red]")
            return False

        if not preflight.ok:
            if context_snapshot is not None:
                self._agent._context.replace_messages(context_snapshot)
            self._print(
                f"[red]Model switch preflight failed: {preflight.reason or 'context is too large'}[/red]"
            )
            self._print(
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
            self._print(f"[red]Failed to switch model: {e}[/red]")
            return False

        if preflight.compacted:
            self._print(
                "[yellow]Context was compacted before switching to fit target model.[/yellow]"
            )

        self._print(
            f"[green]Model switched:[/] [dim]{old_model}[/dim] -> [cyan]{model}[/cyan]"
        )
        self._print(
            f"[dim]Context preserved ({len(self._agent.messages)} messages).[/dim]"
        )
        if manual and auto_state is not None:
            auto_state.sticky_preset = preset_name
            self.clear_auto_switch_state(auto_state)
        return True

    async def ensure_model_for_turn(
        self,
        *,
        has_image: bool,
        auto_state: ModelAutoState,
    ) -> bool:
        if not self._model_presets:
            return True

        if auto_state.sticky_preset is None:
            auto_state.sticky_preset = self.resolve_current_preset_name()

        sticky_is_vision = self.preset_supports_vision(auto_state.sticky_preset)
        current_preset = self.resolve_exact_current_preset_name() or self.resolve_current_preset_name()

        if has_image:
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
                    "[red]Image input requires a vision preset. Configure `vision: true` and `auto_vision_preset` in config/model_presets.json.[/red]"
                )
                return False

            self._print(f"[dim]Auto switch to vision preset: {vision_preset}[/dim]")
            switched = await self.switch_model_preset(
                vision_preset,
                manual=False,
                auto_state=auto_state,
            )
            if not switched:
                self._print(
                    "[red]Failed to switch to vision model. Image request aborted.[/red]"
                )
                return False

            auto_state.auto_switched = True
            auto_state.auto_from_preset = auto_state.sticky_preset
            return True

        if not auto_state.auto_switched:
            return True
        if sticky_is_vision:
            return True

        target_preset = auto_state.auto_from_preset or auto_state.sticky_preset
        if not target_preset:
            self.clear_auto_switch_state(auto_state)
            return True

        self._print(f"[dim]Auto switch back to text preset: {target_preset}[/dim]")
        switched = await self.switch_model_preset(
            target_preset,
            manual=False,
            auto_state=auto_state,
        )
        if switched:
            self.clear_auto_switch_state(auto_state)
        else:
            self._print(
                "[yellow]Auto switch-back failed; continue with current model.[/yellow]"
            )
        return True
