import asyncio
from pathlib import Path

from agent_core.llm.views import ChatInvokeCompletion
from cli.agents_handler import AgentSlashHandler


class PromptDraftLLM:
    model = "test-model"

    @property
    def provider(self) -> str:
        return "test"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):
        self.messages = messages
        self.tool_choice = tool_choice
        return ChatInvokeCompletion(content="```markdown\n你是代码审查 agent。\n```")

    async def astream(self, messages, tools=None, tool_choice=None, **kwargs):
        if False:
            yield None


class SlowPromptDraftLLM(PromptDraftLLM):
    def __init__(self) -> None:
        self.cancelled = False

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return ChatInvokeCompletion(content="不会返回")


class ChoicePrompter:
    def __init__(self) -> None:
        self.calls = []

    async def prompt_choice(self, prompt, choices, default=None):
        self.calls.append((prompt, choices, default))
        return default


class ConfirmPrompter(ChoicePrompter):
    def __init__(self, confirmed: bool) -> None:
        super().__init__()
        self.confirmed = confirmed
        self.confirm_calls = []

    async def prompt_yes_no(self, prompt, default=True):
        self.confirm_calls.append((prompt, default))
        return self.confirmed


def test_prompt_system_prompt_source_offers_llm_when_available(tmp_path: Path) -> None:
    prompter = ChoicePrompter()
    handler = AgentSlashHandler(
        workspace_root=tmp_path,
        llm=PromptDraftLLM(),
        prompter=prompter,
        model_presets={},
    )

    choice = asyncio.run(handler._prompt_system_prompt_source())

    assert choice == "大模型生成"
    assert prompter.calls == [
        ("[5/5] 选择系统提示词草稿来源", ["默认模板", "大模型生成"], "大模型生成")
    ]


def test_reload_registry_does_not_request_system_prompt_refresh(tmp_path: Path) -> None:
    handler = AgentSlashHandler(workspace_root=tmp_path, model_presets={})

    assert handler.system_prompt_refresh_requested is False

    handler._reload_registry()

    assert handler.system_prompt_refresh_requested is False


def test_reload_command_requests_system_prompt_refresh(tmp_path: Path) -> None:
    prompter = ConfirmPrompter(confirmed=True)
    handler = AgentSlashHandler(workspace_root=tmp_path, prompter=prompter, model_presets={})

    handled = asyncio.run(handler._reload())

    assert handled is True
    assert handler.system_prompt_refresh_requested is True
    assert prompter.confirm_calls == [
        ("/agents reload 会重建主 agent 系统提示词并重置当前会话上下文，确认继续吗？", False)
    ]


def test_reload_command_can_be_cancelled_without_refresh(tmp_path: Path) -> None:
    prompter = ConfirmPrompter(confirmed=False)
    handler = AgentSlashHandler(workspace_root=tmp_path, prompter=prompter, model_presets={})

    handled = asyncio.run(handler._reload())

    assert handled is True
    assert handler.system_prompt_refresh_requested is False


def test_prompt_system_prompt_source_uses_template_without_llm(tmp_path: Path) -> None:
    prompter = ChoicePrompter()
    handler = AgentSlashHandler(
        workspace_root=tmp_path,
        llm=None,
        prompter=prompter,
        model_presets={},
    )

    choice = asyncio.run(handler._prompt_system_prompt_source())

    assert choice == "默认模板"
    assert prompter.calls == [("[5/5] 选择系统提示词草稿来源", ["默认模板"], "默认模板")]


def test_generate_default_system_prompt_uses_llm(tmp_path: Path) -> None:
    llm = PromptDraftLLM()
    handler = AgentSlashHandler(workspace_root=tmp_path, llm=llm, model_presets={})

    async def wait_forever():
        await asyncio.Event().wait()

    handler._wait_for_system_prompt_generation_cancel = wait_forever

    prompt = asyncio.run(
        handler._generate_default_system_prompt(
            name="code-reviewer",
            description="重点关注 bug、回归风险和测试缺口",
            model="inherit",
            tools=["read", "grep"],
            disallowed_tools=[],
        )
    )

    assert prompt == "你是代码审查 agent。"
    assert llm.tool_choice == "none"
    assert "code-reviewer" in llm.messages[1].text
    assert "read, grep" in llm.messages[1].text


def test_generate_default_system_prompt_cancel_falls_back_to_template(tmp_path: Path) -> None:
    llm = SlowPromptDraftLLM()
    handler = AgentSlashHandler(workspace_root=tmp_path, llm=llm, model_presets={})

    async def cancel_immediately():
        return True

    handler._wait_for_system_prompt_generation_cancel = cancel_immediately

    prompt = asyncio.run(
        handler._generate_default_system_prompt(
            name="code-reviewer",
            description="重点关注 bug、回归风险和测试缺口",
            model="inherit",
            tools=["read", "grep"],
            disallowed_tools=[],
        )
    )

    assert "你是 code-reviewer agent。" in prompt
    assert "重点关注 bug、回归风险和测试缺口" in prompt
    assert "职责边界" in prompt
    assert "输出格式" in prompt
    assert llm.cancelled is True


def test_generate_default_system_prompt_falls_back_without_llm(tmp_path: Path) -> None:
    handler = AgentSlashHandler(workspace_root=tmp_path, llm=None, model_presets={})

    prompt = asyncio.run(
        handler._generate_default_system_prompt(
            name="qa-agent",
            description="验证发布质量",
            model="inherit",
            tools=None,
            disallowed_tools=["bash"],
        )
    )

    assert "你是 qa-agent agent。" in prompt
    assert "验证发布质量" in prompt
    assert "禁用工具：bash" in prompt
    assert "不要声称完成未经验证的事项" in prompt
    assert "结论：一句话说明结果或建议" in prompt
