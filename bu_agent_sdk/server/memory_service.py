"""Memory service adapter for loading and persisting server session history."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Protocol

from bu_agent_sdk.llm.messages import AssistantMessage, BaseMessage, DeveloperMessage, UserMessage

logger = logging.getLogger("bu_agent_sdk.server.memory")


class MemoryServiceProtocol(Protocol):
    async def load_history(self, session_id: str, user_id: str) -> list[BaseMessage]:
        ...

    async def load_user_memory_context(self, user_id: str) -> DeveloperMessage | None:
        ...

    async def append_round(self, session_id: str, user_id: str, user_message: str, assistant_message: str) -> None:
        ...


class NoopMemoryService:
    async def load_history(self, session_id: str, user_id: str) -> list[BaseMessage]:
        logger.debug(
            f"Memory service disabled, skip history load for session_id={session_id}, user_id={user_id}"
        )
        return []

    async def load_user_memory_context(self, user_id: str) -> DeveloperMessage | None:
        logger.debug(f"Memory service disabled, skip user memory load for user_id={user_id}")
        return None

    async def append_round(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        logger.debug(
            f"Memory service disabled, skip round append for session_id={session_id}, user_id={user_id}"
        )


class TgMemMemoryService:
    """Adapter that translates server session history to tg_mem operations."""

    def __init__(self, memory, max_memory_context_chars: int = 20000) -> None:
        self.memory = memory
        self.max_memory_context_chars = max_memory_context_chars

    async def load_history(self, session_id: str, user_id: str) -> list[BaseMessage]:
        del user_id
        records = await asyncio.to_thread(
            self.memory.db.get_conversation_records,
            session_id=session_id,
        )
        return build_history_messages(records)

    async def load_user_memory_context(self, user_id: str) -> DeveloperMessage | None:
        records = await asyncio.to_thread(
            self.memory.db.list_memory_records,
            user_id=user_id,
            status="ACTE",
            limit=None,
        )
        return build_user_memory_context_message(
            records=records,
            max_chars=self.max_memory_context_chars,
        )

    async def append_round(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        await asyncio.to_thread(
            self.memory.add,
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ],
            user_id=user_id,
            session_id=session_id,
            infer=True,
        )


def build_history_messages(records: list[dict]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for record in records:
        role = str(record.get("role") or "").strip().lower()
        content = str(record.get("content") or "")
        if role == "user":
            messages.append(UserMessage(content=content))
            continue
        if role == "assistant":
            messages.append(AssistantMessage(content=content))
    return messages


def build_user_memory_context_message(
    records: list[dict],
    *,
    max_chars: int,
) -> DeveloperMessage | None:
    header = "User memory context:\n"
    if max_chars <= len(header):
        logger.warning(
            f"max_memory_context_chars={max_chars} is too small; skipping user memory injection"
        )
        return None

    parts: list[str] = []
    current_length = len(header)
    truncated = False

    for record in records:
        memory_text = str(record.get("memory") or record.get("memory_data") or "").strip()
        if not memory_text:
            continue

        line = f"- {memory_text}\n"
        if current_length + len(line) > max_chars:
            truncated = True
            break

        parts.append(line)
        current_length += len(line)

    if not parts:
        return None

    if truncated:
        logger.warning(
            f"User memory context truncated to {max_chars} characters to protect request context"
        )

    return DeveloperMessage(content=f"{header}{''.join(parts).rstrip()}")


def _is_disabled(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"0", "false", "no", "off", "disabled"}


def build_memory_service_from_env() -> TgMemMemoryService | None:
    if _is_disabled(os.getenv("TG_MEM_ENABLED")):
        logger.info("TG memory service is disabled by TG_MEM_ENABLED")
        return None

    db_uri = (os.getenv("TG_MEM_MYSQL_DB_URI") or "").strip()
    if not db_uri:
        logger.info("TG memory service is disabled because TG_MEM_MYSQL_DB_URI is not set")
        return None

    from tg_mem import Memory

    model = os.getenv("TG_MEM_LLM_MODEL") or os.getenv("LLM_MODEL") or "gpt-4.1-nano-2025-04-14"
    base_url = (
        os.getenv("TG_MEM_OPENAI_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("LLM_BASE_URL")
    )
    api_key = os.getenv("TG_MEM_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

    llm_config: dict[str, str] = {"model": model}
    if api_key:
        llm_config["api_key"] = api_key
    if base_url:
        llm_config["openai_base_url"] = base_url

    max_memory_context_chars = int(os.getenv("TG_MEM_MAX_MEMORY_CONTEXT_CHARS", "20000"))

    memory = Memory.from_config(
        {
            "vector_store": {"provider": "none", "config": None},
            "mysql": {"db_uri": db_uri},
            "llm": {
                "provider": "openai_like",
                "config": llm_config,
            },
        }
    )
    logger.info(
        f"Initialized TG memory service from environment with max_memory_context_chars={max_memory_context_chars}"
    )
    return TgMemMemoryService(
        memory=memory,
        max_memory_context_chars=max_memory_context_chars,
    )
