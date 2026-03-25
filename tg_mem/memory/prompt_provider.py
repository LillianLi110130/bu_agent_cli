import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimePromptOverrides:
    fact_extraction_prompt: Optional[str] = None
    update_memory_prompt: Optional[str] = None


class UserPromptProvider:
    """Fetch per-user prompt overrides from an external service."""

    FACT_EXTRACTION_TYPES = {
        "fact_extraction",
        "memory_extraction",
        "extract",
        "extraction",
        "factextraction",
        "memoryextraction",
    }

    UPDATE_MEMORY_TYPES = {
        "memory_update",
        "update_memory",
        "update",
        "memoryupdate",
    }

    def __init__(self, service_url: Optional[str], timeout: float = 3.0):
        self.service_url = (service_url or "").strip() or None
        self.timeout = timeout

    @staticmethod
    def _normalize_user_id(user_id: Any) -> Optional[int]:
        if user_id is None:
            return None

        try:
            return int(str(user_id).strip())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_active_status(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False

        normalized = str(value).strip().lower()
        return normalized in {"1", "true", "active", "enabled"}

    @staticmethod
    def _normalize_prompt_type(value: Any) -> str:
        text = str(value or "").strip().lower()
        return text.replace("-", "_").replace(" ", "_")

    @staticmethod
    def _version_key(value: Any) -> Tuple[int, ...]:
        if value is None:
            return (0,)

        text = str(value).strip()
        if not text:
            return (0,)

        parts = re.findall(r"\d+", text)
        if not parts:
            logger.warning("Invalid prompt version '%s'. Treating as 0.", value)
            return (0,)

        return tuple(int(part) for part in parts)

    def _build_request_url(self, user_id: int) -> str:
        parsed = urlparse(self.service_url or "")
        query_items = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query_items["user_id"] = str(user_id)
        new_query = urlencode(query_items)
        return urlunparse(parsed._replace(query=new_query))

    def _fetch_prompt_list(self, user_id: int) -> List[Dict[str, Any]]:
        if not self.service_url:
            return []

        request_url = self._build_request_url(user_id)
        request = Request(request_url, method="GET")
        request.add_header("Accept", "application/json")

        with urlopen(request, timeout=self.timeout) as response:  # nosec B310 - user supplied URL is explicit config
            raw_body = response.read().decode("utf-8")
            payload = json.loads(raw_body)

        if isinstance(payload, list):
            data = payload
        elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
            data = payload["data"]
        else:
            logger.warning("Prompt service returned unexpected payload format. Falling back to defaults.")
            return []

        return [item for item in data if isinstance(item, dict)]

    def _select_best_prompt(
        self,
        prompts: Iterable[Dict[str, Any]],
        *,
        prompt_types: set[str],
    ) -> Optional[str]:
        best_item: Optional[Dict[str, Any]] = None
        best_version: Tuple[int, ...] = (0,)

        for prompt in prompts:
            prompt_type = self._normalize_prompt_type(prompt.get("promptType"))
            if prompt_type not in prompt_types:
                continue
            if not self._is_active_status(prompt.get("status")):
                continue

            prompt_content = prompt.get("promptContent")
            if not isinstance(prompt_content, str) or not prompt_content.strip():
                continue

            version_key = self._version_key(prompt.get("version"))
            if best_item is None or version_key > best_version:
                best_item = prompt
                best_version = version_key

        if best_item is None:
            return None

        return str(best_item.get("promptContent")).strip()

    def resolve_for_user(self, user_id: Any) -> RuntimePromptOverrides:
        if not self.service_url:
            return RuntimePromptOverrides()

        normalized_user_id = self._normalize_user_id(user_id)
        if normalized_user_id is None:
            logger.warning("Invalid user_id '%s' for prompt lookup. Falling back to defaults.", user_id)
            return RuntimePromptOverrides()

        try:
            prompt_list = self._fetch_prompt_list(normalized_user_id)
        except Exception as e:
            logger.warning("Failed to fetch prompt overrides for user_id=%s: %s", normalized_user_id, e)
            return RuntimePromptOverrides()

        return RuntimePromptOverrides(
            fact_extraction_prompt=self._select_best_prompt(
                prompt_list,
                prompt_types=self.FACT_EXTRACTION_TYPES,
            ),
            update_memory_prompt=self._select_best_prompt(
                prompt_list,
                prompt_types=self.UPDATE_MEMORY_TYPES,
            ),
        )

    async def resolve_for_user_async(self, user_id: Any) -> RuntimePromptOverrides:
        return await asyncio.to_thread(self.resolve_for_user, user_id)
