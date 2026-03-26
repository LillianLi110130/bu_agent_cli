import asyncio
import concurrent
import gc
import hashlib
import json
import logging
import uuid
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional

import pytz
from pydantic import ValidationError

from tg_mem.configs.base import MemoryConfig, MemoryItem
from tg_mem.configs.enums import MemoryType
from tg_mem.configs.prompts import (
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    get_update_memory_messages,
)
from tg_mem.exceptions import ValidationError as Mem0ValidationError
from tg_mem.memory.base import MemoryBase
from tg_mem.memory.prompt_provider import RuntimePromptOverrides, UserPromptProvider
from tg_mem.memory.setup import setup_config
from tg_mem.memory.storage import MySQLManager
from tg_mem.memory.utils import (
    extract_json,
    get_fact_retrieval_messages,
    parse_messages,
    parse_vision_messages,
    remove_code_blocks,
)
from tg_mem.utils.factory import (
    EmbedderFactory,
    GraphStoreFactory,
    LlmFactory,
    VectorStoreFactory,
    RerankerFactory,
)

# Suppress SWIG deprecation warnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")

# Initialize logger early for util functions
logger = logging.getLogger(__name__)

def _build_filters_and_metadata(
    *,  # Enforce keyword-only arguments
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    actor_id: Optional[str] = None,  # For query-time filtering
    input_metadata: Optional[Dict[str, Any]] = None,
    input_filters: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Constructs metadata for storage and filters for querying based on session and actor identifiers.

    This helper supports multiple session identifiers (`user_id`, `agent_id`, and/or `run_id`)
    for flexible session scoping and optionally narrows queries to a specific `actor_id`. It returns two dicts:

    1. `base_metadata_template`: Used as a template for metadata when storing new memories.
       It includes all provided session identifier(s) and any `input_metadata`.
    2. `effective_query_filters`: Used for querying existing memories. It includes all
       provided session identifier(s), any `input_filters`, and a resolved actor
       identifier for targeted filtering if specified by any actor-related inputs.

    Actor filtering precedence: explicit `actor_id` arg → `filters["actor_id"]`
    This resolved actor ID is used for querying but is not added to `base_metadata_template`,
    as the actor for storage is typically derived from message content at a later stage.

    Args:
        user_id (Optional[str]): User identifier, for session scoping.
        agent_id (Optional[str]): Agent identifier, for session scoping.
        run_id (Optional[str]): Run identifier, for session scoping.
        actor_id (Optional[str]): Explicit actor identifier, used as a potential source for
            actor-specific filtering. See actor resolution precedence in the main description.
        input_metadata (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session identifiers for the storage metadata template. Defaults to an empty dict.
        input_filters (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session and actor identifiers for query filters. Defaults to an empty dict.

    Returns:
        tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - base_metadata_template (Dict[str, Any]): Metadata template for storing memories,
              scoped to the provided session(s).
            - effective_query_filters (Dict[str, Any]): Filters for querying memories,
              scoped to the provided session(s) and potentially a resolved actor.
    """

    base_metadata_template = deepcopy(input_metadata) if input_metadata else {}
    effective_query_filters = deepcopy(input_filters) if input_filters else {}

    # ---------- add all provided session ids ----------
    session_ids_provided = []

    if user_id:
        base_metadata_template["user_id"] = user_id
        effective_query_filters["user_id"] = user_id
        session_ids_provided.append("user_id")

    if agent_id:
        base_metadata_template["agent_id"] = agent_id
        effective_query_filters["agent_id"] = agent_id
        session_ids_provided.append("agent_id")

    if run_id:
        base_metadata_template["run_id"] = run_id
        effective_query_filters["run_id"] = run_id
        session_ids_provided.append("run_id")

    if not session_ids_provided:
        raise Mem0ValidationError(
            message="At least one of 'user_id', 'agent_id', or 'run_id' must be provided.",
            error_code="VALIDATION_001",
            details={"provided_ids": {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}},
            suggestion="Please provide at least one identifier to scope the memory operation."
        )

    # ---------- optional actor filter ----------
    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    if resolved_actor_id:
        effective_query_filters["actor_id"] = resolved_actor_id

    return base_metadata_template, effective_query_filters


def _resolve_session_id(session_id: Optional[Any]) -> str:
    """Validate and normalize the explicit session identifier for raw message persistence."""
    if session_id is None:
        raise ValueError("session_id is required")

    resolved_session_id = str(session_id).strip()
    if not resolved_session_id:
        raise ValueError("session_id is required")

    return resolved_session_id


def _build_fact_retrieval_prompt(
    messages,
    metadata,
    custom_fact_extraction_prompt,
    should_use_agent_memory_extraction,
):
    parsed_messages = parse_messages(messages)

    if custom_fact_extraction_prompt:
        return custom_fact_extraction_prompt, f"Input:\n{parsed_messages}"

    is_agent_memory = should_use_agent_memory_extraction(messages, metadata)
    return get_fact_retrieval_messages(parsed_messages, is_agent_memory)


def _extract_facts_from_llm_response(response: str, *, error_log_prefix: str):
    try:
        response = remove_code_blocks(response)
        if not response.strip():
            return []

        try:
            return json.loads(response)["facts"]
        except json.JSONDecodeError:
            extracted_json = extract_json(response)
            return json.loads(extracted_json)["facts"]
    except Exception as e:
        logger.error(f"{error_log_prefix}: {e}")
        return []


def _extract_memory_actions_from_llm_response(
    response: str,
    *,
    empty_response_warning: str,
    invalid_json_error: str,
):
    try:
        if not response or not response.strip():
            logger.warning(empty_response_warning)
            return {}

        response = remove_code_blocks(response)
        return json.loads(response)
    except Exception as e:
        logger.error(f"{invalid_json_error}: {e}")
        return {}


def _build_session_search_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    search_filters: Dict[str, Any] = {}
    if filters.get("user_id"):
        search_filters["user_id"] = filters["user_id"]
    if filters.get("agent_id"):
        search_filters["agent_id"] = filters["agent_id"]
    if filters.get("run_id"):
        search_filters["run_id"] = filters["run_id"]
    return search_filters


def _build_mysql_old_memory_context(existing_memories):
    retrieved_old_memory = []
    temp_uuid_mapping = {}

    for idx, mem in enumerate(existing_memories):
        memory_id = mem.get("id", mem.get("memory_id"))
        if memory_id is None:
            continue

        retrieved_old_memory.append(
            {
                "id": str(idx),
                "text": mem.get("memory_data") or "",
            }
        )
        temp_uuid_mapping[str(idx)] = memory_id

    return retrieved_old_memory, temp_uuid_mapping


def _remap_old_memory_ids_for_llm(retrieved_old_memory):
    temp_uuid_mapping = {}
    for idx, item in enumerate(retrieved_old_memory):
        temp_uuid_mapping[str(idx)] = item["id"]
        retrieved_old_memory[idx]["id"] = str(idx)

    return temp_uuid_mapping


def _normalize_memory_timestamp(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _build_infer_metadata(messages, metadata, should_use_agent_memory_extraction):
    inferred_metadata = deepcopy(metadata)
    if "role" not in inferred_metadata:
        inferred_metadata["role"] = (
            "assistant" if should_use_agent_memory_extraction(messages, metadata) else "user"
        )
    return inferred_metadata


def _resolve_runtime_prompt_overrides(config: MemoryConfig, user_id: Optional[Any]) -> RuntimePromptOverrides:
    if not user_id:
        return RuntimePromptOverrides()

    prompt_provider = UserPromptProvider(
        service_url=getattr(config, "prompt_service_url", None),
        timeout=float(getattr(config, "prompt_service_timeout", 3.0)),
    )
    return prompt_provider.resolve_for_user(user_id)


async def _resolve_runtime_prompt_overrides_async(
    config: MemoryConfig,
    user_id: Optional[Any],
) -> RuntimePromptOverrides:
    if not user_id:
        return RuntimePromptOverrides()

    prompt_provider = UserPromptProvider(
        service_url=getattr(config, "prompt_service_url", None),
        timeout=float(getattr(config, "prompt_service_timeout", 30.0)),
    )
    return await prompt_provider.resolve_for_user_async(user_id)


setup_config()
logger = logging.getLogger(__name__)


def _create_mysql_manager(config: MemoryConfig) -> MySQLManager:
    db_uri = getattr(config.mysql, "db_uri", None) or getattr(config, "history_db_uri", None)

    return MySQLManager(
        db_uri=db_uri,
        pool_size=getattr(config.mysql, "pool_size", 5),
        max_overflow=getattr(config.mysql, "max_overflow", 10),
        pool_timeout=getattr(config.mysql, "pool_timeout", 30),
        pool_recycle=getattr(config.mysql, "pool_recycle", 3600),
        pool_pre_ping=getattr(config.mysql, "pool_pre_ping", True),
    )


class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.custom_fact_extraction_prompt = getattr(self.config, "custom_fact_extraction_prompt", None)
        self.custom_update_memory_prompt = getattr(self.config, "custom_update_memory_prompt", None)
        self.db = _create_mysql_manager(self.config)
        self.api_version = self.config.version
        self.collection_name = "mysql_only"

        self.embedding_model = None
        self.vector_store = None
        self.llm = None
        self._vector_store_available = False

        vector_provider = getattr(self.config.vector_store, "provider", None)
        vector_config = getattr(self.config.vector_store, "config", None)

        if vector_provider and vector_provider != "none" and vector_config is not None:
            try:
                self.embedding_model = EmbedderFactory.create(
                    self.config.embedder.provider,
                    self.config.embedder.config,
                    vector_config,
                )
                self.vector_store = VectorStoreFactory.create(vector_provider, vector_config)
                self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
                self.collection_name = getattr(vector_config, "collection_name", self.collection_name)
                self._vector_store_available = True
            except Exception as e:
                logger.warning(
                    f"Vector stack initialization failed ({e}). Falling back to MySQL-only mode."
                )
                self.embedding_model = None
                self.vector_store = None
                self.llm = None
                self._vector_store_available = False
        else:
            logger.info("Vector store is disabled. Running in MySQL-only mode.")

        # Initialize reranker if configured
        self.reranker = None
        if config.reranker and self._vector_store_available:
            self.reranker = RerankerFactory.create(
                config.reranker.provider,
                config.reranker.config,
            )
        elif config.reranker:
            logger.warning("Reranker is configured but vector store is unavailable. Skipping reranker init.")

        self.enable_graph = False

        if self.config.graph_store.config:
            provider = self.config.graph_store.provider
            try:
                self.graph = GraphStoreFactory.create(provider, self.config)
                self.enable_graph = True
            except Exception as e:
                logger.warning(f"Graph store initialization failed ({e}). Disabling graph features.")
                self.graph = None
        else:
            self.graph = None


    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            processed_config = cls._process_config(config_dict)
            config = MemoryConfig(**processed_config)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        processed_config = deepcopy(config_dict)

        # Allow MySQL-only mode when vector_store is omitted in runtime config.
        vector_store_config = processed_config.get("vector_store")
        if "vector_store" not in processed_config or vector_store_config in (None, {}):
            processed_config["vector_store"] = {"provider": "none", "config": None}

        return processed_config

    def _should_use_agent_memory_extraction(self, messages, metadata):
        """Determine whether to use agent memory extraction based on the logic:
        - If agent_id is present and messages contain assistant role -> True
        - Otherwise -> False
        
        Args:
            messages: List of message dictionaries
            metadata: Metadata containing user_id, agent_id, etc.
            
        Returns:
            bool: True if should use agent memory extraction, False for user memory extraction
        """
        # Check if agent_id is present in metadata
        has_agent_id = metadata.get("agent_id") is not None
        
        # Check if there are assistant role messages
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)
        
        # Use agent memory extraction if agent_id is present and there are assistant messages
        return has_agent_id and has_assistant_messages

    def _is_vector_store_enabled(self) -> bool:
        return bool(
            getattr(self, "_vector_store_available", False)
            and getattr(self, "vector_store", None) is not None
            and getattr(self, "embedding_model", None) is not None
            and getattr(self, "llm", None) is not None
        )

    def _extract_new_facts(self, messages, metadata, *, custom_fact_extraction_prompt, error_log_prefix):
        system_prompt, user_prompt = _build_fact_retrieval_prompt(
            messages=messages,
            metadata=metadata,
            custom_fact_extraction_prompt=custom_fact_extraction_prompt,
            should_use_agent_memory_extraction=self._should_use_agent_memory_extraction,
        )

        response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        return _extract_facts_from_llm_response(response, error_log_prefix=error_log_prefix)

    def _resolve_memory_actions(
        self,
        retrieved_old_memory,
        new_retrieved_facts,
        *,
        custom_update_memory_prompt,
        response_error_log,
        empty_response_warning,
        invalid_json_error,
    ):
        if not new_retrieved_facts:
            return {}

        function_calling_prompt = get_update_memory_messages(
            retrieved_old_memory,
            new_retrieved_facts,
            custom_update_memory_prompt,
        )

        try:
            response = self.llm.generate_response(
                messages=[{"role": "user", "content": function_calling_prompt}],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error(f"{response_error_log}: {e}")
            response = ""

        return _extract_memory_actions_from_llm_response(
            response,
            empty_response_warning=empty_response_warning,
            invalid_json_error=invalid_json_error,
        )

    def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
        channel: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """
        Create a new memory.

        Adds new memories scoped to a single session id (e.g. `user_id`, `agent_id`, or `run_id`). One of those ids is required.

        Args:
            messages (str or List[Dict[str, str]]): The message content or list of messages
                (e.g., `[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]`)
                to be processed and stored.
            user_id (str, optional): ID of the user creating the memory. Defaults to None.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            session_id (str, optional): Stable session identifier for raw record persistence. Required.
            channel (str, optional): Session initiation channel for raw session persistence. Defaults to "".
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            infer (bool, optional): If True (default), an LLM is used to extract key facts from
                'messages' and decide whether to add, update, or delete related memories.
                If False, 'messages' are added as raw memories directly.
            memory_type (str, optional): Specifies the type of memory. Currently, only
                `MemoryType.PROCEDURAL.value` ("procedural_memory") is explicitly handled for
                creating procedural memories (typically requires 'agent_id'). Otherwise, memories
                are treated as general conversational/factual memories.memory_type (str, optional): Type of memory to create. Defaults to None. By default, it creates the short term memories and long term (semantic and episodic) memories. Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.


        Returns:
            dict: A dictionary containing the result of the memory addition operation, typically
                  including a list of memory items affected (added, updated) under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "event": "ADD"}]}`

        Raises:
            Mem0ValidationError: If input validation fails (invalid memory_type, messages format, etc.).
            VectorStoreError: If vector store operations fail.
            GraphStoreError: If graph store operations fail.
            EmbeddingError: If embedding generation fails.
            LLMError: If LLM operations fail.
            DatabaseError: If database operations fail.
        """

        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise Mem0ValidationError(
                message=f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories.",
                error_code="VALIDATION_002",
                details={"provided_type": memory_type, "valid_type": MemoryType.PROCEDURAL.value},
                suggestion=f"Use '{MemoryType.PROCEDURAL.value}' to create procedural memories."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise Mem0ValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )

        resolved_session_id = _resolve_session_id(session_id)
        processed_metadata = deepcopy(processed_metadata)
        processed_metadata["session_id"] = resolved_session_id

        if not processed_metadata.get("user_id"):
            raise ValueError("user_id is required")

        self.db.add_conversation_records(
            session_id=resolved_session_id,
            messages=messages,
            user_id=processed_metadata.get("user_id"),
            agent_id=processed_metadata.get("agent_id"),
            run_id=processed_metadata.get("run_id"),
            channel=channel,
        )

        vector_enabled = self._is_vector_store_enabled()

        runtime_prompt_overrides = RuntimePromptOverrides()
        inferred_metadata = processed_metadata
        if infer:
            runtime_prompt_overrides = _resolve_runtime_prompt_overrides(
                self.config,
                processed_metadata.get("user_id"),
            )
            inferred_metadata = _build_infer_metadata(
                messages,
                processed_metadata,
                self._should_use_agent_memory_extraction,
            )

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            if not vector_enabled:
                logger.warning(
                    "Procedural memory requested but vector stack is unavailable. "
                    "Only raw conversation is persisted to MySQL."
                )
                return {"results": []}
            results = self._create_procedural_memory(messages, metadata=processed_metadata, prompt=prompt)
            return results

        vector_store_result = []
        graph_result = []
        rdb_store_result= []

        if infer and not vector_enabled:
            try:
                rdb_store_result = self._add_to_mysql_only_store(
                    messages,
                    inferred_metadata,
                    effective_filters,
                    custom_fact_extraction_prompt=(
                        runtime_prompt_overrides.fact_extraction_prompt
                        or getattr(self.config, "custom_fact_extraction_prompt", None)
                    ),
                    custom_update_memory_prompt=(
                        runtime_prompt_overrides.update_memory_prompt
                        or getattr(self.config, "custom_update_memory_prompt", None)
                    ),
                )
            except Exception as e:
                logger.warning(f"MySQL-only inferred add failed; preserving raw conversation only. Error: {e}")
                rdb_store_result = []

            if self.enable_graph:
                try:
                    graph_result = self._add_to_graph(messages, effective_filters)
                except Exception as e:
                    logger.warning(f"Graph store add failed; graph relation update skipped. Error: {e}")
                    graph_result = []
                return {
                    "results": rdb_store_result,
                    "relations": graph_result,
                }

            return {"results": rdb_store_result}

        if vector_enabled:
            if self.config.llm.config.get("enable_vision"):
                messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
            else:
                messages = parse_vision_messages(messages)

        if vector_enabled and self.enable_graph:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future1 = executor.submit(
                    self._add_to_vector_store,
                    messages,
                    inferred_metadata,
                    effective_filters,
                    infer,
                    runtime_prompt_overrides.fact_extraction_prompt or getattr(self.config, "custom_fact_extraction_prompt", None),
                    runtime_prompt_overrides.update_memory_prompt or getattr(self.config, "custom_update_memory_prompt", None),
                )
                future2 = executor.submit(self._add_to_graph, messages, effective_filters)

                concurrent.futures.wait([future1, future2])

                try:
                    vector_store_result = future1.result()
                except Exception as e:
                    logger.warning(f"Vector store add failed; preserving only MySQL record. Error: {e}")
                    vector_store_result = []

                try:
                    graph_result = future2.result()
                except Exception as e:
                    logger.warning(f"Graph store add failed; graph relation update skipped. Error: {e}")
                    graph_result = []
        else:
            if vector_enabled:
                try:
                    vector_store_result = self._add_to_vector_store(
                        messages,
                        inferred_metadata,
                        effective_filters,
                        infer,
                        runtime_prompt_overrides.fact_extraction_prompt or getattr(self.config, "custom_fact_extraction_prompt", None),
                        runtime_prompt_overrides.update_memory_prompt or getattr(self.config, "custom_update_memory_prompt", None),
                    )
                except Exception as e:
                    logger.warning(f"Vector store add failed; preserving only MySQL record. Error: {e}")
                    vector_store_result = []

            if self.enable_graph:
                try:
                    graph_result = self._add_to_graph(messages, effective_filters)
                except Exception as e:
                    logger.warning(f"Graph store add failed; graph relation update skipped. Error: {e}")
                    graph_result = []

        if self.enable_graph:
            return {
                "results": vector_store_result,
                "relations": graph_result,
            }

        return {"results": vector_store_result}

    def _add_to_vector_store(
        self,
        messages,
        metadata,
        filters,
        infer,
        custom_fact_extraction_prompt=None,
        custom_update_memory_prompt=None,
    ):
        if custom_fact_extraction_prompt is None:
            custom_fact_extraction_prompt = getattr(self.config, "custom_fact_extraction_prompt", None)
        if custom_update_memory_prompt is None:
            custom_update_memory_prompt = getattr(self.config, "custom_update_memory_prompt", None)

        if not infer:
            returned_memories = []
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    logger.warning(f"Skipping invalid message format: {message_dict}")
                    continue

                if message_dict["role"] == "system":
                    continue

                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]

                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name

                msg_content = message_dict["content"]
                msg_embeddings = self.embedding_model.embed(msg_content, "add")
                mem_id = self._create_memory(msg_content, msg_embeddings, per_msg_meta)

                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            return returned_memories

        new_retrieved_facts = self._extract_new_facts(
            messages,
            metadata,
            custom_fact_extraction_prompt=custom_fact_extraction_prompt,
            error_log_prefix="Error in new_retrieved_facts",
        )

        if not new_retrieved_facts:
            logger.debug("No new facts retrieved from input. Skipping memory update LLM call.")

        retrieved_old_memory = []
        new_message_embeddings = {}
        search_filters = _build_session_search_filters(filters)

        for new_mem in new_retrieved_facts:
            messages_embeddings = self.embedding_model.embed(new_mem, "add")
            new_message_embeddings[new_mem] = messages_embeddings
            existing_memories = self.vector_store.search(
                query=new_mem,
                vectors=messages_embeddings,
                limit=5,
                filters=search_filters,
            )
            for mem in existing_memories:
                retrieved_old_memory.append({"id": mem.id, "text": mem.payload.get("data", "")})

        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item
        retrieved_old_memory = list(unique_data.values())
        logger.info(f"Total existing memories: {len(retrieved_old_memory)}")

        # mapping UUIDs with integers for handling UUID hallucinations
        temp_uuid_mapping = _remap_old_memory_ids_for_llm(retrieved_old_memory)

        new_memories_with_actions = self._resolve_memory_actions(
            retrieved_old_memory,
            new_retrieved_facts,
            custom_update_memory_prompt=custom_update_memory_prompt,
            response_error_log="Error in new memory actions response",
            empty_response_warning="Empty response from LLM, no memories to extract",
            invalid_json_error="Invalid JSON response",
        )

        returned_memories = []
        try:
            for resp in new_memories_with_actions.get("memory", []):
                logger.info(resp)
                try:
                    action_text = resp.get("text")
                    if not action_text:
                        logger.info("Skipping memory entry because of empty `text` field.")
                        continue

                    event_type = resp.get("event")
                    if event_type == "ADD":
                        memory_id = self._create_memory(
                            data=action_text,
                            existing_embeddings=new_message_embeddings,
                            metadata=deepcopy(metadata),
                        )
                        returned_memories.append({"id": memory_id, "memory": action_text, "event": event_type})
                    elif event_type == "UPDATE":
                        self._update_memory(
                            memory_id=temp_uuid_mapping[resp.get("id")],
                            data=action_text,
                            existing_embeddings=new_message_embeddings,
                            metadata=deepcopy(metadata),
                        )
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp.get("id")],
                                "memory": action_text,
                                "event": event_type,
                                "previous_memory": resp.get("old_memory"),
                            }
                        )
                    elif event_type == "DELETE":
                        self._delete_memory(memory_id=temp_uuid_mapping[resp.get("id")])
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp.get("id")],
                                "memory": action_text,
                                "event": event_type,
                            }
                        )
                    elif event_type == "NONE":
                        # Even if content doesn't need updating, update session IDs if provided
                        memory_id = temp_uuid_mapping.get(resp.get("id"))
                        if memory_id and (metadata.get("agent_id") or metadata.get("run_id")):
                            # Update only the session identifiers, keep content the same
                            existing_memory = self.vector_store.get(vector_id=memory_id)
                            updated_metadata = deepcopy(existing_memory.payload)
                            if metadata.get("agent_id"):
                                updated_metadata["agent_id"] = metadata["agent_id"]
                            if metadata.get("run_id"):
                                updated_metadata["run_id"] = metadata["run_id"]
                            updated_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

                            self.vector_store.update(
                                vector_id=memory_id,
                                vector=None,  # Keep same embeddings
                                payload=updated_metadata,
                            )
                            logger.info(f"Updated session IDs for memory {memory_id}")
                        else:
                            logger.info("NOOP for Memory.")
                except Exception as e:
                    logger.error(f"Error processing memory action: {resp}, Error: {e}")
        except Exception as e:
            logger.error(f"Error iterating new_memories_with_actions: {e}")

        return returned_memories

    def _ensure_llm_for_inference(self) -> bool:
        if self.llm is not None:
            return True

        llm_config = getattr(self.config, "llm", None)
        llm_provider = getattr(llm_config, "provider", None)
        llm_provider_config = getattr(llm_config, "config", None)

        if not llm_provider:
            logger.warning("LLM provider is not configured; infer mode is unavailable in MySQL-only path.")
            return False

        try:
            self.llm = LlmFactory.create(llm_provider, llm_provider_config)
        except Exception as e:
            logger.warning(f"LLM initialization failed in MySQL-only path: {e}")
            self.llm = None

        return self.llm is not None

    def _add_to_mysql_only_store(
        self,
        messages,
        metadata,
        filters,
        custom_fact_extraction_prompt=None,
        custom_update_memory_prompt=None,
    ):
        if custom_fact_extraction_prompt is None:
            custom_fact_extraction_prompt = getattr(self.config, "custom_fact_extraction_prompt", None)
        if custom_update_memory_prompt is None:
            custom_update_memory_prompt = getattr(self.config, "custom_update_memory_prompt", None)

        if not self._ensure_llm_for_inference():
            return []

        new_retrieved_facts = self._extract_new_facts(
            messages,
            metadata,
            custom_fact_extraction_prompt=custom_fact_extraction_prompt,
            error_log_prefix="Error in mysql-only new_retrieved_facts",
        )

        if not new_retrieved_facts:
            logger.debug("No new facts retrieved from input in MySQL-only path.")
            return []

        if filters.get("user_id"):
            existing_memories = self.db.list_memory_records(
                user_id=filters.get("user_id"),
                status="ACTE",
                limit=100,
            )
        else:
            existing_memories = []

        retrieved_old_memory, temp_uuid_mapping = _build_mysql_old_memory_context(existing_memories)

        new_memories_with_actions = self._resolve_memory_actions(
            retrieved_old_memory,
            new_retrieved_facts,
            custom_update_memory_prompt=custom_update_memory_prompt,
            response_error_log="Error in mysql-only memory actions response",
            empty_response_warning="Empty response from LLM in mysql-only path, no memories to extract",
            invalid_json_error="Invalid JSON response in mysql-only path",
        )

        returned_memories = []
        for resp in new_memories_with_actions.get("memory", []):
            try:
                event_type = resp.get("event")
                action_text = resp.get("text")

                if event_type in {"ADD", "UPDATE"} and not action_text:
                    logger.info("Skipping memory entry because of empty `text` field in mysql-only path.")
                    continue

                if event_type == "ADD":
                    memory_id = self._create_memory_mysql_only(action_text, metadata=deepcopy(metadata))
                    returned_memories.append({"id": memory_id, "memory": action_text, "event": event_type})
                elif event_type == "UPDATE":
                    memory_id = temp_uuid_mapping.get(resp.get("id"))
                    if not memory_id:
                        logger.info("Skipping UPDATE because target memory id was not resolved in mysql-only path.")
                        continue
                    self._update_memory_mysql_only(memory_id, action_text, metadata=deepcopy(metadata))
                    returned_memories.append(
                        {
                            "id": memory_id,
                            "memory": action_text,
                            "event": event_type,
                            "previous_memory": resp.get("old_memory"),
                        }
                    )
                elif event_type == "DELETE":
                    memory_id = temp_uuid_mapping.get(resp.get("id"))
                    if not memory_id:
                        logger.info("Skipping DELETE because target memory id was not resolved in mysql-only path.")
                        continue
                    self._delete_memory_mysql_only(memory_id, metadata=deepcopy(metadata))
                    returned_memories.append(
                        {
                            "id": memory_id,
                            "memory": action_text,
                            "event": event_type,
                        }
                    )
                elif event_type == "NONE":
                    logger.info("NOOP for Memory in mysql-only path.")
            except Exception as e:
                logger.error(f"Error processing mysql-only memory action: {resp}, Error: {e}")

        return returned_memories

    def _create_memory_mysql_only(self, data, metadata=None):
        metadata = metadata or {}
        created_at = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        memory_id = self.db.create_memory_record(
            memory_data=data,
            user_id=metadata.get("user_id"),
            agent_id=metadata.get("agent_id"),
            run_id=metadata.get("run_id"),
            memory_type=metadata.get("memory_type") or MemoryType.SEMANTIC.value,
            status="ACTE",
            created_at=created_at,
            updated_at=created_at,
        )
        self.db.add_history(
            memory_id,
            None,
            data,
            "ADD",
            created_at=created_at,
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
        )
        return memory_id

    def _update_memory_mysql_only(self, memory_id, data, metadata=None):
        existing_record = self.db.get_memory_record(memory_id)
        if not existing_record:
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        metadata = metadata or {}
        prev_value = existing_record.get("memory_data")
        created_at = existing_record.get("created_at")
        updated_at = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        self.db.update_memory_record(
            memory_id,
            memory_data=data,
            user_id=metadata.get("user_id") or existing_record.get("user_id"),
            agent_id=metadata.get("agent_id"),
            run_id=metadata.get("run_id"),
            memory_type=metadata.get("memory_type") or existing_record.get("memory_type") or MemoryType.SEMANTIC.value,
            status="ACTE",
            created_at=created_at,
            updated_at=updated_at,
        )
        self.db.add_history(
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=created_at,
            updated_at=updated_at,
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
        )
        return memory_id

    def _delete_memory_mysql_only(self, memory_id, metadata=None):
        existing_record = self.db.get_memory_record(memory_id)
        if not existing_record:
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        metadata = metadata or {}
        prev_value = existing_record.get("memory_data", "")
        created_at = existing_record.get("created_at")
        updated_at = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        self.db.update_memory_record(
            memory_id,
            memory_data=prev_value,
            user_id=metadata.get("user_id") or existing_record.get("user_id"),
            agent_id=metadata.get("agent_id"),
            run_id=metadata.get("run_id"),
            memory_type=metadata.get("memory_type") or existing_record.get("memory_type") or MemoryType.SEMANTIC.value,
            status="INAC",
            created_at=created_at,
            updated_at=updated_at,
        )
        self.db.add_history(
            memory_id,
            prev_value,
            None,
            "DELETE",
            created_at=created_at,
            updated_at=updated_at,
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
            is_deleted=1,
        )
        return memory_id

    def _search_mysql_store(self, query, filters, limit, threshold: Optional[float] = None):
        if not filters.get("user_id"):
            return []

        records = self.db.list_memory_records(
            user_id=filters.get("user_id"),
            status="ACTE",
            limit=limit,
        )

        query_text = (query or "").strip().lower()
        memories = []
        for record in records:
            memory_text = record.get("memory_data") or ""
            score = 1.0 if not query_text or query_text in memory_text.lower() else 0.5

            if threshold is not None and score < threshold:
                continue

            memory_item = MemoryItem(
                id=str(record.get("id", record.get("memory_id"))),
                memory=memory_text,
                hash=None,
                created_at=_normalize_memory_timestamp(record.get("created_at")),
                updated_at=_normalize_memory_timestamp(record.get("updated_at")),
                score=score,
            ).model_dump()
            if record.get("user_id") is not None:
                memory_item["user_id"] = record.get("user_id")
            memories.append(memory_item)

        return memories

    def _add_to_graph(self, messages, filters):
        added_entities = []
        if self.enable_graph:
            if filters.get("user_id") is None:
                filters["user_id"] = "user"

            data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])
            added_entities = self.graph.add(data, filters)

        return added_entities

    def get(self, memory_id):
        """
        Retrieve a memory by ID.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        memory = self.vector_store.get(vector_id=memory_id)
        if not memory:
            return None

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        return result_item

    def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ):
        """
        List all memories.

        Args:
            user_id (str, optional): user id
            agent_id (str, optional): agent id
            run_id (str, optional): run id
            filters (dict, optional): Additional custom key-value filters to apply to the search.
                These are merged with the ID-based scoping filters. For example,
                `filters={"actor_id": "some_user"}`.
            limit (int, optional): The maximum number of memories to return. Defaults to 100.

        Returns:
            dict: A dictionary containing a list of memories under the "results" key,
                  and potentially "relations" if graph store is enabled. For API v1.0,
                  it might return a direct list (see deprecation warning).
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`
        """

        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be specified.")


        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._get_all_from_vector_store, effective_filters, limit)
            future_graph_entities = (
                executor.submit(self.graph.get_all, effective_filters, limit) if self.enable_graph else None
            )

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            all_memories_result = future_memories.result()
            graph_entities_result = future_graph_entities.result() if future_graph_entities else None

        if self.enable_graph:
            return {"results": all_memories_result, "relations": graph_entities_result}

        return {"results": all_memories_result}

    def _get_all_from_vector_store(self, filters, limit):
        memories_result = self.vector_store.list(filters=filters, limit=limit)

        # Handle different vector store return formats by inspecting first element
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            first_element = memories_result[0]

            # If first element is a container, unwrap one level
            if isinstance(first_element, (list, tuple)):
                actual_memories = first_element
            else:
                # First element is a memory object, structure is already flat
                actual_memories = memories_result
        else:
            actual_memories = memories_result

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        rerank: bool = True,
    ):
        """
        Searches for memories based on a query
        Args:
            query (str): Query to search for.
            user_id (str, optional): ID of the user to search for. Defaults to None.
            agent_id (str, optional): ID of the agent to search for. Defaults to None.
            run_id (str, optional): ID of the run to search for. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to 100.
            filters (dict, optional): Legacy filters to apply to the search. Defaults to None.
            threshold (float, optional): Minimum score for a memory to be included in the results. Defaults to None.
            filters (dict, optional): Enhanced metadata filtering with operators:
                - {"key": "value"} - exact match
                - {"key": {"eq": "value"}} - equals
                - {"key": {"ne": "value"}} - not equals  
                - {"key": {"in": ["val1", "val2"]}} - in list
                - {"key": {"nin": ["val1", "val2"]}} - not in list
                - {"key": {"gt": 10}} - greater than
                - {"key": {"gte": 10}} - greater than or equal
                - {"key": {"lt": 10}} - less than
                - {"key": {"lte": 10}} - less than or equal
                - {"key": {"contains": "text"}} - contains text
                - {"key": {"icontains": "text"}} - case-insensitive contains
                - {"key": "*"} - wildcard match (any value)
                - {"AND": [filter1, filter2]} - logical AND
                - {"OR": [filter1, filter2]} - logical OR
                - {"NOT": [filter1]} - logical NOT

        Returns:
            dict: A dictionary containing the search results, typically under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`
        """
        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be specified.")

        # Apply enhanced metadata filtering if advanced operators are detected
        if filters and self._has_advanced_operators(filters):
            processed_filters = self._process_metadata_filters(filters)
            effective_filters.update(processed_filters)
        elif filters:
            # Simple filters, merge directly
            effective_filters.update(filters)


        if not self._is_vector_store_enabled():
            original_memories = self._search_mysql_store(query, effective_filters, limit, threshold)
            graph_entities = self.graph.search(query, effective_filters, limit) if self.enable_graph else None
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_memories = executor.submit(self._search_vector_store, query, effective_filters, limit, threshold)
                future_graph_entities = (
                    executor.submit(self.graph.search, query, effective_filters, limit) if self.enable_graph else None
                )

                concurrent.futures.wait(
                    [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
                )

                original_memories = future_memories.result()
                graph_entities = future_graph_entities.result() if future_graph_entities else None

        # Apply reranking if enabled and reranker is available
        if rerank and self.reranker and original_memories:
            try:
                reranked_memories = self.reranker.rerank(query, original_memories, limit)
                original_memories = reranked_memories
            except Exception as e:
                logger.warning(f"Reranking failed, using original results: {e}")

        if self.enable_graph:
            return {"results": original_memories, "relations": graph_entities}

        return {"results": original_memories}

    def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced metadata filters and convert them to vector store compatible format.
        
        Args:
            metadata_filters: Enhanced metadata filters with operators
            
        Returns:
            Dict of processed filters compatible with vector store
        """
        processed_filters = {}
        
        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                return {key: condition}
            
            result = {}
            for operator, value in condition.items():
                # Map platform operators to universal format that can be translated by each vector store
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte", 
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains"
                }
                
                if operator in operator_map:
                    result[key] = {operator_map[operator]: value}
                else:
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            return result
        
        for key, value in metadata_filters.items():
            if key == "AND":
                # Logical AND: combine multiple conditions
                if not isinstance(value, list):
                    raise ValueError("AND operator requires a list of conditions")
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        processed_filters.update(process_condition(sub_key, sub_value))
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                processed_filters["$or"] = []
                for condition in value:
                    or_condition = {}
                    for sub_key, sub_value in condition.items():
                        or_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$or"].append(or_condition)
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                processed_filters["$not"] = []
                for condition in value:
                    not_condition = {}
                    for sub_key, sub_value in condition.items():
                        not_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$not"].append(not_condition)
            else:
                processed_filters.update(process_condition(key, value))
        
        return processed_filters

    def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
        """
        Check if filters contain advanced operators that need special processing.
        
        Args:
            filters: Dictionary of filters to check
            
        Returns:
            bool: True if advanced operators are detected
        """
        if not isinstance(filters, dict):
            return False
            
        for key, value in filters.items():
            # Check for platform-style logical operators
            if key in ["AND", "OR", "NOT"]:
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            if isinstance(value, dict):
                for op in value.keys():
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        return True
            # Check for wildcard values
            if value == "*":
                return True
        return False

    def _search_vector_store(self, query, filters, limit, threshold: Optional[float] = None):
        embeddings = self.embedding_model.embed(query, "search")
        memories = self.vector_store.search(query=query, vectors=embeddings, limit=limit, filters=filters)

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        original_memories = []
        for mem in memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
                score=mem.score,
            ).model_dump()

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            if threshold is None or mem.score >= threshold:
                original_memories.append(memory_item_dict)

        return original_memories

    def update(self, memory_id, data):
        """
        Update a memory by ID.

        Args:
            memory_id (str): ID of the memory to update.
            data (str): New content to update the memory with.

        Returns:
            dict: Success message indicating the memory was updated.

        Example:
            >>> m.update(memory_id="mem_123", data="Likes to play tennis on weekends")
            {'message': 'Memory updated successfully!'}
        """

        existing_embeddings = {data: self.embedding_model.embed(data, "update")}

        self._update_memory(memory_id, data, existing_embeddings)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        self._delete_memory(memory_id)
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None):
        """
        Delete all memories.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        # delete all vector memories and reset the collections
        memories = self.vector_store.list(filters=filters)[0]
        for memory in memories:
            self._delete_memory(memory.id)
        self.vector_store.reset()

        logger.info(f"Deleted {len(memories)} memories")

        if self.enable_graph:
            self.graph.delete_all(filters)

        return {"message": "Memories deleted successfully!"}

    def history(self, memory_id):
        """
        Get the history of changes for a memory by ID.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        return self.db.get_history(memory_id)

    def _create_memory(self, data, existing_embeddings, metadata=None):
        logger.debug(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, memory_action="add")
        memory_id = uuid.uuid4().hex
        metadata = metadata or {}
        metadata["data"] = data
        metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[metadata],
        )
        return memory_id

    def _create_procedural_memory(self, messages, metadata=None, prompt=None):
        """
        Create a procedural memory

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {
                "role": "user",
                "content": "Create procedural memory of the above conversation.",
            },
        ]

        try:
            procedural_memory = self.llm.generate_response(messages=parsed_messages)
            procedural_memory = remove_code_blocks(procedural_memory)
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata["memory_type"] = MemoryType.PROCEDURAL.value
        embeddings = self.embedding_model.embed(procedural_memory, memory_action="add")
        memory_id = self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        new_metadata = deepcopy(metadata) if metadata is not None else {}

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        # Preserve session identifiers from existing memory only if not provided in new metadata
        if "user_id" not in new_metadata and "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" not in new_metadata and "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" not in new_metadata and "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]
        if "actor_id" not in new_metadata and "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        if "role" not in new_metadata and "role" in existing_memory.payload:
            new_metadata["role"] = existing_memory.payload["role"]
        if "memory_type" not in new_metadata and "memory_type" in existing_memory.payload:
            new_metadata["memory_type"] = existing_memory.payload["memory_type"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, "update")

        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        return memory_id

    def _delete_memory(self, memory_id):
        logger.info(f"Deleting memory with {memory_id=}")
        self.vector_store.get(vector_id=memory_id)
        self.vector_store.delete(vector_id=memory_id)
        return memory_id

    def reset(self):
        """
        Reset the memory store by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")

        if self.db:
            self.db.reset()
            self.db.close()

        self.db = _create_mysql_manager(self.config)

        if self._is_vector_store_enabled():
            try:
                if hasattr(self.vector_store, "reset"):
                    self.vector_store = VectorStoreFactory.reset(self.vector_store)
                else:
                    logger.warning("Vector store does not support reset. Recreating collection.")
                    self.vector_store.delete_col()
                    self.vector_store = VectorStoreFactory.create(
                        self.config.vector_store.provider, self.config.vector_store.config
                    )
            except Exception as e:
                logger.warning(f"Vector reset failed. MySQL reset completed; vector state unchanged. Error: {e}")
        else:
            logger.warning("Vector store is unavailable. Skipping vector reset.")

    def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")


class AsyncMemory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.db = _create_mysql_manager(self.config)
        self.api_version = self.config.version
        self.collection_name = "mysql_only"

        self.embedding_model = None
        self.vector_store = None
        self.llm = None
        self._vector_store_available = False

        vector_provider = getattr(self.config.vector_store, "provider", None)
        vector_config = getattr(self.config.vector_store, "config", None)

        if vector_provider and vector_provider != "none" and vector_config is not None:
            try:
                self.embedding_model = EmbedderFactory.create(
                    self.config.embedder.provider,
                    self.config.embedder.config,
                    vector_config,
                )
                self.vector_store = VectorStoreFactory.create(vector_provider, vector_config)
                self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
                self.collection_name = getattr(vector_config, "collection_name", self.collection_name)
                self._vector_store_available = True
            except Exception as e:
                logger.warning(
                    f"Vector stack initialization failed ({e}). Falling back to MySQL-only mode."
                )
                self.embedding_model = None
                self.vector_store = None
                self.llm = None
                self._vector_store_available = False
        else:
            logger.info("Vector store is disabled. Running in MySQL-only mode.")

        # Initialize reranker if configured
        self.reranker = None
        if config.reranker and self._vector_store_available:
            self.reranker = RerankerFactory.create(
                config.reranker.provider,
                config.reranker.config,
            )
        elif config.reranker:
            logger.warning("Reranker is configured but vector store is unavailable. Skipping reranker init.")

        self.enable_graph = False

        if self.config.graph_store.config:
            provider = self.config.graph_store.provider
            try:
                self.graph = GraphStoreFactory.create(provider, self.config)
                self.enable_graph = True
            except Exception as e:
                logger.warning(f"Graph store initialization failed ({e}). Disabling graph features.")
                self.graph = None
        else:
            self.graph = None


    @classmethod
    async def from_config(cls, config_dict: Dict[str, Any]):
        try:
            processed_config = cls._process_config(config_dict)
            config = MemoryConfig(**processed_config)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        processed_config = deepcopy(config_dict)

        # Allow MySQL-only mode when vector_store is omitted in runtime config.
        vector_store_config = processed_config.get("vector_store")
        if "vector_store" not in processed_config or vector_store_config in (None, {}):
            processed_config["vector_store"] = {"provider": "none", "config": None}

        return processed_config

    def _should_use_agent_memory_extraction(self, messages, metadata):
        """Determine whether to use agent memory extraction based on the logic:
        - If agent_id is present and messages contain assistant role -> True
        - Otherwise -> False
        
        Args:
            messages: List of message dictionaries
            metadata: Metadata containing user_id, agent_id, etc.
            
        Returns:
            bool: True if should use agent memory extraction, False for user memory extraction
        """
        # Check if agent_id is present in metadata
        has_agent_id = metadata.get("agent_id") is not None
        
        # Check if there are assistant role messages
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)
        
        # Use agent memory extraction if agent_id is present and there are assistant messages
        return has_agent_id and has_assistant_messages

    def _is_vector_store_enabled(self) -> bool:
        return bool(
            getattr(self, "_vector_store_available", False)
            and getattr(self, "vector_store", None) is not None
            and getattr(self, "embedding_model", None) is not None
            and getattr(self, "llm", None) is not None
        )

    async def _extract_new_facts(self, messages, metadata, *, custom_fact_extraction_prompt, error_log_prefix):
        system_prompt, user_prompt = _build_fact_retrieval_prompt(
            messages=messages,
            metadata=metadata,
            custom_fact_extraction_prompt=custom_fact_extraction_prompt,
            should_use_agent_memory_extraction=self._should_use_agent_memory_extraction,
        )

        response = await asyncio.to_thread(
            self.llm.generate_response,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        return _extract_facts_from_llm_response(response, error_log_prefix=error_log_prefix)

    async def _resolve_memory_actions(
        self,
        retrieved_old_memory,
        new_retrieved_facts,
        *,
        custom_update_memory_prompt,
        response_error_log,
        empty_response_warning,
        invalid_json_error,
    ):
        if not new_retrieved_facts:
            return {}

        function_calling_prompt = get_update_memory_messages(
            retrieved_old_memory,
            new_retrieved_facts,
            custom_update_memory_prompt,
        )

        try:
            response = await asyncio.to_thread(
                self.llm.generate_response,
                messages=[{"role": "user", "content": function_calling_prompt}],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error(f"{response_error_log}: {e}")
            response = ""

        return _extract_memory_actions_from_llm_response(
            response,
            empty_response_warning=empty_response_warning,
            invalid_json_error=invalid_json_error,
        )

    async def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
        channel: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
        llm=None,
    ):
        """
        Create a new memory asynchronously.

        Args:
            messages (str or List[Dict[str, str]]): Messages to store in the memory.
            user_id (str, optional): ID of the user creating the memory.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            session_id (str, optional): Stable session identifier for raw record persistence. Required.
            channel (str, optional): Session initiation channel for raw session persistence. Defaults to "".
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            infer (bool, optional): Whether to infer the memories. Defaults to True.
            memory_type (str, optional): Type of memory to create. Defaults to None.
                                         Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.
            llm (BaseChatModel, optional): LLM class to use for generating procedural memories. Defaults to None. Useful when user is using LangChain ChatModel.
        Returns:
            dict: A dictionary containing the result of the memory addition operation.
        """
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_metadata=metadata
        )

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise ValueError(
                f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise Mem0ValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )

        resolved_session_id = _resolve_session_id(session_id)
        processed_metadata = deepcopy(processed_metadata)
        processed_metadata["session_id"] = resolved_session_id

        if not processed_metadata.get("user_id"):
            raise ValueError("user_id is required")

        await asyncio.to_thread(
            self.db.add_conversation_records,
            session_id=resolved_session_id,
            messages=messages,
            user_id=processed_metadata.get("user_id"),
            agent_id=processed_metadata.get("agent_id"),
            run_id=processed_metadata.get("run_id"),
            channel=channel,
        )

        vector_enabled = self._is_vector_store_enabled()

        runtime_prompt_overrides = RuntimePromptOverrides()
        inferred_metadata = processed_metadata
        if infer:
            runtime_prompt_overrides = await _resolve_runtime_prompt_overrides_async(
                self.config,
                processed_metadata.get("user_id"),
            )
            inferred_metadata = _build_infer_metadata(
                messages,
                processed_metadata,
                self._should_use_agent_memory_extraction,
            )

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            if not vector_enabled:
                logger.warning(
                    "Procedural memory requested but vector stack is unavailable. "
                    "Only raw conversation is persisted to MySQL."
                )
                return {"results": []}
            results = await self._create_procedural_memory(
                messages, metadata=processed_metadata, prompt=prompt, llm=llm
            )
            return results

        vector_store_result = []
        graph_result = []

        if infer and not vector_enabled:
            try:
                vector_store_result = await self._add_to_mysql_only_store(
                    messages,
                    inferred_metadata,
                    effective_filters,
                    custom_fact_extraction_prompt=(
                        runtime_prompt_overrides.fact_extraction_prompt
                        or getattr(self.config, "custom_fact_extraction_prompt", None)
                    ),
                    custom_update_memory_prompt=(
                        runtime_prompt_overrides.update_memory_prompt
                        or getattr(self.config, "custom_update_memory_prompt", None)
                    ),
                )
            except Exception as e:
                logger.warning(f"MySQL-only inferred add failed; preserving raw conversation only. Error: {e}")
                vector_store_result = []

            if self.enable_graph:
                try:
                    graph_result = await self._add_to_graph(messages, effective_filters)
                except Exception as e:
                    logger.warning(f"Graph store add failed; graph relation update skipped. Error: {e}")
                    graph_result = []

            if self.enable_graph:
                return {
                    "results": vector_store_result,
                    "relations": graph_result,
                }

            return {"results": vector_store_result}

        if vector_enabled:
            if self.config.llm.config.get("enable_vision"):
                messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
            else:
                messages = parse_vision_messages(messages)

        vector_store_task = (
            asyncio.create_task(
                self._add_to_vector_store(
                    messages,
                    inferred_metadata,
                    effective_filters,
                    infer,
                    runtime_prompt_overrides.fact_extraction_prompt or getattr(self.config, "custom_fact_extraction_prompt", None),
                    runtime_prompt_overrides.update_memory_prompt or getattr(self.config, "custom_update_memory_prompt", None),
                )
            )
            if vector_enabled
            else None
        )
        graph_task = asyncio.create_task(self._add_to_graph(messages, effective_filters)) if self.enable_graph else None

        if vector_store_task and graph_task:
            vector_result_raw, graph_result_raw = await asyncio.gather(
                vector_store_task,
                graph_task,
                return_exceptions=True,
            )

            if isinstance(vector_result_raw, Exception):
                logger.warning(
                    f"Vector store add failed; preserving only MySQL record. Error: {vector_result_raw}"
                )
            else:
                vector_store_result = vector_result_raw

            if isinstance(graph_result_raw, Exception):
                logger.warning(f"Graph store add failed; graph relation update skipped. Error: {graph_result_raw}")
            else:
                graph_result = graph_result_raw

        elif vector_store_task:
            try:
                vector_store_result = await vector_store_task
            except Exception as e:
                logger.warning(f"Vector store add failed; preserving only MySQL record. Error: {e}")

        elif graph_task:
            try:
                graph_result = await graph_task
            except Exception as e:
                logger.warning(f"Graph store add failed; graph relation update skipped. Error: {e}")

        if self.enable_graph:
            return {
                "results": vector_store_result,
                "relations": graph_result,
            }

        return {"results": vector_store_result}

    async def _add_to_vector_store(
        self,
        messages: list,
        metadata: dict,
        effective_filters: dict,
        infer: bool,
        custom_fact_extraction_prompt=None,
        custom_update_memory_prompt=None,
    ):
        if custom_fact_extraction_prompt is None:
            custom_fact_extraction_prompt = getattr(self.config, "custom_fact_extraction_prompt", None)
        if custom_update_memory_prompt is None:
            custom_update_memory_prompt = getattr(self.config, "custom_update_memory_prompt", None)

        if not infer:
            returned_memories = []
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    logger.warning(f"Skipping invalid message format (async): {message_dict}")
                    continue

                if message_dict["role"] == "system":
                    continue

                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]

                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name

                msg_content = message_dict["content"]
                msg_embeddings = await asyncio.to_thread(self.embedding_model.embed, msg_content, "add")
                mem_id = await self._create_memory(msg_content, msg_embeddings, per_msg_meta)

                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            return returned_memories

        new_retrieved_facts = await self._extract_new_facts(
            messages,
            metadata,
            custom_fact_extraction_prompt=custom_fact_extraction_prompt,
            error_log_prefix="Error in new_retrieved_facts",
        )

        if not new_retrieved_facts:
            logger.debug("No new facts retrieved from input. Skipping memory update LLM call.")

        retrieved_old_memory = []
        new_message_embeddings = {}
        search_filters = _build_session_search_filters(effective_filters)

        async def process_fact_for_search(new_mem_content):
            embeddings = await asyncio.to_thread(self.embedding_model.embed, new_mem_content, "add")
            new_message_embeddings[new_mem_content] = embeddings
            existing_mems = await asyncio.to_thread(
                self.vector_store.search,
                query=new_mem_content,
                vectors=embeddings,
                limit=5,
                filters=search_filters,
            )
            return [{"id": mem.id, "text": mem.payload.get("data", "")} for mem in existing_mems]

        search_tasks = [process_fact_for_search(fact) for fact in new_retrieved_facts]
        search_results_list = await asyncio.gather(*search_tasks)
        for result_group in search_results_list:
            retrieved_old_memory.extend(result_group)

        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item
        retrieved_old_memory = list(unique_data.values())
        logger.info(f"Total existing memories: {len(retrieved_old_memory)}")
        temp_uuid_mapping = _remap_old_memory_ids_for_llm(retrieved_old_memory)

        new_memories_with_actions = await self._resolve_memory_actions(
            retrieved_old_memory,
            new_retrieved_facts,
            custom_update_memory_prompt=custom_update_memory_prompt,
            response_error_log="Error in new memory actions response",
            empty_response_warning="Empty response from LLM, no memories to extract",
            invalid_json_error="Invalid JSON response",
        )

        returned_memories = []
        try:
            memory_tasks = []
            for resp in new_memories_with_actions.get("memory", []):
                logger.info(resp)
                try:
                    action_text = resp.get("text")
                    if not action_text:
                        continue
                    event_type = resp.get("event")

                    if event_type == "ADD":
                        task = asyncio.create_task(
                            self._create_memory(
                                data=action_text,
                                existing_embeddings=new_message_embeddings,
                                metadata=deepcopy(metadata),
                            )
                        )
                        memory_tasks.append((task, resp, "ADD", None))
                    elif event_type == "UPDATE":
                        task = asyncio.create_task(
                            self._update_memory(
                                memory_id=temp_uuid_mapping[resp["id"]],
                                data=action_text,
                                existing_embeddings=new_message_embeddings,
                                metadata=deepcopy(metadata),
                            )
                        )
                        memory_tasks.append((task, resp, "UPDATE", temp_uuid_mapping[resp["id"]]))
                    elif event_type == "DELETE":
                        task = asyncio.create_task(self._delete_memory(memory_id=temp_uuid_mapping[resp.get("id")]))
                        memory_tasks.append((task, resp, "DELETE", temp_uuid_mapping[resp.get("id")]))
                    elif event_type == "NONE":
                        # Even if content doesn't need updating, update session IDs if provided
                        memory_id = temp_uuid_mapping.get(resp.get("id"))
                        if memory_id and (metadata.get("agent_id") or metadata.get("run_id")):
                            # Create async task to update only the session identifiers
                            async def update_session_ids(mem_id, meta):
                                existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=mem_id)
                                updated_metadata = deepcopy(existing_memory.payload)
                                if meta.get("agent_id"):
                                    updated_metadata["agent_id"] = meta["agent_id"]
                                if meta.get("run_id"):
                                    updated_metadata["run_id"] = meta["run_id"]
                                updated_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

                                await asyncio.to_thread(
                                    self.vector_store.update,
                                    vector_id=mem_id,
                                    vector=None,  # Keep same embeddings
                                    payload=updated_metadata,
                                )
                                logger.info(f"Updated session IDs for memory {mem_id} (async)")

                            task = asyncio.create_task(update_session_ids(memory_id, metadata))
                            memory_tasks.append((task, resp, "NONE", memory_id))
                        else:
                            logger.info("NOOP for Memory (async).")
                except Exception as e:
                    logger.error(f"Error processing memory action (async): {resp}, Error: {e}")

            for task, resp, event_type, mem_id in memory_tasks:
                try:
                    result_id = await task
                    if event_type == "ADD":
                        returned_memories.append({"id": result_id, "memory": resp.get("text"), "event": event_type})
                    elif event_type == "UPDATE":
                        returned_memories.append(
                            {
                                "id": mem_id,
                                "memory": resp.get("text"),
                                "event": event_type,
                                "previous_memory": resp.get("old_memory"),
                            }
                        )
                    elif event_type == "DELETE":
                        returned_memories.append({"id": mem_id, "memory": resp.get("text"), "event": event_type})
                except Exception as e:
                    logger.error(f"Error awaiting memory task (async): {e}")
        except Exception as e:
            logger.error(f"Error in memory processing loop (async): {e}")

        return returned_memories

    async def _ensure_llm_for_inference(self) -> bool:
        if self.llm is not None:
            return True

        llm_config = getattr(self.config, "llm", None)
        llm_provider = getattr(llm_config, "provider", None)
        llm_provider_config = getattr(llm_config, "config", None)

        if not llm_provider:
            logger.warning("LLM provider is not configured; infer mode is unavailable in MySQL-only path.")
            return False

        try:
            self.llm = await asyncio.to_thread(LlmFactory.create, llm_provider, llm_provider_config)
        except Exception as e:
            logger.warning(f"LLM initialization failed in MySQL-only async path: {e}")
            self.llm = None

        return self.llm is not None

    async def _add_to_mysql_only_store(
        self,
        messages,
        metadata,
        filters,
        custom_fact_extraction_prompt=None,
        custom_update_memory_prompt=None,
    ):
        if custom_fact_extraction_prompt is None:
            custom_fact_extraction_prompt = getattr(self.config, "custom_fact_extraction_prompt", None)
        if custom_update_memory_prompt is None:
            custom_update_memory_prompt = getattr(self.config, "custom_update_memory_prompt", None)

        if not await self._ensure_llm_for_inference():
            return []

        new_retrieved_facts = await self._extract_new_facts(
            messages,
            metadata,
            custom_fact_extraction_prompt=custom_fact_extraction_prompt,
            error_log_prefix="Error in mysql-only async new_retrieved_facts",
        )

        if not new_retrieved_facts:
            logger.debug("No new facts retrieved from input in mysql-only async path.")
            return []

        if filters.get("user_id"):
            existing_memories = await asyncio.to_thread(
                self.db.list_memory_records,
                user_id=filters.get("user_id"),
                status="ACTE",
                limit=100,
            )
        else:
            existing_memories = []

        retrieved_old_memory, temp_uuid_mapping = _build_mysql_old_memory_context(existing_memories)

        new_memories_with_actions = await self._resolve_memory_actions(
            retrieved_old_memory,
            new_retrieved_facts,
            custom_update_memory_prompt=custom_update_memory_prompt,
            response_error_log="Error in mysql-only async memory actions response",
            empty_response_warning="Empty response from LLM in mysql-only async path, no memories to extract",
            invalid_json_error="Invalid JSON response in mysql-only async path",
        )

        returned_memories = []
        for resp in new_memories_with_actions.get("memory", []):
            try:
                event_type = resp.get("event")
                action_text = resp.get("text")

                if event_type in {"ADD", "UPDATE"} and not action_text:
                    logger.info("Skipping memory entry because of empty `text` field in mysql-only async path.")
                    continue

                if event_type == "ADD":
                    memory_id = await self._create_memory_mysql_only(action_text, metadata=deepcopy(metadata))
                    returned_memories.append({"id": memory_id, "memory": action_text, "event": event_type})
                elif event_type == "UPDATE":
                    memory_id = temp_uuid_mapping.get(resp.get("id"))
                    if not memory_id:
                        logger.info("Skipping UPDATE because target memory id was not resolved in mysql-only async path.")
                        continue
                    await self._update_memory_mysql_only(memory_id, action_text, metadata=deepcopy(metadata))
                    returned_memories.append(
                        {
                            "id": memory_id,
                            "memory": action_text,
                            "event": event_type,
                            "previous_memory": resp.get("old_memory"),
                        }
                    )
                elif event_type == "DELETE":
                    memory_id = temp_uuid_mapping.get(resp.get("id"))
                    if not memory_id:
                        logger.info("Skipping DELETE because target memory id was not resolved in mysql-only async path.")
                        continue
                    await self._delete_memory_mysql_only(memory_id, metadata=deepcopy(metadata))
                    returned_memories.append(
                        {
                            "id": memory_id,
                            "memory": action_text,
                            "event": event_type,
                        }
                    )
                elif event_type == "NONE":
                    logger.info("NOOP for Memory in mysql-only async path.")
            except Exception as e:
                logger.error(f"Error processing mysql-only async memory action: {resp}, Error: {e}")

        return returned_memories

    async def _create_memory_mysql_only(self, data, metadata=None):
        metadata = metadata or {}
        created_at = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        memory_id = await asyncio.to_thread(
            self.db.create_memory_record,
            memory_data=data,
            user_id=metadata.get("user_id"),
            agent_id=metadata.get("agent_id"),
            run_id=metadata.get("run_id"),
            memory_type=metadata.get("memory_type") or MemoryType.SEMANTIC.value,
            status="ACTE",
            created_at=created_at,
            updated_at=created_at,
        )
        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            None,
            data,
            "ADD",
            created_at=created_at,
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
        )
        return memory_id

    async def _update_memory_mysql_only(self, memory_id, data, metadata=None):
        existing_record = await asyncio.to_thread(self.db.get_memory_record, memory_id)
        if not existing_record:
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        metadata = metadata or {}
        prev_value = existing_record.get("memory_data")
        created_at = existing_record.get("created_at")
        updated_at = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        await asyncio.to_thread(
            self.db.update_memory_record,
            memory_id,
            memory_data=data,
            user_id=metadata.get("user_id") or existing_record.get("user_id"),
            agent_id=metadata.get("agent_id"),
            run_id=metadata.get("run_id"),
            memory_type=metadata.get("memory_type") or existing_record.get("memory_type") or MemoryType.SEMANTIC.value,
            status="ACTE",
            created_at=created_at,
            updated_at=updated_at,
        )
        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=created_at,
            updated_at=updated_at,
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
        )
        return memory_id

    async def _delete_memory_mysql_only(self, memory_id, metadata=None):
        existing_record = await asyncio.to_thread(self.db.get_memory_record, memory_id)
        if not existing_record:
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        metadata = metadata or {}
        prev_value = existing_record.get("memory_data", "")
        created_at = existing_record.get("created_at")
        updated_at = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        await asyncio.to_thread(
            self.db.update_memory_record,
            memory_id,
            memory_data=prev_value,
            user_id=metadata.get("user_id") or existing_record.get("user_id"),
            agent_id=metadata.get("agent_id"),
            run_id=metadata.get("run_id"),
            memory_type=metadata.get("memory_type") or existing_record.get("memory_type") or MemoryType.SEMANTIC.value,
            status="INAC",
            created_at=created_at,
            updated_at=updated_at,
        )
        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            None,
            "DELETE",
            created_at=created_at,
            updated_at=updated_at,
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
            is_deleted=1,
        )
        return memory_id

    async def _search_mysql_store(self, query, filters, limit, threshold: Optional[float] = None):
        if not filters.get("user_id"):
            return []

        records = await asyncio.to_thread(
            self.db.list_memory_records,
            user_id=filters.get("user_id"),
            status="ACTE",
            limit=limit,
        )

        query_text = (query or "").strip().lower()
        memories = []
        for record in records:
            memory_text = record.get("memory_data") or ""
            score = 1.0 if not query_text or query_text in memory_text.lower() else 0.5

            if threshold is not None and score < threshold:
                continue

            memory_item = MemoryItem(
                id=str(record.get("id", record.get("memory_id"))),
                memory=memory_text,
                hash=None,
                created_at=_normalize_memory_timestamp(record.get("created_at")),
                updated_at=_normalize_memory_timestamp(record.get("updated_at")),
                score=score,
            ).model_dump()
            if record.get("user_id") is not None:
                memory_item["user_id"] = record.get("user_id")
            memories.append(memory_item)

        return memories

    async def _add_to_graph(self, messages, filters):
        added_entities = []
        if self.enable_graph:
            if filters.get("user_id") is None:
                filters["user_id"] = "user"

            data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])
            added_entities = await asyncio.to_thread(self.graph.add, data, filters)

        return added_entities

    async def get(self, memory_id):
        """
        Retrieve a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        if not memory:
            return None

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        return result_item

    async def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ):
        """
        List all memories.

         Args:
             user_id (str, optional): user id
             agent_id (str, optional): agent id
             run_id (str, optional): run id
             filters (dict, optional): Additional custom key-value filters to apply to the search.
                 These are merged with the ID-based scoping filters. For example,
                 `filters={"actor_id": "some_user"}`.
             limit (int, optional): The maximum number of memories to return. Defaults to 100.

         Returns:
             dict: A dictionary containing a list of memories under the "results" key,
                   and potentially "relations" if graph store is enabled. For API v1.0,
                   it might return a direct list (see deprecation warning).
                   Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`
        """

        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "At least one of 'user_id', 'agent_id', or 'run_id' must be specified for get_all."
            )


        vector_store_task = asyncio.create_task(self._get_all_from_vector_store(effective_filters, limit))

        graph_task = None
        if self.enable_graph:
            graph_get_all = getattr(self.graph, "get_all", None)
            if callable(graph_get_all):
                if asyncio.iscoroutinefunction(graph_get_all):
                    graph_task = asyncio.create_task(graph_get_all(effective_filters, limit))
                else:
                    graph_task = asyncio.create_task(asyncio.to_thread(graph_get_all, effective_filters, limit))

        results_dict = {}
        if graph_task:
            vector_store_result, graph_entities_result = await asyncio.gather(vector_store_task, graph_task)
            results_dict.update({"results": vector_store_result, "relations": graph_entities_result})
        else:
            results_dict.update({"results": await vector_store_task})

        return results_dict

    async def _get_all_from_vector_store(self, filters, limit):
        memories_result = await asyncio.to_thread(self.vector_store.list, filters=filters, limit=limit)

        # Handle different vector store return formats by inspecting first element
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            first_element = memories_result[0]

            # If first element is a container, unwrap one level
            if isinstance(first_element, (list, tuple)):
                actual_memories = first_element
            else:
                # First element is a memory object, structure is already flat
                actual_memories = memories_result
        else:
            actual_memories = memories_result

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    async def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
    ):
        """
        Searches for memories based on a query
        Args:
            query (str): Query to search for.
            user_id (str, optional): ID of the user to search for. Defaults to None.
            agent_id (str, optional): ID of the agent to search for. Defaults to None.
            run_id (str, optional): ID of the run to search for. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to 100.
            filters (dict, optional): Legacy filters to apply to the search. Defaults to None.
            threshold (float, optional): Minimum score for a memory to be included in the results. Defaults to None.
            filters (dict, optional): Enhanced metadata filtering with operators:
                - {"key": "value"} - exact match
                - {"key": {"eq": "value"}} - equals
                - {"key": {"ne": "value"}} - not equals  
                - {"key": {"in": ["val1", "val2"]}} - in list
                - {"key": {"nin": ["val1", "val2"]}} - not in list
                - {"key": {"gt": 10}} - greater than
                - {"key": {"gte": 10}} - greater than or equal
                - {"key": {"lt": 10}} - less than
                - {"key": {"lte": 10}} - less than or equal
                - {"key": {"contains": "text"}} - contains text
                - {"key": {"icontains": "text"}} - case-insensitive contains
                - {"key": "*"} - wildcard match (any value)
                - {"AND": [filter1, filter2]} - logical AND
                - {"OR": [filter1, filter2]} - logical OR
                - {"NOT": [filter1]} - logical NOT

        Returns:
            dict: A dictionary containing the search results, typically under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`
        """

        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("at least one of 'user_id', 'agent_id', or 'run_id' must be specified ")

        # Apply enhanced metadata filtering if advanced operators are detected
        if filters and self._has_advanced_operators(filters):
            processed_filters = self._process_metadata_filters(filters)
            effective_filters.update(processed_filters)
        elif filters:
            # Simple filters, merge directly
            effective_filters.update(filters)


        if not self._is_vector_store_enabled():
            original_memories = await self._search_mysql_store(query, effective_filters, limit, threshold)
            if self.enable_graph:
                if hasattr(self.graph.search, "__await__"):
                    graph_entities = await self.graph.search(query, effective_filters, limit)
                else:
                    graph_entities = await asyncio.to_thread(self.graph.search, query, effective_filters, limit)
            else:
                graph_entities = None
        else:
            vector_store_task = asyncio.create_task(self._search_vector_store(query, effective_filters, limit, threshold))

            graph_task = None
            if self.enable_graph:
                if hasattr(self.graph.search, "__await__"):  # Check if graph search is async
                    graph_task = asyncio.create_task(self.graph.search(query, effective_filters, limit))
                else:
                    graph_task = asyncio.create_task(asyncio.to_thread(self.graph.search, query, effective_filters, limit))

            if graph_task:
                original_memories, graph_entities = await asyncio.gather(vector_store_task, graph_task)
            else:
                original_memories = await vector_store_task
                graph_entities = None

        # Apply reranking if enabled and reranker is available
        if rerank and self.reranker and original_memories:
            try:
                # Run reranking in thread pool to avoid blocking async loop
                reranked_memories = await asyncio.to_thread(
                    self.reranker.rerank, query, original_memories, limit
                )
                original_memories = reranked_memories
            except Exception as e:
                logger.warning(f"Reranking failed, using original results: {e}")

        if self.enable_graph:
            return {"results": original_memories, "relations": graph_entities}

        return {"results": original_memories}

    def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced metadata filters and convert them to vector store compatible format.

        Args:
            metadata_filters: Enhanced metadata filters with operators

        Returns:
            Dict of processed filters compatible with vector store
        """
        processed_filters = {}

        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                return {key: condition}

            result = {}
            for operator, value in condition.items():
                # Map platform operators to universal format that can be translated by each vector store
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte",
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains"
                }

                if operator in operator_map:
                    result[key] = {operator_map[operator]: value}
                else:
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            return result

        for key, value in metadata_filters.items():
            if key == "AND":
                # Logical AND: combine multiple conditions
                if not isinstance(value, list):
                    raise ValueError("AND operator requires a list of conditions")
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        processed_filters.update(process_condition(sub_key, sub_value))
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                processed_filters["$or"] = []
                for condition in value:
                    or_condition = {}
                    for sub_key, sub_value in condition.items():
                        or_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$or"].append(or_condition)
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                processed_filters["$not"] = []
                for condition in value:
                    not_condition = {}
                    for sub_key, sub_value in condition.items():
                        not_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$not"].append(not_condition)
            else:
                processed_filters.update(process_condition(key, value))

        return processed_filters

    def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
        """
        Check if filters contain advanced operators that need special processing.

        Args:
            filters: Dictionary of filters to check

        Returns:
            bool: True if advanced operators are detected
        """
        if not isinstance(filters, dict):
            return False

        for key, value in filters.items():
            # Check for platform-style logical operators
            if key in ["AND", "OR", "NOT"]:
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            if isinstance(value, dict):
                for op in value.keys():
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        return True
            # Check for wildcard values
            if value == "*":
                return True
        return False

    async def _search_vector_store(self, query, filters, limit, threshold: Optional[float] = None):
        embeddings = await asyncio.to_thread(self.embedding_model.embed, query, "search")
        memories = await asyncio.to_thread(
            self.vector_store.search, query=query, vectors=embeddings, limit=limit, filters=filters
        )

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        original_memories = []
        for mem in memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
                score=mem.score,
            ).model_dump()

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            if threshold is None or mem.score >= threshold:
                original_memories.append(memory_item_dict)

        return original_memories

    async def update(self, memory_id, data):
        """
        Update a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to update.
            data (str): New content to update the memory with.

        Returns:
            dict: Success message indicating the memory was updated.

        Example:
            >>> await m.update(memory_id="mem_123", data="Likes to play tennis on weekends")
            {'message': 'Memory updated successfully!'}
        """

        embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")
        existing_embeddings = {data: embeddings}

        await self._update_memory(memory_id, data, existing_embeddings)
        return {"message": "Memory updated successfully!"}

    async def delete(self, memory_id):
        """
        Delete a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        await self._delete_memory(memory_id)
        return {"message": "Memory deleted successfully!"}

    async def delete_all(self, user_id=None, agent_id=None, run_id=None):
        """
        Delete all memories asynchronously.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        memories = await asyncio.to_thread(self.vector_store.list, filters=filters)

        delete_tasks = []
        for memory in memories[0]:
            delete_tasks.append(self._delete_memory(memory.id))

        await asyncio.gather(*delete_tasks)

        logger.info(f"Deleted {len(memories[0])} memories")

        if self.enable_graph:
            await asyncio.to_thread(self.graph.delete_all, filters)

        return {"message": "Memories deleted successfully!"}

    async def history(self, memory_id):
        """
        Get the history of changes for a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        return await asyncio.to_thread(self.db.get_history, memory_id)

    async def _create_memory(self, data, existing_embeddings, metadata=None):
        logger.debug(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, memory_action="add")

        memory_id = uuid.uuid4().hex
        metadata = metadata or {}
        metadata["data"] = data
        metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        await asyncio.to_thread(
            self.vector_store.insert,
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[metadata],
        )

        return memory_id

    async def _create_procedural_memory(self, messages, metadata=None, llm=None, prompt=None):
        """
        Create a procedural memory asynchronously

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            llm (llm, optional): LLM to use for the procedural memory creation. Defaults to None.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        try:
            from langchain_core.messages.utils import (
                convert_to_messages,  # type: ignore
            )
        except Exception:
            logger.error(
                "Import error while loading langchain-core. Please install 'langchain-core' to use procedural memory."
            )
            raise

        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {"role": "user", "content": "Create procedural memory of the above conversation."},
        ]

        try:
            if llm is not None:
                parsed_messages = convert_to_messages(parsed_messages)
                response = await asyncio.to_thread(llm.invoke, input=parsed_messages)
                procedural_memory = response.content
            else:
                procedural_memory = await asyncio.to_thread(self.llm.generate_response, messages=parsed_messages)
                procedural_memory = remove_code_blocks(procedural_memory)
        
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata["memory_type"] = MemoryType.PROCEDURAL.value
        embeddings = await asyncio.to_thread(self.embedding_model.embed, procedural_memory, memory_action="add")
        memory_id = await self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    async def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        new_metadata = deepcopy(metadata) if metadata is not None else {}

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        # Preserve session identifiers from existing memory only if not provided in new metadata
        if "user_id" not in new_metadata and "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" not in new_metadata and "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" not in new_metadata and "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]

        if "actor_id" not in new_metadata and "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        if "role" not in new_metadata and "role" in existing_memory.payload:
            new_metadata["role"] = existing_memory.payload["role"]
        if "memory_type" not in new_metadata and "memory_type" in existing_memory.payload:
            new_metadata["memory_type"] = existing_memory.payload["memory_type"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")

        await asyncio.to_thread(
            self.vector_store.update,
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        return memory_id

    async def _delete_memory(self, memory_id):
        logger.info(f"Deleting memory with {memory_id=}")
        await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        await asyncio.to_thread(self.vector_store.delete, vector_id=memory_id)

        return memory_id

    async def reset(self):
        """
        Reset the memory store asynchronously by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")

        if self._is_vector_store_enabled():
            try:
                await asyncio.to_thread(self.vector_store.delete_col)
                gc.collect()

                if hasattr(self.vector_store, "client") and hasattr(self.vector_store.client, "close"):
                    await asyncio.to_thread(self.vector_store.client.close)

                self.vector_store = VectorStoreFactory.create(
                    self.config.vector_store.provider, self.config.vector_store.config
                )
            except Exception as e:
                logger.warning(f"Vector reset failed. MySQL reset completed; vector state unchanged. Error: {e}")
        else:
            logger.warning("Vector store is unavailable. Skipping vector reset.")

        if self.db:
            await asyncio.to_thread(self.db.reset)
            await asyncio.to_thread(self.db.close)

        self.db = _create_mysql_manager(self.config)

    async def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")
