import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

try:
    # SQLAlchemy uses PyMySQL as the DBAPI driver when `mysql+pymysql://` is used.
    import pymysql  # noqa: F401
except ImportError:
    raise ImportError("MySQL history store requires 'pymysql'. Install with: pip install pymysql") from None

try:
    from sqlalchemy import CheckConstraint, Column, Integer, SmallInteger, String, Text, create_engine, insert, select, text
    from sqlalchemy.engine import Engine, make_url
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import Session, declarative_base, sessionmaker

    from sqlalchemy.dialects.mysql import LONGTEXT, TIMESTAMP, TINYINT
    from sqlalchemy.schema import Index
except ImportError:
    raise ImportError(
        "MySQL history store requires 'sqlalchemy' (and a MySQL driver such as 'pymysql'). "
        "Install with: pip install sqlalchemy pymysql"
    ) from None

from tg_mem.configs.enums import MemoryType

logger = logging.getLogger(__name__)

Base = declarative_base()


class MemoryChangeHistoryModel(Base):
    __tablename__ = "memory_change_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    memory_id = Column(Integer, nullable=True)
    old_memory = Column(Text, nullable=True)
    new_memory = Column(Text, nullable=True)
    event = Column(String(32), nullable=True)
    created_at = Column(
        String(32).with_variant(TIMESTAMP(), "mysql"),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at = Column(
        String(32).with_variant(TIMESTAMP(), "mysql"),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    is_deleted = Column(
        SmallInteger().with_variant(TINYINT(1), "mysql"),
        nullable=True,
        server_default=text("0"),
    )
    actor_id = Column(String(32), nullable=True)
    role = Column(String(32), nullable=True)

    __table_args__ = (
        Index("idx_memory_id", "memory_id"),
        {
            "mysql_engine": "InnoDB",
            "mysql_charset": "utf8mb4",
        },
    )


class SessionModel(Base):
    __tablename__ = "session"

    session_id = Column(String(191), primary_key=True)
    user_id = Column(String(32), nullable=False)
    channel = Column(String(64), nullable=False, server_default=text("''"))
    query = Column(Text().with_variant(LONGTEXT(), "mysql"), nullable=False)
    create_time = Column(
        String(32).with_variant(TIMESTAMP(), "mysql"),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    update_time = Column(
        String(32).with_variant(TIMESTAMP(), "mysql"),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        Index("idx_session_user_id", "user_id"),
        Index("idx_session_update_time", "update_time"),
        {
            "mysql_engine": "InnoDB",
            "mysql_charset": "utf8mb4",
            "mysql_collate": "utf8mb4_unicode_ci",
        },
    )


class RecordModel(Base):
    __tablename__ = "record"

    record_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(191), nullable=False)
    role = Column(String(32), nullable=False)
    content = Column(Text().with_variant(LONGTEXT(), "mysql"), nullable=False)
    create_time = Column(
        String(32).with_variant(TIMESTAMP(), "mysql"),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    update_time = Column(
        String(32).with_variant(TIMESTAMP(), "mysql"),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        Index("idx_record_session_id", "session_id"),
        Index("idx_record_session_time", "session_id", "create_time", "record_id"),
        {
            "mysql_engine": "InnoDB",
            "mysql_charset": "utf8mb4",
            "mysql_collate": "utf8mb4_unicode_ci",
        },
    )


class MemoryModel(Base):
    __tablename__ = "memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(32), nullable=False)
    memory_data = Column(Text().with_variant(LONGTEXT(), "mysql"), nullable=False)
    memory_type = Column(String(32), nullable=False)
    status = Column(
        String(4),
        nullable=False,
        server_default=text("'ACTE'"),
    )
    created_at = Column(
        String(32).with_variant(TIMESTAMP(), "mysql"),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at = Column(
        String(32).with_variant(TIMESTAMP(), "mysql"),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        Index("idx_user", "user_id"),
        Index("idx_status", "status"),
        CheckConstraint("status IN ('ACTE', 'INAC')", name="ck_memory_status"),
        {
            "mysql_engine": "InnoDB",
            "mysql_charset": "utf8mb4",
            "mysql_collate": "utf8mb4_unicode_ci",
        },
    )


class MySQLManager:
    def __init__(
        self,
        db_uri: Optional[str] = None,
        *,
        # --- SQLAlchemy pool tuning ---
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True,
        # --- advanced/testing ---
        engine: Optional[Engine] = None,
    ):
        self.db_uri = db_uri

        self.pool_size = int(pool_size)
        self.max_overflow = int(max_overflow)
        self.pool_timeout = int(pool_timeout)
        self.pool_recycle = int(pool_recycle)
        self.pool_pre_ping = bool(pool_pre_ping)

        self.engine: Optional[Engine] = engine
        self._session_factory: Optional[sessionmaker[Session]] = None

        try:
            # Validate and resolve config early.
            cfg = self._resolve_mysql_config()

            if self.engine is None:
                self._create_database_if_needed(cfg)
                self.engine = self._create_engine(cfg["db_uri"], include_database=True)

            self._session_factory = sessionmaker(
                bind=self.engine,
                autoflush=False,
                expire_on_commit=False,
            )
            self._create_required_tables()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _escape_hash_in_userinfo(db_uri: str) -> str:
        scheme_sep = "://"
        scheme_idx = db_uri.find(scheme_sep)
        if scheme_idx == -1:
            return db_uri

        authority_start = scheme_idx + len(scheme_sep)
        slash_index = db_uri.find("/", authority_start)
        query_index = db_uri.find("?", authority_start)

        authority_end_candidates = [index for index in (slash_index, query_index) if index != -1]
        authority_end = min(authority_end_candidates) if authority_end_candidates else len(db_uri)
        authority = db_uri[authority_start:authority_end]

        userinfo, at, hostinfo = authority.rpartition("@")
        if not at or "#" not in userinfo:
            return db_uri

        escaped_authority = f"{userinfo.replace('#', '%23')}@{hostinfo}"
        return f"{db_uri[:authority_start]}{escaped_authority}{db_uri[authority_end:]}"

    def _parse_mysql_uri(self) -> Dict[str, Any]:
        if not self.db_uri:
            raise ValueError("MySQL history db uri is empty")

        normalized_db_uri = self._escape_hash_in_userinfo(self.db_uri)
        parsed = urlparse(normalized_db_uri)
        if parsed.scheme not in {"mysql", "mysql+pymysql"}:
            raise ValueError(
                "Invalid history db uri. Expected mysql URI, e.g. mysql://user:password@host:3306/dbname"
            )

        host = parsed.hostname or "localhost"
        try:
            port = parsed.port or 3306
        except ValueError as exc:
            raise ValueError("MySQL history db uri has an invalid port") from exc

        user = unquote(parsed.username) if parsed.username else "root"
        password = unquote(parsed.password) if parsed.password else ""
        database = (parsed.path or "/").lstrip("/")
        if not database:
            raise ValueError("MySQL history db uri must include database name")

        query = parse_qs(parsed.query)
        charset = query.get("charset", ["utf8mb4"])[0]

        return {
            "db_uri": normalized_db_uri,
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "charset": charset,
        }

    def _resolve_mysql_config(self) -> Dict[str, Any]:
        if not self.db_uri:
            raise ValueError("MySQL history db uri is empty")

        return self._parse_mysql_uri()

    def _create_engine(self, db_uri: str, *, include_database: bool) -> Engine:
        normalized_db_uri = self._escape_hash_in_userinfo(db_uri)
        url = make_url(normalized_db_uri)

        if url.drivername == "mysql":
            url = url.set(drivername="mysql+pymysql")
        elif url.drivername != "mysql+pymysql":
            raise ValueError(
                "Invalid history db uri. Expected mysql URI, e.g. mysql://user:password@host:3306/dbname"
            )

        if include_database and not url.database:
            raise ValueError("MySQL history db uri must include database name")

        if "charset" not in url.query:
            url = url.update_query_dict({"charset": "utf8mb4"})

        if not include_database:
            url = url.set(database=None)

        return create_engine(
            url,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle,
            pool_pre_ping=self.pool_pre_ping,
        )

    @staticmethod
    def _quote_mysql_identifier(identifier: str) -> str:
        # MySQL uses backticks; escape embedded backticks by doubling.
        safe = str(identifier).replace("`", "``")
        return f"`{safe}`"

    def _create_database_if_needed(self, cfg: Dict[str, Any]) -> None:
        bootstrap_engine = self._create_engine(cfg["db_uri"], include_database=False)
        db_name = self._quote_mysql_identifier(cfg["database"])

        try:
            with bootstrap_engine.connect() as conn:
                conn = conn.execution_options(isolation_level="AUTOCOMMIT")
                conn.exec_driver_sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        finally:
            bootstrap_engine.dispose()

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _normalize_datetime(value: Optional[Any]) -> str:
        if value is None:
            return MySQLManager._utc_now_iso()

        if isinstance(value, datetime):
            dt = value
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt.strftime("%Y-%m-%d %H:%M:%S")

        text_value = str(value).strip()
        if not text_value:
            return MySQLManager._utc_now_iso()

        if text_value.endswith("Z"):
            text_value = text_value[:-1] + "+00:00"

        try:
            dt = datetime.fromisoformat(text_value)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return text_value

    @staticmethod
    def _normalize_identifier(value: Optional[Any], max_length: int = 32) -> Optional[str]:
        if value is None:
            return None

        text_value = str(value).strip()
        if not text_value:
            return None

        if len(text_value) <= max_length:
            return text_value

        return hashlib.md5(text_value.encode("utf-8")).hexdigest()[:max_length]

    @staticmethod
    def _normalize_session_id(value: Optional[Any]) -> Optional[str]:
        return MySQLManager._normalize_identifier(value, max_length=191)

    @staticmethod
    def _normalize_int_identifier(value: Optional[Any]) -> Optional[int]:
        if value is None:
            return None

        if isinstance(value, bool):
            return int(value)

        if isinstance(value, int):
            return value

        text_value = str(value).strip()
        if not text_value:
            return None

        try:
            return int(text_value)
        except ValueError:
            stable = int(hashlib.md5(text_value.encode("utf-8")).hexdigest()[:8], 16)
            return stable % 2_147_483_647 or 1

    @staticmethod
    def _normalize_step_id(value: Optional[Any]) -> Optional[int]:
        if value is None:
            return None

        if isinstance(value, bool):
            return int(value)

        try:
            step_id = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("step_id must be an integer") from exc

        if step_id < 0:
            raise ValueError("step_id must be >= 0")

        return step_id

    @staticmethod
    def _serialize_content(content: Any) -> Optional[str]:
        if content is None:
            return None
        if isinstance(content, str):
            return content
        try:
            return json.dumps(content, ensure_ascii=False)
        except TypeError:
            return str(content)

    @staticmethod
    def _normalize_status(status: Any, *, default: str = "ACTE") -> str:
        if status is None:
            return default

        if not isinstance(status, str):
            raise ValueError("memory.status only accepts string enum values ACTE and INAC")

        normalized = status.strip().upper()
        if normalized not in {"ACTE", "INAC"}:
            raise ValueError("memory.status only accepts string enum values ACTE and INAC")

        return normalized

    def _require_session_factory(self) -> sessionmaker[Session]:
        if self._session_factory is None or self.engine is None:
            raise RuntimeError("MySQLManager is closed")
        return self._session_factory

    def _create_required_tables(self) -> None:
        try:
            if self.engine is None:
                raise RuntimeError("MySQL engine is not initialized")
            Base.metadata.create_all(self.engine)
        except Exception as e:
            logger.error(f"Failed to create required MySQL tables via SQLAlchemy: {e}")
            raise

    def add_history(
        self,
        memory_id: Optional[Any],
        old_memory: Optional[str],
        new_memory: Optional[str],
        event: str,
        *,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        is_deleted: int = 0,
        actor_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> None:
        created_at = self._normalize_datetime(created_at)
        updated_at = self._normalize_datetime(updated_at or created_at)
        normalized_memory_id = self._normalize_int_identifier(memory_id)

        model = MemoryChangeHistoryModel(
            memory_id=normalized_memory_id,
            old_memory=old_memory,
            new_memory=new_memory,
            event=event,
            created_at=created_at,
            updated_at=updated_at,
            is_deleted=int(is_deleted or 0),
            actor_id=self._normalize_identifier(actor_id, max_length=32),
            role=self._normalize_identifier(role, max_length=32),
        )

        try:
            SessionFactory = self._require_session_factory()
            with SessionFactory() as session:
                with session.begin():
                    session.add(model)
        except Exception as e:
            logger.error(f"Failed to add history record: {e}")
            raise

    def get_history(self, memory_id: Any) -> List[Dict[str, Any]]:
        normalized_memory_id = self._normalize_int_identifier(memory_id)
        if normalized_memory_id is None:
            return []

        SessionFactory = self._require_session_factory()
        with SessionFactory() as session:
            stmt = (
                select(MemoryChangeHistoryModel)
                .where(MemoryChangeHistoryModel.memory_id == normalized_memory_id)
                .order_by(MemoryChangeHistoryModel.created_at.asc(), MemoryChangeHistoryModel.updated_at.asc())
            )
            rows = session.execute(stmt).scalars().all()

        return [
            {
                "id": r.id,
                "memory_id": r.memory_id,
                "old_memory": r.old_memory,
                "new_memory": r.new_memory,
                "event": r.event,
                "created_at": r.created_at,
                "updated_at": r.updated_at,
                "is_deleted": bool(r.is_deleted),
                "actor_id": r.actor_id,
                "role": r.role,
            }
            for r in rows
        ]

    def get_session(self, session_id: Any) -> Optional[Dict[str, Any]]:
        normalized_session_id = self._normalize_session_id(session_id)
        if normalized_session_id is None:
            return None

        SessionFactory = self._require_session_factory()
        with SessionFactory() as session:
            row = session.get(SessionModel, normalized_session_id)

        if row is None:
            return None

        return {
            "session_id": row.session_id,
            "user_id": row.user_id,
            "channel": row.channel,
            "query": row.query,
            "create_time": row.create_time,
            "update_time": row.update_time,
        }

    def add_conversation_records(
        self,
        session_id: Any,
        messages: List[Dict[str, Any]],
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        channel: Optional[str] = "",
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ) -> None:
        normalized_session_id = self._normalize_session_id(session_id)
        if normalized_session_id is None:
            raise ValueError("session_id is required")
        if not messages:
            return

        resolved_user_id = self._normalize_identifier(user_id)
        if resolved_user_id is None:
            raise ValueError("user_id is required")

        resolved_channel = self._normalize_identifier(channel, max_length=64) or ""

        create_time = self._normalize_datetime(created_at)
        update_time = self._normalize_datetime(updated_at or created_at)

        rows: List[Dict[str, Any]] = []
        first_user_message_content: Optional[str] = None
        for message in messages:
            payload = message if isinstance(message, dict) else {"content": message}
            role = self._normalize_identifier(payload.get("role"), max_length=32)
            if role is None:
                raise ValueError("message role is required")

            content = self._serialize_content(payload.get("content")) or ""
            if first_user_message_content is None and role.lower() == "user":
                first_user_message_content = content

            rows.append(
                {
                    "session_id": normalized_session_id,
                    "role": role,
                    "content": content,
                    "create_time": create_time,
                    "update_time": update_time,
                }
            )

        try:
            SessionFactory = self._require_session_factory()
            with SessionFactory() as session:
                with session.begin():
                    session_row = session.get(SessionModel, normalized_session_id)
                    if session_row is None:
                        if first_user_message_content is None:
                            raise ValueError("first user message is required for a new session")

                        session.add(
                            SessionModel(
                                session_id=normalized_session_id,
                                user_id=resolved_user_id,
                                channel=resolved_channel,
                                query=first_user_message_content,
                                create_time=create_time,
                                update_time=update_time,
                            )
                        )
                    else:
                        if session_row.user_id != resolved_user_id:
                            raise ValueError("session user_id mismatch")
                        session_row.update_time = update_time

                    session.execute(insert(RecordModel), rows)
        except Exception as e:
            logger.error(f"Failed to add conversation records: {e}")
            raise

    def get_conversation_records(
        self,
        *,
        session_id: Optional[Any] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        SessionFactory = self._require_session_factory()
        with SessionFactory() as session:
            stmt = select(RecordModel)

            if session_id is not None:
                normalized_session_id = self._normalize_session_id(session_id)
                if normalized_session_id is not None:
                    stmt = stmt.where(RecordModel.session_id == normalized_session_id)

            stmt = stmt.order_by(RecordModel.create_time.asc(), RecordModel.record_id.asc())

            if limit is not None:
                stmt = stmt.limit(int(limit))

            rows = session.execute(stmt).scalars().all()

        return [
            {
                "record_id": r.record_id,
                "session_id": r.session_id,
                "role": r.role,
                "content": r.content,
                "create_time": r.create_time,
                "update_time": r.update_time,
            }
            for r in rows
        ]

    def create_memory_record(
        self,
        *,
        memory_data: Optional[Any],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        status: Any = "ACTE",
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ) -> int:
        now = self._utc_now_iso()
        created_at = self._normalize_datetime(created_at or now)
        updated_at = self._normalize_datetime(updated_at or now)

        resolved_user_id = self._normalize_identifier(user_id or agent_id or run_id or "unknown") or "unknown"
        resolved_memory_type = memory_type or MemoryType.SEMANTIC.value
        resolved_status = self._normalize_status(status, default="ACTE")

        record = MemoryModel(
            user_id=resolved_user_id,
            memory_data=self._serialize_content(memory_data) or "",
            memory_type=resolved_memory_type,
            status=resolved_status,
            created_at=created_at,
            updated_at=updated_at,
        )

        try:
            SessionFactory = self._require_session_factory()
            with SessionFactory() as session:
                with session.begin():
                    session.add(record)
                    session.flush()
                    inserted_id = int(record.id)
            return inserted_id
        except Exception as e:
            logger.error(f"Failed to create memory record: {e}")
            raise

    def update_memory_record(
        self,
        memory_id: Any,
        *,
        memory_data: Optional[Any] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        status: Optional[Any] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ) -> None:
        normalized_memory_id = self._normalize_int_identifier(memory_id)
        if normalized_memory_id is None:
            raise ValueError("memory_id is required")

        try:
            SessionFactory = self._require_session_factory()
            with SessionFactory() as session:
                with session.begin():
                    existing = session.get(MemoryModel, normalized_memory_id)
                    if existing is None:
                        raise ValueError(f"Memory record with id={normalized_memory_id} not found")

                    resolved_content = (
                        self._serialize_content(memory_data) if memory_data is not None else existing.memory_data
                    ) or ""

                    resolved_user_id = self._normalize_identifier(user_id or agent_id or run_id) or existing.user_id
                    resolved_memory_type = memory_type or existing.memory_type
                    resolved_status = (
                        self._normalize_status(status, default=existing.status) if status is not None else existing.status
                    )
                    resolved_created_at = self._normalize_datetime(created_at or existing.created_at)
                    resolved_updated_at = self._normalize_datetime(updated_at or self._utc_now_iso())

                    existing.user_id = resolved_user_id
                    existing.memory_data = resolved_content
                    existing.memory_type = resolved_memory_type
                    existing.status = resolved_status
                    existing.created_at = resolved_created_at
                    existing.updated_at = resolved_updated_at
        except Exception as e:
            logger.error(f"Failed to update memory record: {e}")
            raise

    def upsert_memory_record(
        self,
        *,
        memory_id: Optional[Any] = None,
        memory_data: Optional[Any],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        status: Any = "ACTE",
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ) -> int:
        normalized_memory_id = self._normalize_int_identifier(memory_id)

        SessionFactory = self._require_session_factory()
        with SessionFactory() as session:
            with session.begin():
                if normalized_memory_id is not None:
                    existing = session.get(MemoryModel, normalized_memory_id)
                    if existing is not None:
                        resolved_content = (self._serialize_content(memory_data) or existing.memory_data) or ""
                        resolved_user_id = self._normalize_identifier(user_id or agent_id or run_id) or existing.user_id
                        resolved_memory_type = memory_type or existing.memory_type
                        resolved_status = self._normalize_status(status, default=existing.status)
                        resolved_created_at = self._normalize_datetime(created_at or existing.created_at)
                        resolved_updated_at = self._normalize_datetime(updated_at or self._utc_now_iso())

                        existing.user_id = resolved_user_id
                        existing.memory_data = resolved_content
                        existing.memory_type = resolved_memory_type
                        existing.status = resolved_status
                        existing.created_at = resolved_created_at
                        existing.updated_at = resolved_updated_at
                        return normalized_memory_id

                # Not found (or no id provided) -> create
                now = self._utc_now_iso()
                resolved_created_at = self._normalize_datetime(created_at or now)
                resolved_updated_at = self._normalize_datetime(updated_at or now)

                resolved_user_id = self._normalize_identifier(user_id or agent_id or run_id or "unknown") or "unknown"
                resolved_memory_type = memory_type or MemoryType.SEMANTIC.value
                resolved_status = self._normalize_status(status, default="ACTE")

                record = MemoryModel(
                    user_id=resolved_user_id,
                    memory_data=self._serialize_content(memory_data) or "",
                    memory_type=resolved_memory_type,
                    status=resolved_status,
                    created_at=resolved_created_at,
                    updated_at=resolved_updated_at,
                )
                session.add(record)
                session.flush()
                return int(record.id)

    def get_memory_record(self, memory_id: Any) -> Optional[Dict[str, Any]]:
        normalized_memory_id = self._normalize_int_identifier(memory_id)
        if normalized_memory_id is None:
            return None

        SessionFactory = self._require_session_factory()
        with SessionFactory() as session:
            row = session.get(MemoryModel, normalized_memory_id)

        if row is None:
            return None

        return {
            "id": row.id,
            "memory_id": row.id,
            "user_id": row.user_id,
            "memory_data": row.memory_data,
            "memory_type": row.memory_type,
            "status": row.status,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }

    def list_memory_records(
        self,
        *,
        user_id: Optional[str] = None,
        status: Optional[Any] = "ACTE",
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        resolved_user_id = self._normalize_identifier(user_id)
        if not resolved_user_id:
            raise ValueError("user_id is required")

        resolved_status = self._normalize_status(status, default="ACTE") if status is not None else None

        SessionFactory = self._require_session_factory()
        with SessionFactory() as session:
            stmt = select(MemoryModel).where(MemoryModel.user_id == resolved_user_id)
            if resolved_status is not None:
                stmt = stmt.where(MemoryModel.status == resolved_status)

            stmt = stmt.order_by(MemoryModel.updated_at.desc())
            if limit is not None:
                stmt = stmt.limit(int(limit))

            rows = session.execute(stmt).scalars().all()

        return [
            {
                "id": r.id,
                "memory_id": r.id,
                "user_id": r.user_id,
                "memory_data": r.memory_data,
                "memory_type": r.memory_type,
                "status": r.status,
                "created_at": r.created_at,
                "updated_at": r.updated_at,
            }
            for r in rows
        ]

    def reset(self) -> None:
        if self.engine is None:
            raise RuntimeError("MySQL engine is not initialized")

        # Drop both current and legacy tables for backward compatibility.
        tables_to_drop = [
            "memory_change_history",
            "history",
            "session",
            "record",
            "memory",
            "session_step_counter",
            "conversation_records",
            "memory_records",
        ]

        try:
            with self.engine.begin() as conn:
                for table_name in tables_to_drop:
                    conn.exec_driver_sql(f"DROP TABLE IF EXISTS {self._quote_mysql_identifier(table_name)}")

            self._create_required_tables()
        except Exception as e:
            logger.error(f"Failed to reset MySQL tables: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self) -> None:
        engine = getattr(self, "engine", None)
        if engine is None:
            return

        try:
            engine.dispose()
        finally:
            self.engine = None
            self._session_factory = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
