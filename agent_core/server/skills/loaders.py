"""
Skill loaders for different sources.

Loaders are responsible for loading skills from various sources:
- ConfigSkillLoader: From in-memory configuration (built-in skills)
- DatabaseSkillLoader: From a database (runtime management)
- RemoteAPISkillLoader: From a remote API service (centralized management)
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
from datetime import datetime, timedelta

if TYPE_CHECKING:
    from .types import Skill

logger = logging.getLogger("agent_core.server.skills.loaders")


class SkillLoader(ABC):
    """Abstract base class for skill loaders.

    Custom loaders should inherit from this class and implement
    the load() and get() methods.
    """

    @abstractmethod
    async def load(self) -> list["Skill"]:
        """Load all skills from this source.

        Returns:
            List of all Skill instances from this source
        """
        pass

    @abstractmethod
    async def get(self, name: str) -> "Skill | None":
        """Get a specific skill by name.

        Args:
            name: The skill name to retrieve

        Returns:
            The Skill if found, None otherwise
        """
        pass


class ConfigSkillLoader(SkillLoader):
    """Load skills from in-memory configuration.

    This is the default loader for built-in skills. Skills are
    provided as a list of dictionaries at initialization.

    Example:
        config = [
            {
                "name": "calculator",
                "display_name": "Calculator",
                "description": "Perform arithmetic calculations",
                "content": "# Calculator\\n...",
                "category": "Math",
            }
        ]
        loader = ConfigSkillLoader(config)
    """

    def __init__(self, skills_config: list[dict[str, Any]]):
        """Initialize the loader with skills configuration.

        Args:
            skills_config: List of skill configuration dictionaries
        """
        self._skills = self._parse_config(skills_config)

    def _parse_config(self, config: list[dict[str, Any]]) -> dict[str, "Skill"]:
        """Parse skill configurations into Skill objects.

        Args:
            config: List of skill configuration dictionaries

        Returns:
            Dictionary mapping skill names to Skill objects
        """
        from .types import Skill

        skills = {}
        for item in config:
            try:
                skill = Skill(
                    name=item["name"],
                    display_name=item.get("display_name", item["name"]),
                    description=item.get("description", ""),
                    content=item["content"],
                    category=item.get("category", "General"),
                    source="config",
                    enabled=item.get("enabled", True),
                    version=item.get("version", "1.0"),
                    tags=item.get("tags"),
                )
                skills[skill.name] = skill
            except KeyError as e:
                logger.warning(f"Skipping invalid skill config: missing {e}")
            except Exception as e:
                logger.warning(f"Skipping invalid skill config: {e}")

        return skills

    async def load(self) -> list["Skill"]:
        """Load all skills from configuration.

        Returns:
            List of all Skill instances
        """
        return list(self._skills.values())

    async def get(self, name: str) -> "Skill | None":
        """Get a specific skill by name.

        Args:
            name: The skill name to retrieve

        Returns:
            The Skill if found, None otherwise
        """
        return self._skills.get(name)


class DatabaseSkillLoader(SkillLoader):
    """Load skills from a database.

    This loader supports any database with an async interface.
    Skills should be stored in a table with the following schema:
    - name: VARCHAR (primary key)
    - display_name: VARCHAR
    - description: TEXT
    - content: TEXT
    - category: VARCHAR
    - enabled: BOOLEAN
    - version: VARCHAR
    - tags: JSON/ARRAY

    Example:
        loader = DatabaseSkillLoader(
            db_client=my_async_db_client,
            table_name="skills",
        )
    """

    def __init__(
        self,
        db_client: Any,
        table_name: str = "skills",
        cache_ttl_seconds: int = 300,
    ):
        """Initialize the database loader.

        Args:
            db_client: An async database client with query() method
            table_name: Name of the skills table
            cache_ttl_seconds: Cache TTL in seconds (default: 5 minutes)
        """
        self._db = db_client
        self._table = table_name
        self._cache: dict[str, tuple["Skill", datetime]] = {}
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)

    def _is_cache_valid(self, cached_at: datetime) -> bool:
        """Check if a cached entry is still valid."""
        return datetime.now() - cached_at < self._cache_ttl

    async def load(self) -> list["Skill"]:
        """Load all skills from the database.

        Returns:
            List of all Skill instances
        """
        from .types import Skill

        query = f"""
            SELECT name, display_name, description, content,
                   category, enabled, version, tags
            FROM {self._table}
            WHERE enabled = TRUE
        """

        try:
            results = await self._db.query(query)
            skills = []

            for row in results:
                skill = Skill(
                    name=row["name"],
                    display_name=row["display_name"],
                    description=row["description"],
                    content=row["content"],
                    category=row.get("category", "General"),
                    source="database",
                    enabled=row["enabled"],
                    version=row.get("version", "1.0"),
                    tags=row.get("tags"),
                )
                skills.append(skill)
                self._cache[skill.name] = (skill, datetime.now())

            return skills

        except Exception as e:
            logger.error(f"Error loading skills from database: {e}")
            return []

    async def get(self, name: str) -> "Skill | None":
        """Get a specific skill by name.

        Args:
            name: The skill name to retrieve

        Returns:
            The Skill if found, None otherwise
        """
        from .types import Skill

        # Check cache first
        if name in self._cache:
            skill, cached_at = self._cache[name]
            if self._is_cache_valid(cached_at):
                return skill

        # Query database
        query = f"""
            SELECT name, display_name, description, content,
                   category, enabled, version, tags
            FROM {self._table}
            WHERE name = $1 AND enabled = TRUE
        """

        try:
            result = await self._db.query(query, [name])
            if not result:
                return None

            row = result[0]
            skill = Skill(
                name=row["name"],
                display_name=row["display_name"],
                description=row["description"],
                content=row["content"],
                category=row.get("category", "General"),
                source="database",
                enabled=row["enabled"],
                version=row.get("version", "1.0"),
                tags=row.get("tags"),
            )

            self._cache[name] = (skill, datetime.now())
            return skill

        except Exception as e:
            logger.error(f"Error getting skill '{name}' from database: {e}")
            return None


class RemoteAPISkillLoader(SkillLoader):
    """Load skills from a remote API service.

    This loader fetches skills from a centralized HTTP API.
    Supports caching to reduce API calls.

    Example:
        loader = RemoteAPISkillLoader(
            api_url="https://api.example.com/skills",
            auth_token="your-token",
        )
    """

    def __init__(
        self,
        api_url: str,
        auth_token: str | None = None,
        cache_ttl_seconds: int = 300,
        timeout_seconds: int = 10,
    ):
        """Initialize the API loader.

        Args:
            api_url: Base URL of the skills API
            auth_token: Optional bearer token for authentication
            cache_ttl_seconds: Cache TTL in seconds (default: 5 minutes)
            timeout_seconds: HTTP request timeout in seconds
        """
        self._base_url = api_url.rstrip("/")
        self._token = auth_token
        self._cache: dict[str, tuple["Skill", datetime]] = {}
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._timeout = timeout_seconds
        self._session = None

    async def _get_session(self):
        """Get or create an HTTP session."""
        if self._session is None:
            try:
                import aiohttp

                self._session = aiohttp.ClientSession(
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                )
            except ImportError:
                logger.warning("aiohttp not available, using urllib")
        return self._session

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for API requests."""
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _is_cache_valid(self, cached_at: datetime) -> bool:
        """Check if a cached entry is still valid."""
        return datetime.now() - cached_at < self._cache_ttl

    async def load(self) -> list["Skill"]:
        """Load all skills from the remote API.

        Returns:
            List of all Skill instances
        """
        from .types import Skill

        session = await self._get_session()

        if hasattr(session, "get"):
            # Using aiohttp
            try:
                async with session.get(f"{self._base_url}/skills") as response:
                    if response.status != 200:
                        logger.error(f"API returned status {response.status}")
                        return []

                    data = await response.json()
            except Exception as e:
                logger.error(f"Error loading skills from API: {e}")
                return []
        else:
            # Fallback to urllib
            try:
                import urllib.request
                import json

                request = urllib.request.Request(
                    f"{self._base_url}/skills",
                    headers=self._get_headers(),
                )
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    if response.status != 200:
                        logger.error(f"API returned status {response.status}")
                        return []
                    data = json.loads(response.read().decode())
            except Exception as e:
                logger.error(f"Error loading skills from API: {e}")
                return []

        skills = []
        for item in data.get("skills", []):
            skill = Skill.from_dict(item)
            skills.append(skill)
            self._cache[skill.name] = (skill, datetime.now())

        return skills

    async def get(self, name: str) -> "Skill | None":
        """Get a specific skill by name from the API.

        Args:
            name: The skill name to retrieve

        Returns:
            The Skill if found, None otherwise
        """
        from .types import Skill

        # Check cache first
        if name in self._cache:
            skill, cached_at = self._cache[name]
            if self._is_cache_valid(cached_at):
                return skill

        session = await self._get_session()

        if hasattr(session, "get"):
            # Using aiohttp
            try:
                async with session.get(f"{self._base_url}/skills/{name}") as response:
                    if response.status == 404:
                        return None
                    if response.status != 200:
                        logger.error(f"API returned status {response.status}")
                        return None

                    data = await response.json()
                    skill = Skill.from_dict(data)
                    self._cache[name] = (skill, datetime.now())
                    return skill
            except Exception as e:
                logger.error(f"Error getting skill '{name}' from API: {e}")
                return None
        else:
            # Fallback to urllib
            try:
                import urllib.request
                import json

                request = urllib.request.Request(
                    f"{self._base_url}/skills/{name}",
                    headers=self._get_headers(),
                )
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    if response.status == 404:
                        return None
                    if response.status != 200:
                        logger.error(f"API returned status {response.status}")
                        return None
                    data = json.loads(response.read().decode())
                    skill = Skill.from_dict(data)
                    self._cache[name] = (skill, datetime.now())
                    return skill
            except Exception as e:
                logger.error(f"Error getting skill '{name}' from API: {e}")
                return None

    async def close(self):
        """Close the HTTP session if using aiohttp."""
        if self._session and hasattr(self._session, "close"):
            await self._session.close()
