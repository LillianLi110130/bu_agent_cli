from pydantic import BaseModel, Field

_DEFAULT_MYSQL_DB_URI = "mysql+pymysql://root:@localhost:3306/mem0?charset=utf8mb4"


class MysqlConfig(BaseModel):
    db_uri: str = Field(default=_DEFAULT_MYSQL_DB_URI, description="MySQL SQLAlchemy database URI")

    # --- SQLAlchemy connection pool tuning ---
    pool_size: int = Field(default=5, description="SQLAlchemy pool size")
    max_overflow: int = Field(default=10, description="SQLAlchemy max overflow")
    pool_timeout: int = Field(default=30, description="SQLAlchemy pool timeout (seconds)")
    pool_recycle: int = Field(default=3600, description="SQLAlchemy pool recycle (seconds)")
    pool_pre_ping: bool = Field(default=True, description="Enable SQLAlchemy pool pre-ping")
