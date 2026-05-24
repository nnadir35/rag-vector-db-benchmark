import os
from dataclasses import dataclass
from redis import Redis

@dataclass(frozen=True)
class QueueConfig:
    """Configuration for the Redis Queue (RQ) ingestion pipeline."""
    
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    redis_password: str | None = os.getenv("REDIS_PASSWORD", None)
    
    # Queue names
    queue_name: str = "ingestion_queue"
    dlq_name: str = "ingestion_dlq"
    
    # Retry configurations
    max_retries: int = 3
    retry_delay: int = 5  # seconds

    def get_redis_connection(self) -> Redis:
        """Create and return a Redis connection based on the configuration."""
        return Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=self.redis_password,
            decode_responses=False  # RQ requires decode_responses=False (default)
        )
