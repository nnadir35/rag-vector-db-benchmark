import json
import logging
import sys
from datetime import datetime, timezone

from rq import Queue, Worker

from .config import QueueConfig

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("rq.worker")

def dlq_exception_handler(job, exc_type, exc_value, traceback) -> bool:
    """Exception handler to route jobs that exhausted all retries to a Dead Letter Queue (DLQ).

    Args:
        job: The failed RQ Job object.
        exc_type: The exception class.
        exc_value: The exception instance.
        traceback: The traceback object.

    Returns:
        True to continue propagation to other handlers (like RQ's FailedJobRegistry).
    """
    retries_left = getattr(job, "retries_left", None)

    # Eğer henüz deneme hakkı varsa (retries_left > 0), RQ'nun kendi retry handler'ına bırakıyoruz
    if retries_left is not None and retries_left > 0:
        logger.warning(
            f"Job {job.id} failed. Retries left: {retries_left}. "
            f"Error: {exc_type.__name__}: {exc_value}"
        )
        return True

    # Deneme hakları bittiğinde veya retry konfigüre edilmediğinde DLQ'ya aktarıyoruz
    config = QueueConfig()
    connection = job.connection

    error_payload = {
        "job_id": job.id,
        "origin_queue": job.origin,
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "exception_type": exc_type.__name__,
        "exception_message": str(exc_value),
        "job_args": str(job.args) if job.args else [],
        "job_kwargs": str(job.kwargs) if job.kwargs else {}
    }

    try:
        # DLQ olarak belirlenen Redis listesine (list) hata detaylarını yazıyoruz
        connection.rpush(config.dlq_name, json.dumps(error_payload))
        logger.error(
            f"❌ Job {job.id} EXHAUSTED all retries. "
            f"Routed payload to Dead Letter Queue (DLQ): '{config.dlq_name}'"
        )
    except Exception as dlq_err:
        logger.critical(f"Failed to write job {job.id} to DLQ: {dlq_err}", exc_info=True)

    return True

def run_worker() -> None:
    """Start the RQ worker with custom DLQ exception handler and configuration."""
    config = QueueConfig()
    redis_conn = config.get_redis_connection()

    logger.info(f"Starting RQ Worker connecting to Redis at {config.redis_host}:{config.redis_port}")
    logger.info(f"Listening on queue: '{config.queue_name}'")
    logger.info(f"Dead Letter Queue (DLQ) configured as: '{config.dlq_name}' (Redis List)")

    queue = Queue(config.queue_name, connection=redis_conn)

    # Worker initialization with custom DLQ handler
    worker = Worker(
        [queue],
        connection=redis_conn,
        exception_handlers=[dlq_exception_handler]
    )

    try:
        # RQ sinyalleri (SIGINT/SIGTERM) yakalayarak Graceful Shutdown'ı otomatik olarak yönetir.
        # İlk sinyalde yeni iş kabul etmeyi durdurup elindeki işi bitirir.
        # İkinci sinyalde (Cold Shutdown) anında sonlanır.
        worker.work(with_scheduler=True)
    except KeyboardInterrupt:
        logger.info("Worker received KeyboardInterrupt. Exiting gracefully...")
    except Exception as e:
        logger.critical(f"Worker crashed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_worker()
