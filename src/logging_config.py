"""
Logging configuration
"""

import logging
import sys
from datetime import datetime, timezone

from pythonjsonlogger.json import JsonFormatter


STANDARD_LOG_RECORD_FIELDS = set(
    logging.LogRecord(
        name="",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    ).__dict__
) | {"message", "asctime"}


class CustomJsonFormatter(JsonFormatter):
    """Format application logs as structured JSON."""

    def add_fields(self, log_record, record, message_dict):
        for field_name, field_value in record.__dict__.items():
            if field_name not in STANDARD_LOG_RECORD_FIELDS:
                log_record[field_name] = field_value

        log_record["timestamp"] = datetime.now(timezone.utc).isoformat()
        log_record["level"] = record.levelname
        log_record["service"] = "telco-churn-api"
        log_record["logger"] = record.name

        if not log_record.get("message"):
            log_record["message"] = record.getMessage()


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the application's structured logger.

    The handler is replaced on each call so importing the application repeatedly
    in a test runner does not duplicate log messages.
    """
    logger = logging.getLogger("api")
    logger.setLevel(getattr(logging, level.upper()))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s'
    ))
    logger.handlers = []
    logger.addHandler(handler)
    logger.propagate = False

    return logger