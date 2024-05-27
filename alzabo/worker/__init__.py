import logging
import sys

import redis
from celery import Celery, Task
from celery._state import get_current_task
from celery.signals import setup_logging, worker_process_init
from celery.utils.log import ColorFormatter

from ..config import config


def create_app() -> Celery:
    class TaskClass(Task):
        def __init__(self):
            self.redis = redis_connection

    redis_connection = redis.from_url(str(config.redis.url))

    return Celery(
        "alzabo-worker",
        broker=str(config.redis.url),
        backend=str(config.redis.url),
        include=[
            "alzabo.worker.audio",
            "alzabo.worker.ast",
            "alzabo.worker.milvus",
            "alzabo.worker.scsynth",
        ],
        task_cls=TaskClass,
    )


class HybridFormatter(ColorFormatter):
    """
    Custom logging formatter.

    Emits task logs when inside a task, otherwise standard logs.
    """

    def __init__(self):
        base_format = (
            "[%(asctime)s: %(levelname)s/%(processName)s] [%(name)s:%(lineno)d]"
        )
        worker_log_format = f"{base_format} %(message)s"
        worker_task_log_format = (
            f"{base_format} %(task_name)s[%(task_id)s]: %(message)s"
        )
        self.main_formatter = ColorFormatter(worker_log_format)
        self.task_formatter = ColorFormatter(worker_task_log_format)
        for formatter in (self.main_formatter, self.task_formatter):
            formatter.default_time_format = "%Y-%m-%dT%H:%M:%S"
            formatter.default_msec_format = "%s,%03dZ"

    def format(self, record):
        task = get_current_task()
        if task and task.request:
            record.__dict__.update(task_id=task.request.id, task_name=task.name)
            return self.task_formatter.format(record)
        return self.main_formatter.format(record)


@setup_logging.connect()
def on_setup_logging(*args, **kwargs):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(HybridFormatter())
    logger.addHandler(stream_handler)


@worker_process_init.connect
def on_worker_process_init(**kwargs) -> None:
    from ..core import milvus

    milvus.connect()
