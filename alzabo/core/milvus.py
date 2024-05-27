from pymilvus import connections
from typing_extensions import TypedDict

from ..config import config


class Entry(TypedDict):
    digest: str
    start_frame: int
    frame_count: int
    distance: float


def connect() -> None:
    connections.connect(host=config.milvus.url.host, port=config.milvus.url.port)
