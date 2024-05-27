from celery import shared_task

from ..config import config
from ..core import ast, scsynth


@shared_task(bind=True)
def flush_milvus(self, *args, **kwargs) -> None:
    ast.get_ast_collection().flush()
    for index_config in config.analysis.scsynth_indices:
        scsynth.get_scsynth_collection(index_config["alias"]).flush()
