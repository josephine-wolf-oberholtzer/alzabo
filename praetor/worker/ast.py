import json
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from botocore.exceptions import ClientError
from celery import shared_task
from celery.utils.log import get_task_logger

from ..config import config
from ..constants import AST_ENTRIES_FILENAME, AUDIO_FILENAME
from ..core import ast
from ..core.s3 import create_s3_client
from ..core.utils import make_data_key, timer

logger = get_task_logger(__name__)


@shared_task(bind=True)
def analyze_via_ast(
    self,
    job_id_and_digest: tuple[str, str],
    hops: Sequence[int] | None = None,
    lengths: Sequence[int] | None = None,
) -> tuple[str, str]:
    """
    Analyze an audio file via AST.
    """
    job_id, digest = job_id_and_digest
    logger.info(f"Analyzing {digest} ...")
    client = create_s3_client()
    hops_ = hops or config.analysis.hops
    lengths_ = lengths or config.analysis.lengths
    with timer(logger, f"Partititioned {digest} in " + "{time:.03f} seconds"):
        with TemporaryDirectory() as temp_directory:
            source_path = Path(temp_directory) / AUDIO_FILENAME
            client.download_file(
                Filename=str(source_path),
                Bucket=config.s3.data_bucket,
                Key=f"{make_data_key(digest)}/{AUDIO_FILENAME}",
            )
            with timer(logger, "Model loaded in " + "{time:.03f} seconds"):
                model = ast.load_model()
            for hop, length in product(hops_, lengths_):
                logger.info(f"Analyzing {digest} with {hop=} / {length=} ...")
                entries_filename = AST_ENTRIES_FILENAME.format(hop=hop, length=length)
                entries_key = f"{make_data_key(digest)}/{entries_filename}"
                try:
                    client.head_object(Bucket=config.s3.data_bucket, Key=entries_key)
                    logger.info(
                        f"Already partitioned {digest} with {hop=} / {length=}!"
                    )
                    continue
                except ClientError as e:
                    if e.response["Error"]["Code"] != "404":
                        raise
                entries = ast.partition(source_path, model, hop, length)
                entries_path = Path(temp_directory) / entries_filename
                entries_path.write_text(
                    json.dumps(
                        dict(digest=digest, entries=entries, hop=hop, length=length),
                        indent=2,
                        sort_keys=True,
                    )
                )
                client.upload_file(
                    Bucket=config.s3.data_bucket,
                    Filename=str(entries_path),
                    Key=entries_key,
                )
    return job_id, digest


@shared_task(bind=True)
def insert_ast_entries(
    self, job_id_and_digest: tuple[str, str], partition_name=None
) -> tuple[str, str]:
    job_id, digest = job_id_and_digest
    logger.info(f"Inserting {digest} ...")
    # create partition
    ast.create_ast_partition(digest)
    # loop over entry jsons and insert
    client = create_s3_client()
    with timer(logger, f"Inserted {digest} in " + "{time:.03f} seconds"):
        with TemporaryDirectory() as temp_directory:
            for hop, length in product(config.analysis.hops, config.analysis.lengths):
                entries_filename = AST_ENTRIES_FILENAME.format(hop=hop, length=length)
                entries_path = Path(temp_directory) / entries_filename
                client.download_file(
                    Bucket=config.s3.data_bucket,
                    Filename=str(entries_path),
                    Key=f"{make_data_key(digest)}/{entries_filename}",
                )
                data = json.loads(entries_path.read_text())
                ast.insert_ast_entries(
                    digest=digest, entries=data["entries"], partition_name=digest
                )
    return job_id, digest
