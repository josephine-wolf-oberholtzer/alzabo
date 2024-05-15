import asyncio
import json
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import numpy
from botocore.exceptions import ClientError
from celery import shared_task
from celery.utils.log import get_task_logger
from sklearn.preprocessing import StandardScaler

from ..config import config
from ..constants import (
    AUDIO_FILENAME,
    SCSYNTH_ANALYSIS_RAW_FILENAME,
    SCSYNTH_ANALYSIS_WHITENED_FILENAME,
    SCSYNTH_ENTRIES_FILENAME,
)
from ..core import scsynth
from ..core.s3 import create_s3_client, list_digests
from ..core.utils import make_data_key, timer

logger = get_task_logger(__name__)


@shared_task(bind=True)
def analyze_via_scsynth(self, job_id_and_digest: tuple[str, str]) -> tuple[str, str]:
    """
    Analyze an audio file via NRT scsynth and upload to S3.

    Generate and upload a whitened version as well, based on the current
    whitening parameters cached in Redis.
    """
    job_id, digest = job_id_and_digest
    logger.info(f"Analyzing {digest} ...")
    with timer(logger, f"Analyzed {digest} in " + "{time:.03f} seconds"):
        client = create_s3_client()
        data_key = make_data_key(digest)
        try:
            # Return early if both analyses already exist
            client.head_object(
                Bucket=config.s3.data_bucket,
                Key=f"{data_key}/{SCSYNTH_ANALYSIS_RAW_FILENAME}",
            )
            client.head_object(
                Bucket=config.s3.data_bucket,
                Key=f"{data_key}/{SCSYNTH_ANALYSIS_WHITENED_FILENAME}",
            )
            logger.info(f"Already analyzed {digest}!")
            return job_id, digest
        except ClientError:
            pass
        with TemporaryDirectory() as temp_directory:
            source_path = Path(temp_directory) / AUDIO_FILENAME
            client.download_file(
                Filename=str(source_path),
                Bucket=config.s3.data_bucket,
                Key=f"{make_data_key(digest)}/{AUDIO_FILENAME}",
            )
            raw_analysis_array = asyncio.run(scsynth.analyze(source_path))
            whitened_analysis_array = scsynth.whiten(
                array=raw_analysis_array, redis=self.redis
            )
            for filename, array in [
                (SCSYNTH_ANALYSIS_RAW_FILENAME, raw_analysis_array),
                (SCSYNTH_ANALYSIS_WHITENED_FILENAME, whitened_analysis_array),
            ]:
                path = Path(temp_directory) / filename
                path.write_text(json.dumps(array.tolist(), indent=2, sort_keys=True))
                client.upload_file(
                    Filename=str(path),
                    Bucket=config.s3.data_bucket,
                    Key=f"{make_data_key(digest)}/{filename}",
                )
    return job_id, digest


@shared_task(bind=True)
def partition_scsynth_analysis(
    self,
    job_id_and_digest: tuple[str, str],
    hops: Sequence[int] | None = None,
    lengths: Sequence[int] | None = None,
) -> tuple[str, str]:
    """
    Partition an scsynth analysis into multiple entries JSON files.
    """
    job_id, digest = job_id_and_digest
    logger.info(f"Partitioning {digest} ...")
    client = create_s3_client()
    hops_ = hops or config.analysis.hops
    lengths_ = lengths or config.analysis.lengths
    with timer(logger, f"Partitioned {digest} in " + "{time:.03f} seconds"):
        with TemporaryDirectory() as temp_directory:
            raw_analysis_path = Path(temp_directory) / SCSYNTH_ANALYSIS_RAW_FILENAME
            whitened_analysis_path = (
                Path(temp_directory) / SCSYNTH_ANALYSIS_WHITENED_FILENAME
            )
            client.download_file(
                Bucket=config.s3.data_bucket,
                Filename=str(raw_analysis_path),
                Key=f"{make_data_key(digest)}/{SCSYNTH_ANALYSIS_RAW_FILENAME}",
            )
            client.download_file(
                Bucket=config.s3.data_bucket,
                Filename=str(whitened_analysis_path),
                Key=f"{make_data_key(digest)}/{SCSYNTH_ANALYSIS_WHITENED_FILENAME}",
            )
            raw_analysis = numpy.array(json.loads(raw_analysis_path.read_text()))
            whitened_analysis = numpy.array(
                json.loads(whitened_analysis_path.read_text())
            )
            for hop, length in product(hops_, lengths_):
                logger.info(f"Partitioning {digest} with {hop=} / {length=} ...")
                entries_filename = SCSYNTH_ENTRIES_FILENAME.format(
                    hop=hop, length=length
                )
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
                entries = scsynth.partition(
                    hop_ms=hop,
                    length_ms=length,
                    raw_analysis=raw_analysis,
                    whitened_analysis=whitened_analysis,
                )
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
def insert_scsynth_entries(
    self, job_id_and_digest: tuple[str, str], whitened: bool = False
) -> tuple[str, str]:
    """
    Insert entries for ``digest`` into Milvus as a partition.

    Drop any pre-existing partition to prevent duplicates.
    """
    job_id, digest = job_id_and_digest
    logger.info(f"Inserting {digest} ...")
    # loop over entry jsons and insert
    client = create_s3_client()
    with timer(logger, f"Inserted {digest} in " + "{time:.03f} seconds"):
        with TemporaryDirectory() as temp_directory:
            for hop, length in product(config.analysis.hops, config.analysis.lengths):
                entries_filename = (
                    SCSYNTH_ENTRIES_FILENAME if whitened else SCSYNTH_ENTRIES_FILENAME
                ).format(hop=hop, length=length)
                entries_path = Path(temp_directory) / entries_filename
                client.download_file(
                    Bucket=config.s3.data_bucket,
                    Filename=str(entries_path),
                    Key=f"{make_data_key(digest)}/{entries_filename}",
                )
                data = json.loads(entries_path.read_text())
                scsynth.insert_scsynth_entries(
                    digest=digest, entries=data["entries"], partition_name=digest
                )
    return job_id, digest


@shared_task(bind=True)
def whiten(self) -> None:
    logger.info("Whitening ...")
    s3_client = create_s3_client()
    scaler = StandardScaler()
    for digest in list_digests(s3_client):
        logger.info(f"Fitting {digest} ...")
        data = json.loads(
            s3_client.get_object(
                Bucket=config.s3.data_bucket,
                Key=make_data_key(digest) + "/" + SCSYNTH_ANALYSIS_RAW_FILENAME,
            )["Body"].read()
        )
        scaler.partial_fit(data)
    logger.info("... fitting done: {scaler.get_params()}")
    for digest in list_digests(s3_client):
        logger.info(f"Transforming {digest} ...")
        data = json.loads(
            s3_client.get_object(
                Bucket=config.s3.data_bucket,
                Key=make_data_key(digest) + "/" + SCSYNTH_ANALYSIS_RAW_FILENAME,
            )["Body"].read()
        )
        transformed_data = scaler.transform(data).tolist()
        with TemporaryDirectory() as temp_path:
            path = Path(temp_path) / "data.json"
            path.write_text(json.dumps(transformed_data))
            s3_client.upload_file(
                Bucket=config.s3.data_bucket,
                Filename=str(path),
                Key=make_data_key(digest) + "/" + SCSYNTH_ANALYSIS_WHITENED_FILENAME,
            )
    logger.info("... transforming done!")
    scsynth.serialize_whitener(redis=self.redis, scaler=scaler)
