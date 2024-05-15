from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse
from uuid import uuid4

import requests
from botocore.exceptions import ClientError
from celery import shared_task
from celery.utils.log import get_task_logger
from mypy_boto3_s3.type_defs import CopySourceTypeDef

from ..config import config
from ..constants import AUDIO_FILENAME
from ..core.audio import transcode_audio
from ..core.s3 import ChunkedUploader, create_s3_client
from ..core.utils import hash_path, make_data_key, timer

logger = get_task_logger(__name__)


@shared_task(bind=True)
def upload_audio(self, job_id_and_url: tuple[str, str]) -> tuple[str, str]:
    """
    Fetch audio from ``url`` (HTTP or S3), upload to staging bucket
    """
    job_id, url = job_id_and_url
    logger.info(f"Staging {url} ...")
    client = create_s3_client()
    staging_id = str(uuid4())
    parse_result = urlparse(url)
    if parse_result.scheme == "s3":
        # Check if item is already in the uploads bucket.
        # This occurs when using the /audio/upload endpoint.
        if parse_result.netloc == config.s3.uploads_bucket and "/" not in (
            path := parse_result.path.lstrip("/")
        ):
            logger.info(f"Already staged: {url}")
            return job_id, path
        # Copy between S3 buckets
        logger.info(f"Uploading {url} to s3://{config.s3.uploads_bucket}/{staging_id}")
        with timer(logger, f"Staged {url} in " + "{time:.03f} seconds"):
            copy_source = CopySourceTypeDef(
                Bucket=parse_result.netloc, Key=parse_result.path.lstrip("/")
            )
            client.copy(
                CopySource=copy_source, Bucket=config.s3.uploads_bucket, Key=staging_id
            )
    elif parse_result.scheme in ("http", "https"):
        logger.info(f"Uploading {url} to s3://{config.s3.uploads_bucket}/{staging_id}")
        # Stream the HTTP request, upload chunks to S3
        with timer(logger, f"Staged {url} in " + "{time:.03f} seconds"):
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                with ChunkedUploader(
                    bucket=config.s3.uploads_bucket, key=staging_id
                ) as uploader:
                    for chunk in response.iter_content(chunk_size=8192):
                        uploader.write(chunk)
    else:
        raise ValueError(url)
    return job_id, staging_id


@shared_task(bind=True)
def transcode_and_hash_audio(
    self, job_id_and_staging_id: tuple[str, str]
) -> tuple[str, str]:
    """
    Transcode staged audio, hash it, upload to data bucket
    """
    job_id, staging_id = job_id_and_staging_id
    logger.info(f"Transcoding {staging_id} ...")
    client = create_s3_client()
    with timer(logger, f"Transcoded {staging_id} in " + "{time:.03f} seconds"):
        with TemporaryDirectory() as temp_directory:
            temp_path = Path(temp_directory)
            source_path = temp_path / f"{staging_id}-source.wav"
            target_path = temp_path / f"{staging_id}-target.wav"
            # Download unprocessed audio from uploads bucket
            client.download_file(
                Filename=str(source_path),
                Bucket=config.s3.uploads_bucket,
                Key=staging_id,
            )
            # Transcode to mono 48kHz WAV
            transcode_audio(source_path, target_path)
            # Calculate the SHA256
            digest = hash_path(target_path)
            logger.info(f"Hashed {staging_id} to {digest}")
            audio_key = f"{make_data_key(digest)}/{AUDIO_FILENAME}"
            # Check if data exists
            try:
                client.head_object(Bucket=config.s3.data_bucket, Key=audio_key)
                logger.info(f"Already uploaded {digest}!")
                return job_id, digest
            except ClientError as e:
                if e.response["Error"]["Code"] != "404":
                    raise
            # Upload transcoded WAV to data bucket
            logger.info(f"Uploading {staging_id} to {digest}")
            client.upload_file(
                Filename=str(target_path), Bucket=config.s3.data_bucket, Key=audio_key
            )
            # Delete staged file - it's no longer necessary
            client.delete_object(Bucket=config.s3.uploads_bucket, Key=staging_id)
    return job_id, digest
