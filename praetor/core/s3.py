from io import BytesIO
from typing import Generator

import boto3
from aiobotocore.session import get_session
from mypy_boto3_s3.client import S3Client

from ..config import config


def create_s3_client() -> S3Client:
    return boto3.client(
        "s3",
        aws_access_key_id=config.s3.access_key_id,
        aws_secret_access_key=config.s3.secret_access_key,
        endpoint_url=config.s3.endpoint_url,
    )


def create_async_s3_client():
    return get_session().create_client(
        "s3",
        aws_access_key_id=config.s3.access_key_id,
        aws_secret_access_key=config.s3.secret_access_key,
        endpoint_url=config.s3.endpoint_url,
    )


def list_digests(client: S3Client) -> Generator[str, None, None]:
    outer_paginator = client.get_paginator("list_objects_v2")
    for outer_result in outer_paginator.paginate(
        Bucket=config.s3.data_bucket, Delimiter="/"
    ):
        for common_outer_prefix in outer_result.get("CommonPrefixes", []):
            inner_paginator = client.get_paginator("list_objects_v2")
            for inner_result in inner_paginator.paginate(
                Bucket=config.s3.data_bucket,
                Delimiter="/",
                Prefix=common_outer_prefix["Prefix"],
            ):
                for common_inner_prefix in inner_result.get("CommonPrefixes", []):
                    if prefix := common_inner_prefix["Prefix"][3:-1]:
                        yield prefix


class ChunkedUploader:
    def __init__(self, bucket: str, key: str) -> None:
        self.bucket = bucket
        self.key = key
        self.upload_id: str | None = None
        self.etags: list[str] = []
        self.buffer_ = BytesIO()

    async def __aenter__(self) -> "ChunkedUploader":
        client = create_async_s3_client()
        self.client = await client.__aenter__()
        return self

    def __enter__(self) -> "ChunkedUploader":
        self.client = create_s3_client()
        return self

    async def __aexit__(self, *args) -> None:
        if self.buffer_.tell():
            if self.upload_id:
                self.etags.append(
                    (
                        await self.client.upload_part(
                            Body=self.buffer_.getvalue(),
                            Bucket=self.bucket,
                            Key=self.key,
                            PartNumber=len(self.etags),
                            UploadId=self.upload_id,
                        )
                    )["ETag"]
                )
            else:
                await self.client.put_object(
                    Body=self.buffer_.getvalue(), Bucket=self.bucket, Key=self.key
                )
        if self.upload_id:
            await self.client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=self.key,
                MultipartUpload=dict(
                    Parts=[
                        dict(ETag=etag, PartNumber=i)
                        for i, etag in enumerate(self.etags)
                    ]
                ),
                UploadId=self.upload_id,
            )
        await self.client.__aexit__(*args)

    def __exit__(self, *args) -> None:
        if self.buffer_.tell():
            if self.upload_id:
                self.etags.append(
                    self.client.upload_part(
                        Body=self.buffer_.getvalue(),
                        Bucket=self.bucket,
                        Key=self.key,
                        PartNumber=len(self.etags),
                        UploadId=self.upload_id,
                    )["ETag"]
                )
            else:
                self.client.put_object(
                    Body=self.buffer_.getvalue(), Bucket=self.bucket, Key=self.key
                )
        if self.upload_id:
            self.client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=self.key,
                MultipartUpload=dict(
                    Parts=[
                        dict(ETag=etag, PartNumber=i)
                        for i, etag in enumerate(self.etags)
                    ]
                ),
                UploadId=self.upload_id,
            )

    def write(self, chunk: bytes) -> None:
        self.buffer_.write(chunk)
        if self.buffer_.tell() < 5243000:  # 5MiB
            return
        if self.upload_id is None:
            self.upload_id = self.client.create_multipart_upload(
                Bucket=self.bucket, Key=self.key
            )["UploadId"]
        self.etags.append(
            self.client.upload_part(
                Body=self.buffer_.getvalue(),
                Bucket=self.bucket,
                Key=self.key,
                PartNumber=len(self.etags),
                UploadId=self.upload_id,
            )["ETag"]
        )
        self.buffer_ = BytesIO()

    async def write_async(self, chunk: bytes) -> None:
        self.buffer_.write(chunk)
        if self.buffer_.tell() < 5243000:  # 5MiB
            return
        if self.upload_id is None:
            self.upload_id = (
                await self.client.create_multipart_upload(
                    Bucket=self.bucket, Key=self.key
                )
            )["UploadId"]
        self.etags.append(
            (
                await self.client.upload_part(
                    Body=self.buffer_.getvalue(),
                    Bucket=self.bucket,
                    Key=self.key,
                    PartNumber=len(self.etags),
                    UploadId=self.upload_id,
                )
            )["ETag"]
        )
        self.buffer_ = BytesIO()
