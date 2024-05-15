"""
Audio upload routes
"""
import logging
import struct
from uuid import uuid4

from aiohttp import web
from aiohttp_apispec import (
    form_schema,
    json_schema,
    querystring_schema,
    response_schema,
)
from botocore.exceptions import ClientError
from marshmallow import Schema, fields

from ..config import config
from ..constants import AUDIO_FILENAME
from ..core.s3 import ChunkedUploader
from ..core.utils import make_data_key
from ..worker import tasks
from .middleware import auth_middleware

logger = logging.getLogger(__name__)

routes = web.RouteTableDef()


class AudioBatchRequestSchema(Schema):
    urls = fields.List(fields.Str(), required=True)


class AudioBatchResponseSchema(Schema):
    jobs = fields.List(fields.Str(), required=True)


@routes.post("/batch")
@json_schema(AudioBatchRequestSchema)
@response_schema(AudioBatchResponseSchema, 200)
async def batch(request: web.Request) -> web.Response:
    """
    Send a collection of URLs to be batch processed
    """
    job_ids: list[str] = []
    for url in (await request.json())["urls"]:
        job_id = str(uuid4())
        tasks.get_audio_processing_chain(job_id, url)()
        job_ids.append(job_id)
    return web.json_response({"jobs": job_ids})


class AudioFetchRequestSchema(Schema):
    start = fields.Int()
    count = fields.Int()


@routes.get(r"/fetch/{digest:\w+}")
@querystring_schema(AudioFetchRequestSchema)
async def fetch(request: web.Request) -> web.Response:
    digest = request.match_info["digest"]
    start_frame = int(request.query.get("start", 0))
    frame_count = int(request.query.get("count", 0))
    if start_frame and not frame_count:
        raise web.HTTPBadRequest()
    filename = f"{digest}.wav"
    byte_range = ""
    if frame_count:
        offset = 78  # FFMPEG adds additional metadata
        start_byte = offset + (start_frame * 2)
        stop_byte = start_byte + (frame_count * 2)
        byte_range = f"bytes={start_byte}-{stop_byte}"
        filename = f"{digest}-{start_frame}-{frame_count}.wav"
    client = request.config_dict["s3"]
    try:
        s3_response = await client.get_object(
            Bucket=config.s3.data_bucket,
            Key=f"{make_data_key(digest)}/{AUDIO_FILENAME}",
            Range=byte_range,
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            raise web.HTTPNotFound() from e
        raise web.HTTPInternalServerError() from e
    content_length = s3_response["ContentLength"]
    if byte_range:
        content_length += 44 - 1  # synthetic wave header
    response = web.Response(
        content_type="audio/x-wav",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Length": f"{content_length}",
        },
    )
    await response.prepare(request)
    if frame_count:
        header = struct.pack(
            "<4sL4s4sLHHLLHH4sL",
            b"RIFF",
            36 + s3_response["ContentLength"] - 1,  # file size
            b"WAVE",  # WAV description header
            b"fmt ",  # format description header
            16,  # remaining header size
            1,  # wav type format
            1,  # channel count
            48000,  # frame rate
            48000 * 2,  # audio data rate
            2,  # block alignment
            16,  # bits per sample
            b"data",  # data description header
            s3_response["ContentLength"] - 1,  # data chunk size
        )
        await response.write(header)
    async for chunk in s3_response["Body"].iter_chunks(8192):
        await response.write(chunk)
    return response


@routes.get("/partitions")
async def get_partitions(request: web.Request) -> web.Response:
    """
    List partitions.
    """
    partitions: list[str] = []
    for common_prefix_data in (
        await request.config_dict["s3"].list_objects(
            Bucket=config.s3.data_bucket, Delimiter="/"
        )
    )["CommonPrefixes"]:
        for prefix_data in (
            await request.config_dict["s3"].list_objects(
                Bucket=config.s3.data_bucket,
                Delimiter="/",
                Prefix=common_prefix_data["Prefix"],
            )
        )["CommonPrefixes"]:
            partitions.append(prefix_data["Prefix"][3:-1])
    return web.json_response({"partitions": partitions})


class UploadRequestSchema(Schema):
    file = fields.Field(required=True, metadata={"location": "form", "type": "file"})


@routes.post("/upload")
@form_schema(UploadRequestSchema)
async def upload(request: web.Request) -> web.Response:
    """
    Upload an audio sample
    """
    job_id = str(uuid4())
    staging_id = str(uuid4())
    async for field in (await request.multipart()):
        print(f"{request=} {field=} {field.name=}")
        if field.name != "file":
            continue
        async with ChunkedUploader(
            bucket=config.s3.uploads_bucket, key=staging_id
        ) as uploader:
            while chunk := await field.read_chunk():
                await uploader.write_async(chunk)
    tasks.get_audio_processing_chain(
        job_id, f"s3://{config.s3.uploads_bucket}/{staging_id}"
    )()
    return web.json_response({"job": job_id})


def create_audio_app() -> web.Application:
    audio_app = web.Application(middlewares=[auth_middleware])
    audio_app.add_routes(routes)
    return audio_app
