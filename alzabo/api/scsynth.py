"""
scsynth query routes
"""

import asyncio
import base64
import concurrent.futures
import json
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import aiofiles
import ujson
from aiohttp import web
from aiohttp_apispec import json_schema, response_schema
from marshmallow import Schema, fields
from typing_extensions import TypedDict

from ..config import config
from ..core import audio, milvus, scsynth, utils

routes = web.RouteTableDef()


@routes.get(r"/scsynth/data/{digest:\w+}")
async def get_data(request: web.Request) -> web.Response:
    digest = request.match_info["digest"]
    data = []
    for entry in (
        await request.config_dict["s3"].list_objects(
            Bucket=config.s3.data_bucket,
            Prefix=utils.make_data_key(digest) + "/scsynth-entries-",
        )
    ).get("Contents", []):
        body = (
            await request.config_dict["s3"].get_object(
                Bucket=config.s3.data_bucket, Key=entry["Key"]
            )
        )["Body"]
        for entry in json.loads((await body.read()).decode())["entries"]:
            data.append(
                {
                    "count": entry[1],
                    "digest": digest,
                    "features": {
                        **{
                            f"r:chroma:{i}": x
                            for i, x in enumerate(entry[2]["r:chroma"])
                        },
                        **{
                            f"r:mfcc:{i}": x
                            for i, x in enumerate(entry[2]["r:mfcc"][:13])
                        },
                        **{
                            f"w:chroma:{i}": x
                            for i, x in enumerate(entry[2]["w:chroma"])
                        },
                        **{
                            f"w:mfcc:{i}": x
                            for i, x in enumerate(entry[2]["w:mfcc"][:13])
                        },
                        **{
                            key: value
                            for key, value in entry[2].items()
                            if key not in ("r:chroma", "r:mfcc", "w:chroma", "w:mfcc")
                        },
                    },
                    "start": entry[0],
                }
            )
    return web.json_response({"digest": digest, "entries": data})


class QueryScsynthItemSchema(Schema):
    digest = fields.String()
    start = fields.Integer()
    count = fields.Integer()
    distance = fields.Float()


class QueryScsynthRequestSchema(Schema):
    index = fields.String(allow_none=True)
    limit = fields.Integer(load_default=10)
    partitions = fields.List(fields.String, allow_none=True)
    vector = fields.List(fields.Float(), required=True)
    voiced = fields.Boolean(allow_none=True)


class QueryScsynthResponseSchema(Schema):
    entries = fields.List(fields.Nested(QueryScsynthItemSchema))
    timing = fields.Dict(keys=fields.Str(), values=fields.Float())


class QueryScsynthResponseType(TypedDict):
    entries: list[milvus.Entry]
    timing: dict[str, float]


@routes.post("/scsynth")
@json_schema(QueryScsynthRequestSchema)
@response_schema(QueryScsynthResponseSchema, 200)
async def query_scsynth(request: web.Request) -> web.Response:
    """
    Query against an scsynth-derived feature vector.
    """
    if not config.scsynth.enabled:
        raise web.HTTPBadRequest()
    data = QueryScsynthRequestSchema().load(await request.json(loads=ujson.loads))
    try:
        scsynth.get_index_config(data.get("index"))
    except ValueError:
        raise web.HTTPBadRequest()
    if len(data["vector"]) != scsynth.get_vector_size(data.get("index")):
        raise web.HTTPBadRequest()
    with utils.timer(request.app.logger, "Milvus time: {time}") as get_time:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            entries = await asyncio.get_running_loop().run_in_executor(
                pool,
                partial(
                    scsynth.query_scsynth_collection,
                    index_alias=data.get("index", None),
                    is_voiced=data.get("voiced"),
                    limit=data["limit"],
                    partition_names=data.get("partitions") or None,
                    vector=data["vector"],
                ),
            )
    milvus_time = get_time()
    response_body: QueryScsynthResponseType = {
        "entries": list(entries),
        "timing": {"milvus": milvus_time},
    }
    return web.json_response(response_body)


class QueryScsynthUploadRequestSchema(Schema):
    file = fields.String(required=True)
    index = fields.String(allow_none=True)
    limit = fields.Integer(load_default=10)
    partitions = fields.List(fields.String, allow_none=True)


class QueryScsynthUploadResponseSchema(Schema):
    analysis = fields.Dict(required=True)
    entries = fields.List(fields.Nested(QueryScsynthItemSchema))
    timing = fields.Dict(keys=fields.Str(), values=fields.Float())
    vector = fields.List(fields.Float)


class QueryScsynthUploadResponseType(TypedDict):
    analysis: scsynth.Aggregate
    entries: list[milvus.Entry]
    timing: dict[str, float]
    vector: list[float]


@routes.post("/scsynth/upload")
@json_schema(QueryScsynthUploadRequestSchema)
@response_schema(QueryScsynthUploadResponseSchema, 200)
async def query_scsynth_upload(request: web.Request) -> web.Response:
    """
    Upload an audio file and query against it.
    """
    if not config.scsynth.enabled:
        raise web.HTTPBadRequest()
    data = QueryScsynthUploadRequestSchema().load(await request.json(loads=ujson.loads))
    try:
        scsynth.get_index_config(data.get("index"))
    except ValueError:
        raise web.HTTPBadRequest()
    with TemporaryDirectory() as temp_directory:
        source_path = Path(temp_directory) / "source"
        target_path = Path(temp_directory) / "target.wav"
        async with aiofiles.open(source_path, "wb") as file_handle:
            await file_handle.write(base64.b64decode(data.pop("file")))
        with utils.timer(request.app.logger, "FFMPEG time: {time}") as get_time:
            await audio.transcode_audio_async(source_path, target_path)
        ffmpeg_time = get_time()
        with utils.timer(request.app.logger, "Scsynth time: {time}") as get_time:
            analysis = await scsynth.analyze(target_path)
            aggregate = scsynth.aggregate(analysis)
            vector = scsynth.aggregate_to_vector(
                aggregate, index_alias=data.get("index")
            )
        scsynth_time = get_time()
        with utils.timer(request.app.logger, "Milvus time: {time}") as get_time:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                entries = await asyncio.get_running_loop().run_in_executor(
                    pool,
                    partial(
                        scsynth.query_scsynth_collection,
                        index_alias=data.get("index", None),
                        is_voiced=aggregate["is_voiced"],
                        limit=data["limit"],
                        partition_names=data.get("partitions") or None,
                        vector=vector,
                    ),
                )
        milvus_time = get_time()
    response_body: QueryScsynthUploadResponseType = {
        "analysis": aggregate,
        "entries": list(entries),
        "timing": {
            "ffmpeg": ffmpeg_time,
            "scsynth": scsynth_time,
            "milvus": milvus_time,
        },
        "vector": list(vector),
    }
    return web.json_response(response_body)
