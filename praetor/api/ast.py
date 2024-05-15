"""
AST query routes
"""

import asyncio
import base64
import concurrent.futures
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import aiofiles
import ujson
from aiohttp import web
from aiohttp_apispec import json_schema, response_schema
from marshmallow import Schema, fields, validate
from typing_extensions import TypedDict

from ..config import config
from ..core import ast, milvus, utils

routes = web.RouteTableDef()


class QueryAstItemSchema(Schema):
    count = fields.Integer()
    digest = fields.String()
    distance = fields.Float()
    start = fields.Integer()


class QueryAstRequestSchema(Schema):
    limit = fields.Integer(load_default=10)
    partitions = fields.List(fields.String, allow_none=True)
    vector = fields.List(
        fields.Float(),
        required=True,
        validate=[validate.Length(equal=ast.get_vector_size())],
    )


class QueryAstResponseSchema(Schema):
    entries = fields.List(fields.Nested(QueryAstItemSchema))
    timing = fields.Dict(keys=fields.Str(), values=fields.Float())


class QueryAstResponseType(TypedDict):
    entries: list[milvus.Entry]
    timing: dict[str, float]


@routes.post("/ast")
@json_schema(QueryAstRequestSchema)
@response_schema(QueryAstResponseSchema, 200)
async def query_ast(request: web.Request) -> web.Response:
    if not config.ast.enabled:
        return web.json_response({"message": "AST not enabled"}, status=400)
    data = QueryAstRequestSchema().load(await request.json(loads=ujson.loads))
    with utils.timer(request.app.logger, "Milvus time: {time}") as get_time:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            entries = await asyncio.get_running_loop().run_in_executor(
                pool,
                partial(
                    ast.query_ast_collection,
                    limit=data["limit"],
                    partition_names=data.get("partitions") or None,
                    vector=data["vector"],
                ),
            )
    milvus_time = get_time()
    response_body: QueryAstResponseType = {
        "entries": list(entries),
        "timing": {"milvus": milvus_time},
    }
    return web.json_response(response_body)


class QueryAstUploadRequestSchema(Schema):
    file = fields.String(required=True)
    limit = fields.Integer(load_default=10)
    partitions = fields.List(fields.String, allow_none=True)


class QueryAstUploadResponseSchema(Schema):
    entries = fields.List(fields.Nested(QueryAstItemSchema))
    timing = fields.Dict(keys=fields.Str(), values=fields.Float())
    vector = fields.List(fields.Float)


class QueryAstUploadResponseType(TypedDict):
    entries: list[milvus.Entry]
    timing: dict[str, float]
    vector: list[float]


@routes.post("/ast/upload")
@json_schema(QueryAstUploadRequestSchema)
@response_schema(QueryAstUploadResponseSchema, 200)
async def query_ast_upload(request: web.Request) -> web.Response:
    if not config.ast.enabled:
        return web.json_response({"message": "AST not enabled"}, status=400)
    data = QueryAstUploadRequestSchema().load(await request.json(loads=ujson.loads))
    with TemporaryDirectory() as temp_directory:
        target_path = Path(temp_directory) / "target.wav"
        async with aiofiles.open(target_path, "wb") as file_handle:
            await file_handle.write(base64.b64decode(data.pop("file")))
        with concurrent.futures.ThreadPoolExecutor() as pool:
            with utils.timer(request.app.logger, "AST time: {time}") as get_time:
                vector = await asyncio.get_running_loop().run_in_executor(
                    pool, partial(ast.analyze, target_path, request.config_dict["ast"])
                )
            ast_time = get_time()
            with utils.timer(request.app.logger, "Milvus time: {time}") as get_time:
                entries = await asyncio.get_running_loop().run_in_executor(
                    pool,
                    partial(
                        ast.query_ast_collection,
                        limit=data["limit"],
                        partition_names=data.get("partitions") or None,
                        vector=vector,
                    ),
                )
            milvus_time = get_time()
    response_body: QueryAstUploadResponseType = {
        "entries": list(entries),
        "timing": {"ast": ast_time, "milvus": milvus_time},
        "vector": list(vector),
    }
    return web.json_response(response_body)
