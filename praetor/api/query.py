"""
Query routes
"""

from aiohttp import web

from ..core import milvus
from .ast import routes as ast_routes
from .middleware import auth_middleware
from .scsynth import routes as scsynth_routes


def create_query_app() -> web.Application:
    async def connect_to_milvus(app: web.Application) -> None:
        milvus.connect()

    query_app = web.Application(middlewares=[auth_middleware])
    query_app.add_routes(ast_routes)
    query_app.add_routes(scsynth_routes)
    query_app.on_startup.append(connect_to_milvus)
    return query_app
