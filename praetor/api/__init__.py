"""
Praetor API
"""

import aiohttp_cors
import redis
from aiohttp import web
from aiohttp_apispec import setup_aiohttp_apispec

from ..config import config
from ..core import s3
from ..core.ast import load_model
from ..worker import create_app as create_celery_app
from .audio import create_audio_app
from .basic import routes as basic_routes
from .query import create_query_app


def create_app() -> web.Application:
    async def on_startup(app: web.Application) -> None:
        app["ast"] = load_model()
        app["celery"] = create_celery_app()
        app["s3"] = await s3.create_async_s3_client().__aenter__()
        app["redis"] = redis.from_url(str(config.redis.url))

    async def on_shutdown(app: web.Application) -> None:
        await app["s3"].__aexit__(None, None, None)

    app = web.Application()
    # routes
    app.add_routes(basic_routes)
    app.add_subapp("/audio", create_audio_app())
    app.add_subapp("/query", create_query_app())
    # openapi
    setup_aiohttp_apispec(
        app=app,
        title="Praetor",
        version="v1",
        url="/api/docs/swagger.json",
        swagger_path="/api/docs",
    )
    # S3
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    # CORS
    cors = aiohttp_cors.setup(
        app,
        defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True, expose_headers="*", allow_headers="*"
            )
        },
    )
    for route in app.router.routes():
        cors.add(route)
    return app
