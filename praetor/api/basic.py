"""
Basic routes
"""

from aiohttp import web

routes = web.RouteTableDef()


@routes.get("/")
async def index(request: web.Request) -> web.Response:
    return web.Response()


@routes.get("/ping")
async def ping(request: web.Request) -> web.Response:
    """
    Shallow healthcheck.
    """
    return web.Response(text="pong!")


@routes.get("/health")
async def health(request: web.Request) -> web.Response:
    """
    Deep healthcheck.
    """
    # TODO: Integrate with Milvus
    return web.Response()
