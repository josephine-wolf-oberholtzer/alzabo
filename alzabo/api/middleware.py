"""
Auth middleware
"""

import jwt
from aiohttp import web

from ..config import config


@web.middleware
async def auth_middleware(request: web.Request, handler) -> web.Response:
    if config.api.auth_enabled:
        if not (header := request.headers.get("authorization")):
            raise web.HTTPUnauthorized()
        elif not (token := header.split()[-1]):
            raise web.HTTPUnauthorized()
        try:
            jwt.decode(token, config.api.auth_secret, algorithms=["HS256"])
        except jwt.PyJWTError:
            raise web.HTTPUnauthorized()
    return await handler(request)
