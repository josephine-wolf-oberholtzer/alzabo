import pytest_asyncio

import praetor.api


@pytest_asyncio.fixture
async def api_client(aiohttp_client):
    """
    An AIOHTTP Client wrapping the Praetor API.
    """
    return await aiohttp_client(praetor.api.create_app())
