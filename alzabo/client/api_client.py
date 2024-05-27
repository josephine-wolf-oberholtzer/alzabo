import base64
from io import BufferedReader
from pathlib import Path
from typing import Callable, Sequence

import aiofiles
import aiohttp
import ujson

from ..api.ast import QueryAstResponseType, QueryAstUploadResponseType
from ..api.scsynth import QueryScsynthResponseType, QueryScsynthUploadResponseType


class APIClient:
    """
    A client for talking to the Alzabo Cloud API.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str | None = None,
        *,
        connector: aiohttp.BaseConnector | None = None,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.connector = connector

    def _headers(self):
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    async def audio_batch(self, *, urls: Sequence[str]) -> Sequence[str]:
        async with aiohttp.ClientSession(connector=self.connector) as session:
            async with session.post(
                f"{self.api_url}/audio/batch",
                headers=self._headers(),
                json=dict(urls=urls),
            ) as response:
                response.raise_for_status()
                return (await response.json(loads=ujson.loads))["jobs"]

    async def audio_fetch(
        self, *, digest: str, start_frame: int, frame_count: int
    ) -> bytes:
        async with aiohttp.ClientSession(connector=self.connector) as session:
            async with session.get(
                f"{self.api_url}/audio/fetch/{digest}",
                headers=self._headers(),
                params=dict(start=start_frame, count=frame_count),
            ) as response:
                response.raise_for_status()
                return await response.content.read()

    async def audio_upload(
        self,
        *,
        path: Path,
        progress_callback: Callable[[float, float, float], None] | None = None,
    ) -> str:
        with ProgressFileReader(path, callback=progress_callback) as file_pointer:
            form_data = aiohttp.FormData()
            form_data.add_field("file", file_pointer)
            async with aiohttp.ClientSession(connector=self.connector) as session:
                async with session.post(
                    f"{self.api_url}/audio/upload",
                    data=form_data,
                    headers=self._headers(),
                ) as response:
                    response.raise_for_status()
                    return (await response.json(loads=ujson.loads))["job"]

    async def ping(self) -> str:
        async with aiohttp.ClientSession(connector=self.connector) as session:
            async with session.get(f"{self.api_url}/ping") as response:
                response.raise_for_status()
                return await response.text()

    async def query_ast(
        self,
        *,
        limit: int = 10,
        partitions: Sequence[str] | None = None,
        vector: Sequence[float],
    ) -> QueryAstResponseType:
        async with aiohttp.ClientSession(connector=self.connector) as session:
            async with session.post(
                f"{self.api_url}/query/ast",
                json=dict(
                    limit=limit,
                    partitions=list(partitions) if partitions else None,
                    vector=list(vector),
                ),
                headers=self._headers(),
            ) as response:
                response.raise_for_status()
                return await response.json(loads=ujson.loads)

    async def query_ast_upload(
        self, *, path: Path, limit: int = 10, partitions: Sequence[str] | None = None
    ) -> QueryAstUploadResponseType:
        async with aiofiles.open(path, "rb") as file_pointer:
            file_contents = await file_pointer.read()
        async with aiohttp.ClientSession(connector=self.connector) as session:
            async with session.post(
                f"{self.api_url}/query/ast/upload",
                json=dict(
                    file=base64.b64encode(file_contents).decode(),
                    limit=limit,
                    partitions=list(partitions) if partitions else None,
                ),
                headers=self._headers(),
            ) as response:
                response.raise_for_status()
                return await response.json(loads=ujson.loads)

    async def query_scsynth(
        self,
        *,
        index: str | None = None,
        limit: int = 10,
        partitions: Sequence[str] | None = None,
        vector: Sequence[float],
        voiced: bool | None = None,
    ) -> QueryScsynthResponseType:
        async with aiohttp.ClientSession(connector=self.connector) as session:
            async with session.post(
                f"{self.api_url}/query/scsynth",
                json=dict(
                    index=index,
                    limit=limit,
                    partitions=list(partitions) if partitions else None,
                    vector=list(vector),
                    voiced=voiced,
                ),
                headers=self._headers(),
            ) as response:
                response.raise_for_status()
                return await response.json(loads=ujson.loads)

    async def query_scsynth_upload(
        self,
        *,
        path: Path,
        index: str | None = None,
        limit: int = 10,
        partitions: Sequence[str] | None = None,
    ) -> QueryScsynthUploadResponseType:
        async with aiofiles.open(path, "rb") as file_pointer:
            file_contents = await file_pointer.read()
        async with aiohttp.ClientSession(connector=self.connector) as session:
            async with session.post(
                f"{self.api_url}/query/scsynth/upload",
                json=dict(
                    file=base64.b64encode(file_contents).decode(),
                    index=index,
                    limit=limit,
                    partitions=list(partitions) if partitions else None,
                ),
                headers=self._headers(),
            ) as response:
                response.raise_for_status()
                return await response.json(loads=ujson.loads)


class ProgressFileReader(BufferedReader):
    def __init__(
        self, path: Path, callback: Callable[[float, float, float], None] | None = None
    ):
        file_pointer = path.open("rb", buffering=0)
        self._callback = callback
        super().__init__(raw=file_pointer)
        self.length = path.stat().st_size

    def read(self, size=None):
        if self._callback:
            self._callback(size, self.tell(), self.length)
        return super(ProgressFileReader, self).read(size)
