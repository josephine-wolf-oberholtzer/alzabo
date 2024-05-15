from pathlib import Path
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from supriya import AsyncClock, AsyncServer, DoneAction, SynthDef, SynthDefBuilder
from supriya.ugens import BufDur, Line, Out, PlayBuf

from praetor.client import APIClient, Application, BufferManager, Performer


@pytest.fixture
def api_client(api_server: str, milvus_scsynth_collections) -> APIClient:
    return APIClient(api_server)


@pytest.fixture
def application(api_server: str) -> Application:
    return Application()


@pytest.fixture
def buffer_manager(
    api_client: APIClient, context: AsyncServer, tmp_path: Path
) -> BufferManager:
    return BufferManager(
        api_client=api_client, context=context, download_directory=tmp_path
    )


@pytest_asyncio.fixture
async def clock() -> AsyncGenerator[AsyncClock, None]:
    clock = AsyncClock()
    clock.change(beats_per_minute=60 * 4)
    await clock.start()
    yield clock
    await clock.stop()


@pytest.fixture
def performer(
    buffer_manager: BufferManager, clock: AsyncClock, context: AsyncServer
) -> Performer:
    return Performer(buffer_manager=buffer_manager, clock=clock, context=context)


@pytest.fixture
def synthdef() -> SynthDef:
    with SynthDefBuilder(buffer_id=0, out=0) as builder:
        source = PlayBuf.ar(buffer_id=builder["buffer_id"], channel_count=1, loop=0)
        window = Line.kr(
            duration=BufDur.kr(buffer_id=builder["buffer_id"]),
            done_action=DoneAction.FREE_SYNTH,
        ).hanning_window()
        Out.ar(bus=builder["out"], source=source * window)
    return builder.build()
