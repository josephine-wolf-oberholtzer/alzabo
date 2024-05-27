import asyncio
from pathlib import Path
from uuid import uuid4

import pytest
from supriya import AsyncServer
from supriya.osc import OscMessage

from alzabo.client import BufferManager


@pytest.mark.asyncio
async def test_one_sync(
    buffer_manager: BufferManager, context: AsyncServer, data: None, tmp_path: Path
) -> None:
    """
    Acquire one buffer, release it.
    """
    uuid = uuid4()
    # Acquire buffer references
    with context.osc_protocol.capture() as transcript:
        buffers = await buffer_manager.acquire(
            reference=uuid,
            entries=[
                dict(
                    digest="dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969",
                    start_frame=0,
                    frame_count=1024,
                    distance=666.0,
                )
            ],
        )
    assert transcript.filtered(received=False, status=False) == [
        OscMessage(
            "/b_allocRead",
            0,
            str(
                tmp_path
                / "dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969-0-1024.wav"
            ),
            0,
            0,
        ),
        OscMessage("/sync", 1),
    ]
    # Increment them
    with context.osc_protocol.capture() as transcript:
        for i, buffer in enumerate(buffers):
            buffer_manager.increment(reference=i, buffer_id=buffer.id_)
    assert transcript.filtered(received=False, status=False) == []
    # Decrement them
    with context.osc_protocol.capture() as transcript:
        for i in range(len(buffers)):
            buffer_manager.decrement(reference=i)
    assert transcript.filtered(received=False, status=False) == []
    # Release them
    with context.osc_protocol.capture() as transcript:
        buffer_manager.decrement(reference=uuid)
    assert transcript.filtered(received=False, status=False) == [
        OscMessage("/b_free", 0)
    ]


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=5)
async def test_interleaved(
    buffer_manager: BufferManager, context: AsyncServer, data: None, tmp_path: Path
) -> None:
    """
    Acquire one, acquire again, release one, release again.
    """
    uuid_a, uuid_b = uuid4(), uuid4()
    # Acquire buffer references
    with context.osc_protocol.capture() as transcript:
        buffers_a = await buffer_manager.acquire(
            reference=uuid_a,
            entries=[
                dict(
                    digest="dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969",
                    start_frame=0,
                    frame_count=1024,
                    distance=666.0,
                ),
                dict(
                    digest="dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969",
                    start_frame=512,
                    frame_count=1024,
                    distance=666.0,
                ),
            ],
        )
    # Buffer IDs are not deterministic, but there's only two
    # ... so we can re-run the test if it flakes
    assert sorted(
        message.to_list()
        for message in transcript.filtered(received=False, status=False)
    ) == [
        [
            "/b_allocRead",
            0,
            str(
                tmp_path
                / "dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969-0-1024.wav"
            ),
            0,
            0,
        ],
        [
            "/b_allocRead",
            1,
            str(
                tmp_path
                / "dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969-512-1024.wav"
            ),
            0,
            0,
        ],
        ["/sync", 1],
        ["/sync", 2],
    ]
    # Increment the first set of buffers
    with context.osc_protocol.capture() as transcript:
        for i, buffer in enumerate(buffers_a):
            buffer_manager.increment(reference=i, buffer_id=buffer.id_)
    assert transcript.filtered(received=False, status=False) == []
    # Acquire overlapping buffer references
    with context.osc_protocol.capture() as transcript:
        buffers_b = await buffer_manager.acquire(
            reference=uuid_b,
            entries=[
                dict(
                    digest="dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969",
                    start_frame=512,
                    frame_count=1024,
                    distance=666.0,
                ),
                dict(
                    digest="dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969",
                    start_frame=768,
                    frame_count=1024,
                    distance=666.0,
                ),
            ],
        )
    assert sorted(
        message.to_list()
        for message in transcript.filtered(received=False, status=False)
    ) == [
        [
            "/b_allocRead",
            2,
            str(
                tmp_path
                / "dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969-768-1024.wav"
            ),
            0,
            0,
        ],
        ["/sync", 3],
    ]
    # Increment the second set of buffers
    with context.osc_protocol.capture() as transcript:
        for i, buffer in enumerate(buffers_b, len(buffers_a)):
            buffer_manager.increment(reference=i, buffer_id=buffer.id_)
    assert transcript.filtered(received=False, status=False) == []
    # Decrement the first set of buffers
    with context.osc_protocol.capture() as transcript:
        for i in range(len(buffers_a)):
            buffer_manager.decrement(reference=i)
    assert transcript.filtered(received=False, status=False) == []
    # Release the first set of buffers
    with context.osc_protocol.capture() as transcript:
        buffer_manager.decrement(reference=uuid_a)
    assert transcript.filtered(received=False, status=False) == [
        OscMessage("/b_free", 0)
    ]
    # Decrement the second set of buffers
    with context.osc_protocol.capture() as transcript:
        for i, _ in enumerate(buffers_b, len(buffers_a)):
            buffer_manager.decrement(reference=i)
    assert transcript.filtered(received=False, status=False) == []
    # Release the second set of buffers
    with context.osc_protocol.capture() as transcript:
        buffer_manager.decrement(reference=uuid_b)
    assert sorted(
        message.to_list()
        for message in transcript.filtered(received=False, status=False)
    ) == [["/b_free", 1], ["/b_free", 2]]


@pytest.mark.asyncio
async def test_async(
    buffer_manager: BufferManager, context: AsyncServer, data: None, tmp_path: Path
) -> None:
    """
    Acquire multiple buffers in a task, acquire a subset in a second task, release as tasks too
    """
    uuid_a, uuid_b = uuid4(), uuid4()
    with context.osc_protocol.capture() as transcript:
        await asyncio.gather(
            buffer_manager.acquire(
                reference=uuid_a,
                entries=[
                    dict(
                        digest="dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969",
                        start_frame=0,
                        frame_count=1024,
                        distance=666.0,
                    ),
                    dict(
                        digest="dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969",
                        start_frame=512,
                        frame_count=1024,
                        distance=666.0,
                    ),
                ],
            ),
            buffer_manager.acquire(
                reference=uuid_b,
                entries=[
                    dict(
                        digest="dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969",
                        start_frame=512,
                        frame_count=1024,
                        distance=666.0,
                    ),
                    dict(
                        digest="dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969",
                        start_frame=768,
                        frame_count=1024,
                        distance=666.0,
                    ),
                ],
            ),
        )
    sent_messages = [
        message.to_list()
        for message in transcript.filtered(received=False, status=False)
    ]
    assert len(sent_messages) == 6
    sync_messages = [message for message in sent_messages if message[0] == "/sync"]
    assert sync_messages == [["/sync", 1], ["/sync", 2], ["/sync", 3]]
    b_alloc_read_messages = [
        message for message in sent_messages if message[0] == "/b_allocRead"
    ]
    # Order of /b_allocRead (and therefore buffer IDs) is not deterministic
    # ... so we just grab the paths and verify all were allocated
    assert sorted(message[2] for message in b_alloc_read_messages) == [
        str(
            tmp_path
            / "dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969-0-1024.wav"
        ),
        str(
            tmp_path
            / "dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969-512-1024.wav"
        ),
        str(
            tmp_path
            / "dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969-768-1024.wav"
        ),
    ]
    with context.osc_protocol.capture() as transcript:
        buffer_manager.decrement(reference=uuid_a)
        buffer_manager.decrement(reference=uuid_b)
    assert sorted(
        message.to_list()
        for message in transcript.filtered(received=False, status=False)
    ) == [["/b_free", 0], ["/b_free", 1], ["/b_free", 2]]
