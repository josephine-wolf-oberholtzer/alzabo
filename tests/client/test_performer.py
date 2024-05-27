import asyncio
from pathlib import Path
from typing import Sequence

import pytest
from supriya import (
    AsyncServer,
    Buffer,
    BusGroup,
    OscBundle,
    OscMessage,
    Pattern,
    SynthDef,
)
from supriya.patterns import EventPattern, SequencePattern
from uqbar.strings import normalize

from alzabo.client import Performer
from alzabo.client.synthdefs import hdverb, limiter
from alzabo.core.milvus import Entry


@pytest.fixture
def entries() -> list[Entry]:
    return [
        dict(
            digest="dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969",
            start_frame=0,
            frame_count=48000,
            distance=666.0,
        ),
        dict(
            digest="dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969",
            start_frame=512,
            frame_count=48000,
            distance=666.0,
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=5)
async def test_single(
    context: AsyncServer,
    data: None,
    entries: list[Entry],
    performer: Performer,
    synthdef: SynthDef,
    tmp_path: Path,
) -> None:
    def pattern_factory(
        buffers: Sequence[Buffer], reverb_bus_group: BusGroup
    ) -> Pattern:
        return EventPattern(
            buffer_id=SequencePattern(buffers),
            delta=0.25,
            duration=0.0,
            synthdef=synthdef,
        )

    # validate setup
    context.add_synthdefs(synthdef)
    with context.osc_protocol.capture() as transcript:
        await performer.setup()
    assert transcript.filtered(sent=True, received=False, status=False) == [
        OscMessage("/g_new", 1000, 0, 1),
        OscMessage("/p_new", 1001, 0, 1000),
        OscMessage(
            "/d_recv",
            hdverb.compile(),
            OscBundle(
                contents=[
                    OscMessage("/s_new", "hdverb", 1002, 1, 1000, "in_", 16.0),
                    OscMessage(
                        "/s_new", "hdverb", 1003, 1, 1000, "in_", 17.0, "out", 1.0
                    ),
                ]
            ),
        ),
        OscMessage(
            "/d_recv",
            limiter.compile(),
            OscBundle(
                contents=[
                    OscMessage("/s_new", "limiter", 1004, 1, 1000),
                    OscMessage(
                        "/s_new", "limiter", 1005, 1, 1000, "in_", 1.0, "out", 1.0
                    ),
                ]
            ),
        ),
    ]
    # validate performance
    with context.osc_protocol.capture() as transcript:
        player, future = await performer.perform(
            entries=entries, pattern_factory=pattern_factory
        )
        await future
    # TODO: Need easy access to pattern start time to calculate timestamps
    # TODO: Need to group /b_allocRead messages together deterministically
    assert player.initial_seconds is not None
    base_timestamp = player.initial_seconds + context.latency
    assert transcript.filtered(sent=True, received=False, status=False) == [
        OscMessage(
            "/b_allocRead",
            0,
            str(
                tmp_path
                / "dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969-0-48000.wav"
            ),
            0,
            0,
        ),
        OscMessage("/sync", 1),
        OscMessage(
            "/b_allocRead",
            1,
            str(
                tmp_path
                / "dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969-512-48000.wav"
            ),
            0,
            0,
        ),
        OscMessage("/sync", 2),
        OscBundle(
            contents=(OscMessage("/s_new", synthdef.actual_name, 1006, 0, 1001),),
            timestamp=base_timestamp,
        ),
        OscBundle(
            contents=(
                OscMessage(
                    "/s_new", synthdef.actual_name, 1007, 0, 1001, "buffer_id", 1.0
                ),
            ),
            timestamp=base_timestamp + 0.25,
        ),
    ]
    # validate that buffers remain
    assert performer.buffer_manager.reference_to_buffer_ids == {1006: {0}, 1007: {1}}
    # validate server tree
    assert str(await context.query_tree()) == normalize(
        """
        NODE TREE 0 group
            1 group
                1000 group
                    1001 group
                        1007 fd348b76c028d2138722028f6f8ee364
                            buffer_id: 1.0, out: 0.0
                        1006 fd348b76c028d2138722028f6f8ee364
                            buffer_id: 0.0, out: 0.0
                    1002 hdverb
                        decay: 3.5, in_: 16.0, lpf1: 2000.0, lpf2: 6000.0, out: 0.0, predelay: 0.025
                    1003 hdverb
                        decay: 3.5, in_: 17.0, lpf1: 2000.0, lpf2: 6000.0, out: 1.0, predelay: 0.025
                    1004 limiter
                        in_: 0.0, out: 0.0
                    1005 limiter
                        in_: 1.0, out: 1.0
        """
    )
    # sleep until notes are done
    with context.osc_protocol.capture() as transcript:
        await asyncio.sleep(1)
    assert transcript.filtered(sent=True, received=False, status=False) == [
        OscMessage("/b_free", 0),
        OscMessage("/b_free", 1),
    ]
    # validate teardown
    with context.osc_protocol.capture() as transcript:
        await performer.teardown()
    assert transcript.filtered(sent=True, received=False, status=False) == [
        OscMessage("/n_free", 1000)
    ]
