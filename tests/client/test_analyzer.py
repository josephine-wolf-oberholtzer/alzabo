import asyncio
from typing import Any, cast

import pytest
from supriya import AsyncServer, Bus, CalculationRate

from praetor.client import OnlineScsynthAnalyzer


@pytest.fixture
def analyzer(context: AsyncServer) -> OnlineScsynthAnalyzer:
    return OnlineScsynthAnalyzer(
        context=context,
        bus=Bus(
            calculation_rate=CalculationRate.AUDIO,
            context=context,
            id_=context.audio_input_bus_group.id_,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=5)
async def test_analyzer(analyzer: OnlineScsynthAnalyzer, context: AsyncServer):
    # 512 hop / 48kHz SR = 10.667~ms
    # 1000 * 10.667ms = 10.667 seconds to fill up the ring buffer
    # Timing accuracy isn't perfect, so ~1s slop is padded in
    await analyzer.setup()
    await asyncio.sleep(0.1)
    with pytest.raises(ValueError):
        analyzer.emit()
    for sleep_time, sizes in [
        (0.1, [1, 5]),
        (1.1, [5, 10, 100]),
        (5.25, [500]),
        (5.25, [1000]),
    ]:
        await asyncio.sleep(sleep_time)
        for size in sizes:
            aggregate = cast(dict[str, Any], analyzer.emit(size=size))
            for key in aggregate.keys():
                if key in ("r:chroma", "r:mfcc", "w:chroma", "w:mfcc"):
                    assert aggregate[key]
                    assert all(isinstance(x, (float, int)) for x in aggregate[key])
                else:
                    assert isinstance(aggregate[key], (float, int))
            assert len(aggregate["r:chroma"]) == 12
            assert len(aggregate["r:mfcc"]) == 42
            assert len(aggregate["w:chroma"]) == 12
            assert len(aggregate["w:mfcc"]) == 42
            print("MAX_INDEX?", analyzer.max_index, size, sleep_time)
    await analyzer.teardown()
