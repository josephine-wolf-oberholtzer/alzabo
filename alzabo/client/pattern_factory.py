import logging
import math
from typing import Protocol, Sequence

from supriya import AsyncServer, Buffer, BusGroup, Pattern, SynthDef
from supriya.patterns import (
    BusPattern,
    EventPattern,
    FxPattern,
    RandomPattern,
    SequencePattern,
)

from ..core.utils import amplitude_to_decibels
from .synthdefs import build_aux_send, build_basic_playback, build_warp_playback

logger = logging.getLogger(__name__)


class PatternFactoryCallback(Protocol):
    def __call__(
        self, buffers: Sequence[Buffer], reverb_bus_group: BusGroup
    ) -> Pattern:
        pass


class PatternFactory:
    """
    Emits patterns, allocates performance SynthDefs.
    """

    def __init__(self, context: AsyncServer, channel_count: int = 2):
        self.channel_count = channel_count
        self.context = context
        self.synthdefs: dict[str, SynthDef] = {
            "aux-send": build_aux_send(channel_count),
            "basic-playback": build_basic_playback(channel_count),
            "warp-playback": build_warp_playback(channel_count),
        }

    async def setup(self) -> None:
        logger.info(f"Setting up {type(self).__name__} ...")
        # ... allocate synthdefs
        for synthdef in self.synthdefs.values():
            self.context.add_synthdefs(synthdef)

    async def teardown(self) -> None:
        logger.info(f"Tearing down {type(self).__name__} ...")

    def emit(self, polyphony_limit: int, **kwargs: float) -> PatternFactoryCallback:
        return self.emit_warp(polyphony_limit=polyphony_limit, **kwargs)

    def emit_basic(
        self, polyphony_limit: int, **kwargs: float
    ) -> PatternFactoryCallback:
        def basic_pattern_factory(
            buffers: Sequence[Buffer], reverb_bus_group: BusGroup
        ) -> Pattern:
            return BusPattern(
                channel_count=self.channel_count,
                pattern=FxPattern(
                    pattern=EventPattern(
                        amplitude=0.5,
                        buffer_id=SequencePattern(buffers),
                        delta=RandomPattern(0.25, 0.5),
                        duration=0.0,  # < 0 duration means no note off
                        gain=amplitude_to_decibels(1.0 / math.sqrt(polyphony_limit)),
                        panning=RandomPattern(-1.0, 1.0),
                        synthdef=self.synthdefs["basic-playback"],
                    ),
                    aux_out=reverb_bus_group,
                    mix=kwargs["reverb"] ** 2.0,
                    release_time=5.0,
                    synthdef=self.synthdefs["aux-send"],
                ),
            )

        return basic_pattern_factory

    def emit_warp(
        self, polyphony_limit: int, **kwargs: float
    ) -> PatternFactoryCallback:
        def warp_pattern_factory(
            buffers: Sequence[Buffer], reverb_bus_group: BusGroup
        ) -> Pattern:
            return BusPattern(
                channel_count=self.channel_count,
                pattern=FxPattern(
                    pattern=EventPattern(
                        amplitude=0.5,
                        buffer_id=SequencePattern(buffers),
                        delta=RandomPattern(0.25, 0.5),
                        duration=0.0,  # < 0 duration means no note off
                        gain=amplitude_to_decibels(1.0 / math.sqrt(polyphony_limit)),
                        overlaps=8,
                        panning=RandomPattern(-1.0, 1.0),
                        synthdef=self.synthdefs["warp-playback"],
                        time_scaling=4.0 ** ((kwargs["stretch"] * 2.0) - 1.0),
                    ),
                    aux_out=reverb_bus_group,
                    mix=kwargs["reverb"] ** 2.0,
                    release_time=5.0,
                    synthdef=self.synthdefs["aux-send"],
                ),
            )

        return warp_pattern_factory
