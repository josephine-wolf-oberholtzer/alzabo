import asyncio
import logging
from typing import Iterable, cast
from uuid import UUID, uuid4

from supriya import AsyncServer, Buffer, BusGroup, CalculationRate, Group
from supriya.clocks import AsyncClock, ClockContext
from supriya.osc import OscCallback, OscMessage
from supriya.patterns import Event, NoteEvent, PatternPlayer, Priority, StopEvent

from ..config import config
from ..core.milvus import Entry
from .buffer_manager import BufferManager
from .pattern_factory import PatternFactoryCallback
from .synthdefs import hdverb, limiter

logger = logging.getLogger(__name__)


class Performer:
    """
    Orchestrates playing patterns, allocating buffers, cleanup.
    """

    def __init__(
        self,
        buffer_manager: BufferManager,
        clock: AsyncClock,
        context: AsyncServer,
        channel_count: int = 2,
        polyphony_limit: int = 32,
    ) -> None:
        self.buffer_manager = buffer_manager
        self.channel_count = channel_count
        self.clock = clock
        self.context = context
        self.outer_group = Group(context=self.context, id_=1000)  # Will be re-created
        self.inner_group = Group(context=self.context, id_=1001)  # Will be re-created
        self.reverb_bus_group = BusGroup(
            context=self.context, calculation_rate=CalculationRate.AUDIO, id_=16
        )  # Will be recreated
        self.osc_callbacks: list[OscCallback] = []
        self.pattern_futures: dict[UUID, asyncio.Future] = {}
        self.pattern_players: dict[UUID, PatternPlayer] = {}
        self.polyphony_limit = polyphony_limit

    async def setup(self) -> None:
        logger.info(f"Setting up {type(self).__name__} ...")
        # ... register OSC callback
        self.osc_callbacks.append(
            self.context.osc_protocol.register(
                pattern=("/n_end",), procedure=self.on_n_end_osc_message
            )
        )
        # ... setup group
        self.outer_group = self.context.add_group(parallel=False)
        self.inner_group = self.outer_group.add_group(parallel=True)
        # ... add synthdefs and synths
        with self.context.at():
            self.reverb_bus_group = self.context.add_bus_group(
                calculation_rate="AUDIO", count=self.channel_count
            )
            with self.context.add_synthdefs(hdverb):
                for i in range(self.channel_count):
                    self.outer_group.add_synth(
                        add_action="ADD_TO_TAIL",
                        in_=i + int(self.reverb_bus_group),
                        mix=1.0,
                        out=i + config.scsynth.output_bus,
                        synthdef=hdverb,
                    )
        with self.context.at():
            with self.context.add_synthdefs(limiter):
                for i in range(self.channel_count):
                    self.outer_group.add_synth(
                        add_action="ADD_TO_TAIL",
                        in_=i + config.scsynth.output_bus,
                        out=i + config.scsynth.output_bus,
                        synthdef=limiter,
                    )

    async def teardown(self) -> None:
        logger.info(f"Tearing down {type(self).__name__} ...")
        # ... stop patterns
        for player in self.pattern_players.values():
            player.stop()
        # ... wait for patterns to complete
        await asyncio.gather(*self.pattern_futures.values())
        # ... teardown group
        with self.context.at():
            self.outer_group.free()
        # ... deregister OSC callback
        while self.osc_callbacks:
            self.context.osc_protocol.unregister(self.osc_callbacks.pop())
        # ... free any remaining buffers

    def check_polyphony(self) -> bool:
        return len(self.pattern_futures) < self.polyphony_limit

    async def perform(
        self, entries: Iterable[Entry], pattern_factory: PatternFactoryCallback
    ) -> tuple[PatternPlayer, asyncio.Future]:
        logger.info("Performing...")
        for entry in entries:
            logger.info(f"Entry: {entry}")
        # generate a UUID to identify the pattern
        uuid = uuid4()
        # allocate / retrieve the buffers
        buffers = await self.buffer_manager.acquire(reference=uuid, entries=entries)
        # generate the pattern via the factory, using the buffer list as args
        pattern = pattern_factory(
            buffers=buffers, reverb_bus_group=self.reverb_bus_group
        )
        # play it
        self.pattern_players[uuid] = player = pattern.play(
            callback=self.on_pattern_player_callback,
            clock=self.clock,
            context=self.context,
            target_bus=config.scsynth.output_bus,
            target_node=self.inner_group,
            uuid=uuid,
        )
        # save a pattern future so we can track pattern completion,
        # e.g. for managing graceful shutdown
        self.pattern_futures[uuid] = future = asyncio.get_running_loop().create_future()
        return player, future

    def on_n_end_osc_message(self, osc_message: OscMessage):
        logger.debug(f"/n_end callback: {osc_message}")
        self.buffer_manager.decrement(osc_message.contents[0])

    def on_pattern_player_callback(
        self,
        player: PatternPlayer,
        context: ClockContext,
        event: Event,
        priority: Priority,
    ):
        logger.debug(f"Pattern player callback: {event}")
        if isinstance(event, NoteEvent) and priority == Priority.START:
            # print("NOTE", event.delta, event.duration, event.kwargs)
            # TODO: Make this public / simpler
            node_id = int(player._proxies_by_uuid[event.id_])
            if (buffer_id := event.kwargs.get("buffer_id")) is not None:
                # print("IDS", "NODE", int(node_id), "BUFFER", int(buffer_id))
                self.buffer_manager.increment(
                    reference=node_id, buffer_id=int(cast(Buffer, buffer_id))
                )
        elif isinstance(event, StopEvent):
            self.buffer_manager.decrement(player.uuid)
            self.pattern_players.pop(player.uuid)
            self.pattern_futures.pop(player.uuid).set_result(True)
