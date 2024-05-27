import asyncio
import logging
import random
import tempfile
import traceback
from pathlib import Path
from typing import Protocol, Type, TypedDict, cast

from supriya import AsyncClock, AsyncServer, Bus, CalculationRate
from supriya.patterns import PatternPlayer

from ..config import config
from ..core.scsynth import aggregate_to_vector
from ..core.utils import import_class
from .analyzer import OnlineScsynthAnalyzer
from .api_client import APIClient
from .buffer_manager import BufferManager
from .commands import BootCommand, Command, FireCommand
from .listener import Listener
from .midi_client import MidiClient
from .monome_client import MonomeClient
from .pattern_factory import PatternFactory
from .performer import Performer

logger = logging.getLogger(__name__)


class SetupTeardown(Protocol):
    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass


class PerformanceConfig(TypedDict):
    density: float
    history: float
    index: float
    reverb: float
    stretch: float


class Application:
    """
    The Alzabo Client application.

    Harnesses together all application logic.
    """

    def __init__(self) -> None:
        # AsyncIO
        self.background_tasks: set[asyncio.Task] = set()
        self.command_queue: asyncio.Queue[Command] = asyncio.Queue()
        self.boot_future: asyncio.Future | None = None
        self.quit_future: asyncio.Future | None = None
        self.periodic_tasks: list[asyncio.Task] = []
        # Supriya
        self.context = AsyncServer()
        self.clock = AsyncClock()
        self.clock.change(beats_per_minute=60 * 4)
        # Alzabo non-pluggable classes
        self.api_client = APIClient(
            api_url=str(config.api.url), api_key=config.api.key or None
        )
        self.buffer_manager = BufferManager(
            api_client=self.api_client,
            context=self.context,
            download_directory=Path(tempfile.mkdtemp()),
        )
        self.performer = Performer(
            buffer_manager=self.buffer_manager, clock=self.clock, context=self.context
        )
        # Alzabo pluggable classes
        self.analyzer: OnlineScsynthAnalyzer = cast(
            Type[OnlineScsynthAnalyzer], import_class(config.application.analyzer_class)
        )(
            context=self.context,
            bus=Bus(
                calculation_rate=CalculationRate.AUDIO,
                context=self.context,
                id_=config.scsynth.input_bus,
            ),
        )
        self.midi_client: MidiClient = cast(
            Type[MidiClient], import_class(config.midi.client_class)
        )(command_queue=self.command_queue, config=config.midi)
        self.monome_client: MonomeClient = cast(
            Type[MonomeClient], import_class(config.monome.client_class)
        )(command_queue=self.command_queue, config=config.monome)
        self.pattern_factory: PatternFactory = cast(
            Type[PatternFactory], import_class(config.application.pattern_factory_class)
        )(context=self.context)
        # Alzabo state
        self.performance_config: dict[str, float] = dict(
            density=0.5, history=0.05, index=0.0, reverb=0.05, stretch=0.5
        )
        self.teardown_stack: list[SetupTeardown] = []
        self.listeners: set[Listener] = {self.midi_client, self.monome_client}

    # ### LIFECYCLE ###

    async def boot(self) -> None:
        """
        Boot the application.
        """
        logger.info("Booting...")
        if self.boot_future is None:
            raise ValueError
        loop = asyncio.get_running_loop()
        # Start the clock
        await self.clock.start()
        # Boot the context
        logger.info("Booting context ...")
        await self.context.boot(
            block_size=config.scsynth.block_size,
            executable=config.scsynth.executable,
            input_bus_channel_count=config.scsynth.input_count,
            input_device=config.scsynth.input_device,
            memory_size=config.scsynth.memory_size,
            output_bus_channel_count=config.scsynth.output_count,
            output_device=config.scsynth.output_device,
            # ugen_plugins_path="/usr/local/lib/SuperCollider/plugins:/usr/lib/SuperCollider/Extensions",
        )
        # Setup each submodule
        to_setup: list[SetupTeardown] = [
            self.analyzer,
            self.pattern_factory,
            self.performer,
        ]
        if self.midi_client:
            to_setup.append(self.midi_client)
        if self.monome_client:
            to_setup.append(self.monome_client)
        for x in to_setup:
            try:
                await x.setup()
                self.teardown_stack.append(x)
            except Exception:
                logger.warning(f"Something broke while setting up {x}")
                traceback.print_exc()
                await self.quit()
                return
        # Start periodic tasks
        self.periodic_tasks.append(loop.create_task(self.fire_periodic()))
        # Sync server
        await self.context.sync()
        # Done
        self.boot_future.set_result(True)
        logger.info("... booted!")

    async def fire(self) -> tuple[PatternPlayer, asyncio.Future] | None:
        """
        Query the database and fire off a pattern.
        """
        logger.info("Firing...")
        # ... check polyphony
        if not self.performer.check_polyphony():
            logger.warning("... polyphony maxed out!")
            return None
        # ... emit aggregate
        try:
            aggregate = self.analyzer.emit(
                size=int(1000 * self.performance_config["history"])
            )
        except ValueError:
            logger.warning("... analysis not primed!")
            return None
        # ... get index alias
        index_aliases = [
            index_config["alias"] for index_config in config.analysis.scsynth_indices
        ]
        index_alias = index_aliases[
            min(
                int(self.performance_config.get("index", 0) * len(index_aliases)),
                len(index_aliases) - 1,
            )
        ]
        # ... get pattern factory
        pattern_factory = self.pattern_factory.emit(**self.performance_config)
        # ... query milvus
        vector = aggregate_to_vector(aggregate, index_alias=index_alias)
        logger.info(f"{index_alias=} {vector=}")
        if not (
            entries := (
                await self.api_client.query_scsynth(
                    index=index_alias, vector=vector, voiced=aggregate["is_voiced"]
                )
            )["entries"]
        ):
            logger.warning("... no entries queried!")
            return None
        # ... talk to performer
        return await self.performer.perform(
            entries=entries, pattern_factory=pattern_factory
        )

    async def fire_periodic(self):
        """
        Fire off patterns periodically.
        """
        # TODO: Make this pluggable
        while True:
            if random.random() < self.performance_config["density"]:
                await self.command_queue.put(FireCommand())
            await asyncio.sleep(0.1)

    async def notify_listeners(self):
        for listener in self.listeners:
            await listener.notify()

    async def quit(self) -> None:
        """
        Quit the application.
        """
        logger.info("Quitting ...")
        if self.quit_future is None:
            raise ValueError
        # Stop periodic tasks
        while self.periodic_tasks:
            self.periodic_tasks.pop().cancel()
        # Teardown each submodule in reverse order
        while self.teardown_stack:
            await self.teardown_stack.pop().teardown()
        # Quit the context
        logger.info("Quitting context ...")
        await self.context.quit()
        # Stop the clock
        await self.clock.stop()
        # Done
        self.quit_future.set_result(True)
        logger.info("... quit!")

    async def run(self) -> None:
        """
        Create and run an application.

        This is implimented as an async classmethod to guarantee a loop is
        available when the application is instantiated, allowing us to create
        futures bound to that loop for tracking boot/quit status.
        """
        loop = asyncio.get_running_loop()
        self.quit_future = loop.create_future()
        self.boot_future = loop.create_future()
        await self.command_queue.put(BootCommand())
        while not self.quit_future.done():
            # We need a mechanism to breakout of the loop otherwise we risk
            # blocking on the next item while attempting to quit which
            # effectively blocks us forever. Waiting with a timeout solves this.
            try:
                command = await asyncio.wait_for(self.command_queue.get(), 0.1)
            except asyncio.TimeoutError:
                continue
            logger.info(f"Processing command: {command}")
            task = asyncio.create_task(command.do(self))
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
