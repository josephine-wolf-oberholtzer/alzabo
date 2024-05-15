import asyncio
import logging

import monome

from ..config import MonomeConfig
from ..core.utils import clamp
from .commands import Command, NotifyCommand, PerformanceConfigCommand
from .listener import Listener

logger = logging.getLogger(__name__)


class MonomeClient(monome.ArcApp, Listener):
    def __init__(self, *, command_queue: asyncio.Queue[Command], config: MonomeConfig):
        super().__init__()
        self.arc_positions: list[float] = [0.0] * 4
        self.background_tasks: set[asyncio.Task] = set()
        self.buffer = monome.ArcBuffer(4)
        self.command_queue = command_queue
        self.config = config
        self.serialosc = monome.SerialOsc()
        self.serialosc.device_added_event.add_handler(self.on_device_added)

    async def notify(self, performance_config: dict[str, float]) -> None:
        for arc_mapping in self.config.arc:
            if arc_mapping["path"] in performance_config:
                self.positions[arc_mapping["ring"]] = (
                    performance_config[arc_mapping["path"]] * 50.0
                )

    def on_arc_delta(self, ring: int, delta: int) -> None:
        self.arc_positions[ring] = clamp(
            self.arc_positions[ring] + (delta / 8), 0.0, 50.0
        )
        logger.info(f"{ring=} {delta=} {self.arc_positions=}")
        self.render_arc()
        for arc_mapping in self.config.arc:
            if arc_mapping["ring"] != ring:
                continue
            self.command_queue.put_nowait(
                PerformanceConfigCommand(
                    path=arc_mapping["path"],
                    type_=arc_mapping.get("type", "through"),
                    value=self.arc_positions[ring] / 50.0,
                )
            )

    def on_arc_disconnect(self):
        logger.info("Arc disconnected.")

    def on_arc_ready(self):
        logger.info("Ready, clearing all rings...")
        self.command_queue.put_nowait(NotifyCommand())
        self.render_arc()

    def on_device_added(self, id, type, port):
        if "arc" not in type:
            logger.info(
                f"ignoring {id} ({type}) as device does not appear to be an arc"
            )
            return
        logger.info(f"connecting to {id} ({type})")
        task = asyncio.create_task(self.arc.connect("127.0.0.1", port))
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    def render_arc(self):
        for ring in range(4):
            self.buffer.ring_all(ring, 0)
            self.buffer.ring_range(ring, 32 - 7, 32 + 7, 3)
            arc_position = self.arc_positions[ring] % 64.0
            minimum = round(arc_position - 1)
            maximum = round(arc_position + 1)
            for i in range(minimum, maximum + 1):
                self.buffer.ring_set(ring, (i + 39) % 64, 15)
        self.buffer.render(self.arc)

    async def setup(self) -> None:
        await self.serialosc.connect()

    async def teardown(self) -> None:
        pass
