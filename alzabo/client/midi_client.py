import asyncio
import logging
from functools import partial

from rtmidi import MidiIn, SystemError
from rtmidi.midiutil import list_input_ports

from ..config import MidiConfig, MidiMapping
from .commands import Command, PerformanceConfigCommand
from .listener import Listener

logger = logging.getLogger(__name__)


class MidiClient(Listener):
    """
    A MIDI client.
    """

    def __init__(
        self, *, command_queue: asyncio.Queue[Command], config: MidiConfig | None = None
    ):
        self.command_queue = command_queue
        self.config = config
        self.midi_inputs: list[MidiIn] = []

    def __call__(self, device_config: list[MidiMapping], event, data=None):
        (_, note, value), _ = event
        for midi_mapping in device_config:
            if midi_mapping["note"] != note:
                continue
            self.command_queue.put_nowait(
                PerformanceConfigCommand(
                    path=midi_mapping["path"],
                    type_=midi_mapping.get("type", "through"),
                    value=value / 127.0,
                )
            )

    async def notify(self, performance_config: dict[str, float]) -> None:
        pass

    async def setup(self) -> None:
        logger.info(f"Setting up {type(self).__name__} ...")
        ports: list[str] = []
        try:
            ports = list_input_ports()
        except SystemError as e:
            logger.warning(str(e))
        for device_name, device_config in getattr(self.config, "devices", {}).items():
            try:
                index = ports.index(device_name)
            except ValueError:
                logger.warning(f'MIDI port for "{device_name}" not found')
                continue
            self.midi_inputs.append(midiin := MidiIn())
            midiin.open_port(index)
            midiin.set_callback(partial(self.__call__, device_config))

    async def teardown(self) -> None:
        logger.info(f"Tearing down {type(self).__name__} ...")
        while self.midi_inputs:
            self.midi_inputs.pop().close_port()
