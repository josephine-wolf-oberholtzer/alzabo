import asyncio

import pytest
from pytest_mock import MockerFixture

from alzabo.client.commands import Command, PerformanceConfigCommand
from alzabo.client.midi_client import MidiClient
from alzabo.config import MidiConfig, MidiMapping


@pytest.mark.asyncio
async def test_MidiClient(mocker: MockerFixture) -> None:
    mocker.patch("alzabo.client.midi_client.MidiIn")
    mocker.patch(
        "alzabo.client.midi_client.list_input_ports",
        return_value=["Mock Faderboard", "Mock Knobboard", "Mock Keyboard"],
    )
    command_queue: asyncio.Queue[Command] = asyncio.Queue()
    keyboard_config: list[MidiMapping] = [
        dict(path="a", note=64),
        dict(path="b", note=65),
    ]
    knobboard_config: list[MidiMapping] = [
        dict(path="c", note=64),
        dict(path="d", note=65, type="toggle"),
        dict(path="e", note=66),
    ]
    config = MidiConfig(
        devices={"Mock Keyboard": keyboard_config, "Mock Knobboard": knobboard_config},
        enabled=True,
    )
    midi_client = MidiClient(command_queue=command_queue, config=config)
    await midi_client.setup()
    midi_client(keyboard_config, ((None, 64, 72), 0.0))
    midi_client(keyboard_config, ((None, 65, 72), 0.0))
    midi_client(keyboard_config, ((None, 66, 72), 0.0))
    midi_client(knobboard_config, ((None, 64, 72), 0.0))
    midi_client(knobboard_config, ((None, 65, 72), 0.0))
    midi_client(knobboard_config, ((None, 66, 72), 0.0))
    actual_commands: list[Command] = []
    while not command_queue.empty():
        actual_commands.append(await command_queue.get())
    assert actual_commands == [
        PerformanceConfigCommand(path="a", type_="through", value=0.5669291338582677),
        PerformanceConfigCommand(path="b", type_="through", value=0.5669291338582677),
        PerformanceConfigCommand(path="c", type_="through", value=0.5669291338582677),
        PerformanceConfigCommand(path="d", type_="toggle", value=0.5669291338582677),
        PerformanceConfigCommand(path="e", type_="through", value=0.5669291338582677),
    ]
