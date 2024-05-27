import asyncio

import pytest
from pytest_mock import MockerFixture

from alzabo.client.commands import Command, NotifyCommand, PerformanceConfigCommand
from alzabo.client.monome_client import MonomeClient
from alzabo.config import ArcMapping, MonomeConfig


@pytest.mark.asyncio
async def test_MonomeClient(mocker: MockerFixture) -> None:
    mocker.patch("monome.ArcBuffer.render")
    command_queue: asyncio.Queue[Command] = asyncio.Queue()
    config = MonomeConfig(
        arc=[
            ArcMapping(path="foo", ring=0),
            ArcMapping(path="bar", ring=1),
            ArcMapping(path="baz", ring=2),
            ArcMapping(path="quux", ring=3),
        ]
    )
    monome_client = MonomeClient(command_queue=command_queue, config=config)
    await monome_client.setup()
    monome_client.on_arc_ready()
    monome_client.on_arc_delta(0, 16)
    monome_client.on_arc_delta(0, -24)
    monome_client.on_arc_delta(0, 32)
    monome_client.on_arc_disconnect()
    await monome_client.teardown()
    actual_commands: list[Command] = []
    while not command_queue.empty():
        actual_commands.append(await command_queue.get())
    assert actual_commands == [
        NotifyCommand(),
        PerformanceConfigCommand(path="foo", type_="through", value=0.04),
        PerformanceConfigCommand(path="foo", type_="through", value=0.0),
        PerformanceConfigCommand(path="foo", type_="through", value=0.08),
    ]
