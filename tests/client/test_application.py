import asyncio

import pytest
from supriya.contexts.realtime import BootStatus
from uqbar.strings import normalize

from praetor.client.application import Application
from praetor.client.commands import FireCommand, QuitCommand


@pytest.mark.asyncio
async def test_application_atomic(application: Application, data: None) -> None:
    event_loop = asyncio.get_running_loop()
    application.quit_future = event_loop.create_future()
    application.boot_future = event_loop.create_future()
    await application.boot()
    assert application.clock.is_running
    assert application.context.boot_status == BootStatus.ONLINE
    await application.context.sync()
    assert str(await application.context.query_tree()) == normalize(
        """
        NODE TREE 0 group
            2 analysis
                in_: 8.0
            1 group
                1000 group
                    1001 group
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
    await asyncio.sleep(1.0)
    await application.fire()
    await asyncio.sleep(1.0)
    await application.quit()


@pytest.mark.asyncio
async def test_application_run(application: Application, data: None) -> None:
    event_loop = asyncio.get_running_loop()
    # Create and run the application
    application = Application()
    task = event_loop.create_task(application.run())
    await asyncio.sleep(0)
    # Wait for the application to finish booting
    assert application.boot_future is not None
    assert application.quit_future is not None
    await application.boot_future
    # Validate the node tree
    actual_tree = normalize(str(await application.context.query_tree()))
    expected_tree = normalize(
        """
        NODE TREE 0 group
            2 analysis
                in_: 8.0
            1 group
                1000 group
                    1001 group
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
    assert actual_tree == expected_tree
    # Fire a pattern manually
    while (result := await application.fire()) is None:
        await asyncio.sleep(0.1)
    _, pattern_future = result
    await pattern_future
    # Fire a pattern via command
    await application.command_queue.put(FireCommand())
    await asyncio.sleep(1.0)
    # Wait for the application to quit
    await application.command_queue.put(QuitCommand())
    await application.quit_future
    # Wait for the application's run task to complete
    await task
