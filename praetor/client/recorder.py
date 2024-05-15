from pathlib import Path
from typing import Callable, Sequence

from supriya import AsyncClock, AsyncServer, Buffer, Node, SynthDef, SynthDefBuilder
from supriya.osc import OscCallback
from supriya.ugens import DiskOut, In


class Recorder:
    def __init__(
        self,
        context: AsyncServer,
        clock: AsyncClock,
        triples: Sequence[tuple[int, int, Callable[[Path], None]]],
    ) -> None:
        self.context = context
        self.clock = clock
        self.triples = triples
        self.buffer: Buffer | None = None
        self.node: Node | None = None
        self.osc_callbacks: list[OscCallback] = []
        self.buffer_ids_to_paths_and_callbacks: dict[
            int, tuple[Path, Callable[[Path], None]]
        ] = {}
        self.synthdef = self.build_synthdef()

    def build_synthdef(self) -> SynthDef:
        with SynthDefBuilder(buffer_id=0, bus=0) as builder:
            DiskOut.ar(source=In.ar(bus=builder["bus"]))
        return builder.build()

    async def setup(self) -> None:
        # ... clock callback
        # ... setup synthdef
        for duration_ms, hop_ms, callback in self.triples:
            # self.clock.cue()
            pass

    async def teardown(self) -> None:
        # ... rm clock callbacks
        # ... rm node, buffer, osc callbacks
        pass

    async def clock_callback(self):
        with self.context.at():
            if self.node:
                with self.buffer.close():
                    self.node.free()
                    self.buffer.free()
            self.buffer = self.context.add_buffer(channel_count=1)
            with self.buffer:
                with self.buffer.write(leave_open=True):
                    self.node = self.context.add_node()

    async def n_end_callback(self):
        pass
