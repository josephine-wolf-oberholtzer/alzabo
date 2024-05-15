import logging

import numpy
from supriya.contexts import BaseServer, Bus, Node
from supriya.osc import OscCallback, OscMessage

from ..config import config
from ..constants import SCSYNTH_ANALYSIS_SIZE
from ..core.scsynth import Aggregate, aggregate, build_online_analysis_synthdef

logger = logging.getLogger(__name__)


class OnlineScsynthAnalyzer:
    """
    An online analyzer, populating a ~10s circular buffer.
    """

    def __init__(self, context: BaseServer, bus: Bus) -> None:
        self.context = context
        self.bus = bus
        self.nodes: list[Node] = []
        self.osc_callbacks: list[OscCallback] = []
        self.index = 0
        self.max_index = 0
        self.array: numpy.ndarray = numpy.zeros(
            (1000, SCSYNTH_ANALYSIS_SIZE)
        )  # 512 hop / 48kHz * 1000 = ~10s

    async def setup(self) -> None:
        logger.info(f"Setting up {type(self).__name__} ...")
        synthdef = build_online_analysis_synthdef(executable=config.scsynth.executable)
        self.osc_callbacks.append(
            self.context.osc_protocol.register(
                pattern=("/analysis",), procedure=self.update
            )
        )
        with self.context.at():
            with self.context.add_synthdefs(synthdef):
                self.nodes.append(
                    self.context.add_synth(
                        add_action="ADD_TO_HEAD",
                        in_=self.bus,
                        permanent=True,
                        synthdef=synthdef,
                        target_node=self.context.root_node,
                    )
                )

    async def teardown(self) -> None:
        logger.info(f"Tearing down {type(self).__name__} ...")
        with self.context.at():
            while self.nodes:
                self.nodes.pop().free()
            while self.osc_callbacks:
                self.context.osc_protocol.unregister(self.osc_callbacks.pop())

    def emit(self, size=1000) -> Aggregate:
        if size > self.array.shape[0]:
            raise ValueError(size, self.array.shape[0])
        if size > self.max_index:
            raise ValueError(size, self.max_index)
        if (self.index - size) < 0:
            difference = size - self.index
            array = numpy.concatenate(
                [self.array[self.index - size : self.index], self.array[-difference:]]
            )
        else:
            array = self.array[self.index - size : self.index]
        aggregated = aggregate(array)
        logger.info(f"Emitting: {aggregated}")
        return aggregated

    def update(self, osc_message: OscMessage) -> None:
        self.array[self.index] = osc_message.contents[2:]
        self.index = (self.index + 1) % self.array.shape[0]
        self.max_index += 1
