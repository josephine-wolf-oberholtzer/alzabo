import dataclasses
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .application import Application


@dataclasses.dataclass(frozen=True)
class Command:
    async def do(self, app: "Application") -> None:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class BootCommand(Command):
    async def do(self, app: "Application") -> None:
        await app.boot()


@dataclasses.dataclass(frozen=True)
class FireCommand(Command):
    async def do(self, app: "Application") -> None:
        await app.fire()


@dataclasses.dataclass(frozen=True)
class NotifyCommand(Command):
    async def do(self, app: "Application") -> None:
        await app.notify_listeners()


@dataclasses.dataclass(frozen=True)
class PerformanceConfigCommand(Command):
    path: str
    type_: Literal["through", "toggle"]
    value: float

    async def do(self, app: "Application") -> None:
        if self.type_ == "through":
            app.performance_config[self.path] = self.value
        elif self.type_ == "toggle":
            if self.value:
                app.performance_config[self.path] = float(
                    not bool(app.performance_config.get(self.path, False))
                )
        await app.notify_listeners()


@dataclasses.dataclass(frozen=True)
class QuitCommand(Command):
    async def do(self, app: "Application") -> None:
        await app.quit()
