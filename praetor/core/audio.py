import asyncio
import logging
import subprocess
import wave
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


def get_duration(source_path: Path) -> float:
    with source_path.open("rb") as file_pointer:
        with wave.open(file_pointer) as wave_file:
            return wave_file.getnframes() / wave_file.getframerate()


def get_transcode_command(
    source_path: Path,
    target_path: Path,
    *,
    from_seconds: float | None = None,
    sample_rate: int = 48000,
    to_seconds: float | None = None,
) -> Sequence[str]:
    command = ["ffmpeg", "-i", str(source_path), "-ac", "1", "-ar", str(sample_rate)]
    if from_seconds is not None:
        command.extend(["-ss", f"{from_seconds:.03f}"])
    if to_seconds is not None:
        command.extend(["-to", f"{to_seconds:.03f}"])
    command.append(str(target_path))
    return command


def transcode_audio(
    source_path: Path,
    target_path: Path,
    *,
    from_seconds: float | None = None,
    sample_rate: int = 48000,
    to_seconds: float | None = None,
) -> None:
    completed_process = subprocess.run(
        get_transcode_command(
            source_path,
            target_path,
            from_seconds=from_seconds,
            sample_rate=sample_rate,
            to_seconds=to_seconds,
        ),
        capture_output=True,
        text=True,
    )
    if completed_process.returncode:
        for line in completed_process.stdout.splitlines():
            logger.warning(line)


async def transcode_audio_async(
    source_path: Path, target_path: Path, sample_rate: int = 48000
) -> None:
    process = await asyncio.create_subprocess_exec(
        *get_transcode_command(source_path, target_path, sample_rate=sample_rate),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    stdout, _ = await process.communicate()
    if process.returncode:
        for line in stdout.decode().splitlines():
            logger.warning(line)
