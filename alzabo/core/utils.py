import importlib
import math
import time
from contextlib import contextmanager
from hashlib import file_digest
from pathlib import Path
from typing import Type, cast


def clamp(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        raise ValueError(f"Clamp: {maximum} must be greater than {minimum}")
    return max(min(value, maximum), minimum)


def amplitude_to_decibels(amplitude: float) -> float:
    return 20.0 * math.log10(amplitude)


def decibels_to_amplitude(decibels: float) -> float:
    return math.pow(10.0, decibels / 20.0)


def hash_path(path: Path):
    with path.open("rb") as file_pointer:
        return file_digest(file_pointer, "sha256").hexdigest()


def hz_to_midi(frequency: float) -> float:
    return 12.0 * math.log2(frequency / 440.0)


def import_class(import_path: str) -> Type:
    module_path, _, class_name = import_path.rpartition(".")
    module = importlib.import_module(module_path)
    return cast(Type, getattr(module, class_name))


def make_data_key(digest: str) -> str:
    return f"{digest[:2]}/{digest}"


@contextmanager
def timer(logger, message: str):
    def get_time():
        return elapsed_time

    start_time = time.time()
    yield get_time
    elapsed_time = time.time() - start_time
    if logger:
        logger.info(message.format(time=elapsed_time))
