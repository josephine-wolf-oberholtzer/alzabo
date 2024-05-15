from pathlib import Path

import pytest
import torch
import torch.nn

from praetor.core.ast import (
    analyze,
    extract_features,
    load_labels,
    load_model,
    partition,
)


@pytest.fixture(scope="module")
def labels():
    return load_labels()


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_analyze(model, recordings_path: Path) -> None:
    audio_path = recordings_path / "ibn-arabi-44100-1s.wav"
    analysis = analyze(audio_path, model)
    assert len(analysis) == 527


def test_extract_features(model, recordings_path: Path) -> None:
    audio_path = recordings_path / "ibn-arabi-44100-1s.wav"
    features = extract_features(audio_path)
    assert isinstance(features, torch.Tensor)
    assert features.shape == (1024, 128)


def test_load_labels() -> None:
    labels = load_labels()
    assert len(labels) == 527
    assert all(isinstance(x, str) for x in labels)


def test_load_model() -> None:
    model = load_model()
    assert isinstance(model, torch.nn.DataParallel)


def test_partition(model, recordings_path: Path) -> None:
    audio_path = recordings_path / "ibn-arabi-44100-5s.wav"
    hop = 250
    length = 500
    entries = partition(audio_path, model, hop_ms=hop, length_ms=length)
    assert len(entries) == 19
    assert all(len(x) == 3 for x in entries)
    assert all(len(x[-1]) == 527 for x in entries)
