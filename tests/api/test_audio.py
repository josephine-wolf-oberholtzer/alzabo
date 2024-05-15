import uuid
import wave
from io import BytesIO
from unittest import mock

import aiohttp
import pytest

from praetor.config import config
from praetor.constants import AUDIO_FILENAME


@pytest.mark.asyncio
async def test_batch(api_client, mocker, recordings_path, s3_client):
    urls = []
    for filename in [
        "ibn-arabi-44100.wav",
        "nabokov-22050.aiff",
        "vandermeer-24000.mp3",
    ]:
        s3_client.upload_file(
            Bucket="test-source", Key=filename, Filename=recordings_path / filename
        )
        urls.append(f"s3://test-source/{filename}")
    uuids = [uuid.uuid4() for _ in range(len(urls))]
    mocker.patch("praetor.api.audio.uuid4", side_effect=uuids)
    mock_task = mocker.patch("praetor.worker.tasks.get_audio_processing_chain")
    response = await api_client.post("/audio/batch", json=dict(urls=urls))
    assert response.status == 200
    assert await response.json() == {"jobs": [str(x) for x in uuids]}
    assert mock_task.mock_calls == [
        mock.call(str(uuids[0]), urls[0]),
        mock.call()(),
        mock.call(str(uuids[1]), urls[1]),
        mock.call()(),
        mock.call(str(uuids[2]), urls[2]),
        mock.call()(),
    ]


@pytest.mark.parametrize(
    "start_frame, frame_count, expected_frame_count, expected_file_size",
    [(None, None, 4032000, 8064078), (32, 64, 64, 172), (None, 128, 128, 300)],
)
@pytest.mark.asyncio
async def test_fetch(
    api_client,
    expected_file_size,
    expected_frame_count,
    frame_count,
    recordings_path,
    s3_client,
    start_frame,
):
    digest = "dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969"
    key = f"{digest[:2]}/{digest}/{AUDIO_FILENAME}"
    s3_client.upload_file(
        Bucket=config.s3.data_bucket,
        Filename=str(recordings_path / f"{digest}.wav"),
        Key=key,
    )
    params = {}
    if start_frame is not None:
        params["start"] = str(start_frame)
    if frame_count is not None:
        params["count"] = str(frame_count)
    response = await api_client.get(f"/audio/fetch/{digest}", params=params)
    assert response.status == 200
    data = await response.read()
    buffer_ = BytesIO(data)
    buffer_.seek(0)
    with wave.open(buffer_, "r") as wave_file:
        assert wave_file.getframerate() == 48000
        assert wave_file.getnchannels() == 1
        assert wave_file.getsampwidth() == 2
        assert wave_file.getnframes() == expected_frame_count
    actual_file_size = len(data)
    assert actual_file_size == expected_file_size


@pytest.mark.parametrize(
    "filename",
    [
        "ibn-arabi-44100-300s.wav",
        "ibn-arabi-44100-180s.wav",
        "ibn-arabi-44100-5s.wav",
        "ibn-arabi-44100-1s.wav",
        "nabokov-22050.aiff",
        "vandermeer-24000.mp3",
    ],
)
@pytest.mark.asyncio
async def test_upload(api_client, filename, mocker, recordings_path, s3_client):
    uuids = [uuid.uuid4() for _ in range(2)]
    mocker.patch("praetor.api.audio.uuid4", side_effect=uuids)
    mock_task = mocker.patch("praetor.worker.tasks.get_audio_processing_chain")
    data = aiohttp.FormData()
    data.add_field("file", (recordings_path / filename).open("rb"))
    response = await api_client.post("/audio/upload", data=data)
    assert response.status == 200
    assert mock_task.mock_calls == [
        mock.call(str(uuids[0]), f"s3://{config.s3.uploads_bucket}/{uuids[1]}"),
        mock.call()(),
    ]
    s3_client.head_object(Bucket=config.s3.uploads_bucket, Key=str(uuids[1]))
