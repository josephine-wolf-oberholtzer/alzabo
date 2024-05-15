import wave

import pytest
from botocore.exceptions import ClientError

from praetor.config import config
from praetor.constants import AUDIO_FILENAME
from praetor.worker import audio


@pytest.mark.parametrize(
    "source_filename, expected_digest, expected_frame_count",
    [
        (
            "ibn-arabi-44100.wav",
            "dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969",
            4032000,
        ),
        (
            "ibn-arabi-44100-1s.wav",
            "cae67026988403cf60e85089188cbfa9ed44860b35e2d7f857764f3ec433fbb9",
            48000,
        ),
        (
            "ibn-arabi-44100-5s.wav",
            "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f",
            240000,
        ),
    ],
)
def test_transcode_and_hash_audio(
    job_id,
    recordings_path,
    s3_client,
    staging_id,
    tmp_path,
    source_filename,
    expected_digest,
    expected_frame_count,
):
    s3_client.upload_file(
        Bucket=config.s3.uploads_bucket,
        Filename=recordings_path / source_filename,
        Key=staging_id,
    )
    key = f"{expected_digest[:2]}/{expected_digest}/{AUDIO_FILENAME}"
    # Transcoded audio does not yet exist in data bucket
    with pytest.raises(ClientError):
        s3_client.head_object(Bucket=config.s3.data_bucket, Key=key)
    _, actual_digest = audio.transcode_and_hash_audio.delay([job_id, staging_id]).get(
        timeout=60
    )
    # Verify that transcoded audio now exists
    assert actual_digest == expected_digest
    s3_client.head_object(Bucket=config.s3.data_bucket, Key=key)
    s3_client.download_file(
        Bucket=config.s3.data_bucket, Filename=tmp_path / AUDIO_FILENAME, Key=key
    )
    with (tmp_path / AUDIO_FILENAME).open("rb") as file:
        with wave.open(file) as wave_file:
            assert wave_file.getframerate() == 48000
            assert wave_file.getnchannels() == 1
            assert wave_file.getnframes() == expected_frame_count
            assert wave_file.getsampwidth() == 2


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
def test_upload_audio_http(
    filename, job_id, recordings_path, requests_mocker, s3_client
):
    """
    Audio can be uploaded from an HTTP(s) URL
    """
    url = "https://fake.ai/foo/bar/baz"
    content = (recordings_path / filename).read_bytes()
    requests_mocker.get(url, content=content)
    _, staging_id = audio.upload_audio.delay([job_id, url]).get(timeout=60)
    object_ = s3_client.get_object(Bucket=config.s3.uploads_bucket, Key=staging_id)
    assert object_["ContentLength"] == len(content)


def test_upload_audio_s3(job_id, recordings_path, s3_client):
    """
    Audio can be uploaded from an S3 URL
    """
    source_key = "ibn-arabi-44100.wav"
    source_bucket = "test-source"
    s3_client.upload_file(
        Filename=recordings_path / source_key, Bucket=source_bucket, Key=source_key
    )
    s3_client.head_object(Bucket=source_bucket, Key=source_key)
    _, staging_id = audio.upload_audio.delay(
        [job_id, f"s3://{source_bucket}/{source_key}"]
    ).get(timeout=60)
    s3_client.head_object(Bucket=config.s3.uploads_bucket, Key=staging_id)
