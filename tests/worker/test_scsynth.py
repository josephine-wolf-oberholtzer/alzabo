import json
from pathlib import Path

import pytest
from botocore.exceptions import ClientError
from mypy_boto3_s3.client import S3Client
from pymilvus import Collection

from alzabo.config import config
from alzabo.constants import (
    AUDIO_FILENAME,
    SCSYNTH_ANALYSIS_RAW_FILENAME,
    SCSYNTH_ENTRIES_FILENAME,
)
from alzabo.worker import audio, milvus, scsynth


def test_analyze_via_scsynth(
    job_id: str,
    recordings_path: Path,
    s3_client: S3Client,
    staging_id: str,
    tmp_path: Path,
) -> None:
    s3_client.upload_file(
        Bucket=config.s3.uploads_bucket,
        Filename=str(recordings_path / "ibn-arabi-44100.wav"),
        Key=staging_id,
    )
    expected_digest = "dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969"
    key = f"{expected_digest[:2]}/{expected_digest}/{SCSYNTH_ANALYSIS_RAW_FILENAME}"
    with pytest.raises(ClientError):
        s3_client.head_object(Bucket=config.s3.data_bucket, Key=key)
    _, actual_digest = audio.transcode_and_hash_audio.delay([job_id, staging_id]).get(
        timeout=60
    )
    assert scsynth.analyze_via_scsynth.delay([job_id, actual_digest]).get(
        timeout=60
    ) == (job_id, expected_digest)
    s3_client.head_object(Bucket=config.s3.data_bucket, Key=key)
    s3_client.download_file(
        Bucket=config.s3.data_bucket,
        Filename=str(tmp_path / SCSYNTH_ANALYSIS_RAW_FILENAME),
        Key=key,
    )
    analysis = json.loads((tmp_path / SCSYNTH_ANALYSIS_RAW_FILENAME).read_text())
    assert len(analysis) == 7875
    assert analysis[0] == [
        -764.6162109375,
        -764.6162109375,
        69.0,
        0.0,
        0.0,
        0.0,
        0.800000011920929,
        18.21575164794922,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert analysis[-1] == [
        -27.141199111938477,
        -22.028091430664062,
        66.201904296875,
        0.0,
        0.0,
        90.01764678955078,
        0.012303058989346027,
        68.25484466552734,
        0.7673746943473816,
        0.696192741394043,
        0.35394173860549927,
        -0.5621050000190735,
        0.02539624273777008,
        0.25059571862220764,
        0.15206195414066315,
        0.1971874237060547,
        0.359428733587265,
        0.21796418726444244,
        0.01955445110797882,
        0.37760624289512634,
        0.03946411609649658,
        0.21119767427444458,
        0.2638738751411438,
        0.11644460260868073,
        0.26259279251098633,
        0.17650096118450165,
        0.270185649394989,
        0.22566114366054535,
        0.21944472193717957,
        0.19685295224189758,
        0.2785542607307434,
        0.18884457647800446,
        0.24594268202781677,
        0.2137601673603058,
        0.2741971015930176,
        0.24529467523097992,
        0.2736455202102661,
        0.26693791151046753,
        0.23947326838970184,
        0.29601171612739563,
        0.26027923822402954,
        0.2740485966205597,
        0.2647719979286194,
        0.2748633027076721,
        0.2274220734834671,
        0.2940177321434021,
        0.20863233506679535,
        0.26512765884399414,
        0.2044694721698761,
        0.25,
        0.034697987139225006,
        0.01906251162290573,
        0.019729232415556908,
        0.041676826775074005,
        0.12914542853832245,
        0.2031654268503189,
        0.2125513106584549,
        0.12269468605518341,
        0.04393738508224487,
        0.06164771318435669,
        0.05230792984366417,
        0.0593835674226284,
    ]


def test_insert_scsynth_entries(
    job_id: str,
    recordings_path: Path,
    s3_client: S3Client,
    milvus_scsynth_collections: dict[str | None, Collection],
) -> None:
    # prepare pre-conditions
    expected_digest = "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f"
    audio_key = f"{expected_digest[:2]}/{expected_digest}/{AUDIO_FILENAME}"
    s3_client.upload_file(
        Bucket=config.s3.data_bucket,
        Filename=str(recordings_path / f"{expected_digest}.wav"),
        Key=audio_key,
    )
    scsynth.analyze_via_scsynth.delay([job_id, expected_digest]).get(timeout=60)
    scsynth.partition_scsynth_analysis.delay([job_id, expected_digest]).get(timeout=60)
    # run the task
    assert scsynth.insert_scsynth_entries.delay([job_id, expected_digest]).get(
        timeout=60
    ) == (job_id, expected_digest)
    # verify the post-conditions (must flush milvus to make data available!)
    milvus.flush_milvus.delay().get(timeout=60)
    actual = milvus_scsynth_collections[None].query(
        expr=f'digest == "{expected_digest}"',
        output_fields=["id", "start_frame", "frame_count", "f0", "rms"],
    )
    expected = [
        {
            "f0": 62.994198,
            "frame_count": 120320,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-0-120320",
            "rms": -34.162376,
            "start_frame": 0,
        },
        {
            "f0": 63.671158,
            "frame_count": 24064,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-0-24064",
            "rms": -90.969696,
            "start_frame": 0,
        },
        {
            "f0": 64.01272,
            "frame_count": 60416,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-0-60416",
            "rms": -44.90119,
            "start_frame": 0,
        },
        {
            "f0": 66.17291,
            "frame_count": 24064,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-120320-24064",
            "rms": -22.182302,
            "start_frame": 120320,
        },
        {
            "f0": 64.40706,
            "frame_count": 60416,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-120320-60416",
            "rms": -27.188238,
            "start_frame": 120320,
        },
        {
            "f0": -1.0,
            "frame_count": 24064,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-144384-24064",
            "rms": -35.00799,
            "start_frame": 144384,
        },
        {
            "f0": 64.521164,
            "frame_count": 60416,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-144384-60416",
            "rms": -25.302572,
            "start_frame": 144384,
        },
        {
            "f0": 64.276665,
            "frame_count": 24064,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-168448-24064",
            "rms": -18.378633,
            "start_frame": 168448,
        },
        {
            "f0": 66.27423,
            "frame_count": 60416,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-168448-60416",
            "rms": -22.825304,
            "start_frame": 168448,
        },
        {
            "f0": -1.0,
            "frame_count": 24064,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-192512-24064",
            "rms": -28.600578,
            "start_frame": 192512,
        },
        {
            "f0": 63.42105,
            "frame_count": 120320,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-24064-120320",
            "rms": -20.4049,
            "start_frame": 24064,
        },
        {
            "f0": 66.18439,
            "frame_count": 24064,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-24064-24064",
            "rms": -15.04259,
            "start_frame": 24064,
        },
        {
            "f0": 62.376534,
            "frame_count": 60416,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-24064-60416",
            "rms": -17.50328,
            "start_frame": 24064,
        },
        {
            "f0": 62.824722,
            "frame_count": 120320,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-48128-120320",
            "rms": -24.39798,
            "start_frame": 48128,
        },
        {
            "f0": 56.935516,
            "frame_count": 24064,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-48128-24064",
            "rms": -23.150211,
            "start_frame": 48128,
        },
        {
            "f0": 61.026978,
            "frame_count": 60416,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-48128-60416",
            "rms": -19.251835,
            "start_frame": 48128,
        },
        {
            "f0": 64.28271,
            "frame_count": 120320,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-72192-120320",
            "rms": -23.443665,
            "start_frame": 72192,
        },
        {
            "f0": 60.571423,
            "frame_count": 24064,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-72192-24064",
            "rms": -15.876956,
            "start_frame": 72192,
        },
        {
            "f0": 63.47378,
            "frame_count": 60416,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-72192-60416",
            "rms": -20.835308,
            "start_frame": 72192,
        },
        {
            "f0": 65.66328,
            "frame_count": 120320,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-96256-120320",
            "rms": -25.98839,
            "start_frame": 96256,
        },
        {
            "f0": 66.613556,
            "frame_count": 24064,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-96256-24064",
            "rms": -25.77244,
            "start_frame": 96256,
        },
        {
            "f0": 66.13195,
            "frame_count": 60416,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-96256-60416",
            "rms": -23.131735,
            "start_frame": 96256,
        },
    ]
    assert len(actual) == len(expected)
    for i in range(len(actual)):
        for key, value in actual[i].items():
            expected_value = expected[i][key]
            if isinstance(expected_value, float):
                expected_value = round(expected_value, 5)
            # Milvus returns numpy.float32 values instead of float
            actual_value = actual[i][key]
            if type(actual_value).__module__ == "numpy":
                actual_value = round(actual_value.item(), 5)
            assert expected_value == actual_value


@pytest.mark.parametrize("hops", [[500], [250, 500]])
@pytest.mark.parametrize("lengths", [[1000, 2000], [750]])
def test_partition_scsynth_analysis(
    job_id: str,
    recordings_path: Path,
    s3_client: S3Client,
    tmp_path: Path,
    hops: list[int],
    lengths: list[int],
) -> None:
    expected_digest = "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f"
    audio_key = f"{expected_digest[:2]}/{expected_digest}/{AUDIO_FILENAME}"
    s3_client.upload_file(
        Bucket=config.s3.data_bucket,
        Filename=str(recordings_path / f"{expected_digest}.wav"),
        Key=audio_key,
    )
    scsynth.analyze_via_scsynth.delay([job_id, expected_digest]).get(timeout=60)
    assert scsynth.partition_scsynth_analysis.delay(
        [job_id, expected_digest], hops=hops, lengths=lengths
    ).get(timeout=60) == (job_id, expected_digest)
    for hop in hops:
        for length in lengths:
            entries_key = f"{expected_digest[:2]}/{expected_digest}/{SCSYNTH_ENTRIES_FILENAME}".format(
                hop=hop, length=length
            )
            entries_path = tmp_path / SCSYNTH_ENTRIES_FILENAME.format(
                hop=hop, length=length
            )
            s3_client.download_file(
                Bucket=config.s3.data_bucket,
                Filename=str(entries_path),
                Key=entries_key,
            )
            data = json.loads(entries_path.read_text())
            print(json.dumps(data, indent=4, sort_keys=True))
            assert data["hop"] == hop
            assert data["length"] == length
            assert data["digest"] == expected_digest
            assert len(data["entries"])


def test_whiten(data: None) -> None:
    scsynth.whiten.delay().get(timeout=1)
