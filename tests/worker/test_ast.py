import json

import pytest

from praetor.config import config
from praetor.constants import AST_ENTRIES_FILENAME, AUDIO_FILENAME
from praetor.worker import ast, milvus


@pytest.mark.parametrize("hops", [[500], [250, 500]])
@pytest.mark.parametrize("lengths", [[1000, 2000], [750]])
def test_analyze_via_ast(job_id, recordings_path, s3_client, tmp_path, hops, lengths):
    expected_digest = "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f"
    audio_key = f"{expected_digest[:2]}/{expected_digest}/{AUDIO_FILENAME}"
    s3_client.upload_file(
        Bucket=config.s3.data_bucket,
        Filename=recordings_path / f"{expected_digest}.wav",
        Key=audio_key,
    )
    assert ast.analyze_via_ast.delay(
        [job_id, expected_digest], hops=hops, lengths=lengths
    ).get(timeout=120) == (job_id, expected_digest)
    for hop in hops:
        for length in lengths:
            entries_key = f"{expected_digest[:2]}/{expected_digest}/{AST_ENTRIES_FILENAME}".format(
                hop=hop, length=length
            )
            entries_path = tmp_path / AST_ENTRIES_FILENAME.format(
                hop=hop, length=length
            )
            s3_client.download_file(
                Bucket=config.s3.data_bucket, Filename=entries_path, Key=entries_key
            )
            data = json.loads(entries_path.read_text())
            assert data["hop"] == hop
            assert data["length"] == length
            assert data["digest"] == expected_digest
            assert len(data["entries"])
            for entry in data["entries"]:
                assert len(entry) == 3
                assert len(entry[-1]) == 527


def test_insert_ast_entries(data, job_id, milvus_ast_collection):
    # prepare pre-conditions
    expected_digest = "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f"
    # run the task
    assert ast.insert_ast_entries.delay([job_id, expected_digest]).get(timeout=60) == (
        job_id,
        expected_digest,
    )
    # verify the post-conditions (must flush milvus to make data available!)
    milvus.flush_milvus.delay().get(timeout=60)
    actual = milvus_ast_collection.query(
        expr=f'digest == "{expected_digest}"',
        output_fields=["id", "start_frame", "frame_count"],
    )
    expected = [
        {
            "frame_count": 120000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-0-120000",
            "start_frame": 0,
        },
        {
            "frame_count": 24000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-0-24000",
            "start_frame": 0,
        },
        {
            "frame_count": 60000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-0-60000",
            "start_frame": 0,
        },
        {
            "frame_count": 120000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-120000-120000",
            "start_frame": 120000,
        },
        {
            "frame_count": 24000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-120000-24000",
            "start_frame": 120000,
        },
        {
            "frame_count": 60000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-120000-60000",
            "start_frame": 120000,
        },
        {
            "frame_count": 24000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-144000-24000",
            "start_frame": 144000,
        },
        {
            "frame_count": 60000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-144000-60000",
            "start_frame": 144000,
        },
        {
            "frame_count": 24000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-168000-24000",
            "start_frame": 168000,
        },
        {
            "frame_count": 60000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-168000-60000",
            "start_frame": 168000,
        },
        {
            "frame_count": 24000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-192000-24000",
            "start_frame": 192000,
        },
        {
            "frame_count": 24000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-216000-24000",
            "start_frame": 216000,
        },
        {
            "frame_count": 120000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-24000-120000",
            "start_frame": 24000,
        },
        {
            "frame_count": 24000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-24000-24000",
            "start_frame": 24000,
        },
        {
            "frame_count": 60000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-24000-60000",
            "start_frame": 24000,
        },
        {
            "frame_count": 120000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-48000-120000",
            "start_frame": 48000,
        },
        {
            "frame_count": 24000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-48000-24000",
            "start_frame": 48000,
        },
        {
            "frame_count": 60000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-48000-60000",
            "start_frame": 48000,
        },
        {
            "frame_count": 120000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-72000-120000",
            "start_frame": 72000,
        },
        {
            "frame_count": 24000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-72000-24000",
            "start_frame": 72000,
        },
        {
            "frame_count": 60000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-72000-60000",
            "start_frame": 72000,
        },
        {
            "frame_count": 120000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-96000-120000",
            "start_frame": 96000,
        },
        {
            "frame_count": 24000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-96000-24000",
            "start_frame": 96000,
        },
        {
            "frame_count": 60000,
            "id": "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f-96000-60000",
            "start_frame": 96000,
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
