import json
import uuid
import wave
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner
from pytest_mock import MockerFixture

from alzabo import cli
from alzabo.cli import cli as cli_entrypoint


@pytest.fixture
def recording_path(recordings_path: Path) -> Path:
    return recordings_path / "ibn-arabi-44100-1s.wav"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_audio_batch(mocker: MockerFixture, runner: CliRunner) -> None:
    mocker.patch("alzabo.cli._audio_batch")
    result = runner.invoke(cli_entrypoint, ["audio-batch"])
    assert result.exit_code == 0, result.output


@pytest.mark.asyncio
async def test__audio_batch(api_server: str, mocker: MockerFixture) -> None:
    uuids = [uuid.uuid4() for _ in range(2)]
    mocker.patch("alzabo.api.audio.uuid4", side_effect=uuids)
    mock_task = mocker.patch("alzabo.worker.tasks.get_audio_processing_chain")
    await cli._audio_batch(urls=["s3://foo/bar/baz", "s3://quux/wux"])
    assert mock_task.mock_calls == [
        mock.call(str(uuids[0]), "s3://foo/bar/baz"),
        mock.call()(),
        mock.call(str(uuids[1]), "s3://quux/wux"),
        mock.call()(),
    ]


def test_audio_fetch(mocker: MockerFixture, runner: CliRunner) -> None:
    mocker.patch("alzabo.cli._audio_fetch")
    result = runner.invoke(
        cli_entrypoint, ["audio-fetch", "deadbeef", "0", "1024", "foo/bar/baz.aif"]
    )
    assert result.exit_code == 0, result.output


@pytest.mark.asyncio
async def test__audio_fetch(api_server: str, data: None, tmp_path: Path) -> None:
    output_path = tmp_path / "foo.wav"
    await cli._audio_fetch(
        "dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969",
        0,
        1024,
        output_path,
    )
    assert output_path.exists()
    with output_path.open("rb") as file_pointer:
        with wave.open(file_pointer) as wave_file:
            assert wave_file.getframerate() == 48000
            assert wave_file.getnchannels() == 1
            assert wave_file.getsampwidth() == 2
            assert wave_file.getnframes() == 1024


def test_audio_upload(
    mocker: MockerFixture, recording_path: Path, runner: CliRunner
) -> None:
    mocker.patch("alzabo.cli._audio_upload")
    result = runner.invoke(cli_entrypoint, ["audio-upload", str(recording_path)])
    assert result.exit_code == 0, result.output


@pytest.mark.asyncio
async def test__audio_upload(
    api_server: str, mocker: MockerFixture, recording_path: Path
) -> None:
    paths = (str(recording_path),)
    uuids = [uuid.uuid4() for _ in range(2)]
    mocker.patch("alzabo.api.audio.uuid4", side_effect=uuids)
    mock_task = mocker.patch("alzabo.worker.tasks.get_audio_processing_chain")
    await cli._audio_upload(paths, max_concurrency=1)
    assert mock_task.mock_calls == [
        mock.call(str(uuids[0]), f"s3://test-uploads/{uuids[1]}"),
        mock.call()(),
    ]


def test_ensure_buckets(runner: CliRunner) -> None:
    result = runner.invoke(cli_entrypoint, ["ensure-buckets"])
    assert result.exit_code == 0, result.output


def test_ensure_database(runner: CliRunner) -> None:
    result = runner.invoke(cli_entrypoint, ["ensure-database"])
    assert result.exit_code == 0, result.output


def test_ping(mocker: MockerFixture, runner: CliRunner) -> None:
    mocker.patch("alzabo.cli._ping")
    result = runner.invoke(cli_entrypoint, ["ping"])
    assert result.exit_code == 0, result.output


@pytest.mark.asyncio
async def test__ping(api_server: str) -> None:
    assert (await cli._ping()) == "pong!"


def test_query_ast(mocker: MockerFixture, runner: CliRunner) -> None:
    mocker.patch("alzabo.cli._query_ast")
    result = runner.invoke(cli_entrypoint, ["query-ast"])
    assert result.exit_code == 0, result.output


@pytest.mark.asyncio
async def test__query_ast(api_server: str, data: None, recording_path: Path) -> None:
    upload_data = json.loads(
        await cli._query_ast_upload(limit=10, path=recording_path, partition=[])
    )
    query_data = json.loads(
        await cli._query_ast(limit=10, partition=[], vector=upload_data["vector"])
    )
    assert query_data["entries"] == upload_data["entries"]


def test_query_ast_upload(
    mocker: MockerFixture, recording_path: Path, runner: CliRunner
) -> None:
    mocker.patch("alzabo.cli._query_ast_upload")
    result = runner.invoke(cli_entrypoint, ["query-ast-upload", str(recording_path)])
    assert result.exit_code == 0, result.output


@pytest.mark.asyncio
async def test__query_ast_upload(
    api_server: str, data: None, recording_path: Path
) -> None:
    upload_data = json.loads(
        await cli._query_ast_upload(limit=10, path=recording_path, partition=[])
    )
    assert upload_data["entries"]
    assert len(upload_data["vector"]) == 527


def test_query_scsynth(mocker: MockerFixture, runner: CliRunner) -> None:
    mocker.patch("alzabo.cli._query_scsynth")
    result = runner.invoke(cli_entrypoint, ["query-scsynth"])
    assert result.exit_code == 0, result.output


@pytest.mark.asyncio
async def test__query_scsynth(
    api_server: str, data: None, recording_path: Path
) -> None:
    upload_data = json.loads(
        await cli._query_scsynth_upload(
            index=None, limit=10, path=recording_path, partition=[]
        )
    )
    query_data = json.loads(
        await cli._query_scsynth(
            index=None, limit=10, partition=[], vector=upload_data["vector"]
        )
    )
    assert query_data["entries"] == upload_data["entries"]


def test_query_scsynth_upload(
    mocker: MockerFixture, recording_path: Path, runner: CliRunner
) -> None:
    mocker.patch("alzabo.cli._query_scsynth_upload")
    result = runner.invoke(
        cli_entrypoint, ["query-scsynth-upload", str(recording_path)]
    )
    assert result.exit_code == 0, result.output


@pytest.mark.asyncio
async def test__query_scsynth_upload(
    api_server: str, data: None, recording_path: Path
) -> None:
    upload_data = json.loads(
        await cli._query_scsynth_upload(
            index=None, limit=10, path=recording_path, partition=[]
        )
    )
    assert upload_data["analysis"] == {
        "is_voiced": True,
        "r:centroid:mean": 88.17190929125714,
        "r:centroid:std": 20.369653466305373,
        "r:chroma": [
            0.05874799740699781,
            0.04503894776984629,
            0.069211317607457,
            0.1115863282684586,
            0.1336060847311909,
            0.0937659737628369,
            0.04918238589219669,
            0.04224279223275008,
            0.0585237980876725,
            0.07917958861931981,
            0.1072172276373788,
            0.0979341172812862,
        ],
        "r:f0:mean": 65.12738627115885,
        "r:f0:std": 6.799382069906058,
        "r:flatness:mean": 0.08147228711975678,
        "r:flatness:std": 0.17854156684254593,
        "r:mfcc": [
            0.4984107557483899,
            0.015506617484554168,
            0.14230795477026253,
            -0.02995609644279685,
            0.09624801512046527,
            0.1683859033610231,
            0.2544622818628947,
            0.26708049719692556,
            0.25776375766082477,
            0.3831833788464146,
            0.2761819186390087,
            0.3476886949552003,
            0.23867011438774807,
            0.3092360318668427,
            0.2500811139101623,
            0.20612863042662222,
            0.2342671694294099,
            0.18811935891387283,
            0.2664916935146496,
            0.28691720962524414,
            0.2702096606134087,
            0.26736002663771313,
            0.23469349870117762,
            0.24804576830838315,
            0.2903657736637259,
            0.2515834108475716,
            0.22239993881153805,
            0.21388960076916602,
            0.2271559033342587,
            0.2544931064369858,
            0.26196783284346264,
            0.2550640678213489,
            0.24369035565084027,
            0.2496765358473665,
            0.2535663881609517,
            0.24937965969244638,
            0.2350220561668437,
            0.24053794033424827,
            0.25747482959301243,
            0.24746690338016838,
            0.24862708840318906,
            0.25,
        ],
        "r:onsets": 0.07526881720430108,
        "r:peak:mean": -54.702342423059605,
        "r:peak:std": 151.694523232371,
        "r:rms:mean": -53.331234178235455,
        "r:rms:std": 152.032813287156,
        "r:rolloff:mean": 75.15474323559833,
        "r:rolloff:std": 16.64606645139736,
        "w:centroid:mean": 88.17190929125714,
        "w:centroid:std": 20.369653466305373,
        "w:chroma": [
            0.05874799740699781,
            0.04503894776984629,
            0.069211317607457,
            0.1115863282684586,
            0.1336060847311909,
            0.0937659737628369,
            0.04918238589219669,
            0.04224279223275008,
            0.0585237980876725,
            0.07917958861931981,
            0.1072172276373788,
            0.0979341172812862,
        ],
        "w:f0:mean": 65.12738627115885,
        "w:f0:std": 6.799382069906058,
        "w:flatness:mean": 0.08147228711975678,
        "w:flatness:std": 0.17854156684254593,
        "w:mfcc": [
            0.4984107557483899,
            0.015506617484554168,
            0.14230795477026253,
            -0.02995609644279685,
            0.09624801512046527,
            0.1683859033610231,
            0.2544622818628947,
            0.26708049719692556,
            0.25776375766082477,
            0.3831833788464146,
            0.2761819186390087,
            0.3476886949552003,
            0.23867011438774807,
            0.3092360318668427,
            0.2500811139101623,
            0.20612863042662222,
            0.2342671694294099,
            0.18811935891387283,
            0.2664916935146496,
            0.28691720962524414,
            0.2702096606134087,
            0.26736002663771313,
            0.23469349870117762,
            0.24804576830838315,
            0.2903657736637259,
            0.2515834108475716,
            0.22239993881153805,
            0.21388960076916602,
            0.2271559033342587,
            0.2544931064369858,
            0.26196783284346264,
            0.2550640678213489,
            0.24369035565084027,
            0.2496765358473665,
            0.2535663881609517,
            0.24937965969244638,
            0.2350220561668437,
            0.24053794033424827,
            0.25747482959301243,
            0.24746690338016838,
            0.24862708840318906,
            0.25,
        ],
        "w:onsets": 0.5237858632918295,
        "w:peak:mean": -54.702342423059605,
        "w:peak:std": 151.694523232371,
        "w:rms:mean": -53.331234178235455,
        "w:rms:std": 152.032813287156,
        "w:rolloff:mean": 75.15474323559833,
        "w:rolloff:std": 16.64606645139736,
    }
    assert upload_data["entries"]
    assert upload_data["vector"] == [
        65.12738627115885,
        0.4984107557483899,
        0.015506617484554168,
        0.14230795477026253,
        -0.02995609644279685,
        0.09624801512046527,
        0.1683859033610231,
        0.2544622818628947,
        0.26708049719692556,
        0.25776375766082477,
        0.3831833788464146,
        0.2761819186390087,
        0.3476886949552003,
        0.23867011438774807,
        0.07526881720430108,
        -53.331234178235455,
    ]
