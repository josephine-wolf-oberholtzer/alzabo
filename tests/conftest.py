import logging
import subprocess
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
import requests_mock
from aiohttp.test_utils import TestServer
from botocore.exceptions import ClientError
from celery import Celery
from pymilvus import Collection, utility
from supriya.contexts import AsyncServer
from supriya.osc import find_free_port

import praetor.api
import praetor.core.milvus
import praetor.worker
from praetor.config import config
from praetor.core.s3 import create_s3_client


@pytest.fixture(autouse=True)
def logging_setup(caplog) -> None:
    caplog.set_level(logging.WARNING)
    for logger in ("aiohttp", "praetor", "supriya"):
        caplog.set_level(logging.INFO, logger=logger)


@pytest.fixture(autouse=True, scope="session")
def jackd_dummy_process() -> Generator:
    command = ["jackd", "-r", "-ddummy", "-r44100", "-p1024"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(1)
    yield
    process.terminate()
    process.wait()


@pytest_asyncio.fixture
async def context() -> AsyncGenerator[AsyncServer, None]:
    context = AsyncServer()
    yield await context.boot(
        port=find_free_port(),
        # ugen_plugins_path="/usr/local/lib/SuperCollider/plugins:/usr/lib/SuperCollider/Extensions",
    )
    await context.quit()


@pytest_asyncio.fixture
async def api_server(praetor_config, monkeypatch) -> AsyncGenerator[str, None]:
    """
    An AIOHTTP TestServer wrapping the Praetor API.
    """
    server = TestServer(praetor.api.create_app())
    await server.start_server()
    api_url = f"{server.scheme}://{server.host}:{server.port}"
    monkeypatch.setattr(config.api, "url", api_url)
    yield api_url
    await server.close()


@pytest.fixture(autouse=True)
def celery_app(praetor_config: None) -> Celery:
    app = praetor.worker.create_app()
    app.conf.task_always_eager = True
    return app


@pytest.fixture(autouse=True)
def praetor_config(monkeypatch) -> None:
    """
    Patch praetor's config
    """
    monkeypatch.setattr(config.analysis, "ast_collection_prefix", "test_ast")
    monkeypatch.setattr(config.analysis, "scsynth_collection_prefix", "test_scsynth")
    monkeypatch.setattr(config.api, "auth_enabled", False)
    monkeypatch.setattr(config.open_telemetry, "enabled", False)
    monkeypatch.setattr(config.s3, "data_bucket", "test-data")
    monkeypatch.setattr(config.s3, "uploads_bucket", "test-uploads")


@pytest.fixture
def data_path() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture
def data(
    data_path, milvus_ast_collection, milvus_scsynth_collections, s3_client
) -> None:
    for path in data_path.glob("**/*"):
        if not path.is_file():
            continue
        s3_client.upload_file(
            Bucket=config.s3.data_bucket,
            Filename=path,
            Key=str(path.relative_to(data_path)),
        )
    praetor.worker.tasks.insert_ast_entries(
        [
            str(uuid.uuid4()),
            "af5ec6ae3e17614ebf7c2575dc8870cfbb32f12e5b7edabbdda2b02b8b9b7e5f",
        ]
    )
    praetor.worker.tasks.insert_scsynth_entries(
        [
            str(uuid.uuid4()),
            "dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969",
        ]
    )
    praetor.worker.tasks.flush_milvus()


@pytest.fixture(scope="session")
def milvus() -> None:
    praetor.core.milvus.connect()


@pytest.fixture
def milvus_ast_collection(milvus) -> Collection:
    utility.drop_collection(praetor.core.ast.get_ast_collection_name())
    collection = praetor.core.ast.create_ast_collection()
    collection.load()
    return collection


@pytest.fixture
def milvus_scsynth_collections(milvus) -> dict[str | None, Collection]:
    collections = {}
    for index_config in config.analysis.scsynth_indices:
        utility.drop_collection(
            praetor.core.scsynth.get_scsynth_collection_name(index_config["alias"])
        )
        collection = praetor.core.scsynth.create_scsynth_collection(
            index_config["alias"]
        )
        collection.load()
        collections[index_config["alias"]] = collection
    return collections


@pytest.fixture
def recordings_path() -> Path:
    return Path(__file__).parent / "recordings"


@pytest.fixture()
def requests_adapter() -> requests_mock.Adapter:
    """
    Adapter exposes the request history
    """
    return requests_mock.Adapter()


@pytest.fixture(autouse=True)
def requests_mocker(requests_adapter) -> Generator[requests_mock.Mocker, None, None]:
    """
    Prevent external requests
    """
    with requests_mock.Mocker(adapter=requests_adapter) as requests_mocker:
        yield requests_mocker


@pytest.fixture(autouse=True)
def s3_client(praetor_config):
    """
    Mock out S3, and create default buckets
    """
    client = create_s3_client()
    buckets = [config.s3.data_bucket, config.s3.uploads_bucket, "test-source"]
    for bucket in buckets:
        try:
            # If the bucket already exists, delete its contents.
            client.head_bucket(Bucket=bucket)
            for object_ in client.list_objects_v2(Bucket=bucket).get("Contents", []):
                client.delete_object(Bucket=bucket, Key=object_["Key"])
        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                raise
            # If it doesn't exist, create it.
            client.create_bucket(Bucket=bucket)
    yield client
