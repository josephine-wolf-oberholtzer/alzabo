import asyncio
import json
import logging
from pathlib import Path

import aiohttp
import click
from botocore.exceptions import ClientError
from pymilvus import utility
from tqdm import tqdm

from .client import APIClient, Application
from .config import config
from .core import ast, milvus, s3, scsynth


@click.group()
@click.option("--api-url", default="http://localhost:8000")
@click.option("--api-key", type=str)
def cli(api_url, api_key):
    if api_key:
        config.api.key = api_key
    if api_url:
        config.api.url = api_url


### PIPELINE


@cli.command()
def ensure_buckets() -> None:
    client = s3.create_s3_client()
    for bucket in [config.s3.data_bucket, config.s3.uploads_bucket]:
        try:
            client.head_bucket(Bucket=bucket)
            continue
        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                raise
        client.create_bucket(Bucket=bucket)


@cli.command()
def ensure_database() -> None:
    milvus.connect()
    ast.get_or_create_ast_collection().load()
    for index_config in config.analysis.scsynth_indices:
        if not utility.has_collection(
            scsynth.get_scsynth_collection_name(index_config["alias"])
        ):
            scsynth.create_scsynth_collection(index_config["alias"]).load()


### API CLIENT


async def _audio_batch(urls: list[str]) -> str:
    api_client = APIClient(api_url=str(config.api.url), api_key=config.api.key)
    return json.dumps(await api_client.audio_batch(urls=urls), indent=4, sort_keys=True)


@cli.command()
@click.argument("urls", nargs=-1)
def audio_batch(urls: list[str]) -> None:
    print(asyncio.run(_audio_batch(urls=urls)))


async def _audio_fetch(digest: str, start: int, count: int, path: Path) -> None:
    api_client = APIClient(api_url=str(config.api.url), api_key=config.api.key)
    path_ = Path(path)
    data = await api_client.audio_fetch(
        digest=digest, start_frame=start, frame_count=count
    )
    path_.write_bytes(data)


@cli.command()
@click.argument("digest", type=str)
@click.argument("start", type=int)
@click.argument("count", type=int)
@click.argument("path", type=click.Path(exists=False))
def audio_fetch(digest: str, start: int, count: int, path: Path) -> None:
    asyncio.run(_audio_fetch(digest=digest, start=start, count=count, path=path))


async def _audio_upload(paths: tuple[str], *, max_concurrency: int | None) -> str:
    async def upload(paths: list[Path]) -> list[str]:
        tasks = [upload_task(path) for path in paths]
        return await asyncio.gather(*tasks)

    async def upload_task(path: Path) -> str:
        with tqdm(
            total=path.stat().st_size, desc=str(path), leave=False
        ) as progress_bar:
            job_id = await api_client.audio_upload(
                path=path,
                progress_callback=lambda size, current, total: progress_bar.update(
                    size
                ),
            )
            outer_progress_bar.update(1)
            return job_id

    connector: aiohttp.BaseConnector | None = None
    if max_concurrency is not None and max_concurrency < 1:
        raise RuntimeError("Concurrency must be at least 1, got {max_concurrency}")
    elif max_concurrency:
        connector = aiohttp.TCPConnector(limit=max_concurrency)
    api_client = APIClient(
        api_url=str(config.api.url), api_key=config.api.key, connector=connector
    )
    paths_ = sorted(Path(path) for path in paths)
    with tqdm(total=len(paths_), position=0, desc="files") as outer_progress_bar:
        job_ids = await upload(paths_)
    return json.dumps({"jobs": job_ids}, indent=4, sort_keys=True)


@cli.command()
@click.option("--max-concurrency", type=int, help="optional upload concurrency limit")
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
def audio_upload(paths: tuple[str], max_concurrency: int | None = None) -> None:
    print(asyncio.run(_audio_upload(max_concurrency=max_concurrency, paths=paths)))


async def _ping() -> str:
    api_client = APIClient(api_url=str(config.api.url), api_key=config.api.key)
    return await api_client.ping()


@cli.command()
def ping() -> None:
    print(asyncio.run(_ping()))


async def _query_ast(
    limit: int, partition: list[str], vector: tuple[float, ...]
) -> str:
    api_client = APIClient(api_url=str(config.api.url), api_key=config.api.key)
    return json.dumps(
        await api_client.query_ast(limit=limit, partitions=partition, vector=vector),
        indent=4,
        sort_keys=True,
    )


@cli.command()
@click.option("--limit", default=10, type=int)
@click.option("--partition", multiple=True, default=[])
@click.argument("vector", nargs=-1, type=float)
def query_ast(limit: int, partition: list[str], vector: tuple[float, ...]) -> None:
    print(asyncio.run(_query_ast(limit, partition, vector)))


async def _query_ast_upload(limit: int, path: Path, partition: list[str]) -> str:
    api_client = APIClient(api_url=str(config.api.url), api_key=config.api.key)
    return json.dumps(
        await api_client.query_ast_upload(path=path, limit=limit, partitions=partition),
        indent=4,
        sort_keys=True,
    )


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--limit", default=10, type=int)
@click.option("--partition", multiple=True, default=[])
def query_ast_upload(limit: int, path: Path, partition: list[str]) -> None:
    print(asyncio.run(_query_ast_upload(limit=limit, path=path, partition=partition)))


async def _query_scsynth(
    index: str | None, limit: int, partition: list[str], vector: tuple[float, ...]
) -> str:
    api_client = APIClient(api_url=str(config.api.url), api_key=config.api.key)
    return json.dumps(
        await api_client.query_scsynth(
            index=index, limit=limit, partitions=partition, vector=vector
        ),
        indent=4,
        sort_keys=True,
    )


@cli.command()
@click.option("--index", default=None, type=str)
@click.option("--limit", default=10, type=int)
@click.option("--partition", multiple=True, default=[])
@click.argument("vector", nargs=-1, type=float)
def query_scsynth(
    index: str | None, limit: int, partition: list[str], vector: tuple[float, ...]
) -> None:
    print(
        asyncio.run(
            _query_scsynth(index=index, limit=limit, partition=partition, vector=vector)
        )
    )


async def _query_scsynth_upload(
    index: str | None, limit: int, path: Path, partition: list[str]
) -> str:
    api_client = APIClient(api_url=str(config.api.url), api_key=config.api.key)
    return json.dumps(
        await api_client.query_scsynth_upload(
            index=index, limit=limit, partitions=partition, path=path
        ),
        indent=4,
        sort_keys=True,
    )


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--index", default=None, type=str)
@click.option("--limit", default=10, type=int)
@click.option("--partition", multiple=True, default=[])
def query_scsynth_upload(
    index: str | None, limit: int, path: Path, partition: list[str]
) -> None:
    print(
        asyncio.run(
            _query_scsynth_upload(
                index=index, limit=limit, path=path, partition=partition
            )
        )
    )


@cli.command()
def run():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("praetor").setLevel(logging.INFO)
    logging.getLogger("supriya.osc").setLevel(logging.WARNING)
    logging.getLogger("supriya.udp").setLevel(logging.INFO)
    logging.getLogger("supriya.scsynth").setLevel(logging.INFO)
    loop = asyncio.get_event_loop()
    application = Application()
    try:
        loop.run_until_complete(application.run())
    finally:
        loop.run_until_complete(application.quit())
