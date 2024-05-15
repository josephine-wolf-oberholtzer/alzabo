import json
from pathlib import Path

import pytest
from pymilvus import Collection

from praetor.core import scsynth


@pytest.mark.parametrize(
    "entries_filename, expected_count",
    [
        ("scsynth-entries-500-1250.json", 166),
        ("scsynth-entries-500-2500.json", 163),
        ("scsynth-entries-500-500.json", 167),
    ],
)
def test_insert_scsynth_entries(
    data_path: Path,
    entries_filename: str,
    expected_count: int,
    milvus_scsynth_collections: dict[str | None, Collection],
) -> None:
    digest = "dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969"
    # verify count before inserting
    results = milvus_scsynth_collections[None].query(
        expr=f'digest == "{digest}"', output_fields=["id"]
    )
    assert len(results) == 0
    # insert entries
    entries_path = data_path / digest[:2] / digest / entries_filename
    entries = json.loads(entries_path.read_text())["entries"]
    scsynth.insert_scsynth_entries(digest, entries)
    collection = scsynth.get_scsynth_collection()
    collection.flush()
    results = milvus_scsynth_collections[None].query(
        expr=f'digest == "{digest}"', output_fields=["id"]
    )
    assert len(results) == expected_count


def test_query_scsynth_entries(
    data_path: Path, milvus_scsynth_collections: dict[str | None, Collection]
) -> None:
    # insert entries
    digest = "dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969"
    for entries_path in (data_path / digest[:2] / digest).glob(
        "scsynth-entries-*.json"
    ):
        entries = json.loads(entries_path.read_text())["entries"]
        scsynth.insert_scsynth_entries(digest, entries)
    collection = scsynth.get_scsynth_collection()
    collection.flush()
    # verify count after inserting
    results = milvus_scsynth_collections[None].query(
        expr=f'digest == "{digest}"', output_fields=["id"]
    )
    assert len(results) == 496
    # query the collection
    query_result = scsynth.query_scsynth_collection(
        vector=scsynth.aggregate_to_vector(entries[0][-1], index_alias=None), limit=3
    )
    assert len(query_result) == 3
    assert all(
        x["digest"]
        == "dd88610b66f3f053243f8f315345381fc70bca20d48ba32e27a7841d7676f969"
        for x in query_result
    )
