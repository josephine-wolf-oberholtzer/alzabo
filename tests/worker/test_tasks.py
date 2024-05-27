from alzabo.worker import tasks


def test_get_audio_processing_chain(
    job_id,
    milvus_ast_collection,
    milvus_scsynth_collections,
    recordings_path,
    s3_client,
) -> None:
    source_key = "ibn-arabi-44100-5s.wav"
    source_bucket = "test-source"
    s3_client.upload_file(
        Filename=recordings_path / source_key, Bucket=source_bucket, Key=source_key
    )
    assert (
        len(
            milvus_scsynth_collections[None].query(
                expr="f0 >= -1", output_fields=["id"]
            )
        )
        == 0
    )
    tasks.get_audio_processing_chain(
        job_id, f"s3://{source_bucket}/{source_key}"
    ).delay().get(timeout=120)
    assert milvus_scsynth_collections[None].num_entities == 22
    assert milvus_ast_collection.num_entities == 24
