from celery import chain, group

from ..config import config
from .ast import analyze_via_ast, insert_ast_entries
from .audio import transcode_and_hash_audio, upload_audio
from .milvus import flush_milvus
from .scsynth import (
    analyze_via_scsynth,
    insert_scsynth_entries,
    partition_scsynth_analysis,
)

__all__ = [
    "analyze_via_ast",
    "analyze_via_scsynth",
    "flush_milvus",
    "get_audio_processing_chain",
    "insert_ast_entries",
    "insert_scsynth_entries",
    "partition_scsynth_analysis",
    "transcode_and_hash_audio",
    "upload_audio",
]


def get_audio_processing_chain(job_id: str, url: str) -> chain:
    ast_chain = analyze_via_ast.s() | insert_ast_entries.s() | flush_milvus.s()
    scsynth_chain = (
        analyze_via_scsynth.s()
        | partition_scsynth_analysis.s()
        | insert_scsynth_entries.s()
        | flush_milvus.s()
    )
    analysis_group = group(scsynth_chain)
    if config.ast.enabled:
        analysis_group = group(ast_chain, scsynth_chain)
    audio_chain = upload_audio.s([job_id, url]) | transcode_and_hash_audio.s()
    return audio_chain | analysis_group
