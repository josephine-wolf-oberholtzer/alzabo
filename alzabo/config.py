import json
import logging
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import AnyHttpUrl, Field, FilePath, RedisDsn
from pydantic_core import Url
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import NotRequired, TypedDict

from .constants import ScsynthFeatures

logger = logging.getLogger(__name__)

ENV_PREFIX = "PRAETOR"


class ScsynthIndexConfig(TypedDict):
    alias: str | None
    features: list[ScsynthFeatures]
    pitched: bool


class AnalysisConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=f"{ENV_PREFIX}_ANALYSIS_")

    ast_collection_prefix: str = "ast"
    scsynth_collection_prefix: str = "scsynth"
    scsynth_indices: list[ScsynthIndexConfig] = Field(
        default_factory=lambda: [
            ScsynthIndexConfig(
                alias=None,
                features=[
                    ScsynthFeatures.RAW_F0_MEAN,
                    ScsynthFeatures.RAW_MFCC_13,
                    ScsynthFeatures.RAW_ONSETS,
                    ScsynthFeatures.RAW_RMS_MEAN,
                ],
                pitched=True,
            ),
            ScsynthIndexConfig(
                alias="chroma-z",
                features=[
                    ScsynthFeatures.RAW_CHROMA,
                    ScsynthFeatures.RAW_MFCC_13,
                    ScsynthFeatures.RAW_ONSETS,
                    ScsynthFeatures.RAW_RMS_MEAN,
                ],
                pitched=False,
            ),
        ]
    )
    hops: tuple[int, ...] = (500,)
    lengths: tuple[int, ...] = (500, 1250, 2500)


class ApplicationConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=f"{ENV_PREFIX}_APPLICATION_")

    analyzer_class: str = "praetor.client.analyzer.OnlineScsynthAnalyzer"
    pattern_factory_class: str = "praetor.client.pattern_factory.PatternFactory"


class ApiConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=f"{ENV_PREFIX}_API_")

    auth_enabled: bool = True
    auth_secret: str = "change-me"
    key: str | None = None
    url: AnyHttpUrl = Url("http://api:8000")


class AstConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=f"{ENV_PREFIX}_AST_")

    checkpoint_path: FilePath = Path("data/ast/audioset_model.pth")
    enabled: bool = True
    labels_path: FilePath = Path("data/ast/audioset_labels.csv")


class MidiMapping(TypedDict):
    path: str
    note: int
    type: NotRequired[Literal["through", "toggle"]]


class MidiConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=f"{ENV_PREFIX}_MIDI_")

    client_class: str = "praetor.client.midi_client.MidiClient"
    devices: dict[str, list[MidiMapping]] = Field(default_factory=dict)
    enabled: bool = False


class MilvusConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=f"{ENV_PREFIX}_MILVUS_")

    url: AnyHttpUrl = Url("http://milvus:19530")


class ArcMapping(TypedDict):
    path: str
    ring: int
    type: NotRequired[Literal["through", "toggle"]]


class MonomeConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=f"{ENV_PREFIX}_MONOME_")

    arc: list[ArcMapping] = Field(default_factory=list)
    client_class: str = "praetor.client.monome_client.MonomeClient"
    enabled: bool = False


class OpenTelemetryConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=f"{ENV_PREFIX}_OPENTELEMETRY_")

    enabled: bool = False


class RedisConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=f"{ENV_PREFIX}_REDIS_")

    url: RedisDsn = Url("redis://redis")


class S3Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=f"{ENV_PREFIX}_S3_")

    access_key_id: str | None = None
    endpoint_url: str | None = None
    secret_access_key: str | None = None
    data_bucket: str = "praetor-ai-data"
    uploads_bucket: str = "praetor-ai-uploads"


class ScsynthConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=f"{ENV_PREFIX}_SCSYNTH_")

    block_size: int = 256
    enabled: bool = True
    executable: Literal["scsynth", "supernova"] = "scsynth"
    input_bus: int = 8
    input_count: int = 8
    input_device: str | None = None
    memory_size: int = 8192 * 128
    output_bus: int = 0
    output_count: int = 8
    output_device: str | None = None


class PraetorConfig(BaseSettings):
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    application: ApplicationConfig = Field(default_factory=ApplicationConfig)
    ast: AstConfig = Field(default_factory=AstConfig)
    midi: MidiConfig = Field(default_factory=MidiConfig)
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    monome: MonomeConfig = Field(default_factory=MonomeConfig)
    open_telemetry: OpenTelemetryConfig = Field(default_factory=OpenTelemetryConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    s3: S3Config = Field(default_factory=S3Config)
    scsynth: ScsynthConfig = Field(default_factory=ScsynthConfig)


def _init_config():
    data = {}
    if config_path := os.environ.get("PRAETOR_CONFIG_PATH"):
        if (path := Path(config_path)).exists():
            text = path.read_text()
            for loader in (yaml.safe_load, json.loads):
                try:
                    data = loader(text)
                    break
                except Exception:
                    continue
            else:
                logger.warning(f"Could not parse {path}")
        else:
            logger.warning(f"PRAETOR_CONFIG_PATH {path} does not exist")
    config = PraetorConfig(**data)
    return config


config = _init_config()


__all__ = ["PraetorConfig", "config"]
