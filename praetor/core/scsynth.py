import json
import logging
import math
import wave
from hashlib import md5
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Sequence, TypedDict, cast

import numpy
import redis
import soundfile
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility
from sklearn.preprocessing import StandardScaler
from supriya import CalculationRate, Score, SynthDef, synthdef
from supriya.ugens import (
    FFT,
    LPF,
    MFCC,
    Amplitude,
    BufFrames,
    BufWr,
    Impulse,
    In,
    Line,
    LocalBuf,
    Onsets,
    Pitch,
    SampleRate,
    Sanitize,
    SendReply,
    SpecCentroid,
    SpecFlatness,
    SpecPcile,
)
from supriya.ugens.core import UGen, param, ugen

from ..config import ScsynthIndexConfig, config
from ..constants import SCSYNTH_ANALYSIS_SIZE, ScsynthFeatures
from .milvus import Entry

logger = logging.getLogger(__name__)


@ugen(channel_count=12, kr=True, is_multichannel=True, is_pure=True)
class FluidChroma(UGen):
    """
    FluCoMa chromagram UGen.

    Only works with scsynth.
    """

    source = param(None)
    chroma_count = param(12)
    max_chroma_count = param(12)
    reference = param(0.0)
    normalize = param(0)
    min_frequency = param(0.0)
    max_frequency = param(-1.0)
    window_size = param(1024)
    hop_size = param(-1)
    fft_size = param(-1)
    max_fft_size = param(-1)

    @classmethod
    def kr(
        cls,
        source=None,
        chroma_count=12,
        max_chroma_count=12,
        reference=0.0,
        normalize=0,
        min_frequency=0.0,
        max_frequency=-1.0,
        window_size=1024,
        hop_size=-1,
        fft_size=-1,
        max_fft_size=-1,
    ):
        return cls._new_expanded(
            calculation_rate=CalculationRate.CONTROL,
            channel_count=int(max_chroma_count),
            source=source,
            chroma_count=chroma_count,
            max_chroma_count=max_chroma_count,
            reference=reference,
            normalize=normalize,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            window_size=window_size,
            hop_size=hop_size,
            fft_size=fft_size,
            max_fft_size=max_fft_size,
        )


Aggregate = TypedDict(
    "Aggregate",
    {
        "is_voiced": bool,
        "r:centroid:mean": float,
        "r:centroid:std": float,
        "r:chroma": list[float],
        "r:f0:mean": float,
        "r:f0:std": float,
        "r:flatness:mean": float,
        "r:flatness:std": float,
        "r:mfcc": list[float],
        "r:onsets": float,
        "r:peak:mean": float,
        "r:peak:std": float,
        "r:rms:mean": float,
        "r:rms:std": float,
        "r:rolloff:mean": float,
        "r:rolloff:std": float,
        "w:centroid:mean": float,
        "w:centroid:std": float,
        "w:chroma": list[float],
        "w:f0:mean": float,
        "w:f0:std": float,
        "w:flatness:mean": float,
        "w:flatness:std": float,
        "w:mfcc": list[float],
        "w:onsets": float,
        "w:peak:mean": float,
        "w:peak:std": float,
        "w:rms:mean": float,
        "w:rms:std": float,
        "w:rolloff:mean": float,
        "w:rolloff:std": float,
    },
)


def core_synthdef_analysis(
    source,
    executable: Literal["scsynth", "supernova"] = "scsynth",
    frame_length: int = 2048,
    hop_ratio: float = 0.25,
    pitch_detection_frequency_max: float = 3000.0,
    pitch_detection_frequency_min: float = 60.0,
):
    """
    The core UGen graph shared by online and offline analysis SynthDefs.
    """
    peak = Amplitude.ar(source=source).amplitude_to_db()
    rms = LPF.ar(source=source * source, frequency=10.0).sqrt().amplitude_to_db()
    frequency, is_voiced = Pitch.kr(
        source=source,
        max_frequency=pitch_detection_frequency_max,
        min_frequency=pitch_detection_frequency_min,
    )
    pv_chain = FFT.kr(source=source, hop=hop_ratio, window_size=frame_length)
    is_onset = Onsets.kr(
        pv_chain=pv_chain,
        floor_=0.000001,
        relaxtime=0.1,
        threshold=0.01,
        odftype=Onsets.ODFType.WPHASE,
    )
    centroid = SpecCentroid.kr(pv_chain=pv_chain)
    flatness = SpecFlatness.kr(pv_chain=pv_chain)
    rolloff = SpecPcile.kr(pv_chain=pv_chain)
    mfcc = MFCC.kr(pv_chain=pv_chain, coeff_count=42)
    features = [
        peak,
        rms,
        frequency.hz_to_midi(),
        is_voiced,
        is_onset,
        Sanitize.kr(source=centroid.hz_to_midi()),
        flatness,
        Sanitize.kr(source=rolloff.hz_to_midi()),
        *mfcc,
    ]
    if executable == "scsynth":
        chroma = FluidChroma.kr(
            fft_size=2048,
            hop_size=512,
            max_fft_size=2048,
            max_frequency=SampleRate.ir() / 2,
            min_frequency=0.0,
            reference=1.0,
            normalize=1.0,
            source=source,
            window_size=2048,
        )
        features.extend(chroma)
    return features


def build_offline_analysis_synthdef(
    frame_length: int = 2048,
    hop_ratio: float = 0.25,
    pitch_detection_frequency_max: float = 3000.0,
    pitch_detection_frequency_min: float = 60.0,
) -> SynthDef:
    @synthdef()
    def analysis(in_, buffer_id, duration):
        source = In.ar(bus=in_)
        phase = math.floor(Line.kr(
            start=0, stop=BufFrames.kr(buffer_id=buffer_id) - 1, duration=duration
        ))
        analysis_source = core_synthdef_analysis(
            source,
            frame_length=frame_length,
            hop_ratio=hop_ratio,
            pitch_detection_frequency_max=pitch_detection_frequency_max,
            pitch_detection_frequency_min=pitch_detection_frequency_min,
        )
        BufWr.kr(buffer_id=buffer_id, phase=phase, source=analysis_source)

    return analysis


def build_online_analysis_synthdef(
    executable: Literal["scsynth", "supernova"] = "scsynth",
    frame_length: int = 2048,
    hop_ratio: float = 0.25,
    pitch_detection_frequency_max: float = 3000.0,
    pitch_detection_frequency_min: float = 60.0,
) -> SynthDef:
    @synthdef()
    def analysis(in_):
        source = In.ar(bus=in_)
        trigger = Impulse.kr(frequency=SampleRate.ir() / 512)  # pin to sample-accurate
        SendReply.kr(
            command_name="/analysis",
            source=core_synthdef_analysis(
                source,
                executable=executable,
                frame_length=frame_length,
                hop_ratio=hop_ratio,
                pitch_detection_frequency_max=pitch_detection_frequency_max,
                pitch_detection_frequency_min=pitch_detection_frequency_min,
            ),
            trigger=trigger,
        )

    return analysis


async def analyze(audio_path: Path) -> numpy.ndarray:
    with audio_path.open("rb") as audio_file:
        with wave.open(audio_file) as wave_file:
            audio_frame_count = wave_file.getnframes()
    analysis_frame_count = audio_frame_count // 512
    analysis_duration = analysis_frame_count * 512 / 48000
    with TemporaryDirectory() as temp_directory:
        analysis_path = Path(temp_directory) / "analysis.wav"
        score = Score(input_bus_channel_count=1)
        synthdef = build_offline_analysis_synthdef()
        with score.at(0):
            buffer_ = score.add_buffer(
                channel_count=42 + 12 + 8, frame_count=analysis_frame_count
            )
            with score.add_synthdefs(synthdef):
                score.add_synth(
                    synthdef=synthdef,
                    in_=score.audio_input_bus_group,
                    buffer_id=buffer_,
                    duration=analysis_duration,
                )
            with score.at(analysis_duration):
                buffer_.write(
                    file_path=analysis_path, header_format="WAV", sample_format="FLOAT"
                )
        _, exit_code = await score.render(
            input_file_path=audio_path,
            render_directory_path=Path(temp_directory),
            sample_rate=48000,
            suppress_output=True,
            # ugen_plugins_path="/usr/local/lib/SuperCollider/plugins:/usr/lib/SuperCollider/Extensions",
        )
        analysis, _ = soundfile.read(analysis_path)
    return analysis


def partition(
    *,
    raw_analysis: numpy.ndarray,
    whitened_analysis: numpy.ndarray | None = None,
    hop_ms: int,
    length_ms: int,
    sample_rate: int = 48000,
    frame_size: int = 512,
) -> Sequence[tuple[int, int, Aggregate]]:
    if whitened_analysis is None:
        whitened_analysis = raw_analysis
    frame_ms = frame_size / sample_rate * 1000
    indices_per_hop = math.ceil(hop_ms / frame_ms)
    indices_per_entry = math.ceil(length_ms / frame_ms)
    frame_count = raw_analysis.shape[0]
    entries: list[tuple[int, int, Aggregate]] = []
    for start_index in range(0, frame_count, indices_per_hop):
        stop_index = start_index + indices_per_entry
        if stop_index > frame_count:
            break
        entries.append(
            (
                start_index * frame_size,
                (stop_index - start_index) * frame_size,
                aggregate(raw_analysis[start_index:stop_index]),
            )
        )
    return entries


def aggregate(
    raw_analysis: numpy.ndarray, whitened_analysis: numpy.ndarray | None = None
) -> Aggregate:
    if raw_analysis.shape[-1] != SCSYNTH_ANALYSIS_SIZE:
        raise ValueError(raw_analysis.shape)
    if whitened_analysis is None:
        whitened_analysis = raw_analysis
    is_voiced = bool(numpy.median(raw_analysis[:, 3]))
    raw_centroid = raw_analysis[:, 5]
    raw_chroma = raw_analysis[:, -12:]
    raw_f0 = raw_analysis[:, 2][numpy.array(raw_analysis[:, 3], dtype=numpy.bool_)]
    raw_flatness = raw_analysis[:, 6]
    raw_mfcc = raw_analysis[:, 8:-12]
    raw_onsets = float(numpy.mean(raw_analysis[:, 4]))
    raw_peak = raw_analysis[:, 0]
    raw_rms = raw_analysis[:, 1]
    raw_rolloff = raw_analysis[:, 7]
    whitened_centroid = whitened_analysis[:, 5]
    whitened_chroma = whitened_analysis[:, -12:]
    whitened_f0 = whitened_analysis[:, 2][
        numpy.array(raw_analysis[:, 3], dtype=numpy.bool_)
    ]
    whitened_flatness = whitened_analysis[:, 6]
    whitened_mfcc = whitened_analysis[:, 8:-12]
    whitened_peak = whitened_analysis[:, 0]
    whitened_rms = whitened_analysis[:, 1]
    whitened_rolloff = whitened_analysis[:, 7]
    data: Aggregate = {
        "is_voiced": is_voiced,
        "r:centroid:mean": float(numpy.mean(raw_centroid)),
        "r:centroid:std": float(numpy.std(raw_centroid)),
        "r:chroma": numpy.mean(raw_chroma, axis=0).tolist(),
        "r:f0:mean": -1.0,
        "r:f0:std": 0.0,
        "r:flatness:mean": float(numpy.mean(raw_flatness)),
        "r:flatness:std": float(numpy.std(raw_flatness)),
        "r:mfcc": numpy.mean(raw_mfcc, axis=0).tolist(),
        "r:onsets": raw_onsets,
        "r:peak:mean": float(numpy.mean(raw_peak)),
        "r:peak:std": float(numpy.std(raw_peak)),
        "r:rms:mean": float(numpy.mean(raw_rms)),
        "r:rms:std": float(numpy.std(raw_rms)),
        "r:rolloff:mean": float(numpy.mean(raw_rolloff)),
        "r:rolloff:std": float(numpy.std(raw_rolloff)),
        "w:centroid:mean": float(numpy.mean(whitened_centroid)),
        "w:centroid:std": float(numpy.std(whitened_centroid)),
        "w:chroma": numpy.mean(whitened_chroma, axis=0).tolist(),
        "w:f0:mean": -1.0,
        "w:f0:std": 0.0,
        "w:flatness:mean": float(numpy.mean(whitened_flatness)),
        "w:flatness:std": float(numpy.std(whitened_flatness)),
        "w:mfcc": numpy.mean(whitened_mfcc, axis=0).tolist(),
        "w:onsets": raw_onsets**0.25,
        "w:peak:mean": float(numpy.mean(whitened_peak)),
        "w:peak:std": float(numpy.std(whitened_peak)),
        "w:rms:mean": float(numpy.mean(whitened_rms)),
        "w:rms:std": float(numpy.std(whitened_rms)),
        "w:rolloff:mean": float(numpy.mean(whitened_rolloff)),
        "w:rolloff:std": float(numpy.std(whitened_rolloff)),
    }
    if is_voiced:
        data["r:f0:mean"] = raw_f0.mean()
        data["r:f0:std"] = raw_f0.std()
        data["w:f0:mean"] = whitened_f0.mean()
        data["w:f0:std"] = whitened_f0.std()
    return data


def get_index_config(index_alias: str | None) -> ScsynthIndexConfig:
    try:
        return [
            index_config
            for index_config in config.analysis.scsynth_indices
            if index_config["alias"] == index_alias
        ][0]
    except IndexError as e:
        raise ValueError from e


def aggregate_to_vector(
    aggregate: Aggregate, index_alias: str | None
) -> tuple[float, ...]:
    # This is really annoying, but MyPy cannot match a string variable to a
    # TypedDict key
    vector: list[float] = []
    for feature in sorted(get_index_config(index_alias)["features"]):
        if feature == ScsynthFeatures.RAW_CHROMA:
            vector.extend(aggregate["r:chroma"])
        elif feature == ScsynthFeatures.WHITENED_CHROMA:
            vector.extend(aggregate["w:chroma"])
        elif feature == ScsynthFeatures.RAW_MFCC:
            vector.extend(aggregate["r:mfcc"])
        elif feature == ScsynthFeatures.WHITENED_MFCC:
            vector.extend(aggregate["w:mfcc"])
        elif feature == "r:mfcc:13":
            vector.extend(aggregate["r:mfcc"][:13])
        elif feature == "w:mfcc:13":
            vector.extend(aggregate["w:mfcc"][:13])
        elif feature == ScsynthFeatures.IS_VOICED:
            vector.append(aggregate["is_voiced"])
        elif feature == ScsynthFeatures.RAW_CENTROID_MEAN:
            vector.append(aggregate["r:centroid:mean"])
        elif feature == ScsynthFeatures.RAW_CENTROID_STD:
            vector.append(aggregate["r:centroid:std"])
        elif feature == ScsynthFeatures.RAW_F0_MEAN:
            vector.append(aggregate["r:f0:mean"])
        elif feature == ScsynthFeatures.RAW_F0_STD:
            vector.append(aggregate["r:f0:std"])
        elif feature == ScsynthFeatures.RAW_FLATNESS_MEAN:
            vector.append(aggregate["r:flatness:mean"])
        elif feature == ScsynthFeatures.RAW_FLATNESS_STD:
            vector.append(aggregate["r:flatness:std"])
        elif feature == ScsynthFeatures.RAW_ONSETS:
            vector.append(aggregate["r:onsets"])
        elif feature == ScsynthFeatures.RAW_PEAK_MEAN:
            vector.append(aggregate["r:peak:mean"])
        elif feature == ScsynthFeatures.RAW_PEAK_STD:
            vector.append(aggregate["r:peak:std"])
        elif feature == ScsynthFeatures.RAW_RMS_MEAN:
            vector.append(aggregate["r:rms:mean"])
        elif feature == ScsynthFeatures.RAW_RMS_STD:
            vector.append(aggregate["r:rms:std"])
        elif feature == ScsynthFeatures.RAW_ROLLOFF_MEAN:
            vector.append(aggregate["r:rolloff:mean"])
        elif feature == ScsynthFeatures.RAW_ROLLOFF_STD:
            vector.append(aggregate["r:rolloff:std"])
        elif feature == ScsynthFeatures.WHITENED_CENTROID_MEAN:
            vector.append(aggregate["w:centroid:mean"])
        elif feature == ScsynthFeatures.WHITENED_CENTROID_STD:
            vector.append(aggregate["w:centroid:std"])
        elif feature == ScsynthFeatures.WHITENED_F0_MEAN:
            vector.append(aggregate["w:f0:mean"])
        elif feature == ScsynthFeatures.WHITENED_F0_STD:
            vector.append(aggregate["w:f0:std"])
        elif feature == ScsynthFeatures.WHITENED_FLATNESS_MEAN:
            vector.append(aggregate["w:flatness:mean"])
        elif feature == ScsynthFeatures.WHITENED_FLATNESS_STD:
            vector.append(aggregate["w:flatness:std"])
        elif feature == ScsynthFeatures.WHITENED_ONSETS:
            vector.append(aggregate["r:onsets"])
        elif feature == ScsynthFeatures.WHITENED_PEAK_MEAN:
            vector.append(aggregate["w:peak:mean"])
        elif feature == ScsynthFeatures.WHITENED_PEAK_STD:
            vector.append(aggregate["w:peak:std"])
        elif feature == ScsynthFeatures.WHITENED_RMS_MEAN:
            vector.append(aggregate["w:rms:mean"])
        elif feature == ScsynthFeatures.WHITENED_RMS_STD:
            vector.append(aggregate["w:rms:std"])
        elif feature == ScsynthFeatures.WHITENED_ROLLOFF_MEAN:
            vector.append(aggregate["w:rolloff:mean"])
        elif feature == ScsynthFeatures.WHITENED_ROLLOFF_STD:
            vector.append(aggregate["w:rolloff:std"])
    return tuple(vector)


def get_vector_size(index_alias: str | None = None) -> int:
    size = 0
    for feature in sorted(get_index_config(index_alias)["features"]):
        if feature in (ScsynthFeatures.RAW_MFCC, ScsynthFeatures.WHITENED_MFCC):
            size += 42
        elif feature in (ScsynthFeatures.RAW_MFCC_13, ScsynthFeatures.WHITENED_MFCC_13):
            size += 13
        elif feature in (ScsynthFeatures.RAW_CHROMA, ScsynthFeatures.WHITENED_CHROMA):
            size += 12
        else:
            size += 1
    return size


def create_scsynth_collection(index_alias: str | None = None) -> Collection:
    collection = Collection(
        name=get_scsynth_collection_name(index_alias),
        schema=CollectionSchema(
            auto_id=False,
            fields=[
                FieldSchema(
                    name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256
                ),
                FieldSchema(name="digest", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="start_frame", dtype=DataType.INT64),
                FieldSchema(name="frame_count", dtype=DataType.INT64),
                FieldSchema(name="f0", dtype=DataType.FLOAT),
                FieldSchema(name="rms", dtype=DataType.FLOAT),
                FieldSchema(name="is_voiced", dtype=DataType.BOOL),
                FieldSchema(
                    name="vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=get_vector_size(index_alias),
                ),
            ],
        ),
    )
    collection.create_index(
        field_name="vector",
        index_params=dict(
            metric_type="L2", index_type="IVF_FLAT", params=dict(nlist=1024)
        ),
    )
    return collection


def get_scsynth_collection(index_alias: str | None = None) -> Collection:
    collection_name = get_scsynth_collection_name(index_alias)
    if not utility.has_collection(collection_name):
        raise ValueError
    return Collection(name=collection_name)


def get_scsynth_collection_name(index_alias: str | None) -> str:
    features = sorted(get_index_config(index_alias)["features"])
    digest = md5("_".join(features).encode()).hexdigest()
    return config.analysis.scsynth_collection_prefix + "_" + digest


def insert_scsynth_entries(
    digest: str,
    entries: Sequence[tuple[int, int, Aggregate]],
    partition_name: str | None = None,
) -> None:
    stride = 1024 * 16
    for index_alias in [
        index_config["alias"] for index_config in config.analysis.scsynth_indices
    ]:
        collection = get_scsynth_collection(index_alias)
        if not utility.has_partition(collection.name, digest):
            collection.create_partition(digest)
        for i in range(0, len(entries), stride):
            data: dict[str, list] = {}
            for start_frame, frame_count, aggregate in entries[i : i + stride]:
                data.setdefault("id", []).append(
                    f"{digest}-{start_frame}-{frame_count}"
                )
                data.setdefault("digest", []).append(digest)
                data.setdefault("start_frame", []).append(start_frame)
                data.setdefault("frame_count", []).append(frame_count)
                data.setdefault("f0", []).append(numpy.float32(aggregate["r:f0:mean"]))
                data.setdefault("rms", []).append(
                    numpy.float32(aggregate["r:rms:mean"])
                )
                data.setdefault("is_voiced", []).append(aggregate["is_voiced"])
                data.setdefault("vector", []).append(
                    aggregate_to_vector(aggregate, index_alias)
                )
            collection.insert(data=list(data.values()), partition_name=partition_name)


def query_scsynth_collection(
    vector: Sequence[float],
    *,
    index_alias: str | None = None,
    is_voiced: bool | None = None,
    limit: int = 10,
    partition_names: Sequence[str] | None = None,
) -> Sequence[Entry]:
    collection = get_scsynth_collection(index_alias)
    expr: str = ""
    if is_voiced is not None:
        expr = "is_voiced == true" if is_voiced else "is_voiced == false"
    entries: list[Entry] = []
    kwargs = dict(
        anns_field="vector",
        consistency_level=2,
        data=[list(vector)],
        expr=expr,
        limit=limit,
        output_fields=["digest", "start_frame", "frame_count"],
        param={"metric_type": "L2", "params": {"nprobe": 1}},
    )
    if expr is not None:
        kwargs["expr"] = expr
    if partition_names:
        kwargs["partition_names"] = partition_names
    for x in collection.search(**kwargs)[0]:
        entries.append(
            dict(
                digest=x.fields["digest"],
                start_frame=x.fields["start_frame"],
                frame_count=x.fields["frame_count"],
                distance=round(x.distance, 3),
            )
        )
    return entries


class WhiteningConfig(TypedDict):
    mean_: list[float]
    n_samples_seen_: int
    var_: list[float]
    scale_: list[float]


def deserialize_whitener(*, redis: redis.Redis) -> StandardScaler:
    key = config.analysis.scsynth_collection_prefix.replace("_", "-") + ":whitening"
    scaler = StandardScaler()
    if redis.exists(key):
        data: WhiteningConfig = json.loads(cast(str, redis.get(key)))
        scaler.mean_ = numpy.array(data["mean_"])
        scaler.n_samples_seen_ = data["n_samples_seen_"]
        scaler.scale_ = numpy.array(data["scale_"])
        scaler.var_ = numpy.array(data["var_"])
    else:
        scaler.mean_ = numpy.array([0.0 for _ in range(SCSYNTH_ANALYSIS_SIZE)])
        scaler.n_samples_seen = 1
        scaler.scale_ = numpy.array([1.0 for _ in range(SCSYNTH_ANALYSIS_SIZE)])
        scaler.var_ = numpy.array([0.0 for _ in range(SCSYNTH_ANALYSIS_SIZE)])
    return scaler


def serialize_whitener(*, redis: redis.Redis, scaler: StandardScaler) -> None:
    key = config.analysis.scsynth_collection_prefix.replace("_", "-") + ":whitening"
    data: WhiteningConfig = dict(
        mean_=scaler.mean_.tolist(),
        n_samples_seen_=int(scaler.n_samples_seen_),
        scale_=scaler.scale_.tolist(),
        var_=scaler.var_.tolist(),
    )
    redis.set(key, json.dumps(data, indent=0, sort_keys=True))


def whiten(
    *, array: numpy.ndarray, redis: redis.Redis, scaler: StandardScaler | None = None
) -> numpy.ndarray:
    return (scaler or deserialize_whitener(redis=redis)).transform(array)
