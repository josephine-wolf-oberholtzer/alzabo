import csv
import logging
import tempfile
from pathlib import Path
from typing import Sequence

import timm
import torch
import torch.nn as nn
import torchaudio
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility
from timm.models.layers import to_2tuple, trunc_normal_
from torch.cuda.amp import autocast

from ..config import config
from .audio import get_duration, transcode_audio
from .milvus import Entry
from .utils import timer

logger = logging.getLogger(__name__)

LABEL_DIM = 527
INPUT_TDIM = 1024


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram

    """

    def __init__(
        self,
        label_dim=LABEL_DIM,
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=INPUT_TDIM,
    ):
        super(ASTModel, self).__init__()
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        self.v = timm.create_model(
            "vit_deit_base_distilled_patch16_384", pretrained=False
        )
        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches**0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim),
            nn.Linear(self.original_embedding_dim, label_dim),
        )
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        # the linear projection layer
        new_proj = torch.nn.Conv2d(
            1,
            self.original_embedding_dim,
            kernel_size=(16, 16),
            stride=(fstride, tstride),
        )
        self.v.patch_embed.proj = new_proj
        new_pos_embed = nn.Parameter(
            torch.zeros(
                1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim
            )
        )
        self.v.pos_embed = new_pos_embed
        trunc_normal_(self.v.pos_embed, std=0.02)

    @autocast()
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        x = self.mlp_head(x)
        return x

    def get_shape(
        self, fstride: int, tstride: int, input_fdim: int = 128, input_tdim: int = 1024
    ) -> tuple[int, int]:
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(
            1,
            self.original_embedding_dim,
            kernel_size=(16, 16),
            stride=(fstride, tstride),
        )
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim


def load_labels() -> list[str]:
    with config.ast.labels_path.open("r") as file_pointer:
        reader = csv.reader(file_pointer, delimiter=",")
        next(reader)  # consume column header row
        return [row[-1] for row in reader]


def load_model() -> torch.nn.DataParallel:
    try:
        checkpoint = torch.load(config.ast.checkpoint_path, map_location="cuda")
        cuda_enabled = True
    except RuntimeError:
        checkpoint = torch.load(config.ast.checkpoint_path, map_location="cpu")
        cuda_enabled = False
    audio_model = torch.nn.DataParallel(ASTModel(), device_ids=[0])
    audio_model.load_state_dict(checkpoint)
    if cuda_enabled:
        audio_model = audio_model.to(torch.device("cuda:0"))
    audio_model.eval()
    return audio_model


def extract_features(
    audio_path: Path,
    *,
    from_seconds: float | None = None,
    mel_bins: int = 128,
    target_length: int = 1024,
    to_seconds: float | None = None,
) -> torch.Tensor:
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "output.wav"
        transcode_audio(
            audio_path,
            output_path,
            sample_rate=16000,
            from_seconds=from_seconds,
            to_seconds=to_seconds,
        )
        waveform, sample_rate = torchaudio.load(output_path)
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=mel_bins,
        dither=0.0,
        frame_shift=10,
    )
    if (p := target_length - fbank.shape[0]) > 0:
        fbank = torch.nn.ZeroPad2d((0, 0, 0, p))(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank


def analyze(
    audio_path: Path,
    model: torch.nn.DataParallel | None,
    *,
    from_seconds: float | None = None,
    to_seconds: float | None = None,
) -> tuple[float, ...]:
    model_: torch.nn.DataParallel = model or load_model()
    with timer(logger, "Extracted features in " + "{time:.03f} seconds"):
        features = extract_features(
            audio_path, mel_bins=128, from_seconds=from_seconds, to_seconds=to_seconds
        ).expand(1, INPUT_TDIM, 128)
    with timer(logger, "Sent to device in " + "{time:.03f} seconds"):
        try:
            features = features.to(torch.device("cuda:0"))
        except RuntimeError:
            features = features.to(torch.device("cpu"))
    with timer(logger, "Modeled in " + "{time:.03f} seconds"):
        with torch.no_grad():
            with autocast():
                output = torch.sigmoid(model_.forward(features))
    return output.data.cpu().numpy()[0].tolist()


def partition(
    audio_path: Path, model: torch.nn.DataParallel, hop_ms: int, length_ms: int
) -> Sequence[tuple[int, int, tuple[float, ...]]]:
    """
    Partition and analyze an audio file in one go.

    There is no intermediary analysis step like with scsynth.
    """
    entries: list[tuple[int, int, tuple[float, ...]]] = []
    total_time = get_duration(audio_path)
    start_time = 0.0
    while (stop_time := start_time + (length_ms / 1000)) <= total_time:
        logger.info(
            f"Partitioned {stop_time / total_time * 100.0:.03f}% ({stop_time:.03f}s of {total_time:.03f}s)"
        )
        start_frame = int(start_time * 48000)
        stop_frame = int(stop_time * 48000)
        frame_count = stop_frame - start_frame
        entries.append(
            (
                start_frame,
                frame_count,
                analyze(
                    audio_path, model, from_seconds=start_time, to_seconds=stop_time
                ),
            )
        )
        start_time += hop_ms / 1000
    return entries


def get_vector_size() -> int:
    return 527


def create_ast_partition(digest: str) -> None:
    collection = get_or_create_ast_collection()
    if not utility.has_partition(collection.name, digest):
        collection.create_partition(digest)


def create_ast_collection() -> Collection:
    collection = Collection(
        name=get_ast_collection_name(),
        schema=CollectionSchema(
            auto_id=False,
            fields=[
                FieldSchema(
                    name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256
                ),
                FieldSchema(name="digest", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="start_frame", dtype=DataType.INT64),
                FieldSchema(name="frame_count", dtype=DataType.INT64),
                FieldSchema(
                    name="vector", dtype=DataType.FLOAT_VECTOR, dim=get_vector_size()
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


def get_ast_collection() -> Collection:
    collection_name = get_ast_collection_name()
    if utility.has_collection(collection_name):
        return Collection(name=collection_name)
    raise ValueError


def get_or_create_ast_collection() -> Collection:
    # TODO: Separate get and create! No implicit behavior.
    collection_name = get_ast_collection_name()
    if utility.has_collection(collection_name):
        return Collection(name=collection_name)
    return create_ast_collection()


def get_ast_collection_name() -> str:
    return config.analysis.ast_collection_prefix


def insert_ast_entries(
    digest: str,
    entries: Sequence[tuple[int, int, tuple[float, ...]]],
    partition_name: str | None = None,
) -> None:
    collection = get_or_create_ast_collection()
    stride = 1024
    for i in range(0, len(entries), stride):
        data: dict[str, list] = {}
        for start_frame, frame_count, vector in entries[i : i + stride]:
            data.setdefault("id", []).append(f"{digest}-{start_frame}-{frame_count}")
            data.setdefault("digest", []).append(digest)
            data.setdefault("start_frame", []).append(start_frame)
            data.setdefault("frame_count", []).append(frame_count)
            data.setdefault("vector", []).append(vector)
        collection.insert(data=list(data.values()), partition_name=partition_name),


def query_ast_collection(
    vector: Sequence[float],
    limit: int = 10,
    partition_names: Sequence[str] | None = None,
) -> Sequence[Entry]:
    collection = get_or_create_ast_collection()
    entries: list[Entry] = []
    kwargs = dict(
        anns_field="vector",
        consistency_level=2,
        data=[list(vector)],
        limit=limit,
        output_fields=["digest", "start_frame", "frame_count"],
        param={"metric_type": "L2", "params": {"nprobe": 1}},
    )
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
