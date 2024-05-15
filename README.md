# Praetor

A distributed concatenative synthesis engine.

## Setup

Setup `git lfs`: https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage

Build containers, bring them online:

```
make up
```

Install Python deps (for local CLI usage):

```
pip install -e .
```

## CLI

Get help:

```
python -m praetor --help
```

### Load audio into the corpus

Batch audio upload via URLs:

```
python -m praetor \
    audio-batch \
        s3://test-source/bloodborne-dialog.mp3 \
        s3://test-source/bloodborne-enemy-dialog.mp3 \
        s3://test-source/dark-souls-3-part1.mp3 \
        s3://test-source/dark-souls-3-part2.mp3 \
        s3://test-source/dark-souls-3-part3.mp3 \
        s3://test-source/elden-ring-all-tarnshed-voice-lines.mp3 \
        s3://test-source/elden-ring-every-howl-of-shabiri.mp3 \
        s3://test-source/elden-ring-ost.mp3
```

Direct upload an audio file:

```
python -m praetor \
    audio-upload \
        tests/recordings/ibn-arabi-44100.wav
```

### Search the corpus

Upload audio, analyze via scsynth, and query:

```
python -m praetor \
    query-scsynth-upload \
        tests/recordings/ibn-arabi-44100-1s.wav
```

Query via scsynth vector:

```
python -m praetor \
    query-scsynth -- \
        65.12738627115885 \
        -53.331234178235455 \
        0.07526881720430108 \
        0.4984107557483899 \
        0.015506617484554168 \
        0.14230795477026253 \
        -0.02995609644279685 \
        0.09624801512046527 \
        0.1683859033610231 \
        0.2544622818628947 \
        0.26708049719692556 \
        0.25776375766082477 \
        0.3831833788464146 \
        0.2761819186390087 \
        0.3476886949552003 \
        0.23867011438774807
```

Upload audio, analyze via AST, and query:

```
python -m praetor \
    query-ast-upload \
        tests/recordings/ibn-arabi-44100-1s.wav
```

### Download audio segments

Fetch audio segment:

```
python -m praetor \
    audio-fetch \
        ade95a26d07c28af966ec474445379cae82d96c2bc30913b93e972702d9fb208 \
        82756096 \
        24064 \
        downloaded-audio.wav
```
