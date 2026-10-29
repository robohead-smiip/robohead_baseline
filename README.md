# RoboHEARD Baselines

> **Submitting:**
>
> Upload a ZIP archive containing:
>
> - **Track 1:** `hypothesis.seglst.json`; see the [result format](track1/README.md#submission).
> - **Track 2:** `predict.csv` and `partial_results.json`; see the [result format and inference interface](track2/README.md#submission).

| Track                      | Pipeline                                                                             | Metrics                   |
| -------------------------- | ------------------------------------------------------------------------------------ | ------------------------- |
| [Track 1](track1/README.md) | wpe_offline + ODAS; Nemotron SD; GPU GSS; Qwen3-ASR-1.7B (pretrained or fine-tuned) | tcpCER                    |
| [Track 2](track2/README.md) | ODAS; Nemotron SD + Paraformer-zh-streaming                                          | tcpCER, ED P95, TTFST P95 |

## Evaluation

Evaluation script: [robohead_evaluate_v1.sh](robohead_evaluate_v1.sh), run from `track1/` or `track2/`.

> **Model verification:** Models from top-ranked participants will be re-evaluated using this script, including evaluation on supplementary datasets.

| Stage | Track 1                         | Track 2                                                     |
| ----- | ------------------------------- | ----------------------------------------------------------- |
| `0` | Transcribe prepared GSS windows | Fast inference and real-time replay on identical sessions   |
| `1` | Normalize text and score tcpCER | Normalize fast-pass text and score tcpCER                   |
| `2` | —                              | Compute ED/TTFST, RTF/RTFx and check final-text consistency |

- Stages 1–2 score saved predictions. Output: `scores.json`.
- Track 2 status stays `accuracy_only` until stage 2 finishes.

| Track | Ranking       | Requirements                                                                       |
| ----- | ------------- | ---------------------------------------------------------------------------------- |
| 1     | Lowest tcpCER | Complete timestamped transcripts; collar 5 s                                       |
| 2     | Lowest tcpCER | ED P95 ≤ 1.5 s; TTFST P95 ≤ 4.5 s; matching normalized final text in both passes |

## Installation

Place `robohead_baseline/` inside [NeMo Speech](https://github.com/NVIDIA-NeMo/Speech). Install from the checkout root; run each track from its own directory.

```bash
git clone https://github.com/NVIDIA-NeMo/Speech.git
cd Speech
conda create -n robohead python=3.12 pip -y
conda activate robohead
```

Set `cuda` and `cupy` below for the [installed driver](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html):

| Stack                        | Settings                                         |
| ---------------------------- | ------------------------------------------------ |
| CUDA 13, driver ≥ 580       | `cuda=cu130; cupy='cupy-cuda13x[ctk]==14.2.0'` |
| CUDA 12 compatibility option | `cuda=cu128; cupy='cupy-cuda12x[ctk]==13.6.0'` |

```bash
cuda=cu130
cupy='cupy-cuda13x[ctk]==14.2.0'
export PIP_CONSTRAINT="$PWD/robohead_baseline/utils/constraints_$cuda.txt"
pip install torch torchaudio --index-url "https://download.pytorch.org/whl/$cuda"
pip install -e '.[asr]' -r robohead_baseline/requirements.txt "$cupy"
pip install --no-build-isolation 'gss @ https://github.com/desh2608/gss/archive/10fad18cae85e2e4342c77421abc70c9c5da23ed.zip'
pip check
```

Build the frontend:

```bash
sudo apt-get install build-essential cmake pkg-config curl libfftw3-dev libconfig-dev libasound2-dev libpulse-dev libsndfile1 ffmpeg
bash robohead_baseline/utils/setup_frontend.sh "$CONDA_PREFIX/opt/odas"
export PATH="$CONDA_PREFIX/opt/odas/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/opt/odas/lib:${LD_LIBRARY_PATH:-}"
python -m robohead_baseline.utils.check_setup --gpu
```

| Library                        | Source                                                     |
| ------------------------------ | ---------------------------------------------------------- |
| NeMo Speech                    | [NVIDIA-NeMo/Speech](https://github.com/NVIDIA-NeMo/Speech) |
| Qwen3-ASR                      | [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)     |
| Paraformer                     | [modelscope/FunASR](https://github.com/modelscope/FunASR)   |
| ODAS                           | [introlab/odas](https://github.com/introlab/odas)           |
| GPU GSS                        | [desh2608/gss](https://github.com/desh2608/gss)             |
| MeetEval                       | [fgnt/meeteval](https://github.com/fgnt/meeteval)           |
| MFA, Track 2 speech boundaries | [Alignment setup](track2/README.md#mfa-alignment)           |

## Models

Download checkpoints into each track’s `models/` directory:

```bash
python -m robohead_baseline.utils.download_models --track all
```

| Model                               | Download                                                                                                                                                                              | Directory under `robohead_baseline/`                                             |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Nemotron-3-Diarization              | [Hugging Face](https://huggingface.co/nvidia/Nemotron-3-Diarization)                                                                                                                   | `track1/models/Nemotron-3-Diarization`, `track2/models/Nemotron-3-Diarization` |
| Qwen3-ASR-1.7B (default)            | [Hugging Face](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)                                                                                                                             | `track1/models/Qwen3-ASR-1.7B`                                                   |
| Qwen3-ASR-1.7B Finetuned (optional) | [Hugging Face](https://huggingface.co/SMIIP-lab/Qwen3-ASR-FT-robohead)                                                                                                                 | `track1/models/Qwen3-ASR-FT-robohead`                                            |
| Paraformer-zh-streaming             | [Hugging Face](https://huggingface.co/funasr/paraformer-zh-streaming), [ModelScope](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online) | `track2/models/Paraformer-zh-streaming`                                          |

- Model paths: `conf/model.yaml`, relative to each track directory.
- Track 1 training: `conf/finetune.yaml`; output: `exp/qwen3_asr/`.

## Data Preparation

Split the released Train into fixed **train** and **val** partitions.

```text
RoboHEARD/
└── train/
    ├── Audio_AEC/S001.wav
    ├── DOC_AEC/S001.TextGrid
    ├── Video/
    └── speakers.csv
```

- `Audio_AEC/`: six-channel 16 kHz PCM16 WAV after echo cancellation. Both tracks use AEC audio only.
- `DOC_AEC/`: matching AEC TextGrid for each recording. Keep only `P` + digits speaker tiers; exclude the robot tier and other annotation tiers.
- Remove `<...>` tags and intervals with no letters or numbers after cleaning. Keep spoken words, interjections and wake words.
- Exclude `S161`, a duplicate of `S050`, and the two all-zero-audio annotations listed in `utils/splits/annotation_exclusions.json`.
- Six-channel WAVs and six synchronized mono WAVs are also supported.

The provided [session partition](track1/conf/session_partition.tsv) is a preparation input and keeps train/val speakers disjoint:

| Partition | Sessions | Audio hours |
| --------- | -------: | ----------: |
| train     |      135 |      24.893 |
| val       |       28 |       5.044 |

Prepare once for both tracks:

```bash
cd robohead_baseline/track1
bash run_track1.sh --stage 0 --source-dir /path/to/RoboHEARD/train
```

Run the pipelines:

```bash
bash run_track1.sh --stage 1 --stop-stage 7

cd ../track2
bash run_track2.sh --stage 0
bash run_track2.sh --stage 1 --align true
bash run_track2.sh --stage 2 --stop-stage 3
```

- Track 1 output: `data/track1/{train,val}/`.
- Track 2: [crop all 28 val sessions](track2/README.md#prepare-validation) for accuracy and latency evaluation.

## Data Formats

All text files use UTF-8; timestamps are seconds from the start of each WAV.

### Session Manifest

```csv
id,audio_filepath,audio_duration_sec,mfa_speech_start,mfa_speech_end,evaluation_unit
session_001,audio/session_001.wav,40.0,0.5,39.0,session
```

| Field                                    | Requirement                                                      |
| ---------------------------------------- | ---------------------------------------------------------------- |
| `id`                                   | Unique recording ID without whitespace                           |
| `audio_filepath`                       | One six-channel AEC WAV, relative to `--data-root` or absolute |
| `audio_duration_sec`                   | Sample count / 16,000                                            |
| `mfa_speech_start`, `mfa_speech_end` | Track 2 aligned speech bounds; blank for Track 1                 |
| `evaluation_unit`                      | `session`; each supplied WAV is an independent request         |
| `audio_channels`, optional             | Compatibility with a JSON list of six mono files; normally blank |

For existing audio lists, `wav.scp` contains `ID /path/to/session.wav` (six-channel AEC audio):

```bash
python -m robohead_baseline.utils.prepare_manifest \
  --wav-scp /path/to/wav.scp \
  --data-root /path/to/data \
  --output-csv /path/to/sessions.csv
```

`--onset-csv FILE` imports aligned speech bounds. Reference text and aligned bounds are never passed to the model.

### Timestamped Transcripts

References and predictions use a MeetEval SegLST JSON array:

```json
[
  {"session_id": "session_001", "speaker": "speaker_0", "start_time": 0.5, "end_time": 1.2, "words": "你好"}
]
```

- String IDs; consistent speaker IDs within each WAV.
- `0 ≤ start_time < end_time ≤ duration`; overlapping speakers allowed.
- Submission requirements: each track's README.

### tcpCER

**tcpCER** (time-constrained concatenated minimum-permutation character error rate) compares speaker-wise concatenated transcripts under temporal constraints, selecting the speaker mapping with the fewest character errors.

`tcpCER (%) = 100 × (S + D + I) / N`

`S`, `D`, `I`: character substitutions, deletions and insertions; `N`: reference characters. Lower is better.

- NFKC normalization, lowercase, remove `<...>` tags, retain letters and numbers.
- Character-level MeetEval `tcpwer`, collar **5 s**.
- Original reference segment boundaries: `character_based`; hypothesis: `character_based_points`.
- Aggregate errors / reference characters over the complete evaluation manifest.

