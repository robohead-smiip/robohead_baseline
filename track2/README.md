# Track 2 — Streaming Multi-speaker ASR

## Submission

> **Required:** Upload a ZIP archive with the two evaluator-generated files below.

ZIP root: the two UTF-8 files from `exp/val/`.

```text
predict.csv
partial_results.json
```

### Result Format

`predict.csv` contains one fast-pass result per session, with IDs matching `sessions.csv`:

```csv
id,raw_hypos,segments,processing_time_sec
session_001,你好,"[{""speaker"":""speaker_0"",""start_time"":0.02,""end_time"":0.18,""words"":""你好""}]",0.08
```

- `raw_hypos`: complete transcript; empty string for no recognition.
- `segments`: CSV-escaped JSON array using the final-result format; `[]` for no recognition.
- `processing_time_sec`: fast-pass elapsed time in seconds.

`partial_results.json`: session IDs mapped to ordered events and evaluator timestamps, in seconds:

```json
{
  "session_001": {
    "events": [
      {"event": "partial_callback", "time": 100.03, "text": "你"},
      {"event": "partial_callback", "time": 100.14, "text": "你好"},
      {"event": "final_visible", "time": 100.25, "text": "你好",
       "segments": [{"speaker": "speaker_0", "start_time": 0.02, "end_time": 0.18, "words": "你好"}]}
    ],
    "timing": {
      "audio_send_start_time": 100.0, "input_finished_time": 100.2,
      "audio_end_oracle_time": 100.2, "final_visible_time": 100.25,
      "clock": "evaluator_monotonic"
    },
    "capture": {
      "realtime": true, "sample_rate": 16000, "channels": 6,
      "packet_schedule": "immediate_first_100ms", "transport": "in_process_queue",
      "evaluator_pid": 1234, "model_pid": 1234,
      "sender_thread_id": 1001, "decoder_thread_id": 1002,
      "audio_packets": [{"end_sample": 1600, "sent_time": 100.0}, {"end_sample": 3200, "sent_time": 100.1}]
    }
  }
}
```

`scores.json` is local only and is not submitted.

## Inference Interface

**Required:** Implement all methods below in class `Model` in `model.py`. The evaluator calls these interfaces for inference and latency measurement.

```python
class Model:
    def __init__(self):
        ...

    def reset(self) -> None:
        ...

    def set_partial_callback(self, callback) -> None:
        self.callback = callback

    def accept_chunk(self, audio_chunk) -> str:
        ...

    def input_finished(self) -> dict:
        ...
```

| Method                       | Calls                            | Required behavior                                            |
| ---------------------------- | -------------------------------- | ------------------------------------------------------------ |
| `__init__()`               | Once at startup                  | Load configuration and model assets                          |
| `reset()`                  | Before each session in each pass | Clear audio, decoder, speaker and cache state                |
| `set_partial_callback(fn)` | After each reset                 | Register `fn(text: str)`                                   |
| `accept_chunk(chunk)`      | Once per input packet            | Process available audio; return the current transcript       |
| `input_finished()`         | Once after all session audio     | Finish pending decoding and return the complete final result |

**Input:** normalized `float32` arrays, shape `[samples, 6]`, 16 kHz, microphone order 0–5.

- Packets: 1,600 samples (100 ms); the last may be shorter.

### Partial Results

- Call `self.callback(text)` whenever the transcript changes, including corrections.
- Send the **complete current transcript**, not only newly decoded characters.
- Invoke callbacks synchronously within `accept_chunk()` or `input_finished()`; no callbacks after the method returns.
- The evaluator records each callback's text and reception time. The return value of `accept_chunk()` is not a timing event.
- `set_partial_callback()` returns `None`. This does not disable TTFST. Without partial callbacks, TTFST uses final visibility.

### Final Result

The evaluator calls `input_finished()` once after the session ends. Return a JSON-serializable dictionary:

```json
{
  "text": "你好",
  "segments": [
    {"speaker": "speaker_0", "start_time": 0.5, "end_time": 1.2, "words": "你好"}
  ]
}
```

- `text`: complete final session transcript, including earlier decoded speech.
- `segments`: all segments sorted by `start_time`; overlapping speakers are allowed.
- `speaker`: nonempty string without whitespace, consistent within the session.
- `start_time`, `end_time`: acoustic seconds from audio start, with `0 ≤ start < end ≤ duration`.
- `words`: segment text. Concatenated words must equal `text` after normalization.
- Empty recognition: `{"text": "", "segments": []}`. Missing speech counts in tcpCER.

## Baseline

```text
Six-channel packets → ODAS → Nemotron-3-Diarization + Paraformer-zh-streaming
```

- SD: NeMo `SortformerEncLabelModel.forward_streaming_step()`.
- ASR: FunASR `AutoModel.generate()`.
- [Configuration](conf/model.yaml) · [Installation and models](../README.md#installation).

### Validation Results

Val sessions; SD chunk 1.12 s; tcpCER collar 5 s.

| ASR model               | tcpCER (%) | ED P95 (s) | TTFST P95 (s) |   RTF |  RTFx |
| ----------------------- | ---------: | ---------: | ------------: | ----: | ----: |
| Paraformer-zh-streaming |      78.22 |      0.483 |         2.382 | 0.508 | 1.967 |

## Run Stages

Run from `robohead_baseline/track2` in the `robohead` environment. Override paths with command-line options.

| Stage | Operation                                               |
| ----- | ------------------------------------------------------- |
| `0` | Apply fixed conversation boundaries to Track 1 val      |
| `1` | Align and audit all session speech boundaries           |
| `2` | Fast inference and paced replay on all val sessions     |
| `3` | Score accuracy/latency and check final-text consistency |

```bash
cd robohead_baseline/track2
bash run_track2.sh --stage 0
bash run_track2.sh --stage 1
bash run_track2.sh --stage 2 --stop-stage 3 --gpu-id 0
```

- `--stage N`: one stage; `--stop-stage M`: run through M.
- Defaults: `--split val`, `--data-dir ../data/track2`, `--exp-dir exp`.

Stages 2–3 call [robohead_evaluate_v1.sh](../robohead_evaluate_v1.sh):

| Evaluation stage | Operation                                                                      |
| ---------------- | ------------------------------------------------------------------------------ |
| `0`            | Fast inference and real-time replay; write both prediction files               |
| `1`            | Score fast-pass tcpCER with collar 5 s                                         |
| `2`            | Compute ED/TTFST, RTF/RTFx and check latency limits and final-text consistency |

After data preparation, run the shared entry directly:

```bash
bash ../robohead_evaluate_v1.sh \
  --track 2 \
  --stage 0 \
  --stop-stage 2 \
  --gpu-id 0
```

## Prepare Validation

Run Track 1 stage 0 first. Use all 28 val sessions, one recording per session.

To preserve complete turns, [conf/session_crop.tsv](conf/session_crop.tsv) excludes opening counting and recording preparation at session boundaries.

TSV bounds are seconds from the original AEC WAV start:

```text
session_id	speech_start	speech_end
S031	4.89	701.57
```

- `--crop-file FILE`: bounds for every input session; never cut an annotated turn.
- Keep internal pauses and overlapping speech; crop all channels together and shift reference timestamps.
- Context: up to 0.5 s before / 1 s after speech, excluding removed turns.
- Set context with `--leading-seconds` and `--trailing-seconds`.
- `--crop-file ""`: use the first and last annotated speech.

Stage 1 writes VAD/MFA differences to `boundaries.csv` without filtering sessions. Both passes use identical inputs.

### MFA Alignment

- Stage 1 fills `mfa_speech_start` and `mfa_speech_end` using MFA.
- `--align false`: reuse existing aligned bounds.
- tcpCER retains reference timestamps shifted by cropping; MFA affects latency bounds only.

Resources: [MFA](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html), [Mandarin acoustic model](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/acoustic-mandarin_mfa-v3.0.0), [dictionary](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/dictionary-mandarin_china_mfa-v3.0.0), [G2P model](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/g2p-mandarin_china_mfa-v3.0.0).

```bash
export MFA_BIN=/path/to/mfa/environment/bin/mfa
export MFA_G2P_MODEL=/path/to/mandarin_china_mfa.zip
bash run_track2.sh \
  --stage 1 \
  --align true \
  --mfa-dictionary /path/to/mandarin_china_mfa.dict \
  --mfa-acoustic /path/to/mandarin_mfa.zip \
  --mfa-work /path/to/alignment_work
```

- MFA database: `/tmp`; alignment output: `--mfa-work`.
- Alignment defaults: `MFA_BEAM=100`, `MFA_RETRY_BEAM=400`. All sessions must align.

## Evaluation Procedure

| Pass      | Input                    | Delivery                                     | Use                    |
| --------- | ------------------------ | -------------------------------------------- | ---------------------- |
| Fast      | All trimmed val sessions | Sequential 100 ms packets without waiting    | tcpCER, RTF, RTFx      |
| Streaming | The same sessions        | First packet immediately, then 100 ms pacing | ED, TTFST, consistency |

- Both passes use identical packets, model settings and reset behavior.
- The evaluator timestamps partial text and final delivery using a monotonic clock.
- Initialization/reset are outside timing; references are used only for scoring.

## Metrics

```text
ED     = final_visible_time − (audio_send_start_time + audio_duration)
TTFST  = first_stable_partial_time − (audio_send_start_time + mfa_speech_start)
```

| Quantity                      | Definition                                                                                    |
| ----------------------------- | --------------------------------------------------------------------------------------------- |
| `audio_send_start_time`     | First packet enqueue time                                                                     |
| `audio_duration`            | Input sample count / 16,000                                                                   |
| `final_visible_time`        | Time after the final interface return and both threads finish                                 |
| `mfa_speech_start`          | Aligned speech onset relative to WAV start                                                    |
| `first_stable_partial_time` | Earliest event after which the normalized first character always matches the final transcript |

- Endpointing Delay (ED): final delivery delay after WAV end.
- Time to First Stable Token (TTFST): stability of the system's final first character.
- No stable partial or empty final: use final visibility for TTFST.
- One ED/TTFST pair per session; P95 interpolates at index `0.95 × (N − 1)`.
- tcpCER: all val sessions, original reference boundaries, collar **5 s**.
- RTF = total fast-pass time / audio duration; RTFx = 1 / RTF.

**Rank by tcpCER, ascending. Require ED P95 ≤ 1.5 s and TTFST P95 ≤ 4.5 s.**

## Validation

Evaluation stage 2 runs [local/validate.py](local/validate.py). Reject when:

1. ED P95 exceeds 1.5 s or TTFST P95 exceeds 4.5 s.
2. Any session final text differs between fast and paced inference after normalization.
3. Results, input identities or timing records are incomplete or invalid.

`--report-only true` records rejection without a failing exit code; status remains `reject`.

```bash
bash ../robohead_evaluate_v1.sh \
  --track 2 \
  --stage 1 \
  --stop-stage 2 \
  --output-dir exp/val
```

Scores saved predictions. Output: `scores.json`, including metrics, status and rejection reasons.
