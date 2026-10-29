#!/bin/bash

# ============================================================================
# RoboHEARD Inference and Evaluation v1
# Run from track1/ or track2/ after data preparation.
# ============================================================================

# Stages: 0 = inference, 1 = tcpCER, 2 = Track 2 latency and consistency.
# Track 1 runs stages 0-1; Track 2 runs stages 0-2.
# --stage N runs one stage; --stop-stage M runs through M.

# Example: Track 2 inference and scoring.
# bash ../robohead_evaluate_v1.sh \
#   --track 2 \
#   --stage 0 \
#   --stop-stage 2 \
#   --data-root ../data/track2/val \
#   --output-dir exp/val

# Rank by tcpCER (collar 5 s). Track 2 also requires matching final texts,
# ED P95 <= 1.5 s and TTFST P95 <= 4.5 s. 

# Defaults; override with --name value.
track=2                            # 1: offline; 2: streaming
stage=0                            # First evaluation stage: 0, 1 or 2
stop_stage=                        # Last stage; empty = stage
split=val                          # Partition for default paths
gpu_id=0                           # GPU selection for inference

# Shared paths.
data_root=                         ### TODO: prepared data directory; empty -> ../data/track$track/$split
manifest_csv=                      # Input session manifest; empty -> $data_root/sessions.csv
reference_json=                    # Input scoring references; empty -> $data_root/reference.seglst.json
output_dir=                        ### TODO: prediction and score directory; empty -> exp/$split
config=                            # Optional model YAML

# Track 1 inputs.
segments_json=                     # Prepared GSS windows; empty -> exp/frontend/$split/segments.json
asr_model=                         # Optional Qwen3-ASR checkpoint override

# Track 2 inputs.
submission_dir=.                   ### TODO: directory containing model.py; . uses the track baseline
warmup_manifest_csv=               # Optional warm-up manifest; separate session IDs

# Scoring options.
hypothesis_json=                   # Track 1 SegLST input; empty -> $output_dir/hypothesis.seglst.json
scoring_uem=                       # Optional UEM file: session_id 1 start end; accuracy only
report_only=false                  # true: keep reject status but return exit code 0

. ../utils/parse_options.sh || exit 1
case "$track" in
  1|2) ;;
  *) echo "Expected --track 1 or 2" >&2; exit 1 ;;
esac
[ -n "$stop_stage" ] || stop_stage=$stage
if [[ ! "$stage" =~ ^[0-2]$ || ! "$stop_stage" =~ ^[0-2]$ ]] ||
   [ "$stage" -gt "$stop_stage" ] || [ "$stop_stage" -gt "$track" ]; then
  echo "Expected stages 0-$track with stage <= stop_stage" >&2
  exit 1
fi
[ -n "$data_root" ] || data_root=../data/track$track/$split
[ -n "$manifest_csv" ] || manifest_csv=$data_root/sessions.csv
[ -n "$reference_json" ] || reference_json=$data_root/reference.seglst.json
[ -n "$output_dir" ] || output_dir=exp/$split
[ -n "$segments_json" ] || segments_json=exp/frontend/$split/segments.json

export CUDA_VISIBLE_DEVICES=$gpu_id
export PYTHONPATH=../..:$PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "RoboHEARD evaluation v1: track $track, stages $stage-$stop_stage"

# Step 0: Run inference and save predictions.
if [ "$stage" -le 0 ] && [ "$stop_stage" -ge 0 ]; then
  if [ "$track" = 1 ]; then
    [ -n "$config" ] || config=conf/model.yaml
    options=()
    [ -z "$asr_model" ] || options+=(--model-path "$asr_model")
    python ../track1/local/recognize.py \
      --segments-json "$segments_json" \
      --manifest-csv "$manifest_csv" \
      --config "$config" \
      --output-dir "$output_dir" \
      "${options[@]}" || exit 1
  else
    options=()
    [ -z "$config" ] || options+=(--config "$config")
    [ -z "$warmup_manifest_csv" ] || options+=(--warmup-manifest-csv "$warmup_manifest_csv")
    python ../track2/local/decode.py \
      --submission-dir "$submission_dir" \
      --manifest-csv "$manifest_csv" \
      --data-root "$data_root" \
      --output-dir "$output_dir" \
      "${options[@]}" || exit 1
  fi
fi

# Step 1: Score tcpCER; write scores.json.
if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
  options=()
  [ -z "$scoring_uem" ] || options+=(--scoring-uem "$scoring_uem")
  [ -z "$hypothesis_json" ] || options+=(--hypothesis-json "$hypothesis_json")
  python ../utils/evaluate.py \
    --track "$track" \
    --phase accuracy \
    --manifest-csv "$manifest_csv" \
    --reference-json "$reference_json" \
    --output-dir "$output_dir" \
    "${options[@]}" || exit 1
fi

# Step 2: Score Track 2 latency/speed and check consistency; update scores.json.
if [ "$track" = 2 ] && [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then
  options=()
  [ "$report_only" != true ] || options+=(--report-only)
  python ../utils/evaluate.py \
    --track 2 \
    --phase latency \
    --manifest-csv "$manifest_csv" \
    --reference-json "$reference_json" \
    --output-dir "$output_dir" \
    "${options[@]}" || exit 1
fi
