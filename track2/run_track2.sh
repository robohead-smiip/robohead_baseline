#!/bin/bash

stage=0
stop_stage=
gpu_id=0
split=val
source_dir=../data/track1/val
crop_file=conf/session_crop.tsv
leading_seconds=0.5
trailing_seconds=1.0
data_dir=../data/track2
exp_dir=exp
mfa_dictionary=models/mfa/mandarin_china_mfa.dict
mfa_acoustic=models/mfa/mandarin_mfa.zip
mfa_work=../data/track2_mfa
align=true
submission_dir=.
config=
manifest_csv=
data_root=
reference_json=
output_dir=
scoring_uem=
warmup_manifest_csv=
report_only=false

. ../utils/parse_options.sh || exit 1
[ -n "$stop_stage" ] || stop_stage=$stage
[ -n "$data_root" ] || data_root=$data_dir/$split
[ -n "$manifest_csv" ] || manifest_csv=$data_root/sessions.csv
[ -n "$reference_json" ] || reference_json=$data_root/reference.seglst.json
[ -n "$output_dir" ] || output_dir=$exp_dir/$split

export CUDA_VISIBLE_DEVICES=$gpu_id
export PYTHONPATH=../..:$PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Step 0: Apply fixed session boundaries after Track 1 stage 0.
if [ "$stage" -le 0 ] && [ "$stop_stage" -ge 0 ]; then
  options=()
  [ -z "$crop_file" ] || options+=(--crop-file "$crop_file")
  python ../utils/prepare_track2_local.py \
    --split-dir "$source_dir" \
    --output-dir "$data_root" \
    --leading-seconds "$leading_seconds" \
    --trailing-seconds "$trailing_seconds" \
    "${options[@]}" || exit 1
fi

# Step 1: Align and audit speech boundaries for every validation session.
if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
  if [ "$align" = true ]; then
    bash ../utils/mfa/run.sh \
      "$data_root" \
      "$mfa_dictionary" \
      "$mfa_acoustic" \
      "$mfa_work/$split" || exit 1
  fi
  python ../utils/latency_local.py \
    --manifest-csv "$manifest_csv" \
    --data-root "$data_root" \
    --output-csv "$data_root/boundaries.csv" || exit 1
fi

# Step 2: Run fast inference and real-time replay on all validation sessions.
if [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then
  options=()
  [ -z "$config" ] || options+=(--config "$config")
  [ -z "$warmup_manifest_csv" ] || options+=(--warmup-manifest-csv "$warmup_manifest_csv")
  bash ../robohead_evaluate_v1.sh \
    --track 2 \
    --stage 0 \
    --gpu-id "$gpu_id" \
    --submission-dir "$submission_dir" \
    --manifest-csv "$manifest_csv" \
    --data-root "$data_root" \
    --output-dir "$output_dir" \
    "${options[@]}" || exit 1
fi

# Step 3: Compute tcpCER and latency, then validate both passes.
if [ "$stage" -le 3 ] && [ "$stop_stage" -ge 3 ]; then
  options=()
  [ -z "$scoring_uem" ] || options+=(--scoring-uem "$scoring_uem")
  bash ../robohead_evaluate_v1.sh \
    --track 2 \
    --stage 1 \
    --stop-stage 2 \
    --manifest-csv "$manifest_csv" \
    --reference-json "$reference_json" \
    --output-dir "$output_dir" \
    --report-only "$report_only" \
    "${options[@]}" || exit 1
fi

# Submit a ZIP archive with predict.csv and partial_results.json at its root.
# See README.md for the required result formats.
