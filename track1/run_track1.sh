#!/bin/bash

stage=0
stop_stage=
gpu_id=0
source_dir=../../data/RoboHEARD/train
data_dir=../data/track1
split=val
exp_dir=exp
frontend_dir=exp/frontend
finetune_data=../data/finetuning
finetune_config=conf/finetune.yaml
config=conf/model.yaml
asr_model=
manifest_csv=
data_root=
reference_json=
output_dir=
scoring_uem=
hypothesis_json=
score_der=false

. ../utils/parse_options.sh || exit 1
[ -n "$stop_stage" ] || stop_stage=$stage
[ -n "$data_root" ] || data_root=$data_dir/$split
[ -n "$manifest_csv" ] || manifest_csv=$data_root/sessions.csv
[ -n "$reference_json" ] || reference_json=$data_root/reference.seglst.json
[ -n "$output_dir" ] || output_dir=$exp_dir/$split

export CUDA_VISIBLE_DEVICES=$gpu_id
export PYTHONPATH=../..:$PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Step 0: Split released AEC Train into fixed training and validation sessions.
if [ "$stage" -le 0 ] && [ "$stop_stage" -ge 0 ]; then
  python ../utils/prepare_data.py \
    --split-dir "$source_dir" \
    --output-dir "$data_dir" \
    --partition-file conf/session_partition.tsv || exit 1
fi

# Step 1: Prepare training WPE variants and GT-guided GSS.
if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
  python local/frontend.py \
    --stage train \
    --config "$config" \
    --data-dir "$data_dir/train" \
    --output-dir "$frontend_dir/train" || exit 1
fi

# Step 2: Enhance validation audio and run NeMo diarization.
if [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then
  python local/frontend.py \
    --stage diarize \
    --config "$config" \
    --data-dir "$data_root" \
    --output-dir "$frontend_dir/$split" || exit 1
fi

# Step 3: Run GSS with predicted RTTM; no validation labels are used.
if [ "$stage" -le 3 ] && [ "$stop_stage" -ge 3 ]; then
  python local/frontend.py \
    --stage gss \
    --config "$config" \
    --data-dir "$data_root" \
    --output-dir "$frontend_dir/$split" || exit 1
fi

# Step 4: Export training targets and prediction-guided validation windows.
if [ "$stage" -le 4 ] && [ "$stop_stage" -ge 4 ]; then
  python local/prepare_finetune.py \
    --data-dir "$data_dir" \
    --frontend-dir "$frontend_dir" \
    --output-dir "$finetune_data" || exit 1
fi

# Step 5: Fine-tune Qwen3-ASR; select the lowest validation tcpCER.
if [ "$stage" -le 5 ] && [ "$stop_stage" -ge 5 ]; then
  python local/finetune.py \
    --config "$finetune_config" \
    --data-dir "$finetune_data" || exit 1
fi

# Step 6: Transcribe predicted-GSS audio.
if [ "$stage" -le 6 ] && [ "$stop_stage" -ge 6 ]; then
  options=()
  [ -z "$asr_model" ] || options+=(--asr-model "$asr_model")
  bash ../robohead_evaluate_v1.sh \
    --track 1 \
    --stage 0 \
    --gpu-id "$gpu_id" \
    --segments-json "$frontend_dir/$split/segments.json" \
    --manifest-csv "$manifest_csv" \
    --config "$config" \
    --output-dir "$output_dir" \
    "${options[@]}" || exit 1
fi

# Step 7: Score saved predictions when reference transcripts are available.
if [ "$stage" -le 7 ] && [ "$stop_stage" -ge 7 ]; then
  options=()
  [ -z "$scoring_uem" ] || options+=(--scoring-uem "$scoring_uem")
  [ -z "$hypothesis_json" ] || options+=(--hypothesis-json "$hypothesis_json")
  bash ../robohead_evaluate_v1.sh \
    --track 1 \
    --stage 1 \
    --manifest-csv "$manifest_csv" \
    --reference-json "$reference_json" \
    --output-dir "$output_dir" \
    "${options[@]}" || exit 1
  if [ "$score_der" = true ]; then
    python local/score_diarization.py \
      --reference-json "$reference_json" \
      --predicted-rttm "$frontend_dir/$split/pred.rttm" \
      --output-dir "$output_dir" || exit 1
  fi
fi

# Submit a ZIP archive with hypothesis.seglst.json.
# See README.md for the required transcript format.
