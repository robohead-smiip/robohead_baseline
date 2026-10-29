#!/bin/bash
data=${1:?Usage: run.sh PREPARED_DATA DICTIONARY ACOUSTIC_MODEL WORK_DIR}
dictionary=${2:?Set dictionary path}
acoustic=${3:?Set acoustic model path}
work=${4:?Set work directory}
python_bin=$(command -v python) || exit $?
mfa_bin=$(command -v "${MFA_BIN:-mfa}") || exit $?
mfa_scratch=$(mktemp -d /tmp/robohead_mfa.XXXXXX) || exit $?
trap 'rm -rf "$mfa_scratch"' EXIT
export MFA_ROOT_DIR="$mfa_scratch" TMPDIR="$mfa_scratch"
export PATH="$(dirname -- "$mfa_bin"):$PATH"
"$python_bin" -m robohead_baseline.utils.mfa.align prepare \
  --data-dir "$data" \
  --work-dir "$work" \
  --dictionary "$dictionary" || exit $?
oov=$(cat "$work/oov.txt") || exit $?
if [[ -n "$oov" ]]; then
  "$mfa_bin" g2p \
    "$work/oov.txt" \
    "${MFA_G2P_MODEL:?Set MFA_G2P_MODEL to generate missing pronunciations}" \
    "$work/oov.dict" \
    --clean || exit $?
  cat "$work/oov.dict" >> "$work/dictionary.dict" || exit $?
fi
"$mfa_bin" align \
  "$work/corpus" \
  "$work/dictionary.dict" \
  "$acoustic" \
  "$work/aligned" \
  --beam "${MFA_BEAM:-100}" \
  --retry_beam "${MFA_RETRY_BEAM:-400}" \
  --num_jobs "${MFA_JOBS:-4}" \
  --no_tokenization \
  --clean \
  --output_format long_textgrid || exit $?
"$python_bin" -m robohead_baseline.utils.mfa.align export \
  --data-dir "$data" \
  --work-dir "$work"
