#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_PATH=${1:-"${SCRIPT_DIR}/config.env"}
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${CONFIG_PATH}"

REPETITIONS=${REPETITIONS:-1}
if [[ ! "${REPETITIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPETITIONS must be a positive integer, got ${REPETITIONS}" >&2
  exit 1
fi

scale_values=${TRACE_TIME_SCALES:-${TRACE_TIME_SCALE:-}}
read -r -a TRACE_TIME_SCALE_VALUES <<< "${scale_values}"
if (( ${#TRACE_TIME_SCALE_VALUES[@]} == 0 )); then
  echo "Set TRACE_TIME_SCALES or TRACE_TIME_SCALE in ${CONFIG_PATH}" >&2
  exit 1
fi

python3 - "${TRACE_TIME_SCALE_VALUES[@]}" <<'PY'
import math
import sys

seen = set()
for raw_value in sys.argv[1:]:
    try:
        value = float(raw_value)
    except ValueError as error:
        raise SystemExit(f"Invalid trace time scale: {raw_value}") from error
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f"Trace time scale must be finite and positive: {raw_value}")
    if value in seen:
        raise SystemExit(f"Duplicate trace time scale: {raw_value}")
    seen.add(value)
PY

scale_slug() {
  printf '%s' "$1" | tr '.' 'p'
}

SWEEP_ID=$(date "+%Y%m%d-%H%M%S")
SWEEP_ROOT="${RESULT_ROOT}/${SWEEP_ID}"
mkdir -p "${SWEEP_ROOT}"

python3 - \
  "${SWEEP_ROOT}/sweep_manifest.json" \
  "${SWEEP_ID}" \
  "${CONFIG_PATH}" \
  "${REPETITIONS}" \
  "${MOONCAKE_TRACE_PATH}" \
  "${PREFILL_TP_SIZE}" \
  "${DECODE_TP_SIZE}" \
  "${TRACE_TIME_SCALE_VALUES[@]}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output_path,
    sweep_id,
    config_path,
    repetitions,
    trace_path,
    prefill_tp_size,
    decode_tp_size,
    *scale_values,
) = sys.argv[1:]
manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "sweep_id": sweep_id,
    "config_path": config_path,
    "trace_path": trace_path,
    "trace_time_scale_semantics": "recorded_timestamp_seconds_multiplier",
    "trace_time_scales": [float(value) for value in scale_values],
    "repetitions": int(repetitions),
    "prefill_tp_size": int(prefill_tp_size),
    "decode_tp_size": int(decode_tp_size),
    "case_isolation": "fresh_servers_and_warmup_per_scale_and_repetition",
}
Path(output_path).write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

echo "Sweep results: ${SWEEP_ROOT}"
echo "Trace time scales: ${TRACE_TIME_SCALE_VALUES[*]}"
echo "Repetitions: ${REPETITIONS}"

for ((repetition = 1; repetition <= REPETITIONS; repetition++)); do
  ordered_scales=()
  if (( repetition % 2 == 1 )); then
    ordered_scales=("${TRACE_TIME_SCALE_VALUES[@]}")
  else
    for ((index = ${#TRACE_TIME_SCALE_VALUES[@]} - 1; index >= 0; index--)); do
      ordered_scales+=("${TRACE_TIME_SCALE_VALUES[index]}")
    done
  fi

  for scale in "${ordered_scales[@]}"; do
    slug=$(scale_slug "${scale}")
    case_id="${SWEEP_ID}-scale-${slug}-rep-${repetition}"
    case_dir="${SWEEP_ROOT}/scale-${slug}/rep-${repetition}"

    printf '\n=== Scale %s, repetition %s/%s ===\n' \
      "${scale}" "${repetition}" "${REPETITIONS}"
    TRACE_TIME_SCALE_OVERRIDE="${scale}" \
    RUN_ID_OVERRIDE="${case_id}" \
    RUN_ROOT_OVERRIDE="${case_dir}" \
      bash "${SCRIPT_DIR}/run_pd_transfer_bench.sh" "${CONFIG_PATH}"
  done
done

echo "Sweep complete: ${SWEEP_ROOT}"
