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

sleep_values=${TRANSFER_SLEEP_MS_LIST:-${TRANSFER_SLEEP_MS:-0}}
read -r -a TRANSFER_SLEEP_MS_VALUES <<< "${sleep_values}"
if (( ${#TRANSFER_SLEEP_MS_VALUES[@]} == 0 )); then
  echo "Set TRANSFER_SLEEP_MS_LIST or TRANSFER_SLEEP_MS in ${CONFIG_PATH}" >&2
  exit 1
fi

python3 - "${TRANSFER_SLEEP_MS_VALUES[@]}" <<'PY'
import math
import re
import sys

pattern = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
seen = set()
for raw_value in sys.argv[1:]:
    if not pattern.fullmatch(raw_value):
        raise SystemExit(f"Invalid transfer sleep value: {raw_value}")
    value = float(raw_value)
    if not math.isfinite(value) or value < 0:
        raise SystemExit(
            f"Transfer sleep must be finite and non-negative: {raw_value}"
        )
    if value in seen:
        raise SystemExit(f"Duplicate transfer sleep value: {raw_value}")
    seen.add(value)
PY

sleep_slug() {
  printf '%s' "$1" | tr '.' 'p'
}

SWEEP_ID=$(date "+%Y%m%d-%H%M%S")
SWEEP_ROOT="${RESULT_ROOT}/${SWEEP_ID}"
mkdir -p "${SWEEP_ROOT}"

python3 - \
  "${SWEEP_ROOT}/sleep_sweep_manifest.json" \
  "${SWEEP_ID}" \
  "${CONFIG_PATH}" \
  "${REPETITIONS}" \
  "${MOONCAKE_TRACE_PATH}" \
  "${TRACE_TIME_SCALE}" \
  "${PREFILL_TP_SIZE}" \
  "${DECODE_TP_SIZE}" \
  "${TRANSFER_SLEEP_MS_VALUES[@]}" <<'PY'
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
    trace_time_scale,
    prefill_tp_size,
    decode_tp_size,
    *sleep_values,
) = sys.argv[1:]
manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "sweep_id": sweep_id,
    "config_path": config_path,
    "trace_path": trace_path,
    "trace_time_scale": float(trace_time_scale),
    "transfer_sleep_ms_values": [float(value) for value in sleep_values],
    "repetitions": int(repetitions),
    "prefill_tp_size": int(prefill_tp_size),
    "decode_tp_size": int(decode_tp_size),
    "case_isolation": "fresh_servers_and_warmup_per_sleep_and_repetition",
    "injection_semantics": (
        "nonblocking delay after physical NIXL completion and before worker "
        "completion reporting"
    ),
}
Path(output_path).write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

echo "Sleep sweep results: ${SWEEP_ROOT}"
echo "Transfer sleep values (ms): ${TRANSFER_SLEEP_MS_VALUES[*]}"
echo "Repetitions: ${REPETITIONS}"

for ((repetition = 1; repetition <= REPETITIONS; repetition++)); do
  ordered_sleeps=()
  if (( repetition % 2 == 1 )); then
    ordered_sleeps=("${TRANSFER_SLEEP_MS_VALUES[@]}")
  else
    for ((index = ${#TRANSFER_SLEEP_MS_VALUES[@]} - 1; index >= 0; index--)); do
      ordered_sleeps+=("${TRANSFER_SLEEP_MS_VALUES[index]}")
    done
  fi

  for sleep_ms in "${ordered_sleeps[@]}"; do
    slug=$(sleep_slug "${sleep_ms}")
    case_id="${SWEEP_ID}-sleep-${slug}-rep-${repetition}"
    case_dir="${SWEEP_ROOT}/sleep-${slug}/rep-${repetition}"

    printf '\n=== Transfer sleep %s ms, repetition %s/%s ===\n' \
      "${sleep_ms}" "${repetition}" "${REPETITIONS}"
    TRANSFER_SLEEP_MS_OVERRIDE="${sleep_ms}" \
    RUN_ID_OVERRIDE="${case_id}" \
    RUN_ROOT_OVERRIDE="${case_dir}" \
      bash "${SCRIPT_DIR}/run_pd_transfer_bench.sh" "${CONFIG_PATH}"
  done
done

echo "Sleep sweep complete: ${SWEEP_ROOT}"
