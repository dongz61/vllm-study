#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-"tests/pd_transfer/generalization_config.env"}
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${CONFIG_PATH}"

# DATASET_PATH supersedes the original BurstGPT-specific name. Keep the
# fallback so existing local configs continue to work unchanged.
DATASET_PATH=${DATASET_PATH:-${BURSTGPT_DATASET_PATH:-}}
DATASET_LOADER=${DATASET_LOADER:-burstgpt}
WORKLOAD_NAME=${WORKLOAD_NAME:-burstgpt-v1.1}
WORKLOAD_SLUG=${WORKLOAD_SLUG:-burstgpt}
DATASET_SOURCE_FORMAT=${DATASET_SOURCE_FORMAT:-burstgpt-v1.1-csv}
MOONCAKE_BLOCK_SIZE=${MOONCAKE_BLOCK_SIZE:-512}
MOONCAKE_TOKEN_SEED=${MOONCAKE_TOKEN_SEED:-0}
# paired runs OFF and ON in alternating order across repetitions.  Single
# variant modes are useful for smoke tests and workload characterization.
VARIANT_MODE=${VARIANT_MODE:-paired}
# Extra non-blocking delay after NIXL receive completion and before the Decode
# worker reports the remote KV as ready. Keep the default at zero so existing
# OFF/ON generalization runs retain their original behavior.
TRANSFER_DELAY_MS_LIST=${TRANSFER_DELAY_MS_LIST:-0}

require_value() {
  local name=$1
  if [[ -z "${!name:-}" ]]; then
    echo "Required config value is missing: ${name}" >&2
    exit 1
  fi
}

require_positive_integer() {
  local name=$1
  local value=${!name:-}
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer, got: ${value}" >&2
    exit 1
  fi
}

require_nonnegative_integer() {
  local name=$1
  local value=${!name:-}
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "${name} must be a non-negative integer, got: ${value}" >&2
    exit 1
  fi
}

for required_name in \
  MODEL SERVED_MODEL_NAME PREFILL_DEVICES DECODE_DEVICES TP_SIZE HOST \
  PREFILL_PORT DECODE_PORT PROXY_PORT PREFILL_SIDE_CHANNEL_PORT \
  DECODE_SIDE_CHANNEL_PORT PREFILL_ENGINE_ID DECODE_ENGINE_ID \
  DATASET_PATH DATASET_LOADER WORKLOAD_NAME WORKLOAD_SLUG \
  DATASET_SOURCE_FORMAT \
  REQUEST_RATES NUM_PROMPTS REPETITIONS RESULT_ROOT; do
  require_value "${required_name}"
done

for integer_name in \
  TP_SIZE PREFILL_PORT DECODE_PORT PROXY_PORT PREFILL_SIDE_CHANNEL_PORT \
  DECODE_SIDE_CHANNEL_PORT REPETITIONS; do
  require_positive_integer "${integer_name}"
done
require_nonnegative_integer NUM_PROMPTS

if ! [[ "${WORKLOAD_SLUG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "WORKLOAD_SLUG may contain only letters, numbers, dot, underscore, and dash" >&2
  exit 1
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi

if [[ "${DATASET_LOADER}" != "burstgpt" \
    && "${DATASET_LOADER}" != "mooncake" ]]; then
  echo "DATASET_LOADER must be burstgpt or mooncake" >&2
  exit 1
fi
if [[ "${DATASET_LOADER}" == "mooncake" ]]; then
  require_positive_integer MOONCAKE_BLOCK_SIZE
fi

validate_compatible_dataset() {
  python3 - "${DATASET_PATH}" "${DATASET_LOADER}" \
    "${MOONCAKE_BLOCK_SIZE}" "${NUM_PROMPTS}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
loader = sys.argv[2]
block_size = int(sys.argv[3])
num_prompts = int(sys.argv[4])

if loader == "mooncake":
    count = 0
    previous_timestamp = None
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            count += 1
            record = json.loads(line)
            timestamp = float(record["timestamp"])
            input_length = int(record["input_length"])
            output_length = int(record["output_length"])
            hash_ids = record["hash_ids"]
            if timestamp < 0 or (
                previous_timestamp is not None
                and timestamp < previous_timestamp
            ):
                raise SystemExit(
                    f"line {line_number}: timestamps must be non-decreasing"
                )
            if input_length <= 0 or output_length <= 0:
                raise SystemExit(
                    f"line {line_number}: lengths must be positive"
                )
            expected = math.ceil(input_length / block_size)
            if not isinstance(hash_ids, list) or len(hash_ids) != expected:
                raise SystemExit(
                    f"line {line_number}: expected {expected} hash IDs, "
                    f"got {len(hash_ids) if isinstance(hash_ids, list) else 0}"
                )
            previous_timestamp = timestamp
    if count < num_prompts:
        raise SystemExit(
            f"Mooncake trace has {count} requests, fewer than {num_prompts}"
        )
    print(
        f"Validated Mooncake JSONL: {path} "
        f"({count} requests, block_size={block_size})"
    )
    raise SystemExit(0)

with path.open("r", newline="", encoding="utf-8-sig") as file:
    reader = csv.reader(file)
    try:
        header = next(reader)
    except StopIteration:
        raise SystemExit(f"Dataset is empty: {path}")

expected = {1: "Model", 2: "Request tokens", 3: "Response tokens"}
problems = [
    f"column {index + 1} must be {name!r}, got "
    f"{header[index] if index < len(header) else '<missing>'!r}"
    for index, name in expected.items()
    if index >= len(header) or header[index] != name
]
if problems:
    raise SystemExit(
        "CSV is incompatible with the vLLM 0.11 positional BurstGPT loader. "
        "Use BurstGPT v1.1 for this loader.\n"
        + "\n".join(problems)
    )

print(f"Validated vLLM 0.11 BurstGPT-compatible CSV: {path}")
print(f"Columns: {header}")
PY
}

validate_compatible_dataset

if [[ -n "${MAX_CONCURRENCY:-}" ]]; then
  require_positive_integer MAX_CONCURRENCY
fi
if [[ "${SAVE_DETAILED:-1}" != "0" && "${SAVE_DETAILED:-1}" != "1" ]]; then
  echo "SAVE_DETAILED must be 0 or 1" >&2
  exit 1
fi
if [[ "${IGNORE_EOS:-1}" != "0" && "${IGNORE_EOS:-1}" != "1" ]]; then
  echo "IGNORE_EOS must be 0 or 1" >&2
  exit 1
fi
if [[ "${RUN_DIAGNOSTIC_TRACE:-1}" != "0" \
      && "${RUN_DIAGNOSTIC_TRACE:-1}" != "1" ]]; then
  echo "RUN_DIAGNOSTIC_TRACE must be 0 or 1" >&2
  exit 1
fi
DIAGNOSTIC_REPETITION_COUNT=0
if [[ "${RUN_DIAGNOSTIC_TRACE:-1}" == "1" ]]; then
  # Preserve the existing REPETITIONS setting as the default while allowing
  # trace-enabled cases to be repeated independently from performance cases.
  DIAGNOSTIC_REPETITIONS=${DIAGNOSTIC_REPETITIONS:-${REPETITIONS}}
  require_positive_integer DIAGNOSTIC_REPETITIONS
  DIAGNOSTIC_REPETITION_COUNT=${DIAGNOSTIC_REPETITIONS}
fi
case "${VARIANT_MODE}" in
  off|on|paired)
    ;;
  *)
    echo "VARIANT_MODE must be off, on, or paired; got: ${VARIANT_MODE}" >&2
    exit 1
    ;;
esac

read -r -a TRANSFER_DELAY_VALUES <<< "${TRANSFER_DELAY_MS_LIST}"
if (( ${#TRANSFER_DELAY_VALUES[@]} == 0 )); then
  echo "TRANSFER_DELAY_MS_LIST must contain at least one value" >&2
  exit 1
fi
python3 - "TRANSFER_DELAY_MS_LIST" "${TRANSFER_DELAY_VALUES[@]}" <<'PY'
import math
import sys

name, *raw_values = sys.argv[1:]
seen = set()
for raw_value in raw_values:
    try:
        value = float(raw_value)
    except ValueError as error:
        raise SystemExit(f"Invalid transfer delay in {name}: {raw_value}") from error
    if not math.isfinite(value) or value < 0:
        raise SystemExit(
            f"Transfer delay must be finite and non-negative: {raw_value}"
        )
    if value in seen:
        raise SystemExit(f"Duplicate transfer delay in {name}: {raw_value}")
    seen.add(value)
PY

DIAGNOSTIC_TRANSFER_DELAY_VALUES=()
if [[ "${RUN_DIAGNOSTIC_TRACE:-1}" == "1" ]]; then
  diagnostic_transfer_delays=${DIAGNOSTIC_TRANSFER_DELAY_MS_LIST:-${TRANSFER_DELAY_VALUES[0]}}
  read -r -a DIAGNOSTIC_TRANSFER_DELAY_VALUES <<< "${diagnostic_transfer_delays}"
  if (( ${#DIAGNOSTIC_TRANSFER_DELAY_VALUES[@]} == 0 )); then
    echo "DIAGNOSTIC_TRANSFER_DELAY_MS_LIST must contain at least one value" >&2
    exit 1
  fi
  python3 - "DIAGNOSTIC_TRANSFER_DELAY_MS_LIST" \
    "${DIAGNOSTIC_TRANSFER_DELAY_VALUES[@]}" <<'PY'
import math
import sys

name, *raw_values = sys.argv[1:]
seen = set()
for raw_value in raw_values:
    try:
        value = float(raw_value)
    except ValueError as error:
        raise SystemExit(f"Invalid transfer delay in {name}: {raw_value}") from error
    if not math.isfinite(value) or value < 0:
        raise SystemExit(
            f"Transfer delay must be finite and non-negative: {raw_value}"
        )
    if value in seen:
        raise SystemExit(f"Duplicate transfer delay in {name}: {raw_value}")
    seen.add(value)
PY
fi

if [[ "${VARIANT_MODE}" == "paired" ]] \
    && (( ${#TRANSFER_DELAY_VALUES[@]} > 1 \
          || ${#DIAGNOSTIC_TRANSFER_DELAY_VALUES[@]} > 1 )); then
  echo "Transfer-delay sweeps require VARIANT_MODE=off or VARIANT_MODE=on" >&2
  echo "The OFF/ON aggregator does not pair results across multiple delays" >&2
  exit 1
fi
if [[ "${NUM_PROMPTS}" == "0" \
    && "${RUN_DIAGNOSTIC_TRACE:-1}" != "1" ]]; then
  echo "NUM_PROMPTS=0 skips performance runs and requires RUN_DIAGNOSTIC_TRACE=1" >&2
  exit 1
fi

read -r -a REQUEST_RATE_VALUES <<< "${REQUEST_RATES}"
if (( ${#REQUEST_RATE_VALUES[@]} == 0 )); then
  echo "REQUEST_RATES must contain at least one value" >&2
  exit 1
fi

declare -A SEEN_REQUEST_RATES=()
for request_rate in "${REQUEST_RATE_VALUES[@]}"; do
  if ! [[ "${request_rate}" == "inf" \
      || "${request_rate}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    echo "Invalid request rate: ${request_rate}" >&2
    exit 1
  fi
  if [[ "${request_rate}" != "inf" ]] \
      && ! python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)" \
        "${request_rate}"; then
    echo "Request rate must be positive: ${request_rate}" >&2
    exit 1
  fi
  if [[ "${DATASET_LOADER}" == "mooncake" \
      && "${request_rate}" == "inf" ]]; then
    echo "Mooncake arrival-rate scale must be finite" >&2
    exit 1
  fi
  if [[ -n "${SEEN_REQUEST_RATES[${request_rate}]:-}" ]]; then
    echo "Duplicate request rate: ${request_rate}" >&2
    exit 1
  fi
  SEEN_REQUEST_RATES["${request_rate}"]=1
done

DIAGNOSTIC_RATE_VALUES=()
if [[ "${RUN_DIAGNOSTIC_TRACE:-1}" == "1" ]]; then
  diagnostic_rates=${DIAGNOSTIC_REQUEST_RATES:-${DIAGNOSTIC_REQUEST_RATE:-${REQUEST_RATE_VALUES[0]}}}
  read -r -a DIAGNOSTIC_RATE_VALUES <<< "${diagnostic_rates}"
  if (( ${#DIAGNOSTIC_RATE_VALUES[@]} == 0 )); then
    echo "DIAGNOSTIC_REQUEST_RATES must contain at least one value" >&2
    exit 1
  fi
  declare -A SEEN_DIAGNOSTIC_RATES=()
  for diagnostic_rate in "${DIAGNOSTIC_RATE_VALUES[@]}"; do
    if ! [[ "${diagnostic_rate}" == "inf" \
        || "${diagnostic_rate}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
      echo "Invalid diagnostic request rate: ${diagnostic_rate}" >&2
      exit 1
    fi
    if [[ "${diagnostic_rate}" != "inf" ]] \
        && ! python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)" \
          "${diagnostic_rate}"; then
      echo "Diagnostic request rate must be positive: ${diagnostic_rate}" >&2
      exit 1
    fi
    if [[ "${DATASET_LOADER}" == "mooncake" \
        && "${diagnostic_rate}" == "inf" ]]; then
      echo "Mooncake diagnostic arrival-rate scale must be finite" >&2
      exit 1
    fi
    if [[ -n "${SEEN_DIAGNOSTIC_RATES[${diagnostic_rate}]:-}" ]]; then
      echo "Duplicate diagnostic request rate: ${diagnostic_rate}" >&2
      exit 1
    fi
    SEEN_DIAGNOSTIC_RATES["${diagnostic_rate}"]=1
  done
fi

RUN_ID=$(date "+%Y%m%d-%H%M%S")
RUN_ROOT="${RESULT_ROOT}/${RUN_ID}"
mkdir -p "${RUN_ROOT}"

python3 - \
  "${RUN_ROOT}/run_manifest.json" \
  "${MODEL}" \
  "${SERVED_MODEL_NAME}" \
  "${DATASET_PATH}" \
  "${WORKLOAD_NAME}" \
  "${WORKLOAD_SLUG}" \
  "${DATASET_SOURCE_FORMAT}" \
  "${DATASET_LOADER}" \
  "${REQUEST_RATES}" \
  "${NUM_PROMPTS}" \
  "${REPETITIONS}" \
  "${DIAGNOSTIC_REPETITION_COUNT}" \
  "${MAX_CONCURRENCY:-}" \
  "${MOONCAKE_BLOCK_SIZE}" \
  "${MOONCAKE_TOKEN_SEED}" \
  "${VARIANT_MODE}" \
  "${DIAGNOSTIC_RATE_VALUES[*]}" \
  "${TRANSFER_DELAY_VALUES[*]}" \
  "${DIAGNOSTIC_TRANSFER_DELAY_VALUES[*]}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(path, model, served_model_name, dataset_path, workload_name, workload_slug,
 dataset_source_format, dataset_loader, request_rates, num_prompts,
 repetitions, diagnostic_repetitions, max_concurrency, mooncake_block_size,
 mooncake_token_seed, variant_mode, diagnostic_request_rates,
 transfer_delay_ms, diagnostic_transfer_delay_ms) = sys.argv[1:]
manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "workload": workload_name,
    "workload_slug": workload_slug,
    "model": model,
    "served_model_name": served_model_name,
    "dataset_path": dataset_path,
    "dataset_source_format": dataset_source_format,
    "vllm_dataset_loader": dataset_loader,
    "arrival_mode": (
        "trace_timestamps" if dataset_loader == "mooncake" else "generated"
    ),
    "prefix_hash_ids_replayed": dataset_loader == "mooncake",
    "request_rates": request_rates.split(),
    "diagnostic_request_rates": diagnostic_request_rates.split(),
    "transfer_delay_ms": [float(value) for value in transfer_delay_ms.split()],
    "diagnostic_transfer_delay_ms": [
        float(value) for value in diagnostic_transfer_delay_ms.split()
    ],
    "load_values_semantics": (
        "recorded_arrival_rate_multiplier"
        if dataset_loader == "mooncake"
        else "requests_per_second"
    ),
    "mooncake_block_size": (
        int(mooncake_block_size) if dataset_loader == "mooncake" else None
    ),
    "mooncake_token_seed": (
        int(mooncake_token_seed) if dataset_loader == "mooncake" else None
    ),
    "num_prompts": int(num_prompts),
    "repetitions": int(repetitions),
    "diagnostic_repetitions": int(diagnostic_repetitions),
    "max_concurrency": (
        int(max_concurrency) if max_concurrency else None
    ),
    "server_configuration": "vllm-defaults-plus-required-pd-arguments",
    "variant_mode": variant_mode,
    "variant_order": (
        "odd repetitions: off,on; even repetitions: on,off"
        if variant_mode == "paired" else variant_mode
    ),
}
Path(path).write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

git rev-parse HEAD > "${RUN_ROOT}/git_commit.txt" 2>/dev/null || true
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q > "${RUN_ROOT}/nvidia_smi.txt" 2>&1 || true
fi

PIDS=()

kill_tree() {
  local pid=$1
  local child
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    return 0
  fi
  while read -r child; do
    [[ -z "${child}" ]] && continue
    kill_tree "${child}"
  done < <(pgrep -P "${pid}" 2>/dev/null || true)
  kill "${pid}" >/dev/null 2>&1 || true
}

kill_tree_force() {
  local pid=$1
  local child
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    return 0
  fi
  while read -r child; do
    [[ -z "${child}" ]] && continue
    kill_tree_force "${child}"
  done < <(pgrep -P "${pid}" 2>/dev/null || true)
  kill -9 "${pid}" >/dev/null 2>&1 || true
}

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill_tree "${pid}"
  done
  sleep "${CLEANUP_GRACE_SECONDS:-5}"
  for pid in "${PIDS[@]:-}"; do
    kill_tree_force "${pid}"
  done
}
trap cleanup EXIT

is_port_free() {
  local host=$1
  local port=$2
  python3 - "${host}" "${port}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
bind_host = "0.0.0.0" if host in (
    "0.0.0.0", "127.0.0.1", "localhost"
) else host

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((bind_host, port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

check_ports_free() {
  local occupied=0
  local port
  for port in "${PREFILL_PORT}" "${DECODE_PORT}" "${PROXY_PORT}" \
    "${PREFILL_SIDE_CHANNEL_PORT}" "${DECODE_SIDE_CHANNEL_PORT}"; do
    if ! is_port_free "${HOST}" "${port}"; then
      echo "Port ${port} is already in use" >&2
      occupied=1
    fi
  done
  if (( occupied != 0 )); then
    return 1
  fi
}

wait_for_ports_free() {
  local waited=0
  local timeout_sec=${CLEANUP_TIMEOUT_SECONDS:-120}
  while ! check_ports_free >/dev/null 2>&1; do
    sleep 1
    waited=$((waited + 1))
    if (( waited >= timeout_sec )); then
      echo "Timed out waiting for experiment ports to become free" >&2
      check_ports_free || true
      return 1
    fi
  done
}

wait_for_server() {
  local port=$1
  local name=$2
  local waited=0
  local timeout_sec=${SERVER_START_TIMEOUT_SECONDS:-1200}
  until curl -fsS "http://${HOST}:${port}/v1/models" >/dev/null 2>&1; do
    sleep 1
    waited=$((waited + 1))
    if (( waited >= timeout_sec )); then
      echo "Timeout waiting for ${name} on port ${port}" >&2
      return 1
    fi
  done
}

wait_for_proxy() {
  local waited=0
  local timeout_sec=${PROXY_START_TIMEOUT_SECONDS:-300}
  until curl -fsS "http://${HOST}:${PROXY_PORT}/healthcheck" >/dev/null 2>&1; do
    sleep 1
    waited=$((waited + 1))
    if (( waited >= timeout_sec )); then
      echo "Timeout waiting for proxy on port ${PROXY_PORT}" >&2
      return 1
    fi
  done
}

variant_json() {
  case "$1" in
    off)
      printf 'false'
      ;;
    on)
      printf 'true'
      ;;
    *)
      echo "Unknown variant: $1" >&2
      return 1
      ;;
  esac
}

kv_config() {
  local role=$1
  local engine_id=$2
  local variant=$3
  local enabled
  enabled=$(variant_json "${variant}")
  printf '%s\n' \
    "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"${role}\",\"engine_id\":\"${engine_id}\",\"kv_connector_extra_config\":{\"canonicalize_reverse_block_pairs\":${enabled}}}"
}

start_vllm_server() {
  local role=$1
  local port=$2
  local side_channel_port=$3
  local engine_id=$4
  local devices=$5
  local case_dir=$6
  local variant=$7
  local trace_enabled=$8
  local transfer_delay_ms=$9
  local kv_role role_extra_args kv_json log_file

  if [[ "${role}" == "prefill" ]]; then
    kv_role="kv_producer"
    role_extra_args="${PREFILL_EXTRA_ARGS:-}"
    transfer_delay_ms=0
  else
    kv_role="kv_consumer"
    role_extra_args="${DECODE_EXTRA_ARGS:-}"
  fi
  kv_json=$(kv_config "${kv_role}" "${engine_id}" "${variant}")
  log_file="${case_dir}/${role}.log"

  echo "Starting ${variant}/${role} with transfer_delay_ms=${transfer_delay_ms} on port ${port}, log=${log_file}"
  (
    export CUDA_VISIBLE_DEVICES="${devices}"
    export UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
    export UCX_LOG_LEVEL="${PD_UCX_LOG_LEVEL:-warn}"
    export UCX_DEBUG_SIGNO=0
    export VLLM_NIXL_SIDE_CHANNEL_HOST="${HOST}"
    export VLLM_NIXL_SIDE_CHANNEL_PORT="${side_channel_port}"
    export VLLM_PD_TRANSFER_SLEEP_MS="${transfer_delay_ms}"
    if [[ "${trace_enabled}" == "1" ]]; then
      export VLLM_PD_TRACE_PATH="${case_dir}/${role}.trace.jsonl"
      export VLLM_PD_TRACE_ROLE="${role}"
    else
      unset VLLM_PD_TRACE_PATH VLLM_PD_TRACE_ROLE
    fi
    if [[ -n "${TRANSFORMERS_OFFLINE:-}" ]]; then
      export TRANSFORMERS_OFFLINE
    fi
    if [[ -n "${HF_HUB_OFFLINE:-}" ]]; then
      export HF_HUB_OFFLINE
    fi
    vllm serve "${MODEL}" \
      --host "0.0.0.0" \
      --port "${port}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --tensor-parallel-size "${TP_SIZE}" \
      ${SERVER_EXTRA_ARGS:-} \
      ${role_extra_args} \
      --kv-transfer-config "${kv_json}"
  ) >"${log_file}" 2>&1 &
  PIDS+=("$!")
}

start_proxy() {
  local case_dir=$1
  local trace_enabled=$2
  local log_file="${case_dir}/proxy.log"

  echo "Starting proxy on port ${PROXY_PORT}, log=${log_file}"
  (
    if [[ "${trace_enabled}" == "1" ]]; then
      export VLLM_PD_TRACE_PATH="${case_dir}/proxy.trace.jsonl"
      export VLLM_PD_TRACE_ROLE="proxy"
    else
      unset VLLM_PD_TRACE_PATH VLLM_PD_TRACE_ROLE
    fi
    python tests/pd_transfer/pd_proxy.py \
      --host "${HOST}" \
      --port "${PROXY_PORT}" \
      --prefiller-hosts "${HOST}" \
      --prefiller-ports "${PREFILL_PORT}" \
      --decoder-hosts "${HOST}" \
      --decoder-ports "${DECODE_PORT}"
  ) >"${log_file}" 2>&1 &
  PIDS+=("$!")
}

rate_slug() {
  printf '%s' "$1" | tr '.' 'p'
}

delay_slug() {
  printf '%s' "$1" | tr '.' 'p'
}

write_case_event() {
  local case_dir=$1
  local event=$2
  local case_id=$3
  local phase=$4
  local variant=$5
  local repetition=$6
  local request_rate=$7
  local num_prompts=$8
  local transfer_delay_ms=$9
  local ts_ns
  ts_ns=$(date +%s%N)
  printf '{"ts_ns":%s,"role":"bench","event":"%s","case_id":"%s","mode":"pull","dataset":"%s","phase":"%s","variant":"%s","repetition":%s,"request_rate":"%s","num_prompts":%s,"injected_transfer_delay_ms":%s}\n' \
    "${ts_ns}" "${event}" "${case_id}" "${WORKLOAD_SLUG}" "${phase}" "${variant}" \
    "${repetition}" "${request_rate}" "${num_prompts}" "${transfer_delay_ms}" \
    >> "${case_dir}/case.trace.jsonl"
}

run_warmup() {
  local case_dir=$1
  local case_id=$2
  local warmup_seed=$3
  local warmup_prompts=${WARMUP_PROMPTS:-20}

  if [[ "${warmup_prompts}" == "0" ]]; then
    return 0
  fi
  if ! [[ "${warmup_prompts}" =~ ^[1-9][0-9]*$ ]]; then
    echo "WARMUP_PROMPTS must be zero or a positive integer" >&2
    return 1
  fi

  echo "Warm-up: prompts=${warmup_prompts}, seed=${warmup_seed}"
  vllm bench serve \
    --backend vllm \
    --model "${SERVED_MODEL_NAME}" \
    --tokenizer "${BENCH_TOKENIZER:-${MODEL}}" \
    --host "${HOST}" \
    --port "${PROXY_PORT}" \
    --dataset-name random \
    --random-input-len "${WARMUP_INPUT_LEN:-512}" \
    --random-output-len "${WARMUP_OUTPUT_LEN:-32}" \
    --num-prompts "${warmup_prompts}" \
    --max-concurrency "${WARMUP_CONCURRENCY:-1}" \
    --request-rate inf \
    --ignore-eos \
    --seed "${warmup_seed}" \
    --temperature 0 \
    --ready-check-timeout-sec 0 \
    --request-id-prefix "warmup-${case_id}-" \
    ${WARMUP_EXTRA_ARGS:-} 2>&1 | tee "${case_dir}/warmup.log"
}

run_formal_benchmark() {
  local case_dir=$1
  local case_id=$2
  local phase=$3
  local variant=$4
  local repetition=$5
  local request_rate=$6
  local num_prompts=$7
  local transfer_delay_ms=$8
  local -a detailed_args=()
  local -a concurrency_args=()
  local -a ignore_eos_args=()
  local -a dataset_args=()
  local -a traffic_args=()
  local arrival_mode

  if [[ "${SAVE_DETAILED:-1}" == "1" ]]; then
    detailed_args+=(--save-detailed)
  fi
  if [[ -n "${MAX_CONCURRENCY:-}" ]]; then
    concurrency_args+=(--max-concurrency "${MAX_CONCURRENCY}")
  fi
  if [[ "${IGNORE_EOS:-1}" == "1" ]]; then
    ignore_eos_args+=(--ignore-eos)
  fi
  if [[ "${DATASET_LOADER}" == "mooncake" ]]; then
    arrival_mode="trace_timestamps"
    dataset_args+=(
      --dataset-name mooncake
      --dataset-path "${DATASET_PATH}"
      --mooncake-block-size "${MOONCAKE_BLOCK_SIZE}"
      --mooncake-arrival-rate-scale "${request_rate}"
      --mooncake-token-seed "${MOONCAKE_TOKEN_SEED}"
    )
    traffic_args+=(--request-rate inf)
  else
    arrival_mode="generated"
    dataset_args+=(--dataset-name burstgpt --dataset-path "${DATASET_PATH}")
    traffic_args+=(--request-rate "${request_rate}")
  fi

  echo "Benchmark: phase=${phase}, variant=${variant}, repetition=${repetition}, request_rate=${request_rate}, transfer_delay_ms=${transfer_delay_ms}, prompts=${num_prompts}"
  write_case_event "${case_dir}" "bench_case_start" "${case_id}" "${phase}" \
    "${variant}" "${repetition}" "${request_rate}" "${num_prompts}" \
    "${transfer_delay_ms}"
  vllm bench serve \
    --backend vllm \
    --model "${SERVED_MODEL_NAME}" \
    --tokenizer "${BENCH_TOKENIZER:-${MODEL}}" \
    --host "${HOST}" \
    --port "${PROXY_PORT}" \
    "${dataset_args[@]}" \
    --num-prompts "${num_prompts}" \
    "${traffic_args[@]}" \
    "${concurrency_args[@]}" \
    --ready-check-timeout-sec 0 \
    --save-result \
    "${detailed_args[@]}" \
    --result-dir "${case_dir}" \
    --result-filename "result.json" \
    --metadata \
      "phase=${phase}" \
      "variant=${variant}" \
      "repetition=${repetition}" \
      "configured_request_rate=${request_rate}" \
      "injected_transfer_delay_ms=${transfer_delay_ms}" \
      "dataset_loader=${DATASET_LOADER}" \
      "arrival_mode=${arrival_mode}" \
      "workload=${WORKLOAD_NAME}" \
    "${ignore_eos_args[@]}" \
    --seed "${BENCH_SEED:-1024}" \
    --temperature "${BENCH_TEMPERATURE:-0}" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --request-id-prefix "${case_id}-" \
    ${BENCH_EXTRA_ARGS:-} 2>&1 | tee "${case_dir}/benchmark.log"
  write_case_event "${case_dir}" "bench_case_end" "${case_id}" "${phase}" \
    "${variant}" "${repetition}" "${request_rate}" "${num_prompts}" \
    "${transfer_delay_ms}"
}

run_isolated_case() {
  local phase=$1
  local variant=$2
  local repetition=$3
  local request_rate=$4
  local num_prompts=$5
  local trace_enabled=$6
  local transfer_delay_ms=$7
  local slug transfer_delay_slug case_id case_dir warmup_seed

  slug=$(rate_slug "${request_rate}")
  transfer_delay_slug=$(delay_slug "${transfer_delay_ms}")
  case_id="${WORKLOAD_SLUG}-${phase}-rps-${slug}-delay-${transfer_delay_slug}-rep-${repetition}-${variant}"
  case_dir="${RUN_ROOT}/${phase}/${variant}/rps-${slug}/delay-${transfer_delay_slug}/rep-${repetition}"
  mkdir -p "${case_dir}"

  # Every isolated server receives the same warm-up workload. The Random
  # prompts are separate from the formal length-trace workload.
  warmup_seed=${WARMUP_SEED:-900000}

  check_ports_free
  start_vllm_server \
    "prefill" "${PREFILL_PORT}" "${PREFILL_SIDE_CHANNEL_PORT}" \
    "${PREFILL_ENGINE_ID}" "${PREFILL_DEVICES}" "${case_dir}" \
    "${variant}" "${trace_enabled}" "${transfer_delay_ms}"
  start_vllm_server \
    "decode" "${DECODE_PORT}" "${DECODE_SIDE_CHANNEL_PORT}" \
    "${DECODE_ENGINE_ID}" "${DECODE_DEVICES}" "${case_dir}" \
    "${variant}" "${trace_enabled}" "${transfer_delay_ms}"
  wait_for_server "${PREFILL_PORT}" "${variant}/prefill"
  wait_for_server "${DECODE_PORT}" "${variant}/decode"

  start_proxy "${case_dir}" "${trace_enabled}"
  wait_for_proxy
  run_warmup "${case_dir}" "${case_id}" "${warmup_seed}"
  run_formal_benchmark \
    "${case_dir}" "${case_id}" "${phase}" "${variant}" "${repetition}" \
    "${request_rate}" "${num_prompts}" "${transfer_delay_ms}"

  cleanup
  PIDS=()
  wait_for_ports_free
}

echo "Results will be saved to ${RUN_ROOT}"
echo "Workload: ${WORKLOAD_NAME}, mixed input/output lengths"
echo "Dataset loader: ${DATASET_LOADER}"
echo "Dataset source format: ${DATASET_SOURCE_FORMAT}"
echo "Variant mode: ${VARIANT_MODE}"
echo "Transfer delays (ms): ${TRANSFER_DELAY_VALUES[*]}"
if [[ "${DATASET_LOADER}" == "mooncake" ]]; then
  echo "REQUEST_RATES are interpreted as recorded arrival-rate multipliers."
fi
if [[ "${RUN_DIAGNOSTIC_TRACE:-1}" == "1" ]]; then
  echo "Diagnostic request rates: ${DIAGNOSTIC_RATE_VALUES[*]}"
  echo "Diagnostic transfer delays (ms): ${DIAGNOSTIC_TRANSFER_DELAY_VALUES[*]}"
  echo "Diagnostic repetitions: ${DIAGNOSTIC_REPETITION_COUNT}"
fi
echo "Performance runs use vLLM defaults except required PD/topology arguments."

if [[ "${NUM_PROMPTS}" == "0" ]]; then
  echo "Skipping trace-disabled performance runs because NUM_PROMPTS=0."
else
  for ((repetition = 1; repetition <= REPETITIONS; repetition++)); do
    ordered_transfer_delays=()
    if (( repetition % 2 == 1 )); then
      ordered_transfer_delays=("${TRANSFER_DELAY_VALUES[@]}")
    else
      for ((index = ${#TRANSFER_DELAY_VALUES[@]} - 1; index >= 0; index--)); do
        ordered_transfer_delays+=("${TRANSFER_DELAY_VALUES[index]}")
      done
    fi
    case "${VARIANT_MODE}" in
      off|on)
        ordered_variants=("${VARIANT_MODE}")
        ordered_rates=("${REQUEST_RATE_VALUES[@]}")
        ;;
      paired)
        if (( repetition % 2 == 1 )); then
          ordered_variants=(off on)
          ordered_rates=("${REQUEST_RATE_VALUES[@]}")
        else
          ordered_variants=(on off)
          ordered_rates=()
          for ((index = ${#REQUEST_RATE_VALUES[@]} - 1; index >= 0; index--)); do
            ordered_rates+=("${REQUEST_RATE_VALUES[index]}")
          done
        fi
        ;;
    esac

    for request_rate in "${ordered_rates[@]}"; do
      for variant in "${ordered_variants[@]}"; do
        for transfer_delay_ms in "${ordered_transfer_delays[@]}"; do
          run_isolated_case \
            "performance" "${variant}" "${repetition}" "${request_rate}" \
            "${NUM_PROMPTS}" 0 "${transfer_delay_ms}"
        done
      done
    done
  done
fi

if [[ "${RUN_DIAGNOSTIC_TRACE:-1}" == "1" ]]; then
  diagnostic_prompts=${DIAGNOSTIC_NUM_PROMPTS:-500}
  if ! [[ "${diagnostic_prompts}" =~ ^[1-9][0-9]*$ ]]; then
    echo "DIAGNOSTIC_NUM_PROMPTS must be a positive integer" >&2
    exit 1
  fi
  for ((diagnostic_repetition = 1;
        diagnostic_repetition <= DIAGNOSTIC_REPETITION_COUNT;
        diagnostic_repetition++)); do
    ordered_diagnostic_rates=()
    ordered_diagnostic_transfer_delays=()
    if (( diagnostic_repetition % 2 == 1 )); then
      ordered_diagnostic_rates=("${DIAGNOSTIC_RATE_VALUES[@]}")
      ordered_diagnostic_transfer_delays=(
        "${DIAGNOSTIC_TRANSFER_DELAY_VALUES[@]}")
      if [[ "${VARIANT_MODE}" == "paired" ]]; then
        ordered_diagnostic_variants=(off on)
      else
        ordered_diagnostic_variants=("${VARIANT_MODE}")
      fi
    else
      for ((index = ${#DIAGNOSTIC_RATE_VALUES[@]} - 1;
            index >= 0; index--)); do
        ordered_diagnostic_rates+=("${DIAGNOSTIC_RATE_VALUES[index]}")
      done
      for ((index = ${#DIAGNOSTIC_TRANSFER_DELAY_VALUES[@]} - 1;
            index >= 0; index--)); do
        ordered_diagnostic_transfer_delays+=(
          "${DIAGNOSTIC_TRANSFER_DELAY_VALUES[index]}")
      done
      if [[ "${VARIANT_MODE}" == "paired" ]]; then
        ordered_diagnostic_variants=(on off)
      else
        ordered_diagnostic_variants=("${VARIANT_MODE}")
      fi
    fi

    for diagnostic_rate in "${ordered_diagnostic_rates[@]}"; do
      for variant in "${ordered_diagnostic_variants[@]}"; do
        for transfer_delay_ms in \
            "${ordered_diagnostic_transfer_delays[@]}"; do
          run_isolated_case \
            "diagnostic" "${variant}" "${diagnostic_repetition}" \
            "${diagnostic_rate}" "${diagnostic_prompts}" 1 \
            "${transfer_delay_ms}"
        done
      done
    done
  done
fi

if [[ "${VARIANT_MODE}" == "paired" ]]; then
  echo "Done. Aggregate OFF/ON performance with:"
  echo "  python tests/pd_transfer/compare_pd_generalization_perf.py ${RUN_ROOT}"
else
  echo "Done. Single-variant results are in ${RUN_ROOT}."
fi
