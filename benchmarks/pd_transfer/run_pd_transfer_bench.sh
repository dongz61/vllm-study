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

# Keep existing config.env files working while enabling a representative,
# content-disjoint warm-up by default.
WARMUP_PROMPTS=${WARMUP_PROMPTS:-4}
WARMUP_INPUT_LEN=${WARMUP_INPUT_LEN:-4096}
WARMUP_OUTPUT_LEN=${WARMUP_OUTPUT_LEN:-16}
WARMUP_CONCURRENCY=${WARMUP_CONCURRENCY:-1}
WARMUP_SEED=${WARMUP_SEED:-900000}

PIDS=()

kill_tree() {
  local pid=$1 child
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    return
  fi
  while read -r child; do
    [[ -n "${child}" ]] && kill_tree "${child}"
  done < <(pgrep -P "${pid}" 2>/dev/null || true)
  kill "${pid}" >/dev/null 2>&1 || true
}

cleanup() {
  local pid
  for pid in "${PIDS[@]}"; do
    kill_tree "${pid}"
  done
  sleep "${CLEANUP_GRACE_SECONDS:-5}"
  for pid in "${PIDS[@]}"; do
    kill -9 "${pid}" >/dev/null 2>&1 || true
  done
  PIDS=()
}
trap cleanup EXIT

require_positive_integer() {
  local name=$1 value=${!1}
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer, got ${value}" >&2
    exit 1
  fi
}

require_nonnegative_integer() {
  local name=$1 value=${!1}
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "${name} must be a non-negative integer, got ${value}" >&2
    exit 1
  fi
}

validate_devices() {
  local role=$1 devices=$2 tp_size=$3
  local -a device_array
  IFS=',' read -r -a device_array <<< "${devices}"
  if (( ${#device_array[@]} != tp_size )); then
    echo "${role}: ${#device_array[@]} devices but TP size is ${tp_size}" >&2
    exit 1
  fi
}

validate_config() {
  local name p_device d_device
  local -a p_devices d_devices
  for name in PREFILL_TP_SIZE DECODE_TP_SIZE NUM_PROMPTS MAX_CONCURRENCY \
    MOONCAKE_BLOCK_SIZE; do
    require_positive_integer "${name}"
  done
  require_nonnegative_integer WARMUP_PROMPTS
  if (( WARMUP_PROMPTS > 0 )); then
    for name in WARMUP_INPUT_LEN WARMUP_OUTPUT_LEN WARMUP_CONCURRENCY; do
      require_positive_integer "${name}"
    done
    require_nonnegative_integer WARMUP_SEED
    if (( WARMUP_INPUT_LEN + WARMUP_OUTPUT_LEN > MAX_MODEL_LEN )); then
      echo "Warm-up input + output length exceeds MAX_MODEL_LEN" >&2
      exit 1
    fi
  fi
  validate_devices "prefill" "${PREFILL_DEVICES}" "${PREFILL_TP_SIZE}"
  validate_devices "decode" "${DECODE_DEVICES}" "${DECODE_TP_SIZE}"
  if (( PREFILL_TP_SIZE % DECODE_TP_SIZE != 0 \
        && DECODE_TP_SIZE % PREFILL_TP_SIZE != 0 )); then
    echo "PREFILL_TP_SIZE and DECODE_TP_SIZE must divide one another" >&2
    exit 1
  fi
  if [[ ! "${TRACE_TIME_SCALE}" =~ ^0*\.?0*[1-9][0-9]*$ \
        && ! "${TRACE_TIME_SCALE}" =~ ^[1-9][0-9]*(\.[0-9]+)?$ ]]; then
    echo "TRACE_TIME_SCALE must be positive, got ${TRACE_TIME_SCALE}" >&2
    exit 1
  fi
  IFS=',' read -r -a p_devices <<< "${PREFILL_DEVICES}"
  IFS=',' read -r -a d_devices <<< "${DECODE_DEVICES}"
  for p_device in "${p_devices[@]}"; do
    for d_device in "${d_devices[@]}"; do
      if [[ "${p_device}" == "${d_device}" ]]; then
        echo "GPU ${p_device} is assigned to both P and D" >&2
        exit 1
      fi
    done
  done
  if [[ ! -r "${MOONCAKE_TRACE_PATH}" ]]; then
    echo "Mooncake trace is not readable: ${MOONCAKE_TRACE_PATH}" >&2
    exit 1
  fi
}

is_port_free() {
  python3 - "$1" "$2" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
bind_host = "0.0.0.0" if host in ("localhost", "127.0.0.1") else host
with socket.socket() as sock:
    try:
        sock.bind((bind_host, port))
    except OSError:
        raise SystemExit(1)
PY
}

check_ports() {
  local port
  for port in "${PREFILL_PORT}" "${DECODE_PORT}" "${PROXY_PORT}" \
    "${PREFILL_SIDE_CHANNEL_PORT}" "${DECODE_SIDE_CHANNEL_PORT}"; do
    if ! is_port_free "${HOST}" "${port}"; then
      echo "Port ${port} is already in use" >&2
      exit 1
    fi
  done
}

wait_for_url() {
  local url=$1 name=$2 timeout=${3:-1200} waited=0
  until curl -fsS "${url}" >/dev/null 2>&1; do
    sleep 1
    waited=$((waited + 1))
    if (( waited >= timeout )); then
      echo "Timed out waiting for ${name}: ${url}" >&2
      return 1
    fi
  done
}

kv_config() {
  local role=$1 engine_id=$2
  printf '{"kv_connector":"NixlConnector","kv_role":"%s","engine_id":"%s"}' \
    "${role}" "${engine_id}"
}

start_server() {
  local role=$1 port=$2 side_port=$3 engine_id=$4 devices=$5
  local tp_size=$6 trace_dir=$7 log_file=$8
  local kv_role max_batched max_seqs gpu_util extra_args
  if [[ "${role}" == "prefill" ]]; then
    kv_role="kv_producer"
    max_batched=${PREFILL_MAX_NUM_BATCHED_TOKENS}
    max_seqs=${PREFILL_MAX_NUM_SEQS}
    gpu_util=${PREFILL_GPU_MEMORY_UTILIZATION}
    extra_args=${PREFILL_EXTRA_ARGS:-}
  else
    kv_role="kv_consumer"
    max_batched=${DECODE_MAX_NUM_BATCHED_TOKENS}
    max_seqs=${DECODE_MAX_NUM_SEQS}
    gpu_util=${DECODE_GPU_MEMORY_UTILIZATION}
    extra_args=${DECODE_EXTRA_ARGS:-}
  fi
  local kv_json
  kv_json=$(kv_config "${kv_role}" "${engine_id}")

  echo "Starting ${role}: GPUs=${devices}, TP=${tp_size}, port=${port}"
  (
    export CUDA_VISIBLE_DEVICES="${devices}"
    export UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
    export VLLM_NIXL_SIDE_CHANNEL_HOST="${HOST}"
    export VLLM_NIXL_SIDE_CHANNEL_PORT="${side_port}"
    export VLLM_PD_TRACE_PATH="${trace_dir}/"
    export VLLM_PD_TRACE_ROLE="${role}"
    export TRANSFORMERS_OFFLINE HF_HUB_OFFLINE
    # Extra argument strings are intentionally word-split as CLI arguments.
    # shellcheck disable=SC2086
    vllm serve "${MODEL}" \
      --host 0.0.0.0 \
      --port "${port}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --tensor-parallel-size "${tp_size}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${max_batched}" \
      --max-num-seqs "${max_seqs}" \
      --gpu-memory-utilization "${gpu_util}" \
      ${COMMON_VLLM_ARGS:-} \
      ${extra_args} \
      --kv-transfer-config "${kv_json}"
  ) >"${log_file}" 2>&1 &
  PIDS+=("$!")
}

start_proxy() {
  local trace_dir=$1 log_file=$2
  (
    export VLLM_PD_TRACE_PATH="${trace_dir}/"
    export VLLM_PD_TRACE_ROLE="proxy"
    python3 "${SCRIPT_DIR}/pd_proxy.py" \
      --host "${HOST}" \
      --port "${PROXY_PORT}" \
      --prefiller-host "${HOST}" \
      --prefiller-port "${PREFILL_PORT}" \
      --decoder-host "${HOST}" \
      --decoder-port "${DECODE_PORT}"
  ) >"${log_file}" 2>&1 &
  PIDS+=("$!")
}

run_warmup() {
  local run_root=$1 run_id=$2

  if (( WARMUP_PROMPTS == 0 )); then
    echo "Warm-up disabled"
    return 0
  fi

  printf 'Warm-up: prompts=%s, input=%s, output=%s, concurrency=%s\n' \
    "${WARMUP_PROMPTS}" "${WARMUP_INPUT_LEN}" \
    "${WARMUP_OUTPUT_LEN}" "${WARMUP_CONCURRENCY}"
  # Random inputs are deliberately separate from the formal TimedTrace
  # workload, so warm-up cannot create prefix hits in the measured requests.
  # shellcheck disable=SC2086
  vllm bench serve \
    --backend vllm \
    --model "${SERVED_MODEL_NAME}" \
    --tokenizer "${BENCH_TOKENIZER:-${MODEL}}" \
    --host "${HOST}" \
    --port "${PROXY_PORT}" \
    --dataset-name random \
    --random-input-len "${WARMUP_INPUT_LEN}" \
    --random-output-len "${WARMUP_OUTPUT_LEN}" \
    --num-prompts "${WARMUP_PROMPTS}" \
    --max-concurrency "${WARMUP_CONCURRENCY}" \
    --request-rate inf \
    --ignore-eos \
    --seed "${WARMUP_SEED}" \
    --temperature 0 \
    --ready-check-timeout-sec 0 \
    --request-id-prefix "warmup-${run_id}-" \
    ${WARMUP_EXTRA_ARGS:-} 2>&1 | tee "${run_root}/warmup.log"
}

validate_config
check_ports

RUN_ID=$(date "+%Y%m%d-%H%M%S")
RUN_ROOT="${RESULT_ROOT}/${RUN_ID}"
mkdir -p "${RUN_ROOT}/traces/prefill" "${RUN_ROOT}/traces/decode" \
  "${RUN_ROOT}/traces/proxy"

printf '{"run_id":"%s","prefill_tp_size":%s,"decode_tp_size":%s,' \
  "${RUN_ID}" "${PREFILL_TP_SIZE}" "${DECODE_TP_SIZE}" \
  >"${RUN_ROOT}/run_manifest.json"
printf '"prefill_devices":"%s","decode_devices":"%s",' \
  "${PREFILL_DEVICES}" "${DECODE_DEVICES}" \
  >>"${RUN_ROOT}/run_manifest.json"
printf '"warmup_prompts":%s,"warmup_input_len":%s,' \
  "${WARMUP_PROMPTS}" "${WARMUP_INPUT_LEN}" \
  >>"${RUN_ROOT}/run_manifest.json"
printf '"warmup_output_len":%s,"warmup_concurrency":%s,"warmup_seed":%s,' \
  "${WARMUP_OUTPUT_LEN}" "${WARMUP_CONCURRENCY}" "${WARMUP_SEED}" \
  >>"${RUN_ROOT}/run_manifest.json"
printf '"trace_path":"%s","num_prompts":%s,"trace_time_scale":%s}\n' \
  "${MOONCAKE_TRACE_PATH}" "${NUM_PROMPTS}" "${TRACE_TIME_SCALE}" \
  >>"${RUN_ROOT}/run_manifest.json"

start_server prefill "${PREFILL_PORT}" "${PREFILL_SIDE_CHANNEL_PORT}" \
  "${PREFILL_ENGINE_ID}" "${PREFILL_DEVICES}" "${PREFILL_TP_SIZE}" \
  "${RUN_ROOT}/traces/prefill" "${RUN_ROOT}/prefill.log"
start_server decode "${DECODE_PORT}" "${DECODE_SIDE_CHANNEL_PORT}" \
  "${DECODE_ENGINE_ID}" "${DECODE_DEVICES}" "${DECODE_TP_SIZE}" \
  "${RUN_ROOT}/traces/decode" "${RUN_ROOT}/decode.log"

wait_for_url "http://${HOST}:${PREFILL_PORT}/v1/models" prefill
wait_for_url "http://${HOST}:${DECODE_PORT}/v1/models" decode
start_proxy "${RUN_ROOT}/traces/proxy" "${RUN_ROOT}/proxy.log"
wait_for_url "http://${HOST}:${PROXY_PORT}/healthcheck" proxy 300
run_warmup "${RUN_ROOT}" "${RUN_ID}"

echo "Running Mooncake trace: P_TP=${PREFILL_TP_SIZE}, D_TP=${DECODE_TP_SIZE}"
# shellcheck disable=SC2086
vllm bench serve \
  --backend vllm \
  --model "${SERVED_MODEL_NAME}" \
  --tokenizer "${BENCH_TOKENIZER:-${MODEL}}" \
  --host "${HOST}" \
  --port "${PROXY_PORT}" \
  --dataset-name timed_trace \
  --dataset-path "${MOONCAKE_TRACE_PATH}" \
  --timed-trace-chunk-hash-size "${MOONCAKE_BLOCK_SIZE}" \
  --timed-trace-sec-multiplier "${TRACE_TIME_SCALE}" \
  --self-timed \
  --num-prompts "${NUM_PROMPTS}" \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --request-id-prefix "${RUN_ID}-" \
  --save-result \
  --save-detailed \
  --result-dir "${RUN_ROOT}" \
  --result-filename benchmark.json \
  ${BENCH_EXTRA_ARGS:-} 2>&1 | tee "${RUN_ROOT}/benchmark.log"

cleanup
trap - EXIT
python3 "${SCRIPT_DIR}/parse_pd_trace.py" "${RUN_ROOT}" \
  | tee "${RUN_ROOT}/parse.log"
echo "Results: ${RUN_ROOT}"
