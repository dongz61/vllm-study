#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-"tests/pd_transfer/config.env"}
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${CONFIG_PATH}"

case "${CANONICALIZE_REVERSE_BLOCK_PAIRS:-0}" in
  0)
    CANONICALIZE_REVERSE_BLOCK_PAIRS_JSON=false
    ;;
  1)
    CANONICALIZE_REVERSE_BLOCK_PAIRS_JSON=true
    ;;
  *)
    echo "CANONICALIZE_REVERSE_BLOCK_PAIRS must be 0 or 1" >&2
    exit 1
    ;;
esac

RUN_ID=$(date "+%Y%m%d-%H%M%S")
RUN_ROOT="${RESULT_ROOT}/${RUN_ID}"
mkdir -p "${RUN_ROOT}"
printf '{"canonicalize_reverse_block_pairs":%s}\n' \
  "${CANONICALIZE_REVERSE_BLOCK_PAIRS_JSON}" \
  > "${RUN_ROOT}/experiment_config.json"

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
bind_host = "0.0.0.0" if host in ("0.0.0.0", "127.0.0.1", "localhost") else host

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((bind_host, port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
sys.exit(0)
PY
}

check_ports_free() {
  local occupied=0
  local port
  for port in "${PREFILL_PORT}" "${DECODE_PORT}" "${PROXY_PORT}" \
    "${PREFILL_SIDE_CHANNEL_PORT}" "${DECODE_SIDE_CHANNEL_PORT}"; do
    if ! is_port_free "${HOST}" "${port}"; then
      echo "Port ${port} is already in use" >&2
      if command -v ss >/dev/null 2>&1; then
        ss -ltnp "sport = :${port}" >&2 || true
      elif command -v netstat >/dev/null 2>&1; then
        netstat -ltnp 2>/dev/null | grep ":${port}" >&2 || true
      fi
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
      echo "Timed out waiting for PD experiment ports to become free" >&2
      check_ports_free || true
      return 1
    fi
  done
}

wait_for_server() {
  local port=$1
  local name=$2
  local waited=0
  local timeout_sec=1200
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
  local timeout_sec=300
  until curl -fsS "http://${HOST}:${PROXY_PORT}/healthcheck" >/dev/null 2>&1; do
    sleep 1
    waited=$((waited + 1))
    if (( waited >= timeout_sec )); then
      echo "Timeout waiting for proxy on port ${PROXY_PORT}" >&2
      return 1
    fi
  done
}

write_case_event() {
  local case_dir=$1
  local event=$2
  local case_id=$3
  local input_len=$4
  local output_len=$5
  local concurrency=$6
  local num_prompts=$7
  local sleep_ms=$8
  local ts_ns
  ts_ns=$(date +%s%N)
  printf '{"ts_ns":%s,"role":"bench","event":"%s","case_id":"%s","mode":"pull","sleep_ms":%s,"input_len":%s,"output_len":%s,"concurrency":%s,"num_prompts":%s}\n' \
    "${ts_ns}" "${event}" "${case_id}" "${sleep_ms}" "${input_len}" "${output_len}" "${concurrency}" "${num_prompts}" \
    >> "${case_dir}/case.trace.jsonl"
}

kv_config() {
  local role=$1
  local engine_id=$2
  cat <<EOF
{"kv_connector":"NixlConnector","kv_role":"${role}","engine_id":"${engine_id}","kv_connector_extra_config":{"canonicalize_reverse_block_pairs":${CANONICALIZE_REVERSE_BLOCK_PAIRS_JSON}}}
EOF
}

start_vllm_server() {
  local role=$1
  local port=$2
  local side_channel_port=$3
  local engine_id=$4
  local devices=$5
  local trace_file=$6
  local log_file=$7
  local sleep_ms=$8
  local kv_role extra_args kv_json

  if [[ "${role}" == "prefill" ]]; then
    kv_role="kv_producer"
    extra_args="${PREFILL_EXTRA_ARGS:-}"
  else
    kv_role="kv_consumer"
    extra_args="${DECODE_EXTRA_ARGS:-}"
  fi
  kv_json=$(kv_config "${kv_role}" "${engine_id}")

  echo "Starting pull/${role} on port ${port}, log=${log_file}"
  (
    export CUDA_VISIBLE_DEVICES="${devices}"
    export UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
    export VLLM_NIXL_SIDE_CHANNEL_HOST="${HOST}"
    export VLLM_NIXL_SIDE_CHANNEL_PORT="${side_channel_port}"
    export VLLM_PD_TRACE_PATH="${trace_file}"
    export VLLM_PD_TRACE_ROLE="${role}"
    export VLLM_PD_TRANSFER_SLEEP_MS="${sleep_ms}"
    export TRANSFORMERS_OFFLINE HF_HUB_OFFLINE
    vllm serve "${MODEL}" \
      --host "0.0.0.0" \
      --port "${port}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      ${COMMON_VLLM_ARGS} \
      ${extra_args} \
      --kv-transfer-config "${kv_json}"
  ) >"${log_file}" 2>&1 &
  PIDS+=("$!")
}

start_proxy() {
  local trace_file=$1
  local log_file=$2
  echo "Starting pull/proxy on port ${PROXY_PORT}, log=${log_file}"
  (
    export VLLM_PD_TRACE_PATH="${trace_file}"
    export VLLM_PD_TRACE_ROLE="proxy"
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

run_benchmark_case() {
  local case_dir=$1
  local input_len=$2
  local output_len=$3
  local concurrency=$4
  local sleep_ms=$5
  local num_prompts=${6:-$((concurrency * NUM_FOLDS))}
  local case_id="pull-sleep-${sleep_ms}-input-${input_len}-output-${output_len}-concurrency-${concurrency}"
  local result_name="${case_id}.json"
  local bench_log="${case_dir}/bench-sleep-${sleep_ms}-input-${input_len}-output-${output_len}-concurrency-${concurrency}.log"

  if ! [[ "${num_prompts}" =~ ^[1-9][0-9]*$ ]]; then
    echo "num_prompts must be a positive integer, got: ${num_prompts}" >&2
    return 1
  fi

  echo "Benchmark pull: sleep_ms=${sleep_ms}, input=${input_len}, output=${output_len}, concurrency=${concurrency}, prompts=${num_prompts}"
  write_case_event "${case_dir}" "bench_case_start" "${case_id}" "${input_len}" "${output_len}" "${concurrency}" "${num_prompts}" "${sleep_ms}"
  vllm bench serve \
    --backend vllm \
    --model "${SERVED_MODEL_NAME}" \
    --tokenizer "${BENCH_TOKENIZER:-${MODEL}}" \
    --host "${HOST}" \
    --port "${PROXY_PORT}" \
    --dataset-name random \
    --random-input-len "${input_len}" \
    --random-output-len "${output_len}" \
    --random-prefix-len "${RANDOM_PREFIX_LEN}" \
    --num-prompts "${num_prompts}" \
    --max-concurrency "${concurrency}" \
    --save-result \
    --save-detailed \
    --result-dir "${case_dir}" \
    --result-filename "${result_name}" \
    ${BENCH_EXTRA_ARGS} \
    --seed "${BENCH_SEED:-1024}" \
    --temperature "${BENCH_TEMPERATURE:-0}" \
    --request-id-prefix "${case_id}-" 2>&1 | tee "${bench_log}"
  write_case_event "${case_dir}" "bench_case_end" "${case_id}" "${input_len}" "${output_len}" "${concurrency}" "${num_prompts}" "${sleep_ms}"
}

parse_bench_case() {
  local case_spec=$1
  local fields
  IFS=',' read -r -a fields <<< "${case_spec}"
  if (( ${#fields[@]} < 4 || ${#fields[@]} > 5 )); then
    echo "Invalid BENCH_CASES entry: ${case_spec}. Expected input,output,concurrency,sleep[,num_prompts]" >&2
    return 1
  fi

  BENCH_CASE_INPUT_LEN=${fields[0]}
  BENCH_CASE_OUTPUT_LEN=${fields[1]}
  BENCH_CASE_CONCURRENCY=${fields[2]}
  BENCH_CASE_SLEEP_MS=${fields[3]}
  BENCH_CASE_NUM_PROMPTS=${fields[4]:-}

  if ! [[ "${BENCH_CASE_INPUT_LEN}" =~ ^[1-9][0-9]*$ ]] \
      || ! [[ "${BENCH_CASE_OUTPUT_LEN}" =~ ^[1-9][0-9]*$ ]] \
      || ! [[ "${BENCH_CASE_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] \
      || ! [[ "${BENCH_CASE_SLEEP_MS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Invalid BENCH_CASES entry: ${case_spec}. Lengths/concurrency must be positive integers and sleep must be non-negative" >&2
    return 1
  fi

  if [[ -z "${BENCH_CASE_NUM_PROMPTS}" ]]; then
    if ! [[ "${NUM_FOLDS}" =~ ^[1-9][0-9]*$ ]]; then
      echo "NUM_FOLDS must be a positive integer, got: ${NUM_FOLDS}" >&2
      return 1
    fi
    BENCH_CASE_NUM_PROMPTS=$((BENCH_CASE_CONCURRENCY * NUM_FOLDS))
  elif ! [[ "${BENCH_CASE_NUM_PROMPTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid num_prompts in BENCH_CASES entry: ${case_spec}" >&2
    return 1
  fi
}

selected_sleep_values() {
  if [[ -n "${BENCH_CASES:-}" ]]; then
    local case_spec
    for case_spec in ${BENCH_CASES}; do
      parse_bench_case "${case_spec}"
      printf '%s\n' "${BENCH_CASE_SLEEP_MS}"
    done | awk '!seen[$0]++'
  else
    printf '%s\n' ${TRANSFER_SLEEP_MS_LIST:-0}
  fi
}

run_benchmark_cases_for_sleep() {
  local case_dir=$1
  local sleep_ms=$2
  local case_spec matched
  matched=0

  if [[ -n "${BENCH_CASES:-}" ]]; then
    for case_spec in ${BENCH_CASES}; do
      parse_bench_case "${case_spec}"
      if [[ "${BENCH_CASE_SLEEP_MS}" == "${sleep_ms}" ]]; then
        run_benchmark_case \
          "${case_dir}" \
          "${BENCH_CASE_INPUT_LEN}" \
          "${BENCH_CASE_OUTPUT_LEN}" \
          "${BENCH_CASE_CONCURRENCY}" \
          "${sleep_ms}" \
          "${BENCH_CASE_NUM_PROMPTS}"
        matched=1
      fi
    done
    if (( matched == 0 )); then
      echo "No BENCH_CASES entries matched sleep_ms=${sleep_ms}" >&2
      return 1
    fi
    return 0
  fi

  for input_len in ${INPUT_LENS}; do
    for output_len in ${OUTPUT_LENS}; do
      for concurrency in ${CONCURRENCIES}; do
        run_benchmark_case "${case_dir}" "${input_len}" "${output_len}" "${concurrency}" "${sleep_ms}"
      done
    done
  done
}

run_sleep_case() {
  local sleep_ms=$1
  local sleep_dir_name="${sleep_ms//./p}"
  local case_dir="${RUN_ROOT}/pull/sleep-${sleep_dir_name}"
  mkdir -p "${case_dir}"

  check_ports_free

  start_vllm_server "prefill" "${PREFILL_PORT}" "${PREFILL_SIDE_CHANNEL_PORT}" "${PREFILL_ENGINE_ID}" \
    "${PREFILL_DEVICES}" "${case_dir}/prefill.trace.jsonl" "${case_dir}/prefill.log" "${sleep_ms}"
  start_vllm_server "decode" "${DECODE_PORT}" "${DECODE_SIDE_CHANNEL_PORT}" "${DECODE_ENGINE_ID}" \
    "${DECODE_DEVICES}" "${case_dir}/decode.trace.jsonl" "${case_dir}/decode.log" "${sleep_ms}"
  wait_for_server "${PREFILL_PORT}" "pull/prefill"
  wait_for_server "${DECODE_PORT}" "pull/decode"

  start_proxy "${case_dir}/proxy.trace.jsonl" "${case_dir}/proxy.log"
  wait_for_proxy

  run_benchmark_cases_for_sleep "${case_dir}" "${sleep_ms}"

  cleanup
  PIDS=()
  wait_for_ports_free
}

echo "Results will be saved to ${RUN_ROOT}"
echo "Paired-reverse block canonicalization: ${CANONICALIZE_REVERSE_BLOCK_PAIRS_JSON}"
while read -r sleep_ms; do
  [[ -z "${sleep_ms}" ]] && continue
  run_sleep_case "${sleep_ms}"
done < <(selected_sleep_values)

echo "Done. Parse traces with:"
echo "  python tests/pd_transfer/parse_pd_trace.py ${RUN_ROOT}"
