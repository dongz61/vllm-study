#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-"tests/pd_transfer/config.env"}
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${CONFIG_PATH}"

RUN_ID=$(date "+%Y%m%d-%H%M%S")
RUN_ROOT="${RESULT_ROOT}/${RUN_ID}"
mkdir -p "${RUN_ROOT}"

PIDS=()

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
  sleep 2
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT

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
{"kv_connector":"NixlConnector","kv_role":"${role}","engine_id":"${engine_id}"}
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
  local num_prompts=$((concurrency * NUM_FOLDS))
  local case_id="pull-sleep-${sleep_ms}-input-${input_len}-output-${output_len}-concurrency-${concurrency}"
  local result_name="${case_id}.json"
  local bench_log="${case_dir}/bench-sleep-${sleep_ms}-input-${input_len}-output-${output_len}-concurrency-${concurrency}.log"

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
    --result-dir "${case_dir}" \
    --result-filename "${result_name}" \
    ${BENCH_EXTRA_ARGS} 2>&1 | tee "${bench_log}"
  write_case_event "${case_dir}" "bench_case_end" "${case_id}" "${input_len}" "${output_len}" "${concurrency}" "${num_prompts}" "${sleep_ms}"
}

run_sleep_case() {
  local sleep_ms=$1
  local sleep_dir_name="${sleep_ms//./p}"
  local case_dir="${RUN_ROOT}/pull/sleep-${sleep_dir_name}"
  mkdir -p "${case_dir}"

  start_vllm_server "prefill" "${PREFILL_PORT}" "${PREFILL_SIDE_CHANNEL_PORT}" "${PREFILL_ENGINE_ID}" \
    "${PREFILL_DEVICES}" "${case_dir}/prefill.trace.jsonl" "${case_dir}/prefill.log" "${sleep_ms}"
  start_vllm_server "decode" "${DECODE_PORT}" "${DECODE_SIDE_CHANNEL_PORT}" "${DECODE_ENGINE_ID}" \
    "${DECODE_DEVICES}" "${case_dir}/decode.trace.jsonl" "${case_dir}/decode.log" "${sleep_ms}"
  wait_for_server "${PREFILL_PORT}" "pull/prefill"
  wait_for_server "${DECODE_PORT}" "pull/decode"

  start_proxy "${case_dir}/proxy.trace.jsonl" "${case_dir}/proxy.log"
  wait_for_proxy

  for input_len in ${INPUT_LENS}; do
    for output_len in ${OUTPUT_LENS}; do
      for concurrency in ${CONCURRENCIES}; do
        run_benchmark_case "${case_dir}" "${input_len}" "${output_len}" "${concurrency}" "${sleep_ms}"
      done
    done
  done

  cleanup
  PIDS=()
}

echo "Results will be saved to ${RUN_ROOT}"
for sleep_ms in ${TRANSFER_SLEEP_MS_LIST:-0}; do
  run_sleep_case "${sleep_ms}"
done

echo "Done. Parse traces with:"
echo "  python tests/pd_transfer/parse_pd_trace.py ${RUN_ROOT}"
