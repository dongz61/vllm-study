# NIXL descriptor-order microbenchmark

`nixl_desc_order_bench.py` isolates the descriptor-order effect from vLLM.
Run the same script in separate NIXL 0.6.0 and 1.3.2 environments.

## Experiment matrix

Each run measures four cells:

| Input order | Mitigation | Submitted order |
|---|---|---|
| ascending | OFF | ascending |
| descending | OFF | descending |
| ascending | ON | ascending |
| descending | ON | ascending |

Combining the two NIXL environments produces the planned eight cells. The
default workload uses one contiguous 32 MiB buffer represented as 1024
descriptors of 32 KiB each.

The target process owns the source buffer. The initiator performs NIXL `READ`
operations into a second GPU buffer. Both processes explicitly select the UCX
backend and exchange agent metadata and serialized descriptors through a
Python multiprocessing pipe.

The ON path uses the same paired-reverse-run canonicalization as the vLLM
connector: local and remote block sequences are reversed together, preserving
every block mapping.

## Server command

From the vLLM study repository, run:

```bash
python tests/pd_transfer/nixl_desc_order_bench.py \
  --target-gpu 3 \
  --initiator-gpu 4 \
  --descriptors 1024 \
  --descriptor-bytes 32768 \
  --warmup 5 \
  --repeats 30 \
  --nixl-label 0.6.0 \
  --output results/nixl_desc_order_0.6.0.jsonl
```

Repeat in the NIXL 1.3.2 environment, changing the label and output path:

```bash
python tests/pd_transfer/nixl_desc_order_bench.py \
  --target-gpu 3 \
  --initiator-gpu 4 \
  --descriptors 1024 \
  --descriptor-bytes 32768 \
  --warmup 5 \
  --repeats 30 \
  --nixl-label 1.3.2 \
  --output results/nixl_desc_order_1.3.2.jsonl
```

When a container exposes only the selected GPUs, use their container-visible
indices instead. For example, with `CUDA_VISIBLE_DEVICES=3,4`, target and
initiator indices are normally `0` and `1`.

If NIXL was built from source rather than installed as a wheel, set
`NIXL_PLUGIN_DIR` and the library/Python paths required by that build before
running the script.

## Output

The JSONL file contains:

- one `metadata` record;
- one raw `sample` record per measured transfer;
- one `correctness` record for each cell;
- one `summary` record for each cell.

The primary measurements are:

- `canonicalize_ns`;
- `make_xfer_ns`;
- `post_xfer_ns`;
- `poll_xfer_ns`;
- `transfer_total_ns`;
- `end_to_end_ns`;
- `effective_gbps`.

`post_xfer_ns` measures the blocking duration of Python `transfer()`, matching
the vLLM `xfer_submit` measurement. `transfer_total_ns` covers `transfer()`
through the first observed `DONE`.

The four cells are shuffled within every repetition to reduce drift and
temperature bias. Correctness is checked separately after timing by clearing
the initiator buffer and verifying a descriptor-specific data pattern.
