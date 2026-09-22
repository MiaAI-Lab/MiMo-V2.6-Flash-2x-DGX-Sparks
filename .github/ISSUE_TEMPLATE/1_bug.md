---
name: Bug report
about: The kit is not behaving as expected, produces an error, or numbers do not match the README.
title: ""
labels: "bug"
assignees: ""
---

<!-- Thank you for using this model kit!

     If you are looking for support, please check the README first,
     or reach out on X:
      * https://x.com/MiaAI_lab

     If you have found a bug, then fill out the template below.
-->

---

## Environment

<!-- Fill in what applies to your setup. The README's "Defaults" table lists every knob and its value. -->

- Hardware / nodes: <!-- e.g. 2x DGX Spark (GB10, SM121, 121.7 GiB unified memory each) -->
- Interconnect: <!-- e.g. ConnectX-7 RoCE, a switch, 10GbE, ... -->
- `BASE_IMAGE`: <!-- `docker image inspect -f '{{index .Config.Labels "mimo26.base"}}' mimo26-spark:local` -->
- Checkpoint (`HF_REPO` + `HF_REVISION`): <!-- e.g. XiaomiMiMo/MiMo-V2.6-Flash-RL @ 3b38d06 -->
- Drafter: <!-- SPEC_ALGO + SPEC_DRAFT_TOKENS, e.g. DFLASH / 8 or EAGLE / 4 -->
- Memory settings: <!-- MEM_FRACTION_STATIC, MAX_TOTAL_TOKENS, SWA_FULL_TOKENS_RATIO, MEM_GUARD_GIB -->
- Graphs / prefill: <!-- CUDA_GRAPH_MAX_BS_DECODE, CUDA_GRAPH_PREFILL, CHUNKED_PREFILL_SIZE -->
- How you started it: <!-- ./start.sh serve, ./start.sh restart, a custom invocation -->

---

## Steps to Reproduce

<!-- Full steps so that we can reproduce the problem. -->

1. <!-- e.g. `./start.sh serve` — what it printed up to the failure -->
2. ... <!-- the request or action that shows the bug -->
3. ...

**Expected results:** <!-- what did you expect to happen? -->

**Actual results:** <!-- what did you actually see happen? -->

---

### Additional context

<details>
<summary>Minimal reproduction sample</summary>

<!-- If the bug is about model output or API behaviour, attach a minimal request. -->

```bash
curl -s http://10.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiMo-v2.6-Flash",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

</details>

<details>
  <summary>Logs</summary>

<!--
      Paste the output below, and say whether it came from ./start.sh, logs/mimo26.log,
      the worker's container, or a client.

      Please check these first — they are the failures this kit sees most:

        * "REFUSING TO LAUNCH ... keeps no MXFP4 expert layout" — the base image
          predates sgl-project/sglang#40448. Rebuild from the pinned BASE_IMAGE.
          An older engine allocates the experts as FP8 (~151 GiB/rank) and takes
          the node down.

        * Boot dies in MultiLayerEagleMultiStepDraftExtendCudaGraphRunner with a
          Triton CompilationError — decode graph capture does not work under
          EAGLE MTP. Set CUDA_GRAPH_MAX_BS_DECODE=0. This is not a memory
          failure. See the README's "The EAGLE MTP failure".

        * Context smaller than expected — read max_total_num_tokens= from the
          boot log, not MAX_TOTAL_TOKENS from .env. The pin is silently clamped
          to the budget. SWA_FULL_TOKENS_RATIO dominates that budget; 0.03 is
          the shipped value and 0.3 costs ~10x more.

        * "MEMORY GUARD: MemAvailable ... killing sglang" — the boot exceeded the
          floor. Lower MEM_FRACTION_STATIC, or MAX_TOTAL_TOKENS, and do not raise
          CHUNKED_PREFILL_SIZE at a high memory fraction.

        * Node rebooted instead of erroring — on GB10 host RAM is GPU memory and
          driver allocations have no OOM killer. Keep MEM_GUARD_GIB > 0, keep a
          swap file, and keep vm.swappiness=0.

        * Decode much slower than the README tables — check which drafter is live
          (grep speculative_algorithm in the log) and whether it matches your
          workload: DFlash wins code by ~52%, EAGLE MTP wins prose by ~36%.
-->

```

```

</details>
