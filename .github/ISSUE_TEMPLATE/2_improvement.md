---
name: Improvement proposal
about: General improvements to the kit. Anything that exists and works, but could be enhanced or optimized.
title: ""
labels: "enhancement"
assignees: ""
---

<!-- Thank you for using this model kit!

     If you are looking for support, please check the README first,
     or reach out on X:
      * https://x.com/MiaAI_lab

     If you think something can be improved, then fill out the template below.
-->

---

## Improvement description

<!--
      What do you think can be improved, and how does the kit benefit?

      Things that would fit this repo, and where the README already says the work
      is unfinished:

        * CHUNKED_PREFILL_SIZE has never been laddered on this engine. 2048 came
          from a sibling kit; 4096 is an untested rung.
        * Making decode CUDA graphs work under EAGLE MTP. Four candidate fixes
          are listed in "The EAGLE MTP failure" and none has been tried.
        * A drafter sweep on the remaining categories (JSON, structured, math,
          counting) — only code and prose are measured for both drafters.
        * Long-context prefill and needle tests past 256K, now that the KV pool
          holds a full 1M context.
        * Comm and kernel knobs that are still at their defaults:
          enable_fused_moe_sum_all_reduce, flashinfer allreduce fusion,
          SGLANG_FLASHINFER_AUTOTUNE_EXTEND, the Triton attention split/kv-split
          settings, num_continuous_decode_steps.
        * Attention backends other than triton on SM121.
        * Boot-shape warmup so Triton kernels compile once per image instead of
          JIT-ing mid-serve.
-->

---

### Additional context

<details>
<summary>Code sample</summary>

```bash

```

</details>

<details>
  <summary>Measurements</summary>

<!--
      Numbers make the case. If the change affects throughput, please include
      before/after in the README's format: aggregate and per-stream tok/s with
      TTFT at a fixed stream count, and for drafter changes the accepted drafts
      per step. bench/decode.py and bench/cold_prefill.py produce exactly this
      and write JSON receipts.
-->

```

```

</details>
