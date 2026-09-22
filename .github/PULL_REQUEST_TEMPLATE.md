## Description and Motivation

<!--

    What does this PR change, remove or add, and why?

    For this kit, a good description usually covers:
      * which knob, default, script behaviour or engine patch changes
      * whether it moves a measured number (KV pool size, TTFT, prefill or
        decode throughput, accepted drafts per step) and in which direction
      * why it is safe on BOTH ranks — the worker runs the same boot.py and the
        same preflight, and a change that only works on the head will hang the
        pair in a collective
      * for any performance claim: the protocol, not just the result

-->

## Related Issues

<!--

    Issues this PR relates to, and which it resolves (e.g. "Fixes #12").

-->

---

## Testing

<!--

    How did you verify this? For this kit that usually means some of:

      * `python3 -m unittest tests.test_boot_args` — arg planner and memory
        preflight, no GPU or torch needed
      * `bash tests/test_stop_restart.sh` — stop/restart guard rails, safe to
        run against a live fleet
      * the in-image patch checks:
        `docker run --rm --entrypoint python3 -v "$PWD:/w:ro" -w /w $BASE_IMAGE tests/test_engine_patches.py`
      * `bash -n start.sh stop.sh` for shell changes
      * an actual boot, with the boot-log facts it produced: the `preflight:`
        line, `Load weight end`, `max_total_num_tokens=`, the capture sizes,
        and `available_gpu_mem=`
      * for throughput claims: bench/decode.py or bench/cold_prefill.py with the
        receipt, or sparkDash. Same --seed across the arms being compared.

    Two things this kit has learned the hard way, so please confirm them:

      * Do not edit start.sh while a ./start.sh command is running — bash reads
        scripts incrementally and will fail mid-parse.
      * If your change touches memory, say what the boot log reported for
        available_gpu_mem and whether MEM_GUARD_GIB stayed quiet. On GB10 an
        over-allocation reboots the node rather than raising.

-->

---

## Checklist

<!--

    Thanks for contributing to Mia's AI Lab!

    Put an x in each box, like this: [x].

-->

- [ ] I have read the README and kept my change consistent with it.
- [ ] The title and description say what changed, not just "Update README.md".
- [ ] I verified this on hardware, or explained precisely why it could not be.
- [ ] Defaults still boot out of the box; any new knob has a fallback matching the style of the existing ones, and is listed in the README's "Defaults" table.
- [ ] `.env` and `.env.example` stay in sync.
- [ ] I updated the README for any knob, default or measured number I changed, and did not leave a superseded figure behind.
- [ ] Performance claims come with their protocol and a receipt, and unverified ones are labelled as such.
