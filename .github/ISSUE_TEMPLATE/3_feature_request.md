---
name: Feature request
about: Suggest a new idea for this kit. Something that doesn't exist currently, and you want to see.
title: ""
labels: "enhancement"
assignees: ""
---

<!-- Thank you for using this model kit!

     If you are looking for support, please check the README first,
     or reach out on X:
      * https://x.com/MiaAI_lab

     If you want the kit to do something it currently doesn't, then fill out
     the template below.
-->

## Use case

<!--
     What workload are you trying to serve, or what got in your way?

     Ideas that would fit: more node counts than two (nothing in the image is
     TP-specific, but only TP=2 is measured), a switched rather than
     point-to-point fabric, a systemd unit or compose file, per-workload drafter
     routing, an OpenAI-compatible client example, prefill-decode disaggregation,
     or a CI lint/validate step.

     Please also say what you considered instead.
-->

## Proposal

<!--
     Briefly but precisely describe what you would like the kit to do.

     Two constraints worth knowing before proposing:

       * A single Spark cannot serve this model at all — 161 GiB of weights on a
         121.7 GiB box — so anything single-node needs a smaller checkpoint.
       * Host RAM is GPU memory here, so features that pin host memory (caches,
         row stores, extra worker processes) compete directly with the KV pool.
         OFFLOAD_MODE=ram/cpu is refused for this reason.

     If it can live as a small user-land script rather than in the kit, consider
     implementing it and opening a pull request instead.
-->
