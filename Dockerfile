# MiMo-V2.6 stores its 256 routed experts as MXFP4 (U8 nibble pairs plus E8M0
# block scales). SGLang keeps that layout on the GPU only since 2026-09-20
# (sgl-project/sglang#40448, first in the 2026-09-21 nightly). An older base
# allocates the experts as FP8, ~151 GiB per rank at EP=2, and takes the Spark
# down with it. start.sh passes BASE_IMAGE from the profile; boot.py refuses to
# launch on an engine that lacks the packed layout.
ARG BASE_IMAGE=lmsysorg/sglang:nightly-dev-cu13-20261003-65f75914
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
LABEL mimo26.base="${BASE_IMAGE}"
COPY patches/vision-attention.patch patches/dflash-multimodal.patch patches/eagle-swa-draft-extend-cudagraph.patch patches/swa-prefix-window-lock.patch patches/gb10-mimo-extend-tile.patch /opt/mimo26/
# Keep the kernel in its canonical source file for Triton's JIT inspection.
RUN cd /sgl-workspace/sglang/python \
 && printf '%s\n' \
 'f061599bb2c9243d50b565281c47de077ccc90135b9b1e5320a07425bedac112  sglang/kernels/ops/attention/extend_attention.py' \
 'c46492ccb53c64882dfd60a3a9c18b76d08e27c3a6514c4e765761c795a202d7  sglang/srt/environ.py' \
 '03f5ef9efe3b0fdd2175d9bc6e94fb01360e60a9b002e0fb2c17eec68121e570  sglang/srt/mem_cache/unified_cache/components/swa.py' \
 '3c0684320fef964c5320213547a474aaf430c87edf9283de1c9008d89e6e6228  sglang/srt/layers/attention/triton_backend.py' \
 'eb158c4c03e69091eb95a1bce093c9b054ac1ca5aab12769063d2e84630ca7f3  sglang/kernels/ops/attention/prefill_attention.py' \
 'f54da17e49896af7c5e9213675f224a770d31211bd5553ed310fc1ef2002967e  sglang/srt/layers/attention/vision.py' \
 '80a828359da423b13327f2ea28fca9da80bd11d3ca0c9d488cbee656fe0d633e  sglang/srt/models/mimo_v2.py' | sha256sum --check \
 && cp sglang/kernels/ops/attention/prefill_attention.py /opt/mimo26/original_prefill_attention.py \
 && cp sglang/srt/layers/attention/triton_backend.py /opt/mimo26/original_triton_backend.py \
 && cp sglang/srt/models/mimo_v2.py /opt/mimo26/original_mimo_v2.py \
 && patch --batch --fuzz=0 -p1 < /opt/mimo26/vision-attention.patch \
 && patch --batch --fuzz=0 -p1 < /opt/mimo26/dflash-multimodal.patch \
 && patch --batch --fuzz=0 -p1 < /opt/mimo26/eagle-swa-draft-extend-cudagraph.patch \
 && patch --batch --fuzz=0 -p1 < /opt/mimo26/swa-prefix-window-lock.patch \
 && patch --batch --fuzz=0 -p1 < /opt/mimo26/gb10-mimo-extend-tile.patch \
 && python3 -m py_compile sglang/kernels/ops/attention/extend_attention.py sglang/srt/environ.py sglang/srt/mem_cache/unified_cache/components/swa.py sglang/srt/layers/attention/triton_backend.py sglang/kernels/ops/attention/prefill_attention.py sglang/srt/layers/attention/vision.py sglang/srt/models/mimo_v2.py
LABEL mimo26.vision_fix="inclusive-window-real-key0-bias"
LABEL mimo26.dflash_fix="multimodal-dispatch-independent-of-aux-capture"
LABEL mimo26.eagle_swa_draft_extend="swa-window-buffers"
LABEL mimo26.swa_window_lock="41580-page-aligned-window"
LABEL mimo26.gb10_prefill_tile="mimo-tp2-32x64-w4"
# Image, video, and audio. The MiMo processor decodes audio with torchcodec
# and shells out to ffprobe for video.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir torchcodec \
 && python3 -c "import torchaudio; from torchcodec.decoders import AudioDecoder; import shutil; assert shutil.which('ffprobe'), 'ffprobe missing'"
WORKDIR /opt/mimo26
COPY boot.py /opt/mimo26/boot.py
ENV MODEL_PATH=/models/MiMo-V2.6-Flash-RL \
    STATE_PATH=/state \
    OFFLOAD_MODE=off \
    PYTHONUNBUFFERED=1
EXPOSE 8888
HEALTHCHECK --interval=30s --timeout=30s --start-period=40m --retries=3 \
    CMD ["python3", "-S", "/opt/mimo26/boot.py", "health"]
ENTRYPOINT ["python3", "-u", "/opt/mimo26/boot.py"]
CMD ["run"]
