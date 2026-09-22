# MiMo-V2.6 stores its 256 routed experts as MXFP4 (U8 nibble pairs plus E8M0
# block scales). SGLang keeps that layout on the GPU only since 2026-09-20
# (sgl-project/sglang#40448, first in the 2026-09-21 nightly). An older base
# allocates the experts as FP8, ~151 GiB per rank at EP=2, and takes the Spark
# down with it. start.sh passes BASE_IMAGE from the profile; boot.py refuses to
# launch on an engine that lacks the packed layout.
ARG BASE_IMAGE=lmsysorg/sglang:nightly-dev-cu13-20260921-0f6761b5
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
LABEL mimo26.base="${BASE_IMAGE}"
COPY patches/vision-attention.patch patches/dflash-multimodal.patch /opt/mimo26/
# Keep the kernel in its canonical source file for Triton's JIT inspection.
RUN cd /sgl-workspace/sglang/python \
 && printf '%s\n' \
 'eb158c4c03e69091eb95a1bce093c9b054ac1ca5aab12769063d2e84630ca7f3  sglang/kernels/ops/attention/prefill_attention.py' \
 '00ff69ab2f2fdfb74e319789aad47291bf84ba675d6bf227de4e3a5c5d5ca192  sglang/srt/layers/attention/vision.py' \
 'a3dd3c7e9d59345c1e435cf43bf569ed1e5dd0ce75a588881283399de8e79656  sglang/srt/models/mimo_v2.py' | sha256sum --check \
 && cp sglang/kernels/ops/attention/prefill_attention.py /opt/mimo26/original_prefill_attention.py \
 && cp sglang/srt/models/mimo_v2.py /opt/mimo26/original_mimo_v2.py \
 && patch --batch --fuzz=0 -p1 < /opt/mimo26/vision-attention.patch \
 && patch --batch --fuzz=0 -p1 < /opt/mimo26/dflash-multimodal.patch \
 && python3 -m py_compile sglang/kernels/ops/attention/prefill_attention.py sglang/srt/layers/attention/vision.py sglang/srt/models/mimo_v2.py
LABEL mimo26.vision_fix="inclusive-window-real-key0-bias"
LABEL mimo26.dflash_fix="multimodal-dispatch-independent-of-aux-capture"
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
