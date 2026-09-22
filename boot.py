"""Serve XiaomiMiMo/MiMo-V2.6-Flash-RL with SGLang on DGX Spark.

Weights stay GPU-resident unless OFFLOAD_MODE=ssd, which spills the HiCache
L3 tier to a per-rank directory on local NVMe. CPU/RAM offload is refused:
on GB10 host RAM is the same pool as the GPU.

The 256 routed experts are stored MXFP4 (U8 nibble pairs plus E8M0 block
scales, ~150 GiB for all of them). SGLang keeps that layout on the GPU only
since 2026-09-20 (sgl-project/sglang#40448). An older engine allocates the
experts as FP8, about 151 GiB per rank at EP=2, and takes the Spark down
with it: driver allocations on unified memory have no OOM killer, the box
hangs and reboots. preflight() refuses to launch on such an engine, and
whenever the per-rank resident estimate does not fit the static budget.
"""
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request

REPO = 'XiaomiMiMo/MiMo-V2.6-Flash-RL'
REVISION = '3b38d063180c3e4aed9691fdc735f3d10b266ee4'
MODEL = Path(os.environ.get('MODEL_PATH', '/models/MiMo-V2.6-Flash-RL'))
STATE = Path(os.environ.get('STATE_PATH', '/state'))
PORT = int(os.environ.get('SERVER_PORT', '8888'))
NNODES = int(os.environ.get('NNODES', '1'))
NODE_RANK = int(os.environ.get('NODE_RANK', '0'))
TP_SIZE = int(os.environ.get('TP_SIZE', '2'))
EP_SIZE = int(os.environ.get('EP_SIZE', str(TP_SIZE)))
DIST_INIT_ADDR = os.environ.get('DIST_INIT_ADDR', '')
SERVED_MODEL_NAME = os.environ.get('SERVED_MODEL_NAME', 'MiMo-v2.6-Flash')

# First public image with sgl-project/sglang#40448 (MiMo-V2 MXFP4 experts).
MIN_BASE_IMAGE = 'lmsysorg/sglang:nightly-dev-cu13-20260921-0f6761b5'

# safetensors dtype tags → bytes per element.
DTYPE_BYTES = {
    'F8_E4M3': 1, 'F8_E5M2': 1, 'E8M0': 1, 'U8': 1, 'I8': 1, 'BOOL': 1,
    'BF16': 2, 'F16': 2, 'U16': 2, 'I16': 2,
    'F32': 4, 'I32': 4, 'U32': 4,
    'F64': 8, 'I64': 8, 'U64': 8,
}
# Encoders are counted as replicated on every rank (an over-estimate of at
# most ~1 GiB if SGLang shards them).
REPLICATED_TAGS = ('vision', 'visual', 'audio')
# CUDA context, tokenizer/detokenizer processes, NCCL buffers, FlashInfer
# workspace (measured ~4.2 GiB between the preflight and "Load weight begin").
ENGINE_OVERHEAD_GIB = 5.0
# Smallest pool worth booting with: the EAGLE draft (3 MTP layers, ~1.5 GiB
# weights+KV) plus a few GiB of target KV (~5.6 KiB/token/rank fp8 on the
# full-attention layers, plus the SWA pool).
MIN_KV_GIB = 4.0
# SGLang reserves the multimodal embedding cache out of the KV budget
# (SGLANG_VLM_CACHE_SIZE_MB, default 100).
MM_RESERVE_GIB = 0.1
# Userspace OOM guard (boot.py memory_guard): MemAvailable floor in GiB.
MEM_GUARD_GIB_DEFAULT = 4.0


def save(name, value):
    STATE.mkdir(parents=True, exist_ok=True)
    temporary = STATE / (name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(STATE / name)


def key():
    value = os.environ.get('API_KEY', '').strip()
    if value.lower() in ('', 'none', 'off', 'dummy', '0'):
        return ''
    STATE.mkdir(parents=True, exist_ok=True)
    path = STATE / 'api-key'
    path.write_text(value)
    path.chmod(0o600)
    return value


def request(path, payload=None, timeout=10):
    headers = {'Content-Type': 'application/json'}
    secret = key()
    if secret:
        headers['Authorization'] = 'Bearer ' + secret
    req = urllib.request.Request(
        f'http://127.0.0.1:{PORT}' + path,
        headers=headers,
        data=None if payload is None else json.dumps(payload).encode())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body) if body else {}


def prepare():
    required = (
        'config.json',
        'model.safetensors.index.json',
        'preprocessor_config.json',
        'audio_tokenizer/model.safetensors',
        'model_mtp.safetensors',
    )
    missing = [name for name in required if not (MODEL / name).is_file()]
    if missing:
        raise RuntimeError(f'{MODEL} is not the full omni checkpoint, missing {missing}')
    config = json.loads((MODEL / 'config.json').read_text())
    if 'vision_config' not in config or 'audio_config' not in config:
        raise RuntimeError(f'{MODEL}/config.json has no vision_config and audio_config')
    print(f'using omni checkpoint at {MODEL} (text, image, video, audio)', flush=True)


# ─── memory preflight ────────────────────────────────────────────────────────

def safetensors_header(path):
    with open(path, 'rb') as handle:
        length = int.from_bytes(handle.read(8), 'little')
        return json.loads(handle.read(length))


def drafter_files(model):
    """Safetensors the selected speculative algorithm adds to each rank.

    EAGLE uses model_mtp.safetensors (3 MTP layers). DFLASH loads the separate
    Qwen3 draft model under dflash/ and never reads the MTP head. SPEC_ALGO=off
    loads neither.
    """
    spec = os.environ.get('SPEC_ALGO', 'EAGLE').strip()
    if spec.lower() in ('off', 'none', '0', ''):
        return []
    if spec.upper() == 'DFLASH':
        folder = Path(os.environ.get('SPEC_DRAFT_MODEL', '').strip() or (model / 'dflash'))
        return sorted(folder.glob('*.safetensors')) if folder.is_dir() else []
    head = model / 'model_mtp.safetensors'
    return [head] if head.is_file() else []


def resident_estimate(model, tp):
    """Bytes each rank keeps resident, from the safetensors headers as stored.

    experts: '.mlp.experts.' tensors, split across ranks (EP=TP, or TP slices
    when EP<TP; both divide by tp). sharded: the rest of the language model
    and the drafter, tensor-parallel. replicated: vision/audio encoders.
    """
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    files = [model / name for name in sorted(set(index['weight_map'].values()))]
    # Count the drafter that will actually be loaded. EAGLE reads the 1.2 GiB
    # MTP head; DFLASH ignores it and loads the 2.9 GiB Qwen3 draft model
    # instead, so counting the wrong one under-estimates the rank by ~1.8 GiB
    # -- and this estimate is what decides whether the boot is refused.
    for drafter in drafter_files(model):
        if drafter not in files:
            files.append(drafter)
    experts = sharded = replicated = 0
    for path in files:
        for tensor, meta in safetensors_header(path).items():
            if tensor == '__metadata__':
                continue
            nbytes = math.prod(meta['shape']) * DTYPE_BYTES[meta['dtype']]
            if '.mlp.experts.' in tensor:
                experts += nbytes
            elif any(tag in tensor for tag in REPLICATED_TAGS):
                replicated += nbytes
            else:
                sharded += nbytes
    gib = 2 ** 30
    experts_rank = experts / tp
    sharded_rank = sharded / tp
    per_rank = experts_rank + sharded_rank + replicated
    return {
        'files': len(files),
        'tp': tp,
        'experts_packed_gib': experts / gib,
        'experts_per_rank_gib': experts_rank / gib,
        'sharded_per_rank_gib': sharded_rank / gib,
        'replicated_gib': replicated / gib,
        'per_rank_gib': per_rank / gib,
        # FP8 expansion doubles the packed expert weights (E8M0 scales are ~3%).
        'per_rank_if_fp8_experts_gib': (per_rank + experts_rank) / gib,
    }


_ENGINE_PROBE = r'''
import inspect, json, warnings
warnings.filterwarnings("ignore")
out = {"supported": False, "version": "?", "detail": ""}
try:
    import sglang
    out["version"] = str(getattr(sglang, "__version__", "?"))
    from sglang.srt.configs import model_config
    from sglang.srt.models import mimo_v2
    has_cfg = "store_dtype" in inspect.getsource(model_config)
    has_load = "torch.uint8" in inspect.getsource(mimo_v2.MiMoV2ForCausalLM.load_weights)
    out["supported"] = bool(has_cfg and has_load)
    out["detail"] = f"model_config store_dtype={has_cfg} mimo_v2 uint8 experts={has_load}"
except Exception as exc:
    out["detail"] = f"{type(exc).__name__}: {exc}"
print("ENGINE " + json.dumps(out))
'''


def engine_keeps_mxfp4_experts():
    """Probe the installed SGLang in a throwaway interpreter.

    Importing sglang here would keep torch and a CUDA context alive in this
    supervisor for the life of the server; a subprocess keeps it lean.
    """
    proc = subprocess.run(
        [sys.executable, '-c', _ENGINE_PROBE],
        capture_output=True, text=True, timeout=900)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith('ENGINE '):
            return json.loads(line[len('ENGINE '):])
    tail = (proc.stderr or '').strip().splitlines()[-3:]
    return {'supported': False, 'version': '?', 'detail': 'probe failed: ' + ' / '.join(tail)}


def meminfo_gib():
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, value = line.split(':', 1)
        mem[name] = int(value.split()[0]) * 1024
    return mem['MemTotal'] / 2 ** 30, mem['MemAvailable'] / 2 ** 30


def evaluate_fit(est, supported, mem_total_gib, mem_available_gib, mem_fraction,
                 overhead_gib=ENGINE_OVERHEAD_GIB, min_kv_gib=MIN_KV_GIB):
    """Return (problems, summary). Empty problems means launch.

    SGLang's pool on GB10 (kv_cache_configurator._profile_available_bytes):
    MemAvailable after the weights, minus the slack kept outside the pool
    for activations, (memory before the load) × (1 - mem_fraction_static),
    minus the multimodal cache reservation. The EAGLE draft and its KV come
    out of the same pool.
    """
    per_rank = est['per_rank_gib'] if supported else est['per_rank_if_fp8_experts_gib']
    pre_load = max(mem_available_gib - overhead_gib, 0.0)
    reserve = pre_load * (1 - mem_fraction) + MM_RESERVE_GIB
    kv = pre_load - per_rank - reserve
    problems = []
    if not supported:
        problems.append(
            f'this SGLang keeps no MXFP4 expert layout for MiMo-V2 and would allocate the '
            f'routed experts as FP8: ~{per_rank:.0f} GiB per rank on a {mem_total_gib:.0f} GiB '
            f'Spark. Rebuild from BASE_IMAGE={MIN_BASE_IMAGE} or newer '
            f'(./start.sh build)')
    if kv < min_kv_gib:
        problems.append(
            f'MemAvailable {mem_available_gib:.0f} GiB minus {overhead_gib:.0f} GiB engine '
            f'overhead, weights ~{per_rank:.0f} GiB and the {reserve:.0f} GiB kept outside '
            f'the pool ({pre_load:.0f} GiB x (1 - {mem_fraction}) + {MM_RESERVE_GIB:.0f} '
            f'multimodal) leaves {kv:.1f} GiB for the draft and KV; need at least '
            f'{min_kv_gib:.0f} GiB. Raise MEM_FRACTION_STATIC or free host memory')
    if mem_available_gib < per_rank + overhead_gib:
        problems.append(
            f'MemAvailable is {mem_available_gib:.0f} GiB, below the {per_rank + overhead_gib:.0f} GiB '
            f'this rank needs; something else is holding unified memory')
    summary = (
        f'weights ~{per_rank:.1f} GiB/rank (experts {est["experts_per_rank_gib"]:.1f} '
        f'{"packed MXFP4" if supported else "EXPANDED TO FP8"} + '
        f'{est["sharded_per_rank_gib"]:.1f} sharded + {est["replicated_gib"]:.1f} encoders), '
        f'MemAvailable {mem_available_gib:.0f} GiB, {reserve:.0f} GiB kept outside the pool '
        f'at {mem_fraction}, ~{max(kv, 0):.0f} GiB for the draft and KV')
    return problems, summary


def memory_guard(process, floor_gib, interval=0.25):
    """Kill the engine when MemAvailable drops under floor_gib.

    On a Spark, GPU allocations come from host RAM through the NVIDIA
    driver. Running out does not end in a CUDA OOM: the node stalls and
    reboots (four times on 2026-09-22). This thread is the OOM killer the
    driver path lacks; it fires while the kernel can still recover.
    """
    if floor_gib <= 0:
        return None

    def run():
        while process.poll() is None:
            try:
                _, available = meminfo_gib()
            except Exception:
                time.sleep(interval)
                continue
            if available < floor_gib:
                print(
                    f'MEMORY GUARD: MemAvailable {available:.1f} GiB is under the '
                    f'{floor_gib:.1f} GiB floor; killing sglang (pgid {process.pid}) '
                    f'before the node goes down', flush=True)
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return
            time.sleep(interval)

    thread = threading.Thread(target=run, name='memory-guard', daemon=True)
    thread.start()
    return thread


def preflight():
    tp = int(os.environ.get('TP_SIZE', '2'))
    mem_fraction = float(os.environ.get('MEM_FRACTION_STATIC', '0.80'))
    est = resident_estimate(MODEL, tp)
    engine = engine_keeps_mxfp4_experts()
    total, available = meminfo_gib()
    problems, summary = evaluate_fit(est, engine['supported'], total, available, mem_fraction)
    line = f'preflight: sglang {engine["version"]}, {summary}'
    if problems:
        raise RuntimeError(
            line + '\nREFUSING TO LAUNCH: ' + ' | '.join(problems) + f' [{engine["detail"]}]')
    print(line, flush=True)
    return line


# ─── launch ──────────────────────────────────────────────────────────────────

def _flag(name, default):
    value = os.environ.get(name, default).strip()
    return value


def dflash_draft_dir():
    """Path to hand SGLang for --speculative-draft-model-path under DFLASH.

    The published dflash/config.json has a trailing comma before its closing
    brace, so it is not valid JSON and transformers refuses it. The checkpoint
    is mounted read-only (and shared to the worker over NFS), so instead of
    editing it we build a sibling directory under /state that symlinks every
    file and substitutes a re-serialized config.json. Returns the original
    directory untouched when its config already parses.
    """
    # start.sh forwards SPEC_DRAFT_MODEL= unconditionally, so an empty value
    # means "unset" here -- _flag would otherwise hand back '' and resolve to
    # the current directory.
    source = Path(_flag('SPEC_DRAFT_MODEL', '') or (MODEL / 'dflash'))
    if not source.is_dir():
        raise RuntimeError(
            f'SPEC_ALGO=DFLASH needs the draft model at {source}, which does not '
            f'exist. Set SPEC_DRAFT_MODEL, or use SPEC_ALGO=EAGLE (the MTP head '
            f'in model_mtp.safetensors, which is what this fleet has tested).')
    config = source / 'config.json'
    try:
        parsed = json.loads(config.read_text())
    except FileNotFoundError:
        raise RuntimeError(f'{config} is missing')
    except json.JSONDecodeError as exc:
        parsed = None
        reason = exc
    else:
        return source
    repaired = STATE / 'dflash'
    repaired.mkdir(parents=True, exist_ok=True)
    # json5 is not in the image; the only defect seen is a trailing comma, so
    # strip commas that sit before a closing brace or bracket and re-parse. If
    # that does not yield valid JSON, refuse rather than guess.
    text = re.sub(r',(\s*[}\]])', r'\1', config.read_text())
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError(
            f'{config} is not valid JSON ({reason}) and the trailing-comma repair '
            f'did not fix it. Fix the file or point SPEC_DRAFT_MODEL at a good copy.')
    for entry in source.iterdir():
        link = repaired / entry.name
        if entry.name == 'config.json':
            continue
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(entry)
    (repaired / 'config.json').write_text(json.dumps(parsed, indent=2) + '\n')
    print(f'dflash: {config} is invalid JSON ({reason}); serving a repaired copy '
          f'from {repaired} (other files symlinked)', flush=True)
    return repaired


def build_args():
    node_rank = int(os.environ.get('NODE_RANK', '0'))
    nnodes = int(os.environ.get('NNODES', '1'))
    tp_size = int(os.environ.get('TP_SIZE', '2'))
    ep_size = int(os.environ.get('EP_SIZE', str(tp_size)))
    dist_init = os.environ.get('DIST_INIT_ADDR', '')
    served = os.environ.get('SERVED_MODEL_NAME', SERVED_MODEL_NAME)
    mode = os.environ.get('OFFLOAD_MODE', 'off').strip().lower()
    if mode in ('ram', 'cpu'):
        raise RuntimeError(
            'OFFLOAD_MODE=ram/cpu pins host memory, which on DGX Spark is GPU memory. '
            'Use off (resident weights) or ssd (HiCache file tier on local NVMe).')
    if mode not in ('off', 'ssd'):
        raise RuntimeError('OFFLOAD_MODE must be off or ssd')

    context = int(os.environ.get('CONTEXT_LENGTH', '262144'))
    if not 4096 <= context <= 1048576:
        raise RuntimeError('CONTEXT_LENGTH must be 4096 through the model limit 1048576')

    # Flag set of the MiMo-V2.6 SGLang cookbook, tp/ep/nnodes from the Spark
    # profile. --dp / --enable-dp-attention / --mm-enable-dp-encoder only
    # apply when DP_SIZE > 1 (one GPU per Spark cannot also split data-parallel).
    dp_size = int(os.environ.get('DP_SIZE', '1'))
    args = [
        '--trust-remote-code',
        '--model-path', str(MODEL),
        '--served-model-name', served,
        '--enable-multimodal',
        '--tp', str(tp_size),
    ]
    if dp_size > 1:
        args += [
            '--dp', str(dp_size),
            '--enable-dp-attention',
            '--mm-enable-dp-encoder',
        ]
    # MoE: the cookbook's deepep all-to-all needs NVSHMEM over the fabric; on
    # two Sparks the plain EP path (a2a none, all-reduce) is the proven one.
    # The MXFP4 experts need a runner that consumes the packed layout:
    # flashinfer_mxfp4 (SM120 cutlass, MXFP8 activations; used by the
    # DeepSeek-V4.1 profile on this fleet) or marlin (W4A16).
    args += [
        '--ep', str(ep_size),
        '--moe-a2a-backend', _flag('MOE_A2A_BACKEND', 'none'),
        '--moe-dense-tp-size', _flag('MOE_DENSE_TP_SIZE', '1'),
        '--mem-fraction-static', _flag('MEM_FRACTION_STATIC', '0.80'),
        '--max-running-requests', _flag('MAX_RUNNING_REQUESTS', '8'),
        '--chunked-prefill-size', _flag('CHUNKED_PREFILL_SIZE', '32768'),
        '--page-size', _flag('PAGE_SIZE', '64'),
        '--swa-full-tokens-ratio', _flag('SWA_FULL_TOKENS_RATIO', '0.3'),
        '--context-length', str(context),
        '--reasoning-parser', _flag('REASONING_PARSER', 'mimo'),
        '--tool-call-parser', _flag('TOOL_CALL_PARSER', 'mimo'),
        '--host', os.environ.get('HOST', '0.0.0.0'),
        '--port', str(PORT),
    ]
    # Speculative decoding. The checkpoint ships TWO drafters:
    #   EAGLE   model_mtp.safetensors, 1.2 GiB, 3 MTP layers. Tested here.
    #   DFLASH  dflash/, 2.9 GiB Qwen3 draft model. Untested on this fleet.
    # SPEC_DRAFT_TOKENS is the verify window for both: EAGLE spends it on
    # --speculative-num-draft-tokens, and SGLang documents
    # --speculative-dflash-block-size as an alias of that same flag for DFLASH,
    # so one knob covers each.
    spec = _flag('SPEC_ALGO', 'EAGLE')
    if spec and spec.lower() not in ('off', 'none', '0'):
        args += [
            '--speculative-algorithm', spec,
            '--speculative-num-draft-tokens', _flag('SPEC_DRAFT_TOKENS', '4'),
        ]
        if spec.strip().upper() == 'DFLASH':
            args += ['--speculative-draft-model-path', str(dflash_draft_dir())]
        else:
            args += [
                '--speculative-num-steps', _flag('SPEC_STEPS', '3'),
                '--speculative-eagle-topk', _flag('SPEC_TOPK', '1'),
            ]
            if os.environ.get('ENABLE_MTP', '1') == '1':
                args += ['--enable-multi-layer-eagle']
    # GB10: the cookbook leaves attention on the server default (FA3/FA4).
    # Those kernels are not the SM121 path, so Triton stays on unless cleared.
    attn = _flag('ATTENTION_BACKEND', 'triton')
    if attn and attn.lower() not in ('auto', 'none', 'off', '0'):
        args += ['--attention-backend', attn]
    mm_attn = _flag('MM_ATTENTION_BACKEND', 'triton_attn')
    if mm_attn and mm_attn.lower() not in ('auto', 'none', 'off', '0'):
        args += ['--mm-attention-backend', mm_attn]
    moe_runner = _flag('MOE_RUNNER_BACKEND', 'flashinfer_mxfp4')
    if moe_runner and moe_runner.lower() not in ('auto', 'none', 'off', '0'):
        args += ['--moe-runner-backend', moe_runner]
    # CUDA graphs. SGLang's own per-phase defaults are decode=full and
    # prefill=breakable (model_executor/cuda_graph_config.py: CudaGraphConfig),
    # so dropping --disable-cuda-graph turns BOTH phases on -- the decode
    # max-bs flag alone does not select a phase. Decode is the lever worth
    # having: EAGLE runs 4 model passes per step (3 draft + verify) across 48
    # layers, and uncaptured that is thousands of eager launches per token.
    # Prefill (BCG) capture is a separate untested path here and wants the same
    # memory the prefill activation transient does, so it stays off unless
    # CUDA_GRAPH_PREFILL=1: one phase at a time.
    graph_bs = _flag('CUDA_GRAPH_MAX_BS_DECODE', '')
    if graph_bs == '0':
        args += ['--disable-cuda-graph']
    else:
        if graph_bs:
            args += ['--cuda-graph-max-bs-decode', graph_bs]
        if os.environ.get('CUDA_GRAPH_PREFILL', '0') != '1':
            args += ['--disable-prefill-cuda-graph']
    # Unified memory: page cache from the mmap'd shards competes with the
    # GPU pool. Release each shard's cache once its tensors are on the GPU.
    if os.environ.get('WEIGHT_LOADER_DROP_CACHE', '1') != '0':
        args += ['--weight-loader-drop-cache-after-load']
    # The shards stay mmap'd (reclaimable page cache); patches/sitecustomize.py
    # materializes one tensor at a time for the copy to the GPU. Do not add
    # --weight-loader-disable-mmap here: it turns every in-flight shard into
    # anonymous memory (25 GiB for ep0) and took both Sparks down.
    args += [
        '--watchdog-timeout', _flag('WATCHDOG_TIMEOUT', '1800'),
        '--random-seed', '0',
    ]
    # Reporting the bench harnesses read. --enable-cache-report is in the
    # upstream V2.6 recipe and puts prompt_tokens_details.cached_tokens on the
    # OpenAI response; --enable-metrics is what makes /metrics exist at all
    # (without it the endpoint 404s and no cache-hit counters are available).
    # Neither is free-standing overhead worth avoiding here.
    if os.environ.get('ENABLE_CACHE_REPORT', '1') != '0':
        args += ['--enable-cache-report']
    if os.environ.get('ENABLE_METRICS', '1') != '0':
        args += ['--enable-metrics']
    kv = _flag('KV_CACHE_DTYPE', 'fp8_e4m3')
    if kv and kv.lower() not in ('auto', 'none', 'off', '0'):
        args += ['--kv-cache-dtype', kv]
    total = _flag('MAX_TOTAL_TOKENS', '')
    if total and total != '0':
        args += ['--max-total-tokens', total]
    if os.environ.get('SLEEP_ON_IDLE', '1') == '1':
        args += ['--sleep-on-idle']

    if nnodes > 1:
        if not dist_init:
            raise RuntimeError('DIST_INIT_ADDR is required when NNODES>1')
        args += [
            '--nnodes', str(nnodes),
            '--node-rank', str(node_rank),
            '--dist-init-addr', dist_init,
        ]

    env = dict(os.environ)
    patch_dir = Path('/opt/mimo26/patches')
    if patch_dir.is_dir():
        previous = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = str(patch_dir) + ((':' + previous) if previous else '')
    if mode == 'ssd':
        root = Path(os.environ.get('SSD_MOUNT', '/ssd'))
        path = root / f'rank{node_rank}'
        path.mkdir(parents=True, exist_ok=True)
        env['SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR'] = str(path)
        args += [
            '--enable-hierarchical-cache',
            '--hicache-storage-backend', 'file',
            '--hicache-write-policy', _flag('HICACHE_WRITE_POLICY', 'write_through'),
            '--hicache-size', _flag('HICACHE_SIZE_GB', '4'),
            '--hicache-storage-prefetch-policy', _flag('HICACHE_PREFETCH', 'timeout'),
        ]
        print(f'SSD offload: HiCache file tier at {path} '
              f'(host staging {env.get("HICACHE_SIZE_GB", os.environ.get("HICACHE_SIZE_GB", "4"))} GiB)',
              flush=True)
    else:
        print('SSD offload off: weights and KV stay in unified memory', flush=True)
    print('omni: text, image, video, audio (--enable-multimodal)', flush=True)

    extra = os.environ.get('EXTRA_SGLANG_ARGS', '').strip()
    if extra:
        args += extra.split()
    return args, env, mode


def smoke():
    result = request('/v1/chat/completions', {
        'model': SERVED_MODEL_NAME,
        'temperature': 0,
        'max_tokens': 32,
        'chat_template_kwargs': {'enable_thinking': False},
        'messages': [{'role': 'user', 'content': 'What is 19 + 23? Reply only with the number.'}],
    }, timeout=600)
    text = result['choices'][0]['message']['content']
    print(f'smoke: {text!r}', flush=True)
    if '42' not in (text or ''):
        print('WARNING: smoke reply did not contain 42', flush=True)


def serve():
    prepare()
    if os.environ.get('SKIP_PREFLIGHT', '0') != '1':
        preflight()
    args, env, mode = build_args()
    save('launch.json', {
        'revision': REVISION, 'repo': REPO, 'offload_mode': mode, 'args': args,
        'nnodes': int(os.environ.get('NNODES', '1')),
        'node_rank': int(os.environ.get('NODE_RANK', '0')),
        'tp': int(os.environ.get('TP_SIZE', '2')),
        'ep': int(os.environ.get('EP_SIZE', os.environ.get('TP_SIZE', '2'))),
    })
    secret = key()
    cmd = [sys.executable, '-m', 'sglang.launch_server', *args]
    if secret:
        cmd += ['--api-key', secret]
    process = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)
    guard_gib = float(os.environ.get('MEM_GUARD_GIB', str(MEM_GUARD_GIB_DEFAULT)))
    if memory_guard(process, guard_gib) is not None:
        print(f'memory guard armed: sglang is killed if MemAvailable < {guard_gib:.1f} GiB',
              flush=True)

    def stop(*_):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def logs():
        for line in process.stdout:
            print(line.replace(secret, '[REDACTED]') if secret else line, end='', flush=True)

    threading.Thread(target=logs, daemon=True).start()
    try:
        if int(os.environ.get('NODE_RANK', '0')) != 0:
            return process.wait()
        deadline = time.monotonic() + int(os.environ.get('READY_TIMEOUT_S', '3600'))
        while process.poll() is None and time.monotonic() < deadline:
            try:
                request('/health', timeout=3)
                break
            except Exception:
                time.sleep(5)
        else:
            raise RuntimeError('Server failed to become healthy; inspect container logs')
        if os.environ.get('SKIP_SMOKE', '0') != '1':
            smoke()
        if secret:
            print(f'Ready: API on port {PORT} (rank 0). Key is in {STATE}/api-key.', flush=True)
        else:
            print(f'Ready: API on port {PORT} (rank 0), no API key.', flush=True)
        return process.wait()
    finally:
        stop()


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else 'run'
    if command == 'health':
        request('/health')
    elif command == 'smoke':
        smoke()
    elif command == 'plan':
        planned, _, mode = build_args()
        print(json.dumps({'offload_mode': mode, 'args': planned}))
    elif command == 'check':
        # Engine + memory preflight only; used by `start.sh doctor|serve`.
        # Errors go to stdout so the caller can show them without stderr noise.
        try:
            prepare()
            preflight()
        except Exception as exc:
            print(str(exc), flush=True)
            sys.exit(1)
    elif command == 'run':
        sys.exit(serve() or 0)
    else:
        raise SystemExit('Commands: run, plan, check, health, smoke')
