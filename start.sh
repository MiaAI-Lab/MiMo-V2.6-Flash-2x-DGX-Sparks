#!/usr/bin/env bash
# start.sh — MiMo-V2.6-Flash-RL on 2x DGX Spark (GB10 / SM121), TP2/EP2, SGLang.
#
# Profile: .env (copied from .env.example on first run), state/, logs/
#
# Usage:
#   ./start.sh                 doctor -> image -> share -> serve
#   ./start.sh doctor
#   ./start.sh pull            pull $BASE_IMAGE on every node
#   ./start.sh build
#   ./start.sh download        hf download the pinned revision
#   ./start.sh share           NFSv4 export + worker volumes
#   ./start.sh serve
#   ./start.sh restart         stop, wait for the GPU, serve again
#   ./start.sh stop | status | logs | smoke
#
# OFFLOAD_MODE=off (default) keeps weights resident.
# OFFLOAD_MODE=ssd spills the HiCache L3 tier to $SSD_DIR on each node.
# OFFLOAD_MODE=ram|cpu is refused.
#
# The image is rebuilt (and re-sent to every worker) whenever its
# `mimo26.base` label differs from $BASE_IMAGE, so a stale engine cannot be
# launched by accident. The engine must keep MiMo's MXFP4 experts packed
# (sgl-project/sglang#40448, 2026-09-20); boot.py checks that and the memory
# fit before it starts SGLang. See README "Why the first boots crashed".
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ENV_FILE="${ENV_FILE:-$ROOT/.env}"
ENV_EXAMPLE="${ENV_EXAMPLE:-$ROOT/.env.example}"
STATE_DIR="${STATE_DIR:-$ROOT/state}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
SERVE_LOG="${SERVE_LOG:-$LOG_DIR/mimo26.log}"
export ENV_FILE ENV_EXAMPLE STATE_DIR LOG_DIR SERVE_LOG
if [[ ! -f "$ENV_FILE" ]]; then
  [[ -f "$ENV_EXAMPLE" ]] || { echo "missing $(basename "$ENV_EXAMPLE")" >&2; exit 1; }
  cp "$ENV_EXAMPLE" "$ENV_FILE"
  echo "[mimo26] wrote $(basename "$ENV_FILE") from $(basename "$ENV_EXAMPLE") — edit IPs if needed"
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

HEAD_IP="${HEAD_IP:-10.0.0.1}"
_list() { tr ',' ' ' <<<"$1"; }
if [[ -n "${WORKER_IPS:-}" ]]; then
  read -r -a WORKER_IPS <<<"$(_list "$WORKER_IPS")"
else
  echo "WORKER_IPS is required" >&2
  exit 1
fi
if [[ -n "${WORKER_HOSTS:-}" ]]; then
  read -r -a WORKER_HOSTS <<<"$(_list "$WORKER_HOSTS")"
else
  WORKER_HOSTS=("${WORKER_IPS[@]}")
fi
[[ ${#WORKER_HOSTS[@]} -eq ${#WORKER_IPS[@]} ]] || { echo "WORKER_HOSTS and WORKER_IPS differ in length" >&2; exit 1; }
WORKER_USER="${WORKER_USER:-zurih}"
SSH_IDENTITY="${SSH_IDENTITY:-$HOME/.ssh/id_ed25519_shared}"
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enP7s7}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enP7s7}"
IB_HCA="${IB_HCA:-rocep1s0f0,rocep1s0f1}"
NCCL_NET="${NCCL_NET:-IB}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NCCL_HOST_DIR="${NCCL_HOST_DIR:-$HOME/nccl-2.30.7}"
NCCL_PIP_SO="${NCCL_PIP_SO:-/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2}"
NCCL_OVERLAY_PIP="${NCCL_OVERLAY_PIP:-0}"

MODEL_DIR="${MODEL_DIR:-$HOME/NewModels/MiMo-V2.6-Flash-RL}"
COMMON_MODEL="${COMMON_MODEL:-/var/tmp/MiMo-V2.6-Flash-RL}"
HF_REPO="${HF_REPO:-XiaomiMiMo/MiMo-V2.6-Flash-RL}"
HF_REVISION="${HF_REVISION:-3b38d063180c3e4aed9691fdc735f3d10b266ee4}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-64}"

BASE_IMAGE="${BASE_IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20260921-0f6761b5}"
IMAGE="${IMAGE:-mimo26-spark:local}"
HEAD_CTN="${HEAD_CTN:-mimo26-head}"
WORKER_CTN="${WORKER_CTN:-mimo26-worker}"
WORKER_DIR="${WORKER_DIR:-/home/${WORKER_USER}/mimo26}"

NNODES="${NNODES:-$(( ${#WORKER_IPS[@]} + 1 ))}"
[[ "$NNODES" -eq $(( ${#WORKER_IPS[@]} + 1 )) ]] || { echo "NNODES=$NNODES but ${#WORKER_IPS[@]} workers are configured" >&2; exit 1; }
TP_SIZE="${TP_SIZE:-$NNODES}"
EP_SIZE="${EP_SIZE:-$TP_SIZE}"
[[ "$TP_SIZE" -eq "$NNODES" ]] || { echo "TP_SIZE=$TP_SIZE but NNODES=$NNODES (one GPU per Spark)" >&2; exit 1; }
DIST_PORT="${DIST_PORT:-20000}"
PORT="${PORT:-8888}"
OFFLOAD_MODE="${OFFLOAD_MODE:-off}"
case "$OFFLOAD_MODE" in
  off|ssd) ;;
  ram|cpu) echo "OFFLOAD_MODE=$OFFLOAD_MODE pins host RAM, which is GPU memory on Spark. Use off or ssd." >&2; exit 1 ;;
  *) echo "OFFLOAD_MODE must be off or ssd" >&2; exit 1 ;;
esac
SSD_DIR="${SSD_DIR:-$HOME/mimo26-ssd}"
WORKER_SSD_DIR="${WORKER_SSD_DIR:-$WORKER_DIR/ssd}"
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-4}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
HEAD_MEM_FRACTION_STATIC="${HEAD_MEM_FRACTION_STATIC:-$MEM_FRACTION_STATIC}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-32768}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-262144}"
SPEC_ALGO="${SPEC_ALGO:-EAGLE}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiMo-v2.6-Flash}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
API_KEY="${API_KEY:-}"
REMOTE_PY="$ROOT/scripts/remote.py"
NFS_VOLUME="${NFS_VOLUME:-mimo26-weights}"
NFS_SHARE="${NFS_SHARE:-1}"
MODEL_MOUNT="/models/MiMo-V2.6-Flash-RL"

_abs() { readlink -f "$1" 2>/dev/null || echo "$1"; }
MODEL_DIR="$(_abs "$MODEL_DIR")"
SSH_IDENTITY="$(_abs "$SSH_IDENTITY")"
NCCL_HOST_DIR="$(_abs "$NCCL_HOST_DIR")"
SSD_DIR="$(_abs "$SSD_DIR")"

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; NC=$'\033[0m'
info() { echo "${GREEN}[+]${NC} $*"; }
warn() { echo "${YELLOW}[!]${NC} $*"; }
err()  { echo "${RED}[x]${NC} $*" >&2; }
die()  { err "$@"; exit 1; }

mkdir -p "$LOG_DIR" "$STATE_DIR"

# shellcheck source=files/nfs-share.sh
source "$ROOT/files/nfs-share.sh"
# shellcheck source=files/nccl.sh
source "$ROOT/files/nccl.sh"

remote_on() {
  local host="$1"; shift
  local to_args=()
  if [[ "${1:-}" == "--timeout" ]]; then
    to_args=(--timeout "$2"); shift 2
  elif [[ -n "${REMOTE_TIMEOUT:-}" ]]; then
    to_args=(--timeout "$REMOTE_TIMEOUT")
  fi
  python3 "$REMOTE_PY" --env-file "$ENV_FILE" \
    --host "$host" --user "$WORKER_USER" \
    --identity "$SSH_IDENTITY" \
    "${to_args[@]}" \
    "bash -lc $(printf '%q' "$*")"
}
remote_ok_on() { remote_on "$@" >/dev/null 2>&1; }

ssh_rsync_e() {
  printf 'ssh -i %q -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o BatchMode=yes' \
    "$SSH_IDENTITY"
}

api_key() {
  local v="${API_KEY:-}"
  case "${v,,}" in
    ''|none|off|dummy|0) echo "" ;;
    *) echo "$v" ;;
  esac
}

# The image records the engine it was built from (Dockerfile LABEL). A
# mismatch with $BASE_IMAGE means a rebuild, on the head and on every worker.
image_base_local() {
  docker image inspect -f '{{index .Config.Labels "mimo26.base"}}' "$IMAGE" 2>/dev/null || true
}

image_base_remote() {
  remote_on "$1" "docker image inspect -f '{{index .Config.Labels \"mimo26.base\"}}' $(printf '%q' "$IMAGE") 2>/dev/null || true" \
    2>/dev/null | tr -d '\r' | tail -n1
}

image_current_local() {
  [[ "$(image_base_local)" == "$BASE_IMAGE" ]]
}

image_current_remote() {
  [[ "$(image_base_remote "$1")" == "$BASE_IMAGE" ]]
}

# boot.py check: engine keeps MXFP4 experts packed, and the per-rank
# resident estimate fits the static budget. Runs in the image, no GPU.
engine_check_local() {
  docker run --rm --entrypoint python3 \
    -v "$MODEL_DIR:$MODEL_MOUNT:ro" \
    -v "$ROOT/boot.py:/opt/mimo26/boot.py:ro" \
    -e "MODEL_PATH=$MODEL_MOUNT" \
    -e "TP_SIZE=$TP_SIZE" -e "EP_SIZE=$EP_SIZE" -e "NNODES=$NNODES" \
    -e "MEM_FRACTION_STATIC=$MEM_FRACTION_STATIC" \
    "$IMAGE" /opt/mimo26/boot.py check 2>/dev/null
}

gid_index_local() {
  local ip="$1" hex hca g i
  hex=$(printf '%02x%02x:%02x%02x' $(echo "$ip" | tr . ' '))
  local IFS=','
  for hca in $IB_HCA; do
    hca="${hca// /}"
    [[ -d /sys/class/infiniband/"$hca" ]] || continue
    for g in /sys/class/infiniband/"$hca"/ports/1/gids/*; do
      i=${g##*/}
      [[ "$(cat /sys/class/infiniband/"$hca"/ports/1/gid_attrs/types/"$i" 2>/dev/null)" == "RoCE v2" ]] || continue
      case "$(cat "$g")" in *ffff:"$hex") echo "$i"; return 0;; esac
    done
  done
  return 1
}

gid_index_remote() {
  local host="$1" ip="$2" hex
  hex=$(printf '%02x%02x:%02x%02x' $(echo "$ip" | tr . ' '))
  remote_on "$host" "for hca in \$(echo $IB_HCA | tr ',' ' '); do
    [ -d /sys/class/infiniband/\$hca ] || continue;
    for g in /sys/class/infiniband/\$hca/ports/1/gids/*; do
      i=\${g##*/};
      [ \"\$(cat /sys/class/infiniband/\$hca/ports/1/gid_attrs/types/\$i 2>/dev/null)\" = 'RoCE v2' ] || continue;
      case \$(cat \$g) in *ffff:$hex) echo \$i; exit 0;; esac;
    done;
  done"
}

# IPv4 mapped into a RoCE GID (...:ffff:hhhh:hhhh). Link-local is ignored.
gid_ipv4() {
  local gid="$1" a b
  [[ "$gid" =~ ffff:([0-9a-fA-F]{4}):([0-9a-fA-F]{4})$ ]] || return 1
  a="${BASH_REMATCH[1]}"
  b="${BASH_REMATCH[2]}"
  printf '%d.%d.%d.%d\n' "$((16#${a:0:2}))" "$((16#${a:2:2}))" "$((16#${b:0:2}))" "$((16#${b:2:2}))"
}

roce_rows_local() {
  local IFS=',' entry hca g i gid ip
  for entry in $IB_HCA; do
    hca="${entry// /}"
    [[ -d /sys/class/infiniband/$hca/ports/1 ]] || continue
    for g in /sys/class/infiniband/$hca/ports/1/gids/*; do
      i=${g##*/}
      [[ "$(cat "/sys/class/infiniband/$hca/ports/1/gid_attrs/types/$i" 2>/dev/null)" == "RoCE v2" ]] || continue
      gid=$(cat "$g")
      ip=$(gid_ipv4 "$gid") || continue
      [[ "$ip" == 0.0.0.0 || "$ip" == 169.254.* ]] && continue
      printf '%s %s %s\n' "$hca" "$i" "$ip"
    done
  done
}

roce_rows_remote() {
  local host="$1"
  remote_on "$host" "for hca in \$(echo $IB_HCA | tr ',' ' '); do
    [ -d /sys/class/infiniband/\$hca/ports/1 ] || continue
    for g in /sys/class/infiniband/\$hca/ports/1/gids/*; do
      i=\${g##*/}
      [ \"\$(cat /sys/class/infiniband/\$hca/ports/1/gid_attrs/types/\$i 2>/dev/null)\" = 'RoCE v2' ] || continue
      gid=\$(cat \$g)
      case \$gid in
        *ffff:????:????)
          hex=\${gid##*ffff:}
          hi=\${hex%%:*}
          lo=\${hex##*:}
          ip=\$(printf '%d.%d.%d.%d' 0x\${hi:0:2} 0x\${hi:2:2} 0x\${lo:0:2} 0x\${lo:2:2})
          case \$ip in 0.0.0.0|169.254.*) continue;; esac
          echo \$hca \$i \$ip
          ;;
      esac
    done
  done"
}

runtime_env() {
  local -n _a=$1
  local hip="$2" gid="$3" rank="$4" mem="$5" ssd_host="$6" hca="${7:-$IB_HCA}"
  _a=(
    -e "NODE_RANK=$rank"
    -e "NNODES=$NNODES"
    -e "TP_SIZE=$TP_SIZE"
    -e "EP_SIZE=$EP_SIZE"
    -e "DP_SIZE=${DP_SIZE:-1}"
    -e "DIST_INIT_ADDR=${HEAD_IP}:${DIST_PORT}"
    -e "OFFLOAD_MODE=$OFFLOAD_MODE"
    -e "SSD_MOUNT=/ssd"
    -e "HICACHE_SIZE_GB=$HICACHE_SIZE_GB"
    -e "HICACHE_WRITE_POLICY=${HICACHE_WRITE_POLICY:-write_through}"
    -e "HICACHE_PREFETCH=${HICACHE_PREFETCH:-timeout}"
    -e "MODEL_PATH=$MODEL_MOUNT"
    -e "STATE_PATH=/state"
    -e "SERVER_PORT=$PORT"
    -e "HOST=0.0.0.0"
    -e "CONTEXT_LENGTH=$CONTEXT_LENGTH"
    -e "MEM_FRACTION_STATIC=$mem"
    -e "MAX_RUNNING_REQUESTS=$MAX_RUNNING_REQUESTS"
    -e "CHUNKED_PREFILL_SIZE=$CHUNKED_PREFILL_SIZE"
    -e "PAGE_SIZE=${PAGE_SIZE:-64}"
    -e "MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-}"
    -e "CUDA_GRAPH_MAX_BS_DECODE=${CUDA_GRAPH_MAX_BS_DECODE:-}"
    -e "KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}"
    -e "SWA_FULL_TOKENS_RATIO=${SWA_FULL_TOKENS_RATIO:-0.3}"
    -e "ATTENTION_BACKEND=${ATTENTION_BACKEND:-triton}"
    -e "MM_ATTENTION_BACKEND=${MM_ATTENTION_BACKEND:-triton_attn}"
    -e "MOE_A2A_BACKEND=${MOE_A2A_BACKEND:-none}"
    -e "MOE_DENSE_TP_SIZE=${MOE_DENSE_TP_SIZE:-1}"
    -e "MOE_RUNNER_BACKEND=${MOE_RUNNER_BACKEND:-flashinfer_mxfp4}"
    -e "WEIGHT_LOADER_DROP_CACHE=${WEIGHT_LOADER_DROP_CACHE:-1}"
    -e "MEM_GUARD_GIB=${MEM_GUARD_GIB:-4}"
    -e "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=${SGLANG_FLASHINFER_MOE_FUSED_FINALIZE:-0}"
    -e "SKIP_PREFLIGHT=${SKIP_PREFLIGHT:-0}"
    -e "SPEC_ALGO=$SPEC_ALGO"
    -e "SPEC_STEPS=${SPEC_STEPS:-3}"
    -e "SPEC_TOPK=${SPEC_TOPK:-1}"
    -e "SPEC_DRAFT_TOKENS=${SPEC_DRAFT_TOKENS:-4}"
    -e "SPEC_DRAFT_MODEL=${SPEC_DRAFT_MODEL:-}"
    -e "ENABLE_MTP=${ENABLE_MTP:-1}"
    -e "REASONING_PARSER=${REASONING_PARSER:-mimo}"
    -e "TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-mimo}"
    -e "WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-1800}"
    -e "SLEEP_ON_IDLE=${SLEEP_ON_IDLE:-1}"
    -e "SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
    -e "ENABLE_CACHE_REPORT=${ENABLE_CACHE_REPORT:-1}"
    -e "ENABLE_METRICS=${ENABLE_METRICS:-1}"
    -e "SKIP_SMOKE=${SKIP_SMOKE:-0}"
    -e "NCCL_NET=$NCCL_NET"
    -e "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
    -e "NCCL_IB_HCA=$hca"
    -e "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
    -e "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
    -e "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
    -e "NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE"
    -e "NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1}"
    -e "NCCL_IB_MERGE_NICS=${NCCL_IB_MERGE_NICS:-0}"
    -e "NCCL_CUMEM_ENABLE=0"
    -e "NCCL_DEBUG=$NCCL_DEBUG"
    -e "NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-1048576}"
    -e "NCCL_LL128_BUFFSIZE=${NCCL_LL128_BUFFSIZE:-262144}"
    -e "NCCL_PROTO=${NCCL_PROTO:-^LL128}"
    -e "NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-8}"
    -e "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
    -e "CUDA_DEVICE_ORDER=PCI_BUS_ID"
    -e "VLLM_HOST_IP=$hip"
    -e "HOST_IP=$hip"
    -e "NCCL_IB_GID_INDEX=$gid"
    -v "$ssd_host:/ssd"
  )
  if [[ -n "${API_KEY}" ]]; then
    _a+=(-e "API_KEY=$API_KEY")
  fi
  if [[ -n "${EXTRA_SGLANG_ARGS:-}" ]]; then
    _a+=(-e "EXTRA_SGLANG_ARGS=$EXTRA_SGLANG_ARGS")
  fi
}

cmd_doctor() {
  local strict="${1:-}"
  echo "=== doctor (MiMo-V2.6-Flash-RL · ${NNODES}× Spark · TP${TP_SIZE} · SGLang) ==="
  echo "head:     $(hostname) / $(whoami) @ $HEAD_IP"
  echo "workers:  ${WORKER_USER}@${WORKER_HOSTS[*]}  (${WORKER_IPS[*]})"
  echo "image:    $IMAGE  (base $BASE_IMAGE)"
  echo "model:    $MODEL_DIR"
  echo "parallel: nnodes=$NNODES TP=$TP_SIZE EP=$EP_SIZE  ctx=$CONTEXT_LENGTH  util=$MEM_FRACTION_STATIC (head $HEAD_MEM_FRACTION_STATIC)"
  echo "offload:  $OFFLOAD_MODE  hicache=${HICACHE_SIZE_GB}GiB  spec=$SPEC_ALGO  port=$PORT"
  echo
  local bad=0
  if (( 64 % TP_SIZE != 0 || 4 % TP_SIZE != 0 || 8 % TP_SIZE != 0 || 256 % TP_SIZE != 0 )); then
    warn "TP=$TP_SIZE does not divide heads=64, GA kv=4, SWA kv=8, or experts=256"
  fi
  if [[ "$TP_SIZE" != 4 ]]; then
    info "fused qkv is stored TP=4-interleaved; the pinned engine reshards it onto TP=$TP_SIZE (log: 'Resolved deferred qkv scale_inv')"
  fi
  if [[ "$OFFLOAD_MODE" == ssd ]]; then
    info "SSD dir head=$SSD_DIR workers=$WORKER_SSD_DIR (HiCache file tier, ${HICACHE_SIZE_GB} GiB host staging per rank)"
  fi
  command -v docker >/dev/null || { err "docker missing"; bad=1; }
  [[ -d /dev/infiniband ]] || { warn "no /dev/infiniband on head"; bad=1; }
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    warn "image $IMAGE missing — $(profile_bin) build"
  elif image_current_local; then
    info "image present, built from $BASE_IMAGE"
    if [[ -f "$MODEL_DIR/config.json" ]]; then
      local check_out
      if check_out=$(engine_check_local); then
        info "engine: $check_out"
      else
        err "engine check failed: $check_out"
        bad=1
      fi
    fi
  else
    warn "image $IMAGE was built from '$(image_base_local)', profile wants $BASE_IMAGE — $(profile_bin) build (serve does it automatically)"
  fi
  if [[ -f "$MODEL_DIR/config.json" ]]; then
    local n
    n=$(find "$MODEL_DIR" -maxdepth 1 -name 'model_pp0_ep*_shard0.safetensors' | wc -l | tr -d ' ')
    if [[ "$n" -ge "$EXPECTED_SHARDS" \
          && -s "$MODEL_DIR/preprocessor_config.json" \
          && -s "$MODEL_DIR/audio_tokenizer/model.safetensors" \
          && -s "$MODEL_DIR/model_mtp.safetensors" ]]; then
      info "omni checkpoint: $n expert shards, vision preprocessor, audio tokenizer, MTP"
    elif [[ "$n" -ge "$EXPECTED_SHARDS" ]]; then
      warn "language shards are present but the omni files are not (preprocessor_config.json, audio_tokenizer/, model_mtp.safetensors)"
      bad=1
    else
      warn "checkpoint has $n expert shards, expected $EXPECTED_SHARDS — $(profile_bin) download"
    fi
  else
    warn "no checkpoint at $MODEL_DIR — $(profile_bin) download"
  fi
  local h
  for h in "${WORKER_HOSTS[@]}"; do
    if remote_ok_on "$h" "docker info >/dev/null && test -d /dev/infiniband"; then
      info "ssh+docker+IB $h"
      if image_current_remote "$h"; then
        info "image on $h built from $BASE_IMAGE"
      else
        warn "image on $h is '$(image_base_remote "$h")' (want $BASE_IMAGE) — $(profile_bin) build resends it"
      fi
    else
      err "ssh/docker/IB failed on $h"
      bad=1
    fi
  done
  if [[ "$strict" == strict && "$bad" -ne 0 ]]; then
    die "doctor found blocking issues"
  fi
  if [[ "$bad" -ne 0 ]]; then
    warn "doctor: issues found"
    return 1
  fi
  info "doctor: ready"
}

# Is $MODEL_DIR a complete omni checkpoint? Called before sharing it to the
# workers and after a download. Checks the files boot.py's prepare() insists on
# plus every expert shard the index references, because a truncated download
# leaves valid-looking JSON next to missing weights.
local_model_has_weights() {
  local name
  for name in config.json model.safetensors.index.json preprocessor_config.json \
              audio_tokenizer/model.safetensors model_mtp.safetensors; do
    [[ -f "$MODEL_DIR/$name" ]] || { err "missing $name in $MODEL_DIR"; return 1; }
  done
  local shards
  shards=$(python3 - "$MODEL_DIR" <<'PYEOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text())
missing = sorted({f for f in set(index["weight_map"].values()) if not (root / f).is_file()})
print(len(missing))
for name in missing[:5]:
    print(name, file=sys.stderr)
PYEOF
  ) || { err "cannot read model.safetensors.index.json in $MODEL_DIR"; return 1; }
  if [[ "$shards" != 0 ]]; then
    err "$shards expert shard(s) referenced by the index are missing from $MODEL_DIR"
    return 1
  fi
  return 0
}

cmd_download() {
  info "downloading $HF_REPO @ $HF_REVISION → $MODEL_DIR"
  mkdir -p "$MODEL_DIR"
  if command -v hf >/dev/null 2>&1; then
    hf download "$HF_REPO" --revision "$HF_REVISION" --local-dir "$MODEL_DIR"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$HF_REPO" --revision "$HF_REVISION" --local-dir "$MODEL_DIR"
  else
    python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$HF_REPO", revision="$HF_REVISION", local_dir="$MODEL_DIR")
PY
  fi
  local_model_has_weights || die "download finished but shards are incomplete"
  info "download ok"
}

cmd_pull() {
  info "docker pull $BASE_IMAGE"
  docker pull "$BASE_IMAGE"
  local h
  for h in "${WORKER_HOSTS[@]}"; do
    info "docker pull on $h"
    remote_on "$h" --timeout 0 "docker pull $(printf '%q' "$BASE_IMAGE")"
  done
}

sync_recipe() {
  local h
  for h in "${WORKER_HOSTS[@]}"; do
    info "sync recipe to $h:$WORKER_DIR"
    remote_on "$h" "mkdir -p $(printf '%q' "$WORKER_DIR")"
    rsync -a --delete \
      --exclude state --exclude logs \
      --exclude .env --exclude __pycache__ \
      -e "$(ssh_rsync_e)" \
      "$ROOT/" "${WORKER_USER}@${h}:${WORKER_DIR}/"
  done
}

cmd_build() {
  info "=== build $IMAGE from $ROOT (base $BASE_IMAGE) ==="
  docker build --build-arg "BASE_IMAGE=$BASE_IMAGE" -t "$IMAGE" "$ROOT"
  image_current_local || die "built image does not carry label mimo26.base=$BASE_IMAGE"
  local h
  sync_recipe
  for h in "${WORKER_HOSTS[@]}"; do
    info "send $IMAGE to $h"
    docker save "$IMAGE" | ssh -i "$SSH_IDENTITY" \
      -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
      "${WORKER_USER}@${h}" docker load
  done
}

cmd_share() {
  [[ "$NFS_SHARE" == 1 ]] || die "NFS_SHARE must stay 1: every worker loads $MODEL_DIR from the head"
  info "=== share head checkpoint $MODEL_DIR to ${#WORKER_HOSTS[@]} worker(s) over NFSv4 ==="
  local_model_has_weights || die "checkpoint incomplete at $MODEL_DIR"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE missing on the head — $(profile_bin) build"
  local h
  for h in "${WORKER_HOSTS[@]}"; do
    remote_ok_on "$h" "docker image inspect $(printf '%q' "$IMAGE") >/dev/null 2>&1" \
      || die "image $IMAGE missing on $h — $(profile_bin) build"
    remote_on "$h" "docker rm -f $(printf '%q' "$WORKER_CTN") >/dev/null 2>&1 || true" || true
  done
  local index_bytes
  index_bytes=$(stat -c %s "$MODEL_DIR/model.safetensors.index.json")
  nfs_ensure_server
  nfs_publish_model
  local idx
  for idx in "${!WORKER_HOSTS[@]}"; do
    h="${WORKER_HOSTS[$idx]}"
    nfs_ensure_worker_volume "$h" "$idx"
    if nfs_worker_has_model "$h" "$index_bytes"; then
      info "$h loads head weights via nfs://${NFS_LAST_ADDR}${NFS_LAST_DEVICE} (index ${index_bytes} bytes)"
    else
      die "$h cannot read the head checkpoint at $MODEL_DIR (nfs://${NFS_LAST_ADDR}${NFS_LAST_DEVICE}). Workers do not keep a local copy."
    fi
  done
  info "all ${#WORKER_HOSTS[@]} workers load weights from the head ($MODEL_DIR)"
}

cmd_serve() {
  DOCTOR_STRICT=0 cmd_doctor || true
  [[ -f "$MODEL_DIR/config.json" ]] || cmd_download
  mkdir -p "$(dirname "$COMMON_MODEL")"
  ln -sfn "$MODEL_DIR" "$COMMON_MODEL"
  if [[ "$OFFLOAD_MODE" == ssd ]]; then
    mkdir -p "$SSD_DIR"
  fi

  if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]' \
     && [[ "${FORCE:-0}" != 1 ]]; then
    die "GPU is busy. Stop the other stack, or FORCE=1"
  fi

  if ! image_current_local; then
    info "image $IMAGE is missing or not built from $BASE_IMAGE — building"
    cmd_build
  fi
  local h
  for h in "${WORKER_HOSTS[@]}"; do
    if ! image_current_remote "$h"; then
      info "image on $h is missing or not built from $BASE_IMAGE — rebuilding and resending"
      cmd_build
      break
    fi
  done
  [[ "$NFS_SHARE" == 1 ]] || die "NFS_SHARE must stay 1: workers load weights from the head"
  local check_out
  if check_out=$(engine_check_local); then
    info "engine: $check_out"
  else
    die "engine check failed (nothing was started): $check_out"
  fi
  sync_recipe
  cmd_share

  API_KEY="$(api_key)"
  export API_KEY
  if [[ -n "$API_KEY" ]]; then
    (umask 077; printf '%s\n' "$API_KEY" > "$STATE_DIR/api-key")
    chmod 600 "$STATE_DIR/api-key"
  else
    rm -f "$STATE_DIR/api-key"
  fi

  # Management addresses are not in the CX7 GID table, and the same HCA name
  # is a different subnet on each Spark. Pick the port pair that shares a /24.
  local GID_HEAD=""
  local -a HEAD_ROCE=() WORKER_GIDS=() WORKER_HCAS=()
  mapfile -t HEAD_ROCE < <(roce_rows_local)
  [[ ${#HEAD_ROCE[@]} -gt 0 ]] || die "no IPv4 RoCE v2 GID on $IB_HCA"
  local HEAD_HCA="" gi line lh li lip rh ri rip lnet rnet picked
  for gi in "${!WORKER_HOSTS[@]}"; do
    local -a REMOTE_ROCE=()
    mapfile -t REMOTE_ROCE < <(roce_rows_remote "${WORKER_HOSTS[$gi]}" | tr -d '\r')
    picked=""
    for line in "${HEAD_ROCE[@]}"; do
      read -r lh li lip <<<"$line"
      lnet=${lip%.*}
      for rline in "${REMOTE_ROCE[@]}"; do
        read -r rh ri rip <<<"$rline"
        rnet=${rip%.*}
        if [[ -n "$lnet" && "$lnet" == "$rnet" && "$lip" != "$rip" ]]; then
          picked="$lh $li $lip $rh $ri $rip"
          break
        fi
      done
      [[ -n "$picked" ]] && break
    done
    [[ -n "$picked" ]] || die "no shared RoCE subnet between this host and ${WORKER_HOSTS[$gi]}"
    read -r lh li lip rh ri rip <<<"$picked"
    if [[ -z "$HEAD_HCA" ]]; then
      HEAD_HCA="$lh"
      GID_HEAD="$li"
    elif [[ "$HEAD_HCA" != "$lh" ]]; then
      HEAD_HCA="${HEAD_HCA},$lh"
    fi
    WORKER_HCAS+=("$rh")
    WORKER_GIDS+=("$ri")
    info "RoCE ${WORKER_HOSTS[$gi]}: $lh gid $li ($lip) ↔ $rh gid $ri ($rip)"
  done
  [[ -n "$GID_HEAD" && -n "$HEAD_HCA" ]] || die "RoCE port pair was not selected"

  # Unified memory: file cache from the NFS weight read plus the CUDA pool
  # rebooted both Sparks. Drop the cache so the load starts from free RAM.
  sync
  echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
  for h in "${WORKER_HOSTS[@]}"; do
    remote_on "$h" "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true" || true
  done

  docker rm -f "$HEAD_CTN" >/dev/null 2>&1 || true
  for h in "${WORKER_HOSTS[@]}"; do
    remote_on "$h" "docker rm -f $(printf '%q' "$WORKER_CTN") >/dev/null 2>&1 || true" || true
  done

  info "Starting workers (ranks 1..${#WORKER_IPS[@]}) first..."
  local idx=0 wip wgid rank
  for h in "${WORKER_HOSTS[@]}"; do
    wip="${WORKER_IPS[$idx]}"
    wgid="${WORKER_GIDS[$idx]}"
    rank=$((idx + 1))
    local -a wenv=()
    runtime_env wenv "$wip" "$wgid" "$rank" "$MEM_FRACTION_STATIC" "$WORKER_SSD_DIR" "${WORKER_HCAS[$idx]}"
    local wenv_q
    wenv_q=$(printf '%q ' "${wenv[@]}")
    remote_on "$h" "
      set -e
      docker volume inspect $(printf '%q' "$NFS_VOLUME") >/dev/null
      test -d /dev/infiniband
      mkdir -p $(printf '%q' "$WORKER_DIR/state") $(printf '%q' "$WORKER_DIR/logs") $(printf '%q' "$WORKER_SSD_DIR")
      nccl_args=()
      $(declare -f nccl_library nccl_mount_args)
      NCCL_OVERLAY_PIP=$(printf '%q' "$NCCL_OVERLAY_PIP")
      NCCL_HOST_DIR=$(printf '%q' "$NCCL_HOST_DIR")
      NCCL_PIP_SO=$(printf '%q' "$NCCL_PIP_SO")
      nccl_mount_args nccl_args
      docker run -d --name $(printf '%q' "$WORKER_CTN") \
        --network host --ipc host --privileged --cap-add IPC_LOCK --gpus all \
        --shm-size $(printf '%q' "${SHM_SIZE:-32g}") \
        --ulimit memlock=-1:-1 --ulimit stack=67108864 \
        --device /dev/infiniband:/dev/infiniband \
        -v $(printf '%q' "$NFS_VOLUME:$MODEL_MOUNT:ro") \
        -v $(printf '%q' "$WORKER_DIR/boot.py:/opt/mimo26/boot.py:ro") \
        -v $(printf '%q' "$WORKER_DIR/patches:/opt/mimo26/patches:ro") \
        -v $(printf '%q' "$WORKER_DIR/state:/state") \
        -v \"\$HOME/.cache:/root/.cache\" \
        \"\${nccl_args[@]}\" \
        $wenv_q \
        $(printf '%q' "$IMAGE") run
    "
    idx=$((idx + 1))
  done

  info "Starting head (rank 0, API :$PORT)..."
  local -a head_env=()
  runtime_env head_env "$HEAD_IP" "$GID_HEAD" 0 "$HEAD_MEM_FRACTION_STATIC" "$SSD_DIR" "$HEAD_HCA"
  local -a nccl_args=()
  nccl_mount_args nccl_args || die "NCCL mount setup failed"
  local head_cid
  head_cid=$(docker run -d --name "$HEAD_CTN" --restart=no \
    --network host --ipc host --privileged --cap-add IPC_LOCK --gpus all \
    --shm-size "${SHM_SIZE:-32g}" \
    --ulimit memlock=-1:-1 --ulimit stack=67108864 \
    --device /dev/infiniband:/dev/infiniband \
    -v "$MODEL_DIR:$MODEL_MOUNT:ro" \
    -v "$ROOT/boot.py:/opt/mimo26/boot.py:ro" \
    -v "$ROOT/patches:/opt/mimo26/patches:ro" \
    -v "$STATE_DIR:/state" \
    -v "$HOME/.cache:/root/.cache" \
    "${nccl_args[@]}" \
    "${head_env[@]}" \
    "$IMAGE" run)
  [[ -n "$head_cid" ]] || die "docker run produced no container id"
  info "head cid=${head_cid:0:12}"

  mkdir -p "$LOG_DIR"
  : >"$SERVE_LOG"
  info "streaming engine log..."
  ( docker logs -f --tail=20 "$HEAD_CTN" 2>&1 | tee -a "$SERVE_LOG" ) &
  echo $! >"$LOG_DIR/logtail.pid"
  _stop_logtail() {
    local p
    p=$(cat "$LOG_DIR/logtail.pid" 2>/dev/null || true)
    rm -f "$LOG_DIR/logtail.pid"
    if [[ -n "${p:-}" ]]; then
      pkill -TERM -P "$p" 2>/dev/null || true
      kill -TERM "$p" 2>/dev/null || true
    fi
  }
  trap '_stop_logtail' EXIT

  local i=0 st
  while (( i < ${READY_TIMEOUT:-360} )); do
    if docker logs "$HEAD_CTN" 2>&1 | grep -q "^Ready: API on port ${PORT}"; then
      _stop_logtail
      trap - EXIT
      ( setsid docker logs -f --since 1s "$HEAD_CTN" >>"$SERVE_LOG" 2>&1 </dev/null & ) 2>/dev/null
      echo
      info "API is up on :$PORT — engine keeps running, this script is done."
      cmd_status
      echo
      echo "  curl http://$HEAD_IP:$PORT/v1/chat/completions \\"
      echo "    -H 'Content-Type: application/json' \\"
      echo "    -d '{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 19 + 23?\"}],\"max_tokens\":32,\"temperature\":1.0,\"top_p\":0.95,\"chat_template_kwargs\":{\"enable_thinking\":false}}'"
      echo
      echo "  logs: $(profile_bin) logs"
      echo "  stop: $(profile_bin) stop"
      return 0
    fi
    st=$(docker inspect -f '{{.State.Status}}' "$HEAD_CTN" 2>/dev/null || echo missing)
    case "$st" in
      running|created|restarting) ;;
      *)
        _stop_logtail
        trap - EXIT
        err "head is $st cid=${head_cid:0:12}"
        tail -n 80 "$SERVE_LOG" 2>/dev/null || true
        exit 1
        ;;
    esac
    i=$((i + 1))
    sleep 10
  done
  _stop_logtail
  trap - EXIT
  die "timed out waiting for :$PORT"
}

profile_bin() { echo ./start.sh; }

cmd_stop() {
  export ENV_FILE STATE_DIR LOG_DIR HEAD_CTN WORKER_CTN
  exec "$ROOT/stop.sh" "$@"
}

cmd_restart() {
  # stop.sh is called, not exec'd: restart has to outlive it and then serve.
  # Its settle wait is what makes this safe rather than a coin flip -- after
  # `docker rm -f` the driver still holds the CUDA context and ~82 GiB for
  # several seconds, and both cmd_serve's GPU-busy check and SGLang's own KV
  # budget (literally MemAvailable on GB10) read that draining state.
  local arg
  for arg in "$@"; do
    case "$arg" in
      --unmount|-u)
        die "restart does not take --unmount: serve would come back up with no worker volume. Use '$(profile_bin) stop --unmount', then share, then serve."
        ;;
    esac
  done
  export ENV_FILE STATE_DIR LOG_DIR HEAD_CTN WORKER_CTN
  info "restart: stopping the current stack"
  "$ROOT/stop.sh" "$@" ||
    die "stop did not finish cleanly — NOT serving. Clear the node reported above, then $(profile_bin) serve"
  info "restart: bringing it back up"
  cmd_serve
}

cmd_status() {
  echo "== head =="
  docker ps --filter "name=$HEAD_CTN" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' || true
  echo
  local h
  local index_bytes=""
  [[ -f "$MODEL_DIR/model.safetensors.index.json" ]] && index_bytes=$(stat -c %s "$MODEL_DIR/model.safetensors.index.json")
  for h in "${WORKER_HOSTS[@]}"; do
    echo "== worker $h =="
    remote_on "$h" "docker ps --filter $(printf '%q' "name=$WORKER_CTN") --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'" || warn "status SSH $h failed"
    if [[ -n "$index_bytes" ]] && nfs_worker_has_model "$h" "$index_bytes"; then
      echo "weights: head $MODEL_DIR via $NFS_VOLUME"
    else
      echo "weights: MISSING (not the head checkpoint on $NFS_VOLUME)"
    fi
    echo
  done
  echo "== API =="
  if curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/models"; then
    echo
  else
    echo "(not responding on :$PORT)"
  fi
}

cmd_logs() {
  local who="${1:-head}" lines="${2:-120}"
  case "$who" in
    ''|head|[0-9]*)
      [[ "$who" =~ ^[0-9]+$ ]] && lines="$who"
      docker logs --tail="$lines" "$HEAD_CTN" 2>&1 || tail -n "$lines" "$SERVE_LOG"
      ;;
    worker*|w[0-9]*)
      local n="${who#worker}"; n="${n#w}"; n="${n:-1}"
      [[ "$n" =~ ^[0-9]+$ && "$n" -ge 1 && "$n" -le ${#WORKER_HOSTS[@]} ]] || die "no such worker: $who"
      remote_on "${WORKER_HOSTS[$((n - 1))]}" "docker logs --tail=$lines $WORKER_CTN" || true
      ;;
    -f|follow)
      docker logs -f "$HEAD_CTN"
      ;;
    *)
      docker logs --tail="$lines" "$HEAD_CTN" 2>&1 || true
      ;;
  esac
}

cmd_smoke() {
  local key auth=()
  key=$(cat "$STATE_DIR/api-key" 2>/dev/null || true)
  [[ -n "$key" ]] && auth=(-H "Authorization: Bearer $key")
  # Use the id the running server actually serves: the profile may have been
  # renamed since it was started.
  local name
  name=$(curl -fsS --max-time 5 "${auth[@]}" "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)
  name="${name:-$SERVED_MODEL_NAME}"
  info "smoke: 19+23 (thinking off) as model '$name'"
  curl -fsS --max-time 180 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    "${auth[@]}" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$name\",\"temperature\":0,\"max_tokens\":32,\"chat_template_kwargs\":{\"enable_thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"What is 19 + 23? Reply only with the number.\"}]}"
  echo
}

usage() {
  sed -n '2,22p' "$0" | tr -d '#'
}

CMD="${1:-serve}"
shift || true
case "$CMD" in
  serve|start|"") cmd_serve "$@" ;;
  doctor) cmd_doctor strict ;;
  pull) cmd_pull ;;
  build) cmd_build ;;
  download) cmd_download ;;
  share|mount) cmd_share ;;
  stop) cmd_stop "$@" ;;
  restart) cmd_restart "$@" ;;
  status) cmd_status ;;
  logs) cmd_logs "$@" ;;
  smoke) cmd_smoke ;;
  -h|--help|help) usage ;;
  *) die "unknown command: $CMD" ;;
esac
