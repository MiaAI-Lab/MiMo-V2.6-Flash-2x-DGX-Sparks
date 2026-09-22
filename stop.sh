#!/usr/bin/env bash
# stop.sh — tear down the fleet. Reads .env, or $ENV_FILE if set.
#   ./stop.sh                  (same as ./start.sh stop)
#   ./stop.sh --unmount        also drop the worker NFS volumes
#   ./stop.sh --no-wait        do not wait for the driver to release the GPU
#   ./stop.sh --wait-timeout N settle deadline per node (default 180s)
# Exit 0 once every rank's container is gone and, unless --no-wait, the GPU
# has been released on every node. Exit 2 if a node never settled.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

UNMOUNT=0
WAIT=1
WAIT_TIMEOUT="${STOP_WAIT_TIMEOUT:-180}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --unmount|-u) UNMOUNT=1 ;;
    --no-wait) WAIT=0 ;;
    --wait-timeout)
      shift
      [[ "${1:-}" =~ ^[0-9]+$ ]] || { echo "--wait-timeout needs seconds" >&2; exit 1; }
      WAIT_TIMEOUT="$1"
      ;;
    -h|--help)
      sed -n '2,8p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 1
      ;;
  esac
  shift
done

# Runs standalone (./stop.sh) or via ./start.sh stop, which exports ENV_FILE.
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "no profile at $ENV_FILE (run ./start.sh once to create it from .env.example)" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

HEAD_CTN="${HEAD_CTN:-mimo26-head}"
WORKER_CTN="${WORKER_CTN:-mimo26-worker}"
WORKER_USER="${WORKER_USER:-zurih}"
SSH_IDENTITY="${SSH_IDENTITY:-$HOME/.ssh/id_ed25519_shared}"
_list() { tr ',' ' ' <<<"$1"; }
read -r -a WORKER_HOSTS <<<"$(_list "${WORKER_HOSTS:-}")"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
REMOTE_PY="$ROOT/scripts/remote.py"

remote_on() {
  local host="$1"; shift
  python3 "$REMOTE_PY" --env-file "$ENV_FILE" --host "$host" --user "$WORKER_USER" \
    --identity "$SSH_IDENTITY" \
    "bash -lc $(printf '%q' "$*")"
}

# `docker rm -f` returns as soon as the container record is gone, but the
# NVIDIA driver still holds the CUDA context and the ~82 GiB of weights behind
# it for several seconds. Two things read that state immediately after a stop:
# cmd_serve's GPU-busy check (nvidia-smi --query-compute-apps), and SGLang's
# own startup budget, which on GB10 is literally the OS MemAvailable. So a
# restart that does not wait either refuses to launch or sizes its KV pool
# against a budget that is still draining. Wait for the compute apps to clear,
# then for MemAvailable to stop climbing (two reads within 256 MiB).
read -r -d '' SETTLE_BODY <<'SNIP'
deadline=$(( SECONDS + ${SETTLE_TIMEOUT:-180} ))
last=0
while (( SECONDS < deadline )); do
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
    sleep 2
    continue
  fi
  now=$(awk '/^MemAvailable:/ { print int($2 / 1024) }' /proc/meminfo)
  if (( last > 0 && now >= last && now - last < 256 )); then
    echo "settled, MemAvailable ${now} MiB, no compute apps"
    exit 0
  fi
  last=$now
  sleep 2
done
echo "NOT settled within ${SETTLE_TIMEOUT:-180}s" >&2
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null | head -5 >&2
exit 1
SNIP

settle_local() { SETTLE_TIMEOUT="$WAIT_TIMEOUT" bash -c "$SETTLE_BODY"; }
settle_remote() { remote_on "$1" "SETTLE_TIMEOUT=$WAIT_TIMEOUT; $SETTLE_BODY"; }

echo "stopping $HEAD_CTN and $WORKER_CTN on ${WORKER_HOSTS[*]:-no workers}"
if [[ -f "$LOG_DIR/logtail.pid" ]]; then
  p=$(cat "$LOG_DIR/logtail.pid" 2>/dev/null || true)
  [[ -n "${p:-}" ]] && kill -TERM "$p" 2>/dev/null || true
  rm -f "$LOG_DIR/logtail.pid"
fi
# Head first: it is the HTTP endpoint, so dropping it stops new requests being
# admitted. Either order leaves the surviving rank in a collective for the
# ssh round-trip, but this way nothing is accepted that cannot be completed.
docker rm -f "$HEAD_CTN" >/dev/null 2>&1 || true
for h in "${WORKER_HOSTS[@]}"; do
  remote_on "$h" "docker rm -f $(printf '%q' "$WORKER_CTN") >/dev/null 2>&1 || true" || true
done

stragglers=0
if [[ "$WAIT" == 1 ]]; then
  echo "waiting up to ${WAIT_TIMEOUT}s per node for the driver to release the GPU"
  printf '  head: '
  settle_local || stragglers=1
  for h in "${WORKER_HOSTS[@]}"; do
    printf '  %s: ' "$h"
    settle_remote "$h" || stragglers=1
  done
fi

if [[ "$UNMOUNT" == 1 ]]; then
  info() { echo "[+] $*"; }
  die() { echo "[x] $*" >&2; exit 1; }
  # shellcheck source=files/nfs-share.sh
  source "$ROOT/files/nfs-share.sh"
  nfs_unmount_workers
fi

if [[ "$stragglers" != 0 ]]; then
  echo "stopped, but at least one node never released the GPU." >&2
  echo "Do not serve yet: check 'nvidia-smi' there, and if a probe hangs that" >&2
  echo "node needs a reboot (see the CUDA graphs / recovery notes in README)." >&2
  exit 2
fi
echo "stopped"
