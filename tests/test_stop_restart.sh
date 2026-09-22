#!/usr/bin/env bash
# test_stop_restart.sh — argument handling and guard rails for stop.sh and
# `start.sh restart`. Safe to run against a live fleet: every case either
# fails before touching anything, or acts on a throwaway profile whose
# container and volume names do not exist.
#
#   bash tests/test_stop_restart.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pass=0 fail=0
ok()   { printf '  ok   %s\n' "$1"; pass=$((pass + 1)); }
bad()  { printf '  FAIL %s\n' "$1" >&2; fail=$((fail + 1)); }
check() { if [[ "$2" == "$3" ]]; then ok "$1"; else bad "$1 (got ${2@Q}, want ${3@Q})"; fi; }

# A profile that names nothing real. No workers, so no ssh either.
sed -e 's/^HEAD_CTN=.*/HEAD_CTN=mimo26-selftest-head/' \
    -e 's/^WORKER_CTN=.*/WORKER_CTN=mimo26-selftest-worker/' \
    -e 's/^NFS_VOLUME=.*/NFS_VOLUME=mimo26-selftest-volume/' \
    -e 's/^WORKER_HOSTS=.*/WORKER_HOSTS=/' \
    -e 's/^WORKER_IPS=.*/WORKER_IPS=/' \
    .env.example > "$TMP/env.probe"

echo "stop.sh:"

# --help must print only the comment block. The range used to run one line
# long and leaked `set -uo pipefail` into the help text.
help_out="$(bash stop.sh --help 2>&1)"
case "$help_out" in
  *"set -"*) bad "--help does not leak shell directives" ;;
  *) ok "--help does not leak shell directives" ;;
esac
case "$help_out" in
  *"--wait-timeout"*) ok "--help documents --wait-timeout" ;;
  *) bad "--help documents --wait-timeout" ;;
esac

out="$(ENV_FILE="$TMP/env.probe" bash stop.sh --bogus 2>&1)"; rc=$?
check "unknown arg exits 1" "$rc" "1"
case "$out" in *"unknown arg"*) ok "unknown arg is named" ;; *) bad "unknown arg is named" ;; esac

# Standalone ./stop.sh must find the repo's .env without ENV_FILE being set.
out="$(unset ENV_FILE; bash stop.sh --help 2>&1)"; rc=$?
check "standalone --help exits 0" "$rc" "0"
out="$(ENV_FILE=/tmp/mimo-no-such-profile bash stop.sh 2>&1)"; rc=$?
check "a missing profile file exits 1" "$rc" "1"
case "$out" in
  *"no profile at"*) ok "a missing profile names the path" ;;
  *) bad "a missing profile names the path" ;;
esac

out="$(ENV_FILE="$TMP/env.probe" bash stop.sh --wait-timeout 2>&1)"; rc=$?
check "--wait-timeout with no value exits 1" "$rc" "1"
out="$(ENV_FILE="$TMP/env.probe" bash stop.sh --wait-timeout nope 2>&1)"; rc=$?
check "--wait-timeout with a non-number exits 1" "$rc" "1"

# --no-wait must not consult the GPU at all, so this passes even while a real
# server holds it.
out="$(ENV_FILE="$TMP/env.probe" LOG_DIR="$TMP/logs" bash stop.sh --no-wait 2>&1)"; rc=$?
check "--no-wait on a throwaway profile exits 0" "$rc" "0"
case "$out" in
  *"waiting up to"*) bad "--no-wait skips the settle wait" ;;
  *) ok "--no-wait skips the settle wait" ;;
esac

# A logtail pid file is cleaned up even when the process is already gone.
mkdir -p "$TMP/logs"; echo 999999 > "$TMP/logs/logtail.pid"
ENV_FILE="$TMP/env.probe" LOG_DIR="$TMP/logs" bash stop.sh --no-wait >/dev/null 2>&1
[[ -f "$TMP/logs/logtail.pid" ]] && bad "stale logtail.pid is removed" || ok "stale logtail.pid is removed"

echo "restart:"

run_profile() {
  ENV_FILE="$ROOT/.env" ENV_EXAMPLE="$ROOT/.env.example" \
  STATE_DIR="$TMP/state" LOG_DIR="$TMP/logs" bash "$ROOT/start.sh" "$@" 2>&1
}

# Must refuse before stopping anything: serve would return with no volume.
out="$(run_profile restart --unmount)"; rc=$?
check "restart --unmount exits 1" "$rc" "1"
case "$out" in
  *"does not take --unmount"*) ok "restart --unmount explains itself" ;;
  *) bad "restart --unmount explains itself" ;;
esac
case "$out" in
  *"stopping"*) bad "restart --unmount stops nothing first" ;;
  *) ok "restart --unmount stops nothing first" ;;
esac
out="$(run_profile restart -u)"; rc=$?
check "restart -u is rejected too" "$rc" "1"

# The dispatcher must know the verb, and cmd_restart must not exec stop.sh
# (exec would replace the shell and serve would never run).
grep -q 'restart) cmd_restart' start.sh && ok "dispatcher routes restart" || bad "dispatcher routes restart"
if sed -n '/^cmd_restart()/,/^}/p' start.sh | grep -q 'exec .*stop\.sh'; then
  bad "cmd_restart calls stop.sh without exec"
else
  ok "cmd_restart calls stop.sh without exec"
fi
sed -n '/^cmd_restart()/,/^}/p' start.sh | grep -q 'cmd_serve' \
  && ok "cmd_restart reaches cmd_serve" || bad "cmd_restart reaches cmd_serve"

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[[ "$fail" -eq 0 ]]
