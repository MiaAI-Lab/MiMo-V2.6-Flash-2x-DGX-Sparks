# Optional host NCCL overlay. Sourced by start.sh. No side effects at import.

nccl_library() {
  local dir="$1" name
  for name in libnccl.so.2.30.7 libnccl.so.2; do
    if [[ -f "$dir/$name" && -r "$dir/$name" ]]; then
      printf '%s\n' "$dir/$name"
      return 0
    fi
  done
  return 1
}

nccl_mount_args() {
  local -n mounts=$1
  local library
  [[ "${NCCL_OVERLAY_PIP:-0}" == 1 ]] || return 0
  if library=$(nccl_library "$NCCL_HOST_DIR"); then
    mounts+=(-v "$library:$NCCL_PIP_SO:ro")
  else
    echo "NCCL: no readable library in $NCCL_HOST_DIR" >&2
    return 1
  fi
}
