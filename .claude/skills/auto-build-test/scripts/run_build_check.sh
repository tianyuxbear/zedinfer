#!/usr/bin/env bash
set -u

LOG_FILE="logs/build.log"
DEBUG_DIR="docs/debug"

mkdir -p logs "${DEBUG_DIR}"

: > "${LOG_FILE}"

run_cmd() {
  local cmd="$1"

  echo "===== ${cmd} =====" >> "${LOG_FILE}"
  bash -lc "${cmd}" >> "${LOG_FILE}" 2>&1
  local status=$?
  echo "" >> "${LOG_FILE}"
  return ${status}
}

run_cmd "xmake clean --all" || exit 1
run_cmd "xmake f -m release --nv-gpu=y --pytest=y" || exit 1
run_cmd "xmake build zedinfer_ops" || exit 1
run_cmd "xmake build ping" || exit 1
run_cmd "xmake run ping" || exit 1

exit 0