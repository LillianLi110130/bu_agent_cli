#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
PLATFORM_NAME="linux"
BUILD_ROOT="${PROJECT_ROOT}/build/pyinstaller/${PLATFORM_NAME}"
DIST_ROOT="${PROJECT_ROOT}/dist/standalone/${PLATFORM_NAME}"
SPEC_PATH="${PROJECT_ROOT}/packaging/tg-agent.spec"
BINARY_NAME="tg-agent"
BUILT_BINARY="${DIST_ROOT}/${BINARY_NAME}"
PACKAGE_ROOT="${DIST_ROOT}/package"

rm -rf "${BUILD_ROOT}" "${PACKAGE_ROOT}"

"${PYTHON_BIN}" -m PyInstaller --version >/dev/null
"${PYTHON_BIN}" -m PyInstaller \
  --noconfirm \
  --clean \
  --distpath "${DIST_ROOT}" \
  --workpath "${BUILD_ROOT}" \
  "${SPEC_PATH}"

mkdir -p "${PACKAGE_ROOT}"
cp "${BUILT_BINARY}" "${PACKAGE_ROOT}/${BINARY_NAME}"
cp "${SCRIPT_DIR}/install-tg-agent.sh" "${PACKAGE_ROOT}/install-tg-agent.sh"
cp "${PROJECT_ROOT}/tg_crab_worker.json" "${PACKAGE_ROOT}/tg_crab_worker.json"
chmod +x "${PACKAGE_ROOT}/${BINARY_NAME}" "${PACKAGE_ROOT}/install-tg-agent.sh"

echo "Standalone package ready: ${PACKAGE_ROOT}"
