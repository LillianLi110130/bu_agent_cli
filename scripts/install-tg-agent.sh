#!/usr/bin/env bash
set -euo pipefail

default_env_content() {
  cat <<'EOF'
# tg-agent runtime configuration
# Fill in OPENAI_API_KEY after install if your shell or workspace does not already provide it.
OPENAI_API_KEY=
LLM_MODEL=GLM-4.7
LLM_BASE_URL=https://open.bigmodel.cn/api/coding/paas/v4
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="${TG_AGENT_HOME:-$HOME/.tg_agent}"
BIN_DIR="${INSTALL_ROOT}/bin"
SOURCE_BIN="${1:-${SCRIPT_DIR}/tg-agent}"
TARGET_BIN="${BIN_DIR}/tg-agent"
ENV_FILE="${INSTALL_ROOT}/.env"
WORKER_CONFIG="${INSTALL_ROOT}/tg_crab_worker.json"
CONFIG_TEMPLATE="${SCRIPT_DIR}/tg_crab_worker.json"

mkdir -p "${BIN_DIR}"
cp "${SOURCE_BIN}" "${TARGET_BIN}"
chmod +x "${TARGET_BIN}"

if [[ -f "${CONFIG_TEMPLATE}" && ! -f "${WORKER_CONFIG}" ]]; then
  cp "${CONFIG_TEMPLATE}" "${WORKER_CONFIG}"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  default_env_content > "${ENV_FILE}"
fi

PROFILE_FILE="${HOME}/.profile"
PATH_EXPORT='export PATH="$HOME/.tg_agent/bin:$PATH"'
if ! grep -Fq "${PATH_EXPORT}" "${PROFILE_FILE}" 2>/dev/null; then
  printf '\n%s\n' "${PATH_EXPORT}" >> "${PROFILE_FILE}"
fi

echo "Installed tg-agent to ${TARGET_BIN}"
echo "User config directory: ${INSTALL_ROOT}"
echo "Config file: ${ENV_FILE}"
echo "Open a new shell or run: export PATH=\"\$HOME/.tg_agent/bin:\$PATH\""
