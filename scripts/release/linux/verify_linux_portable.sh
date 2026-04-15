#!/usr/bin/env bash
set -Eeuo pipefail

bundle_dir=""
run_smoke=0

usage() {
    cat <<'EOF'
Usage: verify_linux_portable.sh [options]

Options:
  --bundle-dir PATH   Bundle directory to verify
  --run-smoke         Run an isolated HOME smoke install and launcher check
  -h, --help          Show this help
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/../../.." && pwd -P)"

while (($# > 0)); do
    case "$1" in
        --bundle-dir)
            bundle_dir="${2:-}"
            shift 2
            ;;
        --run-smoke)
            run_smoke=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

get_project_version() {
    local pyproject_path="$1"
    awk '
        /^\[project\]/ { in_project=1; next }
        /^\[/ && in_project { exit }
        in_project && $0 ~ /^[[:space:]]*version[[:space:]]*=/ {
            line = $0
            sub(/^[^"]*"/, "", line)
            sub(/".*$/, "", line)
            print line
            exit
        }
    ' "${pyproject_path}"
}

assert_path_exists() {
    local target_path="$1"
    local label="$2"
    if [[ ! -e "${target_path}" ]]; then
        echo "Missing ${label}: ${target_path}" >&2
        exit 1
    fi
}

find_runtime_python() {
    local runtime_root="$1"
    local candidate
    for candidate in \
        "${runtime_root}/bin/python3" \
        "${runtime_root}/bin/python" \
        "${runtime_root}/python/bin/python3" \
        "${runtime_root}/python/bin/python"; do
        if [[ -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return
        fi
    done

    return 1
}

invoke_smoke_run() {
    local bundle_root="$1"
    local deploy_script="${bundle_root}/deploy.sh"
    local launcher="${bundle_root}/tg-agent-launcher.sh"
    local smoke_root="${bundle_root}/.tmp_portable_smoke"
    local home_dir="${smoke_root}/home"
    local workspace_dir="${smoke_root}/workspace"

    rm -rf "${smoke_root}"
    mkdir -p "${home_dir}" "${workspace_dir}"

    (
        export HOME="${home_dir}"
        "${deploy_script}" --skip-profile-update
        cd "${workspace_dir}"
        "${launcher}" --help >/dev/null
    )
}

project_version="$(get_project_version "${repo_root}/pyproject.toml")"
if [[ -n "${bundle_dir}" ]]; then
    resolved_bundle_dir="$(cd "${bundle_dir}" && pwd -P)"
else
    resolved_bundle_dir="${repo_root}/dist/release/tg-agent-linux-x64-v${project_version}-portable"
fi

app_dir="${resolved_bundle_dir}/app"
runtime_dir="${resolved_bundle_dir}/python-runtime"
wheelhouse_dir="${resolved_bundle_dir}/wheelhouse"

assert_path_exists "${resolved_bundle_dir}" "bundle directory"
assert_path_exists "${resolved_bundle_dir}/deploy.sh" "deploy.sh"
assert_path_exists "${resolved_bundle_dir}/tg-agent-launcher.sh" "tg-agent-launcher.sh"
assert_path_exists "${resolved_bundle_dir}/README.txt" "README.txt"
assert_path_exists "${app_dir}" "app directory"
assert_path_exists "${runtime_dir}" "python-runtime directory"
assert_path_exists "${wheelhouse_dir}" "wheelhouse directory"
assert_path_exists "${app_dir}/tg_crab_main.py" "packaged tg_crab_main.py"
assert_path_exists "${app_dir}/agent_core" "packaged agent_core"
assert_path_exists "${app_dir}/cli" "packaged cli"
assert_path_exists "${app_dir}/tools" "packaged tools"
assert_path_exists "${app_dir}/.env" "packaged .env"
assert_path_exists "${app_dir}/tg_crab_worker.json" "packaged worker config"

runtime_python="$(find_runtime_python "${runtime_dir}" || true)"
if [[ -z "${runtime_python}" ]]; then
    echo "Missing bundled runtime Python under: ${runtime_dir}" >&2
    exit 1
fi

project_wheel="$(find "${wheelhouse_dir}" -maxdepth 1 -type f -name 'tg_agent_cli-*.whl' | head -n 1)"
if [[ -z "${project_wheel}" ]]; then
    echo "Project wheel not found in bundled wheelhouse: ${wheelhouse_dir}" >&2
    exit 1
fi

echo "[portable] structure verification passed: ${resolved_bundle_dir}"

if ((run_smoke)); then
    invoke_smoke_run "${resolved_bundle_dir}"
    echo "[portable] smoke verification passed"
fi
