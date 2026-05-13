#!/usr/bin/env bash
set -Eeuo pipefail

version=""
python_executable=""
python_home=""
python_runtime_dir=""
source_wheelhouse=""
output_root=""
allow_conda_python_runtime=0
skip_tar=0
skip_project_wheel_build=0

usage() {
    cat <<'EOF'
Usage: build_linux_portable.sh [options]

Options:
  --version TEXT                 Override project version in bundle name
  --python-executable PATH       Python executable used for wheel builds
  --python-home PATH             Base Python home used when --python-runtime-dir is not provided
  --python-runtime-dir PATH      Portable Linux runtime directory to bundle
  --source-wheelhouse PATH       Prebuilt Linux wheelhouse directory
  --output-root PATH             Output root for bundle and archive
  --allow-conda-python-runtime   Allow copying a conda runtime when using --python-home fallback
  --skip-tar                     Skip tar.gz archive creation
  --skip-project-wheel-build     When using --source-wheelhouse, skip rebuilding the project wheel
  -h, --help                     Show this help
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/../../.." && pwd -P)"

while (($# > 0)); do
    case "$1" in
        --version)
            version="${2:-}"
            shift 2
            ;;
        --python-executable)
            python_executable="${2:-}"
            shift 2
            ;;
        --python-home)
            python_home="${2:-}"
            shift 2
            ;;
        --python-runtime-dir)
            python_runtime_dir="${2:-}"
            shift 2
            ;;
        --source-wheelhouse)
            source_wheelhouse="${2:-}"
            shift 2
            ;;
        --output-root)
            output_root="${2:-}"
            shift 2
            ;;
        --allow-conda-python-runtime)
            allow_conda_python_runtime=1
            shift
            ;;
        --skip-tar)
            skip_tar=1
            shift
            ;;
        --skip-project-wheel-build)
            skip_project_wheel_build=1
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

resolve_python_executable() {
    local explicit_python="${1:-}"
    local -a candidates=()

    add_candidate() {
        local candidate="${1:-}"
        if [[ -z "${candidate}" ]]; then
            return
        fi

        if [[ -d "${candidate}" ]]; then
            candidate="${candidate%/}/bin/python"
        fi

        if [[ ! -f "${candidate}" ]]; then
            return
        fi

        local resolved
        resolved="$(
            cd "$(dirname "${candidate}")" &&
            printf '%s\n' "$(pwd -P)/$(basename "${candidate}")"
        )"

        local existing
        for existing in "${candidates[@]}"; do
            if [[ "${existing}" == "${resolved}" ]]; then
                return
            fi
        done

        candidates+=("${resolved}")
    }

    if [[ -n "${explicit_python}" ]]; then
        add_candidate "${explicit_python}"
        if ((${#candidates[@]} == 0)); then
            echo "Explicit Python executable was provided but not found: ${explicit_python}" >&2
            exit 1
        fi
    else
        [[ -n "${VIRTUAL_ENV:-}" ]] && add_candidate "${VIRTUAL_ENV}/bin/python"
        [[ -n "${CONDA_PREFIX:-}" ]] && add_candidate "${CONDA_PREFIX}/bin/python"
        [[ -n "${CONDA_PYTHON_EXE:-}" ]] && add_candidate "${CONDA_PYTHON_EXE}"
        [[ -n "${PYTHON:-}" ]] && add_candidate "${PYTHON}"

        if command -v python3 >/dev/null 2>&1; then
            add_candidate "$(command -v python3)"
        fi
        if command -v python >/dev/null 2>&1; then
            add_candidate "$(command -v python)"
        fi
    fi

    local candidate
    for candidate in "${candidates[@]}"; do
        if "${candidate}" -c 'import sys; print(sys.executable)' >/dev/null 2>&1; then
            printf '%s\n' "${candidate}"
            return
        fi
    done

    echo "Could not resolve a usable Python executable automatically." >&2
    exit 1
}

resolve_python_home() {
    local python_exe="$1"
    local explicit_python_home="${2:-}"

    if [[ -n "${explicit_python_home}" ]]; then
        (cd "${explicit_python_home}" && pwd -P)
        return
    fi

    "${python_exe}" -c 'import sys; print(sys.base_prefix)'
}

test_conda_python_home() {
    local python_home_dir="$1"
    [[ -d "${python_home_dir}/conda-meta" || -d "${python_home_dir}/condabin" ]]
}

assert_build_backend_available() {
    local python_exe="$1"
    if ! "${python_exe}" -c 'import hatchling' >/dev/null 2>&1; then
        echo "Missing hatchling in ${python_exe}. Run '${python_exe} -m pip install hatchling' first." >&2
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

copy_app_payload() {
    local repo_root_path="$1"
    local target_dir="$2"
    local -a items=(
        "agent_core"
        "cli"
        "tools"
        "tg_mem"
        "skills"
        "plugins"
        "config"
        "template"
        ".devagent"
        "tg_crab_main.py"
        "ralph_init.py"
        "ralph_loop.py"
        ".env"
        "tg_crab_worker.json"
    )

    local item
    for item in "${items[@]}"; do
        local source_path="${repo_root_path}/${item}"
        if [[ ! -e "${source_path}" ]]; then
            echo "Required app payload is missing: ${source_path}" >&2
            exit 1
        fi
        cp -a "${source_path}" "${target_dir}/"
    done
}

copy_python_runtime() {
    local source_runtime_dir="$1"
    local target_runtime_dir="$2"

    mkdir -p "${target_runtime_dir}"
    cp -a "${source_runtime_dir}/." "${target_runtime_dir}/"

    local pattern
    for pattern in \
        "${target_runtime_dir}"/lib/python*/site-packages \
        "${target_runtime_dir}"/lib/python*/test \
        "${target_runtime_dir}"/lib/python*/tests \
        "${target_runtime_dir}"/lib/python*/idlelib \
        "${target_runtime_dir}"/lib/python*/tkinter \
        "${target_runtime_dir}"/lib/python*/turtledemo; do
        if [[ -e "${pattern}" ]]; then
            rm -rf "${pattern}"
        fi
    done

    find "${target_runtime_dir}" \
        \( -type d -name '__pycache__' -o -type f \( -name '*.pyc' -o -name '*.pyo' \) \) \
        -exec rm -rf {} +
}

copy_wheelhouse() {
    local source_wheelhouse_dir="$1"
    local target_wheelhouse_dir="$2"

    if [[ ! -d "${source_wheelhouse_dir}" ]]; then
        echo "Source wheelhouse directory not found: ${source_wheelhouse_dir}" >&2
        exit 1
    fi

    cp -a "${source_wheelhouse_dir}/." "${target_wheelhouse_dir}/"
}

build_project_wheel() {
    local python_exe="$1"
    local repo_root_path="$2"
    local target_wheelhouse_dir="$3"
    local temp_root="$4"

    find "${target_wheelhouse_dir}" -maxdepth 1 -type f -name 'tg_agent_cli-*.whl' -delete

    mkdir -p "${temp_root}"
    pushd "${repo_root_path}" >/dev/null
    local previous_tmpdir="${TMPDIR:-}"
    export TMPDIR="${temp_root}"
    "${python_exe}" -m pip wheel . --no-deps --wheel-dir "${target_wheelhouse_dir}" --no-build-isolation
    popd >/dev/null

    if [[ -n "${previous_tmpdir}" ]]; then
        export TMPDIR="${previous_tmpdir}"
    else
        unset TMPDIR || true
    fi
}

build_full_wheelhouse() {
    local python_exe="$1"
    local repo_root_path="$2"
    local target_wheelhouse_dir="$3"
    local temp_root="$4"

    mkdir -p "${temp_root}"
    pushd "${repo_root_path}" >/dev/null
    local previous_tmpdir="${TMPDIR:-}"
    export TMPDIR="${temp_root}"
    "${python_exe}" -m pip wheel . --wheel-dir "${target_wheelhouse_dir}"
    popd >/dev/null

    if [[ -n "${previous_tmpdir}" ]]; then
        export TMPDIR="${previous_tmpdir}"
    else
        unset TMPDIR || true
    fi
}

write_bundle_readme() {
    local output_path="$1"
    local version_text="$2"

    cat > "${output_path}" <<EOF
crab portable bundle

Version: ${version_text}
Platform: linux
Architecture: x64

Contents:
- deploy.sh
- crab-launcher.sh
- python-runtime/
- wheelhouse/
- app/

First run:
1. Extract this tar.gz.
2. Run ./deploy.sh once.
3. Start ./crab-launcher.sh from the workspace you want to use.

Workspace behavior:
- Running the launcher from a terminal uses the current working directory as the workspace.
- Global config stays in \$HOME/.tg_agent.

Notes:
- deploy.sh creates \$HOME/.tg_agent/.venv using the bundled Python runtime.
- crab and dependencies are installed offline from wheelhouse/.
- deploy.sh also installs a crab command shim into \$HOME/.tg_agent/bin and updates shell profile files.
- deploy.sh registers the crab://open protocol for launching the local CLI from a browser.
- If crab already exists from an older pip install, uninstall the older copy to avoid command precedence conflicts.
- Existing \$HOME/.tg_agent/.env and tg_crab_worker.json are preserved.
- For the most portable Linux bundles, provide a python-build-standalone runtime with --python-runtime-dir.
EOF
}

write_deploy_script() {
    local output_path="$1"

    cat > "${output_path}" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

force_recreate_venv=0
skip_profile_update=0

while (($# > 0)); do
    case "$1" in
        --force-recreate-venv)
            force_recreate_venv=1
            shift
            ;;
        --skip-profile-update)
            skip_profile_update=1
            shift
            ;;
        -h|--help)
            cat <<'USAGE'
Usage: deploy.sh [--force-recreate-venv] [--skip-profile-update]
USAGE
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

bundle_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
runtime_dir="${bundle_root}/python-runtime"
wheelhouse_dir="${bundle_root}/wheelhouse"
app_dir="${bundle_root}/app"
launcher_path="${bundle_root}/crab-launcher.sh"
user_home="${HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)}"
install_root="${TG_AGENT_HOME:-${user_home}/.tg_agent}"
venv_dir="${install_root}/.venv"
bin_dir="${install_root}/bin"
entry_shim="${bin_dir}/crab-entry.py"
command_shim="${bin_dir}/crab"
protocol_launcher="${bin_dir}/crab-protocol-launcher.sh"
env_file="${install_root}/.env"
worker_config="${install_root}/tg_crab_worker.json"
packaged_env_file="${app_dir}/.env"
packaged_worker_config="${app_dir}/tg_crab_worker.json"
desktop_dir="${user_home}/.local/share/applications"
protocol_desktop="${desktop_dir}/crab-protocol.desktop"
profile_line='export PATH="$HOME/.tg_agent/bin:$PATH"'

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

add_path_line_once() {
    local target_file="$1"
    local line="$2"
    mkdir -p "$(dirname "${target_file}")"
    touch "${target_file}"
    if ! grep -Fqx "${line}" "${target_file}"; then
        printf '\n%s\n' "${line}" >> "${target_file}"
        return 0
    fi
    return 1
}

get_command_candidates() {
    type -aP crab 2>/dev/null | awk '!seen[$0]++'
}

get_pip_uninstall_hint() {
    local command_path="${1:-}"
    if [[ -z "${command_path}" ]]; then
        return
    fi

    local command_dir
    command_dir="$(dirname "${command_path}")"
    if [[ "$(basename "${command_dir}")" != "bin" ]]; then
        return
    fi

    local python_candidate
    for python_candidate in \
        "${command_dir}/python3" \
        "${command_dir}/python" \
        "$(dirname "${command_dir}")/bin/python3" \
        "$(dirname "${command_dir}")/bin/python"; do
        if [[ -x "${python_candidate}" ]]; then
            printf '%s -m pip uninstall tg-agent-cli\n' "${python_candidate}"
            return
        fi
    done
}

if ! runtime_python="$(find_runtime_python "${runtime_dir}")"; then
    echo "Bundled Python runtime not found under: ${runtime_dir}" >&2
    exit 1
fi

if [[ ! -d "${wheelhouse_dir}" ]]; then
    echo "Bundled wheelhouse not found: ${wheelhouse_dir}" >&2
    exit 1
fi

project_wheel="$(find "${wheelhouse_dir}" -maxdepth 1 -type f -name 'tg_agent_cli-*.whl' | sort | tail -n 1)"
if [[ -z "${project_wheel}" ]]; then
    echo "Project wheel not found in wheelhouse: ${wheelhouse_dir}" >&2
    exit 1
fi

mkdir -p "${install_root}" "${bin_dir}"

if ((force_recreate_venv)) && [[ -d "${venv_dir}" ]]; then
    rm -rf "${venv_dir}"
fi

venv_python="${venv_dir}/bin/python"
if [[ ! -x "${venv_python}" ]]; then
    "${runtime_python}" -m venv "${venv_dir}"
fi

"${venv_python}" -m ensurepip --upgrade
"${venv_python}" -m pip install --no-index --find-links "${wheelhouse_dir}" --upgrade "${project_wheel}"

cat > "${entry_shim}" <<'PY'
from tg_crab_main import cli_main

if __name__ == "__main__":
    cli_main()
PY

cat > "${command_shim}" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail

user_home="${HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)}"
install_root="${TG_AGENT_HOME:-${user_home}/.tg_agent}"
venv_python="${install_root}/.venv/bin/python"
entry_shim="${install_root}/bin/crab-entry.py"

if [[ ! -x "${venv_python}" ]]; then
    echo "crab is not installed." >&2
    echo "Run ./deploy.sh from the portable bundle first." >&2
    exit 1
fi

if [[ ! -f "${entry_shim}" ]]; then
    echo "crab entry shim is missing." >&2
    echo "Run ./deploy.sh from the portable bundle first." >&2
    exit 1
fi

exec "${venv_python}" -u "${entry_shim}" "$@"
SH
chmod +x "${command_shim}"

cat > "${protocol_launcher}" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail

url="${1:-}"

if [[ "${url}" != crab://open* && "${url}" != crab://launch* ]]; then
  exit 1
fi

user_home="${HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)}"
install_root="${TG_AGENT_HOME:-${user_home}/.tg_agent}"
command_shim="${install_root}/bin/crab"

if [[ ! -x "${command_shim}" ]]; then
  echo "crab is not installed." >&2
  echo "Run ./deploy.sh from the portable bundle first." >&2
  exit 1
fi

if [[ -d "${user_home}/Desktop" ]]; then
  cd "${user_home}/Desktop"
else
  cd "${user_home}"
fi

exec "${command_shim}"
SH
chmod +x "${protocol_launcher}"

if [[ -f "${packaged_env_file}" && ! -f "${env_file}" ]]; then
    cp "${packaged_env_file}" "${env_file}"
fi

if [[ -f "${packaged_worker_config}" && ! -f "${worker_config}" ]]; then
    cp "${packaged_worker_config}" "${worker_config}"
fi

if [[ ! -f "${launcher_path}" ]]; then
    echo "Launcher missing: ${launcher_path}" >&2
    exit 1
fi

mkdir -p "${desktop_dir}"
cat > "${protocol_desktop}" <<EOF_DESKTOP
[Desktop Entry]
Name=Crab Protocol Handler
Comment=Handle crab:// links
Exec=${protocol_launcher} %u
Type=Application
Terminal=true
MimeType=x-scheme-handler/crab;
NoDisplay=true
EOF_DESKTOP

if command -v xdg-mime >/dev/null 2>&1; then
    xdg-mime default "$(basename "${protocol_desktop}")" x-scheme-handler/crab || true
fi
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "${desktop_dir}" >/dev/null 2>&1 || true
fi

profile_updated=0
if ((skip_profile_update == 0)); then
    if add_path_line_once "${user_home}/.profile" "${profile_line}"; then
        profile_updated=1
    fi
    if [[ -f "${user_home}/.bashrc" ]]; then
        add_path_line_once "${user_home}/.bashrc" "${profile_line}" || true
    fi
    if [[ -f "${user_home}/.zshrc" ]]; then
        add_path_line_once "${user_home}/.zshrc" "${profile_line}" || true
    fi
fi

existing_commands=()
while IFS= read -r line; do
    [[ -n "${line}" && "${line}" != "${command_shim}" ]] && existing_commands+=("${line}")
done < <(get_command_candidates || true)

echo "[portable] install root: ${install_root}"
echo "[portable] venv ready: ${venv_dir}"
echo "[portable] command shim: ${command_shim}"
echo "[portable] protocol launcher: ${protocol_launcher}"
if ((profile_updated)); then
    echo "[portable] added to shell profile PATH: ${user_home}/.tg_agent/bin"
else
    echo "[portable] shell profile PATH already contains: ${user_home}/.tg_agent/bin"
fi
echo "[portable] launcher: ${launcher_path}"
echo "[portable] registered protocol: crab://open"
echo "[portable] open a new shell and run: crab"

if ((${#existing_commands[@]} > 0)); then
    conflicting_command="${existing_commands[0]}"
    echo "[portable] warning: detected another crab command on this machine: ${conflicting_command}" >&2
    echo "[portable] warning: the portable install does not overwrite older pip-based crab commands." >&2

    pip_uninstall_hint="$(get_pip_uninstall_hint "${conflicting_command}" || true)"
    if [[ -n "${pip_uninstall_hint}" ]]; then
        echo "[portable] warning: to avoid command conflicts, uninstall the older copy with: ${pip_uninstall_hint}" >&2
    else
        echo "[portable] warning: to avoid command conflicts, uninstall the older crab copy before using the PATH-based command." >&2
    fi

    echo "[portable] warning: you can always launch the portable install directly with: ${command_shim}" >&2
fi

echo "[portable] start the launcher from your workspace directory."
EOF

    chmod +x "${output_path}"
}

write_launcher_script() {
    local output_path="$1"

    cat > "${output_path}" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

user_home="${HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)}"
install_root="${TG_AGENT_HOME:-${user_home}/.tg_agent}"
venv_python="${install_root}/.venv/bin/python"
entry_shim="${install_root}/bin/crab-entry.py"

if [[ ! -x "${venv_python}" ]]; then
    echo "crab is not installed yet." >&2
    echo "Run ./deploy.sh from this bundle first." >&2
    exit 1
fi

if [[ ! -f "${entry_shim}" ]]; then
    echo "crab entry shim is missing." >&2
    echo "Run ./deploy.sh from this bundle first." >&2
    exit 1
fi

exec "${venv_python}" -u "${entry_shim}" "$@"
EOF

    chmod +x "${output_path}"
}

resolved_python_exe="$(resolve_python_executable "${python_executable}")"
resolved_python_home="$(resolve_python_home "${resolved_python_exe}" "${python_home}")"
resolved_python_home="$(cd "${resolved_python_home}" && pwd -P)"

if [[ -n "${python_runtime_dir}" ]]; then
    resolved_runtime_source="$(cd "${python_runtime_dir}" && pwd -P)"
else
    resolved_runtime_source="${resolved_python_home}"
fi

if [[ -z "${version}" ]]; then
    version="$(get_project_version "${repo_root}/pyproject.toml")"
fi

if [[ -n "${output_root}" ]]; then
    mkdir -p "${output_root}"
    resolved_output_root="$(cd "${output_root}" && pwd -P)"
else
    resolved_output_root="${repo_root}/dist/release"
    mkdir -p "${resolved_output_root}"
fi

if [[ -n "${source_wheelhouse}" ]]; then
    resolved_source_wheelhouse="$(cd "${source_wheelhouse}" && pwd -P)"
else
    default_wheelhouse="${repo_root}/dist/package/linux/wheelhouse"
    if [[ -d "${default_wheelhouse}" ]]; then
        resolved_source_wheelhouse="${default_wheelhouse}"
    else
        resolved_source_wheelhouse=""
    fi
fi

if test_conda_python_home "${resolved_python_home}" && ((allow_conda_python_runtime == 0)) && [[ -z "${python_runtime_dir}" ]]; then
    echo "Resolved Python home appears to be a conda runtime: ${resolved_python_home}" >&2
    echo "Use a standard Linux runtime or pass --python-runtime-dir with a portable runtime bundle." >&2
    echo "If you really want to continue with this Python home, pass --allow-conda-python-runtime." >&2
    exit 1
fi

if ! runtime_probe="$(find_runtime_python "${resolved_runtime_source}")"; then
    echo "Could not find a usable runtime Python under: ${resolved_runtime_source}" >&2
    echo "Provide --python-runtime-dir pointing at an extracted Linux runtime, such as python-build-standalone." >&2
    exit 1
fi

assert_build_backend_available "${resolved_python_exe}"

bundle_name="tg-agent-linux-x64-v${version}-portable"
bundle_dir="${resolved_output_root}/${bundle_name}"
app_dir="${bundle_dir}/app"
runtime_dir="${bundle_dir}/python-runtime"
wheelhouse_dir="${bundle_dir}/wheelhouse"
temp_dir="${bundle_dir}/tmp"
launcher_path="${bundle_dir}/crab-launcher.sh"
deploy_sh="${bundle_dir}/deploy.sh"
readme_path="${bundle_dir}/README.txt"
tar_path="${resolved_output_root}/${bundle_name}.tar.gz"

rm -rf "${bundle_dir}"
mkdir -p "${app_dir}" "${runtime_dir}" "${wheelhouse_dir}" "${temp_dir}"

echo "[portable] repo root: ${repo_root}"
echo "[portable] python exe: ${resolved_python_exe}"
echo "[portable] python home: ${resolved_python_home}"
echo "[portable] runtime source: ${resolved_runtime_source}"
echo "[portable] runtime python: ${runtime_probe}"
echo "[portable] bundle dir: ${bundle_dir}"

echo "[portable] copying app payload..."
copy_app_payload "${repo_root}" "${app_dir}"

echo "[portable] copying bundled Linux runtime..."
copy_python_runtime "${resolved_runtime_source}" "${runtime_dir}"

if [[ -n "${resolved_source_wheelhouse}" ]]; then
    echo "[portable] copying source wheelhouse from ${resolved_source_wheelhouse}"
    copy_wheelhouse "${resolved_source_wheelhouse}" "${wheelhouse_dir}"
else
    echo "[portable] no source wheelhouse provided; building dependency wheelhouse from the project"
    build_full_wheelhouse "${resolved_python_exe}" "${repo_root}" "${wheelhouse_dir}" "${temp_dir}"
fi

if [[ -n "${resolved_source_wheelhouse}" && ${skip_project_wheel_build} -eq 0 ]]; then
    echo "[portable] building project wheel..."
    build_project_wheel "${resolved_python_exe}" "${repo_root}" "${wheelhouse_dir}" "${temp_dir}"
fi

rm -rf "${temp_dir}"

write_launcher_script "${launcher_path}"
write_deploy_script "${deploy_sh}"
write_bundle_readme "${readme_path}" "${version}"

if ((skip_tar == 0)); then
    rm -f "${tar_path}"
    echo "[portable] creating tar.gz archive..."
    tar -C "${resolved_output_root}" -czf "${tar_path}" "${bundle_name}"
fi

echo "[portable] build complete"
echo "[portable] bundle: ${bundle_dir}"
if ((skip_tar == 0)); then
    echo "[portable] tar.gz: ${tar_path}"
fi
