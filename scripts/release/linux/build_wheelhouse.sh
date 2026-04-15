#!/usr/bin/env bash
set -Eeuo pipefail

python_executable=""
output_dir=""
clean=0
project_only=0

usage() {
    cat <<'EOF'
Usage: build_wheelhouse.sh [options]

Options:
  --python-executable PATH   Python executable used for wheel builds
  --output-dir PATH          Output wheelhouse directory
  --clean                    Reset the output directory before building
  --project-only             Build only the project wheel and skip dependencies
  -h, --help                 Show this help
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/../../.." && pwd -P)"

while (($# > 0)); do
    case "$1" in
        --python-executable)
            python_executable="${2:-}"
            shift 2
            ;;
        --output-dir)
            output_dir="${2:-}"
            shift 2
            ;;
        --clean)
            clean=1
            shift
            ;;
        --project-only)
            project_only=1
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

    cat >&2 <<'EOF'
Could not resolve a usable Python executable automatically.
Pass --python-executable explicitly, for example:
  --python-executable /opt/python/cp312-cp312/bin/python3
EOF
    exit 1
}

assert_build_backend_available() {
    local python_exe="$1"
    if ! "${python_exe}" -c 'import hatchling' >/dev/null 2>&1; then
        echo "Missing hatchling in ${python_exe}. Run '${python_exe} -m pip install hatchling' first." >&2
        exit 1
    fi
}

resolved_python_exe="$(resolve_python_executable "${python_executable}")"

if [[ -n "${output_dir}" ]]; then
    resolved_output_dir="$(mkdir -p "${output_dir}" && cd "${output_dir}" && pwd -P)"
else
    resolved_output_dir="${repo_root}/dist/package/linux/wheelhouse"
    mkdir -p "${resolved_output_dir}"
fi

output_parent="$(cd "$(dirname "${resolved_output_dir}")" && pwd -P)"
temp_root="${output_parent}/.wheelhouse_build_tmp"
python_version="$("${resolved_python_exe}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"

mkdir -p "${output_parent}"

if ((clean)); then
    rm -rf "${resolved_output_dir}"
    mkdir -p "${resolved_output_dir}"
else
    mkdir -p "${resolved_output_dir}"
    if find "${resolved_output_dir}" -mindepth 1 -maxdepth 1 | read -r _; then
        echo "[wheelhouse] warning: output directory already contains files; use --clean to avoid mixing wheelhouses." >&2
    fi
fi

assert_build_backend_available "${resolved_python_exe}"

mkdir -p "${temp_root}"

echo "[wheelhouse] repo root: ${repo_root}"
echo "[wheelhouse] python exe: ${resolved_python_exe}"
echo "[wheelhouse] python version: ${python_version}"
echo "[wheelhouse] output dir: ${resolved_output_dir}"
if ((project_only)); then
    echo "[wheelhouse] mode: project-only"
else
    echo "[wheelhouse] mode: project-and-dependencies"
fi

pushd "${repo_root}" >/dev/null
previous_tmpdir="${TMPDIR:-}"
export TMPDIR="${temp_root}"

build_args=(-m pip wheel . --wheel-dir "${resolved_output_dir}")
if ((project_only)); then
    build_args+=(--no-deps)
fi

"${resolved_python_exe}" "${build_args[@]}"

popd >/dev/null

if [[ -n "${previous_tmpdir}" ]]; then
    export TMPDIR="${previous_tmpdir}"
else
    unset TMPDIR || true
fi

echo "[wheelhouse] build complete"
