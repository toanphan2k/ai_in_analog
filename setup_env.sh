#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
VENV_DIR="${SCRIPT_DIR}/.venv"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    printf 'Error: Python executable not found: %s\n' "${PYTHON}" >&2
    printf 'Install Python 3 and rerun, or set PYTHON=/path/to/python3.\n' >&2
    exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    if ! "${PYTHON}" -m venv "${VENV_DIR}"; then
        printf 'Python could not seed pip; retrying with a pip-free virtual environment.\n' >&2
        if ! "${PYTHON}" -m venv --without-pip "${VENV_DIR}"; then
            printf 'Error: could not create the virtual environment.\n' >&2
            exit 1
        fi
    fi
fi

if ! "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
    if ! "${PYTHON}" -m pip --python "${VENV_DIR}/bin/python" install --upgrade pip; then
        printf 'Error: pip is unavailable and could not be bootstrapped into the virtual environment.\n' >&2
        printf 'Install the matching python3-venv package or upgrade the system pip, then rerun.\n' >&2
        exit 1
    fi
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${SCRIPT_DIR}/requirements.txt"

printf '\nEnvironment is ready.\n'
printf 'Activate with: source "%s/bin/activate"\n' "${VENV_DIR}"
printf 'Run tests with: cd "%s/.." && "%s/bin/python" -m pytest tests/test_ai_in_analog.py -q\n' \
    "${SCRIPT_DIR}" "${VENV_DIR}"
