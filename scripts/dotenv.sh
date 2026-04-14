#!/usr/bin/env bash
# dotenv.sh — load environment variables from a .env file
#
# Usage (after REPO_ROOT / SCRIPT_DIR are set):
#   source "${SCRIPT_DIR}/dotenv.sh"
#   dotenv_load "${ENV_FILE:-${REPO_ROOT}/.env}"
#
# The file should contain bash-compatible assignments, one per line, e.g.:
#   GEMINI_API_KEY=your_key_here
#   # comments and blank lines are OK
#   export VERTEX_PROJECT=my-project
#
# Values with spaces should be quoted:  FOO="hello world"

dotenv_load() {
    local f="${1:-}"
    if [[ -z "${f}" || ! -f "${f}" ]]; then
        return 0
    fi
    set -a
    # shellcheck disable=SC1090
    source "${f}"
    set +a
}
