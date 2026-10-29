#!/usr/bin/env bash
# Author-operated optional transfer; does not connect to any server.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$#" -lt 2 ]]; then
  echo "Usage: bash export_missing_artifacts.sh RESEARCH_REPO OUTPUT_TAR_GZ [--list-only]" >&2
  exit 2
fi
repo="$1"
output="$2"
shift 2
python "$script_dir/export_missing_artifacts.py" --repo "$repo" --output "$output" "$@"
