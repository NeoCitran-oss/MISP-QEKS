#!/usr/bin/env bash
# Legacy wrapper — use scripts/run_siglip2_extract.sh directly.
exec bash "$(dirname "$0")/scripts/run_siglip2_extract.sh" "$@"
