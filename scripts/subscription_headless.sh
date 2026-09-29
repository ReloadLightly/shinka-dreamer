#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
headless_cli="${SHINKADREAMER_HEADLESS_CLI:-$task_root/.runtime/headless/dist/cli.js}"
# Never rely on Headless's auto mode, or an API key being absent by accident.
unset OPENAI_API_KEY CODEX_API_KEY OPENAI_BASE_URL ANTHROPIC_API_KEY
unset GOOGLE_API_KEY GEMINI_API_KEY AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
export HEADLESS_BILLING=subscription
if ! rg -q 'HEADLESS_BILLING' "$(dirname "$headless_cli")/billing.js"; then
  echo 'Blocked: Headless must implement subscription-only billing (see docs/upstream.md).' >&2
  exit 78
fi
exec node "$headless_cli" "$@"
