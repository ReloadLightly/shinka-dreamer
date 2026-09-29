#!/usr/bin/env bash
# Repository-local dependencies; never changes authentication or global settings.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
mkdir -p .runtime
if [[ ! -d .runtime/headless ]]; then
  git clone https://github.com/RobertTLange/headless-cli.git .runtime/headless
  git -C .runtime/headless checkout --detach 93cd9b06b85f848af1308c41e018991b33907c5e
fi
test "$(git -C .runtime/headless rev-parse HEAD)" = 93cd9b06b85f848af1308c41e018991b33907c5e
npm ci --prefix .runtime/headless --ignore-scripts
npm run --prefix .runtime/headless build
