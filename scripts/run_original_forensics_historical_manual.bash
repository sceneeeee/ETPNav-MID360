#!/usr/bin/env bash
# Run after the current-worktree manual runner.  This never changes its branch.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
worktree="${repo_root}/../etpnav-mid360-c794ba6"
if [[ ! -d "${worktree}/.git" && ! -f "${worktree}/.git" ]]; then
  git -C "${repo_root}" worktree add --detach "${worktree}" c794ba6
fi
echo "Historical worktree prepared at ${worktree}."
echo "c794ba6 predates the forensic hook, so do not claim raw heatmap parity until the hook is backported as an instrumentation-only patch."
echo "Run its normal one-episode evaluation there with the same overrides recorded in outputs/original_forensics_manual/summary.json."
