# Public Release Checklist

This checklist documents the repository state required before changing `ETPNav-MID360` from private to public.

## Repository content

- [x] Default README clearly describes the project as ongoing research.
- [x] Original Chinese research README preserved as `README_CN.md`.
- [x] Upstream ETPNav attribution is explicit.
- [x] Upstream MIT license notice is retained.
- [x] README states that restricted Matterport3D assets, credentials, checkpoints, and local outputs are not redistributed.
- [x] `.gitignore` excludes common secrets, local agent state, checkpoints, caches, generated outputs, and datasets.
- [x] Default-branch code search found no obvious `token`, `secret`, `password`, `api_key`, `hf_`, or `sk-` strings.
- [x] Default-branch code search found no obvious local absolute-path / username leakage patterns from the audit queries.

## Branches and collaboration state

Changing repository visibility does not require rebasing, squashing, deleting, or moving existing branches.

Current research branches should remain unchanged during release preparation. In particular:

- `mid360-sparse-depth-learned-adapter`
- `stage5-m1-data-training`
- `stage5-m1-static-eval`
- `mid360-sparse-depth-zero-shot`
- `mid360-sparse-depth-feature-alignment`
- `etp-mid360`
- `isaac-integration`

Before release, remember that **all branches in the repository become publicly visible** with the repository.

Current collaboration artifacts also become publicly visible, including:

- open Pull Request #3 (`stage5-m1-data-training` -> `mid360-sparse-depth-learned-adapter`);
- Issue #1 (Stage 5 M1 data/cache/training);
- Issue #2 (Stage 5 M1 static evaluation / held-out gate);
- their comments, experiment protocol details, branch names, held-out identifiers, and collaboration discussion.

No release-preparation change should rewrite or invalidate those branches or their PR relationships.

## Final manual checks before switching visibility

- [ ] Decide whether the current open Issues / PR contain research details that are acceptable to disclose publicly.
- [ ] Review all active branch diffs for unexpected datasets, model weights, credentials, local-only reports, or private paths.
- [ ] Run a full Git-history secret scan locally (recommended: Gitleaks or equivalent), because default-branch code search does not prove that deleted historical secrets never existed.
- [ ] If a historical secret is found, rotate/revoke it first; then clean repository history if necessary.
- [ ] Confirm there is no collaborator / supervisor requirement to keep the work private before publication or submission.
- [ ] Check GitHub Actions logs/artifacts, if any, for credentials or private paths before making them public.
- [ ] Confirm repository visibility change only after the checks above are complete.

## Recommended local history audit

Example commands:

```bash
git log --all --stat

git log --all --name-only --pretty=format: | \
  grep -Ei '(\.env|token|secret|password|credential|\.pem|\.key|\.pth|\.pt|\.ckpt|\.safetensors|temp_report)'
```

If available, also run a dedicated secret scanner over the full history, for example:

```bash
gitleaks git .
```

Do not interpret a clean current working tree as proof that the entire Git history is clean.

## Visibility change

The final private -> public switch is intentionally **not** part of this preparation branch. It should be performed only after the manual checks above and an explicit final decision to disclose the active Issues, PR, branches, and historical repository content.
