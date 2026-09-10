# Fix 2: build the image in 12.8 seconds

## The failure

vLLM support for V4.1-Flash lives in PR #56214. No aarch64 wheel is published
for a pull request head. A from-source vLLM build on this hardware costs hours.

The obvious shortcut fails. Take the wheel for the base commit the GitHub API
reports, and the import breaks:

```
ImportError: cannot import name 'resolve_quant_method'
```

## The cause

The GitHub API reports `base.sha 9e2570656` for this pull request. **That is the
tip of `main`. The diff does not apply to it.** Its tree has already moved
past the PR.

PR #56214 is a single commit, `e47aa780`. The commit it applies to is its own
parent.

`wheels.vllm.ai` publishes an official `manylinux_2_28_aarch64` wheel per commit.
A wheel exists for the parent, `29af8bd672d5a780abd7399c0cc624078202e89d`.

## The change

Read the parent from the commit endpoint:

```
GET /repos/vllm-project/vllm/commits/e47aa780...  ->  .parents[0]
```

Then build four layers on top of `eugr/spark-vllm-b12x:latest`:

| Step | Content |
|--:|---|
| 1 | Install the parent-commit wheel with `--no-deps --force-reinstall` |
| 2 | Copy the PR's 87 changed `vllm/*.py` over the installed package |
| 3 | Apply `patch/engram-disk-table.patch` |
| 4 | Copy `vl41_ops.so` to `/opt/vl41/` |

Step 2 reproduces the PR head Python tree exactly. Files the PR does not touch
are byte-identical between the wheel and the PR head tree. 26 of the 87 files are
new. The PR deletes nothing and adds no non-`.py` asset under `vllm/`.

`csrc` and the Rust crate stay at the parent commit. See
[op-shim-apply-q-norm.md](op-shim-apply-q-norm.md) for the one place that
matters.

Script: `build/vl41-build-image.sh`. Total build time 12.8 seconds.

## Why this base image works

`eugr/spark-vllm-b12x:latest` already carries the dependency stack the wheel
wants:

| Package | Version |
|---|---|
| torch | 2.13.0+cu130 |
| flashinfer | 0.6.18, eugr's build with the sm120 sparse-MLA kernels |
| tilelang | 0.1.12 |
| CUDA | 13.0 |
| Python | 3.12 |

## Traps

- **The diff endpoint refuses this PR.** `/pulls/56214` returns HTTP 406, "diff
  exceeded the maximum number of lines (20000)". Use
  `github.com/vllm-project/vllm/pull/56214.diff`, which has no such cap.
- **The official wheel carries no sm_121 cubins.** It ships sm_120 device code
  plus PTX for 80, 89 and 90. Minor-version compatibility held for everything
  exercised here. No audit was run for `sm_120a` kernels.
- **The Rust crate stays at the parent commit.** 23 of the PR's 147 files are
  Rust, including the V4.1 renderer and the DSML tool parser. Nothing here
  exercises them.
