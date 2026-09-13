# Environment Notes — env-v0.1

## Scope: HPC (linux-64) Only

This project version-locks **one** environment: the HPC cluster (AICR), platform
`linux-64`. That is where formal experiment results are produced, so that is what
must be exactly reproducible.

**macOS is intentionally not tracked and not locked.** The local Mac is used only
to prototype and smoke-test pipeline logic on a minimal flow — it never produces
official results. Locking it would cost real maintenance (a second solve, a second
set of platform-specific constraints to validate) for no reproducibility benefit at
the current data scale. If that changes, macOS can be added back as its own
independently solved and validated lock; nothing here forecloses that.

A prior macOS export (`environment.yml`) was removed in the commit that introduced
this note. It remains in git history if ever needed:

```bash
git show 34c1f75:environment/env-v0.1/environment.yml
```

## Files in This Directory

| File | Role |
|------|------|
| `environment-linux-64.yml` | **Declarative.** Human-readable, human-editable list of top-level dependencies only. Says *what we asked for*. |
| `conda-lock.yml` | **The lock.** Exact versions + build strings + hashes for every package, transitive ones included. Says *what actually gets installed*. |
| `NOTES.md` | This file. |

### environment-linux-64.yml — declarative, not a guarantee

Lists top-level dependencies with version pins, but **no build strings and no
hashes**, and it does not pin transitive dependencies at all. Re-solving it against
conda-forge on a different day can produce different builds, or different versions
entirely. It documents intent; it cannot reproduce an environment. Edit this file
when you want to change a dependency — then regenerate the lock.

### conda-lock.yml — the actual reproducibility artifact

Generated from `environment-linux-64.yml`, with top-level pins taken from the
environment actually running on AICR, so the lock reproduces that environment
rather than a fresh same-day guess. Covers `linux-64` only: 149 conda + 36 pip
packages, every one carrying a hash, zero osx-arm64 entries. Verified build-for-
build against the live env (`ffmpeg-9.0.1-gpl_hc1c51de_902`,
`python-3.10.21-h267e890_0_cpython`, `libstdcxx-16.2.0-h934c35e_4`,
`x264-1!164.3095-h7cc23a3_3`, ...).

Recreate the HPC environment:

```bash
conda-lock install --name talkvid conda-lock.yml
```

Regenerate after editing the declarative file (on a compute node, **not** the
login node):

```bash
srun --partition=cpu --cpus-per-task=4 --mem=8G --time=00:50:00 \
  conda-lock lock --file environment-linux-64.yml -p linux-64
```

## Why Lock: Observed Drift Evidence

This is not hypothetical. Comparing the old macOS declarative export against what
conda/pip actually resolved on HPC — **from the same declared intent** — showed the
following already-diverged packages:

| Package | macOS resolved | HPC resolved |
|---------|----------------|--------------|
| `opencv-python` | 4.10.0.84 | 4.11.0.86 |
| `tzdata` | 2026.3.post1 | 2026.4 |
| `ffmpeg` | 6.1.2 | 9.0.1 |
| `libopenvino` | 2024.4.0 | 2026.3.1 |
| `libprotobuf` | 5.28.2 | 7.35.1 |
| `harfbuzz` | 12.2.0 | 14.4.0 |
| `icu` | 75.1 | 78.3 |
| `lame` | 3.100 | 4.0 |
| `libabseil` | 20240722.0 | 20260526.0 |

`opencv-python` and `tzdata` are the most damning entries: both are plain pip
packages with no platform-specific compilation story. The same unpinned
`pip install` simply resolved differently. Native libraries drift further and
faster. Without a lock file, "the same environment" is a claim nobody can check.

A second failure mode the old export demonstrated: it could not be solved for
`linux-64` **at all** (`fontconfig ... requires libuuid >=2.42.2, but none of the
providers can be installed`), because it carried osx-arm64-only packages such as
`libcxx` and `libopenvino-arm-cpu-plugin`. A declarative file exported from one
platform is not portable to another.

## Minimal Dependency Set

Do NOT use the top-level `requirements.txt` from the TalkVid repository as-is.
That file targets TalkVid's own model-training pipeline and pulls in heavy
GPU-training-only dependencies (deepspeed, bitsandbytes, onnxruntime-gpu, etc.)
that are irrelevant to data processing / validation, and may fail to install
on non-CUDA machines.

Minimal dependencies actually needed for this project's validation stage:
yt-dlp, rich, av, mediapipe==0.10.21 (see version pinning rationale below —
do NOT upgrade to 1.0.x yet), opencv-python, pandas, pyarrow, ffmpeg
(see constraint note below).

## mediapipe Version Pinning

Pinned to 0.10.21, the last stable release before the 1.0 line. The 1.0.x line has
a known GPU-acceleration compatibility issue with no confirmed community fix at
the time of writing; 0.10.21 fully avoids it.

On Linux (HPC) the crash was not observed — GPU-unavailable conditions degrade
gracefully to CPU with a warning. The pin is kept anyway for consistent,
reproducible behavior. See `docs/decisions/mediapipe_version_pinning.md`.

Note that mediapipe 0.10.21 is not numpy-2 compatible, which is why `numpy` is
held at 1.26.4.

## ffmpeg: Do Not Apply a `<7` Version Cap on Linux

On Linux, conda-forge resolves ffmpeg 2.8.6 under an `ffmpeg<7` constraint, which
links against an incompatible libx264 (`libx264.so.138: cannot open shared object
file`). Install without a version cap — `conda install -c conda-forge ffmpeg -y`
resolves a self-consistent newer build (9.0.1, the version now locked).

The `<7` cap originated as a macOS-specific workaround. It was wrong to carry it
to Linux, and that mistake is the general lesson worth keeping: **version
constraints validated on one platform must be re-validated when porting to
another — never copy them blindly.**

## HPC (SLURM Cluster) Usage Notes

The login node must not be used to run actual compute workloads, including
moderate-load operations like video download/transcoding or dependency solving --
the scheduler will forcibly terminate such processes with SIGKILL (exit code -9)
to protect the shared login node for all users.

Correct approach: request an interactive compute-node session via
`srun --partition=cpu --cpus-per-task=4 --mem=8G --time=01:00:00 --pty bash`,
then activate the conda environment and run scripts from within that session.

Large-scale batch processing should use sbatch for non-interactive job
submission rather than holding a long-lived interactive session.
