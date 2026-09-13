# Environment Notes — env-v0.1

## Files in This Directory: What Locks and What Doesn't

| File | Platform | Role |
|------|----------|------|
| `conda-lock.yml` | **linux-64 only** | **The lock file.** Exact versions + build strings + hashes. This is what recreates the HPC environment. |
| `environment-linux-64.yml` | linux-64 | Source spec conda-lock consumes: top-level deps only, human-editable. |
| `environment.yml` | osx-arm64 | Historical macOS export. Reference only — see warning below. |

### environment.yml is NOT a reproducibility guarantee

`environment.yml` is a declarative `--no-builds` export: a human-readable list of
package names and versions with **no build strings and no hashes**. Re-solving it
against conda-forge on a different day can yield different builds — or different
versions entirely, since transitive dependencies are not pinned at all. It
documents intent; it does not reproduce an environment.

Additionally, **this particular `environment.yml` was exported from the local
macOS box** (note `prefix: /opt/anaconda3/envs/talkvid`). It pins osx-arm64-only
packages (`libcxx`, `libopenvino-arm-cpu-plugin`, `libintl-devel`) and
macOS-resolved versions that are wrong for Linux — most visibly `ffmpeg=6.1.2`,
where HPC actually runs 9.0.1. It **cannot be solved for linux-64 at all**;
conda-lock fails with `fontconfig ... requires libuuid >=2.42.2, but none of the
providers can be installed`. Do not point tooling at it for HPC work.

### conda-lock.yml is the real lock

Generated from `environment-linux-64.yml`, whose top-level pins were taken from
the environment actually running on AICR, so the lock reproduces that environment
rather than a fresh guess. Verified build-for-build against the live env
(`ffmpeg-9.0.1-gpl_hc1c51de_902`, `python-3.10.21-h267e890_0_cpython`,
`libstdcxx-16.2.0-h934c35e_4`, ...). 149 conda + 36 pip packages, every one with
a hash, zero osx-arm64 entries.

Recreate the HPC environment with:

```bash
conda-lock install --name talkvid conda-lock.yml
```

Regenerate after editing `environment-linux-64.yml` (run on a compute node, not
the login node):

```bash
conda-lock lock --file environment-linux-64.yml -p linux-64
```

### macOS is deliberately not locked

Local macOS is used only to prototype and run the pipeline. It is **not** a
reproducibility target, and nothing in env-v0.1 pins it. Do not assume that
recreating this spec on macOS gives you a matching environment — it will not, and
several pins here (the uncapped ffmpeg in particular) are known to be wrong for
macOS. If macOS reproducibility is ever needed, it requires its own separate
lock, solved and validated independently.

## Minimal Dependency Set

Do NOT use the top-level `requirements.txt` from the TalkVid repository as-is.
That file targets TalkVid's own model-training pipeline and pulls in heavy
GPU-training-only dependencies (deepspeed, bitsandbytes, onnxruntime-gpu, etc.)
that are irrelevant to data processing / validation, and may fail to install
on non-Linux or non-CUDA machines.

Minimal dependencies actually needed for this project's validation stage:
yt-dlp, rich, av, mediapipe==0.10.21 (see version pinning rationale below —
do NOT upgrade to 1.0.x yet), opencv-python, pandas, pyarrow, ffmpeg
(platform-dependent constraint, see below).

## mediapipe Version Pinning

mediapipe 1.0.x has a known Metal-related GPU-acceleration compatibility issue
on macOS (crashes with stack traces containing "DrishtiMetalHelper" /
"Service is unavailable"), with no confirmed community fix at the time of
writing. Pinning to 0.10.21 (the last stable release before the 1.0 line)
fully avoids this.

On Linux (HPC), this issue was not observed -- GPU-unavailable conditions
degrade gracefully to CPU with a warning instead of crashing. We still pin
to 0.10.21 uniformly across platforms for consistent, reproducible behavior.

## ffmpeg Version Constraint Is Platform-Dependent

macOS: install an older ffmpeg via conda install -c conda-forge 'ffmpeg<7',
since some downstream libraries conflict with ffmpeg 8.x.

Linux (HPC): do NOT apply the <7 constraint. On Linux, conda-forge resolves
ffmpeg 2.8.6 under that constraint, which links against an incompatible
libx264 version (libx264.so.138: cannot open shared object file). Instead
run conda install -c conda-forge ffmpeg -y (no version cap), which resolves
a self-consistent newer build (e.g., 9.0.1) that works correctly.

Conclusion: version constraints validated on one platform must be
re-validated when porting to a new platform -- do not copy them blindly.

## HPC (SLURM Cluster) Usage Notes

The login node must not be used to run actual compute workloads, including
moderate-load operations like video download/transcoding -- the scheduler
will forcibly terminate such processes with SIGKILL (exit code -9) to
protect the shared login node for all users.

Correct approach: request an interactive compute-node session via
srun --partition=cpu --cpus-per-task=4 --mem=8G --time=01:00:00 --pty bash,
then activate the conda environment and run scripts from within that session.

Large-scale batch processing should use sbatch for non-interactive job
submission rather than holding a long-lived interactive session.
