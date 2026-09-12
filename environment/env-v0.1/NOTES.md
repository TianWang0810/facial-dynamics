# Environment Notes — env-v0.1

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
