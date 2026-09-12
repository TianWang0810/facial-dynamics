# Engineering Log: HPC (AICR Cluster) Deployment Validation

Format: Symptom -> Diagnosis -> Root Cause -> Fix -> Generalizable Takeaway.

## Issue 1: ffmpeg binary present but fails to run (missing shared library)

Symptom: which ffmpeg locates the binary, but yt-dlp fails with
"ffmpeg is not installed"; running ffmpeg -version directly reveals the
real error: "error while loading shared libraries: libx264.so.138:
cannot open shared object file".

Diagnosis: The binary itself was installed correctly, but a shared
library it links against was version-incompatible.

Root Cause: conda install -c conda-forge 'ffmpeg<7' resolved a very old
ffmpeg build (2.8.6) on Linux, whose statically-recorded x264 dependency
version did not match the x264 build actually present in the conda
environment.

Fix: Removed the <7 version cap; installed
conda install -c conda-forge ffmpeg -y with no constraint, which resolved
a self-consistent, newer build (9.0.1) with matching dependencies.

Takeaway: A version constraint validated on one platform (macOS) does not
transfer automatically to another platform (Linux) -- the same constraint
can resolve an entirely different, and differently broken, dependency
graph. Constraints should be re-validated, not blindly copied, when
porting to a new environment.

## Issue 2: Download job killed with exit code -9

Symptom: Download progress reports 100% completion, but the process
reports "ffmpeg exited with code -9" and every segment fails.

Diagnosis: Signal 9 (SIGKILL) can only be sent by an external process --
the application itself cannot exit this way voluntarily. All prior
commands had been run directly on the SSH login node.

Root Cause: This HPC cluster uses SLURM job scheduling. Login nodes are
reserved for lightweight interactive use (editing code, submitting jobs)
and enforce resource limits; moderate-load tasks such as video
downloading/transcoding exceed those limits and are forcibly terminated
to protect the shared login node for all users.

Fix: Requested an interactive compute-node session with
srun --partition=cpu --cpus-per-task=4 --mem=8G --time=01:00:00 --pty bash,
then re-ran the same commands from within that session. The job completed
successfully.

Takeaway: This was not a code bug but a mismatch between a local-
development mental model ("this machine is mine to use however I want")
and a shared-cluster architecture, where compute must be explicitly
requested on dedicated nodes. Validating that code logic is correct and
validating that it runs within a target deployment environment's resource
constraints are two distinct kinds of feasibility, and both must be
checked before scaling up.

## Cross-Platform Validation Result Comparison

| Metric                  | Local macOS (4 samples) | HPC (5 samples) |
|--------------------------|-------------------------|------------------|
| Download success rate    | 4/5 (1 dead source)     | 5/5              |
| PTS validation           | 4/4 usable              | 5/5 usable       |
| Face detection rate      | 4/4 >= 99.85%           | 5/5 = 100%       |
| Geometry extraction rate | 4/4 = 100%              | 5/5 = 100%       |

Results are highly consistent across platforms, indicating good
cross-platform reliability of the codebase.
