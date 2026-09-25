"""
Input resolution shared by the L1 CLIs: a run directory (paths derived through
src/pipeline/layout.py, never joined by hand) or explicit artifact paths.
"""
import json
import os

import numpy as np

from src.pipeline.layout import STAGE_FEATURES, STAGE_GEOMETRY, RunLayout
from src.semantic.l1 import blendshape_names_from_metadata, gaze_unit_from_metadata


def add_input_arguments(parser) -> None:
    parser.add_argument("--run_dir", default=None, help="A pipeline run directory; artifact paths are derived from it.")
    parser.add_argument("--features_npz", default=None)
    parser.add_argument("--geometry_metadata", default=None)


def resolve_inputs(args) -> tuple:
    """(features.npz, geometry_metadata.json) from --run_dir or the explicit paths."""
    if args.run_dir:
        run_dir = os.path.abspath(args.run_dir)
        layout = RunLayout(root=os.path.dirname(run_dir), run_id=os.path.basename(run_dir))
        return (layout.artifact(STAGE_FEATURES, "features.npz"),
                layout.artifact(STAGE_GEOMETRY, "geometry_metadata.json"))
    if not (args.features_npz and args.geometry_metadata):
        raise SystemExit("give --run_dir, or both --features_npz and --geometry_metadata")
    return args.features_npz, args.geometry_metadata


def load_inputs(args) -> tuple:
    """(npz dict, blendshape names, gaze unit, source paths)."""
    features_path, metadata_path = resolve_inputs(args)
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    with np.load(features_path) as data:
        npz = {key: data[key] for key in data.files}
    source = {"features_npz": os.path.abspath(features_path), "geometry_metadata": os.path.abspath(metadata_path)}
    return npz, blendshape_names_from_metadata(metadata), gaze_unit_from_metadata(metadata), source
