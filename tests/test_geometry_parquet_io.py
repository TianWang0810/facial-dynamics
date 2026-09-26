"""
Geometry parquet writer: a clip with no detected face must neither be dropped nor
poison the file schema.

Self-contained (pyarrow only, temporary directory).

Usage:
    python -m pytest tests/test_geometry_parquet_io.py -q
    python tests/test_geometry_parquet_io.py
"""
import os
import sys
import tempfile

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pyarrow as pa  # noqa: E402

from src.geometry.parquet_io import append_table  # noqa: E402


def clip(name, detected):
    rows = [{"clip_id": name, "frame_idx": k, "detected": detected,
             "eye_distance": 0.1 if detected else None,
             "blendshapes": [0.5, 0.2] if detected else None} for k in range(3)]
    return pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)


def write(tables):
    path = os.path.join(tempfile.mkdtemp(), "geometry.parquet")
    writer, pending = None, []
    for table in tables:
        writer = append_table(writer, pending, table, path)
    writer.close()
    return pq.read_table(path).to_pandas()


def test_a_faceless_clip_after_a_normal_one_is_kept():
    df = write([clip("a", True), clip("b", False)])
    assert list(df.groupby("clip_id").size()) == [3, 3]
    assert not df[df.clip_id == "b"].detected.any() and df[df.clip_id == "b"].eye_distance.isna().all()


def test_a_faceless_first_clip_does_not_fix_a_null_schema():
    df = write([clip("b", False), clip("a", True)])
    assert sorted(df.clip_id.unique()) == ["a", "b"]
    assert df[df.clip_id == "a"].eye_distance.tolist() == [0.1, 0.1, 0.1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
