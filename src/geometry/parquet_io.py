"""
Clip-by-clip parquet writing for the Geometry layer (scripts/run_geometry.py).

Kept apart from the stage script so it can be tested without MediaPipe.
"""
import pyarrow as pa
import pyarrow.parquet as pq


def append_table(writer, pending: list, table, path: str):
    """Write one clip's table; return the (possibly new) writer.

    The file schema comes from the first clip whose columns all have real types. A
    clip where MediaPipe never found a face has only nulls in the per-frame columns,
    so pandas -> arrow infers null types, which the open file rejects ("Table schema
    does not match"); it would also poison the file schema if it came first. Such a
    table waits in `pending` until a typed schema exists and is then cast to it, so
    the clip is kept (detected = False everywhere) and Features rejects it with a
    reason code instead of Geometry dropping it as extraction_failed.
    """
    if writer is None and any(pa.types.is_null(field.type) for field in table.schema):
        pending.append(table)
        return None
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema)
        for held in pending:
            writer.write_table(held.cast(writer.schema))
        pending.clear()
    writer.write_table(table.cast(writer.schema))
    return writer
