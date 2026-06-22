"""T-01-04 - JSONL line writer for trace frames.

Pure I/O: takes already-built envelope dicts and writes them one-per-line. It
validates each envelope against schema v1.1.0 (the same enforcer the live stream
uses) *before* writing, so an invalid frame raises rather than being persisted.
Keys are sorted and separators are compact so a fixed sequence of frames yields
byte-identical output (the reproducibility contract; the weekly smoke test
T-01-07 hashes this).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Any

from src.schema.validate import validate_envelope


class JsonlWriter:
    """Append-by-line JSONL writer for ``sim_frame`` / ``kpi_frame`` envelopes.

    Use as a context manager::

        with JsonlWriter("trace.jsonl") as w:
            w.write(envelope)

    Parameters
    ----------
    path : str or Path
        Output file path. Parent directories are created on open.
    validate : bool, optional
        If ``True`` (default), each envelope is checked with
        :func:`src.schema.validate.validate_envelope` before writing and a
        :class:`src.schema.validate.SchemaError` is raised on violation.
    """

    def __init__(self, path: str | Path, *, validate: bool = True) -> None:
        self._path = Path(path)
        self._validate = validate
        self._fh = None
        self._count = 0

    def __enter__(self) -> "JsonlWriter":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n" keeps line endings stable across OSes (byte-identity).
        self._fh = self._path.open("w", encoding="utf-8", newline="\n")
        return self

    def write(self, envelope: dict[str, Any]) -> None:
        """Validate (if enabled) and write one envelope as a single JSON line.

        Raises
        ------
        RuntimeError
            If called outside the context manager (file not open).
        src.schema.validate.SchemaError
            If ``validate`` is enabled and the envelope is non-conforming.
        """
        if self._fh is None:
            raise RuntimeError("JsonlWriter.write() called while file is not open")
        if self._validate:
            validate_envelope(envelope)
        json.dump(envelope, self._fh, separators=(",", ":"), sort_keys=True)
        self._fh.write("\n")
        self._count += 1

    @property
    def count(self) -> int:
        """Number of frames written so far."""
        return self._count

    @property
    def path(self) -> Path:
        """The output file path."""
        return self._path

    def close(self) -> None:
        """Flush and close the underlying file (idempotent)."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        self.close()
        return False
