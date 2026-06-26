from __future__ import annotations

import csv
import sys
from typing import Any, TextIO

_CSV_FIELD_SIZE_LIMIT_CONFIGURED = False


def dict_reader(handle: TextIO, *args: Any, **kwargs: Any) -> csv.DictReader:
    """Return a DictReader after raising the CSV field-size limit for large payloads."""

    _ensure_csv_field_size_limit()
    return csv.DictReader(handle, *args, **kwargs)


def _ensure_csv_field_size_limit() -> None:
    global _CSV_FIELD_SIZE_LIMIT_CONFIGURED
    if _CSV_FIELD_SIZE_LIMIT_CONFIGURED:
        return

    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
        except OverflowError:
            limit //= 10
            continue
        _CSV_FIELD_SIZE_LIMIT_CONFIGURED = True
        return
