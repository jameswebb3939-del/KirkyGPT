from __future__ import annotations

from typing import Any


_import_error: Exception | None = None

try:
    from . import _kirk_gpt_native
except ImportError as exc:
    _kirk_gpt_native = None  # type: ignore[assignment]
    _import_error = exc


def native_available() -> bool:
    return _kirk_gpt_native is not None


def require_native() -> Any:
    if _kirk_gpt_native is None:
        raise RuntimeError(
            "KirkGPT native extension is not available. "
            "Build the native C++ module first."
        ) from _import_error

    return _kirk_gpt_native