from __future__ import annotations

import sys


def require_typing_extensions_self(*, package: str = "qiskit") -> None:
    """
    Qiskit (and its deps) may require `typing_extensions.Self`.

    If the active environment has an old `typing_extensions`, Qiskit import fails with:
        ImportError: cannot import name 'Self' from 'typing_extensions'

    We check this early to give a clear, actionable error message.
    """
    try:
        import typing_extensions as te  # type: ignore
    except Exception as e:
        raise SystemExit(
            f"{package} import failed because `typing_extensions` is missing.\n"
            "Fix:\n"
            "  python -m pip install -U typing_extensions\n"
        ) from e

    if not hasattr(te, "Self"):
        where = getattr(te, "__file__", "typing_extensions (unknown path)")
        ver = getattr(te, "__version__", "unknown")
        raise SystemExit(
            f"{package} import failed because `typing_extensions.Self` is missing.\n"
            f"- typing_extensions version: {ver}\n"
            f"- typing_extensions path: {where}\n"
            "Fix (run inside your active environment):\n"
            "  python -m pip install -U typing_extensions\n"
        )

