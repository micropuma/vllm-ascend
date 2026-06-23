"""Compile-safe snapshot of ``flash_comm_v1_enabled``.

Written once per forward context setup (:func:`set_ascend_forward_context`),
before ``torch.compile`` ever runs fake tensor propagation.  Fake impls must
read this snapshot instead of ``_EXTRA_CTX`` / ``get_forward_context()``
because ``PiecewiseBackend.compile_all_ranges()`` fires outside any active
forward context.

This module lives outside the ops tree to avoid circular imports with
``prepare_finalize.py`` → ``dispatch.py``.
"""

_FLASH_COMM_V1_SNAPSHOT: bool = False


def set_flash_comm_v1_snapshot(value: bool) -> None:
    global _FLASH_COMM_V1_SNAPSHOT
    _FLASH_COMM_V1_SNAPSHOT = value
