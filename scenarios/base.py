from typing import Callable, Optional

Hook = Callable[[object, int], None]

def make_hook(args=None) -> Hook:
    """
    S0_BASE: no extra operations.
    Returns a hook compatible with core.
     """
    def _hook(qc, cycle_idx: int):
        return  # do nothing
    return _hook
