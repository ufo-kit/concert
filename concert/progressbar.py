"""Progress bar allows you to see the progress of long-term operations."""

import concert.config as cfg
import inspect


try:
    import tqdm
    from tqdm.asyncio import tqdm as atqdm
except ImportError:
    cfg.PROGRESS_BAR = False


def wrap_iterable(iterable, **kwargs):
    """Wrap *iterable* so that a progress bar will be shown on iteration."""
    if cfg.PROGRESS_BAR:
        if inspect.isasyncgen(iterable):
            iterable = atqdm(iterable, **kwargs)
        else:
            iterable = tqdm.tqdm(iterable, **kwargs)

    return iterable
