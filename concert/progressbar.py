"""Progress bar allows you to see the progress of long-term operations."""

import concert.config as cfg


try:
    import tqdm
except:
    cfg.PROGRESS_BAR = False


def wrap_iterable(iterable, **kwargs):
    """Wrap *iterable* so that a progress bar will be shown on iteration."""
    if cfg.PROGRESS_BAR:
        iterable = tqdm.tqdm(iterable, **kwargs)

    return iterable
