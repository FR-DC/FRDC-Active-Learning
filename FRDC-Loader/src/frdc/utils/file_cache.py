from pathlib import Path
from typing import Callable, Hashable

import numpy as np
import xxhash


def file_cache(
    fn_cache_fp: Callable[[str], Path],
    fn_load_object: Callable[[Path], object] = np.load,
    fn_save_object: Callable[[Path, object], None] = np.save,
    fn_make_hashable: Callable[[object], Hashable] = str,
):
    """A decorator that caches the output of a function to a file

    Notes:
        This caching function uses xxhash.xxh32 to hash.
        Thus, arguments must be hashable by xxhash.xxh32.

    Args:
        fn_cache_fp: A function that takes a hash string and returns a Path
            to the cache file.
        fn_load_object: A function that takes a Path and returns an object
        fn_save_object: A function that takes a Path and an object and saves
        fn_make_hashable: A function that takes an object and returns something
            hashable by xxhash.xxh32.

    Examples:
        >>> from pathlib import Path
        >>> from frdc.utils.file_cache import file_cache
        >>> from frdc.conf import ROOT_DIR
        >>>
        >>> @file_cache(cache_dir=ROOT_DIR / ".cache")
        >>> def my_fn(x):
        >>>     np.random.seed(1000)
        >>>     return x + np.random.randint(0, 10)
        >>>
        >>> my_fn(1) # This will be cached
        2
        >>> my_fn(1) # This will be loaded from cache
        2
        >>> my_fn(2) # This will be cached
        3

    Returns:
        A decorator that caches the output of a function to a file
    """

    def decorator(fn: Callable):
        def inner(*args, **kwargs):
            x = xxhash.xxh32()
            for v in [*args, *kwargs.values()]:
                x.update(fn_make_hashable(v))
            hash_str = x.hexdigest()
            cache_fp = fn_cache_fp(hash_str)
            cache_fp.parent.mkdir(parents=True, exist_ok=True)

            if cache_fp.exists():
                return fn_load_object(cache_fp)
            else:
                out = fn(*args, **kwargs)
                fn_save_object(cache_fp, out)
                return out

        return inner

    return decorator
