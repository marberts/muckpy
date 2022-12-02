import numpy as np
from collections import Counter
from typing import Callable

"""
Python implementation of the sps R package on CRAN.
"""

__all__ = [
    "pi",
    "sps",
    "ps",
    "allocate"
]

def _pi(
    x: np.ndarray, 
    n: int
) -> np.ndarray:
    return x * (n / np.sum(x))

def pi(
    x: np.ndarray,
    n: int
) -> np.ndarray:
    """
    First-order inclusion probabilities.

    Parameters
    ----------
    x : np.ndarray
        Sizes in the population. Should be a flat array of strictly 
        positive numbers.
    n : int
        Sample size.

    Returns
    -------
    np.ndarray
        Flat array of inclusion probabilities.
    
    Examples
    --------
    >>> pi([1, 3, 30, 100], 3)
    array([0.25, 0.75, 1.  , 1.  ])
    """
    x = np.asfarray(x).ravel()
    res = _pi(x, n)
    if np.max(res) <= 1:
        return res
    while True:
        ts = np.flatnonzero(res < 1)
        n_ts = n - len(x) + len(ts)
        p = _pi(x[ts], n_ts)
        res[ts] = p
        if np.max(p) <= 1:
            break
    return np.minimum(res, 1)

def sps(
    pi: np.ndarray, 
    n: int, 
    prn: np.ndarray = None,
    rng: np.random.Generator = np.random.default_rng()
) -> np.ndarray:
    """
    Draw a sequential Poisson sample.

    Parameters
    ----------
    pi : np.ndarray
        Inclusion probabilities. Should be a flat array of values
        between 0 and 1.
    n : int
        Sample size.
    prn : np.ndarray, optional
        Permanent random numbers. Should be a flat array of values
        distributed uniform between 0 and 1. The default does not 
        use permanent random numbers.
    rng : np.random.Generator, optional
        Random-number generator. The default is 
        np.random.default_rng().

    Returns
    -------
    np.ndarray
        Indexes for units in the sample.
        
    Examples
    --------
    >>> sps(pi([1, 3, 30, 100], 3), 3)
    array([0, 2, 3])
    """
    pi = np.asfarray(pi).ravel()
    ts = pi < 1
    ta, ts = np.flatnonzero(~ts), np.flatnonzero(ts)
    n_ts = n - len(ta)
    if n_ts == 0:
        return np.sort(ta)
    if prn is None:
        prn = rng.uniform(size=len(pi))
    else:
        prn = np.asfarray(prn).ravel()
    keep = np.argpartition(prn[ts] / pi[ts], n_ts)[:n_ts]
    return np.sort(np.concatenate([ta, ts[keep]]))

def ps(
    pi: np.ndarray, 
    prn: np.ndarray = None,
    rng: np.random.Generator = np.random.default_rng()
) -> np.ndarray:
    if prn is None:
        prn = rng.uniform(size=len(pi))
    else:
        prn = np.asfarray(prn).ravel()
    return np.sort(np.flatnonzero(prn < pi))

def allocate(
    x: dict, 
    n: int, 
    lower: dict = None, 
    upper: dict = None, 
    d: Callable[[int], float] = lambda a: a + 1
) -> Counter:
    res = Counter(dict.fromkeys(x, 0))
    lower = Counter(lower)
    if upper is None:
        upper = Counter(dict.fromkeys(x, n))
    else:
        upper = Counter(upper)
    if lower > upper:
        raise ValueError()
    if lower.total() > n:
        raise ValueError()
    if upper.total() < n:
        raise ValueError()
    for k in res:
        res[k] = lower[k]
    n -= res.total()
    while n > 0:
        div = {k: x[k] / d(a) for k, a in res.items() if upper[k] > a}
        i = max(div.keys(), key=div.get)
        res[i] += 1
        n -= 1
    return res