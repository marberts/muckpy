from math import floor, fsum
from collections import Counter
from typing import Callable

def largest_remainder(size: dict[str, float], n: int) -> Counter[str, int]:
    if n < 0:
        raise ValueError(
            "n must be positive"
            )
    res, remainder, total = Counter(), dict(), 0
    sumx = fsum(size.values())
    for k, x in size.items():
        np = n * x / sumx
        npf = floor(np)
        res[k], remainder[k] = npf, npf - np
        total += npf
    remainders = sorted(remainder, key=remainder.get)[:n - total]
    res.update(dict.fromkeys(remainders, 1))
    return res
            
def highest_average(size: dict[str, float], 
                    n: int, *,
                    divisor: Callable[int, float]=lambda x: x + 1) -> Counter[str, int]:
    if n < 0:
        raise ValueError(
            "n must be positive"
            )
    res = Counter(dict.fromkeys(size, 0))
    while n > 0:
        td = {k: v / divisor(res[k]) for k, v in size.items()}
        res[max(td, key=td.get)] += 1
        n -= 1
    return res

def allocate(size: dict[str, float],
             n:int, *,
             units: dict[str, int]=None,
             initial: int=0,
             rounding: Callable[[dict, int], Counter]=largest_remainder):
    if initial * len(size) > n:
        raise ValueError(
            f"initial allocation cannot be larger than {initial/len(size)}"
            )
    res = Counter(dict.fromkeys(size, initial))
    if units is None:
        units = dict.fromkeys(res, n)
    units = Counter(units)
    n -= initial * len(size)
    while n > 0:
        res.update(rounding(size, n))
        n = 0
        for k, v in res.items():
            if v > units[k]:
                res[k] = units[k]
                n += v - units[k]
                size[k] = 0
    return res

if __name__ == "__main__":
    assert largest_remainder({"a": 1, "c": 1, "b": 1}, 0) == {"a": 0, "c": 0, "b": 0}
    assert largest_remainder({"a": 1, "c": 1, "b": 1}, 1) == {"a": 1, "c": 0, "b": 0}
    assert largest_remainder({"a": 1, "b": 1, "c": 1}, 2) == {"a": 1, "b": 1, "c": 0}
    assert largest_remainder({"a": 1, "c": 1, "b": 1}, 2) == {"a": 1, "c": 1, "b": 0}
    assert largest_remainder({"a": 0.25, "b": 0.2, "c": 0.4, "d": 0.15}, 100) == {"a": 25, "b": 20, "c": 40, "d": 15}
    assert largest_remainder({"a": 0.25, "b": 0.2, "c": 0.4, "d": 0.15}, 10) == {"a": 3, "b": 2, "c": 4, "d": 1}
    assert largest_remainder({"a": 0.25, "b": 0.2, "c": 0.4, "d": 0.15}, 4) == {"a": 1, "b": 1, "c": 2, "d": 0}
    assert largest_remainder({"a": 25, "b": 20, "c": 40, "d": 15, "e": 0}, 4) == {"a": 1, "b": 1, "c": 2, "d": 0, "e": 0}

    assert allocate({"a": 1, "b": 20, "c": 300}, 6) == {"a": 0, "b": 0, "c": 6}
    assert allocate({"a": 1, "b": 20, "c": 300}, 3, units={"a": 6, "b": 6, "c": 0}) == {"a": 0, "b": 3, "c": 0}
    assert allocate({"a": 1, "b": 20, "c": 300}, 6, units={"a": 6, "b": 2, "c": 3}) == {"a": 1, "b": 2, "c": 3}
    assert allocate({"a": 1, "b": 20, "c": 300}, 6, units={"a": 6, "b": 2, "c": 3}, initial=2) == {"a": 2, "b": 2, "c": 2}
    
    print("Passing all tests.")
    