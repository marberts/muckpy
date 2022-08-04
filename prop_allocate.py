from math import floor, fsum
from typing import Callable
    
def allocate(size: dict[str, float], 
             n: int, *,
             initial: int=0,
             divisor: Callable[int, float]=None) -> dict[str, int]:
    """
    Generate a proportional-to-size allocation.

    Parameters
    ----------
    size : dict[str, float]
        DESCRIPTION.
    n : int
        DESCRIPTION.
    initial : int, optional
        DESCRIPTION. The default is 0.
    divisor : Callable[int, float], optional
        DESCRIPTION. The default is None.
        
    Returns
    -------
    dict[str, int]
        DESCRIPTION.

    """
    if n < 0:
        raise ValueError(
            "n must be positive"
            )
    if initial * len(size) > n:
        raise ValueError(
            f"initial allocation cannot be larger than {floor(n / len(size))}"
            )
    n -= initial * len(size)
    res = dict.fromkeys(size, initial)
    if divisor is None:
        remainder, total = dict(), 0
        sumx = fsum(size.values())
        for k, x in size.items():
            np = n * x / sumx
            npf = floor(np)
            res[k] += npf
            remainder[k] = npf - np
            total += npf
        for k in sorted(remainder, key=remainder.get)[:n - total]:
            res[k] += 1
    else:
        while n > 0:
            td = {k: v / divisor(res[k]) for k, v in size.items()}
            res[max(td, key=td.get)] += 1
            n -= 1
    return res

if __name__ == "__main__":
    assert allocate({"a": 1, "b": 1, "c": 1}, 2) == {"a": 1, "b": 1, "c": 0}
    assert allocate({"a": 1, "c": 1, "b": 1}, 2) == {"a": 1, "c": 1, "b": 0}
    assert allocate({"a": 1, "c": 1, "b": 1}, 0) == {"a": 0, "c": 0, "b": 0}
    assert allocate({"a": 0.25, "b": 0.2, "c": 0.4, "d": 0.15}, 100) == {"a": 25, "b": 20, "c": 40, "d": 15}
    assert allocate({"a": 0.25, "b": 0.2, "c": 0.4, "d": 0.15}, 10) == {"a": 3, "b": 2, "c": 4, "d": 1}
    assert allocate({"a": 0.25, "b": 0.2, "c": 0.4, "d": 0.15}, 4) == {"a": 1, "b": 1, "c": 2, "d": 0}
    assert allocate({"a": 0.25, "b": 0.2, "c": 0.4, "d": 0.15, "e": 0}, 4) == {"a": 1, "b": 1, "c": 2, "d": 0, "e": 0}

    print("Passing all tests.")