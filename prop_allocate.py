from math import floor, fsum, sqrt
from collections import Counter
from typing import Callable

def largest_remainder(size: dict[str, float], n: int) -> Counter[str, int]:
    """
    Generate an allocation using the largest remainder method.

    Parameters
    ----------
    size : dict[str, float]
        A dict of positive numbers (not all zero) giving the size for each
        unit in the population.
    n : int
        The number of units in the allocation.

    Returns
    -------
    Counter[str, int]
        An allocation.
        
    Examples
    --------
    # Alabama paradox
    >>> sizes = {"a": 6 / 14, "b": 6 / 14, "c": 2 / 14}
    >>> largest_remainder(sizes, 10)
    Counter({'a': 4, 'b': 4, 'c': 2})
    >>> largest_remainder(sizes, 11)
    Counter({'a': 5, 'b': 5, 'c': 1})
    """
    
    if n < 0:
        raise ValueError(
            "n must be positive"
            )
    if any(x < 0 for x in size.values()):
        raise ValueError(
            "size cannot have negative values"
            )
    res, remainder, total = Counter(), dict(), 0
    sumx = fsum(size.values())
    if sumx == 0:
        raise ValueError(
            "size cannot have all zeros"
            )
    for k, x in size.items():
        np = n * x / sumx
        npf = floor(np)
        res[k], remainder[k] = npf, npf - np
        total += npf
    add_one = sorted(remainder, key=remainder.get)[:n - total]
    res.update(dict.fromkeys(add_one, 1))
    return res

def divisor(name: str) -> Callable[int, float]:
    """
    Divisors for the highest averages method.

    Parameters
    ----------
    name : str
        One of Adams, Dean, Huntington-Hill, Webster, D'Hondt, Imperiali,
        or Danish.

    Returns
    -------
    Callable[int, float]
        The given divisor function.

    Examples
    --------
    >>> [divisor("D'Hondt")(x) for x in range(5)]
    [1, 2, 3, 4, 5]
    >>> [divisor("Webster")(x) for x in range(5)]
    [0.5, 1.5, 2.5, 3.5, 4.5]
    """
    
    if name.lower() == "adams":
        return lambda x: x
    elif name.lower() == "dean":
        return lambda x: x * (x + 1) / (x + 0.5)
    elif name.lower() == "huntington-hill":
        return lambda x: sqrt(x * (x + 1))
    elif name.lower() == "webster":
        return lambda x: x + 0.5
    elif name.lower() == "d'hondt":
        return lambda x : x + 1
    elif name.lower() == "imperiali":
        return lambda x: x + 2
    elif name.lower() == "danish":
        return lambda x: x + 1 / 3
    else:
        raise ValueError(
            "name must be one of Adams, Dean, Huntington-Hill, Webster, D'Hondt, Imperiali, or Danish"
            )
            
def highest_average(
        size: dict[str, float], 
        n: int, *,
        divisor: Callable[int, float]=divisor("D'Hondt")
        ) -> Counter[str, int]:
    
    """
    Generate an allocation using the highest averages method.

    Parameters
    ----------
    size : dict[str, float]
        A dict of positive numbers (not all zero) giving the size for each
        unit in the population.
    n : int
        The number of units in the allocation.
    divisor : Callable[int, float], optional
        The divisor function. The default is the D'Hondt divisor.

    Returns
    -------
    Counter[str, int]
        An allocation.

    Examples
    --------
    >>> sizes = {"a": 6 / 14, "b": 6 / 14, "c": 2 / 14}
    >>> highest_averages(sizes, 10)
    Counter({'a': 5, 'b': 4, 'c': 1})
    >>> highest_averages(sizes, 11)
    Counter({'a': 5, 'b': 5, 'c': 1})
    """
    
    if n < 0:
        raise ValueError(
            "n must be positive"
            )
    if any(x < 0 for x in size.values()):
        raise ValueError(
            "sizes cannot be negative"
            )
    res = Counter(dict.fromkeys(size, 0))
    while n > 0:
        td = {k: v / divisor(res[k]) for k, v in size.items()}
        res[max(td, key=td.get)] += 1
        n -= 1
    return res

def allocate(
        size: dict[str, float],
        n: int, *,
        units: dict[str, int]=None,
        initial: int=0,
        method: Callable[[dict, int], Counter]=largest_remainder
        ) -> Counter[str, int]:
    """
    Generate a proportional-to-size allocation with units of different sizes
    from a population.

    Parameters
    ----------
    size : dict[str, float]
        A dict of positive numbers (not all zero) giving the size for each
        unit in the population.
    n : int
        The number of units in the allocation.
    units : dict[str, int], optional
        The number of available units in the population. The default puts no 
        limit on the allocation size for each unit.
    initial : int, optional
        The initial allocation for each unit. The default is 0.
    method : Callable[[dict, int], Counter], optional
        The method used to round proportions into integers. The default is 
        largest remainder; highest averages can be had by supplying the
        highest_averages function.

    Returns
    -------
    Counter[str, int]
        An allocation.

    """
    if n < 0:
        raise ValueError(
            "n must be positive"
            )
    if initial < 0:
        raise ValueError(
            "intial must be positive"
            )
    if initial * len(size) > n:
        raise ValueError(
            f"initial allocation cannot be larger than {int(initial/len(size))}"
            )
    res = Counter(dict.fromkeys(size, initial))
    if units is None:
        units = Counter(dict.fromkeys(res, n))
    elif set(size) == set(units):
        units = Counter(units)
    else:
        raise ValueError(
            "size and units must have the same keys"
            )
    if any(x < 0 for x in units.values()):
        raise ValueError(
            "units cannot have negative values"
            )
    n -= initial * len(size)
    while n > 0:
        res.update(method(size, n))
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
    # Alabama paradox
    assert allocate({"a": 6 / 14, "b": 6 / 14, "c": 2 / 14}, 10) == {"a": 4, "b": 4, "c": 2}
    assert allocate({"a": 6 / 14, "b": 6 / 14, "c": 2 / 14}, 11) == {"a": 5, "b": 5, "c": 1}
    # No sample
    assert allocate({"a": 1, "b": 2}, 0) == {"a": 0, "b": 0}
    assert allocate({"a": 1, "b": 2}, 0, method=highest_average) == {"a": 0, "b": 0}
    # Three iterations
    assert allocate({"a": 1, "b": 20, "c": 300}, 6) == {"a": 0, "b": 0, "c": 6}
    assert allocate({"a": 1, "b": 20, "c": 300}, 3, units={"a": 6, "b": 6, "c": 0}) == {"a": 0, "b": 3, "c": 0}
    assert allocate({"a": 1, "b": 20, "c": 300}, 6, units={"a": 6, "b": 2, "c": 3}) == {"a": 1, "b": 2, "c": 3}
    assert allocate({"a": 1, "b": 20, "c": 300}, 6, units={"a": 6, "b": 2, "c": 3}, initial=2) == {"a": 2, "b": 2, "c": 2}
    
    print("Passing all tests.")
    