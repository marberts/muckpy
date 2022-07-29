"""
Find the expected number of different colors when drawing balls of
different colors from an urn.

Steve Martin
"""

__all__ = ["balanced_urn", "uniform_urn", "expected_coverage"]

from math import exp, prod, fsum, lgamma, comb
from itertools import repeat
from operator import gt

# Analogous to R's lchoose()
def _lperm(n: int, k: int) -> float:
    return lgamma(n + 1) - lgamma(n - k + 1)

def _urn_matrix(*urns: dict[str, int]) -> dict[str, list]:
    # {'red': 1, 'blue': 2, 'green': 3}        red 1 4 6
    # {'red': 4, 'green': 5}             =>   blue 2 0 7
    # {'red': 6, 'blue': 7}                  green 3 5 0

    colors = set().union(*urns)
    return {c: [u.get(c, 0) for u in urns] for c in colors}

def balanced_urn(balls: int, colors: set[str]) -> dict[str, int]:
    """
    Make an urn with balls distributed as equally as possible across the
    different colors.

    Parameters
    ----------
    balls : int
        Number of balls in the urn.
    colors : set[str]
        The set of colors in the urn.

    Returns
    -------
    dict[str, int]
        An urn of the form {color: number of balls}.
    
    Examples
    --------
    >>> balanced_urn(4, {"red", "blue", "green"})
    {'red': 2, 'blue': 1, 'green': 1}
    """
    
    if balls < 0:
        raise ValueError(
            "cannot make an urn with a negative number of balls"
            )
    n = len(set(colors))
    if n == 0:
        raise ValueError(
            "there must be colors for the urn"
            )
    k = balls // n
    r = balls - k * n # 0 <= d < n
    return dict(zip(colors, [k + 1]*r + [k]*(n - r)))

def uniform_urn(balls: int, colors: set[str]) -> dict[str, int]:
    """
    Make an urn with balls distributed as uniformly as possible across the
    different colors.

    Parameters
    ----------
    balls : int
        Number of balls in the urn.
    colors : set[str]
        The set of colors in the urn.
        
    Returns
    -------
    dict[str, int]
        An urn of the form {color: number of balls}.
        
    Examples
    --------
    >>> uniform_urn(6, {"red", "blue", "green"})
    {'red': 1, 'blue': 2, 'green': 3}
    """
    
    n = len(set(colors))
    if n == 0:
        raise ValueError(
            "there must be colors for the urn"
            )
    k = int(n * (n + 1) / 2)
    if balls < k:
        raise ValueError(
            f"there must be a least {k} balls in the urn"
            )
    a = (balls - k) // n + 1
    r = balls - (a - 1) * n - k
    return dict(zip(colors, [a + i + (i < r) for i in range(n)])) 

def expected_coverage(draws: list[int], 
                      *urns: dict[str, int], 
                      replace: bool=False, 
                      exact: bool=False) -> float:
    """
    Find the expected number of different colors when drawing balls
    independently from a sequence of urns.

    Parameters
    ----------
    draws : list[int]
        A list of integers giving the number of draws from each urn.
    *urns : dict[str, int]
        A collection of urns, one for each element in draws, of the 
        form {color: number of balls}.
    replace : bool, optional
        Is sampling done with replacement? The default is False.
    exact : bool, optional
        If True, use Python's arbitrary-precision integers for calculating the
        ratio of binomial coefficients when replace=False. If False, use a 
        faster floating-point approximation. The default is False.

    Returns
    -------
    float
        Expected number of different colors.

    Examples
    --------
    urn = {"red": 1, "blue": 2, "green": 3}
    >>> expected_coverage([3], urn)
    2.25
    >>> expected_coverage([3], urn, replace=True)
    2.0
    >>> expected_coverage([1, 1], {"a": 1, "b": 1}, {"a": 1, "c": 1})
    1.75
    """
    
    if len(draws) != len(urns):
        raise ValueError(
            "number of draws does not equals the number of urns"
            )
    balls = [sum(u.values()) for u in urns]
    if any(map(gt, draws, balls)) and not replace:
        raise ValueError(
            "cannot draw more balls than are in the urns without replacement"
            )
    urn = _urn_matrix(*urns)
    # Function that gives probability of drawing no balls of a given color
    # Precompute the denominators for each urn for efficiency
    if replace:
        den = repeat(None)
        def p(color, balls, n, den): # den is a dummy argument
            return (1 - color / balls)**n
    else:
        if exact:
            den = list(map(comb, balls, draws))
            def p(color, balls, n, den):
                return comb(balls - color, n) / den
        else:
            den = list(map(_lperm, balls, draws))
            def p(color, balls, n, den):
                if balls - color + 1 > n:
                    return exp(_lperm(balls - color, n) - den)
                else:
                    return 0.0
    # Apply over each urn to get the probability of drawing no balls of a
    # given color across all urns, and add the complements to get the
    # expected number of different colors
    return fsum(1 - prod(map(p, urn[c], balls, draws, den)) for c in urn)
        
if __name__ == "__main__":
    from math import isclose
    
    # Tests for balanced_urn()
    assert balanced_urn(0, range(1)) == {0: 0}
    assert balanced_urn(4, range(1)) == {0: 4}
    assert balanced_urn(4, range(2)) == {0: 2, 1: 2}
    assert balanced_urn(4, range(3)) == {0: 2, 1: 1, 2: 1}
    assert balanced_urn(4, range(4)) == {0: 1, 1: 1, 2: 1, 3: 1}
    assert balanced_urn(4, range(5)) == {0: 1, 1: 1, 2: 1, 3: 1, 4: 0}
    
    # Length and sum are determined by the length of colors and balls
    assert len(balanced_urn(89, range(13))) == 13
    assert sum(balanced_urn(89, range(13)).values()) == 89
    
    # Bounded by floor and ceil
    assert min(balanced_urn(78, range(11)).values()) == 7
    assert max(balanced_urn(78, range(11)).values()) == 8
    
    # Tests for uniform urn
    assert uniform_urn(6, range(3)) == {0: 1, 1: 2, 2: 3}
    assert uniform_urn(7, range(3)) == {0: 2, 1: 2, 2: 3}
    assert uniform_urn(8, range(3)) == {0: 2, 1: 3, 2: 3}
    assert uniform_urn(9, range(3)) == {0: 2, 1: 3, 2: 4}
    
    # Length and sum are determined by the length of colors and balls
    assert len(uniform_urn(189, range(13))) == 13
    assert sum(uniform_urn(189, range(13)).values()) == 189
    
    assert min(uniform_urn(465, range(30)).values()) == 1
    assert max(uniform_urn(465, range(30)).values()) == 30
    
    # Tests for expected_coverage()
    urn1 = {"a": 1, "b": 2, "c": 3, "d": 0}
    assert isclose(expected_coverage([0], urn1), 0)
    assert isclose(expected_coverage([0], urn1, exact=True), 0)
    assert isclose(expected_coverage([0], urn1, replace=True), 0)
    
    assert isclose(expected_coverage([1], urn1), 1)
    assert isclose(expected_coverage([1], urn1, exact=True), 1)
    assert isclose(expected_coverage([1], urn1, replace=True), 1)
    
    assert isclose(expected_coverage([3], urn1), 2.25)
    assert isclose(expected_coverage([3], urn1, exact=True), 2.25)
    assert isclose(expected_coverage([3], urn1, replace=True), 2)

    # Order doesn't matter
    urn2 = {"a": 1, "c": 3, "b": 2}
    assert isclose(expected_coverage([3, 2], urn1, urn2),
                   expected_coverage([2, 3], urn2, urn1))
    assert isclose(expected_coverage([3, 2], urn1, urn2, exact=True),
                   expected_coverage([2, 3], urn2, urn1, exact=True))
    assert isclose(expected_coverage([3, 2], urn1, urn2, replace=True),
                   expected_coverage([2, 3], urn2, urn1, replace=True))
    
    # Perfect coverage
    assert isclose(expected_coverage([1, 1, 1], {"a": 5}, {"b": 4}, {"c": 3}), 3)
    assert isclose(expected_coverage([1, 1, 1], {"a": 5}, {"b": 4}, {"c": 3}, exact=True), 3)
    assert isclose(expected_coverage([1, 1, 1], {"a": 5}, {"b": 4}, {"c": 3}, replace=True), 3)
    
    assert isclose(expected_coverage([3, 3, 3], {"a": 5}, {"b": 5}, {"c": 5}), 3)
    assert isclose(expected_coverage([3, 3, 3], {"a": 5}, {"b": 5}, {"c": 5}, exact=True), 3)
    assert isclose(expected_coverage([3, 3, 3], {"a": 5}, {"b": 5}, {"c": 5}, replace=True), 3)
    
    assert isclose(expected_coverage([3, 2, 4], {"a": 3}, {"b": 2}, {"c": 4}), 3)
    assert isclose(expected_coverage([3, 2, 4], {"a": 3}, {"b": 2}, {"c": 4}, exact=True), 3)
    assert isclose(expected_coverage([3, 2, 4], {"a": 3}, {"b": 2}, {"c": 4}, replace=True), 3)
    
    # Balanced urn
    urn3 = {"a": 3, "b": 3, "c": 3, "d": 3}
    assert isclose(expected_coverage([5], urn3),
                   4 * (1 - comb(12 - 3, 5) / comb(12, 5)))
    assert isclose(expected_coverage([5], urn3, exact=True),
                   4 * (1 - comb(12 - 3, 5) / comb(12, 5)))
    assert isclose(expected_coverage([5], urn3, replace=True),
                   4 * (1 - (1 - 1 / 4)**5))
    
    # Known cases
    assert isclose(expected_coverage([2, 2], urn1, urn1, replace=True),
                   expected_coverage([4], {"a": 2, "b": 4, "c": 6}, replace=True))
    assert isclose(expected_coverage([1, 1], {"a": 1, "b": 1}, {"a": 1, "c": 1}), 1.75)
    assert isclose(expected_coverage([1, 1], {"a": 1, "b": 1}, {"a": 1, "c": 1}, exact=True), 1.75)
    assert isclose(expected_coverage([1, 1], {"a": 1, "b": 1}, {"a": 1, "c": 1}, replace=True), 1.75)
    
    urn4 = {"d": 12, "b": 1}
    assert isclose(expected_coverage([3, 2, 4], urn3, urn2, urn4),
                   3.56335664335664)
    assert isclose(expected_coverage([3, 2, 4], urn3, urn2, urn4, exact=True),
                   3.56335664335664)    
    assert isclose(expected_coverage([3, 2, 4], urn3, urn2, urn4, replace=True),
                   3.4654180416477)
    
    # Simulation to help verify
    # from random import sample, choices
    # from statistics import mean

    # wor, wr = [], []

    # u3, u2, u4 = ["".join(k*v for k, v in zip(urn, urn.values())) 
    #               for urn in [urn3, urn2, urn4]]

    # for i in range(10000):
    #     wor.append(len(set(sample(u3, k=3) + sample(u2, k=2) + sample(u4, k=4))))
    #     wr.append(len(set(choices(u3, k=3) + choices(u2, k=2) + choices(u4, k=4))))

    # mean(wor)
    # mean(wr)

    print("Passing all tests.")    
    