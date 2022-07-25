"""
Higher-order functions.

Steve Martin
"""

__all__ = ["negate", "position", "compose"]

from typing import Callable, Iterator
from functools import reduce

def negate(f: Callable[..., bool]) -> Callable[..., bool]:
    """
    Negate a predicate function.

    Parameters
    ----------
    f : Callable[..., bool]
        Predicate function.

    Returns
    -------
    Callable[..., bool]
        Negation of f.
        
    Examples
    --------
    >>> negate(lambda x: x > 1)(2)
    False
    """
    
    return lambda *args, **kwargs: not f(*args, **kwargs)

def position(f: Callable[..., bool], x: list) -> Iterator[int]:
    """
    Find the positions at which a predicate function returns True.

    Parameters
    ----------
    f : Callable[..., bool]
        Predicate function.
    x : list
        A list of arguments for f.

    Returns
    -------
    Iterator[int]
        An interator that yields the positions at which f(x) is True.
        
    Examples
    --------
    >>> list(position(lambda x: x > 1, [1, 2, 3]))
    [1, 2]
    """
    
    return (i for i, z in enumerate(x) if f(z))
        
def compose(*f: Callable) -> Callable:
    """
    Compose a collection of functions.

    Parameters
    ----------
    *f : Callable
        A function taking a single argument.

    Returns
    -------
    Callable
        The composition of all supplied functions.
        
    Examples
    --------
    >>> def add(n):
            return lambda x: x + n
    >>> compose(add(1), add(2), sum)(range(3))
    6
    """
    
    def circ(f, g):
        return lambda x: f(g(x))
    return reduce(circ, f)

if __name__ == "__main__":
    assert not negate(lambda x: x > 1)(2)
    assert list(position(lambda x: x > 1, [1, 2, 3])) == [1, 2]
    assert compose(lambda x: x + 1, sum)(range(3)) == 4
    
    def add(n):
        return lambda x: x + n
    assert compose(add(1), add(2), sum)(range(3)) == 6

    print("Passing all tests")