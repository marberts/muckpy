from math import floor, fsum
from collections import Counter

def largest_remainder_rounding(p: dict[str, float], n: int) -> Counter[str, int]:
    res, remainder, total = Counter(), {}, 0
    sumx = fsum(p.values())
    for k, x in p.items():
        np = n * x / sumx
        npf = floor(np)
        res[k], remainder[k] = npf, npf - np
        total += npf
    largest_remainder = sorted(remainder, key=remainder.get)[:n - total]
    res.update({k: 1 for k in largest_remainder})
    return res

if __name__ == "__main__":
    assert largest_remainder_rounding({"a": 1, "b": 1, "c": 1}, 2) == Counter({"a": 1, "b": 1, "c": 0})
    assert largest_remainder_rounding({"a": 1, "c": 1, "b": 1}, 2) == Counter({"a": 1, "c": 1, "b": 0})
    assert largest_remainder_rounding({"a": 1, "c": 1, "b": 1}, 0) == Counter({"a": 0, "c": 0, "b": 0})
    assert largest_remainder_rounding({"a": 0.25, "b": 0.2, "c": 0.4, "d": 0.15}, 100) == Counter({"a": 25, "b": 20, "c": 40, "d": 15})
    assert largest_remainder_rounding({"a": 0.25, "b": 0.2, "c": 0.4, "d": 0.15}, 10) == Counter({"a": 3, "b": 2, "c": 4, "d": 1})
    assert largest_remainder_rounding({"a": 0.25, "b": 0.2, "c": 0.4, "d": 0.15}, 4) == Counter({"a": 1, "b": 1, "c": 2, "d": 0})
    assert largest_remainder_rounding({"a": 0.25, "b": 0.2, "c": 0.4, "d": 0.15, "e": 0}, 4) == Counter({"a": 1, "b": 1, "c": 2, "d": 0, "e": 0})
