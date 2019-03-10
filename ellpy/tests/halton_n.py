import math
import numpy as np
from ellpy.tests.vdcorput import vdcorput


def halton_n(k, n, b):
    """Generate base-b Halton sequence 0,..,k

    Arguments:
        k {int} -- maximum sequence index, non-negative integer
        n {int} -- [description]
        b {list(int)} -- sequence base, integer exceeding 1

    Returns:
        {list(float)} -- base-b low discrepancy sequence
    """
    if n == 1:
        for s in vdcorput(k, b[0]):
            yield [s]
        return

    S = halton_n(k, n-1, b[1:])

    for vd in vdcorput(k, b[0]):
        yield [vd] + next(S)


if __name__ == "__main__":
    b = [2, 3]
    points = [p for p in halton_n(10, 2, b)]
    print(points)
