from math import cos, pi, sin, sqrt
from typing import List

twoPI = 2 * pi


def vdc(k: int, base: int = 2) -> float:
    """[summary]

    Arguments:
        k (int): number

    Keyword Arguments:
        base (int): [description] (default: {2})

    Returns:
        int: [description]
    """
    vdc: float = 0.0
    denom: float = 1.0
    while k != 0:
        denom *= base
        remainder: int = k % base
        k //= base
        vdc += remainder / denom
    return vdc


class vdcorput:
    def __init__(self, base: int = 2):
        """[summary]

        Args:
            base (int, optional): [description]. Defaults to 2.
        """
        self._base: int = base
        self._count: int = 0

    def __call__(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        self._count += 1
        return vdc(self._count, self._base)

    def reseed(self, seed: int):
        self._count = seed


class halton:
    """Generate Halton sequence"""

    def __init__(self, base: List[int]):
        self._vdc0 = vdcorput(base[0])
        self._vdc1 = vdcorput(base[1])

    def __call__(self) -> List[float]:
        """Get the next item

        Returns:
            list(float):  the next item
        """
        return [self._vdc0(), self._vdc1()]

    def reseed(self, seed: int):
        self._vdc0.reseed(seed)
        self._vdc1.reseed(seed)


class circle:
    """Generate Circle Halton sequence 0,..,k

    Arguments:
        k (int): maximum sequence index, non-negative integer

    Keyword Arguments:
        base (int): [description] (default: {2})

    Returns:
        ([float]): base-b low discrepancy sequence
    """

    def __init__(self, base: int = 2):
        self._vdc = vdcorput(base)

    def __call__(self) -> List[float]:
        """Get the next item

        Raises:
            StopIteration:  description

        Returns:
            list:  the next item
        """
        theta = twoPI * self._vdc()  # map to [0, 2*math.pi]
        return [sin(theta), cos(theta)]

    def reseed(self, seed: int):
        self._vdc.reseed(seed)


class sphere:
    """Generate Sphere Halton sequence 0,..,k

    Arguments:
        k (int): maximum sequence index, non-negative integer

    Keyword Arguments:
        b ([int]): sequence base, integer exceeding 1

    Returns:
        ([float]): base-b low discrepancy sequence
    """

    def __init__(self, base: List[int]):
        assert len(base) >= 2
        self._vdc = vdcorput(base[0])
        self._cirgen = circle(base[1])

    def __call__(self) -> List[float]:
        """Get the next item

        Returns:
            list:  the next item
        """
        cosphi = 2 * self._vdc() - 1  # map to [-1, 1]
        sinphi = sqrt(1 - cosphi * cosphi)
        cc = self._cirgen()
        return [sinphi * cc[0], sinphi * cc[1], cosphi]

    def reseed(self, seed: int):
        self._cirgen.reseed(seed)
        self._vdc.reseed(seed)


class sphere3_hopf:
    """
    sphere3_hopf   Halton sequence
    INPUTS   : k - maximum sequence index, non-negative integer
               b - sequence base, integer exceeding 1
    """

    def __init__(self, base: List[int]):
        assert len(base) >= 3
        self._vdc0 = vdcorput(base[0])
        self._vdc1 = vdcorput(base[1])
        self._vdc2 = vdcorput(base[2])

    def __call__(self) -> List[float]:
        """Get the next item

        Returns:
            list:  the next item
        """
        phi = self._vdc0() * twoPI  # map to [0, 2*math.pi]
        psy = self._vdc1() * twoPI  # map to [0, 2*math.pi]
        # zzz = self._vdc2() * 2 - 1  # map to [-1., 1.]
        # eta = math.acos(zzz) / 2
        # cos_eta = math.cos(eta)
        # sin_eta = math.sin(eta)
        vd = self._vdc2()
        cos_eta = sqrt(vd)
        sin_eta = sqrt(1 - vd)
        return [
            cos_eta * cos(psy),
            cos_eta * sin(psy),
            sin_eta * cos(phi + psy),
            sin_eta * sin(phi + psy),
        ]

    def reseed(self, seed: int):
        self._vdc0.reseed(seed)
        self._vdc1.reseed(seed)
        self._vdc2.reseed(seed)


class halton_n:
    """Generate base-b Halton sequence

    Arguments:
        n (int): [description]
        b ([int]): sequence base, integer exceeding 1

    Returns:
        ([float]): base-b low discrepancy sequence
    """

    def __init__(self, n: int, base: List[int]):
        self._vec_vdc = [vdcorput(base[i]) for i in range(n)]

    def __call__(self) -> List[float]:
        """Get the next item

        Returns:
            list(float):  the next item
        """
        return [vdc() for vdc in self._vec_vdc]

    def reseed(self, seed: int):
        for vdc in self._vec_vdc:
            vdc.reseed(seed)


if __name__ == "__main__":
    halgen = halton([2, 3])
    for _ in range(10):
        print(halgen())

    halngen = halton_n(4, [2, 3, 5, 7])
    for _ in range(10):
        print(halngen())

    spgen = sphere([2, 3, 5, 7])
    for _ in range(10):
        print(spgen())

    sphgen = sphere3_hopf([2, 3, 5, 7])
    for _ in range(10):
        print(sphgen())
