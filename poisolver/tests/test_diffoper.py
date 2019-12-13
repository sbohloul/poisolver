import numpy as np
from ..diffoper import findifcoef


def test_findifcoef():
    "Finite difference coefficients for various stencils."

    # forward fd
    d, p = 3, 1     # order, accuracy
    n = d + p - 1
    sten = range(n+1)
    coefCalc = findifcoef(sten, d)
    coefExpc = np.array([-1, 3, -3, 1])/6
    assert np.allclose(coefCalc, coefExpc)

    # centeral fd
    d, p = 3, 2     # order, accuracy
    n = d + p - 1
    n = n//2
    sten = range(-n, n+1)
    coefCalc = findifcoef(sten, d)
    coefExpc = np.array([-1, 2, 0, -2, 1])/12
    assert np.allclose(coefCalc, coefExpc)

    # backward fd
    d, p = 4, 1     # order, accuracy
    n = d + p - 1
    sten = range(0, -(n+1), -1)
    coefCalc = findifcoef(sten, d)
    coefExpc = np.array([1, -4, 6, -4, 1])/24
    assert np.allclose(coefCalc, coefExpc)
