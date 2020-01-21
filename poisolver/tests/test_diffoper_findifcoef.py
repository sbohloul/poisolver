'''Testing finite difference coefficients
    d      : d
    p      : puracy
    method : forward (ffd), backward (bfd), center (cfd)    
'''    

import pytest
import numpy as np
import scipy.sparse as sps
from ..diffoper import findifcoef


def findifsten(d, p, method):
    "Finite difference stencil."

    if method == 'ffd':
        sten = np.arange(d + p, dtype=np.int32)        
    elif method == 'bfd':
        sten = np.arange(0, -(d + p), -1, dtype=np.int32)
    elif method == 'cfd':
        smax = (d + p - 1)//2
        sten = np.arange(-smax, smax+1, dtype=np.int32)
    
    return sten

def tabulatedcoef(d, p, method):
    "Return tabulated finite difference coefs."

    if method == 'ffd' or method == 'bfd':
        if d == 1 and p == 2:            
            coef = np.array([-3/2, 2, -1/2])
        if d == 2 and p == 6:
            coef = np.array([469/90, -223/10, 879/20, -949/18, 41, -201/10, 1019/180, -7/10])
        if d == 4 and p == 4:
            coef = np.array([28/3, -111/2, 142, -1219/6, 176, -185/2, 82/3, -7/2])

    if method == 'cfd':
        if d == 1 and p == 2:
            coef = np.array([-1/2, 0, 1/2])
        if d == 2 and p == 6:
            coef = np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
        if d == 4 and p == 4:
            coef = np.array([-1/6, 2, -13/2, 28/3, -13/2, 2, -1/6])

    # For odd derivatives bfd = -ffd
    if method == 'bfd' and d % 2 != 0:            
        coef = -1 * coef 

    return coef

@pytest.mark.parametrize('d, p', [(1, 2), (2, 6), (4, 4)])
def test_findifcoef(d, p):

    for method in ['ffd', 'bfd', 'cfd']:
        sten = findifsten(d, p, method)          # stencil
        coef = findifcoef(sten, d)               # calculated
        refr = tabulatedcoef(d, p, method)       # expected
        assert np.allclose(coef, refr)

