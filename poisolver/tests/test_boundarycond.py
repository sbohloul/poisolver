'''Testing applying boundary condition to finite difference matrix 
'''    

import pytest
import numpy as np
import scipy.sparse as sps
from ..boundarycond import getboundary, applybc


def test_getboundary():

    # reference 
    refint = np.array([13])
    refbnd = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, \
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])

    ngrid = (3, 3, 3)
    indint, indbnd = getboundary(ngrid)
        
    assert np.allclose(refint, indint)
    assert np.allclose(refbnd, indbnd)
