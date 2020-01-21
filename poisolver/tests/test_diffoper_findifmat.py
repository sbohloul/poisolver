'''Testing finite difference operator
    d      : d
    p      : puracy
    method : forward (ffd), backward (bfd), center (cfd)    
'''    

import pytest
import numpy as np
import scipy.sparse as sps
from ..diffoper import findifmatsp


def test_findifmat_3D():

    # 3D mesh
    x, y, z = [np.linspace(-np.pi, np.pi, num=60, endpoint=True)] * 3
    h0, h1, h2 = [c[1] - c[0] for c in (x, y, z)]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    ngrid = X.shape

    tol = 1e-2

    # Analytical test function
    ff = np.exp(-X**2 - Y**2 - Z**2)
    d3fdxdydz = (-2*X) * (-2*Y) * (-2*Z) * ff
    d3fdx2dy = (4*Y - 8*X**2*Y) * ff 

    # ddx
    axis, d, p = 0, 1, 8
    ddx = findifmatsp(ngrid, d, p, h0, axis)
    # ddy
    axis, d, p = 1, 1, 8    
    ddy = findifmatsp(ngrid, d, p, h0, axis)
    # ddz
    axis, d, p = 2, 1, 8    
    ddz = findifmatsp(ngrid, d, p, h0, axis)
    # d2dx2
    axis, d, p = 0, 2, 8
    d2dx2 = findifmatsp(ngrid, d, p, h0, axis) 

    # d3fdx2dy
    numder = ddy @ ff.flatten()
    numder = d2dx2 @ numder
    assert np.allclose(d3fdx2dy.flatten(), numder, rtol=tol)  

    # d3fdxdydz
    numder = ddx @ ff.flatten()
    numder = ddy @ numder
    numder = ddz @ numder
    assert np.allclose(d3fdxdydz.flatten(), numder, rtol=tol)  
