'''Testing finite difference operator
    d      : d
    p      : puracy
    method : forward (ffd), backward (bfd), center (cfd)    
'''    

import pytest
import numpy as np
from ..diffoper import genderoper


def test_genderoper_3D_array():
    "Finite difference derivatives for a 3D array."
    # 3D mesh
    x, y, z = [np.linspace(-np.pi, np.pi, num=150, endpoint=True)] * 3
    h0, h1, h2 = [c[1] - c[0] for c in (x, y, z)]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Analytical test function
    ff = np.exp(-X**2 - Y**2 - Z**2)
    d3fdxdydz = (-2*X) * (-2*Y) * (-2*Z) * ff
    d3fdx2dy = (4*Y - 8*X**2*Y) * ff 

    # ddx
    axis, d, p = 0, 1, 8
    ddx = genderoper(d, p, h0, axis)
    # d2dx2
    axis, d, p = 0, 2, 8
    d2dx2 = genderoper(d, p, h0, axis)
    # ddy
    axis, d, p = 1, 1, 8    
    ddy = genderoper(d, p, h1, axis)
    # ddz
    axis, d, p = 2, 1, 8    
    ddz = genderoper(d, p, h2, axis)

    # d3fdxdydz
    numder = ddz(ddy(ddx(ff)))
    assert np.allclose(d3fdxdydz, numder, rtol=1e-4)  
    
    # d3fdx2dy
    numder = ddy(d2dx2(ff))
    assert np.allclose(d3fdx2dy, numder, rtol=1e-4)  

    # d3fdx2dy using ddx(ddx)
    numder = ddy(ddx(ddx(ff))) 
    assert np.allclose(d3fdx2dy, numder, rtol=1e-4)