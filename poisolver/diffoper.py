import numpy as np


def findifcoef(sten, d):
    """
    Finite difference coeficients for a given stencil and derivative order. It solves A*x = b to find 
    coefficient where A and b are generated from provided stencil and order.

    Parameters
    ----------
    sten : list
        Finite difference stencil.
    d : integer
        Derivatrive order.

    Returns
    -------
    fdcoeff : float
        Finite difference coefficients.

    Examples
    --------
    2nd order center finite difference coefficients for [0, 1, 2]:

    >>> findifcoef([0, 1, 2], 2)
        [ 1. -2.  1.]
    """

    assert d < len(sten), "Order of derivative should be smaller then stencil's length!"  

    # size of the stecil
    n = np.arange(len(sten))

    # left hand side matrix
    A = sten**n.reshape((n.shape[0], 1))   

    # right hand side matrix
    b = np.zeros(n.shape)
    b[d] = 1

    # solve A*x = b
    return np.linalg.solve(A, b)     


def findifder(fun, d, p, h):
    """
    Finite difference derivative for a given fucntion. It uses centeral fd for interior points 
    and forward and backward for points at the boundary, assuming a uniform grid.

    Parameters
    ----------
    fun : array
        Input function.
    d, p : integer
        Derivatrive order and accuracy respectively.
    h : float
        Grid spacing.

    Returns
    -------
    derfun : array
        dth order derivative of input function.
    """    

    # Initialize 
    fact = np.arange(1, d+1).prod()/h**d    # d!/h**d 
    derfun = np.zeros_like(fun)
    smax = (d+p-1)//2

    # [smax:-smax] points centeral findif with [-(d+p1)/2, ..., (d+p-1)/2] stencil
    sten = range(-smax, smax+1)
    coef = findifcoef(sten, d)
    for ss, cc in zip(sten, coef): 
        if ss == smax:    
            derfun[smax:-smax] += cc*fun[smax+ss:]
        else:
            derfun[smax:-smax] += cc*fun[smax+ss:-smax+ss]

    # [0:smax] points forward findif with [0, 1, ..., (d+p-1)] stencil
    sten = range(d+p)
    coef = findifcoef(sten, d)
    for ss, cc in zip(sten, coef): 
        derfun[:smax] += cc*fun[ss:ss+smax]

    # [-smax:] points backward findif with [0, -1, ..., -(d+p-1)] stencil
    sten = range(0, -(d+p), -1)
    coef = findifcoef(sten, d)
    for ss, cc in zip(sten, coef): 
        if ss == 0:    
            derfun[-smax:] += cc*fun[-smax:]
        else:
            derfun[-smax:] += cc*fun[-smax+ss:ss]

    return fact*derfun  


def genderfun(d, p, h):
    """
    Generate function helper for dth order derivative of oth order accuracy assuming a 
    uniform grid with spacing h. Once generated it can be applied to any defined function.

    Parameters
    ----------
    d, p : integer
        Derivatrive order and accuracy respectively.
    h : float
        Grid spacing.

    Returns
    -------
    derfun : function helper
        dth order derivative function helper.
    """    

    return lambda fun: findifder(fun, d, p, h)
    