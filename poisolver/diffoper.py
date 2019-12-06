import numpy as np
from numpy.linalg import inv


def findifcoef(sten, d):
    """
    Finite difference coeficients for a given stencil and derivative order. It solves A*x = b to find 
    coefficient where A and b are generated from provided stencil and order.

    Parameters
    ----------
    sten : list
        Finite difference stencil.
    d : integer
        Derivatrive order

    Returns
    -------
    fdcoeff : float
        finite difference coefficients.

    Examples
    --------
    2nd order center finite difference coefficients for [0, 1, 2]:

    >>> finDifCoef([0, 1, 2], 2)
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
    coef = np.linalg.solve(A, b)

    return coef


def fd(fun, spacing):
    """
    Short discription.

    Parameters
    ----------
    para1 : float
        description.
    para2, para3 : float
        description.

    Returns
    -------
    para : float
        description.

    Examples
    --------
    description.

    >>> fd(para1, para2, para3)
    0.5605584137424605
    """
    # print("I take first order derivative")
    df = np.zeros(np.shape(fun))

    # forward fd for [0:n-1]
    tmp = fun[1] - fun[0]
    df[0] = tmp/spacing

    # backward for [n]
    tmp = fun[1:] - fun[0:-1]  
    df[1:] = tmp/spacing  
    
    return df
