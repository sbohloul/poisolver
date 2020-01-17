import numpy as np
import scipy.sparse as sps


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
    # b[d] = 1
    b[d] = np.arange(1, d+1).prod()

    # solve A*x = b
    coef = np.linalg.solve(A, b)     
    # coef = np.linalg.inv(A) @ b     
    coef[np.abs(coef) < 1e-10] = 0  
    
    return coef


def findifder(fun, d, p, h, axis):
    """
    Finite difference derivative for a given n-dimensional array. It uses centeral fd for interior points 
    and forward and backward for points at the boundary, assuming a uniform grid.

    Parameters
    ----------
    fun : array
        Input n-dimensional function.
    d, p : integer
        Derivatrive order and accuracy respectively.
    h : float
        Grid spacing.
    axis : integer
        Derivative is taken along this axis.

    Returns
    -------
    derfun : array
        dth order derivative of input function.
    """    

    # Initialize 
    fact = np.arange(1, d+1).prod()/h**d    # d!/h**d
    ind = fun.shape[axis]   # grid size along derivative axis
    smax = (d + p - 1)//2   # maximum index of the stencil
    derfun = np.zeros_like(fun)

    # Move derivative axis to 0 
    fun = np.swapaxes(fun, axis, 0)

    # [smax:-smax] points centeral findif with [-(d+p-1)/2, ..., (d+p-1)/2] stencil
    sten = range(-smax, smax+1)
    coef = findifcoef(sten, d)
    for s, c in zip(sten, coef):
        derfun[smax:-smax] += c * fun[smax+s:ind-smax+s]

    # [0:smax] points forward findif with [0, 1, ..., (d+p-1)] stencil
    sten = range(d + p)
    coef = findifcoef(sten, d)
    for s, c in zip(sten, coef): 
        derfun[:smax] += c * fun[s:s+smax]

    # [-smax:] points backward findif with [0, -1, ..., -(d+p-1)] stencil
    sten = range(0, -(d + p), -1)
    coef = findifcoef(sten, d)
    for s, c in zip(sten, coef): 
        derfun[-smax:] += c * fun[-smax+s:ind+s]

    # Multiply the factor and move the derivative axis to its original position 
    # return fact * np.swapaxes(derfun, 0, axis)
    return np.swapaxes(derfun, 0, axis) / h**d 


def genderoper(d, p, h, axis):
    """
    Generate function helper for dth order derivative with pth order accuracy, assuming a 
    uniform grid with spacing h. Once generated it can be applied to any n-dimensional array.
    Parameters
    ----------
    d, p : integer
        Derivatrive order and accuracy respectively.
    h : float
        Grid spacing.
    axis : integer
        Derivative is taken along this axis.        

    Returns
    -------
    derfun : function helper
        dth order derivative function helper.
    """    

    return lambda fun: findifder(fun, d, p, h, axis)


def findifmat(ngrid, d, p, h, axis):
    """
    Finite difference matrix for a given n-dimensional array. It uses centeral fd for interior points 
    and forward and backward for points at the boundary, assuming a uniform grid. It assumes a row-major 
    indexing for grid points.

    Parameters
    ----------
    ngrid : tuple
        Grid dimension.
    d, p : integer
        Derivatrive order and accuracy respectively.
    h : float
        Grid spacing.
    axis : integer
        Derivative is taken along this axis.

    Returns
    -------
    dermat : array
        Matrix for dth order derivative.
    """        

    # Initialize 
    fact  = np.arange(1, d+1).prod()/h**d    # d!/h**d
    smax  = (d + p - 1)//2                   # maximum index of the stencil
    nmat  = np.prod(ngrid)                   # total number of grid points
    npts  = ngrid[axis]                      # size of grid along derivative axis
    igrid = np.arange(nmat).reshape(ngrid)   # row-major indexing for grid points
    igrid = np.swapaxes(igrid, axis, 0)      # move the derivative axis to 0
    fdmat = np.zeros((nmat, nmat))

    # [smax:-smax] points centeral findif with [-(d+p-1)/2, ..., (d+p-1)/2] stencil
    sten = range(-smax, smax+1)
    coef = findifcoef(sten, d)
    row  = igrid[smax:npts-smax].flatten()
    for ss, cc in zip(sten, coef):    
        col = igrid[smax+ss:npts-smax+ss].flatten()    
        fdmat[row, col] = cc

    # [0:smax] points forward findif with [0, 1, ..., (d+p-1)] stencil
    sten = range(d + p)
    coef = findifcoef(sten, d)
    row  = igrid[0:smax].flatten()
    for ss, cc in zip(sten, coef):
        col = igrid[ss:smax+ss].flatten()
        fdmat[row, col] = cc

    # [-smax:] points backward findif with [0, -1, ..., -(d+p-1)] stencil
    sten = range(0, -(d + p), -1)
    coef = findifcoef(sten, d)
    row  = igrid[npts-smax:npts].flatten()
    for ss, cc in zip(sten, coef):
        col = igrid[npts-smax+ss:npts+ss].flatten()
        fdmat[row, col] = cc

    # return fact * fdmat
    return fdmat / h**d 


def findifmatsp(ngrid, d, p, h, axis):
    """
    Finite difference matrix for a given n-dimensional array. It uses centeral fd for interior points 
    and forward and backward for points at the boundary, assuming a uniform grid. It assumes a row-major 
    indexing for grid points.

    Parameters
    ----------
    ngrid : tuple
        Grid dimension.
    d, p : integer
        Derivatrive order and accuracy respectively.
    h : float
        Grid spacing.
    axis : integer
        Derivative is taken along this axis.

    Returns
    -------
    dermat : sparse lil_matrix
        Matrix for dth order derivative.
    """        

    # Initialize 
    fact  = np.arange(1, d+1).prod()/h**d    # d!/h**d
    smax  = (d + p - 1)//2                   # maximum index of the stencil
    nmat  = np.prod(ngrid)                   # total number of grid points
    npts  = ngrid[axis]                      # size of grid along derivative axis    
    igrid = np.arange(nmat).reshape(ngrid)   # row-major indexing for grid points
    igrid = np.swapaxes(igrid, axis, 0)      # move the derivative axis to 0
    fdmat = sps.lil_matrix((nmat, nmat), dtype=np.float64)
    
    # [smax:-smax] points centeral findif with [-(d+p-1)/2, ..., (d+p-1)/2] stencil
    sten = range(-smax, smax+1)
    coef = findifcoef(sten, d)
    row  = igrid[smax:npts-smax].flatten()
    for ss, cc in zip(sten, coef):        
        col = igrid[smax+ss:npts-smax+ss].flatten()    
        fdmat[row, col] = cc

    # [0:smax] points forward findif with [0, 1, ..., (d+p-1)] stencil
    sten = range(d + p)
    coef = findifcoef(sten, d)
    row  = igrid[0:smax].flatten()
    for ss, cc in zip(sten, coef):        
        col = igrid[ss:smax+ss].flatten()
        fdmat[row, col] = cc

    # [-smax:] points backward findif with [0, -1, ..., -(d+p-1)] stencil
    sten = range(0, -(d + p), -1)
    coef = findifcoef(sten, d)
    row  = igrid[npts-smax:npts].flatten()
    for ss, cc in zip(sten, coef):    
        col = igrid[npts-smax+ss:npts+ss].flatten()
        fdmat[row, col] = cc

    # return fact * fdmat
    return fdmat / h**d         