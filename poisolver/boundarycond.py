import numpy as np


def getboundary(ngrid):
    """
    Find index of interior and boundary grid points. 
    
    Parameters
    ----------
    ngrid : tuple or numpy array
        Size of grid
        
    Returns
    -------
    indint : numpy or sparse scipy matrix
        Interior grid points.
    indbnd : 
        Boundary grid points.
        
    Examples
    --------
    >>> ngrid = (3, 3, 3)
    indint, indbnd = getboundary(ngrid)s
    """ 

    ndim = len(ngrid)      # number of grid dimensions
    npts = np.prod(ngrid)  # total number of grid points
    igrd = np.arange(npts, dtype=np.int32).reshape(ngrid)   # index of grid points (row major)

    # index of interior points
    indint = np.copy(igrd)
    for axis in range(ndim):
        indint = np.swapaxes(indint, axis, 0)
        indint = indint[1:-1]
        indint = np.swapaxes(indint, 0, axis)
    indint = indint.flatten() 

    # index of boundary points
    indbnd = igrd[np.isin(igrd, indint, invert=True)].flatten()     

    return indint, indbnd

def applybc(fdmat, ngrid, bcvec=None, type='d'):
    """
    Applying boundary condition to matrix representation of finite difference
    derivative (and to rhs of Ax=b if bcvec is present).

    Parameters
    ----------
    fdmat : numpy or sparse scipy matrix
        Finite difference matrix.
    ngrid : tuple or numpy array
        Size of the grid.
    bcvec : 
        Vector containing know boundary values in Ax=b.        
    type  : string
        Boundary condition's type,
        'd' for Dirichlet
        
    Returns
    -------
    fdmat : numpy or sparse scipy matrix
        Finite difference matrix holding the boundary condition.
    bcval : 
        Vector to be substracted from rhs in Ax = b
        
    Examples
    --------
    >>> 
    """

    # index of interior and boundary points
    indint, indbnd = getboundary(ngrid)
    indint = indint.reshape((-1, 1))    # col vector
    indbnd = indbnd.reshape((1, -1))    # row vector   

    # to be substracted from rhs in Ax = b
    if bcvec is not None:
        if type == 'd':
            # bcvec  = fdmat[indint, indbnd] @ bcvec
            bcvec  = fdmat[indint, indbnd].dot(bcvec)

    # apply bc to findif matrix        
    if type == 'd':
        fdmat = fdmat[indint, indint.reshape((1, -1))]

    if bcvec is not None:
        return fdmat, bcvec  
    else:
        return fdmat


