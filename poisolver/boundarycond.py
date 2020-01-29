import numpy as np
import scipy.sparse as sps


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

    # convert to sparse if not
    issparse = sps.issparse(fdmat)
    if issparse:
        if not sps.isspmatrix_csr(fdmat):
            fdmat = fdmat.tocsr()        
    else:
        fdmat = sps.csr_matrix(fdmat)
       
    # index of interior and boundary points
    indint, indbnd = getboundary(ngrid)  

    # apply boundary condition
    fdmat = fdmat[indint, :]    
    if type == 'd':
        if bcvec is not None:   # to be added to r.h.s of Ax = b
            bcvec = fdmat[:, indbnd].dot(bcvec)
     
        fdmat = fdmat[:, indint]       
        if not issparse:        # put back to dense format 
            fdmat = fdmat.todense()                                   

    if bcvec is not None:
        return fdmat, bcvec  
    else:
        return fdmat


