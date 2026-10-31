import numpy as np

def legendre_polynomial(x, kmax):
    '''
    Computes Legendre polynomials Pk(x) for all orders k=0:kmax and a single
    point 'x' within [-1 : +1]
        
    Parameters
    ----------
        x : float
            abscissa: -1 <= x <= +1
            
        kmax : integer
            zero-based maximum order, k = 0, 1, 2, ... kmax
        
    Returns
    -------
        pk : float [kmax+1]
            Legendre polynomials Pk(x) at a given 'x' and all orders 'k'.
        
    Notes
    -----
        The Bonnet recursion formula [1, 2]:
        
        (k+1)P{k+1}(x) = (2k+1)*P{k}(x) - k*P{k-1}(x),
        
        where k = 1:K, P{0}(x) = 1.0, P{1}(x) = x

    References
    ----------
        1. https://en.wikipedia.org/wiki/Legendre_polynomials
        2. http://mathworld.wolfram.com/LegendrePolynomial.html
    '''

    nk = kmax + 1
    pk = np.zeros(nk)
    pk[0] = 1.0
    
    if kmax > 0:
        pk[1] = x
        for ik in range(2, nk):
            pk[ik] = (2.0 - 1.0 / ik) * x * pk[ik - 1] - (1.0 - 1.0 / ik) * pk[ik - 2]
    # end: if kmax > 0
    
    return pk
# LOC: 10
#-EOF-