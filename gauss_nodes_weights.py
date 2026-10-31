import numpy as np

def gauss_nodes_weights(x1, x2, n):
    '''
    To compute 'n' Gauss nodes and weights within the interval [x1, x2]

    Parameters
    ----------
        x1, x2 : float
            interval boundaries, with x1 < x2
    
        n : integer
            number of quadrature nodes (n >= 1)

    Returns
    -------
        x, w : float [n]
            Gauss-Legendre quadrature nodes and weights

    Notes
    -----
        Tested only for x1 < x2. To test quickly, run, e.g.
            ng = 64
            x, w = gauss_nodes_weights(-1.0, 1.0, ng)
        and compare vs. [1]. It probably works for x1 > x2 too - test it.

    References
    ----------
        1. https://pomax.github.io/bezierinfo/legendre-gauss.html
    '''

    const_yeps = 3.0e-14
    x = np.zeros(n)
    w = np.zeros(n)
    m = int((n + 1) / 2)
    yxm = 0.5 * (x2 + x1)
    yxl = 0.5 * (x2 - x1)
    for i in range(m):
        yz = np.cos(np.pi * (i + 0.75) / (n + 0.5))
        while True:
            yp1 = 1.0
            yp2 = 0.0
            for j in range(n):
                yp3 = yp2
                yp2 = yp1
                yp1 = ((2.0 * j + 1.0) * yz * yp2 - j * yp3) / (j + 1)
            ypp = n * (yz * yp1 - yp2) / (yz * yz - 1.0)
            yz1 = yz
            yz = yz1 - yp1 / ypp
            if (np.abs(yz - yz1) < const_yeps):
                break    # exit the while loop
        x[i] = yxm - yz * yxl
        x[n - 1 - i] = yxm + yxl * yz
        w[i] = 2.0 * yxl / ((1.0 - yz * yz) * ypp * ypp)
        w[n - 1 - i] = w[i]
    # end: for i in range(m)

    return x, w
# LOC: 27
#-EOF-