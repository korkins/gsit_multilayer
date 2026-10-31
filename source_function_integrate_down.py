import numpy as np
from legendre_polynomial import legendre_polynomial
from schmidt_polynomial import schmidt_polynomial

def source_function_integrate_down(m, mu, Ig05, mug, wg, ng2, dtau, mask, 
                                   nlr, nlv, ssa, xk, nlro, nk):
    '''
    Computes m-th Fourier moment of 2+ scattering intensity, I2m, at a
    user-defined downward direction mu > 0 and at all atmospheric levels.

    Parameters
    ----------
        m : integer
            zero-offset Fourier moment: m = 0, 1, ... nm-1

        mu : float
            cos(downward view zenith angle) > 0

        Ig05 : float [nlr, ng2]
            m-th Fourier moment for total scattering intensity at Gaussian
            nodes, averaged at the middle of a layer ilr (hence 05)

        mug, wg : float [ng2]
            Gaussian nodes, mug, and weights, wg, in the entire sphere;
            upward directions, mug[0:ng1] < 0, come first; ng1 = ng2 / 2

        ng2 : integer
            total number of Gaussian nodes (order of Gaussian quadrature)

        dtau : float [nlr]
            optical thickness of layer elements

        mask : integer [nlr]
            optical layer index: ilro = mask[ilr]

        nlr, nlv : integer
            number of layer elements and their levels (boundaries); nlv = nlr + 1

        ssa : float [nlro]
            single scattering albedo in optical layers

        xk : float [nlro, nk]
            phase function expansion moments in optical layers; (2k+1) included

        nlro : integer
            number of optical layers; nlro <= nlr

        nk : integer
            number of expansion moments used to compute I2m

    Returns
    -------
        I2dn : float [nlv]
            multiple (2+) scattering solution at all levels downward

    Notes
    -----
        1. Single scattering and direct sunbeam bouncing are dropped here
        2. Not yet scaled by 1/4pi for TOA irradiance (flux) Fo = 1
        3. TOA boundary condition I2dn[0] = 0

    References
    ----------
        1. https://doi.org/10.1016/j.cpc.2021.108198
    '''

    I2dn = np.zeros(nlv)

    pk = np.zeros((ng2, nk))     # Legendre or Schmidt polynomials
    wpij = np.zeros((nlro, ng2)) # weighted m-th moment of phase function pm(mu, mug)

    # Precompute Legendre Pk(x) or Schmidt Qkm(x) polynomials, depending on m
    if m == 0:
        pku = legendre_polynomial(mu, nk - 1) # mu > 0
        for ig in range(ng2):
            pk[ig, :] = legendre_polynomial(mug[ig], nk - 1)
    else:
        pku = schmidt_polynomial(m, mu,  nk - 1)
        for ig in range(ng2):
            pk[ig, :] = schmidt_polynomial(m, mug[ig], nk - 1)
    # end: if m = 0

    # Precompute the weighted m-th Fourier moment of the phase function for
    # the user direction mu, all Gaussian directions mug, and in all optical layers
    for ilro in range(nlro):
        xmom = xk[ilro, :]
        for ig in range(ng2):
            wpij[ilro, ig] = wg[ig] * np.dot(xmom, pku * pk[ig, :])
    # end: for ilro
    
    for ilv in range(nlv - 1):
        ilr = ilv                   # layer element index (for readability)
        ilro = mask[ilr]

        J = 0.5 * ssa[ilro] * np.dot(wpij[ilro, :], Ig05[ilr])
        
        Tdn = np.exp(-dtau[ilr] / mu)
        I2dn[ilv + 1] = I2dn[ilv] * Tdn + J * (1.0 - Tdn)
    # end: for ilv down

    return I2dn
# LOC: 26
#-EOF-