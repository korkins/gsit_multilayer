import numpy as np
from schmidt_polynomial import schmidt_polynomial
from legendre_polynomial import legendre_polynomial
from single_scattering_up_m import single_scattering_up_m
from single_scattering_down_m import single_scattering_down_m

def gauss_seidel_iterations_m(m, nit, mu0, mug, wg, ng1, ng2,
                              dtau, Texp, mask, nlr, Tsun, nlv,
                              ssa, xk, nlro, nk,
                              srfa):
    '''
    Computes m-th Fourier moment of intensity at the Gaussian nodes and all levels
    in atmosphere over a Lambertian surface using the method of Gauss-Seidel iterations.

    Parameters
    ----------
        m : integer
            zero-based Fourier moment: m = 0, 1, ... nm-1

        nit : integer
            number of iterations

        mu0 : float
            cos(solar zenith angle) > 0

        mug, wg : float [ng2]
            all Gaussian nodes and weights; upward directions: mug[0:ng1] < 0

        ng1, ng2 : integer
            number of Gaussian nodes in hemisphere, ng1, and total, ng2 = 2 * ng1

        dtau : float [nlr]
            optical thickness of layer elements

        Texp : float [nlr, ng1]
            direct transmittance at the Gaussian nodes, T = exp(-dtau[ilr] / mug+)

        mask : integer [nlr]
            relates layer element with respective optical layer: ilro = mask[ilr]

        nlr : integer
            number of layer elements

        Tsun : float [nlv]
            direct solar beam attenuation at all levels, exp(-tau_level / mu0)

        nlv : integer
            number of levels

        ssa : f[nlro]
            single scattering albedo in optical layers

        xk : f[nlro, nk]
            phase function expansion momemnts in optical layeres; (2k+1) included

        nlro : integer
            number of optical layers; nlro <= nlr

        nk : integer
            number of expansion momements used to compute Im

        srfa : float
            Lambertian surface albedo

    Returns
    -------
        Iup, Idn : float [nlv, ng1]
            m-th Fourier moment of intensity at the Gaussian nodes and all levels

    Note
    ----
        1. scale by 1/4pi externally for TOA irradiance (flux) Fo = 1
        2. nit = 0: single scattering will be returned
        3. Idn[0, ...] = 0 - TOA boundary condition
        4. Iup[nlv-1, ...] = 0 if srfa = 0 - BOA boundary condition
        5. nodes are symmetric: mug[ig] = -mug[ig+ng1] for ig = 0: ng1
        6. ToDo: use only mup = mug > 0 on input; use symmetry of the phase function
        7. The code uses -Expm1 = (1 - T) -- note minus; the sign is flipped below
        8. wpij = np.zeros((nlr, ng2, ng2)) -- potential RAM problem:
           1024_layers * 128_ng2 * 128_ng2 * 8byte / 1024_kb / 1024_mb = 128 MB

    Reference
    ---------
        1. https://doi.org/10.1016/j.cpc.2021.108198
    '''

    srfa_min = 0.0001

    mup = mug[ng1:ng2]
    w = wg[ng1:ng2]

    pk = np.zeros((ng2, nk))         # Legendre or Schmidt polynomials
    p = np.zeros((nlr, ng2))         # m-th moment of phase function pm(mu0, mug) for single scattering
    wpij = np.zeros((nlr, ng2, ng2)) # m-th moment of phase function pm(mug_i, mug_j) for multiple scattering, weighted

    I1dn = np.zeros((nlv, ng1))
    I1up = np.zeros_like(I1dn)

    Idn = np.zeros_like(I1dn)
    Iup = np.zeros_like(I1dn)

    # Precompute Pk(x) or Qkm(x), depending on m
    if m == 0:
        pk0 = legendre_polynomial(mu0, nk - 1)
        for ig in range(ng2):
            pk[ig, :] = legendre_polynomial(mug[ig], nk - 1)

    else:
        pk0 = schmidt_polynomial(m, mu0, nk - 1)
        for ig in range(ng2):
            pk[ig, :] = schmidt_polynomial(m, mug[ig], nk - 1)
    # end: if m = 0

    # Precompute m-th moment of the phase function pm(mu0, mu) and pm(mui, muj)
    for ilro in range(nlro):
        xmom = xk[ilro, :]

        for ig in range(ng2):
            p[ilro, ig] = np.dot(xmom, pk[ig, :] * pk0)

        for ig in range(ng2):
            for jg in range(ng2):
                wpij[ilro, ig, jg] = wg[jg] * np.dot(xmom, pk[ig, :] * pk[jg, :]) # apply symmetry
    # end: for ilro

    # Downward initial guess: single scattering at all levels
    I11dn = np.zeros((nlr, ng1))
    for ilv in range(nlv - 1):
        ilr = ilv
        taui = dtau[ilr]
        Tdn = Texp[ilr, :]

        ilro = mask[ilv]
        ssa_phase_m = ssa[ilro] * p[ilro, ng1:ng2]

        I11dn[ilv, :] = single_scattering_down_m(mu0, mup, ng1, taui, ssa_phase_m) * Tsun[ilv]
        I1dn[ilv + 1, :] = I1dn[ilv, :] * Tdn + I11dn[ilv, :]
    # end: for ilv

    # Upward initial guess: single scattering at all levels (optionally with surface)
    I11up = np.zeros((nlr, ng1))
    if m == 0 and srfa > srfa_min:
        I1up[nlv - 1, :] = 4.0 * srfa * mu0 * Tsun[nlv - 1] # Fo*rho/pi = 4pi*rho/pi=4

    for ilv in range(nlv - 2, -1, -1):
        ilr = ilv
        taui = dtau[ilr]
        Tup = Texp[ilr, :]

        ilro = mask[ilv]
        ssa_phase_m = ssa[ilro] * p[ilro, 0:ng1]

        I11up[ilv, :] = single_scattering_up_m(mu0, -mup, ng1, taui, ssa_phase_m) * Tsun[ilv]
        I1up[ilv, :] = I1up[ilv + 1, :] * Tup + I11up[ilv, :]
    # end: for ilv

    # Multiple scattering iterations:
    Iup = I1up.copy()
    Idn = I1dn.copy()

    for itr in range(nit):
        # downward
        for ilv in range(nlv - 1):
            ilr = ilv
            ilro = mask[ilv]

            T = wpij[ilro, 0:ng1, 0:ng1]
            R = wpij[ilro, 0:ng1, ng1:ng2]

            Iup05 = 0.5 * (Iup[ilv, :] + Iup[ilv + 1, :])
            Idn05 = 0.5 * (Idn[ilv, :] + Idn[ilv + 1, :])

            J = 0.5 * ssa[ilro] * (np.dot(R, Iup05) + np.dot(T, Idn05))

            Tdn =  Texp[ilr, :]
            Idn[ilv + 1, :] = I11dn[ilv, :] + Idn[ilv, :] * Tdn + J * (1.0 - Tdn)
        # end: for ilv down

        # upward
        if m == 0 and srfa > srfa_min:
            Iup[nlv - 1, :] = ( 2.0 * srfa * np.dot(Idn[nlv - 1, :], mup * w) +
                                4.0 * srfa * mu0 * Tsun[nlv - 1] ) # Fo*rho/pi = 4pi*rho/pi=4

        for ilv in range(nlv - 2, -1, -1):
            ilr = ilv
            ilro = mask[ilr]

            T = wpij[ilro, 0:ng1, 0:ng1]
            R = wpij[ilro, 0:ng1, ng1:ng2]

            Iup05 = 0.5 * (Iup[ilv, :] + Iup[ilv + 1, :])
            Idn05 = 0.5 * (Idn[ilv, :] + Idn[ilv + 1, :])

            J = 0.5 * ssa[ilro] * (np.dot(T, Iup05) + np.dot(R, Idn05))

            Tup =  Texp[ilr, :]
            Iup[ilv, :] = I11up[ilv, :] + Iup[ilv + 1, :] * Tup + J * (1.0 - Tup)
        # end: for ilv up
    # end: for itr

    return Iup, Idn
# LOC: 77
#-EOF-