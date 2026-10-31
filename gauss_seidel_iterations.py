import numpy as np
from split_optical_tau import split_optical_tau
from single_scattering import single_scattering
from gauss_nodes_weights import gauss_nodes_weights
from gauss_seidel_iterations_m import gauss_seidel_iterations_m
from source_function_integrate_up import source_function_integrate_up
from source_function_integrate_down import source_function_integrate_down

def gauss_seidel_iterations(ng1, nm, nit, dtau_max,               # accuracy parameters
                            mu0, muup, nup, mudn, ndn, azr, naz,  # solar-view geometry
                            tau, ssa, xk, nlro, nxk,              # atmosphere
                            srfa):                                # surface
    '''
    Solves scalar RTE at all levels of a vertically inhomogeneous atmosphere
    over a Lambertian surface using the method of Gauss-Seidel iterations.

    Parameters
    ----------
        ng1 : integer
             number of Gaussian nodes per hemisphere

        nm : integer
             total number of Fourier moments: m = 0, 1, ... nm-1

        nit : integer
            number of iterations

        dtau_max : float
            maximum step for integration over the optical thickness 'tau'; ~ 0.01

        mu0 : float
             cos(solar zenith angle) > 0

        muup : float [nup]
            cos(upward zenith angles) < 0

        nup : integer
            number of upward zenith angles

        mudn : float [ndn]
            cos(downward zenith angles) > 0

        ndn : integer
             number of downward zenith angles

        azr : float [naz]
            relative azimuth angles, in radians: azr = 0 (pi) - forward (back) scattering

        naz : integer
             number of relative azimuth angles

        tau, ssa : float [nlro]
            layer optical thickness and single scattering albedo (from TOA to BOA)

        xk : float [nlro, nxk]
            layer phase function expansion moments (on fast dimension); (2k+1) included

        nlro : integer
             number of optical layers

        nxk : integer
             total number of the phase function expansion moments

        srfa : float
             Lambertian surface albedo

    Returns
    -------
        Iup : float [nlr+1, nup, naz]
            total scattering solution at upward directions and all levels

        Idn : float [nlr+1, ndn, naz]
            total scattering solution at downward directions and all levels

    Note
    ----
        1. TOA irradiance (flux) Fo = 1
        2. I1dn[0, ...] = 0 - TOA boundary condition
        3. I1up[nlr, ...] = 0 if srfa = 0 - BOA boundary condition
        4. nm <= ng2 = 2 * ng1
        5. same nxk is assumed in all optical layers

    Reference
    ---------
        1. https://doi.org/10.1016/j.cpc.2021.108198
    '''

    srfa_min = 0.0001

    ng2 = ng1 * 2
    nk = min(nxk, ng2)

    nlvo = nlro + 1
    Iup = np.zeros((nlvo, nup, naz))
    Idn = np.zeros((nlvo, ndn, naz))

    mup, w = gauss_nodes_weights(0.0, 1.0, ng1)
    mug = np.concatenate((-mup, mup))
    wg = np.concatenate((w, w))

    nlr, dtau, mask, ndtau = split_optical_tau(dtau_max, tau, nlro)
    nlv = nlr + 1

    Tsun = np.ones(nlv)
    Texp = np.ones((nlr, ng1))
    tau0 = 0.0
    for ilr in range(nlr):
        tau0 = tau0 + dtau[ilr]
        Tsun[ilr + 1] = np.exp(-tau0 / mu0)
        Texp[ilr,  :] = np.exp(-dtau[ilr] / mup)

    # Source function integration and Fourier summation: multiple (2+) scattering
    Igboa = np.zeros(ng1)
    Img05 = np.zeros((nlr, ng2))
    I2up = np.zeros((nlv, nup, naz))
    I2dn = np.zeros((nlv, ndn, naz))

    deltm0 = 1.0
    for m in range(nm):
        kron_cmaz = deltm0 * np.cos(m * azr)

        # Get all scattering orders at the Gaussian nodes
        Imup, Imdn = gauss_seidel_iterations_m(m, nit, mu0, mug, wg, ng1, ng2,
                                               dtau, Texp, mask, nlr, Tsun, nlv,
                                               ssa, xk[:, 0:nk], nlro, nk,
                                               srfa)

        # Midlayer solution for source function integration (SFI)
        for ilv in range(nlv - 1):
            Img05[ilv, 0:ng1] = 0.5 * (Imup[ilv, :] + Imup[ilv + 1, :])   # Iup05
            Img05[ilv, ng1:ng2] = 0.5 * (Imdn[ilv, :] + Imdn[ilv + 1, :]) # Idn05

        # SFI: downward
        for imu in range(ndn):
            mu = mudn[imu]
            I2mdn = source_function_integrate_down(m, mu, Img05, mug, wg,
                                                   ng2, dtau, mask, nlr, nlv,
                                                   ssa, xk[:, 0:nk], nlro, nk)
            
            for ilv in range(nlv):
                I2dn[ilv, imu, :] += I2mdn[ilv] * kron_cmaz # inefficient use of indices - see below

        # SFI: upward
        if srfa > srfa_min:
            Igboa = Imdn[nlv - 1, :]

        for imu in range(nup):
            mu = muup[imu]
            I2mup = source_function_integrate_up(m, mu, Img05, mug, wg,
                                                 ng2, dtau, mask, nlr, nlv,
                                                 ssa, xk[:, 0:nk], nlro, nk,
                                                 Igboa, srfa)
            for ilv in range(nlv):
                I2up[ilv, imu, :] += I2mup[ilv] * kron_cmaz # inefficient use of indices - see below

        deltm0 = 2.0
        print(f"(in gsit) m = {m}") # slightly decreases performance, but indicates the code is alive

    # Exact single scattering solution: I1[nlv, nmu, naz] - that is why I2 is shaped this way
    caz = np.cos(azr)
    I1up, I1dn = single_scattering(tau, ssa, nlro, xk, nxk, mu0, muup, nup,
                                   mudn, ndn, caz, naz, srfa)

    pi4 = np.pi * 4.0 # for Fo_toa = 1
    for ilvo in range(nlvo):
        ilv = np.sum(ndtau[0:ilvo])
        Iup[ilvo, :, :] = I1up[ilvo, :, :] + I2up[ilv, :, :] / pi4
        Idn[ilvo, :, :] = I1dn[ilvo, :, :] + I2dn[ilv, :, :] / pi4
    # end: for ilvo

    return Iup, Idn
# LOC: 56
#-EOF-