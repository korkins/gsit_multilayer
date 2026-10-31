import numpy as np
from legendre_polynomial import legendre_polynomial
from single_scattering_up import single_scattering_up
from single_scattering_down import single_scattering_down

def single_scattering(tau, ssa, nlr, xk, nxk, mu0, muup, nup, mudn, ndn, caz, naz, srfa):
    '''
    Calculates single scattering at all levels in atmosphere.

    Parameters
    ----------
        tau, ssa : float [nlr]
              layer optical thickness and single scattering albedo

        nlr : integer
              number of optical layers

        xk : float [nlr, nxk]
             layer phase function moments (on fast dimension); (2k+1) included

        mu0 : float
              cos(solar zenith angle) > 0

        muup : float [nup]
              cos(upward zenith angle) < 0

        nup : integer
              number of upward directions

        mudn : float [ndn]
               cos(downward zenith angle) > 0

        ndn : integer
              number of downward directions

        caz : float [naz]
              cos(relative azimuth angle); forward (back) scattering at caz = 1 (-1)

        naz : integer
              number of relative azimuths

        srfa : float
               Lambertian surface albedo

    Returns
    -------
        I1up : float [nlr+1, nup, naz]
               upward single scattering intensity

        I1dn : float [nlr+1, ndn, naz]
               downward single scattering intensity

    Notes
    -----
        1. TOA irradiance (flux) Fo = 1
        2. I1dn[0, ...] = 0 - TOA boundary condition
        3. I1up[nlv-1, ...] = 0 if srfa = 0 - BOA boundary condition
    '''

    pi = np.pi
    ssa_phase = np.zeros(naz)

    nlv = nlr + 1 # number of levels (boundaries)
    Tsun = np.zeros(nlv)

    taui = 0.0
    Tsun[0] = 1.0
    for ilv in range(1, nlv):
        ilr = ilv - 1
        taui = taui + tau[ilr]
        Tsun[ilv] = np.exp(-taui / mu0)
    # end: for ilv

    smu0 = np.sqrt(1.0 - mu0 * mu0)

    # Legendre polynomials for p(x) = sum{xk * pk(x)}
    pkup = np.zeros((nup, naz, nxk))
    pkdn = np.zeros((ndn, naz, nxk))

    for imu in range(nup):
        mu = muup[imu]
        smu = np.sqrt(1.0 - mu * mu)
        for iaz in range(naz):
            nu = mu * mu0 + smu0 * smu * caz[iaz]
            pkup[imu, iaz, :] = legendre_polynomial(nu, nxk - 1)

    for imu in range(ndn):
        mu = mudn[imu]
        smu = np.sqrt(1.0 - mu * mu)
        for iaz in range(naz):
            nu = mu * mu0 + smu0 * smu * caz[iaz]
            pkdn[imu, iaz, :] = legendre_polynomial(nu, nxk - 1)

    # downward
    I1dn = np.zeros((nlv, ndn, naz))
    for ilv in range(nlv - 1):
        ilr = ilv
        e0 = Tsun[ilr]
        taui = tau[ilr]
        Tdn = np.exp(-taui / mudn)
        xki = xk[ilr, :]
        ssai = ssa[ilr]

        for imu in range(ndn):
            mu = mudn[imu]

            for iaz in range(naz):
                ssa_phase[iaz] = ssai * np.dot(xki, pkdn[imu, iaz, :])

            I11dn = single_scattering_down(mu0, mu, taui, ssa_phase, naz)
            I1dn[ilv + 1, imu, :] = I1dn[ilv, imu, :] * Tdn[imu] + I11dn * e0

    # upward
    I1up = np.zeros((nlv, nup, naz))
    if srfa > 0.0:
        I1up[nlv - 1, :, :] = (srfa / pi) * mu0 * Tsun[nlv - 1] # Fo_toa = 1: Fo*rho/pi

    for ilv in range(nlv - 2, -1, -1):
        ilr = ilv
        e0 = Tsun[ilr]
        taui = tau[ilr]
        Tup = np.exp(taui / muup)
        xki = xk[ilr, :]
        ssai = ssa[ilr]

        for imu in range(nup):
            mu = muup[imu]

            for iaz in range(naz):
                ssa_phase[iaz] = ssai * np.dot(xki, pkup[imu, iaz, :])

            I11up = single_scattering_up(mu0, mu, taui, ssa_phase, naz)
            I1up[ilv, imu, :] = I1up[ilv + 1, imu, :] * Tup[imu] + I11up * e0

    return I1up, I1dn
# LOC: 58
#-EOF-