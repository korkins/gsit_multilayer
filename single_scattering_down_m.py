import numpy as np

def single_scattering_down_m(mu0, mu, nmu, tau, ssa_phase_m):
    '''
    Computes m-th Fourier moment of intensity at bottom of a homogeneous layer
    in the single scattering approximation.

    Parameters
    ----------
        mu0 : float
            cos(solar zenith angle) > 0

        mu : float [nmu]
            cos(downward zenith angle) > 0

        nmu : integer
            number of mu-s

        tau : float
            layer optical thickness

        ssa_phase_m : float [nmu]
            ssa * phase_function_m(mu0, mu) -- note moment 'm' here

    Returns
    -------
        I1dn : float [nmu]
            m-th Fourier moment of intensity at the layer bottom: Ibot = f(mu0, mu, m)

    Note
    ----
        1) The moment 'm' comes externally via ssa_phase_m
        2) Fo = 4pi - scaling for Fo = 1 happens later, in the calling program
        3) This function is usually used at the Gaussian quadrature nodes, so
           mu = mug and nmu = ng1
    '''

    tiny = 1.0e-8

    e0 = np.exp(-tau / mu0)

    I1dn = np.zeros_like(mu)

    for imu in range(nmu):
        mui = mu[imu]
        if abs(mui - mu0) < tiny:
            I1dn[imu] = ssa_phase_m[imu] * tau * e0 / mu0
        else:
            e = np.exp(-tau / mui)
            I1dn[imu] = ssa_phase_m[imu] * mu0 / (mu0 - mui) * (e0 - e)  
    # end: for imu
        
    return I1dn
# LOC: 13
#-EOF-