import numpy as np

def single_scattering_up_m(mu0, mu, nmu, tau, ssa_phase_m):
    '''
    Computes m-th Fourier moment of intensity at top of a homogeneous layer
    in the single scattering approximation.

    Parameters
    ----------
        mu0 : float
            cos(solar zenith angle) > 0

        mu : float [nmu]
            cos(upward zenith angle) < 0

        nmu : integer
            number of mu-s

        tau : float
            layer optical thickness

        ssa_phase_m : float [nmu]
            ssa * phase_function_m(mu0, mu) -- note moment 'm' here

    Returns
    -------
        I1up : float [nmu]
            m-th Fourier moment of intensity at the layer top: Itop = f(mu0, mu, m)

    Note
    ----
        1) The moment 'm' comes externally via ssa_phase_m
        2) Fo = 4pi - scaling for Fo = 1 happens later, in the calling program
        3) This function is usually used at the Gaussian quadrature nodes, so 
           mu = mug and nmu = ng1
    '''

    I1up = ssa_phase_m * mu0 / (mu0 - mu) * (1.0 - np.exp(tau / mu - tau / mu0))

    return I1up
# LOC: 4
#-EOF-