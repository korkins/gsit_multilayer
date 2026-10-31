import numpy as np

def single_scattering_down(mu0, mu, tau, ssa_phase, naz):
    '''
    Computes azimuthal distribution of intensity at bottom of a homogeneous
    layer in the single scattering approximation.

    Parameters
    ----------
        mu0 : float
            cos(solar zenith angle) > 0

        mu : float
            cos(downward zenith angle) > 0

        tau : float
            layer optical thickness

        ssa_phase : float [naz]
            ssa * phase_function(mu0, mu, relative azimuth)
                
        naz : integer 
            number of relative azimuths

    Returns
    -------
        I1dn : float [naz]
            azimuth distribution of intensity at the layer bottom: Ibot = f(mu0, mu, raz)

    Note
    ----
        1) relative azimuth angles come externally via ssa_phase
        2) forward (back) scattering corresponds to azimuth 0o (180o)
        3) TOA scaling factor: Fo = 1.0
    '''

    tiny = 1.0e-8
  
    e0 = np.exp(-tau / mu0)

    if abs(mu - mu0) < tiny:
        I1dn = ssa_phase * tau * e0 / mu0
    else:
        e = np.exp(-tau / mu)
        I1dn = ssa_phase * mu0 / (mu0 - mu) * (e0 - e)

    return I1dn / (4.0 * np.pi)
# LOC: 10
#-EOF-