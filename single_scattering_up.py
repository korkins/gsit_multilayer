import numpy as np

def single_scattering_up(mu0, mu, tau, ssa_phase, naz):
    '''
    Computes azimuthal distribution of intensity at top of a homogeneous
    layer in the single scattering approximation.
    
    Parameters
    ----------
        mu0 : float
            cos(solar zenith angle) > 0

        mu : float
            cos(upward zenith angle) < 0

        tau : float
            layer optical thickness

        ssa_phase : float [naz]
            ssa * phase_function(mu0, mu, relative azimuth)
            
        naz : integer
            number of relative azimuths

    Returns
    -------
        I1up : float [naz]
           azimuthal distribution of intensity at the layer top: Itop = f(mu0, mu, raz)

    Note
    ----
        1) Relative azimuth angles come externally via ssa_phase
        2) Forward (back) scattering corresponds to azimuth 0o (180o)
        3) TOA scaling factor: Fo = 1.0
    '''
    
    I1up = ssa_phase * mu0 / (mu0 - mu) * (1.0 - np.exp(tau / mu - tau / mu0))

    return I1up / (4.0 * np.pi)
# LOC: 4
#-EOF-