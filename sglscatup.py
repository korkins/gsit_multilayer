#import time
import numpy as np

def sglscatup(mu, mu0, dtau, ssa_phf, naz):
    '''
    Task:
        To compute single scattering at top of a homogeneous layer.
    In:
        mu       f       cos(vza) < 0 (up)
        mu0      f       cos(sza) > 0
        dtau     f       layer thickness
        ssa_phf  f[naz]  ssa*phase_function(mu, mu0, raz)
        naz      i       number of azimuths
    Out:
        I1up     f       Itop = f(mu, mu0, azr)
    Tree:
        -
    Note:
        TOA scaling factor: Fo = 1.0
    Refs:
        1. -
    '''

    I1up = ssa_phf * mu0 / (mu0 - mu) * (1.0 - np.exp(dtau / mu - dtau / mu0))

    return I1up / (4.0 * np.pi)
#==============================================================================
# #
# if __name__ == "__main__":
# #
#     tau0 = 1.0/3.0
#     xk = np.array([1.0, 0.0, 0.5])
#     mu0 = np.linspace(0.1, 1.0, 91)
#     mu = -mu0
#     azr = np.linspace(0.0, np.pi, 1801)
#     nmu0 = len(mu0)
#     nmu = len(mu)
#     naz = len(azr)
#     I1up = np.zeros((nmu0, nmu, naz))
#     time_start = time.time()
#     for imu0 in range(len(mu0)):
#         for imu in range(len(mu)):
#             I1up[imu0, imu, :] = sglscatup(mu[imu], mu0[imu0], azr, tau0, xk)
#     time_end = time.time()
# #
#     print("sglsup = %.3f sec."%(time_end-time_start))
#==============================================================================