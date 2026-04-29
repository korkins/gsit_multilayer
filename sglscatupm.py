#import time
import numpy as np

def sglscatupm(mu0, mu, nmu, dtau, ssapm):
    '''
    Task:
        To compute m-th Fourier moment in single scattering at top of a homogeneous layer.
    In:
        mu0     f        cos(sza) > 0
        mu      f[nmu]   cos(vza) < 0 (up)
        nmu     i        len(mu)
        dtau    f        layer thickness
        ssapm   f[nmu]   ssa*phase_function_m(mu0, mu) -- note moment 'm' here
    Out:
        I1up    f[nmu]   Itop = f(mu0, mu, m)
    Note:
        1) The moment 'm' comes from outside via ssapm
        2) Fo = 4pi - scaling for Fo = 1 happens later, in gsit(...)
        3) This function is usually used at Gaussian nodes, so mu = mug and nmu = ng1
        4) for very small dtau, (dtau / mu - dtau / mu0 ) = tiny - tiny. Solution:
           x = dtau / mu - dtau / mu0
           I1up = ssapm * (mu0 / (mu0 - mu)) * ( -np.expm1(x) ) -- note expm1 here.
    Refs:
        1. -
    '''

    I1up = ssapm * mu0 / (mu0 - mu) * (1.0 - np.exp(dtau / mu - dtau / mu0))

    return I1up
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