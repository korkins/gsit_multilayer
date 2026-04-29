import numpy as np
from gauszw import gauszw
from splittau import splittau
from gsitm import gsitm
from srcfintdn import srcfintdn
from srcfintup import srcfintup
from sglscat import sglscat

def gsit(ng1, nm, nit,                         # accuracy parameters
         mu0, muup, nup, mudn, ndn, azr, naz,  # solar-view geometry
         tau, ssa, xk, nlro, nxk,              # atmopshere
         srfa):                                # surface
    '''
    Task:
        To solve scalar RTE at all levels of a verticaly inhomogeneous atmopshere
        over a Lambertian surface using the method of Gauss-Seidel iterations.
    In:
        ng1     i             number of Gaussian nodes per hemisphere
        nm      i             total number of Fourier moments; m = 0, 1, ... nm-1
        nit     i             number of iterations
        mu0     f             cos(solar zenith angle) > 0
        muup    f[nup]        cos(zenith angle) < 0 for upward directions
        nup     i             len(muup)
        mudn    f[ndn]        cos(zenith angle) > 0 for downward directions
        ndn     i             len(mudn)
        azr     f[naz]        relative azimuth in radians: azd = 0 (pi) - glint (hotspot)
        naz     i             len(azd)
        tau     f[nlr]        layer optical thicknesses, from top to bottom
        ssa     f[nlr]        layer single scattering albedo
        xk      f[nlr, nxk]   layer phase function moments; (2k+1) included
        nlr     i             number of (optical) layers
        nxk     i             number of expansion moments
        srfa    f             Lambertian surface albedo
    Out:
        Iup   f[nlr+1, nup, naz]   solution at upward directions and all levels
        Idn   f[nlr+1, ndn, naz]   solution at downward directions and all levels
    Notes:
        1. TOA irradiance (flux) Fo = 1
        2. I1dn[0, ...] = 0 - TOA boundary condition
        3. I1up[nlr, ...] = 0 if srfa = 0 - BOA boundary condition
        4. nm <= ng2 = 2*ng1
    Refs:
        1. https://doi.org/10.1016/j.cpc.2021.108198
    '''

    dtau_max = 0.01
    srfa_min = 0.0001

    ng2 = ng1 * 2
    nk = min(nxk, ng2)
    
    nlvo = nlro + 1
    Iup = np.zeros((nlvo, nup, naz))
    Idn = np.zeros((nlvo, ndn, naz))
    
    mup, w = gauszw(0.0, 1.0, ng1)
    mug = np.concatenate((-mup, mup))
    wg = np.concatenate((w, w))
    
    nlr, dtau, mask, ndtau = splittau(dtau_max, tau, nlro)
    nlv = nlr + 1
    tau0_chk = np.sum(dtau)
    print(f"(in gsit) tau0_chk = {tau0_chk: 9.6f}")
    
    Tsun = np.ones(nlv)
    Texp = np.ones((nlr, ng1))
    taue = 0.0
    for ilr in range(0, nlr):
        taue = taue + dtau[ilr]
        Tsun[ilr + 1] = np.exp(-taue / mu0)
        Texp[ilr,  :] = np.exp(-dtau[ilr] / mup)
    tau0 = np.sum(dtau)
    T0 = np.exp(-tau0/mu0)
    
    # Source function integration and Fourier summation: multiple (2+) scattering
    Igboa = np.zeros(ng1)
    Img05 = np.zeros((nlr, ng2))
    I2up = np.zeros((nlv, nup, naz))
    I2dn = np.zeros((nlv, ndn, naz))

    deltm0 = 1.0
    for m in range(nm):
        kron_cmaz = deltm0 * np.cos(m * azr)
        
        # Get all scattering orders at the Gaussian nodes
        Imup, Imdn = gsitm(m, nit, mu0, T0, mug, wg, ng1, ng2, dtau, Texp, 
                             mask, nlr, Tsun, nlv, ssa, xk[:, 0:nk], nlro, nk, srfa)
        
        # Midlayer solution for source function integration (SFI)
        for ilv in range(nlv-1):
            Img05[ilv, 0:ng1] = 0.5 * (Imup[ilv, :] + Imup[ilv+1, :])   # Iup05
            Img05[ilv, ng1:ng2] = 0.5 * (Imdn[ilv, :] + Imdn[ilv+1, :]) # Idn05
        
        # SFI: upward 
        for imu in range(ndn):
            mu = mudn[imu]
            I2mdn = srcfintdn(m, mu, mu0, Img05, mug, wg, ng2, dtau, mask, nlr, 
                              nlv, ssa, xk[:, 0:nk], nlro, nk)
            for ilv in range(nlv):
                I2dn[ilv, imu, :] += I2mdn[ilv] * kron_cmaz # inefficient use of indices - see below
        
        # SFI: downward
        if srfa > srfa_min:
            Igboa = Imdn[nlv-1, :]

        for imu in range(nup):
            mu = muup[imu]
            I2mup = srcfintup(m, mu, mu0, Img05, mug, wg, ng2, dtau, mask, nlr, 
                              nlv, ssa, xk[:, 0:nk], nlro, nk, Igboa, srfa)
            for ilv in range(nlv):
                I2up[ilv, imu, :] += I2mup[ilv] * kron_cmaz # inefficient use of indices - see below

        deltm0 = 2.0
        print(f"(in gsit) m = {m}") # reduces perfmoance a bit, but indicates the code is alive

    # Exact single scattering solution: I1[nlv, nmu, naz] - that is why I2 is shaped this way
    caz = np.cos(azr)
    I1up, I1dn = sglscat(tau, ssa, nlro, xk, nxk, mu0, muup, nup, mudn, ndn, caz, naz, srfa)
    
    four_pi = 4.0 * np.pi # for Fo_toa = 1
    for ilvo in range(nlvo):
        idx_ilvo = np.sum(ndtau[0:ilvo])
        Iup[ilvo, :, :] = I1up[ilvo, :, :] + I2up[idx_ilvo, :, :] / four_pi
        Idn[ilvo, :, :] = I1dn[ilvo, :, :] + I2dn[idx_ilvo, :, :] / four_pi
    
    return Iup, Idn
#-EOF-