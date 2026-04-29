import os
import time
import numpy as np

from polleg import polleg
from sglscatdn import sglscatdn
from sglscatup import sglscatup

#
def sglscat(tau, ssa, nlr, xk, nxk, mu0, muup, nup, mudn, ndn, caz, naz, srfa):
    '''
    Calculates single scattering at all atmospheric boundaries
    
    Parameters
    ----------
        tau, ssa : float [nlr]
              Layer optical thicknesses and single scattering albedos
              
        nlr : integer
              Number of optical layers; nlr = nh - 1
              
        xk : float [nlr, nxk]
             Layer phase function moments (on fast dimension); (2k+1) included
        
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
              Number of realtive azimuths
              
        srfa : float
               Lambertian surface albedo

    Returns
    -------
        I1up : float [nh, nup, naz]
               Upward intensity
               
        I1dn : float [nh, ndn, naz]
               Downward intensity
        
    Notes
    -----
        1. TOA irradiance (flux) Fo = 1
        2. I1dn[0, ...] = 0 - TOA boundary condition
        3. I1up[nh-1, ...] = 0 if srfa = 0 - BOA boundary condition

    References
    ----------
        1. N/A
    '''

    pi = np.pi
    ssaphf = np.zeros(naz)
    
    nh = nlr + 1 # number of heights

    taue = np.zeros(nh)
    for ih in range(1, nh):
        taue[ih] = taue[ih-1] + tau[ih-1]

    Tsun = np.exp(-taue / mu0)

    smu0 = np.sqrt(1.0 - mu0**2)

    # Legendre polynomials for p(x) = sum{xk * pk(x)}
    pkup = np.zeros((nup, naz, nxk))
    pkdn = np.zeros((ndn, naz, nxk))
    
    for imu in range(nup):
        mu = muup[imu]
        smu = np.sqrt(1.0 - mu**2)
        for iaz in range(naz):
            nu = mu*mu0 + smu0*smu*caz[iaz]
            pkup[imu, iaz, :] = polleg(nu, nxk-1)
    for imu in range(ndn):
        mu = mudn[imu]
        smu = np.sqrt(1.0 - mu**2)
        for iaz in range(naz):
            nu = mu*mu0 + smu0*smu*caz[iaz]
            pkdn[imu, iaz, :] = polleg(nu, nxk-1)

    # downward
    I1dn = np.zeros((nh, ndn, naz))
    for ih in range(nh-1):
        ilr = ih
        e0 = Tsun[ilr]
        tau_ilr = tau[ilr]
        Tdn = np.exp(-tau_ilr/mudn)
        xk_ilr = xk[ilr, :]
        ssa_ilr = ssa[ilr]

        for imu in range(ndn):
            mu = mudn[imu]
            
            for iaz in range(naz):
                ssaphf[iaz] = ssa_ilr * np.dot(xk_ilr, pkdn[imu, iaz, :])
                
            I11dn = sglscatdn(mu, mu0, tau_ilr, ssaphf, naz)
            I1dn[ih+1, imu, :] = I1dn[ih, imu, :] * Tdn[imu] + I11dn * e0

    # upward
    I1up = np.zeros((nh, nup, naz))
    if srfa > 0.0:
        I1up[nh-1, :, :] = (srfa / pi) * mu0 * Tsun[nh-1] # Fo_toa = 1: Fo*rho/pi 
    
    for ih in range(nh-2, -1, -1):
        ilr = ih
        e0 = Tsun[ilr]
        tau_ilr = tau[ilr]
        Tup = np.exp(tau_ilr/muup)
        xk_ilr = xk[ilr, :]
        ssa_ilr = ssa[ilr]
        
        for imu in range(nup):
            mu = muup[imu]
            
            for iaz in range(naz):
                ssaphf[iaz] = ssa_ilr * np.dot(xk_ilr, pkup[imu, iaz, :])
                
            I11up = sglscatup(mu, mu0, tau_ilr, ssaphf, naz)
            I1up[ih, imu, :] = I1up[ih+1, imu, :] * Tup[imu] + I11up * e0
        
    return I1up, I1dn
#==============================================================================

if __name__ == "__main__":
#
    ssa1 = 0.99999999    # no absorption case: single scattering albedo ~= 1
    rad = np.pi/180.0

    idx_bmrk = 3       # 1: R only; 2: R + G; 3: R + G + A -- see weblink above

#   Common input
    rootdir =  '..\\iprt_data_ll'
    depf = 0.03   # depolarization factor
    
    vza_up = np.array([180.0, 165.0, 150.0, 135.0, 120.0, 105.0])
    vza_dn = np.array([75.0, 50.0, 25.0, 0.0])
    raz = np.array([0.0, 45.0, 90.0, 135.0, 180.0])
    
    ivza_toa = [0, 3, 6, 9, 12, 15]
    ivza_boa = [2, 7, 12, 17]
    ivza_loa = [0, 3, 6, 9, 12, 15, 20, 25, 30, 35]
    
    d = (1.0 - depf) / (1.0 + 0.5 * depf) # if depf = 0 then d = 1 ... 
    xk_ray = np.array([1.0, 0.0, 0.5*d])  # ... and xkr[k=2] = 0.5 - ok
    nk_ray = len(xk_ray)

    if idx_bmrk == 1:
        
        print("\nIPRT-B1: Rayleigh only\n")
        folder = 'b1_rayleigh_nlr30'
        file = 'taur_profile.txt'
        fpath = os.path.join(rootdir, folder, file)
        print(fpath)
        
        dat = np.loadtxt(fpath, skiprows=1)
        hkm = dat[:, 0]
        nhkm = len(hkm)
        
        tau_ray = dat[:, 1]
        tau_gas = np.zeros_like(tau_ray)
        tau_aer = np.zeros_like(tau_ray)
        ssa_aer = 0.0 # not an array
        
        xk_aer = np.zeros_like(xk_ray)
        nk_aer = nk_ray
        nxk = max(nk_ray, nk_aer)

        sza = 60.0
        srfa = 0.0
        
        folder_bmrk = "iprt_b1/RT_rad/vect"
        path_bmrk = os.path.join(rootdir, folder, folder_bmrk)
        
        file_bmrk_toa = "RT_rad_vect_toa_ss.out"
        I_bmrk_toa = np.loadtxt(os.path.join(path_bmrk, file_bmrk_toa))[:, 3].reshape(5, 18).T[ivza_toa, :]
        
        file_bmrk_boa = "RT_rad_vect_boa_ss.out"
        I_bmrk_boa = np.loadtxt(os.path.join(path_bmrk, file_bmrk_boa))[:, 3].reshape(5, 18).T[ivza_boa, :]
        
        file_bmrk_loa = "RT_rad_vect_loa_ss.out"
        I_bmrk_loa = np.loadtxt(os.path.join(path_bmrk, file_bmrk_loa))[:, 3].reshape(5, 18*2).T[ivza_loa, :]

    elif idx_bmrk == 2:
        
        print("\nIPRT-B2: Rayleigh & Gas\n")
        folder = 'b2_rayleigh_absgas_nlr30'
        file = 'taur_taug_profile.txt'
        fpath = os.path.join(rootdir, folder, file)
        print(fpath)
        
        dat = np.loadtxt(fpath, skiprows=1)
        hkm = dat[:, 0]
        nhkm = len(hkm)
        
        tau_ray = dat[:, 1]
        tau_gas = dat[:, 3]
        tau_aer = np.zeros_like(tau_ray)
        ssa_aer = 0.0 # not an array
        
        xk_aer = np.zeros_like(xk_ray)
        nk_aer = nk_ray
        nxk = max(nk_ray, nk_aer)

        sza = 60.0
        srfa = 0.0
        
        folder_bmrk = "iprt_b2/RT_rad/vect"
        path_bmrk = os.path.join(rootdir, folder, folder_bmrk)
        
        file_bmrk_toa = "RT_rad_vect_toa_ss.out"
        I_bmrk_toa = np.loadtxt(os.path.join(path_bmrk, file_bmrk_toa))[:, 3].reshape(5, 18).T[ivza_toa, :]
        
        file_bmrk_boa = "RT_rad_vect_boa_ss.out"
        I_bmrk_boa = np.loadtxt(os.path.join(path_bmrk, file_bmrk_boa))[:, 3].reshape(5, 18).T[ivza_boa, :]
        
        file_bmrk_loa = "RT_rad_vect_loa_ss.out"
        I_bmrk_loa = np.loadtxt(os.path.join(path_bmrk, file_bmrk_loa))[:, 3].reshape(5, 18*2).T[ivza_loa, :]
        
    else:
        
        print("\nIPRT-B3: Rayleigh & Gas & Aerosol\n")

        folder = 'b3_rayleigh_absgas_aerosol_nlr30'
        file = 'taur_taug_taua_profile.txt'
        fpath = os.path.join(rootdir, folder, file)
        print(fpath)
        
        dat = np.loadtxt(fpath, skiprows=1)
        hkm = dat[:, 0]
        nhkm = len(hkm)
        
        tau_ray = dat[:, 1]
        tau_gas = dat[:, 2]
        tau_aer = dat[:, 3]
        ssa_aer = 0.787581
        
        file = 'xk1000_0838.txt'
        fpath = os.path.join(rootdir, folder, file)
        print(fpath)
        
        dat = np.loadtxt(fpath, skiprows=8)
        xk_aer = dat[:, 0]
        nk_aer = len(xk_aer)
        nxk = max(nk_ray, nk_aer)

        sza = 30.0
        srfa = 0.0
        
        folder_bmrk = "iprt_b3/RT_rad/vect"
        path_bmrk = os.path.join(rootdir, folder, folder_bmrk)
        
        file_bmrk_toa = "RT_rad_vect_toa_ss.out"
        I_bmrk_toa = np.loadtxt(os.path.join(path_bmrk, file_bmrk_toa))[:, 3].reshape(5, 18).T[ivza_toa, :]
        
        file_bmrk_boa = "RT_rad_vect_boa_ss.out"
        I_bmrk_boa = np.loadtxt(os.path.join(path_bmrk, file_bmrk_boa))[:, 3].reshape(5, 18).T[ivza_boa, :]
        
        file_bmrk_loa = "RT_rad_vect_loa_ss.out"
        I_bmrk_loa = np.loadtxt(os.path.join(path_bmrk, file_bmrk_loa))[:, 3].reshape(5, 18*2).T[ivza_loa, :]
        
        
# --- Begin calculations ---
    t0 = time.time()

#   get aerosol scattering & absorption, combined scattering, 
#       and combined optical thicknesses, and combined single scattering albedo
    tau_sca_aer = ssa_aer * tau_aer[1:]
    tau_abs_aer = np.maximum(tau_aer[1:] - tau_sca_aer, 0.0)
    tau_sca = tau_ray[1:] + tau_sca_aer
    tau_abs = tau_gas[1:] + tau_abs_aer
    tau = tau_sca + tau_abs
    
    nlr = len(tau)
    ssa = tau_sca / tau
    
    print("hkm >> tau >> ssa")
    print(f"hkm(TOA): {hkm[0]: .3f}")
    for ilr in range(nlr):
        print(f"{hkm[ilr+1]: 8.3f}    {tau[ilr]: 10.6f}    {ssa[ilr]: 8.4f}")

    tau0 = np.sum(tau)
    print(f"tau0 = {tau0: .6f}")

    # mix rayleigh & aerosol
    xk = np.zeros((nlr, nxk))
    for ilr in range(nlr):
        wray = tau_ray[ilr+1] / tau_sca[ilr]
        waer = max(0.0, 1.0 - wray)
        for ik in range(nk_ray):
            xk[ilr, ik] = wray * xk_ray[ik] + waer * xk_aer[ik]
        for ik in range(nk_ray, nxk):
            xk[ilr, ik] = waer * xk_aer[ik]
    
    # solar-view geometry
    mu0 = np.cos(sza * rad)
    smu0 = np.sin(sza * rad)
    mu_up = np.cos(vza_up * rad)
    mu_dn = np.cos(vza_dn * rad)
    caz = np.cos(raz * rad)

    nmu_up = len(mu_up)
    nmu_dn = len(mu_dn)
    naz = len(raz)

    I1up, I1dn = sglscat(tau, ssa, nlr, xk, nxk, mu0, mu_up, nmu_up, mu_dn, nmu_dn, caz, naz, srfa) 
            
    rerr = 100.0*np.abs(I1up[0, :, :]/I_bmrk_toa - 1.0)
    emax = np.amax(rerr)
    eavr = np.average(rerr)
    print(f"TOA: emax = {emax: .2f} %, eavr = {eavr: .2f} %")
    
    I1updn = np.vstack([I1up[nhkm-2, :, :], I1dn[nhkm-2, :, :]])
    
    rerr = 100.0*np.abs(I1updn/I_bmrk_loa - 1.0)
    emax = np.amax(rerr)
    eavr = np.average(rerr)
    print(f"LOA: emax = {emax: .2f} %, eavr = {eavr: .2f} %")

    rerr = 100.0*np.abs(I1dn[nhkm-1, :, :]/I_bmrk_boa - 1.0)
    emax = np.amax(rerr)
    eavr = np.average(rerr)
    print(f"BOA: emax = {emax: .2f} %, eavr = {eavr: .2f} %")
    
#   --- Elapsed Time ---
    elapsed_time_sec = time.time() - t0
    print("done! elapsed time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time_sec)))
#-EOF-