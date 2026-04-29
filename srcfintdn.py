import os
import time
import numpy as np

from sglscat import sglscat
from polleg import polleg
from polqkm import polqkm

#
def srcfintdn(m, mu, mu0, Ig05, mug, wg, ng2, dtau, mask, nlr, nlv, ssa, xk, nol, nk):
    '''
    Task:
        To compute m-th Fourier moment of 2+ scattering intensity, I2m, at a
        user-defined downward direction mu and at many atmopsheric levels.
    In:
        m      i             Fourier moment; m = 0, 1, ... nm-1
        mu     f             cos(vza_dn) > 0: down
        mu0    f             cos(solar zenith angle) > 0
        Ig05   f[nlev, ng2]  m-th Fourier momemnt for averaged intensity at Gauss nodes (all scattering orders)
        mug    f[ng2]        Gaussian nodes in whole sphere; upward: mug[0:ng1] < 0
        wg     f[ng2]        respective Gaussian weights
        ng1    i             total number of Gauss nodes per hemisphere
        ng2    i             ng1*2
        dtau   f[nlr]        optical thickness of element layers
        mask   i[nlr]        optical layer index = mask[ilr]; ilr = 0 : nlr-1
        nlr    i             number of element layrs; len(dtau)=len(Texp)=len(mask)
        nlv    i             number of levels; nlv=nlr+1
        ssa    f[nol]        single scattering albedo in optical layers
        xk     f[nol, nk]    phase function expansion momemnts in optical layeres; (2k+1) included
        nol    i             number of optical layers; nol <= nlr
        nk     i             number of expansion momements used to compute Im
    Out:
        Idn   f[nlv]         2+ scattering orders solution at all levels downward
    Notes:
        1. Single scattering is dropped here
        1. Scale by 1/4pi for TOA irradiance (flux) Fo = 1
        2. Idn[0, ...] = 0 - TOA boundary condition
    Refs:
        1. https://doi.org/10.1016/j.cpc.2021.108198
    '''

    I2dn = np.zeros(nlv)

    pk = np.zeros((ng2, nk))    # Legendre or Schmidt polynomials
    wpij = np.zeros((nlr, ng2)) # m-th moment of phase function pm(mu, mug_j) for multiple scattering, weighted

    # Precompute Pk(x) or Qkm(x), depending on m
    if m == 0:
        pku = polleg(mu, nk-1) # mu > 0
        for ig in range(ng2):
            pk[ig, :] = polleg(mug[ig], nk-1)
    else:
        pku = polqkm(m, mu,  nk-1)
        for ig in range(ng2):
            pk[ig, :] = polqkm(m, mug[ig], nk-1)
    # end: if m = 0

    # Precompute m-th moment of phase function pm(mu0, mu) and pm(mui, muj)
    for iol in range(nol):
        xmom = xk[iol, :]
        for ig in range(ng2):
            wpij[iol, ig] = wg[ig]*np.dot(xmom, pku*pk[ig, :])
    # end: for iol
    
    for ilv in range(nlv-1):
        iol = mask[ilv]
        Tdn = np.exp(-dtau[ilv]/mu)
        J = 0.5 * ssa[iol] * np.dot(wpij[iol, :], Ig05[ilv])
        I2dn[ilv+1] = I2dn[ilv] * Tdn + (1.0 - Tdn) * J # note: SS is neither added nor subtracted
    # end: for ilv down

    return I2dn
#==============================================================================

if __name__ == "__main__":
#
    ssa1 = 0.99999999    # no absorption case: single scattering albedo ~= 1
    rad = np.pi/180.0

    idx_bmrk = 3       # 1: R only; 2: R + G; 3: R + G + A -- see weblink above

#   Common input
    rootdir =  '..\\iprt_data_ll'
    depf = 0.03   # depolarization factor
    srfa = 0.0
    dtau_max = 0.01
    
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
        
        ng1 = 8*4
        nm = 3
        
        folder_bmrk = "iprt_b1/RT_rad/scal"
        path_bmrk = os.path.join(rootdir, folder, folder_bmrk)
        
        file_bmrk_toa = "RT_rad_scal_toa_ms.out"
        I_bmrk_toa = np.loadtxt(os.path.join(path_bmrk, file_bmrk_toa))[:, 3].reshape(5, 18).T[ivza_toa, :]
        
        file_bmrk_boa = "RT_rad_scal_boa_ms.out"
        I_bmrk_boa = np.loadtxt(os.path.join(path_bmrk, file_bmrk_boa))[:, 3].reshape(5, 18).T[ivza_boa, :]
        
        file_bmrk_loa = "RT_rad_scal_loa_ms.out"
        I_bmrk_loa = np.loadtxt(os.path.join(path_bmrk, file_bmrk_loa))[:, 3].reshape(5, 18*2).T[ivza_loa, :]
        
        fname_npz = 'iprt-b1.npz'

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
        
        ng1 = 8
        nm = 3
        
        folder_bmrk = "iprt_b2/RT_rad/scal"
        path_bmrk = os.path.join(rootdir, folder, folder_bmrk)
        
        file_bmrk_toa = "RT_rad_scal_toa_ms.out"
        I_bmrk_toa = np.loadtxt(os.path.join(path_bmrk, file_bmrk_toa))[:, 3].reshape(5, 18).T[ivza_toa, :]
        
        file_bmrk_boa = "RT_rad_scal_boa_ms.out"
        I_bmrk_boa = np.loadtxt(os.path.join(path_bmrk, file_bmrk_boa))[:, 3].reshape(5, 18).T[ivza_boa, :]
        
        file_bmrk_loa = "RT_rad_scal_loa_ms.out"
        I_bmrk_loa = np.loadtxt(os.path.join(path_bmrk, file_bmrk_loa))[:, 3].reshape(5, 18*2).T[ivza_loa, :]
        
        fname_npz = 'iprt-b2.npz'
        
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
        ng1 = 32
        nm = 24
        
        folder_bmrk = "iprt_b3/RT_rad/scal"
        path_bmrk = os.path.join(rootdir, folder, folder_bmrk)
        
        file_bmrk_toa = "RT_rad_scal_toa_ms.out"
        I_bmrk_toa = np.loadtxt(os.path.join(path_bmrk, file_bmrk_toa))[:, 3].reshape(5, 18).T[ivza_toa, :]
        
        file_bmrk_boa = "RT_rad_scal_boa_ms.out"
        I_bmrk_boa = np.loadtxt(os.path.join(path_bmrk, file_bmrk_boa))[:, 3].reshape(5, 18).T[ivza_boa, :]
        
        file_bmrk_loa = "RT_rad_scal_loa_ms.out"
        I_bmrk_loa = np.loadtxt(os.path.join(path_bmrk, file_bmrk_loa))[:, 3].reshape(5, 18*2).T[ivza_loa, :]
        
        fname_npz = 'iprt-b3.npz'
        
# --- Begin calculations ---
    t0 = time.time()

#   get aerosol scattering & absorption, combined scattering, 
#       and combined optical thicknesses, and combined single scattering albedo
    tau_sca_aer = ssa_aer * tau_aer[1:]
    tau_abs_aer = np.maximum(tau_aer[1:] - tau_sca_aer, 0.0)
    tau_sca = tau_ray[1:] + tau_sca_aer
    tau_abs = tau_gas[1:] + tau_abs_aer
    tau = tau_sca + tau_abs
    
    nol = len(tau)
    ssa = tau_sca / tau

    # mix rayleigh & aerosol
    xk = np.zeros((nol, nxk))
    for iol in range(nol):
        wray = tau_ray[iol+1] / tau_sca[iol]
        waer = max(0.0, 1.0 - wray)
        for ik in range(nk_ray):
            xk[iol, ik] = wray * xk_ray[ik] + waer * xk_aer[ik]
        for ik in range(nk_ray, nxk):
            xk[iol, ik] = waer * xk_aer[ik]
    
    # solar-view geometry
    mu0 = np.cos(sza * rad)
    smu0 = np.sin(sza * rad)
    caz = np.cos(raz * rad)
    mu_up = np.cos(vza_up * rad)
    mu_dn = np.cos(vza_dn * rad)
    
    print("reading npz file:", fname_npz)
    data = np.load(fname_npz)
    Im_up_sfi = data['Im_up_sfi']
    Im_dn_sfi = data['Im_dn_sfi']
    mug = data['mug']
    wg = data['wg']
    dtau = data['dtau']
    mask = data['mask']
    ssa = data['ssa']
    xk_multiscat = data['xk']
    
    nlv, ng1, nm =  Im_up_sfi.shape
    ng2 = ng1 * 2
    nlr = nlv - 1
    nol, nk = xk_multiscat.shape

    naz = len(raz)
    
    nup = len(vza_up)
    ndn = len(vza_dn)
    I2up = np.zeros((nlv, nup, naz)) 
    I2dn = np.zeros((nlv, ndn, naz))
    
    Img05 = np.zeros((nlr, ng2))
    
    deltm0 = 1.0
    for m in range(nm):
        kron_cmaz = deltm0 * np.cos(m*raz*np.pi/180.0)
        
        for ilv in range(nlv-1):
            Iup05 = 0.5 * (Im_up_sfi[ilv, :, m] + Im_up_sfi[ilv+1, :, m])
            Idn05 = 0.5 * (Im_dn_sfi[ilv, :, m] + Im_dn_sfi[ilv+1, :, m])
            Img05[ilv, :] = np.concatenate((Iup05, Idn05))  
        
        for imu in range(ndn):
            mu = mu_dn[imu]
            I2dnm = srcfintdn(m, mu, mu0, Img05, mug, wg, ng2, dtau, mask, nlr, nlv, ssa, xk_multiscat, nol, nk)
            for ilv in range(nlv):
                I2dn[ilv, imu, :] += I2dnm[ilv] * kron_cmaz
        deltm0 = 2.0        
        print(f"m = {m: d}")
    # Scale to unit flux on TOA - like in the original gsit - to compare with MYSTIC
    #Iup = Iup / (4.0 * np.pi)
    I2dn = I2dn / (4.0 * np.pi)
    
    I1up, I1dn = sglscat(tau, ssa, nol, xk, nxk, mu0, mu_up, nup, mu_dn, ndn, caz, naz, srfa)
    
    
    
    
    
    
    
    
    Idn_boa = I1dn[nol, :, :] + I2dn[nlr, :, :]
    rerr = 100.0*np.abs(Idn_boa/I_bmrk_boa - 1.0)
    emax = np.amax(rerr)
    eavr = np.average(rerr)
    print(f"BOA: emax = {emax: .2f} %, eavr = {eavr: .2f} %")
    
    
    #ix_hkm_1km = 29 # by definition
    #ix_loa = np.sum(ndtau[0:ix_hkm_1km])
        
    
    
    # --- Elapsed Time ---
    elapsed_time_sec = time.time() - t0
    print("done! elapsed time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time_sec)))
#-EOF-