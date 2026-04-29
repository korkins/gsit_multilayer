import os
import time
import numpy as np
import matplotlib.pyplot as plt

from gauszw import gauszw
from splittau import splittau
from polleg import polleg
from polqkm import polqkm
from sglscatdnm import sglscatdnm
from sglscatupm import sglscatupm

def gsitm(m, nit,
          mu0, Tsrf, 
          mug, wg, ng1, ng2,
          dtau, Texp, mask, nlr,
          Tsun, nlv,
          ssa, xk, nlro, nk,
          srfa):

    '''
    Task:
        To compute m-th Fourier moment of intensity, Im, at Gaussian nodes, grid of
        tau-s, over a Lambertian surface using the method of Gauss-Seidel iterations.
    In:
        m      i              Fourier moment; m = 0, 1, ... nm-1
        nit    i              number of iterations
        mu0    f              cos(solar zenith angle) > 0
        Tsrf   f              exp(-tau0 / mu0)
        mug    f[ng2]         Gaussian nodes in whole sphere; upward: mug[0:ng1] < 0
        wg     f[ng2]         respective Gaussian weights
        ng1    i              number of Gaussian nodes per hemisphere
        ng2    i              ng1*2
        dtau   f[nlr]         OTs of element layers
        Texp   f[nlr, ng1]    T = exp(-dtau[ilr]/mug_positive)
        mask   i[nlr]         optical layer index = mask[ilr]; ilr = 0 : nlr-1
        nlr    i              number of element layrs; len(dtau)=len(Texp)=len(mask)
        Tsun   f[nlev]        solar bean attenuation at all levels, exp(-tau_embed/mu0)
        nlv    i              number of levels; nlv=nlr+1
        ssa    f[nlro]        single scattering albedo in optical layers
        xk     f[nlro, nk]    phase function expansion momemnts in optical layeres; (2k+1) included
        nlro   i              number of optical layers; nlro <= nlr
        nk     i              number of expansion momements used to compute Im
        srfa   f              Lambertian surface albedo
    Out:
        Iup   f[nlv, ng1]   solution at upward directions and all levels
        Idn   f[nlv, ng1]   solution at downward directions and all levels
    Notes:
        1. Scale by 1/4pi for TOA irradiance (flux) Fo = 1
        2. nit = 0: single scattering will be returned
        2. Idn[0, ...] = 0 - TOA boundary condition
        3. Iup[nlv-1, ...] = 0 if srfa = 0 - BOA boundary condition
        4. nodes are symmetric: mug[ig] = -mug[ig+ng1] for ig = 0: ng1
    
        5. THINK ME: use only mup = mug > 0 on input; use symmetry of the phase function
        
        6. wpij = np.zeros((nlr, ng2, ng2)) -- potential RAM problem: 
           1024_layers * 128_ng2 * 128_ng2 * 8byte / 1024_kb / 1024_mb = 128 MB
    Refs:
        1. https://doi.org/10.1016/j.cpc.2021.108198
    '''

    srfa_min = 0.0001
    
    mup = mug[ng1:ng2]
    w = wg[ng1:ng2]
        
    pk = np.zeros((ng2, nk))         # Legendre or Schmidt polynomials
    p = np.zeros((nlr, ng2))         # m-th moment of phase function pm(mu0, mug) for single scattering
    wpij = np.zeros((nlr, ng2, ng2)) # m-th moment of phase function pm(mug_i, mug_j) for multiple scattering, weighted
    
    I1dn = np.zeros((nlv, ng1))
    I1up = np.zeros_like(I1dn)
    
    Idn = np.zeros_like(I1dn)
    Iup = np.zeros_like(I1dn)
            
    # Precompute Pk(x) or Qkm(x), depending on m
    if m == 0:
        pk0 = polleg(mu0, nk-1)
        
        for ig in range(ng2):
            pk[ig, :] = polleg(mug[ig], nk-1)
    else:
        pk0 = polqkm(m, mu0, nk-1)
        
        for ig in range(ng2):
            pk[ig, :] = polqkm(m, mug[ig], nk-1)
    # end: if m = 0
    
    # Precompute m-th moment of phase function pm(mu0, mu) and pm(mui, muj)
    for ilro in range(nlro):
        xmom = xk[ilro, :]
        
        for ig in range(ng2):
            p[ilro, ig] = np.dot(xmom, pk[ig, :]*pk0)
            
        for ig in range(ng2):
            for jg in range(ng2):
                wpij[ilro, ig, jg] = wg[jg]*np.dot(xmom, pk[ig, :]*pk[jg, :]) # symmetry ?
    # end: for ilro
                
    # Downward initial guess: single scattering at all levels
    I11dn = np.zeros((nlr, ng1))
    for ilv in range(nlv-1):        
        e0 = Tsun[ilv]
        taui = dtau[ilv]
        Tdn = Texp[ilv, :]
        
        ilro = mask[ilv]
        ssa_phfm = ssa[ilro] * p[ilro, :]
        
        I11dn[ilv, :] = sglscatdnm(mu0, mup, ng1, taui, ssa_phfm[ng1:ng2]) * e0
        I1dn[ilv+1, :] = I1dn[ilv, :] * Tdn + I11dn[ilv, :]
    # end: for ilv
    
    # Upward initial guess: single scattering at all levels (optionally with surface)
    I11up = np.zeros((nlr, ng1))
    if m == 0 and srfa > srfa_min:
        I1up[nlv-1, :] = 4.0 * srfa * mu0 * Tsrf # Fo*rho/pi = 4pi*rho/pi=4
    
    for ilv in range(nlv-2, -1, -1):
        e0 = Tsun[ilv]
        taui = dtau[ilv]
        Tup = Texp[ilv, :]
        
        ilro = mask[ilv]
        ssa_phfm = ssa[ilro] * p[ilro, :]
        
        I11up[ilv, :] = sglscatupm(mu0, -mup, ng1, taui, ssa_phfm[0:ng1]) * e0
        I1up[ilv, :] = I1up[ilv+1, :] * Tup + I11up[ilv, :]
    # end: for ilv
    
    # Multiple scattering iterations:
    Iup = I1up.copy()
    Idn = I1dn.copy()
    
    for itr in range(nit):
        # downward
        for ilv in range(nlv-1):
            ilro = mask[ilv]
            
            T = wpij[ilro, 0:ng1, 0:ng1]
            R = wpij[ilro, 0:ng1, ng1:ng2]
            
            Iup05 = 0.5 * (Iup[ilv, :] + Iup[ilv+1, :])
            Idn05 = 0.5 * (Idn[ilv, :] + Idn[ilv+1, :])
            
            J = 0.5 * ssa[ilro] * (np.dot(R, Iup05) + np.dot(T, Idn05))

            Tdn = Texp[ilv, :]
            Idn[ilv+1, :] = Idn[ilv, :] * Tdn + I11dn[ilv, :] + (1.0 - Tdn) * J
        # end: for ilv down
        
        # upward
        if m == 0 and srfa > srfa_min:
            Iup[nlv-1, :] = ( 2.0 * srfa * np.dot(Idn[nlv-1, :], mup * w) + 
                              4.0 * srfa * mu0 * Tsrf ) # Fo*rho/pi = 4pi*rho/pi=4

        for ilv in range(nlv-2, -1, -1):
            ilro = mask[ilv]
            
            T = wpij[ilro, 0:ng1, 0:ng1]
            R = wpij[ilro, 0:ng1, ng1:ng2]
            
            Iup05 = 0.5 * (Iup[ilv, :] + Iup[ilv+1, :])
            Idn05 = 0.5 * (Idn[ilv, :] + Idn[ilv+1, :])
            
            J = 0.5 * ssa[ilro] * (np.dot(T, Iup05) + np.dot(R, Idn05))

            Tup = Texp[ilv, :]
            Iup[ilv, :] = Iup[ilv+1, :] * Tup + I11up[ilv, :] + (1.0 - Tup) * J
        # end: for ilv up
    # end: for itr

    return Iup, Idn
#==============================================================================

if __name__ == "__main__":
#
    ssa1 = 0.99999999    # no absorption case: single scattering albedo ~= 1
    rad = np.pi/180.0

    idx_bmrk = 2       # 1: R only; 2: R + G; 3: R + G + A -- see weblink above

#   Common input
    rootdir =  '..\\iprt_data_ll'
    depf = 0.03   # depolarization factor
    srfa = 0.0
    dtau_max = 0.01
    nit = 10
    
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
    
    nlro = len(tau)
    ssa = tau_sca / tau

    # mix rayleigh & aerosol
    xk = np.zeros((nlro, nxk))
    for ilro in range(nlro):
        wray = tau_ray[ilro+1] / tau_sca[ilro]
        waer = max(0.0, 1.0 - wray)
        for ik in range(nk_ray):
            xk[ilro, ik] = wray * xk_ray[ik] + waer * xk_aer[ik]
        for ik in range(nk_ray, nxk):
            xk[ilro, ik] = waer * xk_aer[ik]
    
    # solar-view geometry
    mu0 = np.cos(sza * rad)
    smu0 = np.sin(sza * rad)
    caz = np.cos(raz * rad)
    
    mup, w = gauszw(0.0, 1.0, ng1)
    mug = np.concatenate((-mup, mup))
    wg = np.concatenate((w, w))
    print("(gsit) Gauss zeros & weights:")
    for ig in range(ng1*2):
         print(f"{mug[ig]: .6f}  {wg[ig]: .6f}")
         
    ng2 = ng1*2
    nk = min(nxk, ng2) # to be used for multiple scattering 

    naz = len(raz)
    
    tau0 = np.sum(tau)
    nlr, dtau, mask, ndtau = splittau(dtau_max, tau, nlro)
    tau0_chk = np.sum(dtau)
    nlv = nlr + 1
    
    print(f"nlr = {nlr: d}")
    for ilr in range(nlr):
         print(f"ilr = {ilr: 4d}   dtau = {dtau[ilr]:.4e}   mask = {mask[ilr]: 3d}")

    print(f"input : tau0 = {tau0    : .6e}")
    print(f"regrid: tau0 = {tau0_chk: .6e}")
    
    
    taue = np.zeros(nlv)
    Tsun = np.ones(nlv)
    Texp = np.ones((nlr, ng1))
    for ilr in range(0, nlr):
        taue[ilr+1] = taue[ilr] + dtau[ilr]
        Tsun[ilr+1] = np.exp(-taue[ilr+1] / mu0)
        Texp[ilr, :] = np.exp(-dtau[ilr] / mup)
    tau0 = np.sum(dtau)
    T0 = np.exp(-tau0/mu0)
    #print("(gsitm) T0", T0)
    
    
    Iup = np.zeros((nlv, ng1, naz)) 
    Idn = np.zeros_like(Iup)

    Im_up_sfi = np.zeros((nlv, ng1, nm))
    Im_dn_sfi = np.zeros((nlv, ng1, nm))   
    
    deltm0 = 1.0
    for m in range(nm):
        Im_up, Im_dn = gsitm(m, nit, mu0, T0, mug, wg, ng1, ng2, dtau, Texp, mask, nlr, Tsun, nlv, ssa, xk[:, 0:nk], nlro, nk, srfa)
        cmaz = np.cos(m*raz*np.pi/180.0)
        for ilv in range(nlv):
            for ig in range(ng1):
                Iup[ilv, ig, :] += deltm0 * Im_up[ilv, ig] * cmaz
                Idn[ilv, ig, :] += deltm0 * Im_dn[ilv, ig] * cmaz
        deltm0 = 2.0
        
        Im_up_sfi[:, :, m] = Im_up
        Im_dn_sfi[:, :, m] = Im_dn
        
        print(f"m = {m: d}")
        
    np.savez(fname_npz, Im_up_sfi=Im_up_sfi, Im_dn_sfi=Im_dn_sfi, mug=mug, wg=wg, dtau=dtau, mask=mask, ssa=ssa, xk=xk[:, 0:nk])
    
    vza_up_gauss = np.acos(mug[0:ng1])*180.0/np.pi
    vza_dn_gauss = np.acos(mug[ng1:ng2])*180.0/np.pi
    
    # Scale to unit flux on TOA - like in the original gsit - to compare with MYSTIC
    Iup = Iup / (4.0 * np.pi)
    Idn = Idn / (4.0 * np.pi)
    
    ix_hkm_1km = 29 # by definition
    ix_loa = np.sum(ndtau[0:ix_hkm_1km])
        
    fig, axs = plt.subplots(2, 2, figsize=(16, 8))
    
    colors = ['r', 'g', 'b', 'm', 'k']
    
    I_loa_up = I_bmrk_loa[0:6, :]
    I_loa_dn = I_bmrk_loa[6:10, :]
    
    # TOA
    ax = axs[0, 0]
    for i, az in enumerate(raz):
        ax.plot(vza_up, I_bmrk_toa[:, i], 'o', color=colors[i]) #, label=f'{az}')
        ax.plot(vza_up_gauss, Iup[0, :, i], '-', color=colors[i], label=f'{az}')
    ax.set_title('TOA')
    ax.grid(True)
    
    ax.legend(title='RAZ')
    
    # LOA up
    ax = axs[0, 1]
    for i, az in enumerate(raz):
        ax.plot(vza_up, I_loa_up[:, i], 'o', color=colors[i])
        ax.plot(vza_up_gauss, Iup[ix_loa, :, i], '-', color=colors[i])
    ax.set_title('LOA (upward)')
    ax.grid(True)
    
    # BOA
    ax = axs[1, 0]
    for i, az in enumerate(raz):
        ax.plot(vza_dn, I_bmrk_boa[:, i], 'o', color=colors[i])
        ax.plot(vza_dn_gauss, Idn[nlv-1, :, i], '-', color=colors[i])
    ax.set_title('BOA')
    ax.grid(True)
    
    # LOA down
    ax = axs[1, 1]
    for i, az in enumerate(raz):
        ax.plot(vza_dn, I_loa_dn[:, i], 'o', color=colors[i])
        ax.plot(vza_dn_gauss, Idn[ix_loa, :, i], '-', color=colors[i])
    ax.set_title('LOA (downward)')
    ax.grid(True)
    
    #plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('radiance_plot.jpg', dpi=600, format='jpg', bbox_inches='tight')
    plt.show()
    
    # --- Elapsed Time ---
    elapsed_time_sec = time.time() - t0
    print("done! elapsed time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time_sec)))
#-EOF-