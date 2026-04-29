import os
import time
import numpy as np
#
def splittau(dtau_max, tau, nlr):
    '''
    Task:
        To split atmopsheric layers into elements whose optical thickness
        does not exceed dtau_max; returned are opticla_layer = mask(element) and
        number of layer elements in each atmospheric layer
    In:
        dtau_max   f        maximum OT of layer element
        tau        f[nlr]   OTs of layers to be regridded; tau[...] > 0.0
        nlr        i        number of input layers; len(tau)
    Out:
        nelem   i          total number of layer lements
        dtau    f[nelem]   regridded tau: new OTs of layer elements
        mask    i[nelem]   ilr=mask[ielem]: maps dtay and tau
        ndtau   i[nlr]     number of layer elements ielem in each input layer ilr
    Notes:
        1. Constant extinction is assumed for optical layers
        2. If tau[ilr] < dtau_max then tau[ilr] is returned; otherwise tau[ilr] is
           split into even elements not thicker than dtau_max
        3. ndtau is used to connect levels in tau[] and dtau[]
    Refs:
        1. -
    '''
    
    # define the number of element layers
    nelem = 0
    for ilr in range(nlr):
        nelem += int(np.ceil(tau[ilr]/dtau_max))
        
    dtau = np.zeros(nelem)
    mask = np.zeros(nelem, dtype=np.int32)
    ndtau = np.zeros(nlr, dtype=np.int32)

    idxel = -1
    for ilr in range(nlr):
        tau_ilr = tau[ilr]
        nel = int(np.ceil(tau_ilr/dtau_max))
        dtau_ilr = tau_ilr / nel
        ndtau[ilr] = nel
        
        for iel in range(nel):
            idxel += 1
            dtau[idxel] = dtau_ilr
            mask[idxel] = ilr

    return nelem, dtau, mask, ndtau
#==============================================================================
#
if __name__ == "__main__":
    t0 = time.time()

    print("\nIPRT-B3: Rayleigh & Gas & Aerosol\n")
    rootdir =  '..\\iprt_data_ll'
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
    
    tau = tau_ray[1:] + tau_gas[1:] + tau_aer[1:]
    nlayers = len(tau)

    dtau_max = 0.01
    
    nelem, dtau, mask, ndtau = splittau(dtau_max, tau, nlayers)
    
    print(f"nelem = {nelem: d}")
    for iel in range(nelem):
        ilr = mask[iel]
        nel = ndtau[ilr]
        print(f"iel = {iel: 3d}   dtau = {dtau[iel]:.4e}   mask = {mask[iel]: 3d}   nel = {nel: 3d}")
    
    print(f"input : tau0 = {np.sum(tau) : .6e}")
    print(f"regrid: tau0 = {np.sum(dtau): .6e}")


#   --- Elapsed Time ---
    elapsed_time_sec = time.time() - t0
    print("done! elapsed time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time_sec)))
#-EOF-