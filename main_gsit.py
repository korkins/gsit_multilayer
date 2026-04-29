import time
import numpy as np
from gsit import gsit

# Benchmark ID:
#   1 - Rayleigh profile; black surface
#   2 - Rayleigh and gas profiles; black surface
#   3 - Rayleigh, gas, and aerosol profiles; black surface
#   4 - Case 3 with Lambertian surface, srfa = 0.3

bmrk_id = 4

# common data
rootdir =  "./bmrk_data"

# number of iterations
nit = 10

# Rayleigh scattering & depolarization factor
depf = 0.03
d = (1.0 - depf) / (1.0 + 0.5 * depf) # if depf = 0 then d = 1 ... 
xk_ray = np.array([1.0, 0.0, 0.5*d])  # ... and xkr[k=2] = 0.5 - ok
nk_ray = len(xk_ray)

srfa = 0.0 # redefined for case 4

vza_up = np.arange(180.0, 95.0 - 1.0, -5.0)
vza_dn = np.arange(85.0, 0.0 - 1.0, -5.0)
azd = np.array([0.0, 45.0, 90.0, 135.0, 180.0])

nup = len(vza_up)
ndn = len(vza_dn)
nvz = nup + ndn
naz = len(azd)

fname_bmrk_toa = "RT_rad_scal_toa_ms.out"
fname_bmrk_loa = "RT_rad_scal_loa_ms.out"
fname_bmrk_boa = "RT_rad_scal_boa_ms.out"

if bmrk_id == 1:
    print("Test 1: Rayleigh only (IPRT-B1)")
    
    sza = 60.0
    ng1 = 8
    nm = 3
    
    subdir = "iprt_b1"
    path = rootdir + '/' + subdir
    print("path:", path)    
    
    file = 'taur.txt'
    fpath = path + '/' + file
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

elif bmrk_id == 2:
    print("Test 2: Rayleigh & Gas (IPRT-B2)")
    
    sza = 60.0
    ng1 = 8
    nm = 3
    
    subdir = "iprt_b2"
    path = rootdir + '/' + subdir
    print("path:", path)    
    
    file = 'taur_taug.txt'
    fpath = path + '/' + file
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
    
elif bmrk_id == 3:

    print("Test 3: Rayleigh & Gas & Aerosol (IPRT-B3)")
    
    sza = 30.0   # note different SZA
    ng1 = 32
    nm = 16
    
    subdir = "iprt_b3"
    path = rootdir + '/' + subdir
    print("path:", path)    
    
    file = 'taur_taug_taua.txt'
    fpath = path + '/' + file
    dat = np.loadtxt(fpath, skiprows=1)
    hkm = dat[:, 0]
    nhkm = len(hkm)
        
    tau_ray = dat[:, 1]
    tau_gas = dat[:, 2]
    tau_aer = dat[:, 3]
    ssa_aer = 0.787581
    
    file = 'xk1000_0838.txt'
    fpath = fpath = path + '/' + file  
    dat = np.loadtxt(fpath, skiprows=1)
    xk_aer = dat[:, 1]
    nk_aer = len(xk_aer)
    nxk = max(nk_ray, nk_aer)

else:
    print("Test 4: Rayleigh & Gas & Aerosol & Lambertian surface")
    
    sza = 30.0
    ng1 = 32
    nm = 16
    
    subdir = "iprt_b3_lambert"
    path = rootdir + '/' + subdir
    print("path:", path)    
    
    file = 'taur_taug_taua.txt'
    fpath = path + '/' + file
    dat = np.loadtxt(fpath, skiprows=1)
    hkm = dat[:, 0]
    nhkm = len(hkm)

    tau_ray = dat[:, 1]
    tau_gas = dat[:, 2]
    tau_aer = dat[:, 3]
    ssa_aer = 0.787581
    
    file = 'xk1000_0838.txt'
    fpath = fpath = path + '/' + file  
    dat = np.loadtxt(fpath, skiprows=1)
    xk_aer = dat[:, 1]
    nk_aer = len(xk_aer)
    nxk = max(nk_ray, nk_aer)
    
    srfa = 0.3
    
# --- Read and shape benchmark data ---
I_bmrk_toa = np.loadtxt(path + '/' + fname_bmrk_toa)[:, 3].reshape(naz, nup).T
I_bmrk_loa = np.loadtxt(path + '/' + fname_bmrk_loa)[:, 3].reshape(naz, nvz).T
I_bmrk_boa = np.loadtxt(path + '/' + fname_bmrk_boa)[:, 3].reshape(naz, ndn).T

# --- Begin calculations ---
t0 = time.time()

# --- Convert individual components to total tau & ssa ---
tau_sca_aer = ssa_aer * tau_aer[1:]
tau_abs_aer = np.maximum(tau_aer[1:] - tau_sca_aer, 0.0)
tau_sca = tau_ray[1:] + tau_sca_aer
tau_abs = tau_gas[1:] + tau_abs_aer
tau = tau_sca + tau_abs    
nlro = len(tau)
nlvo = nlro + 1
ssa = tau_sca / tau
tau0 = np.sum(tau)
print(f"tau0 = {tau0:9.6f}")

# --- Mix rayleigh & aerosol ---
xk = np.zeros((nlro, nxk))
for ilro in range(nlro):
    wray = tau_ray[ilro + 1] / tau_sca[ilro]
    waer = max(0.0, 1.0 - wray)
    for ik in range(nk_ray):
        xk[ilro, ik] = wray * xk_ray[ik] + waer * xk_aer[ik]
    for ik in range(nk_ray, nxk):
        xk[ilro, ik] = waer * xk_aer[ik]
        
# --- Solar-view geometry ---
rad = np.pi / 180.0
mu0 = np.cos(sza * rad)
muup = np.cos(vza_up * rad)
mudn = np.cos(vza_dn * rad)
azr = azd * rad

# --- Solve RTE ---
Iup, Idn = gsit(ng1, nm, nit, mu0, muup, nup, mudn, ndn, azr, naz, 
                tau, ssa, xk, nlro, nxk, srfa)

# --- Accuracy analysis ---
rerr = 100.0*np.abs(Iup[0, :, :]/I_bmrk_toa - 1.0)
emax = np.amax(rerr)
eavr = np.average(rerr)
print(f"TOA: emax = {emax: .2f} %, eavr = {eavr: .2f} %")

I1updn = np.vstack([Iup[nlvo-2, :, :], Idn[nlvo-2, :, :]]) # h=1km <--> nlv-2

rerr = 100.0*np.abs(I1updn/I_bmrk_loa - 1.0)
emax = np.amax(rerr)
eavr = np.average(rerr)
print(f"LOA: emax = {emax: .2f} %, eavr = {eavr: .2f} %")

rerr = 100.0*np.abs(Idn[nlvo-1, :, :]/I_bmrk_boa - 1.0)
emax = np.amax(rerr)
eavr = np.average(rerr)
print(f"BOA: emax = {emax: .2f} %, eavr = {eavr: .2f} %")

# --- Elapsed Time ---
elapsed_time_sec = time.time() - t0

hours = int(elapsed_time_sec // 3600)
minutes = int((elapsed_time_sec % 3600) // 60)
seconds = elapsed_time_sec % 60  # keeps fraction

print(f"elapsed time: {hours:02d}:{minutes:02d}:{seconds:04.1f}")
#-EOF-