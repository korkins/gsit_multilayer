# GSIT Radiative Transfer Code

## Overview

This repository provides a Python implementation of the **Gauss–Seidel Iterative Technique (GSIT)** for solving the scalar radiative transfer equation in **vertically inhomogeneous (multilayer) atmospheres**. The code computes radiation intensity fields throughout the atmosphere, including internal levels.

---

## Brief Instructions

In `main_gsit.py`, **line 11**, select one of four benchmark test cases:

### Case 1 — Pure Rayleigh atmosphere

* **TOA**: emax = 0.05 %, eavr = 0.02 %
* **LOA**: emax = 0.24 %, eavr = 0.06 %
* **BOA**: emax = 0.05 %, eavr = 0.02 %

### Case 2 — Rayleigh atmosphere with gas absorption

* **TOA**: emax = 0.03 %, eavr = 0.01 %
* **LOA**: emax = 0.09 %, eavr = 0.02 %
* **BOA**: emax = 0.05 %, eavr = 0.01 %

### Case 3 — Rayleigh + gas absorption + aerosol (dust)

* **TOA**: emax = 0.05 %, eavr = 0.01 %
* **LOA**: emax = 0.27 %, eavr = 0.05 %
* **BOA**: emax = 0.39 %, eavr = 0.06 %

### Case 4 — Case 3 + Lambertian surface (albedo = 0.3)

* **TOA**: emax = 0.05 %, eavr = 0.01 %
* **LOA**: emax = 0.20 %, eavr = 0.03 %
* **BOA**: emax = 0.33 %, eavr = 0.05 %

---

## Benchmark Notes

* Errors are computed against the Monte Carlo model **MYSTIC** (part of libRadtran: https://www.libradtran.org).
* Cases 1–3 correspond to benchmark cases B1–B3 from the IPRT intercomparison.
* Case 4 extends Case 3 by adding a Lambertian surface (new Monte Carlo results).
* **LOA (Level of Atmosphere)** is defined at **1 km**.
* The original benchmark includes polarization; here, Monte Carlo simulations were performed with **polarization disabled**.

---

## Code Structure (TREE & LOC)

```
main_gsit
│
├── gauszw (25)        # Gaussian nodes and weights
├── splittau (14)      # Splits layers into elements (dtau < dtau_max)
├── gsitm (49)         # Fourier moments via Gauss-Seidel iterations
│   ├── polleg (13)    # Legendre polynomials Pk(x)
│   ├── polqkm (14)    # Associated polynomials Qkm(x)
│   ├── sglscatdnm (10)# Single scattering (downward, BOA)
│   └── sglscatupm (2) # Single scattering (upward, TOA)
│
├── srcfintdn (21)     # Source function integration (downward)
├── srcfintup (25)     # Source function integration (upward)
│
└── sglscat (47)       # Exact single scattering
    ├── sglscatdn (8)  # Downward
    └── sglscatup (2)  # Upward
```

**Total executable lines of code (LOC): 230**

---

## Reference

The theoretical background for the GSIT method (single-layer formulation) is described in:

* https://doi.org/10.1016/j.cpc.2021.108198

---

## Notes

* TOA: Top of Atmosphere

* LOA: Level within Atmosphere (1 km)

* BOA: Bottom of Atmosphere

* The code is designed for clarity and prototyping, with structure suitable for later translation to C/Fortran if needed.

* Thanks to ChatGPT for help with creating this README file

---
