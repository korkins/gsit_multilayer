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
* Cases 1–3 correspond to benchmark cases B1–B3 from the IPRT intercomparison (https://www.meteo.physik.uni-muenchen.de/~iprt/doku.php?id=intercomparisons:intercomparisons).
* Case 4 extends Case 3 by adding a Lambertian surface (new Monte Carlo results).
* **LOA (Level of Atmosphere)** is defined at **1 km**.
* The original benchmark includes polarization; here, Monte Carlo simulations were performed with **polarization disabled**.

---

## Code Structure (Tree & LOC)

```
main_gsit                             # reads input, runs gsit(), tests vs. benchmarks
        |
        +-gsit (57)                   # interface for RT code
             |
             +-gauszw (25)            # Gauss quadrature nodes (zeros) and weights
             |
             +-splittau (14)          # Splits input atmospheric (optical) layers into element layers of dtau < dtau_max
             |
             +-gsitm (49)             # m-th Fourier order of intensity (all scatterings) at Gauss nodes and all levels
             |     |
             |     +-polleg (13)      # Legendre polynomials, Pk(x)
             |     |
             |     +-polqkm (14)      # Associated polynomials, Qkm(x)
             |     |
             |     +-sglscatdnm (10)  # m-th Fourier moment in single scattering at BOA of a homogeneous layer
             |     |
             |     +-sglscatupm (2)   # m-th Fourier moment in single scattering at TOA of a homogeneous layer
             |
             +-srcfintdn (21)         # Source function integration downward
             |         |
             |         +-polleg
             |         |
             |         +-polqkm
             |
             +-srcfintup (25)         # Source function integration upward
             |         |
             |         +-polleg
             |         |
             |         +-polqkm
             |
             +-sglscat (47)           # Exact single scattering
                     |
                     +-sglscatdn (8)  # single scattering at BOA of a homogeneous layer
                     |
                     +-sglscatup (2)  # single scattering at TOA of a homogeneous layer
```

**Total executable lines of code (LOC): 287**

---

## Reference

The theoretical background for the GSIT method (single-layer formulation) is described in:

* https://doi.org/10.1016/j.cpc.2021.108198

---

## Notes

* TOA: Top of Atmosphere

* LOA: Level of Atmosphere (1 km)

* BOA: Bottom of Atmosphere

* The code is designed for clarity and prototyping, with structure suitable for later translation to C/Fortran if needed.

* Thanks to ChatGPT for help with creating this README file

---
