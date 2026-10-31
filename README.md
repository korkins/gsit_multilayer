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
main_gsit    # reads input,  runs gauss_seidel_iterations(...), tests vs. benchmarks
|
+-gauss_seidel_iterations (56)
                        |
                        +-gauss_nodes_weights (27)
                        |
                        +-split_optical_tau (19)
                        |
                        +-gauss_seidel_iterations_m (77)
                        |                         |
                        |                         +-legendre_polynomial (10)
                        |                         |
                        |                         +-schmidt_polynomial (16)
                        |                         |
                        |                         +-single_scattering_up_m (4)
                        |                         |
                        |                         +-single_scattering_down_m (13)
                        |
                        +-source_function_integrate_up (30)
                        |                            |
                        |                            +-legendre_polynomial
                        |                            |
                        |                            +-schmidt_polynomial
                        |
                        +-source_function_integrate_down (26)
                        |                              |
                        |                              +-legendre_polynomial
                        |                              |
                        |                              +-schmidt_polynomial
                        |
                        +-single_scattering (58)
                                          |
                                          +-single_scattering_up (4)
                                          |
                                          +-single_scattering_down (10)
```

**Total executable lines of code (LOC): 350**

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
