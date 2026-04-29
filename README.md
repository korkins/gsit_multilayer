BRIEF INSTRUCTIONS:

In main_gsit.py, line 11, you can define one of 4 test cases (maximum and averaged errors will be printed out):

bmrk_id = 1: pure Rayleigh atmosphere
TOA: emax =  0.05 %, eavr =  0.02 %
LOA: emax =  0.24 %, eavr =  0.06 %
BOA: emax =  0.05 %, eavr =  0.02 %

bmrk_id = 2: Rayleigh atmosphere with gas absorption
TOA: emax =  0.03 %, eavr =  0.01 %
LOA: emax =  0.09 %, eavr =  0.02 %
BOA: emax =  0.05 %, eavr =  0.01 %

bmrk_id = 3: Rayleigh atmosphere with gas absorption and aerosol (dust particles)
TOA: emax =  0.05 %, eavr =  0.01 %
LOA: emax =  0.27 %, eavr =  0.05 %
BOA: emax =  0.39 %, eavr =  0.06 %

bmrk_id = 4: Rayleigh atmosphere with gas absorption and aerosol (dust particles) over a Lambertian surface (albedo 0.3)
TOA: emax =  0.05 %, eavr =  0.01 %
LOA: emax =  0.20 %, eavr =  0.03 %
BOA: emax =  0.33 %, eavr =  0.05 %

The maximum (emax) and average (eavr) errors are calculated vs. Monte-Carlo code MYSTIC (https://www.libradtran.org). Cases 1-3 correspond to B1-3 here: https://www.meteo.physik.uni-muenchen.de/~iprt/doku.php?id=intercomparisons:intercomparisons.
Case 4 is atmosphere from the Case 3 + surface (new Monte-Carlo results were generated for this test).

LOA (level of atmosphere) is defined at 1km in the mentioned published benchmark (note, the benchmark is polarized, but I re-run Monte-Carlo with polarization turned OFF).

TREE & LOC:
main_gsit # reads input, runs gsit(), tests vs. benchmarks
        |
        +-gauszw( 25 ) # Gaussian zeros and weights
        |
        +-splittau( 14 ) # Splits input atmopsheric (optical) layers into element layeres of dtau < dtau_max
        |
        +-gsitm( 49 ) # m-th Fourier order of intensity (all scatterings) at Gauss nodes and all levels
        |     |
        |     +-polleg( 13 ) # Legendre polynomials, Pk(x)
        |     |
        |     +-polqkm( 14 ) # Associated polynomials, Qkm(x)
        |     |
        |     +-sglscatdnm( 10 ) # m-th Fourier moment in single scattering at BOA of a homogeneous layer
        |     |
        |     +-sglscatupm( 2 ) # m-th Fourier moment in single scattering at TOA of a homogeneous layer
        |
        +-srcfintdn( 21 ) # Source function integration downward
        |
        +-srcfintup( 25 ) # Source function integration upward
        |
        +-sglscat( 47 ) # Exact single scattering
                |
                +-sglscatdn( 8 ) # single scattering at BOA of a homogeneous layer
                |
                +-sglscatup( 2 ) # single scattering at TOA of a homogeneous layer


LOC = 25 + 14 + 49 + 13 + 14 + 10 + 2 + 21 + 25 + 47 + 8 + 2 = 230

REFERENCES:
Theoretical background for this Gauss-Seidel RT code (single layer version) is described here: https://doi.org/10.1016/j.cpc.2021.108198
