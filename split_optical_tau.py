import numpy as np

def split_optical_tau(dtau_max, tau, nlro):
    '''
    Splits atmospheric optical layers into layer elements, whose optical thickness
    does not exceed dtau_max (accuracy parameter), and maps the layer elements
    to corresponding optical layers using a mask.

    Parameters
    ----------
        dtau_max : float
            maximum allowable optical thickness of a layer element; dtau_max > 0

        tau : float [nlro]
            optical layer thicknesses; all tau[:] > 0

        nlro : integer
            number of optical layers

    Returns
    -------
        nlr : integer
            total number of layer elements

        dtau : float [nlr]
            optical thicknesses of layer elements

        mask : integer [nlr]
            ilro = mask[ilr]: maps dtau and tau

        ndtau : integer [nlro]
            number of layer elements in each input optical layer

    Notes
    -----
        1. Constant extinction is assumed in each optical layer
        2. If tau[ilro] < dtau_max then tau[ilro] is returned; otherwise tau[ilro] is
           split into even elements not thicker than dtau_max
        3. ndtau is used to connect levels in tau[] and dtau[]
    '''

    nlr = 0
    ndtau = np.zeros(nlro, dtype=np.int32)
    for ilro in range(nlro):
        nel = int(np.ceil(tau[ilro] / dtau_max))
        ndtau[ilro] = nel
        nlr += nel

    dtau = np.zeros(nlr)
    mask = np.zeros(nlr, dtype=np.int32)

    idxel = 0
    for ilro in range(nlro):
        nel = ndtau[ilro]
        dtau_ilro = tau[ilro] / nel

        for iel in range(nel):
            dtau[idxel] = dtau_ilro
            mask[idxel] = ilro
            idxel += 1
    # end: ilro

    return nlr, dtau, mask, ndtau
# LOC: 19
#-EOF-