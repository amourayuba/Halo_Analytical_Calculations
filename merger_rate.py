from formation_time import *
from autograd import grad
from scipy.special import gamma


def lc_mrate(M0, z, xi=np.linspace(0.1, 0.9, 100), prescription1=True, sig8=sigma8, h=h, kmax=30, window='TopHat',
             prec=1000, om0=om,
             ol0=oml, omb=omb, camb=False, colos=True):
    '''
    Computes the lacy & cole merger rate  (eq 2.18) as a function of the ratio of the two mergers xi = M1/M2 and mass of decendent M0
    '''
    if prescription1:  # the function is not symmetric with respect to the merger masses, so you have to choose a prescription
        M1 = M0 * xi / (1 + xi)
    else:
        M1 = M0 / (1 + xi)
    S1 = sigma(M1, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos) ** 2
    S2 = sigma(M0, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos) ** 2
    dSdM = -(sigma(M0 * 1.0001, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos) ** 2 - sigma(M0 * 0.9999,                                                             sig8, h,
            kmax, window,'M', prec, om0, ol0, omb, camb, colos) ** 2) / (0.0002 * M0)
    w = delta_c(z, om0, ol0)
    frac1 = (2 * np.pi) ** (-0.5) * (S1 / (S2 * (S1 - S2))) ** 1.5
    frac2 = np.exp(-0.5 * w ** 2 * (S1 - S2) / (S1 * S2))
    frac3 = dSdM * (M0 - M1)
    return frac1 * frac2 * frac3


def sph_mrate_per_n(M0, z, xi=np.linspace(0.1, 0.9, 100), sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000, om0=om,
                    ol0=oml, omb=omb, camb=False, colos=True):
    ''' Spherical collapse merger rate per halo per unit redshift and mass ratio xi following Fakhouri 2008
    '''
    Mi = M0 * xi / (xi + 1)
    DeltaSi = sigma(Mi, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos) ** 2 - sigma(M0, sig8, h, kmax,
               window, 'M', prec, om0, ol0, omb, camb, colos) ** 2
    dwdz = (delta_c(z + 0.0001 * z, om0, ol0) - delta_c(z - 0.0001 * z, om0, ol0)) / (0.0002 * z)
    if type(xi) == np.ndarray:
        dSdMi = -(DeltaSi[2:] - DeltaSi[:-2]) / (Mi[2:] - Mi[:-2])
        frac1 = dwdz * M0 ** 2 / ((1 + xi[1:-1]) ** 2 * Mi[1:-1])
        frac2 = 1 / (DeltaSi[1:-1] * np.sqrt(2 * np.pi * DeltaSi[1:-1]))
        return frac1 * frac2 * dSdMi
    else:
        dSdMi = -(sigma(Mi + Mi * 0.0001, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos) ** 2 - sigma(
            Mi - Mi * 0.0001, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos) ** 2) / (Mi * 0.0002)
        return dwdz * M0 ** 2 * dSdMi / ((1 + xi) ** 2 * Mi * DeltaSi * np.sqrt(2 * np.pi * DeltaSi))


def ell_mrate_per_n(M0, z, xi=np.linspace(0.1, 0.9, 100), sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000, om0=om,
                    ol0=oml, omb=omb, camb=False, colos=True):
    '''Ellipsoidal collapse merger rate per halo per unit redshift and mass ratio xi following Zhang, Fahouri and Ma 2008
    '''
    sph = sph_mrate_per_n(M0, z, xi, sig8, h, kmax, window, prec, om0, ol0, omb, camb, colos)
    Mi = M0 * xi / (xi + 1)
    S0 = sigma(M0, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos) ** 2
    dSi = sigma(Mi, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos) ** 2 - S0

    nu0 = delta_c(z, om0, ol0) ** 2 / S0
    A0 = 0.8661 * (1 - 0.133 * nu0 ** (-0.615))
    A1 = 0.308 * nu0 ** (-0.115)
    A2 = 0.0373 * nu0 ** (-0.115)
    Sbar = dSi[1:-1] / S0
    return sph * A0 * np.exp(-0.5 * A1 ** 2 * Sbar) * (1 + A2 * Sbar ** 1.5 * (1 + A1 * Sbar ** 0.5 / gamma(1.5)))


def integ_mrate(M0, z, xi_min, xi_max, nxibins=10000, mass=False, model='EC', sig8=sigma8, h=h, kmax=30, window='TopHat',
                prec=1000, om0=om,ol0=oml, omb=omb, camb=False, colos=True):
    #xis = np.linspace(xi_min, xi_max, nxibins)
    xis = np.logspace(np.log10(xi_min), np.log10(xi_max), nxibins)
    dxi = 0.5 * (xis[2:] - xis[:-2])
    if model == 'EC':
        diff_mrate = ell_mrate_per_n(M0, z, xis, sig8, h, kmax, window, prec, om0, ol0, omb, camb, colos)
    elif model == 'SC':
        diff_mrate = sph_mrate_per_n(M0, z, xis, sig8, h, kmax, window, prec, om0, ol0, omb, camb, colos)
    if mass:
        return np.sum(diff_mrate * dxi * M0 * xis[1:-1])
    else:
        return np.sum(diff_mrate * dxi)
