from halo_mass_function import *
from autograd import grad
from math import factorial


def upcrossing(M1, M2, z1, z2, sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000, om0=om,
               ol0=oml, omb=omb, camb=False):
    """
    Upcrossing rate between halos at masses M1 to M2 at redshifts from z1 to z2 with spherical collapse
    :param M1: float Initial mass
    :param M2: float Final mass
    :param z1: float initial redshift
    :param z2: float final redshift
    :param sig8: float sigma_8 cosmological parameter
    :param h: float H0/100 parameter
    :param kmax: int or float maximum wavenumber up to which to integrate the power spectrum
    :param window: str window function to choose
    :param prec: int how many power spectrum bins used for integration to calculate sigma(R)
    :param om0: float matter density parameter today
    :param ol0: float dark energy density parameter today
    :param omb: float baryon density today
    :param camb: bool whether to use the camb power spectrum
    :return: float conditional probability of going from M1,z1 to M2, s2
    """
    w1 = delta_c(z1, om0, ol0)  # critical overdensity at z1
    w2 = delta_c(z2, om0, ol0)
    S1 = sigma(M1, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb) ** 2
    S2 = sigma(M2, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb) ** 2
    dw = w1 - w2
    dS = S1 - S2
    return np.exp(-dw ** 2 / (2 * dS)) * dw / np.sqrt(2 * np.pi * dS ** 3)  # follows eq. 2.15 of lacey&cole93


def K(ds, dw, model='sheth', A=0.27, a=0.707, p=0.3):
    """
    Conditional probability function form. Either spherical or ellipsoilda collapse model
    :param ds: float difference sigma(M1) - sigma(M2)
    :param dw: float delta_c(z1) - delta_c(z2)
    :param model: str if 'sheth' uses sheth&tormen function otherwise uses lacey&cole one
    :param A: float sheth&tormen parameter
    :param a: float sheth&tormen parameter
    :param p: float sheth&tormen parameter
    :return: float conditional probability
    """
    if model == 'sheth':
        ndw = np.sqrt(a) * dw
        return A * (1 + (ds / ndw ** 2) ** p) * ndw * np.exp(-ndw ** 2 / (2 * ds)) / np.sqrt(2 * np.pi * ds ** 3)
    else:
        return dw * np.exp(-dw ** 2 / (2 * ds)) / np.sqrt(2 * np.pi * ds ** 3)


def mu(St, a):
    return (1 + (2 ** a - 1) * St) ** (1 / a)


def f_ec(S1, S0, w1, w0):
    """
    Ellipsoidal collapse multiplicity function.
    :param S1: float sigma(M1) where M1 is the initial mass
    :param S0: float sigma(M0) where M0 is the descendant mass
    :param w1: float delta_c(z1) where z1 is the initial redshift
    :param w0: float delta_c(z0) where z0 is the final redshift
    :return: value of the EC multiplicity function
    """
    dw = w1 - w0
    dS = S1 - S0
    nu0 = w0 ** 2 / S0
    A0 = 0.8661 * (1 - 0.133 * nu0 ** (-0.615))
    A1 = 0.308 * nu0 ** (-0.115)
    A2 = 0.0373 * nu0 ** (-0.115)
    Sbar = dS / S0
    A3 = A0 ** 2 + 2 * A0 * A1 * np.sqrt(dS * Sbar) / dw
    return A0 * (2 * np.pi) ** (-0.5) * dw * dS ** (-1.5) * \
        np.exp(-0.5 * A1 ** 2 * Sbar) * (np.exp(-0.5 * A3 * dw ** 2 / dS) + A2 * Sbar ** 1.5 * (
                1 + 2 * A1 * np.sqrt(Sbar / np.pi)))  # equation 5 of Zhang et al. (2008)


def f_sc(S1, S0, w1, w0):
    """
       Spherical collapse multiplicity function.
       :param S1: float sigma(M1) where M1 is the initial mass
       :param S0: float sigma(M0) where M0 is the descendant mass
       :param w1: float delta_c(z1) where z1 is the initial redshift
       :param w0: float delta_c(z0) where z0 is the final redshift
       :return: value of the EC multiplicity function
       """
    dw = w1 - w0
    ds = S1 - S0
    return (2 * np.pi) ** -0.5 * dw * ds ** (-1.5) * np.exp(-0.5 * dw ** 2 / ds)  # lacey&cole multiplicity function


def Barrier(s, delta, alpha, beta, a):
    return np.sqrt(a) * delta * (1 + beta * s ** alpha / (a * delta ** 2) ** alpha)


def my_grad(fun, order):
    for i in range(order):
        fun = grad(fun)
    return fun


def nth_grad(x, y, axis=0, order=1):
    dx = np.gradient(x, axis=axis, edge_order=2)
    dy = np.gradient(y, axis=axis, edge_order=2)
    if order == 1:
        return dy / dx
    for i in range(1, order):
        dy = np.gradient(dy, axis=axis, edge_order=2)
    return dy / dx ** order


def ngTaylor(s, x0, dB, order, axis=0):
    res = dB
    for j in range(1, order):
        f = nth_grad(s, dB, axis=axis, order=j)
        res = res + (x0 - s) ** j * f / factorial(j)
    return res


def proba(M, zf, frac=0.5, acc=1000, zi=0.0, sig8=sigma8, h=h, kmax=30, window='TopHat', prec=300, om0=om,
          ol0=oml, omb=omb, camb=False, model='EC', colos=False, alpha=0.615, beta=0.485, a=0.7, order=3):
    """
     Probability density of a halo of mass M at redshift zi has had a fraction frac of its mass at z=zf.
     :param M: float. mass of the halo considered
     :param zf: float or ndarray. formation redshift(s) at which we want to calculate the probability
     :param frac: float between 0 and 1. Fraction of mass to define formation redshift. Default :0.5
     :param acc: int. Number redshift steps. Default : 1000.
     :param zi: float. Observed redshift of considered halo. Default :0.
     :param sig8: float : sigma 8 cosmo parameter
     :param h: float : H0/100 cosmo parameter
     :param kmax: float or int : maximum wavenumber for CAMB power spectrum.
     :param window:  str : type of smoothing window function. either "TopHat", "Gauss" or k-Sharp'.
     :param prec: int : number of bins for integral calculations.
     :param om0: float : fraction matter density
     :param ol0: float : fraction dark energy density
     :param omb: float : fraction baryon density
     :param camb:  boolean : if using camb spectrum or analytical version of Eisenstein and Hu.
     :param model: if Press&Schechter mass function "press" or ellipsoidal collapse "EC"
     :param colos: :param Colos : boolan : using Colossus halo mass function or not
     :return: Probability density function of redshift at which halos had x fraction of their mass
     """
    S0 = sigma(M, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos) ** 2  # variance of the field at mass M
    w0 = delta_c(zi, om0, ol0)  # critical density at observed redshift
    if type(zf) == np.ndarray:  # for probability distribution. This is to have a parallel version with no for loops
        mass = np.logspace(np.log10(M * frac), np.log10(M), acc)  # size (0, acc) masses to calculate the integral
        l = len(zf)  # number of steps in PDF
        mat_zf = np.array([zf] * acc)  # (acc, l)
        mat_mass = np.array([mass] * l).transpose()  # duplicating mass array to vectoralize calculations (acc, l)
        mat_wf = delta_c(mat_zf, om0, ol0) - w0  # (acc, l)
        mat_S = sigma(mat_mass, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb,
                      colos) ** 2 - S0  # variance difference of all masses (acc, l)
        mat_S[-1, :] = 1e-10  # nonzero value to avoid numerical effects
        mat_nu = mat_wf / np.sqrt(mat_S)  # (acc, l) peak height
        if model == 'EC':  # Ellipsoidal collapse probability density function
            mat_f = f_ec(mat_S[:, :] + S0, S0, mat_wf[:, :] + w0,
                         w0)  # Computing the multiplicity function at each mass
            mat_ds = 0.5 * (mat_S[2:, :] - mat_S[:-2, :])  # differential to use to integrate over
            return -M * np.sum(mat_ds * mat_f[1:-1, :] / mat_mass[1:-1, :], axis=0)
        elif model == 'sheth2002':
            b1 = Barrier(mat_S + S0, mat_wf + w0, alpha, beta, a)
            b2 = Barrier(S0, w0, alpha, beta, a)
            dB = b2 - b1
            T1 = np.abs(ngTaylor(mat_S + S0, S0, dB, axis=0, order=order))
            gradS = np.gradient(mat_S + S0, axis=0, edge_order=2)
            mat_f = T1 * np.exp(-0.5 * dB ** 2 / mat_S) / (np.sqrt(2 * np.pi) * mat_S ** 1.5)
            return -M * np.sum(gradS * mat_f / mat_mass, axis=0)
        else:
            mat_f = fps(mat_nu[:-1, :]) / mat_nu[:-1, :]  # (acc-1, l) # Press & Schechter multiplicity function
            mat_dnu = (mat_nu[2:-1, :] - mat_nu[:-3, :]) * 0.5  # (acc-3, l) # differential to integrate with
            return M * np.sum(mat_dnu * mat_f[1:-1, :] / mat_mass[1:-2, :], axis=0)  # (acc-3, l)
    else:  # case of only one value of redshift to get the probability distribution of.
        mass = np.logspace(np.log10(M * frac), np.log10(M), acc)
        wf = delta_c(zf, om0, ol0) - w0
        S = sigma(mass, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos) ** 2 - S0
        S[-1] = 1e-10
        nu = wf / np.sqrt(S)
        if model == 'EC':
            f = f_ec(S + S0, S0, wf + w0, w0)
            ds = 0.5 * (S[2:] - S[:-2])
            return -M * np.sum(ds * f[1:-1] / mass[1:-1])

        elif model == 'sheth2002':
            f = np.zeros((acc, 1))
            for i in range(acc):
                b1 = Barrier(S[i] + S0, wf[i] + w0, alpha, beta, a)
                b2 = Barrier(S0, w0, alpha, beta, a)
                dB = b2 - b1
                T1 = np.abs(ngTaylor(S[i] + S0, S0, dB, order=order))
                f[i] = T1 * np.exp(-0.5 * (b1 - b2) ** 2 / S[i]) / (np.sqrt(2 * np.pi) * S[i] ** 1.5)
            ds = 0.5 * (S[2:] - S[:-2])
            return -M * np.sum(ds * f[1:-1] / mass[1:-1])
        else:
            f = fps(nu) / nu
            dnu = (nu[2:] - nu[:-2]) * 0.5
            return M * np.sum(dnu * f[1:-1] / mass[1:-1])


def M_integ_proba(masses, weights=None, zf=np.linspace(0, 7, 20), frac=0.5, acc=1000, zi=0.0, sig8=sigma8, h=h,
                  kmax=30, window='TopHat', prec=1000, om0=om, diff=False,
                  ol0=oml, omb=omb, camb=False, model='EC', colos=False):
    """
    Mass weighted cummulative probability of zf
    :param masses: list or np.array masses of halos to get the average zf
    :param weights: list, array or None  weights of the masses
    :param zf: float or array redshifts where to give probability
    See proba() function for the rest of the parameters
    :return:
    """
    res = []
    if not (type(weights) == np.ndarray or type(weights) == list):
        for mass in masses:
            if diff:
                prob = proba(mass, zf, frac, acc, zi, sig8, h, kmax, window, prec, om0, ol0, omb, camb,
                             model, colos)
                dz = zf[2:] - zf[:-2]
                res.append((prob[2:] - prob[:-2]) / dz)
            else:
                res.append(proba(mass, zf, frac, acc, zi, sig8, h, kmax, window, prec, om0, ol0, omb, camb,
                                 model, colos))
        ares = np.array(res)
        return np.sum(ares, axis=0) / len(masses)
    else:
        for i in range(len(masses)):
            mass = masses[i]
            w = weights[i] / np.sum(weights)
            if diff:
                prob = proba(mass, zf, frac, acc, zi, sig8, h, kmax, window, prec, om0, ol0, omb, camb,
                             model, colos)
                dz = zf[2:] - zf[:-2]
                res.append(-(prob[2:] - prob[:-2]) * w / dz)
            else:
                res.append(proba(mass, zf, frac, acc, zi, sig8, h, kmax, window, prec, om0, ol0, omb, camb,
                                 model, colos) * w)
        ares = np.array(res)
        return np.sum(ares, axis=0)


def median_formation(M, z, frac=0.5, acc=100, nzeds=10000, sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000,
                     om0=om, ol0=oml, omb=omb, model='EC', camb=False, colos=True, outc=False):
    """
    Calculates the median formation redshift of halos of mass M at redshift z, and gets the concentration if needed    :param M: float M
    :param z: float redshift
    :param outc: bool if True outputs concentration parameter estimation
    :return: float : z50 or c(z50)
    """
    if type(M) == list or type(M) == np.ndarray:
        raise TypeError("M should not be an array")
    zs = np.linspace(z + 0.1, 6 + z, nzeds)
    res = []
    for red in zs:
        res.append(proba(M, red, frac, acc, z, sig8, h, kmax, window, prec, om0, ol0, omb, camb, model, colos))
    res = np.array(res)
    zf = np.max(zs[res > 0.5])
    if outc:
        return 0.7 + 0.77 * np.log10(zf)
    else:
        return zf


def average_formation(M, z, frac=0.5, acc=100, sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000,
                      om0=om, ol0=oml, omb=omb, camb=False, colos=True, outc=False):
    # Gets the average z50 of a population of halos at mass M and redshift z
    if type(M) == list or type(M) == np.ndarray:
        raise TypeError("M should not be an array")

    zs = np.linspace(z + 1.2 * sig8 / (2.7 + 0.2 * np.log10(M) + 0.1 * om0), z + 8, acc)
    res = []
    for red in zs:
        res.append(proba(M, red, frac, acc, z, sig8, h, kmax, window, prec, om0, ol0, omb, camb, 'EC', colos))
    res = np.array(res)
    dens = (res[2:] - res[:-2]) / (zs[2:] - zs[:-2])
    dz = zs[1] - zs[0]
    lower = -dz * np.sum(zs[1:-1] * dens)
    deltap = (zs[0] - z) * dens[0]
    upper = lower - deltap
    if outc:
        return 0.7 + 0.77 * np.log10(lower)
    else:
        return [lower, upper]


def peak_formation(M, z, frac=0.5, acc=100, sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000,
                   om0=om, ol0=oml, omb=omb, camb=False, colos=True, outc=False):
    # Gets the redshift at which the z50 probability distribution peaks

    if type(M) == list or type(M) == np.ndarray:
        raise TypeError("M should not be an array")

    zs = np.linspace(z + 0.1, z + 6, acc)
    res = []
    for red in zs:
        res.append(proba(M, red, frac, acc, z, sig8, h, kmax, window, prec, om0, ol0, omb, camb, 'EC', colos))
    res = np.array(res)
    dens = -(res[2:] - res[:-2]) / (zs[2:] - zs[:-2])
    zf = zs[np.argmax(dens)]
    if outc:
        return 0.7 + 0.77 * np.log10(zf)
    else:
        return zf


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    zi = np.linspace(0, 1, 20)
    res1 = []
    res2 = []
    res3 = []
    for el in zi:
        res1.append(peak_formation(M=4e13, z=el, colos=True))
        res2.append(median_formation(M=4e13, z=el, colos=True))
        res3.append(average_formation(M=4e13, z=el, colos=True))
    plt.plot(zi, res1, label='Maximum probability')
    plt.plot(zi, res2, label='Mean')
    plt.plot(zi, res3, label='Average')
    plt.legend()
    plt.xlabel('$z_{0}$', size=15)
    plt.ylabel('$z_{formation}$', size=15)
    plt.show()

    # zs = np.linspace(0.1, 2, 50)
    # masses = [1e8, 1e11, 1e14]
    # for ms in masses:
    #     res = proba(ms, zf=zs, acc=400, prec=400, colos=False, model="EC")
    #     res2 = proba(ms, zf=zs, acc=400, prec=400, colos=True, model="EC")
    #     #dpdw = (res[2:] - res[:-2])/(zs[2:] - zs[:-2])
    #     #plt.plot(zs[1:-1], -dpdw, label='log M='+str(np.log10(ms)), linewidth=2.5)
    #     plt.plot(zs, res, label='log M='+str(np.log10(ms)))
    #     plt.plot(zs, res2, label='Colossus')
    # plt.legend(fontsize='large', fancybox=True)
    # plt.xlabel('z', size=25)
    # plt.ylabel('$P(z_f>z)$', size=20)
    # #plt.ylabel('$dP/dz$', size=20)
    # plt.xticks(size=18)
    # plt.yticks(size=18)
    # plt.show()

    # M = np.logspace(8, 14, 100)
    # zfs = np.linspace(0.05, 2, 50)
    # plt.plot(M, median_formation(1e13, z=0.1, colos=False))
    # plt.plot(zfs, proba(1e13, zfs, prec=500, acc=400))
    # print(proba(1e13, 0.3))
