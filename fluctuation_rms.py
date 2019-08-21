from __future__ import  division
from power_spectrum_analytic import *
from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology('planck15');


def W_th(k, R):
    """Smoothing window function in fourier space.
     Type : Top Hat smoothing"""
    return 3*(np.sin(k*R)-k*R*np.cos(k*R))/(k*R)**3

def W_gauss(k, R):
    """Smoothing window function in fourier space.
         Type : Gaussia,"""
    return np.exp(-(k*R)**2/2)

def W_ksharp(k, R):
    """Smoothing window function in fourier space.
         Type : Sharp k cutoff"""
    return (k*R <= 1) + 0


def power_spectrum(k, sigma8=sigma8, h=h, om0=om, omb=omb, ns=ns, test=False):
    """
    Power spectrum from analytical approx of Eisenstein and Hu
    :param k: float: wavenumber
    :param sigma8: float: sigma 8 cosmo param
    :param h: float : H0/100 : cosmo parameter
    :param om0: faction matter density
    :param omb: float : fraction baryon density
    :param ns: float : initial spectrum power law index
    :param test: bool : used to get out of the recursive loop for the sigma8 normalisation
    :return: float : power spectrum
    """
    if test:
        return k**ns*Transfer(k*h, om0, omb, h)**2  #getting a first value
    else:
        val_8 = sigma_R(8, sigma8, h, omb, om0, test=True)   #getting the value of sigma8
        return (sigma8/val_8)**2*k**ns*Transfer(k*h, om0, omb, h)**2  #normalising by the value of sigma8 we want


def Delta(k, sigma8=sigma8, h=h, om0=om, omb=omb, ns=ns):
    """
    dimenstionless power spectrum function
    :param k: float: wavenumber
    :param sigma8: float: sigma 8 cosmo param
    :param h: float : H0/100 : cosmo parameter
    :param om0: faction matter density
    :param omb: float : fraction baryon density
    :param ns: float : initial spectrum power law index
    :param test: bool : used to get out of the recursive loop for the sigma8 normalisation
    :return: float : power spectrum
    """
    return np.sqrt(k**3*power_spectrum(k, sigma8, h, om0, omb, ns)/(2*np.pi**2))


def camb_power_spectrum(h=h, ombh2=ombh2, omch2=omch2, ns=ns, sig8=sigma8, kmin = 2e-5, kmax=100, linear=True,
                        npoints=1000, nonlinear=False, omk=0.0, cosmomc_theta=None, thetastar=None,
                        neutrino_hierarchy='degenerate', num_massive_neutrinos=1, mnu=0.06, nnu=3.046, YHe=None,
                        meffsterile=0.0, standard_neutrino_neff=3.046, TCMB=2.7255,tau=None, deltazrei=None, Alens=1.0,
                        bbn_predictor=None, theta_H0_range=[10, 100]):
    """Provides the power spectrum using CAMB code
    Variables :
    cosmological parameters in camb.set_cosmo (see camp.set_cosmo for reference
    ns : float initial power spectrum index
    kmax : max value of wave number for calculating the power spectrum
    nonlinear/linear : Boolean, if wanting the nonlinear/linear power spectrum for result
    npoints : number of desired points for the power spectrum restult"""
    import camb
    from camb import model
    pars = camb.CAMBparams()
    pars.set_cosmology(100*h, ombh2, omch2, omk, cosmomc_theta, thetastar, neutrino_hierarchy,
                       num_massive_neutrinos, mnu, nnu, YHe, meffsterile, standard_neutrino_neff, TCMB,
                       tau, deltazrei, Alens, bbn_predictor, theta_H0_range)

    pars.InitPower.set_params(ns=ns)
    pars.set_matter_power(redshifts=[0], kmax=kmax)
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    csig8 = results.get_sigma8()
    norm = (sig8/csig8) ** 2
    if not nonlinear:
        kh, zs, pk = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints=npoints)
        return kh, zs, norm*pk[0]
    elif not linear:
        pars.NonLinear = model.NonLinear_both
        results.calc_power_spectra(pars)
        kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints=npoints)
        return kh_nonlin, z_nonlin, norm*pk_nonlin[0]
    else:
        kh, zs, pk = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints=npoints)

        pars.NonLinear = model.NonLinear_both
        results.calc_power_spectra(pars)
        kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints=npoints)
        return ([kh, zs, norm*pk[0]], [kh_nonlin, z_nonlin, norm*pk_nonlin[0]])


def sigma_R(R, sig8=sigma8, h=h, omb=omb, om0=om, ol0 = oml, ns=ns, kmax=30,  prec=1000, window=W_th, camb=False,
            test=False, Colos=False):
    """
    fluctuation rms of smoothed field with a scale R
    :param R: float: smoothing scale
    :param sig8: float: sigma 8 cosmological parameter
    :param h: float : H0/100 cosmo parameter
    :param omb: float : baryon fraction density
    :param om0: float : matter fraction density
    :param ol0: float : dark energy fraction density
    :param ns: float : initial power spectrum power law index
    :param kmax: float : maximum wavenumber to get power spectrum. default : 30. useful for camb spectrum
    :param prec: float : number of bins of k or integral
    :param window: function : window function for smoothing
    :param camb: boolean : using camb spectrum or analytical approx of eisenstein and hu
    :param test: boolean : used to get out of the recursive loop for the sigma8 normalisation
    :return: float : fluctuation rms at scale R
    """
    if Colos:
        my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': om0, 'Ode0': ol0, 'Ob0': omb, 'sigma8': sig8, 'ns': ns}
        cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
        return cosmo.sigma(R, 0.0)
    else:
        if camb:
            ombh2 = omb * h ** 2
            omch2 = (om0 - omb) * h ** 2
            k, z, pk = camb_power_spectrum(h=h, kmax=kmax, ombh2=ombh2, omch2=omch2, sig8=sig8, npoints=prec,
                                           ns=ns, omk=1-om0-ol0, nonlinear=False, linear=True)
            if type(R) == list or type(R) == np.ndarray:
                # In units of Mpc/h
                n = len(R)  # size of Mass imput
                m = prec  # size of wavenumbers imput
                Rmat = np.array([R] * m)  # Duplicating the R 1D array to get a n*m size matric
                kmat = np.array([k] * n).transpose()  # Dupicating the k and pk 1D arrays to get n*m size matrix
                pkmat = np.array([pk] * n).transpose()

                winres = window(kmat, Rmat)  # Calculating every value of the window function without a boucle
                dlk = np.log(np.max(k) / np.min(k)) / len(k)  # element of k for approximating the integral
                res = pkmat * kmat ** 3 * winres ** 2  # Values inside the integral foreach k
                integ = np.sum(res, axis=0) * dlk  # approximate evaluation of the integral through k.
                return np.sqrt(integ / (2 * np.pi ** 2))
            else:
                winres = window(k, R)
                dlk = np.log(np.max(k) / np.min(k)) / len(k)  # element of k for approximating the integral
                res = pk*k**3*winres**2
                integ = np.sum(res)*dlk
                return np.sqrt(integ / (2 * np.pi ** 2))
        else:
            prec=10*prec   # having more bins is less expensive than in the camb power spectrum case
            k = np.logspace(-7, 5, prec)  #creating a k array in log space for the integration
            pk = power_spectrum(k, sig8, h, om0, omb, ns, test)  #corresponding power spectrum values

            if type(R) == list or type(R) == np.ndarray: #for the code to be more robust, we seprate unique R or arrays of R
                # In units of Mpc/h
                n = len(R)  # size of Mass imput
                m = prec  # size of wavenumbers imput
                Rmat = np.array([R] * m)  # Duplicating the R 1D array to get a n*m size matric
                kmat = np.array([k] * n).transpose()  # Dupicating the k and pk 1D arrays to get n*m size matrix
                pkmat = np.array([pk] * n).transpose()

                winres = window(kmat, Rmat)  # Calculating every value of the window function without a boucle
                dlk = np.log(np.max(k) / np.min(k)) / len(k)  # element of k for approximating the integral
                res = pkmat * kmat ** 3 * winres ** 2  # Values inside the integral foreach k
                integ = np.sum(res, axis=0) * dlk  # approximate evaluation of the integral through k.
                return np.sqrt(integ / (2 * np.pi ** 2))
            else:
                winres = window(k, R)
                dlk = np.log(np.max(k) / np.min(k)) / len(k)  # element of k for approximating the integral
                res = pk * k ** 3 * winres ** 2
                integ = np.sum(res) * dlk
                return np.sqrt(integ / (2 * np.pi ** 2))

def sigma(x, sig8=sigma8, h=h, kmax=30, window='TopHat', xin='M', prec=1000, om0=om, ol0=oml, omb=omb,
          camb=False, Colos=False):
    """
    fluctuation rms of smoothed field with a scale R or of a mass M
    :param x : float: either M mass in Msun/h or smoothin scale R in Mpc/h
    :param xin : string : either 'R' or 'M' if the variable is smoothing scale or Mass
    :param sig8: float: sigma 8 cosmological parameter
    :param h: float : H0/100 cosmo parameter
    :param omb: float : baryon fraction density
    :param om0: float : matter fraction density
    :param ol0: float : dark energy fraction density
    :param ns: float : initial power spectrum power law index
    :param kmax: float : maximum wavenumber to get power spectrum. default : 30. useful for camb spectrum
    :param prec: float : number of bins of k or integral
    :param window: function : window function for smoothing
    :param camb: boolean : using camb spectrum or analytical approx of eisenstein and hu
    :param test: boolean : used to get out of the recursive loop for the sigma8 normalisation
    :return: float : fluctuation rms at scale R or at mass M
    """

    if xin == 'R':
        if window == 'TopHat':
            return sigma_R(x, sig8, h, omb, om0, ol0, ns, kmax, prec, W_th, camb, Colos=Colos)
        elif window == 'Gauss':
            return sigma_R(x, sig8, h, omb, om0, ol0, ns, kmax, prec, W_gauss, camb, Colos=Colos)
        elif window == 'k-Sharp':
            return sigma_R(x, sig8, h, omb, om0, ol0, ns, kmax, prec, W_ksharp, camb, Colos=Colos)
    elif xin == 'M':
        if window == 'TopHat':
            R = (3 *x / (4 *np.pi*rho_m(0, om0))) ** (1 / 3)
            return sigma_R(R, sig8, h, omb, om0, ol0, ns, kmax, prec, W_th, camb, Colos=Colos)
        elif window == 'Gauss':
            R = (x/rho_m(0, om0))**(1/3)/np.sqrt(np.pi)
            return sigma_R(R, sig8, h, omb, om0, ol0, ns, kmax, prec, W_gauss, camb, Colos=Colos)
        elif window == 'k-Sharp':
            R = (x/(6*np.pi**2*rho_m(0, om0)))**(1/3)
            return sigma_R(R, sig8, h, omb, om0, ol0, ns, kmax, prec, W_ksharp, camb, Colos=Colos)





#############################-------------------Comparison with Colossus-----------------###############################



'''import matplotlib.pyplot as plt
R = 10**np.arange(-3 , 2.4, 0.05)
sigma_tophat = cosmo.sigma(R, 0.0)
sigma_sharpk = cosmo.sigma(R, 0.0, filt = 'sharp-k')
sigma_gaussian = cosmo.sigma(R, 0.0, filt = 'gaussian')
y1 = sigma_R(R, window=W_gauss, kmax=1000)
y2 = sigma_R(R, window= W_th, kmax=1000)
y3 = sigma_R(R, window= W_ksharp, kmax = 1000)
plt.figure()
plt.loglog()
plt.xlabel('R [Mpc/]')
plt.ylabel('$\sigma(R)$')
plt.plot(R, sigma_tophat, '--b', label = 'Colossus tophat')
plt.plot(R, sigma_sharpk, '--k', label = 'Colossus sharp-k')
plt.plot(R, sigma_gaussian, '--g', label = 'Colossus gaussian')
plt.plot(R, y2, '-b', label = 'CAMB tophat')
plt.plot(R, y3, '-k', label = 'CAMB sharp-k')
plt.plot(R, y1, '-g', label = 'CAMB gaussian')
plt.legend()
plt.show()'''



'''import matplotlib.pyplot as plt
R = 10**np.arange(-3 , 2.4, 0.005)
yc2 = cosmo.sigma(R, 0.0)
yc3 = cosmo.sigma(R, 0.0, filt = 'sharp-k')
yc1 = cosmo.sigma(R, 0.0, filt = 'gaussian')
y1 = sigma_R(R, sig8=0.8159, window=W_gauss, kmax=1000)
y2 = sigma_R(R, window=W_th, sig8 = 0.8159, kmax=1000)
y3 = sigma_R(R, window=W_ksharp, sig8 = 0.8159, kmax=1000)
plt.xlabel('R [Mpc/h]', size = 15)
plt.ylabel('$\sigma_{CAMB}-\sigma_{col}/\sigma_{col}$', size = 15)
plt.plot(R, (y2-yc2)/yc2, '-b', label = 'tophat')
#plt.plot(R, (y3-yc3)/yc3, '-k', label = 'sharp-k')
plt.plot(R, (y1-yc1)/yc1, '-g', label = 'gaussian')
plt.xlim(0.01, 200)
plt.ylim(-0.02, 0.02)
plt.xscale('log')
plt.legend()
plt.show()'''


#################------------------------Evolution with redshift comparison ------------------#################

'''import matplotlib.pyplot as plt
z = [0, 1, 2, 4]
R = 10**np.arange(-2, 2.4, 0.005)
for el in z:
    yc2 = cosmo.sigma(R, el)
    y2 = sigma_R(R, sig8=0.8159, window=W_th, kmax=400)
    plt.loglog(R, y2, label= 'z='+str(el)+' CAMB')
    plt.loglog(R, yc2, '--', label = 'Colossus')
plt.xlabel('R [Mpc/h]', size=15)
plt.ylabel('$\sigma(R)$', size=15)
plt.legend()
plt.show()'''

#############------------------------Evolution with sigma8----------------------##############################

'''import matplotlib.pyplot as plt
s8 = np.arange(0.4, 1.6, 0.2)
R = 10**np.arange(-2, 2.4, 0.005)
for el in s8:
    my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': om, 'Ode0': oml, 'Ob0': 0.048, 'sigma8':el, 'ns': ns}
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
    yc2 = cosmo.sigma(R, 0)
    y2 = sigma_R(R, sig8= el, window=W_th, kmax=400)
    plt.loglog(R, y2, label= '$\sigma_8$ = '+str(round(el, 1))+' Analytical')
    plt.loglog(R, yc2, '--', label = 'Colossus')
plt.xlabel('R [Mpc/h]', size=15)
plt.ylabel('$\sigma(R)$', size=15)
plt.legend()
plt.show()'''


#############-------------------Evolution with omega_m-----------------------------###########################

'''import matplotlib.pyplot as plt
omg = np.arange(0.1, 0.9, 0.2)
R = 10**np.arange(-2, 2.4, 0.005)
for el in omg:
    my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': el, 'Ode0': oml, 'Ob0': 0.048, 'sigma8': 0.8, 'ns': ns}
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
    yc2 = cosmo.sigma(R, 0)
    y2 = sigma_R(R, om0= el, ol0=1-el, sig8 =  0.8, window=W_th, kmax=400)
    plt.loglog(R, y2, label= '$\Omega_m$ = '+str(round(el, 1))+' Analytical')
    plt.loglog(R, yc2, '--', label = 'Colossus')
plt.xlabel('R [Mpc/h]', size=15)
plt.ylabel('$\sigma(R)$', size=15)
plt.legend()
plt.show()'''



'''import matplotlib.pyplot as plt
omg = np.arange(0.1, 0.7, 0.1)
k = np.logspace(-3, 2, 1000)
for el in omg:
    my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': el, 'Ode0': 1-el, 'Ob0': cosmo.Ob0, 'sigma8': 0.8102, 'ns': 0.9665,'relspecies' : True, 'Tcmb0' : 2.7255, 'Neff' : 3.0460}
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
    h = cosmo.H0 / 100
    ombh2 = cosmo.Ob0 * h ** 2
    omh2 = el * h ** 2
    omch2 = omh2 - ombh2
    kh, z, pk = camb_power_spectrum(z=[0], h=h, kmin= 1e-3, kmax=200, ombh2=ombh2, omch2=omch2, sig8=0.8102, npoints=1000, ns=ns, omk=0, nonlinear=False, linear=True)
    yc2 = cosmo.matterPowerSpectrum(kh, 0)
    #plt.loglog(kh, pk[0], label= '$\Omega_m$ = '+str(round(el, 1))+' CAMB')
    #plt.loglog(k, yc2, '--', label = 'Colossus')
    plt.plot(kh, (yc2-pk)/pk,  label= '$\Omega_m$ = '+str(round(el, 1)))
plt.xlabel('R [Mpc/h]', size=15)
plt.ylabel('$\Delta$P(k)', size=15)
plt.xscale('log')
plt.legend()
plt.show()'''

