from __future__ import  division
import numpy as np
from scipy.integrate import quad
from cosmo_parameters import *
from power_spectrum_analytic import *


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

def sigma_camb(M, k, pk, window = W_gauss):
    """
    :param M: Mass, could be an nd.array
    :param k: wavenumbers for calculating the integral nd array
    :param pk: power spectrum values nd array same size as k
    :param window: function. Smoothing window function
    :return: nd array : values of the rms of the smoothed density field for the mass array entered
    """
    R = (3*np.pi*M/(4*rho_c*0.3))**(1/3)      #In units of Mpc/h
    n = len(R)      #size of Mass imput
    m = len(k)      #size of wavenumbers imput

    Rmat = np.array([R]*m)   #Duplicating the R 1D array to get a n*m size matric
    kmat = np.array([k]*n).transpose()  #Dupicating the k and pk 1D arrays to get n*m size matrix
    pkmat = np.array([pk]*n).transpose()

    winres = window(kmat, Rmat)    #Calculating every value of the window function without a boucle
    dk = (np.max(k)-np.min(k))/len(k)  #element of k for approximating the integral
    res = pkmat*kmat**2*winres         #Values inside the integral foreach k
    integ = np.sum(res, axis=0)*dk     # approximate evaluation of the integral through k.
    return np.sqrt(integ/(2*np.pi**2))



def sigma_a_M(M, window='Gauss', z=0, camb_ps = False, sig8=0.8, h=h, om0=om, omb= omb):
    """Provides the rms of the linear density field as a function of mass.
    Uses power_spectrum_a() function from power_spectrum_analytic.py. And a bunch of cosmo parameters from that file.
    :param M: array or list of Mass for which we want the rms (Msun/h)
    :param window: either 'TopHat', 'Gaussian' or 'k-Sharp' type of window function
    :param z: redshift. Default = 0
    :return: list of arrays containting the rms result and absolute error
    """
    if window == 'Gauss':
        res = []
        R = (3*M / (4*np.pi*rho_m(z, om0))) ** (1 / 3) #/ np.sqrt(2 * np.pi)
        for rad in R:
            def f(k):
                return k ** 2 * power_spectrum_a(k, z, camb_ps, sig8, h, om0, omb) * W_gauss(k, rad)**2
            res.append(np.sqrt(quad(f, 0, np.inf)[0] / (2 * np.pi ** 2)))
        return np.array(res)

    elif window == 'TopHat':
        res = []
        R = (3 * M / (4 * np.pi * rho_m(z, om0))) ** (1 / 3)
        for rad in R:
            def f(k):
                return k ** 2 * power_spectrum_a(k, z, camb_ps, sig8, h, om0, omb) * W_th(k, rad)**2
            res.append(np.sqrt(quad(f, 0, np.inf)[0] / (2*np.pi**2)))
        return np.array(res)

    elif window == 'k-Sharp':
        res = []
        R = (3*M/(4*rho_m(z, om0)*np.pi))**(1/3)
        for rad in R:
            def f(k):
                return k ** 2 * power_spectrum_a(k, z, camb_ps, sig8, h, om0, omb) * W_ksharp(k, rad)**2
            res.append(np.sqrt(quad(f, 0, np.inf)[0]/(2*np.pi**2)))
        return np.array(res)
    else:
        return ValueError('window argument has to be either Gauss TopHat or k-Sharp')


def sigma_a_R(R, window='Gauss', z=0, camb_ps = False, sig8=0.8, h=h, om0=om, omb= omb):
    """Provides the rms of the linear density field as a function of mass
    :param R: array or list of scales for which we want the rms (Mpc/h)
    :param window:  either 'TopHat', 'Gaussian' or 'k-Sharp' type of window function
    :param z:redshift. Default = 0
    :return: list of arrays containting the rms result and absolute error
    """
    if type(R) == np.ndarray or type(R) == list:
        if window == 'Gauss':
            res = []
            for rad in R:
                def f(k):
                    return k ** 2 * power_spectrum_a(k, z, camb_ps, sig8, h, om0, omb) * W_gauss(k, rad)**2
                res.append(np.sqrt(quad(f, 0, np.inf)[0]/(2*np.pi**2)))
            return np.array(res)

        elif window == 'TopHat':
            res = []
            for rad in R:
                def f(k):
                    return k ** 2 * power_spectrum_a(k, z, camb_ps, sig8, h, om0, omb) * W_th(k, rad)**2
                res.append(np.sqrt(quad(f, 0, np.inf)[0]/(2*np.pi**2)))
            return np.array(res)

        elif window == 'k-Sharp':
            res = []
            for rad in R:
                def f(k):
                    return k ** 2 * power_spectrum_a(k, z, camb_ps, sig8, h, om0, omb) * W_ksharp(k, rad)**2
                res.append(np.sqrt(quad(f, 0, np.inf)[0]/(2*np.pi**2)))
            return np.array(res)
        else:
            return ValueError('window argument has to be either Gauss TopHat or k-Sharp')
    else:
        if window == 'Gauss':
            def f(k):
                return k ** 2 * power_spectrum_a(k, z, camb_ps, sig8, h, om0, omb) * W_gauss(k, R) ** 2
            return np.sqrt(quad(f, 0, np.inf)[0] / (2 * np.pi ** 2))

        elif window == 'TopHat':
            def f(k):
                return k ** 2 * power_spectrum_a(k, z, camb_ps, sig8, h, om0, omb) * W_th(k, R) ** 2
            return np.sqrt(quad(f, 0, np.inf)[0] / (2 * np.pi ** 2))

        elif window == 'k-Sharp':
            def f(k):
                return k ** 2 * power_spectrum_a(k, z, camb_ps, sig8, h, om0, omb) * W_ksharp(k, R) ** 2
            return np.sqrt(quad(f, 0, np.inf)[0] / (2 * np.pi ** 2))
        else:
            return ValueError('window argument has to be either Gauss TopHat or k-Sharp')


'''omv = np.linspace(0.01, 0.6, 30)
#olv = 1-omv
ombv = 0.13*omv
sig8 = np.linspace(0.1, 1.5, 30)
nom = np.zeros((30,30))
mt = np.logspace(13,16, 100)
for i in range(30):
    for j in range(30):
        nom[i,j] = np.sum(sigma_a_M(mt, sig8=sig8[j], om0=omv[i], omb=ombv[i]))#nom[i,j] = integrated_hmf(1, 16, prec = 50, sigma8=sig8[j], om0=omv[i])
plt.contourf(omv, sig8, np.log(nom+1), levels=30, cmap='RdGy')
plt.xlabel('$\Omega_m$')
plt.ylabel('$\sigma_8$')
plt.colorbar()
plt.show()'''

