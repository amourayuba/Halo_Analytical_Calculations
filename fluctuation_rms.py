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



def sigma_a_M(M, window='Gauss', z=0, camb_ps = False, sigma8=1, H0=67.5, ombh2=0.022, omch2=0.122, ns=0.965, kmax=10):
    """Provides the rms of the linear density field as a function of mass.
    Uses power_spectrum_a() function from power_spectrum_analytic.py. And a bunch of cosmo parameters from that file.
    :param M: array or list of Mass for which we want the rms (Msun/h)
    :param window: either 'TopHat', 'Gaussian' or 'k-Sharp' type of window function
    :param z: redshift. Default = 0
    :return: list of arrays containting the rms result and absolute error
    """
    if window == 'Gauss':
        res = []
        R = (3*M / (4*np.pi*rho_m(z))) ** (1 / 3) #/ np.sqrt(2 * np.pi)
        for rad in R:
            def f(k):
                return k ** 2 * power_spectrum_a(k, delta_H, z, camb_ps, sigma8) * W_gauss(k, rad)**2
            res.append(np.sqrt(quad(f, 0, np.inf)[0] / (2 * np.pi ** 2)))
        return np.array(res)

    elif window == 'TopHat':
        res = []
        R = (3 * M / (4 * np.pi * rho_m(z))) ** (1 / 3)
        for rad in R:
            def f(k):
                return k ** 2 * power_spectrum_a(k, delta_H, z, camb_ps, sigma8)  * W_th(k, rad)**2
            res.append(np.sqrt(quad(f, 0, np.inf)[0] / (2*np.pi**2)))
        return np.array(res)

    elif window == 'k-Sharp':
        res = []
        R = (3*M/(4*rho_m(z)*np.pi))**(1/3)
        for rad in R:
            def f(k):
                return k ** 2 * power_spectrum_a(k, delta_H, z, camb_ps, sigma8)*W_ksharp(k, rad)**2
            res.append(np.sqrt(quad(f, 0, np.inf)[0]/(2*np.pi**2)))
        return np.array(res)
    else:
        return ValueError('window argument has to be either Gauss TopHat or k-Sharp')


def sigma_a_R(R, window='Gauss', z=0, camb_ps = False, sigma8=1, H0=67.5, ombh2=0.022, omch2=0.122, ns=0.965, kmax=10):
    """Provides the rms of the linear density field as a function of mass
    :param R: array or list of scales for which we want the rms (Mpc/h)
    :param window:  either 'TopHat', 'Gaussian' or 'k-Sharp' type of window function
    :param z:redshift. Default = 0
    :return: list of arrays containting the rms result and absolute error
    """

    if window == 'Gauss':
        res = []
        for rad in R:
            def f(k):
                return k ** 2 * power_spectrum_a(k, delta_H, z, camb_ps, sigma8)* W_gauss(k, rad)**2
            res.append(np.sqrt(quad(f, 0, np.inf)[0]/(2*np.pi**2)))
        return np.array(res)

    elif window == 'TopHat':
        res = []
        for rad in R:
            def f(k):
                return k ** 2 * power_spectrum_a(k, delta_H, z, camb_ps, sigma8)* W_th(k, rad)**2
            res.append(np.sqrt(quad(f, 0, np.inf)[0]/(2*np.pi**2)))
        return np.array(res)

    elif window == 'k-Sharp':
        res = []
        for rad in R:
            def f(k):
                return k ** 2 * power_spectrum_a(k, delta_H, z, camb_ps, sigma8)* W_ksharp(k, rad)**2
            res.append(np.sqrt(quad(f, 0, np.inf)[0]/(2*np.pi**2)))
        return np.array(res)
    else:
        return ValueError('window argument has to be either Gauss TopHat or k-Sharp')
