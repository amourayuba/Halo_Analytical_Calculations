from __future__ import  division
from cosmo_today import *
from cosmo_parameters import *

def W_th(k, R):
    """Smoothing window function in fourier space.
     Type : Top Hat smoothing"""
    return 3*(np.sin(k*R)-k*R*np.cos(k*R))/(k*R)**3

def W_gauss(k, R):
    """Smoothing window function in fourier space.
         Type : Gaussia,"""
    return np.exp(-(k*R)**2/20)

def W_ksharp(k, R):
    """Smoothing window function in fourier space.
         Type : Sharp k cutoff"""
    return (k*R <= 1) + 0

def sigma(M, k, pk, window = W_gauss):
    """
    :param M: Mass, could be an nd.array
    :param k: wavenumbers for calculating the integral nd array
    :param pk: power spectrum values nd array same size as k
    :param window: function. Smoothing window function
    :return: nd array : values of the rms of the smoothed density field for the mass array entered
    """
    R = (3*np.pi*M/(4*rho_c))**(1/3)      #In units of Mpc/h
    n = len(R)      #size of Mass imput
    m = len(k)      #size of wavenumbers imput

    Rmat = np.array([R]*m)   #Duplicating the R 1D array to get a n*m size matric
    kmat = np.array([k]*n).transpose()  #Dupicating the k and pk 1D arrays to get n*m size matrix
    pkmat = np.array([pk]*n).transpose()

    winres = window(kmat, Rmat)    #Calculating every value of the window function without a boucle
    dk = (np.max(k)-np.min(k))/len(k)  #element of k for approximating the integral
    res = pkmat*kmat**2*winres         #Values inside the integral foreach k
    integ = np.sum(res, axis=0)*dk     # approximate evaluation of the integral through k.

    return integ

