from __future__ import  division
import numpy as np
from scipy.integrate import quad
from cosmo_parameters import *
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


def sigma_a_M(M, window='TopHat', z=0, sig8=0.8, h=h, om0=om, omb= omb):
    """Provides the rms of the linear density field as a function of mass.
    Uses power_spectrum_a() function from power_spectrum_analytic.py. And a bunch of cosmo parameters from that file.
    :param M: array or list of Mass for which we want the rms (Msun/h)
    :param window: either 'TopHat', 'Gaussian' or 'k-Sharp' type of window function
    :param z: redshift. Default = 0
    :return: list of arrays containting the rms result and absolute error
    """
    if type(M) == np.ndarray or type(M) == list:
        if window == 'Gauss':
            res = []
            R = (M / (np.pi*rho_m(z, om0))) ** (1 / 3)/np.sqrt(2 * np.pi)
            for rad in R:
                def f(k):
                    return k ** 2 * power_spectrum_a2(k, z, sig8, h, om0, omb) * W_gauss(k, rad)**2
                res.append(np.sqrt(quad(f, 0, np.inf)[0] / (2 * np.pi ** 2)))
            return np.array(res)

        elif window == 'TopHat':
            res = []
            R = (3 * M / (4 * np.pi * rho_m(z, om0))) ** (1 / 3)
            for rad in R:
                def f(k):
                    return k ** 2 * power_spectrum_a2(k, z, sig8, h, om0, omb) * W_th(k, rad)**2
                res.append(np.sqrt(quad(f, 0, np.inf)[0] / (2*np.pi**2)))
            return np.array(res)

        elif window == 'k-Sharp':
            res = []
            R = (M/ (6 * np.pi **2* rho_m(z, om0))) ** (1 / 3)
            for rad in R:
                def f(k):
                    return k ** 2 * power_spectrum_a2(k, z, sig8, h, om0, omb) * W_ksharp(k, rad)**2
                res.append(np.sqrt(quad(f, 0, np.inf)[0]/(2*np.pi**2)))
            return np.array(res)
        else:
            return ValueError('window argument has to be either Gauss TopHat or k-Sharp')
    else:
        if window == 'Gauss':
            R = (1/np.sqrt(2*np.pi))*(M / (np.pi * rho_m(z, om0))) ** (1 / 3)
            def f(k):
                return k ** 2 * power_spectrum_a2(k, z, sig8, h, om0, omb) * W_gauss(k, R) ** 2
            return np.sqrt(quad(f, 0, np.inf)[0] / (2 * np.pi ** 2))

        elif window == 'TopHat':
            R = (3 * M / (4 * np.pi * rho_m(z, om0))) ** (1 / 3)
            def f(k):
                return k ** 2 * power_spectrum_a2(k, z, sig8, h, om0, omb) * W_th(k, R) ** 2
            return np.sqrt(quad(f, 0, np.inf)[0] / (2 * np.pi ** 2))

        elif window == 'k-Sharp':
            R = (M/ (6 * np.pi **2* rho_m(z, om0))) ** (1 / 3)
            def f(k):
                return k ** 2 * power_spectrum_a2(k, z, sig8, h, om0, omb) * W_ksharp(k, R) ** 2
            return np.sqrt(quad(f, 0, np.inf)[0] / (2 * np.pi ** 2))
        else:
            return ValueError('window argument has to be either Gauss TopHat or k-Sharp')

def sigma_a_R(R, window='Gauss', z=0, sig8=0.8, h=h, om0=om, omb= omb):
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
                    return k ** 2 * power_spectrum_a2(k, z, sig8, h, om0, omb) * W_gauss(k, rad)**2
                res.append(np.sqrt(quad(f, 0, np.inf)[0]/(2*np.pi**2)))
            return np.array(res)

        elif window == 'TopHat':
            res = []
            for rad in R:
                def f(k):
                    return k ** 2 * power_spectrum_a2(k, z, sig8, h, om0, omb) * W_th(k, rad)**2
                res.append(np.sqrt(quad(f, 0, np.inf)[0]/(2*np.pi**2)))
            return np.array(res)

        elif window == 'k-Sharp':
            res = []
            for rad in R:
                def f(k):
                    return k ** 2 * power_spectrum_a2(k, z, sig8, h, om0, omb) * W_ksharp(k, rad)**2
                res.append(np.sqrt(quad(f, 0, np.inf)[0]/(2*np.pi**2)))
            return np.array(res)
        else:
            return ValueError('window argument has to be either Gauss TopHat or k-Sharp')
    else:
        if window == 'Gauss':
            def f(k):
                return k ** 2 * power_spectrum_a2(k, z, sig8, h, om0, omb) * W_gauss(k, R) ** 2
            return np.sqrt(quad(f, 0, np.inf)[0] / (2 * np.pi ** 2))

        elif window == 'TopHat':
            def f(k):
                return k ** 2 * power_spectrum_a2(k, z, sig8, h, om0, omb) * W_th(k, R) ** 2
            return np.sqrt(quad(f, 0, np.inf)[0] / (2 * np.pi ** 2))

        elif window == 'k-Sharp':
            def f(k):
                return k ** 2 * power_spectrum_a2(k, z, sig8, h, om0, omb) * W_ksharp(k, R) ** 2
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

'''R = 10**np.arange(-3 , 2.4, 0.005)
sigma_tophat = cosmo.sigma(R, 0.0)
sigma_sharpk = cosmo.sigma(R, 0.0, filt = 'sharp-k')
sigma_gaussian = cosmo.sigma(R, 0.0, filt = 'gaussian')
y1 = sigma_a_R(R)
y2 = sigma_a_R(R, window='TopHat')
y3 = sigma_a_R(R, window='k-Sharp')
plt.figure()
plt.loglog()
plt.xlabel('R(Mpc/h)')
plt.ylabel('sigma(R)')
plt.plot(R, sigma_tophat, '--b', label = 'Colossus tophat')
plt.plot(R, sigma_sharpk, '--k', label = 'Colossus sharp-k')
plt.plot(R, sigma_gaussian, '--g', label = 'Colossus gaussian')
plt.plot(R, y2, '-b', label = 'Yuba tophat')
plt.plot(R, y3, '-k', label = 'Yuba sharp-k')
plt.plot(R, y1, '-g', label = 'Yuba gaussian')
plt.legend()
plt.show()'''



'''R = 10**np.arange(-3 , 2.4, 0.005)
yc2 = cosmo.sigma(R, 0.0)
yc3 = cosmo.sigma(R, 0.0, filt = 'sharp-k')
yc1 = cosmo.sigma(R, 0.0, filt = 'gaussian')
y1 = sigma_a_R(R, sig8=0.8159)
y2 = sigma_a_R(R, window='TopHat', sig8 = 0.8159)
y3 = sigma_a_R(R, window='k-Sharp', sig8 = 0.8159)
plt.xlabel('R [Mpc/h]', size = 15)
plt.ylabel('$\sigma_{yuba}-\sigma_{col}/\sigma_{col}$', size = 15)
plt.plot(R, (y2-yc2)/yc2, '-b', label = 'tophat')
#plt.plot(R, (y3-yc3)/yc3, '-k', label = 'sharp-k')
plt.plot(R, (y1-yc1)/yc1, '-g', label = 'gaussian')
#plt.xlim(0.001, 200)
plt.ylim(-0.12, 0.05)
plt.xscale('log')
plt.legend()
plt.show()'''