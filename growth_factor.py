from __future__ import  division
from scipy import special
from cosmo_today import *
from cosmo_parameters import *


def growth_factor_pt(ol, om):
    """approx furmula for growth factor given by press and turner"""
    return (5*om/2)*1/(om**(4/7)-ol+(1+om/2)*(1+ol/70))


def ibeta(x, a, b):
    return special.beta(a, b)*special.betainc(a, b, x)

def growth_factor_z(z, N=50):
    """Calculate the linear growth factor as a function of redshift"""
    h = hubble_ratio(z, omega_l0, omega_m0, omega_r0)  #Calculating H(z)/H0
    ol = omega_l(omega_l0, h)        #Calculating the Dark Energy density at z
    om = omega_m(z, omega_m0, h)     #Matter density at z
    ok = 1-ol-om
    n = np.arange(N)
    tab = ((-1)**n*special.poch(1.5, n)/special.factorial(n))*(ok/(om**(2/3)*ol**(1/3)))**n*ibeta(ol/(om+ol),(5+2*n)/6,(2+2*n)/3)
    res = 5*om**(1/3)/(6*ol**(5/6))*np.sum(tab)
    return res

def growth(om, ol, N=50):
    """growth factor function in function of cosmo parameters only"""
    ok = 1-om-ol
    n = np.arange(N)
    tab = ((-1)**n*special.poch(1.5, n)/special.factorial(n))*(ok/(om**(2/3)*ol**(1/3)))**n*ibeta(ol/(om+ol),(5+2*n)/6,(2+2*n)/3)
    res = 5*om**(1/3)/(6*ol**(5/6))*np.sum(tab)
    return res

def D(z):
    """Normalised linear growth factor"""
    return growth_factor_z(z)/((1+z)*growth_factor_z(0))

def delta_c(z):
    """critical overdensity"""
    return 1.68/D(z)
