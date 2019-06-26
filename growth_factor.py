from __future__ import  division
from scipy import special
from cosmo_parameters import *


def growth_factor_pt(ol, om):
    """approx formula for growth factor given by carroll press and turner(1992)"""
    return (5*om/2)/(om**(4/7)-ol+(1+om/2)*(1+ol/70))


def ibeta(x, a, b):
    return special.beta(a, b)*special.betainc(a, b, x)

def growth_factor_z(z, om0, ol0, ok, N=50):
    """Calculate the linear growth factor as a function of redshift"""
    or0 = 1-om0-ol0-ok  #Calculating H(z)/H0
    ol = omega_l(ol0, h)        #Calculating the Dark Energy density at z
    om = omega_m(z, om0, ol0, )     #Matter density at z
    ok = 1-ol-om
    if ok == 0:
        return (5*om/2)*1/(om**(4/7)-ol+(1+om/2)*(1+ol/70))
    else:
        n = np.arange(N)
        tab = ((-1)**n*special.poch(1.5, n)/special.factorial(n))*(ok/(om**(2/3)*ol**(1/3)))**n*ibeta(ol/(om+ol),(5+2*n)/6,(2+2*n)/3)
        res = 5*om**(1/3)/(6*ol**(5/6))*np.sum(tab)
        return res

def growth(z, om0, ol0):
    """Calculate the linear growth factor as a function of redshift"""
    ol = omega_l(z, ol0, om0)        #Calculating the Dark Energy density at z
    om = omega_m(z, om0, ol0)     #Matter density at z
    return (5*om/2)*1/(om**(4/7)-ol+(1+om/2)*(1+ol/70))

def D(z, om0, ol0):
    """Normalised linear growth factor"""
    return growth(z, om0, ol0)/((1+z)*growth(0, om0, ol0))

def delta_c(z, om0=om, ol0=oml):
    """critical overdensity"""
    return 1.686*growth(z, om0, ol0)/growth(0, om0, ol0)
