from __future__ import division
import numpy as np
import cosmo_today

omega_l0 = cosmo_today.omega_l0
omega_r0 = cosmo_today.omega_r0
omega_m0 = cosmo_today.omega_m0

def hubble_ratio(z, omega_l0, omega_m0, omega_r0):
    """The value of H(z)/H0 at any given redshift for any set of present content of the
    universe"""
    omega_0 = omega_l0+omega_m0+omega_r0
    return np.sqrt(omega_l0 + (1-omega_0)*(1+z)**2 + omega_m0*(1+z)**3 + omega_r0*(1+z)**4)

def omega_m (z, omega_m0, h):
    """Value of the density of matter in the universe as a function of redshift"""
    return omega_m0*(1+z)**3/h**2

def omega_r (z, omega_r0, h):
    """Value of the density of radiation in the universe as a function of redshift"""
    return omega_r0*(1+z)**4/h**2

def omega_l (omega_l0, h):
    """Value of the density of DE in the universe as a function of redshift"""
    return omega_l0/h**2

def omega(z, omega_0, h):
    """Value of the total density in the universe as a function of redshift"""
    return 1+(omega_0-1)*(1+z)**2/h**2

