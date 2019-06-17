from __future__ import division
import numpy as np

### Fundamental set of parameters
G = 4.30091e-9 #Units kpc/Msun x (km/s)^2
H_0 = 70    #km/s/Mpc
h = H_0/100
omega_m0 = 0.299
omega_l0 = 0.7
omega_r0 = 0.001
omega_0 = omega_m0 + omega_r0 + omega_l0
rho_c = 3*100**2/(8*np.pi*G)       # h^2xMsun/Mpc**3

### New set of cosmo parameters used here using
#Eiseinstein and Hu conventions and Planck 2018 values

h = 0.6766
c = 3e5       #speed of light km/s
ombh2 = 0.02242   #Density of baryons in units of h2
omb = ombh2/h**2  #Density of baryons
omch2 = 0.11933   #Density of CDM in units h2
omc = omch2/h**2    #Density of CDM
omnuh2 = 0.0006451439   #Density of neutrinos in units h²
omnu = omnuh2/h**2     #density of neutrinos
omh2 = ombh2 + omch2 + omnuh2   #Density of matter in units h²
om = omb + omc          #Density of matter
fcb = (omch2+ombh2)/omh2
fc = ombh2/omh2
fb = omch2/omh2
fnub = (ombh2 + omnuh2)/omh2
fnu = omnuh2/omh2

sigma8=0.8
ns = 0.9626   #spectral index
oml = 0.6889   #density of Lambda
om0 = oml + omb + omc  #Total density including lambda

Tcmb = 2.7255      #Temperature of CMB
theta_27 = Tcmb/2.7  #A random parameter they introduced for idontknowhy
Nnu = 1           #Number of massive neutrino species
mnu = 91.5*omnuh2/Nnu   #Mass of neutrino part
zeq = 3387              #z at matter-radiation equality
zdrag = 1060.01        # z at compton drag
yd = (1+zeq)/(1+zdrag)
s = 147.21           #Sound horizon at recombination in Mpc
#delta_H = 1.94*10**(-5)*om**(-0.785-0.05*np.log(om))*np.exp((ns-1) + 1.97*(ns-1)**2)


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

def rho_m(z=0, om0 = omega_m0):
    return omega_m(z, om0, hubble_ratio(z, omega_l0, om0, omega_r0))*(3*100**2)/(8*np.pi*G)
