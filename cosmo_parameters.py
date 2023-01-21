from __future__ import division
import numpy as np

### Fundamental set of parameters
G = 4.30091e-9   #Units Mpc/Msun x (km/s)^2
rho_c = 3*100**2/(8*np.pi*G)       # h^2xMsun/Mpc**3

### New set of cosmo parameters used here using
#Eiseinstein and Hu conventions and Planck 2018 values

h = 0.7
c = 3e5       #speed of light km/s
ombh2 = 0.02242   #Density of baryons in units of h2
omb = ombh2/h**2  #Density of baryons
omch2 = 0.122   #Density of CDM in units h2
omc = omch2/h**2    #Density of CDM
omnuh2 = 0.0006451439   #Density of neutrinos in units h²
omnu = omnuh2/h**2     #density of neutrinos
omh2 = ombh2 + omch2 + omnuh2   #Density of matter in units h²
om = omb + omc          #Density of matter
omr = 1e-4            #Upper limit estimation of radiation density

sigma8=0.8    #fluctuation rms normalisation at 8Mpc
ns = 0.9626   #spectral index for initial power spectrum
oml = 0.685   #density of Lambda
om0 = oml + omb + omc + omr  #Total density including lambda

Tcmb = 2.7255      #Temperature of CMB
theta_cmb = Tcmb/2.7  #Normalized CMB temperature
Nnu = 1           #Number of massive neutrino species
mnu = 91.5*omnuh2/Nnu   #Mass of neutrino part
zeq = 3387              #z at matter-radiation equality
zdrag = 1060.01        # z at compton drag
yd = (1+zeq)/(1+zdrag)
s = 147.21           #Sound horizon at recombination in Mpc


########################--------------Univers composition-----------------------########################################

def hubble_ratio(z, omega_l0=oml, omega_m0=om, omega_r0=omr):
    """The value of H(z)/H0 at any given redshift for any set of present content of the
    universe"""
    omega_0 = omega_l0+omega_m0+omega_r0
    return np.sqrt(omega_l0 + (1-omega_0)*(1+z)**2 + omega_m0*(1+z)**3 + omega_r0*(1+z)**4)

def omega_m (z, omgm0=om, omgl0=oml, omgr0=omr):
    """Value of the fraction density of matter in the universe as a function of redshift"""
    return omgm0*(1+z)**3/hubble_ratio(z, omgl0, omgm0, omgr0)**2

def omega_r (z, omgr0=omr, omgl0=oml, omgm0=om):
    """Value of the fraction density of radiation in the universe as a function of redshift"""
    return omgr0*(1+z)**4/hubble_ratio(z, omgl0, omgm0, omgr0)**2

def omega_l (z=0, omgl0=oml, omgm0=om, omgr0=omr):
    """Value of the fraction density of DE in the universe as a function of redshift"""
    return omgl0/hubble_ratio(z, omgl0, omgm0, omgr0)**2

def omega(z, omega_l0=oml, omega_m0=om, omega_r0=omr):
    """Value of the total fraction density in the universe as a function of redshift"""
    omega_0 = omega_r0+omega_l0+omega_m0
    return 1+(omega_0-1)*(1+z)**2/hubble_ratio(z, omega_l0, omega_m0, omega_r0)**2

def rho_m(z=0, om0 = om):
    """Denisty of matter """
    return omega_m(z, om0)*(3*100**2)/(8*np.pi*G)

#######################--------------------linear perturbations evolution--------------------###########################

def growth_factor_pt(ol, om):
    """approx formula for growth factor given by carroll press and turner(1992)"""
    return (5*om/2)/(om**(4/7)-ol+(1+om/2)*(1+ol/70))

def growth(z, om0=om, ol0=oml):
    """Calculate the linear growth factor as a function of redshift"""
    ol = omega_l(z, ol0, om0)        #Calculating the Dark Energy density at z
    om = omega_m(z, om0, ol0)     #Matter density at z
    return (5*om/2)*1/(om**(4/7)-ol+(1+om/2)*(1+ol/70))

def D(z, om0, ol0):
    """Normalised linear growth factor"""
    return growth(z, om0, ol0)/((1+z)*growth(0, om0, ol0))

def delta_c(z, om0=om, ol0=oml):
    """critical overdensity"""
    return 1.686*growth(0, om0, ol0)*(1+z)/growth(z, om0, ol0)
def delta_ec(z, sig, om0=om, ol0=oml):
    return np.sqrt(0.707)*delta_c(z, om0, ol0)*(1 + 0.47*(sig**2/delta_c(z, om0, ol0)**2)**0.615)



#######################################------------PLOTS --------------------------#####################################
if __name__ == "__main__":
    '''import matplotlib.pyplot as plt
    #oms = np.linspace(0.1, 0.7, 5)
    oms = [0.15, 0.3, 0.5, 0.7]
    zs = np.linspace(0, 2, 1000)
    for el in oms:
        plt.plot(zs, delta_c(zs, el, 1-el), linewidth = 4, label='$\Omega_m = $'+str(el))
    plt.legend(fontsize='x-large')
    plt.xlabel('z', size= 25)
    plt.xlim(0, 2)
    plt.ylim(1.65, 4.5)
    plt.ylabel('$\delta_c$', size= 25)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()'''


    '''import matplotlib.pyplot as plt
    oms = np.linspace(0.1, 0.7, 50)
    
    plt.plot(oms, omega_m(z=2, omgm0=oms))
    plt.show()'''