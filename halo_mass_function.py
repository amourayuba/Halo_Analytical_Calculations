from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from cosmo_parameters import *
from power_spectrum_analytic import *
from growth_factor import *
from fluctuation_rms import *



def hmf(M, z=0, window='Gauss', sigma8=sigma8, om0=omega_m0, ol0=omega_l0, H0=H_0, ombh2=ombh2, omch2=omch2, ns=ns):
    del_c = delta_c(z, om0, ol0)
    sig = sigma_a_M(M, window=window, z=0, sigma8=sigma8, H0=H0, ombh2=ombh2, omch2=omch2, ns=ns, kmax=10)
    dlsig = np.log(sig[1:]/sig[:-1])
    dlM = np.log(M[1:]/M[:-1])
    new_sig = (sig[1:]+sig[:-1])*0.5
    new_m = (M[1:]+M[:-1])*0.5

    ra1 = np.sqrt(2/np.pi)*rho_c*del_c/(new_m**2*new_sig)
    ra2 = np.exp(-del_c**2/(2*new_sig**2))
    ra3 = dlsig/dlM
    return ra1*ra2*ra3

def fps(nu):
    return np.sqrt(2/np.pi)*nu*np.exp(-nu**2/2)

M = np.logspace(11,15, 100)
z = [0,0.5,1,2]
for el in z:
    y1 = hmf(M, z=el, sigma8=0.8)
    plt.loglog(M[1:], -y1, label='z='+str(el))
plt.xlabel('M [$h^{-1}M_\odot$]', size = 15)
plt.ylabel('n(M) [$h^4/Mpc^{3}/M_\odot$]', size = 15)
plt.title('Press and Schechter halo mass function', size = 15)
plt.legend()
plt.show()

sigma8 = [0.6,0.7,0.8,0.9,1,1.1]
for el in sigma8:
    y1 = hmf(M, z=0, sigma8=el)
    plt.loglog(M[1:], -y1, label='$\sigma_8=$'+str(el))
plt.xlabel('M [$h^{-1}M_\odot$]', size = 15)
plt.ylabel('n(M) [$h^4/Mpc^{3}/M_\odot$]', size = 15)
plt.title('Press and Schechter halo mass function', size = 15)
plt.legend()
plt.show()


om1 = [0.1,0.2,0.3,0.4,0.5]
for el in om1:
    y1 = hmf(M, z=0.5, sigma8=0.8, om0=el)
    plt.loglog(M[1:], -y1, label='$\Omega_m=$'+str(el))
plt.xlabel('M [$h^{-1}M_\odot$]', size = 15)
plt.ylabel('n(M) [$h^4/Mpc^{3}/M_\odot$]', size = 15)
plt.title('Press and Schechter halo mass function', size = 15)
plt.legend()
plt.show()