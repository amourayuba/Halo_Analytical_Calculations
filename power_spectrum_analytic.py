import numpy as np
import matplotlib.pyplot as plt
from growth_factor import *
import camb
from cosmo_parameters import *

def q(k):            #Normalised version of k see eq (5)
    return (k/19)/np.sqrt(omh2*10000*(1+zeq))

def g(z):           #To stay with the paper (Eiseintein and Hu) conventions
    return hubble_ratio(z, oml, om, 0)

def D1(z):
    ge = g(z)
    omz = omega_m(z, om, ge)
    olz = omega_l(oml, ge)
    rap = (1+zeq)/(1+z)
    return 2.5*rap*omz/(omz**(4/7)-olz+(1+0.5*omz)*(1+olz/70))

def yfs(q, fnu):
    return 17.2*fnu*(1+0.488*fnu**(-7/6))*(Nnu*q/fnu)**2

def Dcb(z,q):
    pcb = 0.25*(5-np.sqrt(1+24*fcb))
    ra1 = (D1(z)/(1+yfs(q, fnu)))**0.7
    return ((1+ra1)**(pcb/0.7))*D1(z)**(1-pcb)

def Dcbn(z, q):
    pcb = 0.25*(5-np.sqrt(1+24*fcb))
    ra1 = (D1(z)/(1+yfs(q, fnu)))**0.7
    f = fcb**(0.7/pcb)
    return ((f+ra1)**(pcb/0.7))*D1(z)**(1-pcb)

def alphanu(yd):
    pcb = 0.25 * (5 - np.sqrt(1 + 24 * fcb))
    pc = 0.25 * (5 - np.sqrt(1 + 24 * fc))
    ra1 = fc/fcb
    ra2 = (5-2*(pc+pcb))/(5-4*pcb)
    ra3 = (1-0.553*fnub + 0.126*fnub**3)/(1 - 0.193*np.sqrt(fnu*Nnu) + 0.169*fnu*Nnu**0.2)
    ra4 = (1 + yd)**(pcb-pc)
    ra5 = 1 + 0.5*(pc-pcb)*(1 + 1/((3-4*pc)*(7-4*pcb)))/(1 + yd)
    return ra1*ra2*ra3*ra4*ra5

alnu = alphanu(yd)

def gamm_eff(k):
    return omh2*(np.sqrt(alnu) + (1-np.sqrt(alnu))/(1 + (0.43*k*s)**4))

def qeff(k):
    return (k*theta_27**2)/gamm_eff(k)

def Tsup(k):
    beta_c = 1/(1-0.949*fnub)
    L = np.log(np.e + 1.84*beta_c*np.sqrt(alnu)*qeff(k))
    C = 14.4 + 325/(1 + 60.5*qeff(k)**1.08)
    return L/(L + C*qeff(k)**2)

def qnu(k):
    return 3.92*q(k)*np.sqrt(Nnu/fnu)

def B(k):
    return 1 + (1.24*fnu**0.64*Nnu**(0.3 + 0.6*fnu))/(qnu(k)**(-1.6) + qnu(k)**0.8)
def Tmaster(k):
    return Tsup(k)*B(k)

def Tcb(k, z):
    return Tmaster(k)*Dcb(z, q(k))/D1(z)


#hmmm well, the results are not completely non sense. Still, nothing garantees they are right

################-------------------------------------------- Power spectrum from MBW page 200-----------------------#################

### Limit \Omega_b << Omega_m

def q2(k, h, om0, omb):
    gamma = om0*h*np.exp(-omb*(1 + np.sqrt(2*h)/om0))
    return k/(gamma)

def T2(k, h, om0, omb):
    qr = q2(k, h, om0, omb)
    a1 = np.log(1 + 2.34*qr)/(2.34*qr)
    a2 = (1 + 3.89*qr + (16.1*qr)**2 + (5.46*qr)**3 + (6.71*qr)**4)**(-0.25)
    return a1*a2

#k = np.linspace(0.01, 10, 100000)
#y1 = Tcb(k, 0)
#y2 = T2(k)
#plt.loglog(k/h, y1)
#plt.loglog(k/h, y2)
#plt.xlabel('k $\Omega_0h**2Mpc^{-1}$')
#plt.ylabel('T(k)')
#plt.legend(['Eiseinstein and Hu', 'MBW'])
#plt.title('Transfer functions from Eiseinstein and Hu and from Mo, Bosch and White')
#plt.show()

def sigma8_normalisation(sigma8):
    return (sigma8*1.046808e-4*0.8/(2.3*1.77e-8))**2

#(sigma8/0.41949012)**2 #(sigma8/0.6675070983462029)**2
def nor_sigma8_camb(sigma8):
    return (sigma8/5.18185339e-08)**2 #(sigma8 / 7.113556413980577e-08) ** 2



pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)
pars.set_matter_power(redshifts=[0], kmax=10)
results = camb.get_results(pars)

def primordial_PK(k):
    return results.Params.scalar_power(k)

def power_spectrum_a(k,  z=0, camb_ps = False, sigma8=0.8, h=h, om0=omega_m0, omb=omb, ns=0.965):
    a1 = 2*np.pi**2*(0.01*c)**(ns+3)*(D1(z)/D1(0))**2
    return sigma8_normalisation(sigma8)*a1*k**ns*T2(k, h, om0, omb)**2


def power_spectrum_a2(k,  z=0, sigma8=0.8, h=h, om0=omega_m0, ol0 = omega_l0, omb=omb, ns=0.965):
    a1 = (D(z, om0, ol0))**2
    return sigma8_normalisation(sigma8)*a1*k**ns*T2(k, h, om0, omb)**2