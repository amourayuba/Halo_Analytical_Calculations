import numpy as np
import matplotlib.pyplot as plt
from growth_factor import *
import camb
from camb import model
from cosmo_parameters import *
from colossus.cosmology import cosmology

#def q(k):            #Normalised version of k see eq (5)
#    return (k/19)/np.sqrt(omh2*10000*(1+zeq))



########################################################################################################################

##########################------------------EISEINSTEIN & HU ACCURATE FITTING FORMULAE-----------------#################

########################################################################################################################

def zeq(omega=om, h=h, Tcmb=Tcmb):
    theta_cmb = Tcmb/2.7
    return 25000*omega*h**2/theta_cmb**4


def keq(omega=om, h=h, Tcmb=Tcmb):
    theta_cmb = Tcmb / 2.7
    return 0.0746*omega*h**2/theta_cmb**2

def zd(omega=om, omega_b=omb, h=h):
    b1 = 0.313*(omega*h**2)**(-0.419)*(1 + 0.607*(omega*h**2)**0.674)
    b2 = 0.238*(omega*h**2)**0.223
    return 1291*(1 + b1*(omega_b*h**2)**b2)*(omega*h**2)**0.251/(1 + 0.659*(omega*h**2)**0.828)


def q(k, omega=om, h=h, Tcmb=Tcmb):
    theta_cmb = Tcmb/2.7
    return theta_cmb**2*k/(omega*h**2)


def sound(omega=om,  omega_b=omb, h=h, Tcmb=Tcmb):
    theta_cmb = Tcmb/2.7
    k = keq(omega, h, Tcmb)
    z_eq = zeq(omega, h, Tcmb)
    z_d = zd(omega, omega_b, h)
    Rd = 31.5*omega_b*h**2/(theta_cmb**4*0.001*z_d)
    Req = 31.5 * omega_b * h ** 2 / (theta_cmb ** 4 * 0.001 * z_eq)
    return (2/(3*k))*np.sqrt(6/Req)*np.log((np.sqrt(1 + Rd)+ np.sqrt(Rd + Req))/(1 + np.sqrt(Req)))


def ksilk(omega=om, omega_b=omb, h=h):
    return 1.6*(omega_b*h**2)**0.52*(omega*h**2)**0.73*(1 + (10.4*omega*h**2)**(-0.95))



def alpha_c(omega=om, omega_b=omb, h=h):
    a1 = (46.9*omega*h**2)**0.670*(1 + (32.1*omega*h**2)**(-0.532))
    a2 = (12*omega*h**2)**0.424*(1 + (45*omega*h**2)**(-0.582))
    return a1**(-omega_b/omega)*a2**(-(omega_b/omega)**3)


def beta_c(omega=om, omega_b=omb, h=h):
    b1 = 0.944/(1 + (458*omega*h**2)**(-0.708))
    b2 = (0.395*omega*h**2)**(-0.0266)
    omc = omega - omega_b
    return 1/(1 + b1*((omc/omega)**b2 -1))

def G(y):
    te = np.sqrt(1 + y)
    return y*(-6*te + (2 + 3*y)*np.log((te + 1)/(te - 1)))

def alpha_b(omega=om,  omega_b=omb, h=h, Tcmb=Tcmb):
    theta_cmb = Tcmb / 2.7
    k = keq(omega, h, Tcmb)
    z_eq = zeq(omega, h, Tcmb)
    z_d = zd(omega, omega_b, h)
    Rd = 31.5 * omega_b * h ** 2 / (theta_cmb ** 4 * 0.001 * z_d)
    s = sound(omega, omega_b, h, Tcmb)
    return 2.07*k*s*(1 + Rd)**(-0.75)*G((1 + z_eq)/(1 + z_d))

def beta_b(omega, omega_b, h):
    return 0.5 + omega_b/omega + (3-2*omega_b/omega)*np.sqrt((17.2*omega*h**2)**2 +1)

def frac(k, omega=om, omega_b=omb, h=h, Tcmb=Tcmb):
    s = sound(omega, omega_b, h, Tcmb)
    return 1/(1 + (k*s/5.4)**4)

def T0_tilde(nq, al_c, bet_c):
    C = 14.2/al_c + 386/(1 + 69.9*nq**1.08)
    return np.log(np.e + 1.8*bet_c*nq)/(np.log(np.e + 1.8*bet_c*nq) + C*nq**2)

def Tc(k, omega=om, omega_b=omb, h=h, Tcmb=Tcmb):
    nq = q(k, omega, h, Tcmb)
    al_c = alpha_c(omega, omega_b, h)
    bet_c = beta_c(omega, omega_b, h)
    f = frac(k, omega, omega_b, h, Tcmb)
    return f*T0_tilde(nq, 1, bet_c) + (1 - f)*T0_tilde(nq, al_c, bet_c)


def stilde(k, omega=om, omega_b=omb, h=h, Tcmb=Tcmb):
    s = sound(omega, omega_b, h, Tcmb)
    bnode = 8.41*(omega*h**2)**0.435
    return s/(1 + (bnode/(k*s)**3))**(1/3)

def Tb(k, omega=om, omega_b=omb, h=h, Tcmb=Tcmb):
    s = sound(omega, omega_b, h, Tcmb)
    nq = q(k, omega, h, Tcmb)
    al_b = alpha_b(omega, omega_b, h, Tcmb)
    bet_b = beta_b(omega, omega_b, h)
    ks = ksilk(omega, omega_b, h)
    st = stilde(k, omega, omega_b, h)
    ra1 = T0_tilde(nq, 1, 1)/(1 + (k*s/5.2)**2) + al_b*np.exp(-(k/ks)**1.4)/(1 + (bet_b/(k*s))**3)
    ra2 = np.sin(k*st)/(k*st)
    return ra1*ra2

def Transfer(k, omega=om, omega_b=omb, h=h, Tcmb=Tcmb):
    return (omega_b/omega)*Tb(k, omega, omega_b, h, Tcmb) + (1 - omega_b/omega)*Tc(k, omega, omega_b, h, Tcmb)


def delta_h(omega=om, ns=ns):
    nt = ns - 1
    return omega**(-0.785 - 0.05*np.log(omega))*np.exp(-0.95*nt - 0.169*nt**2)

########################################################################################################################

#####################---------------------- Eisenstein and Hu 1998 fitting formulae------------#########################

########################################################################################################################


def T0(q):
    C0 = 14.2 + 731/(1 + 62.5*q)
    L0 = np.log(2*np.e + 1.8*q)
    return L0/(L0 + C0*q**2)

def g_eff(k, alph, omega=om, omega_b=omb, h=h):
    ns = 44.5*np.log(9.83/(omega*h**2))/np.sqrt(1 + 10*(omega_b*h**2)**0.75)
    return omega*h*(alph + (1-alph)/(1 + (0.43*k*ns)**4))

def alpha_g(omega=om, omega_b=omb, h=h):
    return 1 - 0.328*np.log(431*omega*h**2)*omega_b/omega + 0.38*np.log(22.3*omega*h**2)*(omega_b/omega)**2

def q_eff(k, omega=om, omega_b=omb, h=h, T_cmb=Tcmb):
    theta_cmb = T_cmb/2.7
    al = alpha_g(omega, omega_b, h)
    return k*theta_cmb**2/g_eff(k, al, omega, h)


def T_app(k, omega=om, omega_b=omb, h=h):
    nq = q_eff(k, omega, omega_b, h)
    return T0(nq)



#hmmm well, the results are not completely non sense. Still, nothing garantees they are right

################-------------------------------------------- Power spectrum from MBW page 200-----------------------#################

### Limit \Omega_b << Omega_m

def sigma8_normalisation(sigma8):
    return (sigma8/0.0004807375)**2

#(sigma8/0.41949012)**2 #(sigma8/0.6675070983462029)**2
def nor_sigma8_camb(sigma8):
    return (sigma8/5.18185339e-08)**2 #(sigma8 / 7.113556413980577e-08) ** 2



'''pars = camb.CAMBparams()
pars.set_cosmology(H0=100*h, ombh2=ombh2, omch2=omch2)
pars.InitPower.set_params(ns=ns)
pars.set_matter_power(redshifts=[0], kmax=10)
results = camb.get_results(pars)'''

def primordial_PK(k):
    return results.Params.scalar_power(k)

def Delta(k, sigma8=sigma8, h=h, om0=om, omb=omb, ns=ns):
    return np.sqrt(k**3*power_spectrum_a2(k, sigma8, h, om0, omb, ns)/(2*np.pi**2))






####################-----------------------TRANSFER FUNCTION TEST VS COLOSSUS-------------##############################

##############################---------------------------TRANSFER FUNCTION DEPENDENCE ON OMEGA_M----------------########

'''omg = np.arange(0.1, 0.9, 0.2)
for el in omg:
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * h, ombh2=ombh2, omch2=el*h**2 - ombh2)
    pars.InitPower.set_params(ns=ns)
    pars.set_matter_power(redshifts=[0], kmax=10)
    results = camb.get_results(pars)
    sig = results.get_sigma8()
    trans = results.get_matter_transfer_data()
    kh = trans.transfer_data[0,:,0]
    W = trans.transfer_data[model.Transfer_tot-1,:,0]
    nW = W/W[0]
    y2 = Transfer(kh*h, omega=el)
    plt.plot(kh, (y2 - nW) / nW, label='$\Omega_m = $'+str(el))
plt.xscale('log')
plt.xlabel('k [h/Mpc]')
plt.ylabel('$\Delta T/T_{CAMB}$')
plt.legend()
plt.show()'''

'''plt.figure()
plt.loglog(kh, W, label='CAMB')
plt.loglog(kh, y2, label='Analytical')
plt.xlabel('k [h/Mpc]')
plt.ylabel('Transfer function')
plt.legend()
plt.show()'''


############################-------------------POWER SPECTRUM DEPENDENCE ON OMEGA_M---------############################


'''omg = np.arange(0.1, 0.9, 0.2)
res = []
for el in omg:
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * h, ombh2=ombh2, omch2=el*h**2 - ombh2)
    pars.InitPower.set_params(ns=ns)
    pars.set_matter_power(redshifts=[0], kmax=10)
    results = camb.get_results(pars)
    sig = results.get_sigma8()
    #kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=10, npoints=1000)
    trans = results.get_matter_transfer_data()
    kh = trans.transfer_data[0, :, 0]
    W = trans.transfer_data[model.Transfer_tot-1,:,0]
    nW = W/W[0]
    y3 = results.Params.scalar_power(kh)*kh*nW**2
    y4 = results.Params.scalar_power(kh)*kh*Transfer(kh*h, el)**2
    y2 = power_spectrum_a2(kh, om0=el, sigma8=0.8)
    #y1 = pk[0]
    #res.append(y1[0]/y2[0])
    #plt.plot(kh, (y2 - y1) / y1, label='$\Omega_m = $'+str(el))
    plt.plot(kh, (y3-y4)/y4, label='$\Omega_m = $' + str(el))
#plt.loglog(omg, res)
plt.xscale('log')
#plt.yscale('log')
plt.xlabel('k [h/Mpc]')
plt.ylabel('$\Delta P/P_{CAMB}$')
plt.legend()
plt.show()'''
