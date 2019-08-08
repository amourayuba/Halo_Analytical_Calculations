from cosmo_parameters import *

########################################################################################################################

##########################------------------EISEINSTEIN & HU ACCURATE FITTING FORMULAE-----------------#################

########################################################################################################################

def Transfer(k, omega=om, omega_b=omb, h=h, Tcmb=Tcmb):
    """
    Transfer function from Eisenstein and Hu 1998
    :param k : wave number
    :param omega: fraction density of matter
    :param omega_b: fraction density of baryons
    :param h: normalized hubble value
    :param Tcmb: CMB temperature
    :return: transfer function
    """
    ####----Cosmo params--------######
    omc = omega - omega_b      #### Cold Dark Matter density
    theta_cmb = Tcmb / 2.7     # Normalised CMB Temperature

    ##-------equation 2
    z_eq = 25000*omega*h**2/theta_cmb**4   #redshift at equivalence between rad and matter

    ##-------equation 3
    k_eq = 0.0746*omega*h**2/theta_cmb**2  #scale of the horizon at equality

    ##-------equation 4
    b1 = 0.313 * (omega * h ** 2) ** (-0.419) * (1 + 0.607 * (omega * h ** 2) ** 0.674)
    b2 = 0.238 * (omega * h ** 2) ** 0.223
    z_d = 1291*(1 + b1*(omega_b*h**2)**b2)*(omega*h**2)**0.251/(1 + 0.659*(omega*h**2)**0.828) #redshift at decoupling

    ##-------equation 5
    Rd = 31.5 * omega_b * h ** 2 / (theta_cmb ** 4 * 0.001 * z_d)   #Ratio of photon to baryon momentum density
    Req = 31.5 * omega_b * h ** 2 / (theta_cmb ** 4 * 0.001 * z_eq)

    ##-------equation 6 :sound horizon at drag epoch
    s = (2/(3*k_eq))*np.sqrt(6/Req)*np.log((np.sqrt(1 + Rd)+ np.sqrt(Rd + Req))/(1 + np.sqrt(Req)))

    ##----##-------equation 7
    ks = 1.6*(omega_b*h**2)**0.52*(omega*h**2)**0.73*(1 + (10.4*omega*h**2)**(-0.95)) #silk damping scale

    ##-------equation  10
    nq = theta_cmb**2*k/(omega*h**2)   #normalized wavenumber

    ##-------equation 11
    a1 = (46.9 * omega * h ** 2) ** 0.670 * (1 + (32.1 * omega * h ** 2) ** (-0.532))
    a2 = (12 * omega * h ** 2) ** 0.424 * (1 + (45 * omega * h ** 2) ** (-0.582))
    al_c = a1 ** (-omega_b / omega) * a2 ** (-(omega_b / omega) ** 3)   # approx param 1

    ##-------equation 12
    b1 = 0.944 / (1 + (458 * omega * h ** 2) ** (-0.708))
    b2 = (0.395 * omega * h ** 2) ** (-0.0266)
    bet_c = 1/(1 + b1*((omc/omega)**b2 -1))

    ##-------equation  15
    y = (1 + z_eq)/(1 + z_d)
    te = np.sqrt(1 + y)
    G = y * (-6 * te + (2 + 3 * y) * np.log((te + 1) / (te - 1)))

    ##-------equation 14
    al_b = 2.07 * k_eq * s * (1 + Rd) ** (-0.75) * G

    ##-------equation 24
    bet_b = 0.5 + omega_b / omega + (3 - 2 * omega_b / omega) * np.sqrt((17.2 * omega * h ** 2) ** 2 + 1)

    ##-------equation 18
    f = 1/(1 + (k*s/5.4)**4)

    ##-------equation 20
    C = 14.2 / al_c + 386 / (1 + 69.9 * nq ** 1.08)
    C1 = 14.2 + 386 / (1 + 69.9 * nq ** 1.08)

    ##-------equation 19
    T0 = np.log(np.e + 1.8 * bet_c * nq) / (np.log(np.e + 1.8 * bet_c * nq) + C * nq ** 2)
    T01 = np.log(np.e + 1.8*bet_c*nq)/(np.log(np.e + 1.8*bet_c*nq) + C1*nq**2)
    T011 = np.log(np.e + 1.8*1*nq)/(np.log(np.e + 1.8*1*nq) + C1*nq**2)
    ##-------equation 17
    Tc = f*T01 + (1-f)*T0

    ##-------equation 24
    bnode = 8.41 * (omega * h ** 2) ** 0.435

    ##-------equation 22
    st = s / (1 + (bnode / (k * s)) ** 3) ** (1 / 3)

    ##-------equation 21
    ra1 = T011 / (1 + (k * s / 5.2) ** 2) + al_b * np.exp(-(k / ks) ** 1.4) / (1 + (bet_b / (k * s)) ** 3)
    ra2 = np.sin(k * st) / (k * st)
    Tb = ra1*ra2

    ##-------equation 16
    return (omega_b / omega) * Tb + (1 - omega_b / omega) * Tc



###############################-----------------CAMB USAGE FOR COMPARISON-----------------##############################
'''
import camb
from camb import model
pars = camb.CAMBparams()
pars.set_cosmology(H0=100*h, ombh2=ombh2, omch2=omch2)
pars.InitPower.set_params(ns=ns)
pars.set_matter_power(redshifts=[0], kmax=10)
results = camb.get_results(pars)

def primordial_PK(k):
    return results.Params.scalar_power(k)
'''



####################-----------------------TRANSFER FUNCTION TEST VS COLOSSUS-------------##############################

##############################---------------------------TRANSFER FUNCTION DEPENDENCE ON OMEGA_M----------------########

'''
import matplotlib.pyplot as plt
omg = np.arange(0.1, 0.9, 0.2)
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

'''
import matplotlib.pyplot as plt
plt.figure()
plt.loglog(kh, W, label='CAMB')
plt.loglog(kh, y2, label='Analytical')
plt.xlabel('k [h/Mpc]')
plt.ylabel('Transfer function')
plt.legend()
plt.show()'''


############################-------------------POWER SPECTRUM DEPENDENCE ON OMEGA_M---------############################


'''
import matplotlib.pyplot as plt
omg = np.arange(0.1, 0.9, 0.2)
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



