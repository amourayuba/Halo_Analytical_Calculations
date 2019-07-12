from __future__ import division
from colossus.lss import mass_function
from colossus.lss import peaks
from fluctuation_rms import *
import astropy as aspy
from astropy.cosmology import LambdaCDM

my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': om, 'Ode0': oml, 'Ob0': omb, 'sigma8': sigma8, 'ns': ns}
cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)





########################################################################################################################

########################----------------------- PRESS AND SCHECHTER HALO MASS FUNCTION--------------###################

########################################################################################################################


def hmf(M, z=0, window='TopHat', sig8=sigma8, om0=om, ol0=oml, omb=omb, h=h, kmax=30, prec=1000, out='hmf', camb=False):
    if type(z)==np.ndarray:
        l = len(z)
        del_c = delta_c(z, om0, ol0)  #shape (l, )

        if type(M) == np.ndarray or type(M) == list:
            n = len(M)-1
            sig = sigma(M, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb) #shape : (n, )
            dlsig = np.log(sig[1:] / sig[:-1])            #shape : (n-1, )
            dlM = np.log(M[1:] / M[:-1])        #shape : (n-1, )
            new_sig = (sig[1:] + sig[:-1]) * 0.5    #shape : (n-1, )
            new_m = (M[1:] + M[:-1]) * 0.5        #shape : (n-1, )

            mat_new_m = np.array([new_m]*l)  #shape (l, n-1)
            mat_new_sig = np.array([new_sig] * l)  #shape (l, n-1)
            mat_del_c = np.array([del_c]*n).transpose()  #shape (l, n-1)

            ra1 = rho_m(z=0, om0=om0) / mat_new_m ** 2        #shape : (l, n-1 )
            ra2 = np.exp(-mat_del_c ** 2 / (2 * mat_new_sig ** 2)) * np.sqrt(2 / np.pi) * mat_del_c / mat_new_sig
            ra3 = np.array([dlsig / dlM]*l)
            if out == 'hmf':
                return -ra1 * ra2 * ra3
            elif out == 'dn/dlnM':
                return -mat_new_m * ra1 * ra2 * ra3
            elif out == 'dimensionless':
                return -ra2 * ra3
        else:
            nM = np.array([0.999 * M, 1.001 * M])
            sig = sigma(nM, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb=camb)
            dlsig = np.log(sig[1:] / sig[:-1])
            dlM = np.log(nM[1:] / nM[:-1])
            new_sig = (sig[1:] + sig[:-1]) * 0.5
            new_m = (nM[1:] + nM[:-1]) * 0.5

            ra1 = np.sqrt(2 / np.pi) * rho_m(z=0, om0=om0) * del_c / (new_m ** 2 * new_sig)
            ra2 = np.exp(-del_c ** 2 / (2 * new_sig ** 2))
            ra3 = dlsig / dlM

            if out == 'hmf':
                return -ra1 * ra2 * ra3
            elif out == 'dn/dlnM':
                return -new_m * ra1 * ra2 * ra3
            elif out == 'dimensionless':
                return -ra2 * ra3
    else:
        del_c = delta_c(z, om0, ol0)
        if type(M) == np.ndarray or type(M) == list:
            sig = sigma(M, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb=camb)
            dlsig = np.log(sig[1:]/sig[:-1])
            dlM = np.log(M[1:]/M[:-1])
            new_sig = (sig[1:]+sig[:-1])*0.5
            new_m = (M[1:]+M[:-1])*0.5

            ra1 = rho_m(z=0, om0=om0)/new_m**2
            ra2 = np.exp(-del_c**2/(2*new_sig**2))*np.sqrt(2/np.pi)*del_c/new_sig
            ra3 = dlsig/dlM
            if out=='hmf':
                return -ra1*ra2*ra3
            elif out =='dn/dlnM':
                return -new_m*ra1*ra2*ra3
            elif out == 'dimensionless':
                return -ra2*ra3
        else:
            nM = np.array([0.999*M, 1.001*M])
            sig = sigma(nM, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb=camb)
            dlsig = np.log(sig[1:] / sig[:-1])
            dlM = np.log(nM[1:] / nM[:-1])
            new_sig = (sig[1:] + sig[:-1]) * 0.5
            new_m = (nM[1:] + nM[:-1]) * 0.5

            ra1 = np.sqrt(2 / np.pi) * rho_m(z=0, om0=om0) * del_c / (new_m ** 2 * new_sig)
            ra2 = np.exp(-del_c ** 2 / (2 * new_sig ** 2))
            ra3 = dlsig / dlM

            if out=='hmf':
                return -ra1*ra2*ra3
            elif out =='dn/dlnM':
                return -new_m*ra1*ra2*ra3
            elif out == 'dimensionless':
                return -ra2*ra3




#######################-------------------------------Halo Mass Function plot----------------###########################



'''M = np.logspace(11,15, 100)
z = np.array([0, 0.5, 2, 4])
y1 = hmf(M, z, kmax=50, prec=100, out='dn/dlnM')
for i in range(4):
    plt.loglog(M[1:], y1[i,:], label='Analytic  z='+str(z[i]))
    plt.loglog(M[1:], mass_function.massFunction(M[1:], z[i], model='press74', q_out='dndlnM'), '--', label='Colossus')
plt.xlabel('M [$h^{-1}M_\odot$]', size = 15)
plt.ylabel('dn/dlnM [$h^3/Mpc^{3}$]', size = 15)
plt.title('Press and Schechter halo mass function', size = 15)
plt.ylim(1e-10, 0.5)
plt.xlim(1.5e11, 9e14)
plt.legend()
plt.show()'''


##############################"---------------------HMF sig8 evolution---------------------#############################

'''M = np.logspace(11,16, 100)
sigma8 = [0.6,0.7,0.8,0.9,1,1.1]
for el in sigma8:
    my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': om, 'Ode0': oml, 'Ob0': omb, 'sigma8': el, 'ns': ns}
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
    y1 = hmf(M, z=0, sig8=el, out='dn/dlnM')
    plt.loglog(M[1:], y1, label='Analytic  $\sigma_8=$'+str(el))
    plt.loglog(M[1:], mass_function.massFunction(M[1:], 0, model='press74', q_out='dndlnM'), '--', label='Colossus')
plt.xlabel('M [$h^{-1}M_\odot$]', size = 15)
plt.ylabel('dn/dlnM [$h^3/Mpc^{3}$]', size = 15)
plt.xlim(5e12, 6e15)
plt.ylim(0.004, 1e-7)
plt.title('Press and Schechter multiplicity function', size = 15)
plt.legend()
plt.show()'''

######################------------------- HMF Omega_m evolution------------------------################################

'''M = np.logspace(11,16, 100)
omv = [0.1,0.3,0.5,0.7]
for el in omv:
    my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': el, 'Ode0': 1-el, 'Ob0': omb, 'sigma8': sigma8, 'ns': ns}
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
    y1 = hmf(M, z=0, om0=el, ol0=1-el, out='dn/dlnM')
    plt.loglog(M[1:], y1, label='Analytic  $\Omega_m=$'+str(el))
    plt.loglog(M[1:], mass_function.massFunction(M[1:], 0, model='press74', q_out='dndlnM'), '--', label='Colossus')
plt.xlabel('M [$h^{-1}M_\odot$]', size = 15)
plt.ylabel('dn/dlnM [$h^3/Mpc^{3}$]', size = 15)
plt.xlim(5e12, 6e15)
plt.ylim(0.004, 1e-7)
plt.title('Press and Schechter multiplicity function', size = 15)
plt.legend()
plt.show()'''



########################################################################################################################

########################----------------------- PEAK HEIGHT AND MULTIPLICITY FUNCTION--------------#####################

########################################################################################################################


def fps(nu):
    return np.sqrt(2/np.pi)*nu*np.exp(-nu**2/2)

def nu(M, z=[0], om0=om, ol0=oml, omb=omb, sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000, camb=False ):
    return delta_c(z[0], om0, ol0)/sigma(M, sig8, h,
                                              kmax, window, 'M', prec, om0, ol0, omb, camb)


################-------------------Peak Heigh CAMB-COLOSSUS COMPARISON------------###################################
'''M = np.logspace(9, 17, 1000)
z = [0, 1, 2, 4]
my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': om, 'Ode0': oml, 'Ob0': omb, 'sigma8': sigma8, 'ns': ns}
cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
for el in z:
    nu1 = nu(M, z=[el])
    nu2 = peaks.peakHeight(M, z=el)
    plt.loglog(M, nu1, label='Analytic  z='+str(el))
    plt.loglog(M, nu2,'--', label='COLOSSUS')
plt.xlabel('M [$M_\odot/h$]', size=15)
plt.ylabel(r'$\nu$', size=15)
plt.legend()
plt.show()'''


'''M = np.logspace(9, 17, 1000)
sig8 = [0.4, 0.6, 0.8, 1, 1.2]
for el in sig8:
    nu1 = nu(M, z=[0], sig8=el)
    my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': om, 'Ode0': oml, 'Ob0': omb, 'sigma8': el, 'ns': ns}
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
    nu2 = peaks.peakHeight(M, z=0)
    plt.loglog(M, nu1, label='Analytic  $\sigma_8$='+str(el))
    plt.loglog(M, nu2,'--', label='COLOSSUS')
plt.xlabel('M [$M_\odot/h$]', size=15)
plt.ylabel(r'$\nu$', size=15)
plt.legend()
plt.show()'''


'''M = np.logspace(9, 17, 1000)
omegam = [0.1, 0.2, 0.3, 0.4, 0.5]
for el in omegam:
    nu1 = nu(M, z=[0], om0=el, ol0=1-el)
    my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': el, 'Ode0': 1-el, 'Ob0': omb, 'sigma8': 0.81, 'ns': ns}
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
    nu2 = peaks.peakHeight(M, z=0)
    plt.loglog(M, nu1, label='Analytic  $\Omega_m$='+str(el))
    plt.loglog(M, nu2,'--', label='COLOSSUS')
plt.xlabel('M [$M_\odot/h$]', size=15)
plt.ylabel(r'$\nu$', size=15)
plt.legend()
plt.show()'''





######################----------------------- Multiplicity function plot------------------##############################


######################---------------------- Comparison with collossus---------------###################################
'''M = np.logspace(11,16, 100)
z = [0, 2, 4]
for el in z:
    y1 = fps(nu(M, z=[el], kmax=50, prec=200))
    y2 = mass_function.massFunction(M, z=el, model='press74')
    plt.loglog(M, y1, label = 'z='+str(el))
    plt.loglog(M, y2, '--', label = ' Colossus')
plt.xlabel('M [$h^{-1}M_\odot$]', size = 15)
plt.ylabel('f', size = 15)
plt.xlim(2e11, 8e15)
plt.ylim(1e-14, 1)
plt.title('Press and Schechter multiplicity function', size = 15)
plt.legend()
plt.show()'''

###############"-------------------Multiplicity compaarison with colossus sig8 evolution----------------################

'''M = np.logspace(13,17, 100)
sigma8 = [0.6, 0.8,1, 1.2]
for el in sigma8:
    y1 = fps(nu(M, z=0, sig8 = el, win='TopHat'))
    my_cosmo = {'flat': True, 'H0': 100*h, 'Om0': om0, 'Ob0': 0.043, 'sigma8': el, 'ns': ns}
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
    y2 = fps(peaks.peakHeight(M, z= 0))
    #mfunc = mass_function.massFunction(M, z=0, mdef='fof', model='press74', q_out='f')
    plt.loglog(M, y1, label = 'sigma='+str(el))
    plt.loglog(M, y2, '--', label = ' Colossus')
plt.xlabel('M [$h^{-1}M_\odot$]', size = 15)
plt.ylabel('f', size = 15)
plt.xlim(2e13, 8e16)
plt.ylim(1e-14, 1)
plt.title('Press and Schechter multiplicity function', size = 15)
plt.legend()
plt.show()'''


##########################------------  same : omega_m evolution--------------------------##############################

'''M = np.logspace(13,17, 100)
om1 = [0.1, 0.3, 0.5, 0.7]
for el in om1:
    y1 = fps(nu(M, z=0, om0=el, ol0=1-el,  sig8 = 0.8154, win='TopHat'))
    my_cosmo = {'flat': True, 'H0': 100*h, 'Om0': el, 'Ode0' : 1-el, 'Ob0': 0.043, 'sigma8': 0.8154, 'ns': ns}
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
    y2 = fps(peaks.peakHeight(M, z= 0))
    #mfunc = mass_function.massFunction(M, z=0, mdef='fof', model='press74', q_out='f')
    plt.loglog(M, y1, label = '$\Omega_m=$'+str(el)+' Yuba')
    plt.loglog(M, y2, '--', label = ' Colossus')
plt.xlabel('M [$h^{-1}M_\odot$]', size = 15)
plt.ylabel('f', size = 15)
plt.xlim(2e13, 8e16)
plt.ylim(1e-14, 1)
plt.title('Press and Schechter multiplicity function', size = 15)
plt.legend()
plt.show()'''






########################################################################################################################

########################----------------------- PS Caracteristic non linear Mass M^star--------------###################

########################################################################################################################


def Mstar(lMmin=6, lMmax=15, npoints = 10000, z=0, h=h, om0=om, ol0=oml, omb=omb, sigma8 = sigma8,
               prec = 1000, kmax=100, win='TopHat', camb=False):
    mass = np.logspace(lMmin, lMmax, npoints)
    res = nu(mass, [z], om0, ol0, omb, sigma8,h, kmax, win, prec, camb)
    return np.min(mass[res>1])




########################---------------------------Plotting M_\star----------------#####################################

'''zs = np.linspace(0, 4, 50)
res = []
my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': om, 'Ode0': oml, 'Ob0': omb, 'sigma8': sigma8, 'ns': ns}
cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
Ms_co = peaks.nonLinearMass(zs)
for el in zs:
    res.append(Mstar(z=el, npoints=1000, kmax=100))
plt.loglog(1+zs, res, label='Analytic')
plt.loglog(1+zs, Ms_co, label='Colossus')
plt.xlabel('z', size = 15)
plt.ylabel('$M^\star$[$h^{-1}M_\odot$]', size = 15)
plt.xlim(0, 4)
plt.ylim(1e8, 1e13)
plt.title('Press and Schechter caracteristic non linear mass', size=15)
plt.legend()
plt.show()'''


'''om = np.linspace(0.1, 0.8, 50)
res = []
res2 = []
for el in om:
    my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': el, 'Ode0': 1-el, 'Ob0': omb, 'sigma8': sigma8, 'ns': ns}
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
    res.append(Mstar(z=0, om0=el, ol0=1-el, npoints=300, kmax=5))
    res2.append(peaks.nonLinearMass(0))
plt.plot(om, res, '-g', label='Analytic')
plt.plot(om, res2, '--k', label='Colossus')
plt.yscale('log')
plt.xlabel('$\Omega_m$', size = 15)
plt.ylabel('$M^\star$[$h^{-1}M_\odot$]', size = 15)
plt.xlim(0.1, 0.8)
plt.ylim(1e11, 1e14)
plt.legend()
plt.title('Press and Schechter caracteristic non linear mass', size=15)
plt.show()'''


'''s8 = np.linspace(0.1, 1.5, 50)
res = []
res2 = []
for el in s8:
    my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': om, 'Ode0': oml, 'Ob0': omb, 'sigma8': el, 'ns': ns}
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
    res.append(Mstar(z=0, sigma8=el, om0=om, ol0=oml, npoints=300, kmax=50))
    res2.append(peaks.nonLinearMass(0))
plt.plot(s8, res, '-g', label='Analytic')
plt.plot(s8, res2, '--k', label = 'Colossus')
plt.yscale('log')
plt.xlabel('$\sigma_8$', size = 15)
plt.ylabel('$M^\star$[$h^{-1}M_\odot$]', size = 15)
plt.xlim(0.1, 1.5)
plt.ylim(2e10, 5e14)
plt.title('Press and Schechter caracteristic non linear mass', size=15)
plt.legend()
plt.show()'''


#####################------------------------omega_m vs sigma8 Mstar=const-----------------------######################"

'''sze = 15
omv = np.linspace(0.1, 0.7, sze)
sig8 = np.linspace(0.4, 1.5, sze)
nom = np.zeros((sze,sze))
for i in range(sze):
    for j in range(sze):
        nom[i,j] = Mstar(lMmin=1, lMmax=18,  z=0, om0 = omv[i], ol0=1-omv[i], sigma8=sig8[j],
                              npoints=1000, prec=100, kmax=100)
plt.contourf(omv, sig8, np.log10(nom), levels=100, cmap='jet')
plt.xlabel('$\Omega_m$', size = 15)
plt.ylabel('$\sigma_8$', size = 15)
plt.colorbar()
plt.title('$\log M^\star$')
plt.show()'''

'''sze = 15
omv = np.linspace(0.1, 0.7, sze)
sig8 = np.linspace(0.4, 1.5, sze)
nom = np.zeros((sze,sze))
for i in range(sze):
    for j in range(sze):
        my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': omv[i], 'Ode0': 1-omv[i], 'Ob0': omb,
                    'sigma8': sig8[j], 'ns': ns}
        cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
        nom[i,j] = peaks.nonLinearMass(0)
plt.contourf(omv, sig8, np.log10(nom), levels=100, cmap='jet')
plt.xlabel('$\Omega_m$', size = 15)
plt.ylabel('$\sigma_8$', size = 15)
plt.colorbar()
plt.title('$\log M^\star$ Colossus')
plt.show()'''





########################################################################################################################

########################----------------------- Integrated HMF N(>M) -----------------------############################

########################################################################################################################



def nofm(M, lMmax=18, z=0, window='TopHat', sigma8=sigma8, om0=om, ol0=oml, omb=omb, h=h, kmax=30, prec=300, camb=False):
    def dn(x):
        return np.exp(x)*hmf(np.exp(x), z, window, sigma8, om0, ol0, omb, h, kmax, prec, out='hmf', camb=camb)
    return quad(dn, np.log(M), lMmax*np.log(10))[0]


def nofm_man(M, lMmax=20, z=0, window='TopHat', sigma8=sigma8, om0=om, ol0=oml, omb=omb, h=h, kmax=30,
                     prec=100, acc=np.int(1e4), Colos=False, camb=False):
    Ms = np.logspace(np.log10(M), lMmax, acc)
    dlM = np.log(10)*(lMmax-np.log10(M))/acc

    if Colos :
        my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': om0, 'Ode0': ol0, 'Ob0': omb, 'sigma8': sigma8, 'ns': ns}
        cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
        y = mass_function.massFunction(Ms, z, model='press74', q_out='dndlnM')
        return np.sum(y*dlM)
    elif type(z) == np.ndarray:
        y = hmf(Ms, z, window, sigma8, om0, ol0, omb, h, kmax, prec, out='hmf', camb=camb)
        l, m = y.shape
        mat_Ms = np.array([Ms[1:]]*l)
        return np.sum(mat_Ms*dlM*y, axis=1)
    else:
        y = hmf(Ms, z, window, sigma8, om0, ol0, omb, h, kmax, prec, out='hmf', camb=camb)
        return np.sum(Ms[1:]*dlM*y)




#########################------------------------PLOTS OF VARIOUS QUANTITIES----------------############################


#########################----------------omega_m vs sigma8 at n = const--------------------############################


'''size = 15
omv = np.linspace(0.25, 0.35, size)
olv = 1-omv
#ombv = 0.13*omv
sig8 = np.linspace(0.7, 0.95, size)
nom = np.zeros((size,size))
mt = 4e13
for i in range(size):
    for j in range(size):
        nom[i,j] = np.log10(nofm_manual(mt, lMmax= 18, sigma8=sig8[j], om0=omv[i], ol0=olv[i],
                                             kmax=5, prec=100, Colos=False))
plt.contourf(omv, sig8, nom, levels=60, cmap='jet')
plt.xlabel('$\Omega_m$', size = 15)
plt.ylabel('$\sigma_8$', size = 15)
plt.colorbar()
plt.title(r'$N(>4\times10^{13})$')
plt.show()'''





########################################################################################################################

##################################--------------------N(z) for a given volume-----------################################

########################################################################################################################


def N(z, M, solid_angle, lMmax=20, window='TopHat', sigma8=sigma8, om0=om, ol0=oml, omb=omb, h=h, kmax=30,prec=100,
      acc=np.int(1e4), Colos=False, differential=True, z2=None, prec2=None, camb=False):
    cosmo = LambdaCDM(H0=100 * h, Om0=om0, Ode0=ol0, Ob0=omb)

    if differential:
        vol = cosmo.differential_comoving_volume(z).value * h ** 3
        Ntot = nofm_man(M, lMmax, z, window, sigma8, om0, ol0, omb, h, kmax, prec, acc, Colos, camb)
        return Ntot*solid_angle*vol
    else:
        zs = np.linspace(z, z2, prec2)
        vol = cosmo.differential_comoving_volume(zs).value * h ** 3
        dz = (z2-z)/prec2
        Nofz = nofm_man(M, lMmax, zs, window, sigma8, om0, ol0, omb, h, kmax, prec, acc, Colos=False, camb=camb)
        return np.sum(Nofz*vol)*dz*solid_angle



'''size = 15
omv = np.linspace(0.25, 0.35, size)
olv = 1-omv
#ombv = 0.13*omv
sig8 = np.linspace(0.65, 1.1, size)
nom = np.zeros((size,size))
mt = 3e13
ang = 1000*np.pi**2/180**2
for i in range(size):
    for j in range(size):
        nom[i,j] = np.log10(N(z=0.15, M=mt, solid_angle=ang,  lMmax= 18, sigma8=sig8[j], om0=omv[i], ol0=olv[i],
                                             kmax=5, prec=100, Colos=False, differential=False, z2=0.2, prec2=100))
plt.contourf(omv, sig8, nom, levels=60, cmap='jet')
plt.xlabel('$\Omega_m$', size = 15)
plt.ylabel('$\sigma_8$', size = 15)
plt.colorbar()
plt.title(r'$N(>3\times10^{13}, 0.15<z<0.2, \omega=1000 deg^2)$')
plt.show()'''