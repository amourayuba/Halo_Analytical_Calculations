from halo_mass_function import *
from scipy.integrate import quad
import matplotlib.pyplot as plt

def upcrossing(M1, M2, z1, z2, sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000, om0=om, ol0=oml, omb=omb, camb=False):
    w1 = delta_c(z1, om0, ol0)
    w2 = delta_c(z2, om0, ol0)
    S1 = sigma(M1, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb)**2
    S2 = sigma(M2, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb)**2
    dw = w1-w2
    dS = S1-S2
    return np.exp(-dw**2/(2*dS))*dw/np.sqrt(2*np.pi*dS**3)


def formation_time(M2, z1, z2, sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000, acc = np.int(1e4), om0=om,
                   ol0=oml, omb=omb, camb=False):
    #Mass = np.logspace(np.log10(M2/2), np.log10(M2), acc)
    Mass = np.linspace(M2/2, M2, acc)
    sigs = sigma(Mass, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb)**2

    #M1 = np.sqrt(Mass[2:]*Mass[:-2])
    M1 = (Mass[2:]+Mass[:-2])/2
    dsig = -(sigs[2:] - sigs[:-2])
    fS = upcrossing(M1, M2, z1, z2, sig8, h, kmax, window, prec, om0, ol0, omb, camb)

    return M2*np.sum(fS*dsig/M1)

'''def test(S1, S2):
    ds = 2*(S1-S2)
    return 1/np.sqrt(np.pi*ds)
M2 = 1e14
Mh = M2/2
masses = np.linspace(Mh, 0.999*M2, 10000)
sigs = sigma(masses, prec=100)**2
S2 = sigma(M2)**2
dsig = -sigs[1]+sigs[0]
te = test(sigs, S2)
res = M2*dsig*np.sum(te/masses)'''

def K(ds, dw):
    return dw*np.exp(-dw**2/(2*ds))/np.sqrt(2*np.pi*ds**3)
def mu(St, a):
    return (1 + (2**a -1)*St)**(1/a)

def proba(M,  zf,  acc=1000, zi=0.0, sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000, om0=om,
          ol0=oml, omb=omb, camb=False, colos=False):
    S0 = sigma(M, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos)**2
    Sh = sigma(M/2, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos)**2
    w0 = delta_c(zi, om0, ol0)
    if type(zf) == np.ndarray:
        mass = np.logspace(np.log10(M/2), np.log10(M), acc)
        l = len(zf)
        mat_zf = np.array([zf]*acc)   #(acc, l)
        mat_mass = np.array([mass]*l).transpose()
        mat_wf = delta_c(mat_zf, om0, ol0)
        mat_wt = (mat_wf-w0)/np.sqrt(Sh-S0)
        mat_S = sigma(mat_mass, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos)**2
        mat_St = (mat_S - S0)/(Sh - S0)   # (acc, l)
        #mat_St[-1, :] = 1e-10
        mat_Ks = K(mat_St, mat_wt)
        mat_ds = (mat_St[2:, :] - mat_St[:-2,:])*0.5
        return -M*np.sum(mat_ds*mat_Ks[1:-1,:]/mat_mass[1:-1,:], axis=0)
    else:
        mass = np.logspace(np.log10(M/2), np.log10(M), acc)
        wf = delta_c(zf, om0, ol0)
        wt = (wf-w0)/np.sqrt(Sh-S0)
        #St = np.logspace(-10, 0, acc)
        S = sigma(mass, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos)**2
        St = (S - S0)/(Sh - S0)
        #St[-1] = 1e-10
        Ks = K(St, wt)
        ds = (St[2:] - St[:-2])*0.5
        return -M*np.sum(ds*Ks[1:-1]/mass[1:-1])

def proba2(wf, a, man=False, acc=1000):
    if man == True:
        xs = np.logspace(-10, 0, acc)
        Ks = K(xs, wf)
        mus = mu(xs, a)
        dx = (xs[2:] - xs[:-2])*0.5
        return -np.sum(Ks[1:-1]*mus[1:-1]*dx)
    else :
        def f(x):
            return K(x, wf)*mu(x, a)
        return -quad(f, 0, 1)[0]

'''wf = np.linspace(1e-4, 4, 10000)
res = []
for el in wf:
    res.append(proba2(el, man = True, acc=1000, a=-2))
res = np.array(res)
dpdw = (res[2:] - res[:-2])/(wf[2:] - wf[:-2])
plt.plot(wf[1:-1], dpdw)
#plt.plot(wf, -res)
plt.show()'''

'''zs = np.linspace(0.1, 2, 1000)
masses = [1e12, 1e13, 1e14]
for ms in masses:
    res = proba(ms, zf=zs, om0=0.7, colos=True)
    dpdw = (res[2:] - res[:-2])/(zs[2:] - zs[:-2])
    #plt.plot(zs[1:-1], -dpdw, label='log M='+str(np.log10(ms)))
    plt.plot(zs, res, label='log M='+str(np.log10(ms)))
plt.legend()
plt.xlabel('z', size=15)
plt.ylabel('$P(z_f>z)$', size=15)
#plt.ylabel('$dP/dz$', size=15)
plt.show()'''


'''zi = [0, 0.2, 0.5, 1]
for red in zi:
    zs = np.linspace(red+0.1, 2, 1000)
    res = proba(M=1e12, zi=red, zf=zs, acc=1000, colos=True)
    res = np.array(res)
    dpdw = (res[2:] - res[:-2])/(zs[2:] - zs[:-2])
    #plt.plot(zs[1:-1], dpdw)
    plt.plot(zs, res, label='$z_i=$'+str(red))
plt.legend()
plt.xlabel('z', size=15)
plt.ylabel('$P(z_f>z)$', size=15)
plt.show()'''



def average_formation(M, z, acc=1000, sig8=sigma8, h=h, kmax=30, window='TopHat', prec=1000,
                      om0=om, ol0=oml, omb=omb, camb=False, colos=False):
    zs = np.linspace(z+0.1, 2, acc)
    res = proba(M, zs, acc, z, sig8, h, kmax, window, prec, om0, ol0, omb, camb, colos)
    return np.max(zs[res>0.5])

'''zi = np.linspace(0, 1, 20)
res = []
for red in zi:
    res.append(average_formation(M=1e12, z=red, colos=True))
    #plt.plot(zs[1:-1], dpdw)'''

'''mass = np.logspace(9, 14, 20)
res = []
for el in mass:
    res.append(average_formation(M=el, z=0, colos=True))
    #plt.plot(zs[1:-1], dpdw)

plt.plot(mass, res)
plt.legend()
plt.xlabel('$M$', size=15)
plt.xscale('log')
plt.ylabel('$z_{formation}$', size=15)
plt.show()'''


'''s8 = np.linspace(0.5, 1.1, 20)
res = []
for el in s8:
    res.append(average_formation(M=1e12, z=0, sig8=el, colos=True))
    #plt.plot(zs[1:-1], dpdw)

plt.plot(s8, res)
plt.legend()
plt.xlabel('$\sigma_8$', size=15)
plt.ylabel('$z_{formation}$', size=15)
plt.show()'''



'''omeg = np.linspace(0.1, 0.7, 20)
res = []
for el in omeg:
    res.append(average_formation(M=1e12, z=0, om0=el, colos=True))
    #plt.plot(zs[1:-1], dpdw)

plt.plot(omeg, res)
plt.legend()
plt.xlabel('$\Omega_m$', size=15)
plt.ylabel('$z_{formation}$', size=15)
plt.show()'''

'''sze = 15
omv = np.linspace(0.1, 0.7, sze)
s8 = np.linspace(0.4, 1.5, sze)
nom = np.zeros((sze,sze))
x = np.array([omv]*sze).transpose()
y = np.array([s8]*sze)
for i in range(sze):
    for j in range(sze):
        nom[i,j] = average_formation(M=4e13, z=0.15, om0=omv[i], sig8= s8[j], colos=True)
plt.contourf(x, y, nom, levels=100, cmap='jet')
plt.xlabel('$\Omega_m$', size = 15)
plt.ylabel('$\sigma_8$', size = 15)
plt.colorbar()
plt.title('mean $z_f$ for M=4e13 at z=0.2', size=15)
plt.show()'''


r = 5000
l = 2000
zs = np.linspace(0.14, 2, l)
age = proba(4e13, zf=zs, zi = 0.1, colos=True)
rand = np.random.uniform(0,np.max(age), size=r)


mat_rand = np.array([rand]*l)  #lxr
mat_age = np.array([age]*r).transpose() #lxr
mat_zs = np.array([zs]*r).transpose()

res = np.max(mat_zs*(mat_age>mat_rand), axis=0)

plt.hist(res, histtype='step', bins=30)
plt.show()

