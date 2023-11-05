import numpy as np
import cosmo_parameters as cp
from fluctuation_rms import  sigma

def parkinson08(zi, Mi, Mres, zf, dz = 1e-3,frac=0.5, acc=10000, sig8=cp.sigma8, h=cp.h, kmax=30, window='TopHat', prec=1000, om0=cp.om,
          ol0=cp.oml, omb=cp.omb, camb=False, colos=True):
    zs = np.arange(zi, zf, dz)
    mass_tree = [[Mi]]
    for i in range(len(zs)-1):
        mass_tree_M = []
        for M in mass_tree[i]:
            if M > 5*Mres:
                R = np.random.uniform(0,1)
                S0 = sigma(M, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos)**2 #variance of the field at mass M
                w0 = cp.delta_c(zs[i], om0, ol0)  #critical density at observed redshift
                dw = cp.delta_c(zs[i+1], om0, ol0) - w0
                M1s = np.logspace(np.log10(Mres), np.log10(frac*M), acc)
                S1s = sigma(M1s, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos)**2 #variance of the field at mass M
                dS1 = (S1s[2:] - S1s[:-2])*0.5
                dM1 = (M1s[2:]-M1s[:-2])*0.5
                #dlsig = np.log(S1s[2:]) - np.log(S1s[:-2])*0.5
                #dlM =  np.log(M1s[2:]) - np.log(M1s[:-2])*0.5
                #dNdM1 = -M*S1s[1:-1]*(S1s[1:-1]-S0)**(-1.5)*dw*dlsig/(np.sqrt(2*np.pi)*dz*dlM*M1s**2)
                #dM1 = M1s[2:]-M1s[:-2]
                dNdM1 = -M*(S1s[1:-1]-S0)**(-1.5)*dw*dS1/(np.sqrt(2*np.pi)*dM1*M1s[1:-1])
                P = np.sum(dNdM1*dM1)
                Pcum = np.cumsum(dNdM1*dM1)
                M2s = np.logspace(-5, np.log10(Mres), acc)
                S2s = sigma(M2s, sig8, h, kmax, window, 'M', prec, om0, ol0, omb, camb, colos)**2
                #F = special.erfc(dw/np.sqrt(2*(S2s[0]-S2s[-1])))
                dS2 = (S2s[2:] - S2s[:-2])*0.5
                dM2 = (M2s[2:] - M2s[:-2])*0.5
                dNdM2 = -(S2s[1:-1]-S0)**(-1.5)*dw*dS2/(np.sqrt(2*np.pi)*dM2)
                F = np.sum(dNdM2*dM2)
                if R > P:
                    mass_tree_M.append(M*(1-F))
                else:
                    mass = np.min(M1s[1:-1][Pcum>R])
                    mass_tree_M.append(mass)
                    mass_tree_M.append(-mass + M*(1-F))
            else:
                 mass_tree_M.append(M)
        mass_tree.append(mass_tree_M)
        #print(np.sum(np.array(mass_tree[i+1]))/Mi)
    return mass_tree, zs

tree, reds = parkinson08(0.1, 1e13, 1e10, zf=5, dz=1e-4)

'''zi = 0.1
zf = 2
M = 1e13
Mres = 1e9
dz = 1e-3
mass_tree = [[M]]
i=4
mass_tree_M = []
for M in mass_tree[i]:
    S0 = sigma(M, Colos=True)**2 #variance of the field at mass M
    w0 = delta_c(zi, om0=om0)  #critical density at observed redshift
    dw = delta_c(zi+dz, om0=om0) - w0
    M1s = np.logspace(np.log10(Mres), np.log10(0.5*M), 10000)
    S1s = sigma(M1s, Colos=True)**2
    #dlsig = (np.log(S1s[2:]) - np.log(S1s[:-2]))*0.5
    dS1 = (S1s[2:] - S1s[:-2])*0.5
    dM1 = (M1s[2:]-M1s[:-2])*0.5
    #dlM =  (np.log(M1s[2:]) - np.log(M1s[:-2]))*0.5
    dNdM1 = -M*(S1s[1:-1]-S0)**(-1.5)*dw/(np.sqrt(2*np.pi)*M1s[1:-1])
    M2s = np.logspace(-22, np.log10(Mres), 1000000)
    S2s = sigma(M2s, Colos=True)**2
    dS2 = (S2s[2:] - S2s[:-2])*0.5
    dM2 = (M2s[2:] - M2s[:-2])*0.5
    dNdM2 = -(S2s[1:-1]-S0)**(-1.5)*dw/(np.sqrt(2*np.pi))
    #dNdM2 = -np.exp(-0.5*dw**2/(S2s[1:-1]-S0))*(S2s[1:-1]-S0)**(-1.5)*dw/(np.sqrt(2*np.pi))
    F = np.sum(dNdM2*dS2)
    #F = special.erfc(dw/np.sqrt(2*))
    P = np.sum(dNdM1*dS1)
    Pcum = np.cumsum(dNdM1*dS1)
    R = np.random.uniform(0,1)
    #mass_tree = [[M]]
    if R > P:
        mass_tree_M.append(M*(1-F))
    else:
        mass = np.min(M1s[1:-1][Pcum>R])
        mass_tree_M.append(mass)
        mass_tree_M.append(-mass + M*(1-F))
mass_tree.append(mass_tree_M)
print(len(mass_tree[i+1]))
#plt.plot(M1s[1:-1], dNdM1)
#plt.xscale('log')
#plt.show()'''

'''def cond(ds, dw):
    return dw*(np.pi*2)**(-0.5)*np.exp(-0.5*dw**2/ds)*ds**(-1.5)

def dfdz(ds, dw):
    return dw*(np.pi*2)**(-0.5)*ds**(-1.5)
sh = sigma(1, Colos=True)
s2 = sigma(1e14, Colos=True)

s1 = np.linspace(s2, sh, 100000)
deltas = s1-s2
ds = s1[1]-s1[0]
dw = np.logspace(-10, 0, 100)
res = []
res2 = []
y = special.erfc(dw/np.sqrt(2*(sh-s2)))
for el in dw:
    res.append(np.sum(cond(deltas[1:], el)*ds))
    res2.append(np.sum(dfdz(deltas[1:], el)*ds))
plt.plot(dw, res, label='non approx')
#plt.plot(dw, res2, label='approx')
plt.plot(dw, y, label = 'erfc')
plt.legend()
plt.show()'''