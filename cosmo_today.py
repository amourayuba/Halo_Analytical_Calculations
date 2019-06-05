from __future__ import division
import numpy as np

G = 4.30091e-9 #Units kpc/Msun x (km/s)^2
H_0 = 70    #km/s/Mpc
h = H_0/100
omega_m0 = 0.299
omega_l0 = 0.7
omega_r0 = 0.001
omega_0 = omega_m0 + omega_r0 + omega_l0
rho_c = 3*100**2/(8*np.pi*G)       # h^2xMsun/Mpc**3
