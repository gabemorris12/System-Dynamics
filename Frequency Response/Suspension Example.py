#!/usr/bin/env python
# coding: utf-8

# In[1]:


import control as ct
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('../maroon_py.mplstyle')

# From lecture 17, we are finding the frequency response of the two mass suspension model whose equations are

# In[2]:

m1_, m2_ = 250, 25
k1_, k2_ = 1.0975e4, 1e5
c1_ = 943

# In[3]:


# In[4]:

den = [m1_*m2_, c1_*m1_ + c1_*m2_, k1_*m1_ + k1_*m2_ + k2_*m1_, c1_*k2_, k1_*k2_]

T1 = ct.tf([c1_*k2_, k2_*k1_], den)
T2 = ct.tf([k2_*m1_, k2_*c1_, k2_*k1_], den)

# In[7]:

# Getting the frequency response
omegas = np.linspace(1, 100, 10_000)
mag1, phase1, _ = ct.frequency_response(T1, omegas)
mag2, phase2, _ = ct.frequency_response(T2, omegas)

# The phase1 output does not go more negative than 180, so we need to fix that here
phase1[phase1 > 0] = phase1[phase1 > 0] - 2*np.pi

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
ax1.set_xscale('log')
ax2.set_xscale('log')

ax1.plot(omegas, mag1, label='$x_1(t)$')
ax1.plot(omegas, mag2, label='$x_2(t)$')
ax2.plot(omegas, np.rad2deg(phase1), label='$x_1(t)$')
ax2.plot(omegas, np.rad2deg(phase2), label='$x_2(t)$')

ax1.set_ylabel(r'$M(\omega)$')
ax2.set_ylabel(r'$\phi(omega)$ ($deg$)')
ax2.set_xlabel(r'$\omega$ ($rad/s$)')
ax1.legend()
ax2.legend()
plt.show()
