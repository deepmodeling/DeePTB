#!/usr/bin/env python3
import numpy  as np
import matplotlib.pyplot as plt

plt.figure(figsize=(6,6),dpi=100)
trans = np.load('transmission.npy',allow_pickle='True').tolist()
plt.plot(trans['E'],trans['T']['SourcetoDrain'].real,'r',lw=1)
plt.xlim(-3,3)
plt.ylim(0,8)
plt.tick_params(direction='in')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Energy (eV)',fontsize=12)
plt.ylabel('Transmission',fontsize=12)
plt.savefig('Transmission.png',dpi=300)

plt.figure(figsize=(6,6),dpi=100)
curr = np.load('current.npy',allow_pickle='True').tolist()
plt.plot(curr['bias'],curr['current'].real,'r-',lw=1)
inter=5
plt.plot(curr['bias'][::inter],curr['current'][::inter].real,'r^')
plt.xlim(0,6)
plt.ylim(0,20)
plt.tick_params(direction='in')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Bias (V)',fontsize=12)
plt.ylabel(r'Current  $(2e^2/h)$',fontsize=12)
plt.savefig('Current.png',dpi=300)