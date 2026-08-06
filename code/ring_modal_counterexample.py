#!/usr/bin/env python3
"""A reproducible M=4,5 modal reduction showing why g is not an EP test.

Parameters: A=0.33, beta=0.30, Delta=0.02, omega=0,
K0=(1+A)/2 and K1=(1-A)/4.  The uniform coherent relative equilibrium is
stable over eta in [0,1].  Directionality turns a reflection-symmetry doublet
into a normal rotational 2x2 block B=aI+g[[0,1],[-1,0]].  Thus |g| is a
well-defined projected circulation in the Euclidean z metric, while eta=0 is
a scalar, semisimple degeneracy rather than an exceptional point.
"""

from pathlib import Path
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0,str(Path(__file__).parent))
from explore_rings import (
    quotient_jacobian,
    real_schur_pair_block,
    uniform_relative_equilibrium,
)

HERE=Path(__file__).resolve().parent
DATA=HERE.parent/'data'
REPORTS=HERE.parent/'reports'
DATA.mkdir(parents=True,exist_ok=True)
REPORTS.mkdir(parents=True,exist_ok=True)
A=.33; beta=.30; Delta=.02
etas=np.linspace(0,1,201)
allrows=[]
fig,axs=plt.subplots(1,3,figsize=(9.0,2.8),layout='constrained')
colors={4:'#0072B2',5:'#D55E00'}
for M in [4,5]:
    rows=[];z,Om=uniform_relative_equilibrium(M,beta,Delta)
    for eta in etas:
        C,_=quotient_jacobian(z,Om,M,A,beta,eta,Delta)
        ev=np.linalg.eigvals(C)
        target=ev[np.argmax(ev.real+1e-8*ev.imag)]
        B,_=real_schur_pair_block(C,target)
        a,b,c,d=B[0,0],B[0,1],B[1,0],B[1,1]
        g=(b-c)/2;s=(b+c)/2;D=(a-d)**2+4*b*c
        vals,V=np.linalg.eig(B)
        rows.append([M,eta,target.real,abs(target.imag),g,s,D,np.linalg.cond(V),ev.real.max()])
    rows=np.asarray(rows);allrows.append(rows)
    lab=fr'$M={M}$'
    axs[0].plot(rows[:,1],rows[:,2],color=colors[M],label=lab)
    axs[1].plot(rows[:,1],abs(rows[:,4]),color=colors[M],label=lab)
    axs[2].plot(rows[1:,1],rows[1:,7],color=colors[M],label=lab)
np.savetxt(DATA/'ring_modal_counterexample.csv',np.vstack(allrows),delimiter=',',
           header='M,eta,Re_lambda,abs_Im_lambda,g,s,D,cond_eigenvectors,max_Re_full',comments='')
axs[0].axhline(0,color='k',lw=.8);axs[0].set(xlabel=r'directionality $\eta$',ylabel=r'$\Re\lambda$')
axs[1].set(xlabel=r'directionality $\eta$',ylabel=r'projected $|g|$')
axs[2].set(xlabel=r'directionality $\eta$',ylabel=r'eigenvector condition $\kappa_2(V)$',ylim=(.99999994,1.00000006))
for ax,l in zip(axs,['(a)','(b)','(c)']):
    ax.grid(alpha=.2);ax.legend(frameon=False);ax.text(-.18,1.04,l,transform=ax.transAxes,fontweight='bold')
fig.savefig(REPORTS/'ring_modal_counterexample.pdf')
fig.savefig(REPORTS/'ring_modal_counterexample.png',dpi=300)
