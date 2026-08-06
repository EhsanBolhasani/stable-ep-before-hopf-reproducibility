#!/usr/bin/env python3
"""Exploratory, reproducible calculations for M-population OA rings.

The calculations test the normalization K0=(1+A)/2, K1=(1-A)/4, for which
the alternating M=4 subspace reproduces the standard two-population
benchmark exactly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import null_space, schur
from scipy.optimize import root


HERE = Path(__file__).resolve().parent


def coupling(M: int, A: float, eta: float) -> np.ndarray:
    """Target-row/source-column nearest-neighbor coupling matrix."""
    mu = (1.0 + A) / 2.0
    k1 = (1.0 - A) / 4.0
    K = np.zeros((M, M))
    for s in range(M):
        K[s, s] = mu
        K[s, (s + 1) % M] = k1 * (1.0 + eta)
        K[s, (s - 1) % M] = k1 * (1.0 - eta)
    return K


def oa_complex(z: np.ndarray, M: int, A: float, beta: float, eta: float,
               Delta: float = 0.0, omega: float = 0.0) -> np.ndarray:
    K = coupling(M, A, eta)
    alpha = np.pi / 2.0 - beta
    H = K @ z
    Hc = K @ np.conj(z)
    return (-Delta + 1j * omega) * z + 0.5 * (
        np.exp(-1j * alpha) * H - np.exp(1j * alpha) * Hc * z**2
    )


def real_rhs(t: float, y: np.ndarray, M: int, A: float, beta: float,
             eta: float, Delta: float = 0.0) -> np.ndarray:
    z = y[:M] + 1j * y[M:]
    f = oa_complex(z, M, A, beta, eta, Delta)
    return np.r_[f.real, f.imag]


def two_rhs(x: np.ndarray, A: float, beta: float) -> np.ndarray:
    r, psi = x
    mu = (1 + A) / 2
    nu = (1 - A) / 2
    dr = (1 - r*r) / 2 * (mu*r*np.sin(beta) + nu*np.sin(psi + beta))
    dp = (-mu*np.cos(beta) - nu*r*np.cos(beta-psi)
          + (1+r*r)/(2*r)*(mu*r*np.cos(beta) + nu*np.cos(psi+beta)))
    return np.array([dr, dp])


def two_equilibrium(A: float, beta: float, guess=(0.8, -0.2)) -> np.ndarray:
    sol = root(lambda x: two_rhs(x, A, beta), guess, tol=1e-12)
    if not sol.success or np.linalg.norm(two_rhs(sol.x, A, beta)) > 1e-9:
        raise RuntimeError(sol.message)
    return sol.x


def alternating_m4(A: float, beta: float, eta: float,
                   guess=(0.8, -0.2)) -> tuple[np.ndarray, float]:
    """Exact alternating relative equilibrium for the M=4 ring."""
    r, psi = two_equilibrium(A, beta, guess=guess)
    z = np.array([np.exp(1j*psi), r+0j, np.exp(1j*psi), r+0j])
    mu = (1 + A) / 2
    nu = (1 - A) / 2
    Om = -(1+r*r)/(2*r) * (mu*r*np.cos(beta) + nu*np.cos(psi+beta))
    err = np.linalg.norm(oa_complex(z, 4, A, beta, eta) - 1j*Om*z)
    if err > 1e-8:
        raise RuntimeError(f"embedded-state residual {err}")
    return z, Om


def uniform_relative_equilibrium(M: int, beta: float, Delta: float) -> tuple[np.ndarray, float]:
    """Uniform OA relative equilibrium when sin(beta)>2*Delta."""
    r2 = 1.0 - 2.0 * Delta / np.sin(beta)
    if r2 <= 0:
        raise ValueError("uniform coherent state does not exist")
    r = np.sqrt(r2)
    z = np.full(M, r + 0j)
    Om = -0.5 * (1.0 + r2) * np.cos(beta)
    return z, Om


def jacobian_real(fun, y: np.ndarray, h: float = 2e-6) -> np.ndarray:
    n = y.size
    J = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n); e[j] = h
        J[:, j] = (fun(y + e) - fun(y - e)) / (2*h)
    return J


def quotient_jacobian(z: np.ndarray, Om: float, M: int, A: float, beta: float,
                      eta: float, Delta: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal phase quotient inherited from Euclidean complex-z metric."""
    y = np.r_[z.real, z.imag]
    def corot(v):
        zz = v[:M] + 1j*v[M:]
        f = oa_complex(zz, M, A, beta, eta, Delta) - 1j*Om*zz
        return np.r_[f.real, f.imag]
    J = jacobian_real(corot, y)
    u = np.r_[-z.imag, z.real]
    E = null_space(u[None, :])
    C = E.T @ J @ E
    return C, E


def schur_pair_block(C: np.ndarray, target: complex) -> tuple[np.ndarray, np.ndarray]:
    """Return orthonormal real invariant plane closest to target pair.

    For complex pairs scipy's real Schur form directly supplies a 2x2 block.
    For two real eigenvalues, the span of the two right eigenvectors is
    orthonormalized and restricted.  This is sufficient away from exact
    degeneracy and is used only for exploration here.
    """
    vals, vecs = np.linalg.eig(C)
    # Select the two eigenvalues nearest target and its conjugate, without reuse.
    i = int(np.argmin(abs(vals - target)))
    rem = [j for j in range(len(vals)) if j != i]
    j = rem[int(np.argmin(abs(vals[rem] - np.conj(target))))]
    V = vecs[:, [i, j]]
    if np.max(abs(V.imag)) < 1e-7:
        Q, _ = np.linalg.qr(V.real)
    elif abs(vals[j] - np.conj(vals[i])) < 1e-6:
        Q, _ = np.linalg.qr(np.column_stack([vecs[:, i].real, vecs[:, i].imag]))
    else:
        Q, _ = np.linalg.qr(np.column_stack([V.real[:, 0], V.real[:, 1]]))
    B = Q.T @ C @ Q
    return B, Q


def real_schur_pair_block(C: np.ndarray, target: complex) -> tuple[np.ndarray, np.ndarray]:
    """Return an ordered real-Schur block for an isolated target pair."""
    vals = np.linalg.eigvals(C)
    dist = np.minimum(abs(vals - target), abs(vals - np.conj(target)))
    order = np.sort(dist)
    if len(order) < 3 or not order[2] > order[1] + 100 * np.finfo(float).eps:
        raise ValueError("target pair is not spectrally isolated")
    cutoff = (order[1] + order[2]) / 2

    def select(wr: float, wi: float) -> bool:
        lam = complex(wr, wi)
        return min(abs(lam - target), abs(lam - np.conj(target))) < cutoff

    T, Q, selected = schur(C, output="real", sort=select)
    if selected != 2:
        raise RuntimeError(
            f"ordered real Schur selected {selected} modes, expected 2"
        )
    return T[:2, :2], Q[:, :2]


def embedded_scan(beta: float = 0.1, eta: float = 0.2) -> None:
    rows = []
    guess=np.array([.815,-.217])
    for A in np.linspace(0.17836, 0.285, 250):
        z, Om = alternating_m4(A, beta, eta,guess=guess)
        guess=np.array([abs(z[1]),np.angle(z[0])])
        C, _ = quotient_jacobian(z, Om, 4, A, beta, eta)
        vals = np.linalg.eigvals(C)
        J2=jacobian_real(lambda xx: two_rhs(xx,A,beta),guess)
        target_vals=np.linalg.eigvals(J2)
        # Assignment of two full-system eigenvalues to the exact invariant-slice pair.
        i=int(np.argmin(abs(vals-target_vals[0])))
        rem=[j for j in range(len(vals)) if j!=i]
        j=rem[int(np.argmin(abs(vals[rem]-target_vals[1])))]
        pair_idx=np.array([i,j]); pair=vals[pair_idx]
        B, _ = schur_pair_block(C, target_vals[np.argmax(target_vals.imag)])
        a,b,c,d = B[0,0],B[0,1],B[1,0],B[1,1]
        s=(b+c)/2; g=(b-c)/2
        D=(a-d)**2+4*b*c
        others=np.delete(vals,pair_idx)
        rows.append([A, *np.sort_complex(pair), np.max(others.real), a,b,c,d,s,g,D,*guess])
    arr=np.asarray(rows, dtype=complex)
    header="A,pair1,pair2,maxRe_other,a,b,c,d,s,g,D,r,psi"
    np.savetxt(HERE/'embedded_m4_scan.csv', arr, delimiter=',', header=header, comments='', fmt='%s')
    print("A range",arr[0,0],arr[-1,0])
    print("minimum |D|",arr[np.argmin(abs(arr[:,10])),[0,10]])
    print("max other real",arr[:,3].real.max())
    for A0 in [0.17836,0.17845421,0.20,0.27777548,0.285]:
        k=np.argmin(abs(arr[:,0].real-A0)); print(arr[k])


def classify_summary(M: int, A: float, beta: float, eta: float, Delta: float,
                     seed: int, T: float = 1200.0, dt_sample: float = 0.2):
    rng=np.random.default_rng(seed)
    rho=rng.uniform(0.15,0.95,M)
    phi=rng.uniform(-np.pi,np.pi,M)
    z0=rho*np.exp(1j*phi)
    y0=np.r_[z0.real,z0.imag]
    ts=np.arange(0,T+dt_sample/2,dt_sample)
    sol=solve_ivp(real_rhs,(0,T),y0,t_eval=ts,args=(M,A,beta,eta,Delta),
                  method='DOP853',rtol=2e-10,atol=2e-12,max_step=0.2)
    z=sol.y[:M]+1j*sol.y[M:]
    rho=np.abs(z[:,sol.t>=T*0.6])
    means=rho.mean(axis=1); std=rho.std(axis=1)
    speed=np.linalg.norm(np.diff(rho,axis=1),axis=0)/dt_sample
    return dict(means=means,std=std,Srho=means.max()-means.min(),
                Sinst=np.mean(rho.max(axis=0)-rho.min(axis=0)),
                speed_mean=speed.mean(),speed_max=speed.max(),
                rho_final=rho[:,-1])


def sample_grid() -> None:
    for Delta in [0.0,0.01,0.02]:
        print("DELTA",Delta)
        for M in [4,5]:
            for beta,eta in [(0.1,0.0),(0.1,0.1),(0.1,0.2),(0.2,0.0),(0.2,0.1),(0.25,0.05)]:
                q=classify_summary(M,.33,beta,eta,Delta,seed=1234+M,T=400)
                print(M,beta,eta,"S",q['Srho'],q['Sinst'],"std",q['std'].max(),"rho",q['rho_final'])


def uniform_spectrum_scan() -> None:
    for M in [4,5]:
        print("M",M)
        for beta in [.06,.08,.10,.15,.20,.25,.30]:
            row=[]
            for eta in np.linspace(0,1,101):
                try:
                    z,O=uniform_relative_equilibrium(M,beta,.02)
                except ValueError:
                    continue
                C,_=quotient_jacobian(z,O,M,.33,beta,eta,.02)
                ev=np.linalg.eigvals(C)
                row.append((eta,ev.real.max(),sum(abs(ev.imag)>1e-7),ev))
            if not row: continue
            k=min(range(len(row)),key=lambda i: abs(row[i][1]))
            print(" beta",beta,"maxRe eta0/1",row[0][1],row[-1][1],"near",row[k][0],row[k][1],"ncomplex",row[k][2])
            print("   eta0",np.sort_complex(row[0][3]))
            print("   near",np.sort_complex(row[k][3]))


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['embedded','grid','uniform'])
    ns=ap.parse_args()
    {'embedded':embedded_scan,'grid':sample_grid,'uniform':uniform_spectrum_scan}[ns.mode]()
