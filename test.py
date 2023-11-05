import numpy as np
import Conditions as Cond


def developQ(Q, dt, InvTau, InvC, n_prime, Rg):
    InvTauEigenVal, InvTauEigenVec = np.linalg.eig(InvTau)
    Q_eigenbasis = InvTauEigenVec.dot(Q)
    b = -InvTauEigenVec.dot(InvC.dot(n_prime) / Rg)

    exponent = np.exp(InvTauEigenVal * dt)
    Q_eigenbasis = Q_eigenbasis * exponent + (b / InvTauEigenVal) * (exponent - 1)
    Q = np.linalg.solve(InvTau, Q_eigenbasis)
    return Q


Q = np.array([1]*16)

Q = developQ(Q, 0.1, Cond.tau_inv_matrix, Cond.C_inverse,
             np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1]) + Cond.VxCix(0.04, 0), Cond.Rg)

print(Q)
