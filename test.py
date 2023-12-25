import numpy as np


def developQ(Q, dt, InvTau, b):
    InvTauEigenVal, InvTauEigenVec = np.linalg.eig(InvTau)
    Q_eigenbasis = InvTauEigenVec.dot(Q)
    b = InvTauEigenVec.dot(b)

    exponent = np.exp(InvTauEigenVal * dt)
    Q_eigenbasis = Q_eigenbasis * exponent + (b / InvTauEigenVal) * (exponent - 1)
    Q = np.linalg.solve(InvTauEigenVec, Q_eigenbasis)
    return Q


X = np.array([0])

X =

print(Q)
