"""SHADOWOPT: compatibility access paths and acquisition planning over a store of randomized Pauli measurements.

Conventions (fixed for the whole package):
  * basis codes: 0 = X, 1 = Y, 2 = Z (same as PennyLane recipes); identity in a term is -1 / 'I'.
  * outcome bit o = 0 means eigenvalue +1, o = 1 means eigenvalue -1.
  * qubit q of an n-qubit statevector is bit q of the basis-state index (little-endian).
"""
BASIS_CHARS = "XYZ"
