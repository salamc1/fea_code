"""Code for solving a general 2D finite element problem"""
import numpy as np
import sympy as sp
from dataclasses import dataclass
from sympy import init_printing


@dataclass
class SymbolicElement:
    dimension: int
    order: int
    nodes: int
    iso_cord_vetor: sp.matrices.dense.MutableDenseMatrix
    d_cord_vector: sp.matrices.dense.MutableDenseMatrix
    
    def __post_init__(self):
        if self.dimension != 2:
            raise NotImplementedError
        if self.order != 1:
            raise NotImplementedError
        assert self.iso_cord_vetor.shape == self.d_cord_vector.shape

    def get_N_list(self):
        s, t = sp.symbols('s t')
        N1 = 0.25*(s-1)*(t-1)
        N2 = -0.25*(s+1)*(t-1)
        N3 = 0.25*(s+1)*(t+1)
        N4 = -0.25*(s-1)*(t+1)
        N = sp.Matrix([N1, N2, N3, N4])
        return N

    def get_N_matrix(self):
        s, t = sp.symbols('s t')
        N1 = 0.25*(s-1)*(t-1)
        N2 = -0.25*(s+1)*(t-1)
        N3 = 0.25*(s+1)*(t+1)
        N4 = -0.25*(s-1)*(t+1)
        N = sp.Matrix([[N1, 0, N2, 0, N3, 0, N4, 0],
                        [0, N1, 0, N2, 0, N3, 0, N4]])
        return N

    def get_x(self):
        s, t = sp.symbols('s t')
        N = self.get_N_matrix()
        return N*self.d_cord_vector

    def get_J(self):
        s, t = sp.symbols('s t')
        xvec = self.get_x()
        x = xvec[0]
        y = xvec[1]
        dxds = sp.diff(x, s)
        dxdt = sp.diff(x, t)
        dyds = sp.diff(y, s)
        dydt = sp.diff(y, t)
        return sp.Matrix([[dxds, dyds], [dxdt, dydt]])

    def get_dNiso(self):
        s, t = sp.symbols('s t')
        N = self.get_N_list()
        dN = sp.zeros(N.shape[0]*2, 1)
        for i in range(N.shape[0]*2):
            if i % 2 == 0:
                dN[i] = sp.diff(N[int(i/2)], s)
            else:
                dN[i] = sp.diff(N[int((i-1)/2)], t)
        return sp.Matrix(dN)

    def _get_2comp_dN(self, dN):
        J = self.get_J()
        Jinv = J.inv()
        return Jinv*dN

    def get_dN(self):
        dNiso = self.get_dNiso()
        dN = sp.zeros(8, 1)
        for i in range(4):
            dN[i], dN[i+1] = self._get_2comp_dN(sp.Matrix(dNiso[i:(i+2)]))
        return dN

    def get_B(self):
        dN = self.get_dN()
        B = sp.Matrix([[dN[0], 0, dN[2], 0, dN[4], 0, dN[6], 0], 
                        [0, dN[1], 0, dN[3], 0, dN[5], 0, dN[7]],
                        [dN[1], dN[0], dN[3], dN[2], dN[5], dN[4], dN[7], dN[6]]])
        return B
        

def main():
    init_printing()
    e = SymbolicElement(2, 1, 4, sp.Matrix([-1,-1,1,-1,1,1,-1,1]), sp.Matrix([0,0,10,0,8,10,2,7]))
    # print(e.get_N_matrix())
    # print(e.get_x())
    # print(e.get_J())
    print(e.get_B().evalf(subs={'s':0, 't':0}))

if __name__ == '__main__':
    main()
