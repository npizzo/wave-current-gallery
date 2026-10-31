import unittest
import numpy as np
from shear_stability import operator,leading_rate,native_shear


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.h=.5;self.z=-self.h+(np.arange(192)+.5)*self.h/192
    def test_inviscid_opposite_shear_conserves_energy(self):
        # When Q=-Us_z constant, the exact perturbation energy is conserved.
        a=operator(self.z,np.full(192,-.2),np.full(192,.2),12,self.h,0,n=24)
        self.assertLess(np.max(abs(a+a.T)),1e-13)
        self.assertLess(np.max(np.linalg.eigvals(a).real),1e-12)
    def test_viscosity_damps_opposite_shear(self):
        g,_=leading_rate(self.z,np.full(192,-.2),np.full(192,.2),12,self.h,1e-4,n=24)
        self.assertLessEqual(g,-1e-4*12**2+1e-11)
    def test_aligned_shear_bounded_inviscid_growth(self):
        # The inviscid bounded vertical sine mode has this exact growth; the
        # complete cosine representation of u converges to it in L2.
        expected=np.sqrt(.2*.4)*12/np.sqrt(12**2+(np.pi/self.h)**2)
        g,_=leading_rate(self.z,np.full(192,.2),np.full(192,.4),12,self.h,0,n=48)
        self.assertLess(abs(g/expected-1),1e-4)
    def test_no_eulerian_shear_has_no_exponential_growth(self):
        g,_=leading_rate(self.z,np.zeros(192),np.full(192,.4),12,self.h,0,n=24)
        self.assertLess(abs(g),1e-12)
    def test_constant_current_offset_does_not_change_Q(self):
        args=(self.h,6.1,.036649447,.75,.036649447,.002,3.333333333e-6)
        a=native_shear(-.75*.036649447*np.exp(12.2*self.z),*args)
        b=native_shear(7-.75*.036649447*np.exp(12.2*self.z),*args)
        np.testing.assert_allclose(a[0],b[0],atol=2e-12,rtol=1e-10)
        self.assertGreater(a[2][1],0)
    def test_boundary_wind_contribution(self):
        q,b,bounds=native_shear(np.zeros(192),.5,6.1,.036649447,.75,.036649447,.002,3.333333333e-6)
        self.assertAlmostEqual(bounds[1],-.75*12.2*.036649447+1.2,places=8)
    def test_quadrature_refinement(self):
        q=.1*np.exp(12.2*self.z)-.02;s=.4*np.exp(12.2*self.z)
        a=leading_rate(self.z,q,s,24,.5,3.33e-6,n=24,quadrature=384)[0]
        b=leading_rate(self.z,q,s,24,.5,3.33e-6,n=24,quadrature=768)[0]
        self.assertLess(abs(a-b),2e-4)


if __name__=='__main__':unittest.main()
