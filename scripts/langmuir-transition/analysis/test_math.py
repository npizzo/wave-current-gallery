import unittest
import numpy as np
from analyze import terms,dx,dz
class Math(unittest.TestCase):
 def test_zero_feedback_for_y_invariant_fields(self):
  x=np.arange(64)[:,None]*2*np.pi/64;z=np.linspace(-1,0,65);U=np.sin(x)*z[None,:];W=np.cos(x)*z[None,:]**2
  d=dict(U=U,W=W,UU=U*U,WW=W*W,UW=U*W)
  t=terms(d,z,2*np.pi/64,2.,.03)
  self.assertEqual(np.max(abs(t['R'])),0)
 def test_manufactured_covariance_curl(self):
  n=256;x=np.arange(n)[:,None]*2*np.pi/n;z=np.linspace(-1,0,129);rxx=2+np.sin(x)*np.exp(z);rzz=np.ones_like(rxx);rxz=.1*np.cos(x)*np.exp(z)
  d=dict(U=np.zeros_like(rxx),W=np.zeros_like(rxx),UU=rxx,WW=rzz,UW=rxz)
  R=terms(d,z,2*np.pi/n,2,.03)['R'];expected=1.2*np.cos(x)*np.exp(z)
  self.assertLess(np.linalg.norm((R-expected)[:,3:-3])/np.linalg.norm(expected[:,3:-3]),.001)
 def test_stokes_background_sign(self):
  z=np.linspace(-.4,0,129);k=9.3;S=.05;U=np.broadcast_to(3*S*np.exp(2*k*z),(32,len(z)));W=np.zeros_like(U)
  Q=terms(dict(U=U,W=W,UU=U*U,WW=W,UW=W),z,1.,k,S)['Q'];expected=4*k*S*np.exp(2*k*z)
  self.assertLess(np.max(abs(Q[:,3:-3]-expected[None,3:-3]))/expected.max(),.003)
if __name__=='__main__':unittest.main()
