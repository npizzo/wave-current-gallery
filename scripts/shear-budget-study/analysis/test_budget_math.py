import unittest
import numpy as np
from budget_math import cumulative,field_closure,moment_budget,mode_budget,identity_error


class BudgetMathTests(unittest.TestCase):
    def test_known_2d_source_is_zero(self):
        t=np.linspace(0,2,21);x=np.sin(np.arange(8)*2*np.pi/8)[:,None];z=np.linspace(1,2,6)[None,:];shape=x*z
        q=np.broadcast_to(shape,(len(t),8,6));zero=np.zeros_like(q)
        r=moment_budget(q,dict(A0=zero,C=zero,W=zero,D=zero,F=zero),t)
        np.testing.assert_array_equal(r['power']['eddy']['C'],0)
        np.testing.assert_array_equal(r['integrals']['eddy']['C'],0)
        self.assertEqual(r['closure']['eddy']['relative_to_change_and_gross'],0)
    def test_signed_direct_source_and_exact_closure(self):
        t=np.linspace(0,2,21);shape=np.sin(np.arange(8)*2*np.pi/8)[:,None]*np.ones((1,6))
        q=(1-.1*t[:,None,None])*shape;src=np.broadcast_to(-.1*shape,q.shape);zero=np.zeros_like(q)
        terms=dict(A0=zero,C=src,W=zero,D=zero,F=zero)
        self.assertLess(field_closure(q,terms,t)['relative_to_change'],1e-14)
        r=moment_budget(q,terms,t);self.assertTrue(np.all(r['power']['eddy']['C']<0))
        np.testing.assert_allclose(r['energy']['eddy']-r['energy']['eddy'][0],r['integrals']['eddy']['C'],atol=1e-15)
    def test_cancellation_not_called_absence_of_sources(self):
        t=np.arange(3.);q=np.ones((3,4,5));a=np.full_like(q,2);c=-a;z=np.zeros_like(q)
        r=moment_budget(q,dict(A0=a,C=c,W=z,D=z,F=z),t)
        self.assertEqual(r['integrals']['Q']['C'][-1],-4)
        self.assertEqual(r['integrals']['Q']['A0'][-1],4)
        self.assertEqual(r['integrals']['Q']['total'][-1],0)
    def test_profile_eddy_enstrophy_partition(self):
        rng=np.random.default_rng(2);q=rng.normal(size=(3,8,5));z=np.zeros_like(q)
        r=moment_budget(q,dict(A0=z,C=z,W=z,D=z,F=z),np.arange(3.))
        np.testing.assert_allclose(r['energy']['Q'],r['energy']['eddy']+r['energy']['profile'],atol=1e-15)
    def test_jet_mode_budget_with_nonzero_initial_amplitude(self):
        t=np.linspace(0,3,31);a=np.stack((2-.2*t,1+.1*t),axis=1);c=np.broadcast_to([-.2,.1],a.shape);z=np.zeros_like(a)
        r=mode_budget(a,dict(A0=z,C=c,W=z,D=z,F=z),t,np.array([.5,.5]))
        self.assertLess(max(r['amplitude_relative_error']),1e-14);self.assertLess(max(r['energy_relative_error']),1e-14)
        self.assertLess(r['work']['C'][-1,0],0);self.assertGreater(r['work']['C'][-1,1],0)
    def test_wrong_identity_is_detected(self):
        z=np.zeros((4,5));fields=dict(A0=z,C=z,W=z,D=z,F=z,total=z+1)
        self.assertEqual(identity_error(fields)['relative'],1)


if __name__=='__main__':unittest.main()
