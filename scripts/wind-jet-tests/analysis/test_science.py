import unittest
import numpy as np
from science import *

class DiagnosticsTests(unittest.TestCase):
    def test_gaussian_preserves_signed_constant_and_exposes_edge_width(self):
        x=np.linspace(0,1,21);k,width=gaussian_matrix(x,np.ones(21)*.05,.15)
        np.testing.assert_allclose(k@np.ones(21)*-2,-2,atol=1e-14)
        self.assertLess(width[0],width[10])
        self.assertTrue(np.allclose(k.sum(1),1))
    def test_filter_is_linear_not_product_of_filtered_means(self):
        x=np.linspace(0,1,21);k,_=gaussian_matrix(x,np.ones(21)*.05,.15)
        a=np.outer(np.sin(6*x),np.cos(4*x));b=a*a
        np.testing.assert_allclose(filter_field(a+b,k,k),filter_field(a,k,k)+filter_field(b,k,k),atol=1e-14)
        self.assertGreater(np.linalg.norm(filter_field(a*a,k,k)-filter_field(a,k,k)**2),.01)
    def test_native_correction_does_not_discard_2d_eddy_advection(self):
        a=np.array([[1.,2.],[-1.,3.]]);z=np.zeros_like(a)
        f={'Q':a,'Qdot_advection_full':a,'Qdot_advection_mean':a,'Qdot_wave':z,'Qdot_viscous':z,'Qdot_forcing':z,'Qdot_total':a}
        r=feedback_metrics(f,np.ones(2),q0=z,qt=a)
        self.assertEqual(r['native_3d_correction_rms'],0)
        self.assertGreater(r['native_mean_rhs_rms'],0)
        self.assertEqual(r['native_temporal_residual_rms'],0)
    def test_scalar_eddy_transport_is_retained(self):
        a=np.array([[1.,-1.],[2.,-2.]]);z=np.zeros_like(a)
        r=scalar_metrics({'Q':a,'Qt':-a,'A':a,'Rbar':z},np.ones(2))
        self.assertEqual(r['scalar_observed_rms'],0)
        self.assertEqual(r['rbar_to_scalar_evolution'],0)
    def test_forced_viscous_scalar_balance(self):
        a=np.ones((3,4));r=scalar_metrics({'Q':a,'Qt':a,'A':2*a,'Rbar':a,'Qdot_viscous':-a,'Qdot_forcing':3*a},np.ones(4))
        self.assertAlmostEqual(r['scalar_unclosed_rms'],0)
    def test_zero_denominator_is_undefined(self):
        self.assertIsNone(ratio(0,0));self.assertEqual(ratio(-1,2),-.5)
    def test_native_parseval_signed_mean_energy_supply(self):
        n=16;x=np.arange(n)*2*np.pi/n;u=np.sin(2*x)[:,None]*np.ones((1,3));w=np.zeros((n,4))
        r=x_mode_supply(u,w,-3*u,w,np.ones(3),2*np.pi)
        self.assertAlmostEqual(r['sum_supply'],-1.5)
        self.assertAlmostEqual(r['supply'][2],-1.5)
        self.assertAlmostEqual(r['cumulative_supply'][-1],-1.5)
    def test_nonuniform_temporal_derivative_quadratic(self):
        t=np.array([0.,1.,3.]);a=[np.ones((2,2))*s*s for s in t]
        np.testing.assert_allclose(centered_time_derivative(t,a,1),2)
        self.assertIsNone(centered_time_derivative(t,a,0))
    def test_transverse_growth_and_twin_collapse(self):
        t=np.arange(20.);e=1e-8*np.exp(.04*t);b=np.ones(20)
        r=transverse_growth(t,e,b,1e-8)
        self.assertAlmostEqual(r['amplitude_growth_rate_s_inv'],.02)
        twin=twin_collapse(t,e,1e-8,t,100*e,1e-6)
        np.testing.assert_allclose(twin['relative_difference'],0,atol=1e-15)
    def test_twin_disagreement_does_not_qualify_growth(self):
        t=np.arange(20.);e=1e-8*np.exp(.04*t);bad=1e-6*np.exp(.4*t)
        result=qualified_twin_growth(t,e,1e-8,np.ones(20),bad,1e-6,np.ones(20))
        self.assertFalse(result['available'])
        self.assertEqual(result['stop_reason'],'twin_amplitudes_do_not_collapse')
        good=qualified_twin_growth(t,e,1e-8,np.ones(20),100*e,1e-6,np.ones(20))
        self.assertTrue(good['available']);self.assertAlmostEqual(good['amplitude_growth_rate_s_inv'],.02)
    def test_paired_error_preserves_full_xz_eddies(self):
        a=np.array([[1.,-1.],[-1.,1.]]);z=np.zeros_like(a)
        f={'Uc':a,'Wc':a,'Q':a};b={'Uc':z,'Wc':z,'Q':z}
        r=paired_fields(f,b,np.ones(2),b,1)
        self.assertAlmostEqual(r['inplane_error_rms'],np.sqrt(2))
        self.assertEqual(r['jet_profile_error_rms'],0)

if __name__=='__main__':unittest.main()
