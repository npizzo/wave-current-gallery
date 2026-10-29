import copy
import unittest
import numpy as np
from wind_science import (closure_window, modal_response, profile_properties,
                          common_time_indices, DEFAULT_CRITERIA)


def synthetic_forecast():
    nx, nz = 8, 24
    t = np.arange(600., 801., 10.)
    z = -.5+(np.arange(nz)+.5)*.5/nz; dz = np.full(nz, .5/nz)
    x = np.arange(nx)*2*np.pi/nx
    q0 = np.sin(x)[:,None]*np.cos(np.pi*(z+.5)/.5)[None,:]*.02
    q = np.array([q0+(.0001*i)*np.sin(x)[:,None] for i in range(len(t))])
    u = np.array([.004*np.cos(x)[:,None]*np.ones((1,nz))+i*.00001*np.cos(np.pi*(z+.5)/.5)[None,:] for i in range(len(t))])
    parent = dict(time=t, z=z, dz=dz, Q=q, Uc=u, C=np.zeros_like(q))
    config = dict(k=6.1,eddy_rms=.004,injection_wavelength=.25)
    return parent, copy.deepcopy(parent), config


class ClosureTests(unittest.TestCase):
    def test_exact_reduction_passes_both_regions(self):
        parent, forecast, config = synthetic_forecast()
        result, history = closure_window(parent,forecast,config)
        self.assertEqual(result['assessment'],'PASS')
        self.assertTrue(all(r['epsilon_C']==0 for r in result['regions']))
        self.assertEqual(len(history),3*len(parent['time']))

    def test_background_does_not_change_normalization(self):
        parent,forecast,config=synthetic_forecast()
        a,_=closure_window(parent,forecast,config)
        shear=100*np.exp(parent['z'])
        parent['Q']+=shear[None,None,:];forecast['Q']+=shear[None,None,:]
        b,_=closure_window(parent,forecast,config)
        for x,y in zip(a['regions'],b['regions']):
            self.assertAlmostEqual(x['q_reference_s_inverse'],y['q_reference_s_inverse'],places=12)

    def test_signed_cancellation_does_not_hide_correction(self):
        parent,forecast,config=synthetic_forecast()
        parent['C']=np.sign(np.sin(np.arange(len(parent['time']))))[:,None,None]*np.ones_like(parent['Q'])*.01
        result,_=closure_window(parent,forecast,config)
        self.assertEqual(result['assessment'],'INCONCLUSIVE')
        self.assertTrue(all(r['epsilon_C']>1 for r in result['regions']))
        self.assertTrue(all('Outside' in r['reason'] for r in result['regions']))

    def test_weak_correction_bad_forecast_requires_refinement(self):
        parent,forecast,config=synthetic_forecast()
        forecast['Q'][1:]+=1
        result,_=closure_window(parent,forecast,config)
        self.assertEqual(result['assessment'],'INCONCLUSIVE')
        self.assertTrue(all(r['apparent_failure'] for r in result['regions']))

    def test_quiet_current_not_pass(self):
        parent,forecast,config=synthetic_forecast()
        parent['Uc'][:]=parent['Uc'][0];forecast['Uc'][:]=forecast['Uc'][0]
        result,_=closure_window(parent,forecast,config)
        self.assertEqual(result['assessment'],'INCONCLUSIVE')
        self.assertTrue(all(not r['active'] for r in result['regions']))

    def test_initial_state_mismatch_is_detected(self):
        parent,forecast,config=synthetic_forecast();forecast['Uc'][0]+=.01
        result,_=closure_window(parent,forecast,config)
        self.assertTrue(all(not r['initial_mean_matches'] for r in result['regions']))

    def test_incomplete_window_rejected(self):
        parent,forecast,config=synthetic_forecast()
        for key in ('time','Q','Uc','C'):forecast[key]=forecast[key][:-1]
        with self.assertRaises(ValueError):closure_window(parent,forecast,config)

    def test_native_map_timing_roundoff(self):
        a,b=common_time_indices([0,10,20],[0,10+1e-10,20],0,20)
        np.testing.assert_array_equal(a,[0,1,2]);np.testing.assert_array_equal(b,[0,1,2])


def synthetic_probe(gamma=.002, mode=2):
    t=np.arange(600.,801.,5.);z=-.5+(np.arange(24)+.5)*.5/24;dz=np.full(24,.5/24)
    phi=np.cos(mode*np.pi*(z+.5)/.5);amp=.0002;A=np.exp(gamma*(t-t[0]))
    plus=dict(time=t,z=z,dz=dz);minus=dict(time=t,z=z,dz=dz)
    for sign, branch in ((1,plus),(-1,minus)):
        branch['UE']=np.exp(2*z)[None,:]+sign*amp*A[:,None]*phi[None,:]
        for name in ('adv_full','adv_ky0','adv_3d','closure','forcing','wave','pressure_mean'):
            branch[name]=np.zeros_like(branch['UE'])
        branch['adv_full']=sign*amp*gamma*A[:,None]*phi[None,:]
        branch['adv_ky0']=branch['adv_full'].copy()
    return plus,minus,amp


class ResponseTests(unittest.TestCase):
    def test_growing_budgeted_response(self):
        p,m,a=synthetic_probe();r,rows=modal_response(p,m,2,a,.5)
        self.assertEqual(r['assessment'],'POSITIVE_RESPONSE_CANDIDATE')
        self.assertTrue(r['eddy_reinforcement'])
        self.assertAlmostEqual(r['descriptive_log_slope_s_inverse'],.002,places=10)
        self.assertLess(r['response_budget_relative_residual'],.001)

    def test_decay_not_claimed_jet_failure(self):
        p,m,a=synthetic_probe(gamma=-.002);r,_=modal_response(p,m,2,a,.5)
        self.assertEqual(r['assessment'],'NO_POSITIVE_RESPONSE_DEMONSTRATED')
        self.assertFalse(r['eddy_reinforcement'])

    def test_common_wind_response_cancels(self):
        p,m,a=synthetic_probe();r,_=modal_response(p,m,2,a,.5)
        p['UE']+=np.arange(len(p['time']))[:,None]*.2;m['UE']+=np.arange(len(m['time']))[:,None]*.2
        r2,_=modal_response(p,m,2,a,.5)
        self.assertAlmostEqual(r['final_absolute_gain'],r2['final_absolute_gain'],places=9)

    def test_missing_budget_rejects_growth(self):
        p,m,a=synthetic_probe();p['adv_full'][:]=0;m['adv_full'][:]=0
        r,_=modal_response(p,m,2,a,.5)
        self.assertEqual(r['assessment'],'INCONCLUSIVE')

    def test_monotonic_profile_not_interior_jet(self):
        z=np.linspace(-.5,0,24);r=profile_properties(z,z,np.ones(24)*z,np.full(24,.5/24))
        self.assertEqual(r['interior_maxima'],[]);self.assertEqual(r['interior_minima'],[])


if __name__=='__main__':unittest.main()
