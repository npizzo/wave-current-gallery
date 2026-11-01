import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import forced_vorticity_identity,check_signed_vorticity_rows,representative_cases


def signed_fixture():
    times=np.array([0.,.1,.2]);data={'time_s':times,'interval_dt':np.array([0.,.1,.1])}
    for region in ('full','interior','stokes'):
        for part,weight in (('Q',2.),('profile',1.),('eddy',1.)):
            data[region+'_Z_'+part]=weight*(1+times)
            for term in ('A0','C','W','D','F','total'):
                power=weight if term in ('A0','total') else 0.
                data[region+'_P_'+part+'_'+term]=np.full(3,power)
                data[region+'_I_'+part+'_'+term]=power*times
            data[region+'_balance_residual_'+part]=np.zeros(3)
            data[region+'_split_identity_'+part]=np.zeros(3)
        for term in ('A0','C','W','D','F','total'):
            data[region+'_source_rms_'+term]=np.full(3,1. if term in ('A0','total') else 0.)
        data[region+'_enstrophy_decomposition_error']=np.zeros(3)
    array=np.empty(3,dtype=[(name,float) for name in data])
    for name,value in data.items():array[name]=value
    return array


class ForcedVorticityOperationTests(unittest.TestCase):
    def test_representative_selection_retains_refinement_projection_and_timestep(self):
        def case(name,source,kind,wind,dt=.02):
            return dict(id=name,source_id=source,initial_kind=kind,env=dict(WIND_USTAR=wind,MAX_DT=dt))
        cases=[case('coarse-low','coarse','full',.001),case('coarse-high','coarse','full',.0015),
               case('coarse-mean','coarse','mean',.00125),case('fine-low','fine','full',0.),
               case('fine-high','fine','full',.002),case('fine-mean','fine','mean',.002),
               case('fine-halfdt-low','fine','full',.00125,.01),case('fine-halfdt-high','fine','full',.002,.01)]
        selected={case['id'] for case in representative_cases(cases)}
        self.assertEqual(selected,{'coarse-high','coarse-mean','fine-high','fine-mean','fine-halfdt-high'})

    def test_signed_temporal_budget_closes_independently(self):
        data=signed_fixture();result=check_signed_vorticity_rows(data,two_d=True)
        self.assertEqual(len(result),9)
        data['full_I_eddy_A0'][-1]+=.01
        with self.assertRaises(AssertionError):check_signed_vorticity_rows(data,two_d=True)

    def test_self_consistent_saved_residual_still_needs_small_temporal_error(self):
        data=signed_fixture()
        for region in ('full','interior','stokes'):
            for part,weight in (('Q',2.),('profile',1.),('eddy',1.)):
                data[region+'_Z_'+part]+=weight*data['time_s']
                data[region+'_balance_residual_'+part]=weight*data['time_s']
        with self.assertRaises(AssertionError):check_signed_vorticity_rows(data,two_d=True)

    def test_all_native_terms_close_with_signed_3d_feedback(self):
        arrays={name:np.arange(12,dtype=float).reshape(3,4)*factor for name,factor in (
            ('Qdot_advection_mean',.03),('Qdot_3d_native',-.04),('Qdot_wave',.07),
            ('Qdot_viscous',-.02),('Qdot_forcing',.01))}
        arrays['Qdot_total']=sum(arrays.values())
        result=forced_vorticity_identity(arrays)
        self.assertEqual(result['absolute_max'],0)
        arrays['Qdot_3d_native']*=-1
        with self.assertRaises(AssertionError):forced_vorticity_identity(arrays)

    def test_omitted_forcing_cannot_pass_balance(self):
        arrays={name:np.zeros((3,4)) for name in ('Qdot_advection_mean','Qdot_3d_native','Qdot_wave','Qdot_viscous','Qdot_forcing')}
        arrays['Qdot_forcing'][:]=1e-5;arrays['Qdot_total']=sum(arrays.values())
        forced_vorticity_identity(arrays)
        arrays['Qdot_forcing'][:]=0
        with self.assertRaises(AssertionError):forced_vorticity_identity(arrays)

if __name__=='__main__':unittest.main()
