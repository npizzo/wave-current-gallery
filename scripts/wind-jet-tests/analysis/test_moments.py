import json,tempfile,unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch
from moment_science import (MOMENTS,SOURCES,FLUXES,production_components,second_moments,
    interval_closure,integral_metrics,regional_profile,eulerian_profile)
from analyze_moments import (render_moments,analyze_moment_run,compact_csv,profile_groups,mean_comparisons)
from analyze import write_json,digest,cache_valid


class MomentScienceTests(unittest.TestCase):
    def test_2d3c_tilting_cancels_with_nonzero_transverse_velocity_gradients(self):
        rng=np.random.default_rng(34);q=rng.normal(size=(5,1,7));vx=rng.normal(size=q.shape);vz=rng.normal(size=q.shape)
        zero=np.zeros_like(q);a=q-.3
        p=production_components(q,a,zero,zero,vx,zero,vz)
        np.testing.assert_array_equal(p['T'],0)
        np.testing.assert_array_equal((-vz)*vx+vx*vz,0)
        for key in p:np.testing.assert_array_equal(p[key],0)
        self.assertGreater(np.linalg.norm(vx)+np.linalg.norm(vz),1)

    def test_production_sum_and_spatial_cancellation_are_not_a_null(self):
        q=np.ones((2,));p=production_components(q,q,np.array([1.,-1.]),0.,0.,0.,0.)
        self.assertEqual(np.mean(p['Pq_T']),0)
        self.assertEqual(np.mean(p['Pq_T_abs']),1)
        np.testing.assert_allclose(p['Pq_T'],sum(p['Pq_'+n] for n in ('stretch','tilt_wy_vx','tilt_minus_uy_vz')))
        m=integral_metrics([0,1,2],[0,0,0],[1,1,1])
        self.assertEqual(m['cancellation_fraction'],1);self.assertEqual(m['absolute_local_integral'],2)
        with self.assertRaises(ValueError):integral_metrics([0,1],[2,2],[1,1])

    def test_second_moment_partition_and_native_rate_need_full_field(self):
        q=np.array([[[1.],[3.]]]);p=second_moments(q,1.)
        np.testing.assert_allclose(p['Zq'],p['Zmean']+p['Zvariance'])
        np.testing.assert_allclose(p['Za'],p['Zmean_anomaly']+p['Zvariance'])
        qdot=np.array([[[-1.],[1.]]])
        self.assertEqual(float(np.mean(q*qdot)),1)
        self.assertEqual(float(np.mean(q)*np.mean(qdot)),0)

    def test_temporal_and_spatial_closure_are_separate(self):
        t=np.linspace(0,2,21);c=interval_closure(t,t*t,2*t,2*t+.1)
        np.testing.assert_allclose(c['temporal_residual'],0,atol=1e-14)
        np.testing.assert_allclose(c['spatial_residual'],-.1*t,atol=1e-14)
        np.testing.assert_allclose(c['total_residual'],c['temporal_residual']+c['spatial_residual'],atol=1e-14)
        err=[]
        for n in (11,21):
            tt=np.linspace(0,1,n);cc=interval_closure(tt,tt**3,3*tt**2,3*tt**2)
            err.append(abs(cc['temporal_residual'][-1]))
        self.assertAlmostEqual(err[0]/err[1],4,places=10)

    def test_regional_budget_keeps_basal_flux_and_exchange(self):
        n=4;p={'z_m':np.arange(n)+.5,'dz_m':np.ones(n),'time_s':np.zeros(n)}
        for family in MOMENTS:
            p[family]=np.ones(n)
            p['d'+family+'_native']=np.zeros(n)
            for name in SOURCES[family]:p[name]=np.zeros(n)
            for name in ('viscous','forcing','pressure'):p[family+'_'+name]=np.zeros(n)
            p['flux_'+FLUXES[family]+'_lower']=np.arange(n,dtype=float)
            p['flux_'+FLUXES[family]+'_upper']=np.arange(1,n+1,dtype=float)
        p['Pmean_exchange']=np.ones(n)*3;p['Pvariance_exchange']=np.ones(n)*-3
        p['Pq_T']=np.ones(n)*2;p['Pmean_T']=np.ones(n)*1.2;p['Pvariance_T']=np.ones(n)*.8
        p['dZq_native']=np.ones(n)
        p['Zq']=np.ones(n)*2;p['Zmean']=np.ones(n);p['Zvariance']=np.ones(n)
        r=regional_profile(p,[False,False,True,True])
        self.assertEqual(r['Zq_boundary_supply'],-1)
        self.assertEqual(r['Zq_continuum_rate'],1)
        self.assertEqual(r['Zq_spatial_residual'],0)
        self.assertEqual(r['exchange_sum'],0);self.assertEqual(r['partition_residual'],0)
        with self.assertRaises(ValueError):regional_profile(p,[True,False,True,False])

    def test_no_wave_conversion_preserves_eulerian_background(self):
        z=np.array([-.1,-.05]);ul=-.75*.03*np.exp(12.2*z)
        np.testing.assert_array_equal(eulerian_profile(ul,z,{'S':0,'k':6.1}),ul)
        np.testing.assert_allclose(eulerian_profile(ul+.03*np.exp(12.2*z),z,{'S':.03,'k':6.1}),ul)


class MomentReportTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def fixture(self,name='base',eta=0):
        root=self.root;folder=root/'data/runs'/name;folder.mkdir(parents=True)
        zf=np.linspace(-.2,0,9);z=(zf[:-1]+zf[1:])/2;dz=np.diff(zf)
        config=dict(nx=8,ny=8,nz=8,lx=1.,ly=1.,h=.2,k=6.1,S=.03,S_reference=.03,alpha=.75,uoffset=0.,eddy_rms=.004,injection_wavelength=.25,perturb_energy_ratio=eta,seed=1,stop_time=2.,max_dt=.01,nu=1e-5,base_ic_sha256='base',initial_condition_sha256=name)
        write_json(folder/'configuration.json',config);write_json(folder/'grid.json',dict(z_centers=z,z_faces=zf));write_json(folder/'completed.json',dict(time=2.,initial_condition_sha256=name))
        q0=-.75*12.2*.03*np.exp(12.2*z);ul=.25*.03*np.exp(12.2*z);ue=-.75*.03*np.exp(12.2*z)
        p={'z_m':z,'dz_m':dz,'time_s':np.zeros(8)}
        for family in MOMENTS:
            p[family]=np.zeros(8);p['d'+family+'_native']=np.zeros(8)
            for term in SOURCES[family]:p[term]=np.zeros(8)
            for term in ('inviscid','viscous','forcing','pressure'):p[family+'_'+term]=np.zeros(8)
            for side in ('lower','upper'):p['flux_'+FLUXES[family]+'_'+side]=np.zeros(8)
        p['Zq']=p['Zmean']=q0*q0/2;p['q0_native_center']=p['q0_analytic_center']=q0
        for term in ('Pq_T_abs','Pa_T_abs','Pmean_feedback_native','Pmean_anomaly_feedback_native','T_rms','v_rms','Pq_advection','Pa_advection'):p[term]=np.zeros(8)
        for prefix in ('Pq','Pa'):
            for part in ('stretch','tilt_wy_vx','tilt_minus_uy_vz'):
                p[prefix+'_'+part]=np.zeros(8);p[prefix+'_'+part+'_abs']=np.zeros(8)
        profiles=[];rows=[];jets=[]
        for time in (0.,1.,2.):
            p['time_s']=np.full(8,time)
            profiles.extend({k:float(v[i]) for k,v in p.items()} for i in range(8))
            rows.append({'time_s':time,**{k:float(np.sum(v*dz)/sum(dz)) for k,v in p.items() if k not in ('time_s','z_m','dz_m') and not k.startswith('flux_')}})
            jets.extend(dict(time_s=time,z_m=z[i],dz_m=dz[i],UL=ul[i],UE=ue[i]) for i in range(8))
        # Fixtures use full precision: receipt integrity is independent of public compact CSV.
        from analyze import write_csv
        write_csv(folder/'moments.csv',rows);write_csv(folder/'moment-profiles.csv',profiles);write_csv(folder/'jet-profiles.csv',jets)
        write_csv(folder/'regime-budget.csv',[dict(time_s=0)]);write_csv(folder/'jet-budget.csv',[dict(time_s=0)])
        receipt=dict(id=name,source_version='fixture',completed=True,files={p.name:digest(p) for p in folder.iterdir()})
        entry=dict(id=name,path='data/runs/'+name,matched_group='central',seed=1,group='2d' if eta==0 else 'finite')
        return entry,receipt

    def test_streaming_verified_analysis_cache_and_profiles(self):
        entry,receipt=self.fixture();cache=self.root/'cache/base'
        result=analyze_moment_run(self.root,entry,receipt,cache)
        self.assertEqual(result['max_abs_3d_production'],0)
        self.assertEqual(result['closure']['full']['Zq']['max_abs_temporal_residual'],0)
        self.assertTrue((cache/'current-profiles.png').is_file())
        (self.root/entry['path']/'moments.csv').unlink()
        self.assertEqual(analyze_moment_run(self.root,entry,receipt,cache),result)
        data=np.load(cache/'profile-history.npz')
        self.assertTrue(np.all(data['UE']<0));data.close()

    def test_changed_profile_rejected(self):
        entry,receipt=self.fixture();path=self.root/entry['path']/'moment-profiles.csv';path.write_text(path.read_text()+'\n')
        with self.assertRaises(ValueError):analyze_moment_run(self.root,entry,receipt,self.root/'cache/base')

    def test_pending_page_has_no_inferred_results(self):
        output=self.root/'public';output.mkdir()
        render_moments(dict(runs=[dict(id='pending',status='waiting')],counts=dict(waiting=1),figures=[],source_version='test',generated_at='test'),output)
        page=(output/'index.html').read_text()
        self.assertIn('Results will appear after verified completion',page)
        self.assertIn('one component',page);self.assertIn('2D3C',page)
        self.assertNotIn('{{',page);self.assertNotIn('runs/pending/',page)
        self.assertIn('q<sub>0</sub>',page)

    def test_direct_mean_comparison_has_all_four_disturbance_levels(self):
        entries=[];results=[];cache=self.root/'cache';out=self.root/'public';out.mkdir()
        with patch('analyze_moments.plot_run'):
            for name,eta in [('base',0.),('tiny',1e-6),('moderate',.01),('finite',1.)]:
                e,r=self.fixture(name,eta);entries.append(e)
                results.append(analyze_moment_run(self.root,e,r,cache/name))
        figures,rows=mean_comparisons(entries,results,cache,out)
        self.assertEqual(len(figures),1);self.assertTrue((out/figures[0]['file']).is_file())
        self.assertEqual({r['run_id'] for r in rows},{'base','tiny','moderate','finite'})
        self.assertTrue(all(r['mean_current_change_error_m_s']==0 for r in rows))


if __name__=='__main__':unittest.main()
