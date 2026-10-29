"""End-to-end synthetic bundle: verification, analysis, plotting, and cache."""
import copy,csv,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from analyze_wind import digest,write_json,Bundle,current_case,qualify_results
from wind_science import DEFAULT_CRITERIA,mask_regions,rms_xz


def make_fixture(root):
    case='fixture';bundle=root/'data/runs'/case;bundle.mkdir(parents=True)
    nx,nz,h=8,24,.5;z=-h+(np.arange(nz)+.5)*h/nz;dz=np.full(nz,h/nz)
    x=np.arange(nx)*2*np.pi/nx;eddy=.02*np.sin(x)[:,None]*np.cos(np.pi*(z+h)/h)[None,:]
    config=dict(nx=nx,ny=8,nz=nz,lx=2.,ly=1.,h=h,k=6.1,S=.036,S_reference=.036,alpha=.75,
        eddy_rms=.004,injection_wavelength=.25,perturb_energy_ratio=1.,seed=1,nu=1e-5,
        wind_ustar=.004,max_dt=.02,start_time=0,stop_time=1200,initial_snapshot_path='/private/source/never-publish')
    def q_at(t):return eddy+np.cos(np.pi*(z+h)/h)[None,:]*t*1e-6
    def u_at(t):return .004*np.cos(x)[:,None]*np.ones((1,nz))-.025*np.exp(12.2*z)[None,:]+t*1e-6*np.cos(np.pi*(z+h)/h)[None,:]
    def csvout(path,rows):
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def branch(name,start,stop,mode=0,sign=0):
        folder=bundle/name;folder.mkdir();cfg=config|dict(start_time=start,stop_time=stop)
        if name.startswith('forecast'):cfg['ny']=1
        if mode:cfg.update(jet_mode=mode,jet_sign=sign,jet_amplitude=.0002,initial_snapshot_sha256='same-state')
        write_json(folder/'configuration.json',cfg);write_json(folder/'grid.json',dict(z_centers=z.tolist(),z_faces=np.linspace(-h,0,nz+1).tolist()))
        write_json(folder/'completed.json',dict(time=stop))
        if not mode:
            for t in np.arange(start,stop+1,10):
                stem=f'mean_{t:012.6f}';arrays={}
                for key,a in dict(Uc=u_at(t),Wc=np.zeros((nx,nz)),Q=q_at(t),Qdot_3d_native=np.zeros((nx,nz))).items():
                    file=stem+'_'+key+'.bin';a.astype('<f8').ravel(order='F').tofile(folder/file)
                    arrays[key]=dict(file=file,shape=list(a.shape),dtype='float64',order='F',sha256=digest(folder/file))
                write_json(folder/(stem+'.json'),dict(time=float(t),arrays=arrays))
        rows=[]
        phi=np.cos(mode*np.pi*(z+h)/h)
        for t in np.arange(start,stop+1,5):
            a=np.exp(.002*(t-start));u=u_at(t).mean(axis=0)
            if mode:u=u+sign*.0002*a*phi
            for j,zz in enumerate(z):
                adv=sign*.0002*.002*a*phi[j] if mode else 0.
                rows.append(dict(time_s=t,z_m=zz,dz_m=dz[j],UL=u[j]+.036*np.exp(12.2*zz),UE=u[j],
                    adv_full=adv,adv_ky0=adv,adv_3d=0.,wave=0.,closure=0.,forcing=0.,pressure_mean=0.,
                    flux_total=0.,flux_ky0=0.,flux_3d=0.,continuum_flux_rhs=0.))
        csvout(folder/'jet-profiles.csv',rows)
        if name=='parent':
            csvout(folder/'regime-budget.csv',[dict(time_s=t,eddy_rms=.004,chi_3d=.01,native_divergence_max=1e-15,native_pressure_curl_max=1e-15) for t in np.arange(0,1201,5)])
            regions=mask_regions(z,6.1);rows=[]
            for t in np.arange(0,1201,5):
                r=dict(time_s=t,roll_energy_xmean=1e-7)
                for name,mask in regions.items():
                    suffix='stokes' if name=='active_interior' else name
                    r['C_native_rms_'+suffix]=0.
                    q=q_at(t);qe=q-q.mean(axis=0,keepdims=True)
                    r['q_eddy_rms_'+suffix]=rms_xz(qe[:,mask],dz[mask])
                rows.append(r)
            csvout(folder/'wind-budget.csv',rows)
    branch('parent',0,1200)
    branch('forecast-600',600,800);branch('forecast-1000',1000,1200)
    for mode in (1,2,4):
        branch(f'probe-m{mode}-plus',600,800,mode,1);branch(f'probe-m{mode}-minus',600,800,mode,-1)
    files={str(p.relative_to(bundle)):digest(p) for p in bundle.rglob('*') if p.is_file()}
    receipt=dict(id=case,source_version='fixture-source',completed=True,persistent_backup_verified=True,files=files)
    write_json(root/'data/status'/case/'completed.json',receipt)
    write_json(root/'source-manifest.json',dict(version='fixture-source',files={}))
    write_json(root/'campaign.json',dict(runs=[dict(id=case,path='data/runs/'+case,group='primary')],criteria={}))
    (root/'PROTOCOL.txt').write_text('Synthetic test protocol, not scientific results.\n')
    return receipt


class ReportTests(unittest.TestCase):
    def test_verified_bundle_and_idempotent_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'root';root.mkdir();out=Path(tmp)/'page';make_fixture(root)
            cmd=[sys.executable,str(Path(__file__).with_name('analyze_wind.py')),'--root',str(root),'--output',str(out)]
            first_run=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(first_run.returncode,0,first_run.stdout+'\n'+first_run.stderr)
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['counts']['complete'],1)
            self.assertEqual(summary['closure_assessments']['PASS'],2)
            self.assertEqual(summary['mechanism_assessment'],'INCONCLUSIVE')
            self.assertEqual(summary['supported_positive_responses'],[])
            first=(out/'summary.json').read_bytes()
            repeat=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(repeat.returncode,0,repeat.stdout+'\n'+repeat.stderr)
            self.assertEqual(first,(out/'summary.json').read_bytes())
            self.assertFalse(list(out.rglob('*.bin')))
            self.assertNotIn('/private/source', (out/'summary.json').read_text())
            self.assertGreater((out/'closure-summary.png').stat().st_size,1000)
            # Tampering after successful caching must still be rejected.
            (root/'data/runs/fixture/parent/grid.json').write_text('{}')
            changed=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(changed.returncode,0)
            self.assertIn('Changed completed diagnostic',changed.stderr)

    def test_changed_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);receipt=make_fixture(root)
            bundle=Bundle(root,dict(id='fixture'),receipt)
            p=root/'data/runs/fixture/parent/grid.json';p.write_text('{}')
            with self.assertRaises(ValueError):bundle.file('parent/grid.json')

    def test_exact_2d_without_requested_probes_publishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'root';root.mkdir();out=Path(tmp)/'page';make_fixture(root)
            campaign=json.loads((root/'campaign.json').read_text())
            campaign['runs'][0]['probe_modes']=[]
            write_json(root/'campaign.json',campaign)
            cmd=[sys.executable,str(Path(__file__).with_name('analyze_wind.py')),'--root',str(root),'--output',str(out)]
            run=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+'\n'+run.stderr)
            result=json.loads((out/'runs/fixture/summary.json').read_text())
            self.assertEqual(result['mode_responses'],[])
            self.assertEqual((out/'runs/fixture/response-history.csv').read_text().count('\n'),1)

    def test_approval_list_does_not_expose_unapproved_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_fixture(root)
            status,receipt=current_case(root,dict(id='fixture'),'fixture-source',set())
            self.assertNotEqual(status,'complete');self.assertIsNone(receipt)

    def test_failure_requires_both_spatial_and_timestep(self):
        def result(name,nz=96,dt=.02):
            region=dict(region='interior',apparent_failure=True,epsilon_C=.02,max_q_forecast_error=.2,max_current_forecast_error=.2,assessment='INCONCLUSIVE')
            return dict(id=name,status='complete',configuration=dict(nz=nz,max_dt=dt),parameters=dict(seed=1,u_star_m_s=.004),
                closure_windows=[dict(start_s=600,assessment='INCONCLUSIVE',regions=[region])],mode_responses=[])
        base=result('base');fine=result('fine',192,.01);step=result('step',96,.01)
        entries=[dict(id='base'),dict(id='fine',verify_of='base'),dict(id='step',verify_of='base')]
        qualify_results([base,fine],entries,DEFAULT_CRITERIA)
        self.assertEqual(base['closure_windows'][0]['assessment'],'INCONCLUSIVE')
        qualify_results([base,fine,step],entries,DEFAULT_CRITERIA)
        self.assertEqual(base['closure_windows'][0]['assessment'],'FAIL')

    def test_near_boundary_failure_not_resolved(self):
        def result(name,nz,error,dt):
            region=dict(region='interior',apparent_failure=True,epsilon_C=.02,max_q_forecast_error=error,max_current_forecast_error=error,assessment='INCONCLUSIVE')
            return dict(id=name,status='complete',configuration=dict(nz=nz,max_dt=dt),parameters=dict(seed=1,u_star_m_s=.004),
                closure_windows=[dict(start_s=600,assessment='INCONCLUSIVE',regions=[region])],mode_responses=[])
        base=result('base',96,.101,.02);fine=result('fine',192,.105,.01);step=result('step',96,.108,.01)
        qualify_results([base,fine,step],[dict(id='base'),dict(id='fine',verify_of='base'),dict(id='step',verify_of='base')],DEFAULT_CRITERIA)
        self.assertEqual(base['closure_windows'][0]['assessment'],'INCONCLUSIVE')


if __name__=='__main__':unittest.main()
