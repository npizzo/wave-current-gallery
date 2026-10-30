import copy,csv,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from analyze_decay import (laminar_mean,mode_coefficients,digest,write_json,assess,Run,compare_pair,extrema,persistent_extrema,run_state)


def make_fixture(root):
    n=24;nx=8;h=.5;nu=1e-5;dz=h/n;z=-h+(np.arange(n)+.5)*dz;time=np.arange(1000.,1600.1,5);tau=time-1000
    stokes=.036*np.exp(12.2*z);initial=.025*np.exp(12.2*z)-.004
    source='verified-shared-parent';entries=[];receipts={}
    def csvout(path,rows):
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def diffusion(a,ustar):
        value=np.zeros_like(a);value[1:-1]=nu*(a[2:]-2*a[1:-1]+a[:-2])/dz**2;value[0]=nu*(a[1]-a[0])/dz**2;value[-1]=nu*(a[-2]-a[-1])/dz**2+ustar**2/dz
        return value
    for id,ustar in [('wind-off',0.),('wind-on',.002)]:
        folder=root/'data/runs'/id;folder.mkdir(parents=True)
        cfg=dict(nx=nx,ny=4,nz=n,lx=2.,ly=1.,h=h,k=6.1,S=.036,S_reference=.036,alpha=.75,nu=nu,max_dt=.02,wind_ustar=ustar,start_time=1000.,stop_time=1600.,waves=True,advection='Centered2',closure='constant',velocity_convention='Lagrangian',initial_snapshot_sha256=source,initial_snapshot_path='/private/never-publish')
        write_json(folder/'configuration.json',cfg);write_json(folder/'completed.json',dict(time=1600.));write_json(folder/'grid.json',dict(z_centers=z.tolist(),z_faces=np.linspace(-h,0,n+1).tolist()))
        base=laminar_mean(initial,tau,h,nu,ustar);phi=np.cos(2*np.pi*(z+h)/h);eig=-4*nu/dz**2*np.sin(2*np.pi/(2*n))**2
        amplitude=.0005*(1-np.exp(-tau/200));ampdot=.0005/200*np.exp(-tau/200)
        residual=amplitude[:,None]*phi[None,:];ul=base+residual;ue=ul-stokes
        rows=[]
        for i,t in enumerate(time):
            visc=diffusion(ul[i],ustar);adv=(ampdot[i]-eig*amplitude[i])*phi;force=-ustar**2/h;total=visc+adv+force
            for j,zz in enumerate(z):rows.append(dict(time_s=t,z_m=zz,dz_m=dz,UL=ul[i,j],UE=ue[i,j],adv_full=adv[j],adv_ky0=adv[j],adv_3d=0.,wave=0.,closure=visc[j],forcing=force,pressure_mean=0.,native_total=total[j],flux_total=0.,flux_ky0=0.,flux_3d=0.))
        csvout(folder/'jet-profiles.csv',rows)
        energy=.5*.004**2;budget=[];wind=[]
        for t in time:
            budget.append(dict(time_s=t,Eeddy=energy,Eky0_eddy=energy*.2,Eperp=energy*.8,chi_3d=.8,eddy_rms=.004))
            wind.append(dict(time_s=t,wind_ustar=ustar,roll_energy_xmean=energy*.7,C_native_rms_full=0.,C_native_rms_interior=0.,C_native_rms_stokes=0.,q_eddy_rms_full=.02,q_eddy_rms_interior=.02,q_eddy_rms_stokes=.02))
        csvout(folder/'regime-budget.csv',budget);csvout(folder/'wind-budget.csv',wind)
        for i in range(0,len(time),2):
            t=time[i];stem='mean_%012.6f'%t;arrays={};x=np.arange(nx)*2*np.pi/nx
            q=.02*np.sin(x)[:,None]*np.ones((1,n));q[:,-1]+=.5*ustar**2/nu
            for key,a in dict(Uc=ul[i][None,:]+.001*np.sin(x)[:,None]*np.ones((1,n)),Wc=np.zeros((nx,n)),Q=q).items():
                name=stem+'_'+key+'.bin';a.astype('<f8').ravel(order='F').tofile(folder/name);arrays[key]=dict(file=name,shape=list(a.shape),dtype='float64',order='F',sha256=digest(folder/name))
            write_json(folder/(stem+'.json'),dict(time=t,arrays=arrays))
        files={str(p.relative_to(folder)):digest(p) for p in folder.rglob('*') if p.is_file()}
        receipt=dict(id=id,completed=True,persistent_backup_verified=True,source_version='synthetic',files=files);write_json(root/'data/status'/id/'completed.json',receipt);receipts[id]=receipt
        entries.append(dict(id=id,arm='off' if ustar==0 else 'on',path='data/runs/'+id,env=dict(START_TIME=1000,STOP_TIME=1600,WIND_USTAR=ustar)))
    write_json(root/'source-manifest.json',dict(version='synthetic',files={}))
    write_json(root/'campaign.json',dict(runs=entries,criteria={}))
    (root/'PROTOCOL.txt').write_text('Synthetic test only.\n')
    return entries,receipts


class BaselineTests(unittest.TestCase):
    def test_unforced_discrete_mode_decay(self):
        n=48;h=.5;nu=1e-5;m=4;phi=np.cos(m*np.pi*(np.arange(n)+.5)/n);t=np.array([0.,100.,600.]);a=laminar_mean(2.+.1*phi,t,h,nu,0)
        eig=-4*nu/(h/n)**2*np.sin(m*np.pi/(2*n))**2
        np.testing.assert_allclose(a,2.+.1*np.exp(t[:,None]*eig)*phi,atol=2e-14)
    def test_stress_force_balance_and_initial_slope(self):
        n=32;h=.5;nu=1e-5;us=.002;t=np.array([0.,1e-6,200.]);u=laminar_mean(np.full(n,.1),t,h,nu,us)
        np.testing.assert_allclose(u.mean(axis=1),.1,atol=1e-14)
        slope=(u[1]-u[0])/1e-6;expect=np.full(n,-us**2/h);expect[-1]+=us**2/(h/n)
        np.testing.assert_allclose(slope,expect,rtol=1e-5,atol=1e-9)
    def test_constant_is_not_an_extremum(self):
        self.assertEqual(extrema(np.ones(20),np.linspace(-.5,0,20),.00004),[])
    def test_inherited_extremum_flagged(self):
        z=np.linspace(-.5,0,97);u=np.cos(2*np.pi*(z+.5)/.5)*.001;t=np.arange(0,605,5)
        found=persistent_extrema(np.repeat(u[None,:],len(t),axis=0),t,z,.00004,u)
        self.assertTrue(found);self.assertTrue(all(e['matches_initial_extremum'] for e in found))


class ReportTests(unittest.TestCase):
    def test_science_and_identical_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);entries,receipts=make_fixture(root);results=[];curves={}
            for e in entries:
                result,data,_=assess(root,e,receipts[e['id']]);results.append(result);curves[e['id']]=data
                self.assertLess(max(result['mode_budget_relative_errors']),.005)
                self.assertLess(result['mean_momentum_budget_relative_error'],.005)
                self.assertTrue(all(w['classification']=='WEAK_CORRECTION' for w in result['windows']))
                self.assertEqual(len(result['profile_windows']),3)
                self.assertTrue(all('laminar_extrema' in w for w in result['profile_windows']))
            comparison=compare_pair(results,curves);self.assertTrue(comparison['identical_restart_verified']);self.assertEqual(comparison['assessment'],'INCONCLUSIVE')
            self.assertAlmostEqual(comparison['initial_top_cell_Q_off_minus_on_s_inverse'],-.2)
            results[1]['configuration']['nu']=2e-5
            with self.assertRaisesRegex(ValueError,'physical parameters'):compare_pair(results,curves)
            results[1]['configuration']['nu']=1e-5
            curves['wind-on']['maps']['Q'][0,0,-1]+=.001
            with self.assertRaisesRegex(ValueError,'boundary-gradient'):compare_pair(results,curves)
    def test_changed_diagnostic_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);e,r=make_fixture(root);p=root/'data/runs/wind-off/grid.json';p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed verified'):Run(root,e[0],r['wind-off']).file('grid.json')
    def test_unapproved_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);e,r=make_fixture(root);status,receipt=run_state(root,e[0],'synthetic',set());self.assertNotEqual(status,'complete');self.assertIsNone(receipt)
    def test_end_to_end_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'root';root.mkdir();out=Path(tmp)/'page';make_fixture(root)
            cmd=[sys.executable,str(Path(__file__).with_name('analyze_decay.py')),'--root',str(root),'--output',str(out)]
            run=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stdout+'\n'+run.stderr)
            summary=json.loads((out/'summary.json').read_text());self.assertEqual(summary['counts']['complete'],2);self.assertTrue(summary['comparison']['identical_restart_verified'])
            self.assertFalse(list(out.rglob('*.bin')));self.assertNotIn('/private/never-publish',(out/'summary.json').read_text())
            for p in ('mean-profiles.png','primary-600s.png','depth-time.png','mean-eddies.png','band-budget.png'):self.assertGreater((out/p).stat().st_size,1000)
            first=(out/'summary.json').read_bytes();run=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(first,(out/'summary.json').read_bytes())
    def test_waiting_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'root';root.mkdir();out=Path(tmp)/'page';write_json(root/'campaign.json',dict(runs=[dict(id='wind-off'),dict(id='wind-on')],criteria={}));write_json(root/'source-manifest.json',dict(version='none',files={}))
            run=subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_decay.py')),'--root',str(root),'--output',str(out)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(json.loads((out/'summary.json').read_text())['counts']['waiting'],2)


if __name__=='__main__':unittest.main()
