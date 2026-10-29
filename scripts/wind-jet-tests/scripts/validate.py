#!/usr/bin/env python3
"""Software/backend validation, with short full-grid tests; not a science verdict."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from guard import verify,write_json,digest
from launch import snapshot_at


def snapshot(folder,initial=False):
    config=json.loads((folder/'configuration.json').read_text())
    if initial:
        path=sorted(folder.glob('t*.json'))[0]
    else:
        path=folder/json.loads((folder/'completed.json').read_text())['final_snapshot']
    manifest=json.loads(path.read_text());arrays={}
    for key,item in manifest['arrays'].items():
        source=folder/item['file']
        assert digest(source)==item['sha256'], 'Native snapshot hash mismatch'
        data=np.fromfile(source,dtype='<f8').reshape(item['shape'],order='F')
        assert np.all(np.isfinite(data)), 'Nonfinite native snapshot'
        arrays[key]=data
    u,v,w=(arrays[key] for key in ('u','v','w'))
    dx,dy,dz=(config['lx']/config['nx'],config['ly']/config['ny'],config['h']/config['nz'])
    div=(np.roll(u,-1,axis=0)-u)/dx+(np.roll(v,-1,axis=1)-v)/dy+np.diff(w,axis=2)/dz
    scale=max(np.max(np.abs(u))/dx,np.max(np.abs(v))/dy,np.max(np.abs(w))/dz,1e-15)
    scaled_div=float(np.max(np.abs(div))/scale)
    assert scaled_div<1e-10, ('Native divergence',scaled_div)
    assert np.max(np.abs(w[:,:,[0,-1]]))<1e-12, 'Impermeable boundary'
    return arrays,{'time':manifest['time'],'scaled_divergence_max':scaled_div,'shape':list(u.shape)}


def compare(a,b):
    first,_=snapshot(a);second,_=snapshot(b)
    errors={k:float(np.linalg.norm((first[k]-second[k]).ravel())/max(np.linalg.norm(first[k].ravel()),1e-14)) for k in first}
    assert max(errors.values())<1e-8, ('Backend/restart mismatch',errors)
    return errors


def budgets(folder,two_d=False):
    data=np.genfromtxt(folder/'jet-budget.csv',delimiter=',',names=True)
    data=np.atleast_1d(data)
    assert len(data)>2 and np.all(np.diff(data['time_s'])>0), 'Missing/duplicate diagnostic times'
    for key in data.dtype.names:
        values=data[key]
        # A first sample has no preceding interval; its derivatives are undefined.
        if key.startswith(('interval_','dEjet_dt_')):values=values[1:]
        if key=='ky0_eddy_fraction':values=values[data['Eeddy']>0]
        assert np.all(np.isfinite(values)), 'Nonfinite scalar '+key
    profile=np.genfromtxt(folder/'jet-profiles.csv',delimiter=',',names=True)
    times=np.unique(profile['time_s']);nz=len(profile)//len(times)
    assert len(profile)==len(times)*nz
    ul=profile['UL'].reshape(len(times),nz);rhs=profile['native_total'].reshape(len(times),nz)
    delta=ul[-1]-ul[0];integral=np.trapezoid(rhs,x=times,axis=0)
    denom=np.linalg.norm(delta)+np.trapezoid(np.sqrt(np.sum(rhs*rhs,axis=1)),x=times)
    relative=float(np.linalg.norm(delta-integral)/max(denom,1e-15))
    assert relative<0.03, ('Mean-momentum time integration closure',relative)
    pressure=float(np.max(np.abs(data['pressure_mean_max'])))
    assert pressure<1e-9, ('Periodic pressure-mean identity',pressure)
    assert float(np.max(np.abs(data['flux_decomposition_max'])))<1e-12
    extra=float(np.max(np.abs(profile['adv_3d'])))
    if two_d:assert extra<1e-10, ('Nonzero3D term for exactly2D run',extra)
    return {'mean_momentum_relative_integrated_residual':relative,
            'max_native_3d_acceleration':extra,'pressure_mean_max':pressure,
            'rows':len(data),'final_Ejet_L':float(data['Ejet_L'][-1])}


def regime_outputs(folder,two_d=False):
    config=json.loads((folder/'configuration.json').read_text())
    trim=int(config.get('diagnostic_trim_cells',max(2,(2*int(config['nz'])+95)//96)))
    data=np.atleast_1d(np.genfromtxt(folder/'regime-budget.csv',delimiter=',',names=True))
    assert len(data)>2 and np.all(np.diff(data['time_s'])>0)
    for name in data.dtype.names:
        assert np.all(np.isfinite(data[name])), 'Nonfinite regime statistic '+name
    assert np.max(data['viscous_cfl'])<=.201, 'Diffusive time-step restriction'
    assert np.max(data['native_pressure_curl_max'])<1e-9, 'Native pressure-curl identity'
    if two_d:assert np.max(data['Eperp'])<1e-22, 'Exact 2D flow gained transverse energy'
    maps=[]
    required={'Uc','Wc','Q','A','Rbar','Qdot_advection_full','Qdot_advection_mean',
              'Qdot_wave','Qdot_viscous','Qdot_forcing','Qdot_total','Qdot_3d_native',
              'continuum_scalar_residual','continuum_R_identity_error'}
    for path in sorted(folder.glob('mean_*.json')):
        manifest=json.loads(path.read_text());arrays={}
        assert required<=set(manifest['arrays']), 'Missing regime mean-map term'
        for name,item in manifest['arrays'].items():
            file=folder/item['file'];assert digest(file)==item['sha256'], 'Mean-map checksum mismatch'
            arrays[name]=np.fromfile(file,dtype='<f8').reshape(item['shape'],order='F')
            assert np.all(np.isfinite(arrays[name])), 'Nonfinite mean-map '+name
        shape=arrays['Q'].shape
        assert all(arrays[k].shape==shape for k in required)
        assert np.max(abs(arrays['continuum_R_identity_error']))<1e-11
        if two_d:assert np.max(abs(arrays['Qdot_3d_native']))<1e-10
        maps.append((manifest['time'],arrays))
    assert len(maps)>=3, 'Need at least three saved maps for an independent temporal check'
    times=np.array([t for t,a in maps]);assert np.all(np.diff(times)>0)
    errors=[]
    for i in range(1,len(maps)-1):
        h0,h1=times[i]-times[i-1],times[i+1]-times[i]
        qt=(-h1/(h0*(h0+h1)))*maps[i-1][1]['Q']+((h1-h0)/(h0*h1))*maps[i][1]['Q']+(h0/(h1*(h0+h1)))*maps[i+1][1]['Q']
        native=maps[i][1]['Qdot_total'];qt=qt[:,trim:-trim];native=native[:,trim:-trim]
        errors.append(float(np.linalg.norm(qt-native)/max(np.linalg.norm(qt)+np.linalg.norm(native),1e-15)))
    assert max(errors)<.03, ('Native vorticity temporal closure',max(errors))
    return {'map_count':len(maps),'interior_trim_cells':trim,
            'maximum_native_vorticity_temporal_relative_residual':max(errors),
            'maximum_viscous_cfl':float(np.max(data['viscous_cfl'])),
            'final_Eperp':float(data['Eperp'][-1])}


def check_wind_momentum(folder,ustar,h):
    rows=np.atleast_1d(np.genfromtxt(folder/'jet-profiles.csv',delimiter=',',names=True))
    times=np.unique(rows['time_s']);nz=len(rows)//len(times)
    ul=rows['UL'].reshape(len(times),nz)
    closure=rows['closure'].reshape(len(times),nz).mean(axis=1)
    forcing=rows['forcing'].reshape(len(times),nz).mean(axis=1)
    expected=ustar**2/h
    assert np.max(abs(closure-expected))<max(1e-12,expected*1e-9), ('Stress sign/magnitude',closure,expected)
    assert np.max(abs(forcing+expected))<max(1e-12,expected*1e-9), ('Bulk compensation',forcing,expected)
    drift=float(np.max(abs(ul.mean(axis=1)-ul[0].mean())))
    assert drift<1e-10, ('Unexpected bulk momentum drift',drift)
    return dict(kinematic_stress=ustar**2,expected_mean_acceleration=expected,
                maximum_bulk_momentum_drift=drift)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();source=verify(root)
    label=os.environ.get('SLURM_JOB_ID',datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    output=root/'validation'/('attempt-'+label);output.mkdir(parents=True,exist_ok=False)
    julia=os.environ['JULIA'];python=os.environ.get('PYTHON',sys.executable)
    started=time.monotonic()
    report=dict(passed=False,source_version=source['version'],job_id=label,runs={},comparisons={})

    def run_case(name,settings,initial=None,kind='full',two_d=False):
        folder=output/name
        base={'ARCH':'GPU','NX':16,'NY':16,'NZ':16,'LX':.5,'LY':.5,'H':.5,'K0':6.1,
              'ALPHA':.75,'EDDY_RMS':.004,'INJECTION_WAVELENGTH':.25,'PERTURB_ENERGY_RATIO':1,
              'NU':1e-5,'MAINTAIN_BACKGROUND':0,'ADVECTION':'Centered2','REGIME_INTERVAL':.05,
              'SEED':1,'WAVES':1,'WIND_USTAR':.004,'STOP_TIME':.5,'START_TIME':0,
              'TWO_D_THREE_C_RMS':0,'MOMENTS_ENABLED':0,'MOMENT_INTERVAL':.025,'MAX_DT':.005,
              'DIAGNOSTIC_INTERVAL':.025,'MEAN_INTERVAL':0,'CHECKPOINT_WALL':600,'KEEP_CHECKPOINT':0,
              'SNAPSHOT_TIMES':'0,.25,.4,.5','JET_MODE':0,'JET_SIGN':0,'JET_AMPLITUDE':0,
              'INIT_SNAPSHOT':'','INIT_KIND':kind}
        base.update(settings);base['OUT_DIR']=str(folder)
        if initial:
            base['INIT_SNAPSHOT']=str(initial)
            start=float(json.loads(initial.read_text())['time']);base['START_TIME']=start
            base['SNAPSHOT_TIMES']=f'{start:g},{float(base["STOP_TIME"]):g}'
        env=dict(os.environ)
        for key in ('PICKUP','INIT_SNAPSHOT','INIT_KIND','START_TIME','JET_MODE','JET_SIGN','JET_AMPLITUDE'):
            env.pop(key,None)
        env.update({k:str(v) for k,v in base.items()})
        print('VALIDATION_CASE',name,flush=True)
        beginning=time.monotonic()
        with (output/(name+'.log')).open('w') as stream:
            subprocess.run([julia,'--startup-file=no',str(root/'run/run_jet.jl')],env=env,
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
        _,native=snapshot(folder);_,initial_native=snapshot(folder,True)
        diagnostics=budgets(folder,two_d=two_d);regime=regime_outputs(folder,two_d=two_d)
        momentum=check_wind_momentum(folder,float(base['WIND_USTAR']),float(base['H']))
        report['runs'][name]=dict(native=native,initial_native=initial_native,budget=diagnostics,
             regime=regime,momentum=momentum,wall_seconds=time.monotonic()-beginning,
             completed=json.loads((folder/'completed.json').read_text()))
        write_json(output/'progress.json',report)
        print('VALIDATED',name,native,flush=True)
        return folder

    try:
        # These are tests of the actual newly added surface-stress and branch operators.
        for arch in ('CPU','GPU'):
            print('TEST wind driver',arch,flush=True)
            subprocess.run([julia,'--startup-file=no',str(root/'run/test_wind_driver.jl')],
                           env=dict(os.environ,DRIVER_TEST_ARCH=arch),check=True)
        subprocess.run([python,'-m','unittest','discover','-s',str(root/'scripts'),'-p','test_wind_*.py','-v'],check=True)
        subprocess.run([python,'-m','unittest','discover','-s',str(root/'analysis'),'-p','test_wind_*.py','-v'],check=True)
        cpu=run_case('cpu-parent',{'ARCH':'CPU'})
        gpu=run_case('gpu-parent',{})
        report['comparisons']['cpu_gpu']=compare(cpu,gpu)
        run_case('gpu-flat2d',{'NY':1,'PERTURB_ENERGY_RATIO':0},two_d=True)
        parent_snapshot=snapshot_at(gpu,.25)
        full=run_case('gpu-reload',{},initial=parent_snapshot)
        a,_=snapshot(gpu);b,_=snapshot(full)
        errors={k:float(np.linalg.norm(a[k]-b[k])/max(np.linalg.norm(a[k]),1e-14)) for k in a}
        assert max(errors.values())<1e-5, ('Reload trajectory disagreement',errors)
        report['comparisons']['full_snapshot_reload']=errors
        forecast=run_case('gpu-forecast',{'NY':1},initial=parent_snapshot,kind='mean',two_d=True)
        projected,_=snapshot(forecast,True)
        _,initial_parent_native=snapshot(gpu,True)
        parent_arrays={}
        for key,item in json.loads(parent_snapshot.read_text())['arrays'].items():
            parent_arrays[key]=np.fromfile(gpu/item['file'],dtype='<f8').reshape(item['shape'],order='F')
        projection={k:float(np.max(abs(projected[k]-parent_arrays[k].mean(axis=1,keepdims=True)))) for k in projected}
        assert max(projection.values())<1e-12, ('Incorrect transverse-mean initialization',projection)
        report['comparisons']['mean_projection_max_abs']=projection
        plus=run_case('gpu-probe-plus',{'JET_MODE':2,'JET_SIGN':1,'JET_AMPLITUDE':2e-4},initial=parent_snapshot)
        minus=run_case('gpu-probe-minus',{'JET_MODE':2,'JET_SIGN':-1,'JET_AMPLITUDE':2e-4},initial=parent_snapshot)
        ap,_=snapshot(plus,True);am,_=snapshot(minus,True)
        for key in ap:
            assert np.max(abs((ap[key]+am[key])/2-parent_arrays[key]))<1e-12, ('Probe symmetry',key)
        perturb=(ap['u']-am['u'])/2
        z=(-.5+(np.arange(16)+.5)*.5/16)
        expected=2e-4*np.cos(2*np.pi*(z+.5)/.5)
        assert np.max(abs(perturb-expected[None,None,:]))<1e-12
        assert abs(perturb.mean())<1e-13, 'Probe changed net momentum'
        report['comparisons']['probe_initial_symmetry']=True
        defaults=json.loads((root/'campaign.json').read_text())['runs'][0]['env']
        fullsettings=dict(defaults,NX=128,NY=64,NZ=96,WIND_USTAR=.008,STOP_TIME=1,
                          DIAGNOSTIC_INTERVAL=.05,MOMENT_INTERVAL=.05,REGIME_INTERVAL=.1,
                          SNAPSHOT_TIMES='0,1',MAX_DT=.01,KEEP_CHECKPOINT=0)
        primary=run_case('primary-grid',fullsettings)
        fine=run_case('refined-grid',dict(fullsettings,NX=256,NY=128,NZ=192,MAX_DT=.005))
        report['comparisons']['full_grid_checks']={name:report['runs'][name] for name in ('primary-grid','refined-grid')}
        verify(root)
        report.update(passed=True,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic()-started,
                      scope='Software, actual wind-stress momentum balance, exact 2D null, CPU/GPU agreement, full/mean snapshot initialization, symmetric jet probes and short primary/refined GPU checks. Scientific closure and jet-onset verdicts remain production outcomes.')
        write_json(output/'result.json',report);write_json(root/'validation/passed.json',report)
        print('VALIDATION PASSED',source['version'],flush=True)
    except Exception as error:
        report.update(error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        write_json(output/'result.json',report)
        raise

if __name__=='__main__':main()
