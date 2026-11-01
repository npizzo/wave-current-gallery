#!/usr/bin/env python3
"""Software/backend and full-state/projection validation; not a science verdict."""
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
from references import verify_references
from inputs import branch_settings, approved_input, input_locations


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


def forced_vorticity_identity(arrays):
    """Exact native averaged equation, including the omitted 3D feedback."""
    names=('Qdot_advection_mean','Qdot_3d_native','Qdot_wave','Qdot_viscous','Qdot_forcing')
    rhs=sum(arrays[name] for name in names)
    residual=arrays['Qdot_total']-rhs
    absolute=float(np.max(np.abs(residual)))
    scale=max(float(np.max(np.abs(arrays['Qdot_total']))),sum(float(np.max(np.abs(arrays[name]))) for name in names),1e-15)
    if absolute>max(1e-11,scale*1e-10):
        raise AssertionError(('Forced transverse-mean vorticity decomposition',absolute,scale))
    return {'absolute_max':absolute,'relative_to_sum_of_term_maxima':absolute/scale,
            'term_maxima':{name:float(np.max(np.abs(arrays[name]))) for name in names}}


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
    maps=[];identities=[]
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
        identities.append(forced_vorticity_identity(arrays))
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
            'final_Eperp':float(data['Eperp'][-1]),
            'forced_vorticity_identity_max_absolute':max(r['absolute_max'] for r in identities),
            'forced_vorticity_identity_max_relative':max(r['relative_to_sum_of_term_maxima'] for r in identities)}


def signed_vorticity_budget(folder,two_d=False):
    data=np.atleast_1d(np.genfromtxt(folder/'vorticity-budget.csv',delimiter=',',names=True))
    return check_signed_vorticity_rows(data,two_d=two_d)


def check_signed_vorticity_rows(data,two_d=False):
    times=data['time_s'];assert len(times)>2 and np.all(np.diff(times)>0)
    for key in data.dtype.names:
        assert np.all(np.isfinite(data[key])), 'Nonfinite signed vorticity statistic '+key
    assert np.allclose(data['interval_dt'][1:],np.diff(times),rtol=1e-10,atol=1e-10)
    assert abs(data['interval_dt'][0])<1e-12
    terms=('A0','C','W','D','F');result={}
    for region in ('full','interior','stokes'):
        zscale=max(float(np.max(data[region+'_Z_Q'])),1e-15)
        decomposition=float(np.max(abs(data[region+'_Z_Q']-data[region+'_Z_profile']-data[region+'_Z_eddy'])))
        assert decomposition<max(1e-12,zscale*1e-10), 'Native profile/eddy enstrophy split differs'
        assert np.max(abs(data[region+'_enstrophy_decomposition_error']))<max(1e-12,zscale*1e-10)
        if two_d:assert np.max(abs(data[region+'_source_rms_C']))<1e-10, 'Exact 2D flow has3D vorticity feedback'
        for part in ('Q','profile','eddy'):
            prefix=region+'_P_'+part+'_'
            powers=np.stack([data[prefix+term] for term in terms],axis=1)
            ptotal=data[prefix+'total'];pscale=max(float(np.max(np.sum(abs(powers),axis=1))),1e-15)
            difference=float(np.max(abs(powers.sum(axis=1)-ptotal)))
            assert difference<max(1e-11,pscale*1e-10), 'Signed vorticity production split differs'
            assert np.max(abs(data[region+'_split_identity_'+part]))<max(1e-11,pscale*1e-10)
            for term in (*terms,'total'):
                expected=np.concatenate(([0.],np.cumsum(.5*(data[prefix+term][1:]+data[prefix+term][:-1])*np.diff(times))))
                recorded=data[region+'_I_'+part+'_'+term]
                assert np.max(abs(expected-recorded))<max(1e-11,float(np.max(abs(expected)))*1e-9), 'Signed cumulative work differs'
            Z=data[region+'_Z_'+part];delta=Z-Z[0];I=data[region+'_I_'+part+'_total']
            residual=delta-I
            assert np.max(abs(residual-data[region+'_balance_residual_'+part]))<max(1e-11,zscale*1e-10)
            total_variation=np.concatenate(([0.],np.cumsum(.5*(abs(powers[1:]).sum(axis=1)+abs(powers[:-1]).sum(axis=1))*np.diff(times))))
            denom=abs(delta)+total_variation
            active=denom>1e-14
            relative=float(np.max(abs(residual[active])/denom[active])) if np.any(active) else 0.
            assert relative<.02, ('Signed mean-vorticity temporal closure',region,part,relative)
            result[region+'_'+part]=dict(maximum_relative_temporal_residual=relative,
                maximum_absolute_temporal_residual=float(np.max(abs(residual))),production_split_max=difference)
    return result


def check_wind_momentum(folder,ustar,h):
    config=json.loads((folder/'configuration.json').read_text())
    assert config.get('maintain_background') is True, 'Inherited background maintenance disabled'
    rows=np.atleast_1d(np.genfromtxt(folder/'jet-profiles.csv',delimiter=',',names=True))
    times=np.unique(rows['time_s']);nz=len(rows)//len(times)
    ul=rows['UL'].reshape(len(times),nz)
    closure=rows['closure'].reshape(len(times),nz).mean(axis=1)
    forcing=rows['forcing'].reshape(len(times),nz).mean(axis=1)
    expected=ustar**2/h
    baseline=float(config['nu'])*(float(config['baseline_gradient_top'])-float(config['baseline_gradient_bottom']))/h
    tolerance=max(1e-12,abs(baseline+expected)*1e-9)
    assert np.max(abs(closure-baseline-expected))<tolerance, ('Incremental stress sign/magnitude',closure,baseline,expected)
    assert np.max(abs(forcing+baseline+expected))<tolerance, ('Original maintenance plus wind return',forcing,baseline,expected)
    wind=np.atleast_1d(np.genfromtxt(folder/'wind-budget.csv',delimiter=',',names=True))
    assert np.max(abs(wind['kinematic_surface_stress']-ustar**2))<1e-12
    assert np.max(abs(wind['total_surface_stress']-wind['baseline_surface_stress']-ustar**2))<1e-12
    assert np.max(abs(wind['total_bottom_stress']-wind['baseline_bottom_stress']))<1e-12
    assert np.max(abs(wind['wind_return_acceleration']+expected))<1e-12
    assert np.max(abs(wind['background_force_depth_mean']+baseline))<1e-12
    drift=float(np.max(abs(ul.mean(axis=1)-ul[0].mean())))
    assert drift<1e-10, ('Unexpected bulk momentum drift',drift)
    return dict(incremental_kinematic_stress=ustar**2,expected_incremental_mean_acceleration=expected,
                inherited_boundary_mean_acceleration=baseline,background_maintenance_preserved=True,
                maximum_bulk_momentum_drift=drift)


def read_map(path):
    manifest=json.loads(path.read_text());arrays={};hashes={path.name:digest(path)}
    for name in ('Uc','Wc','Q'):
        entry=manifest['arrays'][name];file=path.parent/entry['file']
        if digest(file)!=entry['sha256']:raise RuntimeError('Old/new mean-map hash differs')
        a=np.fromfile(file,dtype='<f8').reshape(entry['shape'],order='F')
        if not np.all(np.isfinite(a)):raise RuntimeError('Nonfinite comparison map')
        arrays[name]=a;hashes[file.name]=entry['sha256']
    return arrays,hashes


def map_at(folder,time):
    matches=[p for p in folder.glob('mean_*.json') if abs(float(json.loads(p.read_text())['time'])-time)<1e-7]
    if len(matches)!=1:raise RuntimeError('Missing or ambiguous reference mean map')
    return matches[0]


def representative_cases(cases):
    """Exercise each imported grid/projection/timestep, at its largest stress."""
    groups={}
    for case in cases:
        key=(case['source_id'],case.get('initial_kind','full'),float(case['env']['MAX_DT']))
        groups.setdefault(key,[]).append(case)
    return [max(group,key=lambda case:float(case['env']['WIND_USTAR']))
            for key,group in sorted(groups.items())]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();source=verify(root)
    label=os.environ.get('SLURM_JOB_ID',datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    output=root/'validation'/('attempt-'+label);output.mkdir(parents=True,exist_ok=False)
    julia=os.environ['JULIA'];python=os.environ.get('PYTHON',sys.executable)
    started=time.monotonic();report=dict(passed=False,source_version=source['version'],job_id=label,runs={},comparisons={})
    cases=json.loads((root/'campaign.json').read_text())['runs']

    def run_case(name,case,duration,diagnostic=.025,mean_interval=.05):
        settings,provenance=branch_settings(root,case,source)
        start=float(settings['START_TIME']);stop=start+duration;folder=output/name
        settings.update(STOP_TIME=stop,OUT_DIR=str(folder),DIAGNOSTIC_INTERVAL=diagnostic,
                        MOMENT_INTERVAL=diagnostic,REGIME_INTERVAL=mean_interval,
                        SNAPSHOT_TIMES=f'{start:g},{stop:g}',ARCH='GPU')
        env=dict(os.environ)
        for key in ('PICKUP','INIT_SNAPSHOT','INIT_METADATA','INIT_KIND','INIT_APPROVAL','INIT_SOURCE_ID','INIT_APPROVAL_SHA256',
                    'START_TIME','JET_MODE','JET_SIGN','JET_AMPLITUDE'):env.pop(key,None)
        env.update({k:str(v) for k,v in settings.items()})
        print('VALIDATION_CASE',name,flush=True);beginning=time.monotonic()
        with (output/(name+'.log')).open('w') as stream:
            subprocess.run([julia,'--startup-file=no',str(root/'run/run_jet.jl')],env=env,
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
        _,native=snapshot(folder);initial,_=snapshot(folder,True)
        parent=Path(settings['INIT_SNAPSHOT']);parent_manifest=json.loads(parent.read_text())
        initialization={}
        for key,entry in parent_manifest['arrays'].items():
            expected=np.fromfile(parent.parent/entry['file'],dtype='<f8').reshape(entry['shape'],order='F')
            if settings['INIT_KIND']=='mean':expected=expected.mean(axis=1,keepdims=True)
            difference=float(np.max(abs(initial[key]-expected)));scale=float(np.max(abs(expected)))
            if difference>max(64*np.finfo(float).eps*scale,1e-14):
                raise RuntimeError('Restart changed the native velocity field: '+key)
            initialization[key]=dict(max_abs_difference=difference,parent_max_abs=scale)
        report['runs'][name]=dict(native=native,initialization=initialization,
            budget=budgets(folder,two_d=settings['INIT_KIND']=='mean'),regime=regime_outputs(folder,two_d=settings['INIT_KIND']=='mean'),
            momentum=check_wind_momentum(folder,float(settings['WIND_USTAR']),float(settings['H'])),
            signed_vorticity=signed_vorticity_budget(folder,two_d=settings['INIT_KIND']=='mean'),
            wall_seconds=time.monotonic()-beginning,provenance=provenance,
            completed=json.loads((folder/'completed.json').read_text()))
        write_json(output/'progress.json',report);print('VALIDATED',name,flush=True)
        return folder

    try:
        report['external_references']=verify_references(root,source)
        # Every production intervention is resolved against the frozen approval;
        # short integrations cover the distinct grid/projection/timestep groups.
        report['approved_production_settings']={}
        for case in cases:
            settings,provenance=branch_settings(root,case,source)
            report['approved_production_settings'][case['id']]=dict(
                source_id=case['source_id'],initial_kind=settings['INIT_KIND'],
                wind_ustar=float(settings['WIND_USTAR']),
                grid=[int(settings[name]) for name in ('NX','NY','NZ')],
                max_dt=float(settings['MAX_DT']),provenance=provenance)
        subprocess.run([julia,'--startup-file=no',str(root/'run/test_shear_helpers.jl')],check=True)
        subprocess.run([julia,'--startup-file=no',str(root/'run/test_vorticity_budget.jl')],check=True)
        for arch in ('CPU','GPU'):
            print('TEST onset driver',arch,flush=True)
            subprocess.run([julia,'--startup-file=no',str(root/'run/test_onset_driver.jl')],
                env=dict(os.environ,DRIVER_TEST_ARCH=arch,ONSET_TEST_ROOT=str(output/('unit-'+arch.lower()))),check=True)
        for branch in ('control','wind-075','wind-125','wind-on','mean-control','mean-wind-on'):
            report['comparisons']['cpu_gpu_'+branch]=compare(output/'unit-cpu'/branch,output/'unit-gpu'/branch)
        subprocess.run([python,'-m','unittest','discover','-s',str(root/'run'),'-p','test_*.py','-v'],check=True)
        subprocess.run([python,'-m','unittest','discover','-s',str(root/'scripts'),'-p','test_*.py','-v'],check=True)
        subprocess.run([python,'-m','unittest','discover','-s',str(root/'analysis'),'-p','test_*.py','-v'],check=True)
        # One representative per imported grid/projection/timestep is sufficient
        # to check the shared code; all prescribed stresses were approved above.
        representatives=representative_cases(cases)
        report['full_grid_representatives']=[case['id'] for case in representatives]
        actual={}
        for case in representatives:
            actual[case['id']]=run_case('actual-'+case['id'],case,.2)
        # Compare only branches with the same approved native parent/grid.
        # Refined native parents must retain their own provenance and shape.
        source_ids=sorted({case['source_id'] for case in representatives})
        for source_id in source_ids:
            group=[case for case in representatives if case['source_id']==source_id]
            full_case=next((case for case in group if case.get('initial_kind','full')=='full'),None)
            if full_case is None:continue # Direct parent/projection check above remains compulsory.
            full_initial,_=snapshot(actual[full_case['id']],True)
            for case in group:
                fields,_=snapshot(actual[case['id']],True)
                expected={k:(v.mean(axis=1,keepdims=True) if case.get('initial_kind','full')=='mean' else v) for k,v in full_initial.items()}
                if any(fields[k].shape!=expected[k].shape for k in fields):raise RuntimeError('Same-source branch grid mismatch')
                differences={k:float(np.max(abs(fields[k]-expected[k]))) for k in fields}
                if max(differences.values())>1e-14:raise RuntimeError('Initial field/projection mismatch')
                report['comparisons']['prescribed_initial_field_'+case['id']]=differences
        full_case=next(case for case in representatives if case.get('initial_kind','full')=='full' and
                       approved_input(root,case['source_id'],source)[1].get('validation_reference_files'))
        # A separate, non-production full-state zero-wind branch is retained
        # solely to compare this driver against the preserved historical20s run.
        control=dict(full_case,id='validation-full-control',env=dict(full_case['env'],WIND_USTAR=0.0))
        actual[control['id']]=run_case('actual-'+control['id'],control,.2)
        added=max((case for case in representatives if case['source_id']==full_case['source_id'] and case.get('initial_kind','full')=='full'),key=lambda case:float(case['env']['WIND_USTAR']))
        initial_time=float(control['env']['START_TIME'])
        c0,_=read_map(map_at(actual[control['id']],initial_time));w0,_=read_map(map_at(actual[added['id']],initial_time))
        expected_q_jump=float(added['env']['WIND_USTAR'])**2/(2*float(added['env']['NU']))
        qdifference=w0['Q']-c0['Q']
        assert np.max(abs(qdifference[:,:-1]))<1e-11, 'Wind onset changed interior initial Q'
        assert np.max(abs(qdifference[:,-1]-expected_q_jump))<1e-11, 'Wind onset boundary Q jump differs from imposed stress'
        report['comparisons']['initial_boundary_vorticity_step']=dict(expected_top_cell_jump=expected_q_jump,
            max_interior_difference=float(np.max(abs(qdifference[:,:-1]))),
            max_top_cell_error=float(np.max(abs(qdifference[:,-1]-expected_q_jump))),
            scope='Native interior velocities are identical; changed top gradient alters only the initial boundary-centered Q definition.')
        continuation=run_case('old-parent-continuation',control,20,diagnostic=.1,mean_interval=.5)
        parent,approval,_=approved_input(root,control['source_id'],source);compare_time=initial_time+20
        reference_folder=input_locations(root,control['source_id'])['validation_reference']
        old,old_hashes=read_map(map_at(reference_folder,compare_time));new,new_hashes=read_map(map_at(continuation,compare_time))
        if any(approval.get('validation_reference_files',{}).get(name)!=sha for name,sha in old_hashes.items()):
            raise RuntimeError('Continuation reference is not pinned by the frozen approval')
        differences={}
        for key in old:
            reference=old[key][:,2:-2];value=new[key][:,2:-2]
            absolute=float(np.sqrt(np.mean((value-reference)**2)));reference_rms=float(np.sqrt(np.mean(reference**2)))
            relative=absolute/max(reference_rms,1e-10)
            if relative>.01:raise RuntimeError('Restarted trajectory differs from old parent by over1%: '+key)
            differences[key]=dict(rms_absolute=absolute,rms_reference=reference_rms,relative=relative)
        report['comparisons']['old_parent_twenty_second_continuation']=dict(time=compare_time,fields=differences,
            old_reference_hashes=old_hashes,new_reference_hashes=new_hashes,
            scope='Control trajectory retains original background forcing and boundary gradients with zero added wind. Adaptive restart timesteps can differ. Interior error threshold1%, not bitwise equality.')
        verify(root)
        report['external_references']=verify_references(root,source)
        for case in cases:branch_settings(root,case,source)
        report.update(passed=True,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
            scope='Software and approved restart validation: CPU/GPU agreement, unchanged full initial fields or exact transverse projections, retained background maintenance, incremental wind-stress/bulk momentum balance, native forced-vorticity identities, signed enstrophy work/temporal closure and twenty-second old-parent control continuation. No scientific outcome is inferred.')
        write_json(output/'result.json',report);write_json(root/'validation/passed.json',report)
        print('VALIDATION PASSED',source['version'],flush=True)
    except Exception as error:
        report.update(error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        write_json(output/'result.json',report);raise

if __name__=='__main__':main()


