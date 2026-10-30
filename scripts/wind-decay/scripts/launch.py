#!/usr/bin/env python3
"""One isolated matched continuation per array task; never alters its parent."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
from guard import verify, write_json, digest, safe_relative
from inputs import branch_settings, approved_input
from backup import tree_hashes, backup_run


def snapshot_at(folder,time):
    matches=[]
    for path in sorted(folder.glob('t*.json')):
        if abs(float(json.loads(path.read_text()).get('time',-1))-time)<1e-7:matches.append(path)
    if len(matches)!=1:raise RuntimeError('Expected one native snapshot at prescribed time')
    path=matches[0]
    for item in json.loads(path.read_text())['arrays'].values():
        if digest(folder/safe_relative(item['file']))!=item['sha256']:
            raise RuntimeError('Native snapshot checksum mismatch')
    if not path.with_suffix('.toml').is_file():raise RuntimeError('Native snapshot metadata missing')
    return path


def verify_run(folder,settings):
    result=json.loads((folder/'completed.json').read_text())
    start,stop=(float(settings[k]) for k in ('START_TIME','STOP_TIME'))
    if abs(float(result.get('time',-1))-stop)>1e-7:raise RuntimeError('Branch lacks prescribed completed time')
    for name in ('configuration.json','grid.json','completed.json','jet-budget.csv','jet-profiles.csv',
                 'jet-fluxes.csv','regime-budget.csv','wind-budget.csv'):
        if not (folder/name).is_file():raise RuntimeError('Required diagnostic missing: '+name)
    config=json.loads((folder/'configuration.json').read_text())
    for name,key in (('nx','NX'),('ny','NY'),('nz','NZ'),('wind_ustar','WIND_USTAR'),
                     ('start_time','START_TIME'),('stop_time','STOP_TIME')):
        if abs(float(config[name])-float(settings[key]))>1e-10:
            raise RuntimeError('Completed branch configuration differs: '+name)
    if config.get('initial_snapshot_sha256')!=digest(Path(settings['INIT_SNAPSHOT'])):
        raise RuntimeError('Branch did not start from its prescribed native state')
    imported=config.get('approved_parent_import') or {}
    if (imported.get('source_id')!=settings['INIT_SOURCE_ID'] or
        imported.get('approval_sha256')!=settings['INIT_APPROVAL_SHA256']):
        raise RuntimeError('Branch lacks the prescribed approved-parent provenance')
    interval=float(settings['REGIME_INTERVAL']);times=[]
    expected=sorted(set([start,stop,*[i*interval for i in range(round(start/interval),round(stop/interval)+1)
                                     if start-1e-7<=i*interval<=stop+1e-7]]))
    for path in sorted(folder.glob('mean_*.json')):
        manifest=json.loads(path.read_text());times.append(float(manifest['time']))
        if 'Qdot_3d_native' not in manifest['arrays']:raise RuntimeError('Native omitted tendency missing')
        for item in manifest['arrays'].values():
            if digest(folder/safe_relative(item['file']))!=item['sha256']:
                raise RuntimeError('Mean-map checksum mismatch')
    if len(times)!=len(expected) or any(abs(a-b)>1e-7 for a,b in zip(times,expected)):
        raise RuntimeError('Mean-map schedule is incomplete')
    for time in {start,stop,*map(float,str(settings['SNAPSHOT_TIMES']).split(','))}:
        if start<=time<=stop:snapshot_at(folder,time)
    return dict(completed=True,time=stop,start_time=start,map_count=len(times),files=tree_hashes(folder))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int,required=True)
    p.add_argument('--finalize-existing',action='store_true',help='Only finalize an already verified finished branch; never integrate a partial run.')
    a=p.parse_args();root=a.root.resolve();source=verify(root,require_validation=True)
    runs=json.loads((root/'campaign.json').read_text())['runs']
    if not 0<=a.index<len(runs):raise ValueError('Array index out of range')
    case=runs[a.index];folder=root/safe_relative(case['path']);status=root/'data/status'/case['id'];status.mkdir(parents=True,exist_ok=True)
    common=dict(id=case['id'],source_version=source['version'],job_id=os.environ.get('SLURM_JOB_ID'),array_index=a.index,env=case['env'])
    with (status/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (status/'completed.json').exists():raise RuntimeError('Completed branch preserved; refusing duplicate integration')
        try:
            settings,provenance=branch_settings(root,case,source)
            if folder.exists() and not a.finalize_existing:raise RuntimeError('Existing branch preserved; no automatic overwrite or restart')
            paths=json.loads((root/'runtime/paths.json').read_text());backup=Path(paths['backup']);backup.mkdir(parents=True,exist_ok=True)
            if shutil.disk_usage(root).free<8*1024**3:raise RuntimeError('Scratch free space below 8 GiB')
            if shutil.disk_usage(backup).free<6*1024**3:raise RuntimeError('Persistent free space below 6 GiB')
            if folder.exists():
                if not (status/'integration-verified.json').is_file():
                    raise RuntimeError('No prior verified completed integration; partial output is preserved')
                proof=verify_run(folder,settings);saved=json.loads((status/'integration-verified.json').read_text())
                if saved['files']!=proof['files'] or saved['source_version']!=source['version']:
                    raise RuntimeError('Previously verified integration changed')
            else:
                if a.finalize_existing:raise RuntimeError('No existing branch to finalize')
                write_json(status/'started.json',common|provenance|dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
                env=dict(os.environ)
                for key in ('PICKUP','INIT_SNAPSHOT','INIT_KIND','INIT_APPROVAL','INIT_SOURCE_ID','INIT_APPROVAL_SHA256',
                            'START_TIME','JET_MODE','JET_SIGN','JET_AMPLITUDE'):
                    env.pop(key,None)
                env.update({k:str(v) for k,v in settings.items()});env['OUT_DIR']=str(folder)
                subprocess.run([os.environ['JULIA'],'--startup-file=no',str(root/'run/run_jet.jl')],env=env,check=True)
                verify(root,require_validation=True);approved_input(root,case['source_id'],source)
                proof=verify_run(folder,settings)
                write_json(status/'integration-verified.json',proof|dict(source_version=source['version']))
            receipt=common|provenance|dict(completed=True,completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                          files=proof['files'],persistent_backup_verified=True)
            backup_run(folder,backup,case,receipt);write_json(status/'completed.json',receipt)
            print('COMPLETED_AND_BACKED_UP',case['id'],flush=True)
        except Exception as error:
            write_json(status/'failed.json',common|dict(failed=True,error_type=type(error).__name__,message=str(error),
                                                       finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            raise

if __name__=='__main__':main()
