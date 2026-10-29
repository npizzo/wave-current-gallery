#!/usr/bin/env python3
"""One isolated wind experiment, including its forecasts and paired probes."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
from guard import verify, write_json, digest, safe_relative
from backup import tree_hashes, backup_run


def stage_plan(case):
    """Every branch starts from the parent, never from another branch."""
    base = dict(case['env'])
    stop = float(base['STOP_TIME'])
    starts = case.get('forecast_times', case.get('forecasts', [600, 1000]))
    probe_start = float(case.get('probe_start', 600))
    duration = float(case.get('branch_duration', 200))
    modes = case.get('probe_modes', [1, 2, 4])
    snapshots = sorted(set([0, stop, *([probe_start] if modes else []), *map(float, starts)]))
    if any(t < 0 or t + duration > stop for t in [*([probe_start] if modes else []), *map(float, starts)]):
        raise ValueError('Branch interval must fit in parent history')
    if len(set(modes)) != len(modes) or any(int(m) != m or m < 1 for m in modes):
        raise ValueError('Jet modes must be distinct positive integers')
    base.update(INIT_SNAPSHOT='', INIT_KIND='full', START_TIME=0,
                JET_MODE=0, JET_SIGN=0, JET_AMPLITUDE=0,
                SNAPSHOT_TIMES=','.join(f'{t:g}' for t in snapshots),
                MAINTAIN_BACKGROUND=0, MOMENTS_ENABLED=0, KEEP_CHECKPOINT=0)
    result = [('parent', base, None)]
    for start in starts:
        env = dict(base, NY=1, INIT_KIND='mean', START_TIME=start,
                   STOP_TIME=float(start)+duration, SNAPSHOT_TIMES=f'{start:g},{float(start)+duration:g}')
        result.append((f'forecast-{start:g}', env, float(start)))
    for mode in modes:
        for sign, label in [(1, 'plus'), (-1, 'minus')]:
            env = dict(base, INIT_KIND='full', START_TIME=probe_start,
                       STOP_TIME=probe_start+duration, JET_MODE=mode, JET_SIGN=sign,
                       JET_AMPLITUDE=float(case.get('probe_amplitude', 2e-4)),
                       SNAPSHOT_TIMES=f'{probe_start:g},{probe_start+duration:g}')
            result.append((f'probe-m{mode}-{label}', env, probe_start))
    return result


def snapshot_at(folder, time):
    matches = []
    for path in sorted(folder.glob('t*.json')):
        manifest = json.loads(path.read_text())
        if abs(float(manifest.get('time', -1))-time) < 1e-7:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one native snapshot at {time}, got {len(matches)}')
    path = matches[0]
    for item in json.loads(path.read_text())['arrays'].values():
        if digest(folder/safe_relative(item['file'])) != item['sha256']:
            raise RuntimeError('Native parent snapshot checksum mismatch')
    return path


def verify_stage(folder, settings):
    result = json.loads((folder/'completed.json').read_text())
    start = float(settings.get('START_TIME', 0)); stop = float(settings['STOP_TIME'])
    if abs(float(result.get('time', -1))-stop) > 1e-7:
        raise RuntimeError('Stage lacks its prescribed completed time')
    for name in ('configuration.json', 'grid.json', 'completed.json', 'jet-budget.csv',
                 'jet-profiles.csv', 'jet-fluxes.csv', 'regime-budget.csv', 'wind-budget.csv'):
        if not (folder/name).is_file():
            raise RuntimeError('Required diagnostic missing: '+name)
    maps = sorted(folder.glob('mean_*.json'))
    if not maps:
        raise RuntimeError('Mean maps are missing')
    interval = float(settings['REGIME_INTERVAL'])
    expected = sorted(set([start, stop, *[i*interval for i in range(round(start/interval), round(stop/interval)+1)
                                         if start-1e-7 <= i*interval <= stop+1e-7]]))
    times = []
    for path in maps:
        fields = json.loads(path.read_text()); times.append(float(fields['time']))
        if 'Qdot_3d_native' not in fields['arrays']:
            raise RuntimeError('Native closure tendency missing')
        for item in fields['arrays'].values():
            if digest(folder/safe_relative(item['file'])) != item['sha256']:
                raise RuntimeError('Mean-map checksum mismatch')
    if len(times) != len(expected) or any(abs(a-b)>1e-7 for a,b in zip(times,expected)):
        raise RuntimeError(f'Mean-map schedule incomplete: {times[:3]} ... {times[-3:]}')
    snapshot_at(folder, start); snapshot_at(folder, stop)
    return {'completed': True, 'time': stop, 'start_time': start,
            'map_count': len(maps), 'files': tree_hashes(folder)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--index', type=int, required=True)
    p.add_argument('--resume-completed-stages', action='store_true',
                   help='Only reuse independently verified complete stages; partial stages still stop.')
    a=p.parse_args(); root=a.root.resolve(); source=verify(root, require_validation=True)
    runs=json.loads((root/'campaign.json').read_text())['runs']
    if not 0 <= a.index < len(runs): raise ValueError('Array index out of range')
    case=runs[a.index]; folder=root/safe_relative(case['path']); status=root/'data/status'/case['id']
    status.mkdir(parents=True, exist_ok=True)
    common=dict(id=case['id'], source_version=source['version'], job_id=os.environ.get('SLURM_JOB_ID'),
                array_index=a.index, env=case['env'])
    with (status/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (status/'completed.json').exists():
            raise RuntimeError('Completed experiment is preserved; refusing duplicate integration')
        try:
            if folder.exists() and not a.resume_completed_stages:
                raise RuntimeError('Existing experiment is preserved; explicit completed-stage resume is required')
            if shutil.disk_usage(root).free < 12*1024**3:
                raise RuntimeError('Scratch free space is below 12 GiB')
            paths=json.loads((root/'runtime/paths.json').read_text())
            backup=Path(paths['backup']); backup.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(backup).free < 8*1024**3:
                raise RuntimeError('Persistent storage free space is below 8 GiB')
            write_json(status/'started.json', common|dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            stages=[]
            for name, settings, parent_time in stage_plan(case):
                target=folder/name
                if parent_time is not None:
                    settings['INIT_SNAPSHOT']=str(snapshot_at(folder/'parent', parent_time))
                if target.exists():
                    if not a.resume_completed_stages or not (target/'completed.json').exists():
                        raise RuntimeError(f'Existing partial stage {name} is preserved; no automatic restart')
                    proof=verify_stage(target, settings)
                    saved=json.loads((status/'stages'/f'{name}.json').read_text())
                    if saved['files'] != proof['files'] or saved['source_version'] != source['version']:
                        raise RuntimeError('Completed stage changed; explicit investigation required')
                    print('REUSING_VERIFIED_STAGE', case['id'], name, flush=True)
                else:
                    env=dict(os.environ)
                    for key in ('PICKUP', 'INIT_SNAPSHOT', 'INIT_KIND', 'START_TIME', 'JET_MODE', 'JET_SIGN', 'JET_AMPLITUDE'):
                        env.pop(key, None)
                    env.update({k:str(v) for k,v in settings.items()}); env['OUT_DIR']=str(target)
                    write_json(status/'active-stage.json', common|dict(stage=name, start_time=settings.get('START_TIME',0), stop_time=settings['STOP_TIME']))
                    print('STAGE', case['id'], name, flush=True)
                    subprocess.run([os.environ['JULIA'], '--startup-file=no', str(root/'run/run_jet.jl')], env=env, check=True)
                    verify(root, require_validation=True)
                    proof=verify_stage(target, settings)
                    write_json(status/'stages'/f'{name}.json', proof|dict(source_version=source['version'], settings=settings))
                stages.append({'name':name,'start_time':settings.get('START_TIME',0),'stop_time':settings['STOP_TIME']})
            receipt=common|dict(completed=True, completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                stages=stages, files=tree_hashes(folder), persistent_backup_verified=True)
            backup_run(folder, backup, case, receipt)
            write_json(status/'completed.json', receipt)
            print('COMPLETED_AND_BACKED_UP',case['id'],flush=True)
        except Exception as error:
            write_json(status/'failed.json',common|dict(failed=True,error_type=type(error).__name__,
                       message=str(error),finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            raise

if __name__=='__main__': main()
