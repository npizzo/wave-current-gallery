"""Resolve private restart paths against the frozen, public approval record."""
import json
import fcntl
from pathlib import Path
import shutil
import tempfile
import tomllib
from guard import digest, safe_relative, write_json


def input_locations(root,source_id):
    value=json.loads((Path(root)/'runtime/input-paths.json').read_text())['sources'][source_id]
    if not isinstance(value,dict):raise RuntimeError('Onset inputs require explicit snapshot, metadata and parent_configuration paths')
    result={k:Path(value[k]) for k in ('snapshot','metadata','parent_configuration')}
    result['validation_reference']=Path(value.get('validation_reference',result['snapshot'].parent))
    if any(not path.is_absolute() for path in result.values()):raise RuntimeError('Private input locations must be absolute')
    return result


def approved_input(root, source_id, source_manifest=None):
    root=Path(root)
    approval=root/'source_provenance.toml'
    if source_manifest is not None:
        expected=source_manifest['files'].get('source_provenance.toml')
        if not expected or digest(approval)!=expected:
            raise RuntimeError('Restart approval differs from frozen source')
    document=tomllib.loads(approval.read_text())
    if document.get('format')!='wind-onset-parent-approval-v1' or document.get('campaign')!='unity-wind-onset-tests':
        raise RuntimeError('Incorrect restart approval format/campaign')
    record=document['sources'][source_id]
    locations=input_locations(root,source_id)
    path=locations['snapshot']
    if not path.is_absolute() or path.suffix!='.json':
        raise RuntimeError('Restart path must be an absolute native JSON manifest')
    metadata_path=locations['metadata']
    if digest(path)!=record['snapshot_json_sha256'] or digest(metadata_path)!=record['snapshot_metadata_sha256']:
        raise RuntimeError('Approved restart manifest/metadata hash differs')
    data=json.loads(path.read_text());metadata=tomllib.loads(metadata_path.read_text())
    if (metadata.get('translation')!='legacy-native-json-plus-original-configuration-v1' or
        digest(locations['parent_configuration'])!=record['parent_configuration_sha256'] or
        metadata.get('original_configuration_sha256')!=record['parent_configuration_sha256']):
        raise RuntimeError('Translated metadata does not bind the original parent configuration')
    if record.get('expected_maintain_background') is not True or metadata.get('parameters',{}).get('maintain') is not True:
        raise RuntimeError('Original background maintenance is not retained in the approved state')
    if metadata.get('json_sha256')!=record['snapshot_json_sha256']:
        raise RuntimeError('Restart metadata does not bind the approved JSON')
    if abs(float(data['time'])-float(record['snapshot_time']))>1e-9 or abs(float(metadata['time'])-float(record['snapshot_time']))>1e-9:
        raise RuntimeError('Approved restart physical time differs')
    for name in ('parameters','source_hashes','arrays'):
        if metadata[name]!=record[name]:raise RuntimeError('Approved restart '+name+' differs')
    if data['arrays']!=record['arrays']:raise RuntimeError('Approved native array inventory differs')
    for entry in record['arrays'].values():
        relative=safe_relative(entry['file'])
        if len(relative.parts)!=1:raise RuntimeError('Native array filename must be a basename')
        if digest(path.parent/relative)!=entry['sha256']:raise RuntimeError('Approved native array changed')
    for name,sha in record.get('validation_reference_files',{}).items():
        relative=safe_relative(name)
        if len(relative.parts)!=1:raise RuntimeError('Validation reference filename must be a basename')
        if digest(locations['validation_reference']/relative)!=sha:raise RuntimeError('Approved continuation reference changed')
    return path,record,digest(approval)


def branch_settings(root,case,source_manifest):
    path,record,approval_hash=approved_input(root,case['source_id'],source_manifest)
    env=dict(case['env'])
    if float(env['START_TIME'])!=float(record['snapshot_time']):
        raise RuntimeError('Branch does not begin at its approved state')
    if float(env['STOP_TIME'])<=float(env['START_TIME']):
        raise RuntimeError('Branch duration must be positive')
    if not any(abs(float(env['WIND_USTAR'])-float(x))<1e-15 for x in record['allowed_branch_ustar']):
        raise RuntimeError('Wind stress is not an approved intervention')
    if int(env.get('MAINTAIN_BACKGROUND',0))!=1:raise RuntimeError('Onset branches must retain original background maintenance')
    env.update(INIT_SNAPSHOT=str(path),INIT_KIND='full',INIT_APPROVAL=str(Path(root)/'source_provenance.toml'),
               INIT_METADATA=str(input_locations(root,case['source_id'])['metadata']),
               INIT_SOURCE_ID=case['source_id'],INIT_APPROVAL_SHA256=approval_hash,
               JET_MODE=0,JET_SIGN=0,JET_AMPLITUDE=0,
               MOMENTS_ENABLED=0,KEEP_CHECKPOINT=0)
    return env,dict(source_id=case['source_id'],snapshot_json_sha256=record['snapshot_json_sha256'],
                    snapshot_metadata_sha256=record['snapshot_metadata_sha256'],
                    parent_configuration_sha256=record['parent_configuration_sha256'],
                    approval_sha256=approval_hash,source_time=float(record['snapshot_time']))


def preserve_inputs(root,backup,source_manifest):
    """Keep an exact, independently verified parent copy on persistent storage."""
    root,backup=Path(root),Path(backup);parent=backup/'initial-states';parent.mkdir(parents=True,exist_ok=True)
    source_ids=sorted({case['source_id'] for case in json.loads((root/'campaign.json').read_text())['runs']})
    with (parent/'.copy.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for source_id in source_ids:
            if len(safe_relative(source_id).parts)!=1:raise RuntimeError('Unsafe source identifier')
            path,record,approval_hash=approved_input(root,source_id,source_manifest)
            locations=input_locations(root,source_id)
            origins={path.name:path,locations['metadata'].name:locations['metadata'],locations['parent_configuration'].name:locations['parent_configuration']}
            if len(origins)!=3:raise RuntimeError('Input filenames collide in persistent copy')
            wanted={path.name:record['snapshot_json_sha256'],locations['metadata'].name:record['snapshot_metadata_sha256'],
                    locations['parent_configuration'].name:record['parent_configuration_sha256']}
            wanted.update({v['file']:v['sha256'] for v in record['arrays'].values()})
            origins.update({v['file']:path.parent/v['file'] for v in record['arrays'].values()})
            wanted.update(record.get('validation_reference_files',{}))
            origins.update({name:locations['validation_reference']/name for name in record.get('validation_reference_files',{})})
            destination=parent/source_id
            if destination.exists():
                proof=json.loads((destination/'verified-input.json').read_text())
                if proof.get('files')!=wanted or proof.get('approval_sha256')!=approval_hash:
                    raise RuntimeError('Persistent parent approval differs')
                if any(digest(destination/name)!=sha for name,sha in wanted.items()):
                    raise RuntimeError('Persistent parent copy changed')
                continue
            size=sum(origins[name].stat().st_size for name in wanted)
            if shutil.disk_usage(parent).free<size+2*1024**3:raise RuntimeError('Insufficient space to preserve the approved parent')
            temp=Path(tempfile.mkdtemp(prefix='.'+source_id+'-',dir=parent))
            try:
                for name,sha in wanted.items():
                    shutil.copy2(origins[name],temp/name)
                    if digest(temp/name)!=sha:raise RuntimeError('Persistent parent copy hash mismatch')
                write_json(temp/'verified-input.json',dict(source_id=source_id,files=wanted,approval_sha256=approval_hash,
                    scope='Exact original snapshot JSON, metadata and native arrays, plus any explicitly approved continuation-reference files. Original simulation is unchanged.'))
                temp.rename(destination)
            finally:
                if temp.exists():shutil.rmtree(temp)
