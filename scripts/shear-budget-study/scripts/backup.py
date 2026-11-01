"""Verified archive on persistent storage before science completion is advertised."""
import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from guard import digest,write_json,safe_relative

def tree_hashes(folder):
    result={}
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():raise RuntimeError('Scientific output contains a symlink')
        if path.is_file():result[path.relative_to(folder).as_posix()]=digest(path)
        elif not path.is_dir():raise RuntimeError('Scientific output contains a nonregular file')
    if not result:raise RuntimeError('Cannot back up an empty run')
    return result

def archive_hashes(path):
    result={}
    with tarfile.open(path,'r|gz') as archive:
        for member in archive:
            safe_relative(member.name)
            if not member.isfile() or member.name in result:raise RuntimeError('Unsafe or duplicate backup member')
            stream=archive.extractfile(member)
            result[member.name]=hashlib.file_digest(stream,'sha256').hexdigest()
    return result

def backup_run(folder,backup,case,receipt):
    root=Path(backup);root.mkdir(parents=True,exist_ok=True)
    # The two active bundles share one quota. Serialize the space check and
    # archive write so both cannot reserve the same free bytes independently.
    with (root/'.archive.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _backup_run(folder,root,case,receipt)

def _backup_run(folder,backup,case,receipt):
    parent=Path(backup)/'runs';parent.mkdir(parents=True,exist_ok=True)
    destination=parent/case['id'];files=receipt['files']
    size=sum((folder/safe_relative(n)).stat().st_size for n in files)
    if shutil.disk_usage(parent).free < size+2*1024**3:
        raise RuntimeError('Persistent backup has insufficient free space; native run preserved on scratch')
    if destination.exists():
        saved=json.loads((destination/'completion-receipt.json').read_text())
        proof=json.loads((destination/'COMPLETE.json').read_text())
        if (saved!=receipt or proof.get('complete') is not True or
            proof.get('archive_sha256')!=digest(destination/'native-output.tar.gz') or
            archive_hashes(destination/'native-output.tar.gz')!=files):
            raise RuntimeError('Existing persistent backup differs')
        return destination
    temporary=Path(tempfile.mkdtemp(prefix='.'+case['id']+'.incomplete-',dir=parent))
    try:
        archive_path=temporary/'native-output.tar.gz'
        with tarfile.open(archive_path,'w:gz',compresslevel=1) as archive:
            for name in sorted(files):archive.add(folder/name,arcname=name,recursive=False)
        if archive_hashes(archive_path)!=files or tree_hashes(folder)!=files:
            raise RuntimeError('Completed files changed during persistent copy')
        write_json(temporary/'completion-receipt.json',receipt)
        write_json(temporary/'COMPLETE.json',dict(complete=True,source_version=receipt['source_version'],
            run_id=case['id'],receipt_sha256=digest(temporary/'completion-receipt.json'),
            archive_sha256=digest(archive_path),archive_file='native-output.tar.gz',
            scope='Every regular run file, including all native snapshots and mean maps; verified by streaming each archived member.'))
        for path in temporary.iterdir():
            with path.open('rb') as stream:os.fsync(stream.fileno())
        temporary.rename(destination)
        descriptor=os.open(parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        return destination
    finally:
        if temporary.exists():shutil.rmtree(temporary)
