#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(data,stream,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)

def safe_relative(value):
    p=Path(value)
    if p.is_absolute() or not p.parts or '..' in p.parts or str(p)=='.':
        raise ValueError('Unsafe relative path')
    return p

def inventory(root):
    root=Path(root)
    files=[root/'campaign.json',root/'PROTOCOL.txt']
    suffixes={'.py','.jl','.sh','.sbatch','.toml','.txt','.html','.css','.js','.json','.csv'}
    for folder in ('run','analysis','scripts','environment','reference'):
        files.extend(p for p in (root/folder).rglob('*') if p.is_file() and
                     '__pycache__' not in p.parts and p.suffix in suffixes)
    return {str(p.relative_to(root)):digest(p) for p in sorted(files)}

def version_for(files):
    return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()[:16]

def verify(root,require_validation=False):
    root=Path(root).resolve();saved=json.loads((root/'source-manifest.json').read_text())
    if version_for(saved['files'])!=saved['version'] or inventory(root)!=saved['files']:
        raise RuntimeError('Campaign source/environment changed; validation must be repeated')
    if require_validation:
        result=json.loads((root/'validation/passed.json').read_text())
        if not result.get('passed') or result['source_version']!=saved['version']:
            raise RuntimeError('Matching validation has not passed')
    return saved

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','verify']);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if a.action=='freeze':
        files=inventory(a.root);version=version_for(files)
        if (a.root/'submission.json').exists():
            old=json.loads((a.root/'source-manifest.json').read_text())
            if old['version']!=version:raise RuntimeError('Cannot replace source of a submitted campaign')
        write_json(a.root/'source-manifest.json',{'version':version,'files':files});print(version)
    else: print(verify(a.root)['version'])
if __name__=='__main__':main()
