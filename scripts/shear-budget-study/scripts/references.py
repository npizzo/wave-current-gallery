"""Verify immutable completed references; paths are private, hashes are frozen."""
import json
from pathlib import Path
from guard import digest,safe_relative


def verify_references(root,source=None):
    root=Path(root)
    manifest=root/'references-manifest.json'
    if source is not None and digest(manifest)!=source['files'].get('references-manifest.json'):
        raise RuntimeError('Reference fingerprints differ from frozen source')
    frozen=json.loads(manifest.read_text()).get('references',{})
    cases=json.loads((root/'campaign.json').read_text()).get('references',[])
    if {c['id'] for c in cases}!=set(frozen):raise RuntimeError('Reference inventory differs from campaign')
    if not cases:return {}
    locations=json.loads((root/'runtime/references.json').read_text())['references']
    result={}
    for case in cases:
        identity=case['id'];pin=frozen[identity];place=locations[identity]
        run=Path(place['root']);campaign=Path(place['campaign_root'])
        if not run.is_absolute() or not campaign.is_absolute():raise RuntimeError('Reference paths must be absolute')
        if place.get('files',pin['files'])!=pin['files']:raise RuntimeError('Private reference hashes differ from frozen pins')
        if pin['source_version']!=case['source_version'] or pin['case_id']!=case['case_id']:
            raise RuntimeError('Reference identity differs from frozen pins')
        receipt_path=campaign/'data/status'/case['case_id']/'completed.json'
        receipt=json.loads(receipt_path.read_text())
        if receipt.get('completed') is not True or receipt.get('persistent_backup_verified') is not True or receipt.get('source_version')!=pin['source_version'] or receipt.get('id')!=case['case_id']:
            raise RuntimeError('External reference is not a verified completed run')
        if pin.get('receipt_sha256') and digest(receipt_path)!=pin['receipt_sha256']:
            raise RuntimeError('External completion receipt differs from pin')
        for name,sha in pin['files'].items():
            relative=safe_relative(name)
            if receipt['files'].get(name)!=sha or digest(run/relative)!=sha:
                raise RuntimeError('External reference file changed: '+identity+'/'+name)
        # Receipt claims are backed by the original archive completion proof.
        paths=json.loads((campaign/'runtime/paths.json').read_text());backup=Path(paths['backup'])/'runs'/case['case_id']
        proof=json.loads((backup/'COMPLETE.json').read_text())
        saved=backup/'completion-receipt.json'
        if proof.get('complete') is not True or proof.get('source_version')!=pin['source_version'] or proof.get('receipt_sha256')!=digest(saved) or json.loads(saved.read_text())!=receipt or not (backup/'native-output.tar.gz').is_file():
            raise RuntimeError('External reference persistent backup proof differs')
        result[identity]=dict(root=str(run),campaign_root=str(campaign),source_version=pin['source_version'],case_id=case['case_id'],files=pin['files'],receipt_sha256=digest(receipt_path))
    return result
