"""Export verified, already averaged fields; leave completed simulations untouched."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root, out = args.root, args.output
    out.mkdir(parents=True, exist_ok=True)
    campaign = json.loads((root / 'campaign.json').read_text())
    source = json.loads((root / 'source-manifest.json').read_text())
    shutil.copyfile(root / 'campaign.json', out / 'campaign.json')
    records = []
    for case in campaign['runs']:
        rid = case['id']
        cache = root / 'data/analysis-cache' / rid
        receipt = json.loads((root / 'data/status' / rid / 'completed.json').read_text())
        marker = json.loads((cache / 'cache.json').read_text())
        key = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
        if not receipt['completed'] or not receipt['persistent_backup_verified']:
            raise ValueError('Unverified completion: ' + rid)
        if receipt['source_version'] != source['version'] or marker['key'] != key:
            raise ValueError('Source or receipt mismatch: ' + rid)
        for name in ('pair-fields.npz', 'summary.json'):
            if sha(cache / name) != marker['files'][name]:
                raise ValueError('Changed analysis cache: ' + rid + '/' + name)
        target = out / 'data' / rid
        target.mkdir(parents=True, exist_ok=True)
        with np.load(cache / 'pair-fields.npz', allow_pickle=False) as data:
            times = data['time']
            indices = sorted(set(int(np.argmin(abs(times-t))) for t in (0, 120, 300, 600)))
            fields = {name: data[name] for name in ('Uc', 'Wc', 'Q')}
            for field in fields.values():
                if not np.isfinite(field).all():
                    raise ValueError('Nonfinite field: ' + rid)
            np.savez_compressed(target / 'pair-fields.npz', time=times[indices],
                z=data['z'], dz=data['dz'], profile_time=times,
                profile_Uc=fields['Uc'].mean(axis=1),
                **{name: field[indices] for name, field in fields.items()})
        shutil.copyfile(cache / 'summary.json', target / 'summary.json')
        record = {'id': rid, 'source_version': source['version'], 'receipt_sha256': key,
            'original_pair_fields_sha256': marker['files']['pair-fields.npz'],
            'exported_pair_fields_sha256': sha(target / 'pair-fields.npz'),
            'summary_sha256': sha(target / 'summary.json'),
            'selected_times_s': [float(times[i]) for i in indices]}
        records.append(record)
    (out / 'export-provenance.json').write_text(json.dumps({
        'scope': 'Saved transverse means only; selected maps and all saved horizontal mean profiles. No native 3D snapshots.',
        'campaign_sha256': sha(out / 'campaign.json'), 'runs': records}, indent=2) + '\n')
    print('Exported verified mean fields for', len(records), 'completed runs.')

if __name__ == '__main__':
    main()
