"""Validate the exact directory published by GitHub Pages."""
from pathlib import Path, PurePosixPath
import argparse
import json
import re

ALLOWED = {'.html', '.css', '.js', '.json', '.csv', '.png', '.jpg', '.jpeg', '.svg', '.pdf', '.webp', '.mp4', '.txt'}
PRIVATE = re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]{20,}|/(?:Users|home|u|work)/[A-Za-z0-9_.-]+|privkey\.key')


def local_asset(root, value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or ':' in value or '\\' in value:
        raise ValueError(f'Unsafe gallery asset reference: {value!r}')
    target = root / path
    if not target.is_file() or target.is_symlink():
        raise ValueError(f'Missing gallery asset: {value!r}')


def check(root):
    root = Path(root).resolve()
    total = 0
    for path in root.rglob('*'):
        if path.is_symlink(): raise ValueError('Symlinks must not be published')
        if not path.is_file(): continue
        if path.suffix.lower() not in ALLOWED or any(x.startswith('.') for x in path.relative_to(root).parts):
            raise ValueError(f'Non-public artifact in gallery: {path.relative_to(root)}')
        size = path.stat().st_size
        if size > 90 * 1024**2: raise ValueError('Individual gallery file exceeds 90 MiB')
        total += size
        if path.suffix in {'.html', '.css', '.js', '.json', '.csv', '.svg', '.txt'}:
            if PRIVATE.search(path.read_text()): raise ValueError(f'Private path or credential pattern: {path.relative_to(root)}')
    if total > 800 * 1024**2: raise ValueError('Gallery exceeds 800 MiB')
    data = json.loads((root / 'data/campaign.json').read_text())
    assert data['schema_version'] == 1
    assert data['campaign']['total_pairs'] == 7
    for item in data.get('members', []) + [{'figures': data.get('results', []), 'downloads': data.get('downloads', [])}]:
        for figure in item.get('figures', []): local_asset(root, figure['src'])
        for download in item.get('downloads', []): local_asset(root, download['href'])
    print(f'Gallery checked: {total / 1024**2:.2f} MiB; only derived public artifacts.')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gallery', type=Path)
    check(parser.parse_args().gallery)

