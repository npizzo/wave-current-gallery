"""Export a completed-output snapshot and push only the public gallery."""
from pathlib import Path
import argparse
import fcntl
import os
import shlex
import subprocess
import sys
import re
import tempfile

sys.dont_write_bytecode = True


def run(argv, root, **kwargs):
    return subprocess.run(argv, cwd=root, check=True, text=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--repository', required=True, help='GitHub owner/name')
    parser.add_argument('--key', required=True, type=Path)
    parser.add_argument('--known-hosts', required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',args.repository):
        raise ValueError('Expected GitHub owner/repository')
    root = Path(__file__).resolve().parents[1]
    state = root / '.unity-publisher'
    state.mkdir(exist_ok=True, mode=0o700)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg', MPLCONFIGDIR=str(state/'matplotlib'))
    env['GIT_SSH_COMMAND'] = ' '.join(['ssh', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
        '-o', 'StrictHostKeyChecking=yes', '-o', shlex.quote('UserKnownHostsFile='+str(args.known_hosts.resolve())),
        '-i', shlex.quote(str(args.key.resolve()))])
    with (state / 'publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        campaign = args.campaign.resolve()
        python = campaign / 'runtime/python-env/bin/python3'
        # A fresh disposable checkout prevents a failed render or rejected push
        # from leaving a dirty publication tree that blocks every later event.
        # Read and write remotes are deliberately derived from the SAME repository.
        for attempt in range(3):
            with tempfile.TemporaryDirectory(prefix='publish-',dir=state) as scratch:
                checkout=Path(scratch)/'repository'
                run(['git','clone','--single-branch','--branch','main',
                     f'https://github.com/{args.repository}.git',str(checkout)],root,env=env)
                run([str(python), str(root/'scripts/export_campaign.py'), '--campaign-root', str(campaign),
                     '--gallery', str(checkout/'gallery'), '--cache', str(state/'cache')],root,env=env)
                run([str(python),str(root/'scripts/check_gallery.py'),str(checkout/'gallery')],root,env=env)
                run(['git','add','--','gallery'],checkout)
                changed = subprocess.run(['git','diff','--cached','--quiet'],cwd=checkout).returncode
                if changed == 0:
                    print('Published gallery already matches this snapshot.',flush=True)
                    return
                if changed != 1: raise RuntimeError('Unable to inspect staged gallery changes')
                run(['git','-c','user.name=Wave-current gallery','-c','user.email=7075590+npizzo@users.noreply.github.com',
                     'commit','-m','Update completed Unity wave-packet results'],checkout)
                pushed=subprocess.run(['git','push',f'git@github.com:{args.repository}.git','HEAD:main'],cwd=checkout,env=env)
                if pushed.returncode==0:
                    print('Public gallery snapshot pushed. GitHub Pages deployment follows automatically.',flush=True)
                    return
                print(f'Push failed on attempt {attempt+1}; rebuilding against the current remote branch.',flush=True)
        raise RuntimeError('Publication failed after three fresh attempts. Inspect this job log; no force push was used.')


if __name__ == '__main__': main()

