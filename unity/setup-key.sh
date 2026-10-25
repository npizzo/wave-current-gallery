#!/usr/bin/env bash
# Run on Unity inside this separate gallery checkout.
set -euo pipefail
gallery_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
state="$gallery_root/.unity-publisher"
mkdir -p "$state/logs"
chmod 700 "$state"
if [[ ! -f "$state/id_ed25519" ]]; then
  ssh-keygen -q -t ed25519 -N '' -C 'Unity wave-current-gallery publisher' -f "$state/id_ed25519"
fi
chmod 600 "$state/id_ed25519"
# Authenticate GitHub's host keys through its official HTTPS metadata endpoint.
python3 - "$state/known_hosts" <<'PY'
import json,sys,urllib.request
from pathlib import Path
req=urllib.request.Request('https://api.github.com/meta',headers={'User-Agent':'wave-current-gallery-setup'})
with urllib.request.urlopen(req,timeout=30) as response: keys=json.load(response)['ssh_keys']
assert keys and all(k.startswith(('ssh-ed25519 ','ecdsa-sha2-nistp256 ','ssh-rsa ')) and '\n' not in k for k in keys)
Path(sys.argv[1]).write_text(''.join('github.com '+key+'\n' for key in keys))
PY
echo 'Add this PUBLIC key as a write-enabled deploy key on the wave-current-gallery repository:'
cat "$state/id_ed25519.pub"
echo 'The private key stays in this Unity checkout and is excluded from Git.'

