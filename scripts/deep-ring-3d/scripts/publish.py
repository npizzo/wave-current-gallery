"""Publish the deep-ring 3D page (derived results + reproducibility source) through the shared gallery publisher."""
import os,sys,json,fcntl,subprocess,shutil,tempfile,shlex,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; PAGE='deep-ring-3d'
PUBLISHER=Path('/scratch3/workspace/nicholas_pizzo_uri_edu-regime-map-20261026/publisher/.unity-publisher')
SOURCE=['README.txt','PROVENANCE.sha256','PROVENANCE-notes.txt','GATE.txt','screen.cases','prod3d.cases','prod3d-rb1.cases','prod3d-rb1-none.cases','movie.cases',
 'run/ring3d.jl','run/ring3d.production.jl','run/regime_initial_conditions.jl','run/test_ring.jl','analysis/summarize.py','analysis/extract_movie.py',
 'analysis/figs34.py','analysis/diag.py','analysis/render_movie.py','analysis/build_page.py','analysis/page_body.html','scripts/case.sbatch','scripts/test.sbatch','scripts/publish.py','scripts/compress_regime_map.py',
 'analysis/long2d.py','analysis/rhines_composite.py','analysis/kx_collapse.py','analysis/vallis_style.py','analysis/ens_vallis.py','analysis/vallis_fig.py','analysis/none_check.py','analysis/snap35.py','analysis/build_spectral_page.py','long2d.cases','exp2d.cases','ens8.cases','lx4.cases','scripts/analysis.sbatch','environment/Project.toml','environment/Manifest.toml']
def main():
 page=ROOT/'gallery-page'; assert (page/'index.html').is_file()
 manifest={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCE}
 env=dict(os.environ);env['GIT_SSH_COMMAND']=' '.join(['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o',shlex.quote('UserKnownHostsFile='+str(PUBLISHER/'known_hosts')),'-i',shlex.quote(str(PUBLISHER/'id_ed25519'))])
 with (PUBLISHER/'publish.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for attempt in range(4):
   with tempfile.TemporaryDirectory(prefix='deepring-publish-',dir=PUBLISHER,ignore_cleanup_errors=True) as td:
    repo=Path(td)/'repo';subprocess.run(['git','clone','--quiet','--depth=1','--branch=main','https://github.com/npizzo/wave-current-gallery.git',str(repo)],check=True,env=env)
    # one-time space recovery approved by the PI (4 Nov 2026): gzip regime-map CSV downloads, separate commit
    subprocess.run([sys.executable,str(ROOT/'scripts/compress_regime_map.py'),str(repo)],check=True)
    subprocess.run(['git','add','--','gallery/experiments/regime-map'],check=True,cwd=repo)
    rm_changed=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=repo,text=True).splitlines()
    if not all(n.startswith('gallery/experiments/regime-map/') for n in rm_changed):raise RuntimeError('Unrelated path in regime-map compression')
    if rm_changed:subprocess.run(['git','-c','user.name=Wave-current gallery','-c','user.email=7075590+npizzo@users.noreply.github.com','commit','-m','Compress regime-map CSV downloads (gzip, identical content) to free gallery space'],check=True,cwd=repo,stdout=subprocess.DEVNULL)
    dest=repo/'gallery/experiments'/PAGE
    if dest.exists():shutil.rmtree(dest)
    shutil.copytree(page,dest)
    src=repo/'scripts'/PAGE
    if src.exists():shutil.rmtree(src)
    for n in SOURCE:
     t=src/n;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,t)
    (src/'source-manifest.json').write_text(json.dumps(manifest,indent=1)+'\n')
    landing=repo/'gallery/index.html';html=landing.read_text();href='experiments/'+PAGE+'/'
    if 'href="'+href+'"' not in html:
     anchor='<div class="nav-links">'
     if anchor not in html:raise RuntimeError('Landing navigation changed')
     html=html.replace(anchor,anchor+'<a href="'+href+'">Deep-water jets in 3D</a>',1);landing.write_text(html)
    subprocess.run([sys.executable,str(repo/'scripts/check_gallery.py'),str(repo/'gallery')],check=True)
    allowed=['gallery/experiments/'+PAGE,'scripts/'+PAGE,'gallery/index.html']
    subprocess.run(['git','add','--',*allowed],check=True,cwd=repo)
    changed=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=repo,text=True).splitlines()
    if not all(any(n==p or n.startswith(p+'/') for p in allowed) for n in changed):raise RuntimeError('Unrelated publication path')
    if changed or rm_changed:
     if changed:
      subprocess.run(['git','-c','user.name=Wave-current gallery','-c','user.email=7075590+npizzo@users.noreply.github.com','commit','-m','Deep-water jets in 3D CL: add 2D spectral-exclusion study'],check=True,cwd=repo,stdout=subprocess.DEVNULL)
     pushed=subprocess.run(['git','push','git@github.com:npizzo/wave-current-gallery.git','HEAD:main'],cwd=repo,env=env)
     if pushed.returncode:
      time.sleep(3);continue
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=repo).strip()
    (ROOT/'last-publication.json').write_text(json.dumps(dict(commit=commit,changed=len(changed),regime_map_compressed_files=len(rm_changed),url='https://npizzo.github.io/wave-current-gallery/experiments/'+PAGE+'/'),indent=1)+'\n');print('published',commit,len(changed),'files');return
  raise RuntimeError('Publication failed; simulations preserved')
if __name__=='__main__':main()
