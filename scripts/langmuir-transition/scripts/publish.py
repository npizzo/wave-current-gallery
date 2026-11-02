import os,sys,json,fcntl,subprocess,shutil,tempfile,shlex,time
from pathlib import Path
from ops import ROOT,read,save,verify,digest
sys.path.insert(0,str(ROOT/'analysis'))
from analyze import main as render
PAGE='langmuir-transition'
def main():
 source=verify();paths=read(ROOT/'deployment.json');publisher=Path(paths['publisher'])/'.unity-publisher'
 env=dict(os.environ);env['GIT_SSH_COMMAND']=' '.join(['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o',shlex.quote('UserKnownHostsFile='+str(publisher/'known_hosts')),'-i',shlex.quote(str(publisher/'id_ed25519'))])
 with (publisher/'publish.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  with tempfile.TemporaryDirectory(prefix='langmuir-render-',dir=ROOT/'runtime') as tmp:
   page=Path(tmp)/'page';render(ROOT,page)
   # Fit this campaign within the existing gallery allowance; never alter other pages.
   if sum(p.stat().st_size for p in page.rglob('*') if p.is_file())>3*1024**2:
    from PIL import Image
    for p in page.rglob('*.webp'):
     with Image.open(p) as im:
      copy=im.copy();copy.thumbnail((1300,1300));copy.save(p,quality=62,method=6)
   for n in ['PROTOCOL.txt','campaign.json','source-manifest.json']:shutil.copy2(ROOT/n,page/n)
   backup=Path(paths['backup']);backup.mkdir(parents=True,exist_ok=True)
   shutil.copytree(page,backup/'gallery',dirs_exist_ok=True)
   for attempt in range(4):
    with tempfile.TemporaryDirectory(prefix='langmuir-publish-',dir=publisher) as td:
     repo=Path(td)/'repo';subprocess.run(['git','clone','--quiet','--depth=1','--branch=main','https://github.com/npizzo/wave-current-gallery.git',str(repo)],check=True,env=env)
     dest=repo/'gallery/experiments'/PAGE
     if dest.exists():shutil.rmtree(dest)
     shutil.copytree(page,dest)
     src=repo/'scripts'/PAGE
     if src.exists():shutil.rmtree(src)
     for name in source['files']:
      target=src/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
     shutil.copy2(ROOT/'source-manifest.json',src/'source-manifest.json')
     landing=repo/'gallery/index.html';html=landing.read_text();href='experiments/'+PAGE+'/'
     if 'href="'+href+'"' not in html:
      anchor='<div class="nav-links">'
      if anchor not in html:raise RuntimeError('Landing navigation changed')
      html=html.replace(anchor,anchor+'<a href="'+href+'">Langmuir to jets</a>',1);landing.write_text(html)
     subprocess.run([sys.executable,str(repo/'scripts/check_gallery.py'),str(repo/'gallery')],check=True)
     verify();allowed=['gallery/experiments/'+PAGE,'scripts/'+PAGE,'gallery/index.html']
     subprocess.run(['git','add','--',*allowed],check=True,cwd=repo)
     changed=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=repo,text=True).splitlines()
     if not all(any(n==p or n.startswith(p+'/') for p in allowed) for n in changed):raise RuntimeError('Unrelated publication path')
     if changed:
      subprocess.run(['git','-c','user.name=Wave-current gallery','-c','user.email=7075590+npizzo@users.noreply.github.com','commit','-m','Update deep-water Langmuir-to-jet experiment'],check=True,cwd=repo,stdout=subprocess.DEVNULL)
      pushed=subprocess.run(['git','push','git@github.com:npizzo/wave-current-gallery.git','HEAD:main'],cwd=repo,env=env)
      if pushed.returncode:
       subprocess.run(['git','fetch','--quiet','origin','main'],check=True,cwd=repo)
       if subprocess.run(['git','merge-base','--is-ancestor','HEAD','FETCH_HEAD'],cwd=repo).returncode:time.sleep(2);continue
     commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=repo).strip();save(ROOT/'last-publication.json',dict(commit=commit,source=source['version'],completed=read(page/'summary.json')['completed'],url='https://npizzo.github.io/wave-current-gallery/experiments/'+PAGE+'/'));return
   raise RuntimeError('Publication failed; simulations preserved')
if __name__=='__main__':main()
