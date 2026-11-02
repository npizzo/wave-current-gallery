import os,json,hashlib,subprocess,sys,fcntl,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def science():
 return {str(p.relative_to(ROOT)):digest(p) for d in ['run','upstream'] for p in sorted((ROOT/d).rglob('*')) if p.is_file()}
def verify():
 m=read(ROOT/'source-manifest.json')
 for n,h in m['files'].items():
  if digest(ROOT/n)!=h:raise RuntimeError('Source changed: '+n)
 return m
def call(cmd):
 print('RUN '+' '.join(map(str,cmd)),flush=True);subprocess.run(list(map(str,cmd)),check=True,cwd=ROOT)
def julia(script,*args):return [os.environ['JULIA'],'--startup-file=no','--project='+str(ROOT/'upstream'),str(ROOT/script),*map(str,args)]
def validate():
 before=science();save(ROOT/'validation/started.json',dict(job=os.getenv('SLURM_JOB_ID'),science=before))
 if (ROOT/'validation/tests').exists():raise RuntimeError('Preserving previous validation attempt; inspect before retry')
 call(julia('run/validate.jl',ROOT/'validation/tests'))
 if science()!=before:raise RuntimeError('Source changed in validation')
 report=read(ROOT/'validation/tests/passed.json');assert report['passed']
 report['science']=before;save(ROOT/'validation/passed.json',report)
def batch(script,extra=()):
 # Write intent BEFORE asking Slurm. A missing response is not safe to resubmit.
 journal=ROOT/'submission-journal.json';items=read(journal) if journal.exists() else []
 token=f'langmuir-transition-{int(time.time()*1e6)}';item=dict(token=token,script=script,extra=list(extra),outcome='unknown');items.append(item);save(journal,items)
 p=subprocess.run(['sbatch','--parsable','--comment='+token,*extra,str(ROOT/'scripts'/script)],cwd=ROOT,text=True,capture_output=True)
 if p.returncode:raise RuntimeError('Slurm submission failed; inspect journal and scheduler before retry: '+p.stderr)
 job=p.stdout.strip().split(';')[0]
 if not job.isdigit():raise RuntimeError('Ambiguous Slurm response: '+p.stdout)
 item.update(outcome='submitted',job=job);save(journal,items);return job

def submit():
 verify();v=read(ROOT/'validation/passed.json');assert v['passed'] and v['science']==science()
 if digest(v['ic'])!=v['ic_sha256']:raise RuntimeError('IC changed')
 path=ROOT/'submission.json'
 if path.exists():print('Already submitted:',path.read_text());return
 if (ROOT/'submission-journal.json').exists():raise RuntimeError('Submission journal already exists; inspect Slurm before any retry')
 a=batch('run.sbatch',['--array=0-3%2']);s=dict(array=a,source=verify()['version'],publishers=[],validation=os.getenv('SLURM_JOB_ID'));save(path,s)
 for index in range(4):
  p=batch('publish.sbatch',['--dependency=afterany:'+a+'_'+str(index)]);s['publishers'].append(dict(job=p,after=a+'_'+str(index)));save(path,s)
 p=batch('publish.sbatch');s['publishers'].append(dict(job=p,after='immediate'));save(path,s)
 print(json.dumps(s,indent=2))
def launch(index):
 source=verify();v=read(ROOT/'validation/passed.json');assert v['science']==science() and v['passed']
 s=read(ROOT/'submission.json');assert os.environ['SLURM_ARRAY_JOB_ID']==s['array']
 if digest(v['ic'])!=v['ic_sha256']:raise RuntimeError('IC changed')
 case=read(ROOT/'campaign.json')['runs'][index];out=ROOT/'data/runs'/case['id']
 if out.exists():raise RuntimeError('Refusing to overwrite '+str(out))
 out.mkdir(parents=True);save(out/'started.json',dict(case=case,job=os.environ['SLURM_JOB_ID'],source=source['version'],ic_sha256=v['ic_sha256']))
 call(julia('run/driver.jl',out,ROOT/'data/ic',case['member'],case['alpha'],case['id']))
 r=read(out/'integrated.json');assert abs(r['time']-600)<1e-7
 verify();save(out/'completed.json',dict(completed=True,case=case,source=source['version'],ic_sha256=v['ic_sha256'],integration=r))
 # Native final state is persistent; intermediate raw fields remain in30-day scratch.
 backup=Path(read(ROOT/'deployment.json')['backup'])/'runs'/case['id'];backup.mkdir(parents=True,exist_ok=True)
 final=read(out/'native_000600.000.json')
 for n in ['completed.json','grid.json','native_000600.000.json',*[x['file'] for x in final['arrays'].values()]]:
  shutil.copy2(out/n,backup/n)
  if digest(out/n)!=digest(backup/n):raise RuntimeError('Backup mismatch')
 save(out/'backup.json',dict(verified=True,path=str(backup)))
if __name__=='__main__':
 action=sys.argv[1]
 with (ROOT/('.'+action+'.lock')).open('a') as f:
  if action!='run':fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if action=='validate':validate()
  elif action=='submit':submit()
  elif action=='run':launch(int(sys.argv[2]))
