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
 campaign=read(ROOT/'campaign.json');case=campaign['runs'][index];job=os.environ['SLURM_ARRAY_JOB_ID']
 for attempt in range(30):
  # A retry record is written immediately after sbatch returns; allow for a fast start.
  s=read(ROOT/'submission.json');t=[x for x in s.get('retries',[]) if x['array']==job]
  if job==s['array'] or t:break
  time.sleep(10)
 if job==s['array']:dt=campaign['dt']
 else:
  assert len(t)==1 and case['id'] in t[0]['runs'],'Array '+job+' is not an authorized retry of '+case['id']
  dt=t[0]['dt'][case['id']]
 if digest(v['ic'])!=v['ic_sha256']:raise RuntimeError('IC changed')
 out=ROOT/'data/runs'/case['id']
 if out.exists():raise RuntimeError('Refusing to overwrite '+str(out))
 out.mkdir(parents=True);save(out/'started.json',dict(case=case,job=os.environ['SLURM_JOB_ID'],array=job,dt=dt,source=source['version'],ic_sha256=v['ic_sha256']))
 call(julia('run/driver.jl',out,ROOT/'data/ic',case['member'],case['alpha'],case['id'],dt))
 r=read(out/'integrated.json');assert abs(r['time']-600)<1e-7
 verify();save(out/'completed.json',dict(completed=True,case=case,dt=dt,array=job,source=source['version'],ic_sha256=v['ic_sha256'],integration=r))
 # Native final state is persistent; intermediate raw fields remain in30-day scratch.
 backup=Path(read(ROOT/'deployment.json')['backup'])/'runs'/case['id'];backup.mkdir(parents=True,exist_ok=True)
 final=read(out/'native_000600.000.json')
 for n in ['completed.json','grid.json','native_000600.000.json',*[x['file'] for x in final['arrays'].values()]]:
  shutil.copy2(out/n,backup/n)
  if digest(out/n)!=digest(backup/n):raise RuntimeError('Backup mismatch')
 save(out/'backup.json',dict(verified=True,path=str(backup)))
def retry_plan():
 c=read(ROOT/'campaign.json');r=c.get('retry')
 if not r:raise RuntimeError('No authorized retry in campaign.json')
 ids=[x['id'] for x in c['runs']];return r,[ids.index(n) for n in r['runs']]
def placement(r):return ['--exclude='+r['exclude_nodes']]+(['--constraint='+r['constraint']] if r.get('constraint') else [])+(['--partition='+r['partition']] if r.get('partition') else [])
def amend():
 # Revalidate the amended source, then retry only the failed members (see PROTOCOL amendment).
 verify();r,idx=retry_plan();path=ROOT/'amendment.json';val=ROOT/'validation'
 busy=subprocess.run(['squeue','-h','-u',os.environ['USER'],'-n','lj-transition,lj-validate,lj-dispatch,lj-retry'],text=True,capture_output=True,check=True).stdout.strip()
 if busy:raise RuntimeError('Campaign jobs still queued or running:\n'+busy)
 if path.exists():
  # Only a FAILED amendment validation may be superseded; its attempt is preserved, never deleted.
  prev=read(path)
  if read(ROOT/'submission.json').get('retries'):print('Retry already submitted');return
  state=subprocess.run(['sacct','-X','-n','-P','-j',prev['validation'],'-o','State'],text=True,capture_output=True,check=True).stdout.split()
  if not state or not all(x.startswith(('FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY')) for x in state):raise RuntimeError('Previous amendment validation not failed: '+repr(state))
  old=prev['validation'];ic=prev['ic_sha256'];(ROOT/'amendments').mkdir(exist_ok=True)
  moves=[(path,ROOT/'amendments'/('amendment-'+old+'.json')),(val/'tests',val/('attempt-'+old)),(val/'started.json',val/('started-'+old+'.json'))]
 else:
  old=read(val/'started.json')['job'];ic=read(val/'passed.json')['ic_sha256']
  moves=[(val/x,val/y) for x,y in [('tests','attempt-'+old),('passed.json','passed-'+old+'.json'),('started.json','started-'+old+'.json')]]
 for x,y in moves:
  if y.exists():raise RuntimeError('Refusing to overwrite '+str(y))
 for x,y in moves:
  if x.exists():x.rename(y)
 a=dict(source=verify()['version'],prior_validation=old,ic_sha256=ic,placement=placement(r));save(path,a)
 a['validation']=batch('validate.sbatch',placement(r));save(path,a)
 a['dispatch']=batch('retry.sbatch',['--dependency=afterok:'+a['validation']]);save(path,a)
 print(json.dumps(a,indent=2))
def retry():
 verify();v=read(ROOT/'validation/passed.json');assert v['passed'] and v['science']==science()
 if digest(v['ic'])!=v['ic_sha256']:raise RuntimeError('IC changed')
 a=read(ROOT/'amendment.json')
 if v['ic_sha256']!=a['ic_sha256']:raise RuntimeError('Revalidated IC differs from the production IC')
 path=ROOT/'submission.json';s=read(path)
 if s.get('retries'):print('Retry already submitted:',json.dumps(s['retries']));return
 r,idx=retry_plan()
 for n in r['runs']:
  out=ROOT/'data/runs'/n
  if (out/'completed.json').exists():raise RuntimeError('Refusing to retry completed run '+n)
  if out.exists():
   dst=ROOT/'data/failed'/(n+'-array-'+s['array'])
   if dst.exists():raise RuntimeError('Refusing to overwrite '+str(dst))
   dst.parent.mkdir(parents=True,exist_ok=True);out.rename(dst)
 x=batch('run.sbatch',['--array='+','.join(map(str,idx))+'%2',*placement(r)])
 t=dict(array=x,source=verify()['version'],runs=r['runs'],dt=r['dt'],validation=a['validation'],publishers=[])
 s.setdefault('retries',[]).append(t);save(path,s)
 for i in idx:
  p=batch('publish.sbatch',['--dependency=afterany:'+x+'_'+str(i)]);t['publishers'].append(dict(job=p,after=x+'_'+str(i)));save(path,s)
 p=batch('publish.sbatch');t['publishers'].append(dict(job=p,after='immediate'));save(path,s)
 print(json.dumps(t,indent=2))
if __name__=='__main__':
 action=sys.argv[1]
 with (ROOT/('.'+action+'.lock')).open('a') as f:
  if action!='run':fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if action=='validate':validate()
  elif action=='submit':submit()
  elif action=='run':launch(int(sys.argv[2]))
  elif action=='amend':amend()
  elif action=='retry':retry()
