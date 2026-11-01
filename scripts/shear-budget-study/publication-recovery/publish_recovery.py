import sys,shutil
from pathlib import Path
root=Path('/scratch3/workspace/nicholas_pizzo_uri_edu-shear-budget-study-20261101/unity-shear-budget-study')
sys.path.insert(0,str(root/'scripts'))
import publish as p
here=Path(__file__).parent
original=p.subprocess.run
def run(args,*a,**kw):
 if len(args)>1 and str(args[1])==str(root/'analysis/analyze_budget.py'):
  args=[args[0],str(here/'analyze_recovery.py'),*args[2:]]
 return original(args,*a,**kw)
p.subprocess.run=run
copy=p.copy_source
def copy_source(root,checkout,source):
 copy(root,checkout,source)
 shutil.copytree(here,checkout/'scripts/shear-budget-study/publication-recovery',ignore=shutil.ignore_patterns('__pycache__'))
p.copy_source=copy_source
sys.argv=[sys.argv[0],'--root',str(root)]
p.main()
