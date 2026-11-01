import sys,json,hashlib
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--root')+1]);out=Path(sys.argv[sys.argv.index('--output')+1])
sys.path.insert(0,str(root/'analysis'))
import analyze_budget as a
from check_moments import check_scalar_maps
a.check_scalar_maps=check_scalar_maps
a.main()
revision={'description':'Cancellation-aware scalar/map comparison: original relative tolerance plus128machine-epsilon times Cauchy-Schwarz product bound. No diagnostic values or physical tolerances changed.',
'original_simulation_source':'890c873d1ef47d15','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
for name in ['summary.json','provenance.json']:
 p=out/name;d=json.loads(p.read_text());d['analysis_revision']=revision;p.write_text(json.dumps(d,indent=2)+'\n')
p=out/'index.html';s=p.read_text().replace('<footer>','<footer><p>Analysis correction: scalar/map comparison includes a roundoff allowance based on the product norm when signed products cancel. The simulation and diagnostic values are unchanged. Details are recorded in provenance.json.</p>');p.write_text(s)
