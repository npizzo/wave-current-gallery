from pathlib import Path
import gzip,csv,io,json,hashlib,shutil
from PIL import Image
I=Path('outputs/deep-water-mechanism-pilot100-20261106/analysis');P=I/'gallery';private=I/'full-spectral-exports';private.mkdir(exist_ok=True)
# Preserve all original rows in logically labeled smaller per-window/time files.
for p in list((P/'data').glob('*-spectra.csv.gz')):
 rows=list(csv.reader(io.StringIO(gzip.decompress(p.read_bytes()).decode())));head=rows[0];groups={}
 for r in rows[1:]:groups.setdefault(tuple(r[:3]),[]).append(r)
 dest=P/'data'/p.name.replace('-spectra.csv.gz','-spectra');dest.mkdir(exist_ok=True)
 for key,rr in groups.items():
  text=io.StringIO();w=csv.writer(text);w.writerow(head);w.writerows(rr);data=gzip.compress(text.getvalue().encode(),mtime=0)
  (dest/('_'.join(key)+'.csv.gz')).write_bytes(data)
 shutil.move(p,private/p.name)
# Export lightweight web assets; keep original high-resolution PNGs/PDFs locally.
original=I/'original-figure-exports';original.mkdir(exist_ok=True)
for p in P.glob('*.webp'):
 if p.stat().st_size>170000:
  shutil.copy2(p,original/p.name)
  im=Image.open(p);im.thumbnail((1400,1400));im.save(p,quality=82)
for p in P.glob('*.pdf'):
 if p.stat().st_size>170000:
  shutil.copy2(p,original/p.name);im=Image.open(P/(p.stem+'.png')).convert('RGB');im.thumbnail((1400,1400));im.save(p,'PDF',resolution=150,quality=80)
# All per-window parts are individually reproducible; page gives grouped links through an index.
rows=[]
for p in sorted((P/'data').rglob('*')):
 if p.is_file():rows.append({'path':str(p.relative_to(P)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(P/'data-manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
h='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mechanism-pilot data</title><style>body{font:16px/1.6 system-ui;max-width:950px;margin:auto;padding:30px}a{color:#175a98}li{margin:5px}</style><a href="../">← Mechanism assessment</a><h1>Numerical data</h1><p>Spectral files are split by case, vertical window, estimator and snapshot/time average. Every row of the exported crops is retained. No energies are normalized, clipped or smoothed. Coordinates are cycles per original length. <a href="../data-manifest.json">File checksums</a>.</p>'
for case in ['deep-linear-s1','deep-ql-s1','deep-full-s1','nowave-linear-s1','nowave-ql-s1','nowave-full-s1']:
 h+='<h2>'+case+'</h2><ul>'
 for r in rows:
  if Path(r['path']).name.startswith(case) or '/'+case+'-spectra/' in r['path']:h+=f'<li><a href="{r["path"][5:]}">{r["path"][5:]}</a> ({r["bytes"]/1024:.0f} KB)</li>'
 h+='</ul>'
h+='</html>';(P/'data/index.html').write_text(h)
print('Largest packaged files',sorted([(p.stat().st_size,str(p.relative_to(P))) for p in P.rglob('*') if p.is_file() and p.suffix!='.png'],reverse=True)[:8])
