"""Split explicitly approved public datasets into independently readable CSV parts.
Rows are preserved exactly; this avoids oversized inline upload review requests.
"""
from pathlib import Path
import gzip,csv,io,json,hashlib,re
from collections import Counter
R=Path('outputs/deep-water-long-analysis');P=R/'gallery';parts=P/'data';parts.mkdir(exist_ok=True);manifest=[]
for filename in ['upper-quarter-spectra.csv.gz','evolution.csv.gz']:
 raw=gzip.decompress((P/filename).read_bytes());lines=raw.decode().splitlines(keepends=True);header=lines[0];columns=next(csv.reader([header]));data=lines[1:];groups={}
 for line in data:
  row=dict(zip(columns,next(csv.reader([line]))))
  if filename.startswith('upper'):
   group='spectra-'+row['stage_Te']+'Te'
  else:
   t=int(row['t_Te']);lo=min(t//125,7)*125;hi=lo+124 if lo<875 else 1000;group=f'evolution-{lo:04d}-{hi:04d}Te'
  groups.setdefault(group,[]).append(line)
 assert Counter(x for v in groups.values() for x in v)==Counter(data)
 entry={'original_filename':filename,'original_csv_sha256':hashlib.sha256(raw).hexdigest(),'rows':len(data),'parts':[]}
 for group,rows in groups.items():
  body=(header+''.join(rows)).encode();b=gzip.compress(body,compresslevel=9,mtime=0);path=parts/(group+'.csv.gz');path.write_bytes(b);entry['parts'].append({'path':'data/'+path.name,'rows':len(rows),'bytes':len(b),'csv_sha256':hashlib.sha256(body).hexdigest()})
 manifest.append(entry)
(P/'data-parts.json').write_text(json.dumps(manifest,indent=2)+'\n')
p=P/'index.html';s=p.read_text();s=re.sub(r'<li><a href="(?:upper-quarter-spectra|evolution)\.csv\.gz">.*?</a></li>','',s)
s=s.replace('<h3>Downloads and source</h3>','<h3>Downloads and source</h3><p><a href="#full-data">Full spectral arrays and evolution records</a> are supplied below as smaller CSV parts; every row of both approved datasets is included.</p>')
blocks='<details id="full-data"><summary>Full spectral arrays and evolution records</summary><p>Each gzip file contains an ordinary CSV header and its labeled subset of rows. Splitting changes neither values nor precision. <a href="data-parts.json">Part manifest, row counts and checksums</a>.</p>'
for d in manifest:
 blocks+='<h4>'+('Upper-quarter spectra by stage' if d['original_filename'].startswith('upper') else 'Evolution by time interval; both cases in each part')+'</h4><ul>'
 for part in d['parts']:blocks+=f'<li><a href="{part["path"]}">{Path(part["path"]).name}</a> ({part["rows"]:,} rows)</li>'
 blocks+='</ul>'
blocks+='</details>'
s=s.replace('<p class="note">Spectral CSV energies',blocks+'<p class="note">Spectral CSV energies');p.write_text(s)
p=R/'source/README.txt';s=p.read_text();s+='\nThe approved full spectral and evolution datasets are published as independently readable CSV.gz parts, with row counts and checksums in data-parts.json. Run split_public_data.py after build_page.py to produce these parts and their download links. All CSV row values and precision are preserved.\n';p.write_text(s)
(R/'source/split_public_data.py').write_text(Path(__file__).read_text())
p=R/'upload-files.json';files=json.loads(p.read_text());files=[x for x in files if Path(x['path']).name not in ['upper-quarter-spectra.csv.gz','evolution.csv.gz']]
for path in [*parts.glob('*.csv.gz'),P/'data-parts.json',R/'source/split_public_data.py']:
 target=('scripts/deep-water-long/'+path.name) if path.parent.name=='source' else 'gallery/experiments/deep-ring-3d/spectral-exclusion/deep-water-long/'+str(path.relative_to(P));files=[x for x in files if x['path']!=target];files.append(dict(local=str(path.resolve()),path=target))
for x in files:
 b=Path(x['local']).read_bytes();x.update(bytes=len(b),git_sha=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest())
p.write_text(json.dumps(files,indent=2)+'\n');print(json.dumps(manifest,indent=2))
