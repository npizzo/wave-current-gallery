import json, os, sys, shutil, gzip, glob, html
from PIL import Image
H=os.path.expanduser('~'); FIG=f'{H}/mnt/i-want-reorganize-this-text-for/outputs/deep-ring-3d'; SRC=f'{H}/drpage/src'; P=f'{H}/drpage/page'
shutil.rmtree(P,ignore_errors=True); os.makedirs(P+'/runs')
def webp(name,out,w=1600):
    with Image.open(f'{FIG}/{name}') as im:
        im=im.convert('RGB'); im.thumbnail((w,w)); im.save(f'{P}/{out}',quality=82,method=6)
webp('fig3-deep-ring-3d.png','fig3.webp'); webp('fig4-deep-ring-3d.png','fig4.webp',1300); webp('diagnostics.png','diagnostics.webp',1400)
shutil.copy(f'{FIG}/deep-ring-movie.mp4',f'{P}/movie.mp4'); webp('deep-ring-movie-last.png','movie-poster.webp',1920)
for n in ['GATE.txt','README.txt','PROVENANCE-notes.txt']: shutil.copy(f'{SRC}/{n}',f'{P}/{n}')
shutil.copy(f'{SRC}/PROVENANCE.sha256',f'{P}/PROVENANCE-sha256.txt')
S=json.load(open(f'{H}/drfig/summary-all.json')); json.dump(dict(runs=S,movie=json.load(open(f'{H}/drfig/movie-summary.json'))),open(f'{P}/summary.json','w'),indent=1)
for d in sorted(glob.glob(f'{H}/drfig/d3-*')+glob.glob(f'{H}/drfig/s2d-*')):
    with open(f'{d}/timeseries.csv','rb') as f, gzip.open(f'{P}/runs/{os.path.basename(d)}.timeseries.csv.gz','wb') as g: g.write(f.read())
for r in ['d3-deep-rb1-e1e-2-s1','d3-none-rb1-e1e-2-s1','s2d-deep-rb1-s1','s2d-none-rb1-s1']:
    with open(f'{H}/drfig/{r}/profiles.csv','rb') as f, gzip.open(f'{P}/runs/{r}.profiles.csv.gz','wb') as g: g.write(f.read())
def row(r):
    f=r['final']; return '<tr><td>%s</td><td>%s</td><td>%g</td><td>%d</td><td>%s</td><td>%.3f</td><td>%.2f</td><td>%.3f</td><td>%.3f</td><td>%.2f</td></tr>'%(r['name'],'3D' if r['ny']>1 else '2D','%g'%r['rb'] and r['rb'],r['seed'],'%g'%r['eps3d'] if r['ny']>1 else '–',f['jet_frac'],f['anisotropy'],f['frac3d_upper'],f['frac3d_lower'],f['maxUL_over_u0'])
S.sort(key=lambda r:(r['rb'],r['ny']==1,'none' in r['name'],r['eps3d'],r['seed']))
table='<table><tr><th>run</th><th>dim</th><th>r<sub>b</sub></th><th>seed</th><th>ε<sub>3D</sub></th><th>jet fraction</th><th>anisotropy</th><th>3D frac, z&gt;−0.3</th><th>3D frac, z&lt;−0.3</th><th>max|U<sub>L</sub>|/u<sub>0</sub></th></tr>'+''.join(row(r) for r in S)+'</table>'
body=open(f'{H}/drpage/body.html').read().replace('{{TABLE}}',table)
css='body{font:17px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:20px;color:#173047;background:#f6f8fa}img,video{width:100%;background:white;margin:12px 0}section{margin:50px 0}a{color:#135da8}table{border-collapse:collapse;font-size:14px;width:100%;background:white}td,th{border-bottom:1px solid #d8dee4;padding:4px 8px;text-align:right}td:first-child,th:first-child{text-align:left}.cap{font-size:15px;color:#3b5368}'
open(f'{P}/index.html','w').write('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Deep-water jets in 3D CL</title><style>'+css+'</style>'+body)
tot=sum(os.path.getsize(os.path.join(dp,f)) for dp,_,fs in os.walk(P) for f in fs); print('page MB',tot/1e6)
