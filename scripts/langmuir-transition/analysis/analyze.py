import sys,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ops import read,save,digest

def array(root,meta,check=True):
 p=root/meta['file']
 if check and digest(p)!=meta['sha256']:raise RuntimeError('Diagnostic checksum mismatch: '+str(p))
 return np.memmap(p,dtype=meta['dtype'],mode='r',shape=tuple(meta['shape']),order='F')
def dx(a,h):return (np.roll(a,-1,axis=0)-np.roll(a,1,axis=0))/(2*h)
def dz(a,z):return np.gradient(a,z,axis=-1,edge_order=2)
def terms(d,z,h,k,S):
 U,W=d['U'],d['W'];rxx=d['UU']-U*U;rzz=d['WW']-W*W;rxz=d['UW']-U*W
 Cu=-dx(rxx,h)-dz(rxz,z);Cw=-dx(rxz,h)-dz(rzz,z)
 R=dx(Cw,h)-dz(Cu,z);vort=dx(W,h)-dz(U,z)
 adv=U*dx(vort,h)+W*dz(vort,z);wave=-4*k*k*S*np.exp(2*k*z)[None,:]*W
 return dict(R=R,adv=adv,wave=wave,vort=vort,Cu=Cu,Q=-vort-2*k*S*np.exp(2*k*z)[None,:])
def avg(a,weights):return np.sum(np.mean(a,axis=0)*weights)/np.sum(weights)
def analyze_run(root,out):
 out.mkdir(parents=True,exist_ok=True);g=read(root/'grid.json');z=np.array(g['zc']);zf=np.array(g['zf']);weights=np.diff(zf);h=g['Lx']/g['nx'];S=g['S'] if g['member'].startswith('steady') else 0.;k=g['k'];paths=sorted(root.glob('mean_*.json'))
 rows=[];profiles=[];maps=[];initial=None;history=[]
 for path in paths:
  m=read(path);d={n:np.array(array(root,a),dtype=float).squeeze(axis=1) for n,a in m['arrays'].items()};U,V,W=(d[n] for n in ['U','V','W']);T=terms(d,z,h,k,S)
  profile=U.mean(axis=0);initial=profile.copy() if initial is None else initial
  # Native transverse-averaged continuity; no center interpolation in this check.
  div=(np.roll(d['Un'],-1,axis=0)-d['Un'])/h+np.diff(d['Wn'],axis=1)/weights
  totalvar=d['UU']+d['VV']+d['WW']-(profile[None,:]**2+V.mean(axis=0)[None,:]**2+W.mean(axis=0)[None,:]**2)
  transversevar=d['UU']+d['VV']+d['WW']-U**2-V**2-W**2
  Eeddy=.5*avg(totalvar,weights);E3d=.5*avg(transversevar,weights)
  active=(z>=-2/(2*k));active[:2]=False;active[-2:]=False;interior=np.ones(len(z),bool);interior[:2]=False;interior[-2:]=False
  norm=lambda a,mask:np.sqrt(avg(a[:,mask]**2,weights[mask]))
  rms={name:norm(T[name],active) for name in ['R','adv','wave']}
  stats=dict(time=m['time'],Eeddy=Eeddy,E3d=E3d,fraction3d=E3d/max(Eeddy,1e-30),w_rms=np.sqrt(max(avg(d['WW'],weights),0)),
    w_rms_active=np.sqrt(max(avg(d['WW'][:,active],weights[active]),0)),R_active=rms['R'],adv_active=rms['adv'],wave_active=rms['wave'],
    R_ratio_active=rms['R']/max(rms['adv']+rms['wave'],1e-30),continuity_rms=np.sqrt(avg(div**2,weights)),
    mean_current_change_rms=np.sqrt(np.sum((profile-initial)**2*weights)/sum(weights)),
    direct_R_current_work=avg((profile-np.sum(profile*weights)/sum(weights))[None,:]*T['Cu'],weights),cfl=m['cfl'])
  rows.append(stats);profiles.append(np.stack([profile,profile-S*np.exp(2*k*z),T['Q'].mean(axis=0),dz(T['Q'].mean(axis=0),z),d['WW'].mean(axis=0)]))
  # Keep only adjacent maps in memory when estimating continuum time derivatives.
  history.append((m['time'],T,stats))
  if len(history)==3:
   a,b,c=history;ht=np.array([a[0],b[0],c[0]]);der=np.gradient(np.stack([a[1]['vort'],b[1]['vort'],c[1]['vort']]),ht,axis=0,edge_order=2)[1]
   residual=der+b[1]['adv']-b[1]['wave']-b[1]['R'];b[2]['continuum_residual_active']=norm(residual,active);b[2]['vorticity_t_active']=norm(der,active);history.pop(0)
  if any(abs(m['time']-t)<1e-6 for t in [0,10,60,600]):maps.append((m['time'],U-S*np.exp(2*k*z)[None,:]))
 t=np.array([x['time'] for x in rows]);p=np.array(profiles)
 np.savetxt(out/'profiles.csv',np.column_stack([np.repeat(t,len(z)),np.tile(z,len(t)),*[p[:,j,:].reshape(-1) for j in range(5)]]),delimiter=',',header='time_s,z_m,UL_m_s,UE_m_s,Q_s_inv,beta_m_inv_s_inv,wvariance_m2_s2',comments='')
 save(out/'diagnostics.json',rows)
 fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
 for target in [0,10,30,60,200,600]:
  i=np.argmin(abs(t-target))
  if abs(t[i]-target)>1e-6:continue
  for ax,j,scale in zip(axs,[1,0,2],[1000,1000,1]):ax.plot(p[i,j]*scale,z,label=f'{target:g} s')
 for ax,title in zip(axs,['Eulerian current (mm/s)','Lagrangian current (mm/s)','Mean Eulerian shear Q (1/s)']):ax.set(xlabel=title,ylabel='z (m)');ax.grid(alpha=.2)
 axs[0].legend();fig.savefig(out/'profiles.png',dpi=150);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 for ax,j,title in zip(axs,[1,0],['Eulerian current','Lagrangian current']):
  im=ax.pcolormesh(t,z,1000*p[:,j].T,shading='auto',cmap='RdBu_r');fig.colorbar(im,ax=ax,label='mm/s');ax.set(xlabel='Time (s)',ylabel='z (m)',title=title)
 fig.savefig(out/'time-depth.png',dpi=150);plt.close(fig)
 fig,axs=plt.subplots(2,len(maps),figsize=(3.5*len(maps),6),squeeze=False,layout='constrained');umax=max(np.max(abs(x[1])) for x in maps)*1000
 for i,(tm,U) in enumerate(maps):
  im=axs[0,i].pcolormesh(np.linspace(0,g['Lx'],g['nx']),z,U.T*1000,shading='auto',cmap='RdBu_r',vmin=-umax,vmax=umax);axs[0,i].set(title=f't={tm:g} s',xlabel='x (m)',ylabel='z (m)')
  nf=root/f'native_{tm:010.3f}.json'
  if nf.exists():
   n=read(nf);w=array(root,n['arrays']['w']);sl=np.array(w[g['nx']//2,:,:]);im2=axs[1,i].pcolormesh(np.linspace(0,g['Ly'],g['ny']),zf,sl.T*1000,cmap='RdBu_r',shading='auto',vmin=-50,vmax=50);axs[1,i].set(xlabel='y (m)',ylabel='z (m)')
 fig.colorbar(im,ax=axs[0,:],label='Transverse-mean Eulerian u (mm/s)');fig.colorbar(im2,ax=axs[1,:],label='Vertical velocity at x=Lx/2 (mm/s)');fig.savefig(out/'evolution.png',dpi=130);plt.close(fig)
 summary=dict(final=rows[-1],complete=(root/'completed.json').exists(),scope='Single realization; initial shear already present. New jets require persistent new extrema and separation from inherited shear. R is a continuum postprocessing diagnostic, not an exact native tendency split.')
 save(out/'summary.json',summary);return summary

def main(root,out):
 out.mkdir(parents=True,exist_ok=True);campaign=read(root/'campaign.json');runs=[]
 for c in campaign['runs']:
  r=root/'data/runs'/c['id'];entry=dict(c)
  if (r/'completed.json').exists():
   if not (r/'backup.json').exists():raise RuntimeError('Final state not backed up')
   receipt=read(r/'completed.json');assert receipt['source']==read(root/'source-manifest.json')['version']
   entry['result']=analyze_run(r,out/c['id'])
  elif (r/'status.json').exists():entry['status']=read(r/'status.json')
  else:entry['status']='queued or not yet started'
  runs.append(entry)
 fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained');count=0
 for r in runs:
  if 'result' not in r:continue
  count+=1;d=read(out/r['id']/'diagnostics.json');t=[x['time'] for x in d]
  for ax,key in zip(axs.flat,['w_rms_active','mean_current_change_rms','fraction3d','R_ratio_active']):ax.plot(t,[x[key] for x in d],label=r['id'])
 for ax,title in zip(axs.flat,['Vertical rms in upper two Stokes depths (m/s)','Change from initial current profile: rms (m/s)','y-dependent / total eddy energy','R / (mean advection + wave term): rms norms']):ax.set(xlabel='Time (s)',title=title);ax.grid(alpha=.2)
 if count:axs[0,0].legend();fig.savefig(out/'comparison.png',dpi=150)
 plt.close(fig)
 summary=dict(runs=runs,completed=count,total=4,source=read(root/'source-manifest.json')['version'],scope='Candidate screen, not a convergence or mechanism proof.');save(out/'summary.json',summary)
 sections=''
 for r in runs:
  sections+=f'<section><h2>{r["id"]}</h2>'
  if 'result'in r:
   sections+=''.join(f'<img src="{r["id"]}/{name}.png" alt="{r["id"]}: {name}">' for name in ['evolution','time-depth','profiles'])+f'<p><a href="{r["id"]}/diagnostics.json">Diagnostics</a> · <a href="{r["id"]}/profiles.csv">Profiles</a></p>'
  else:sections+=f'<p>{r["status"]}</p>'
  sections+='</section>'
 comparison='<img src="comparison.png" alt="Matched wave and no-wave diagnostics">' if count else ''
 (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Langmuir turbulence to jets?</title><style>body{{font:17px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:20px;color:#173047;background:#f6f8fa}}img{{width:100%;background:white;margin:12px 0}}section{{margin:50px 0}}a{{color:#135da8}}</style><a href="../../">Wave–current gallery</a><h1>Can deep-water Langmuir turbulence evolve into jets?</h1><p>{count}/4 runs completed. Two initial Eulerian shears, α=1 and2, each with steady waves and a matched wave-free control. Original12×0.8×0.4m periodic tank;1536×128×128 grid;600s;single seed. No sustained wind or imposed two-dimensionalization.</p><p>The initial smooth shear is already a current. A later mean profile or small3D energy fraction alone does not demonstrate newly formed jets. We test whether persistent new current extrema arise after the Langmuir phase, while following the evolving Eulerian shear and complete transverse Reynolds-stress curl.</p><p>R uses continuum derivatives of centered transverse moments. Temporal sampling, interpolation and implicit numerical dissipation contribute to its budget residual; this is not an exact native tendency split or a convergence demonstration.</p><p><a href="PROTOCOL.txt">Protocol</a> · <a href="summary.json">Status and summary</a> · <a href="source-manifest.json">Source provenance</a></p>{comparison}{sections}</html>''')
if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
