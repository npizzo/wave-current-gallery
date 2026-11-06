from pathlib import Path
import json,csv,gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
I=Path('outputs/deep-water-mechanism-pilot100-20261106/analysis');O=I/'gallery'
plt.rcParams.update({'font.size':10,'savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
 fig.savefig(O/(name+'.png'),dpi=170,bbox_inches='tight');fig.savefig(O/(name+'.pdf'),bbox_inches='tight');im=Image.open(O/(name+'.png'));im.save(O/(name+'.webp'),quality=87);plt.close(fig)
D={p:np.load(I/(p+'-full-s1.npz')) for p in ['deep','nowave']}
for tt in [25,100]:
 fig,ax=plt.subplots(2,2,figsize=(9,7),gridspec_kw={'height_ratios':[.6,1]},layout='constrained');bins=np.arange(-34,35,4);centers=(bins[:-1]+bins[1:])/2;es=[]
 for p in ['deep','nowave']:
  a=D[p];kx=a['top025_hann_kx'];e=a[f'top025_hann_E_{tt}'].copy();e[:,kx==0]=0
  b=np.stack([e[:,(kx>=lo)&(kx<hi)].sum(1) for lo,hi in zip(bins[:-1],bins[1:])],axis=1);es.append(b)
 maxe=max(x.max() for x in es)
 for j,p in enumerate(['deep','nowave']):
  a=D[p];im=ax[0,j].pcolormesh(a['mapx'],a['mapz'],a[f'u_map_{tt}']/float(a['u0']),cmap='RdBu_r',vmin=-3,vmax=3,shading='nearest',rasterized=True);ax[0,j].set(xlabel='$x$',ylabel='$z$',ylim=(-.25,0),title='Deep water' if p=='deep' else 'No waves')
  sp=ax[1,j].contourf(centers,a['top025_hann_ell'],np.maximum(es[j]/maxe,1e-5),levels=[.001,.003,.01,.03,.1,.3,1],norm=LogNorm(.001,1),cmap='Greys',extend='min');ax[1,j].set_aspect('equal');ax[1,j].set(xlim=(-32,32),ylim=(-32,32),xlabel=r'$K/(2\pi)$',ylabel=r'$\ell/(2\pi)$')
 fig.colorbar(im,ax=ax[0,:],label=r'$u^L/u_0$',shrink=.8);fig.colorbar(sp,ax=ax[1,:],label=r'$E/E_*$');save(fig,f'full-comparison-{tt}')
fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for p,color in [('deep','#176c9e'),('nowave','#ce7b28')]:
 a=D[p];v=a['pv_top0125'];ax[0].plot(a['t'],v[:,1]/v[0,1],color=color,label='Deep water' if p=='deep' else 'No waves')
 for tag,ls in [('top0125','-'),('top025','--')]:
  q=a['pv_'+tag];ax[1].plot(a['t'],q[:,3]/q[0,3],ls,color=color)
ax[0].set(xlabel='$t/T_e$',ylabel='Top 0.125 PV variance / initial');ax[1].set(xlabel='$t/T_e$',ylabel='Eddy PV variance / initial');ax[0].legend(frameon=False);save(fig,'pv-resolution-limit')
# Clean machine-readable NaNs and state inapplicable diagnostics explicitly.
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
p=O/'assessment-data.json';s=json.loads(p.read_text())
for r in s['momentum']:
 r['stress_closure_applicable']='linear' not in r['case']
 if not r['stress_closure_applicable']:r['stress_integral_vs_advection_relative']=None
s=clean(s);p.write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
rr=s['momentum']
with (O/'momentum-metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
# All plotted spectra in per-case compressed tables; preserve native bins, no images masquerading as data.
for name in sorted(I.glob('*-s1.npz')):
 a=np.load(name);out=O/'data';out.mkdir(exist_ok=True)
 with gzip.open(out/(name.stem+'-spectra.csv.gz'),'wt') as f:
  w=csv.writer(f);w.writerow(['window','method','time_or_average','K_cycles','ell_cycles','energy'])
  for tag in ['top0125_hann','top025_hann','top05_hann','top0125_reflect','top025_reflect','top05_reflect']:
   for suffix in ['E_0','E_10','E_25','E_50','E_100','avg10_25','avg50_100']:
    e=a[tag+'_'+suffix]
    for iz,l in enumerate(a[tag+'_ell']):
     for ix,k in enumerate(a[tag+'_kx']):w.writerow([*tag.split('_'),suffix,f'{k:g}',f'{l:g}',f'{e[iz,ix]:.9g}'])
 # User-facing compact evolution and profile data for reproducing figures.
 with gzip.open(out/(name.stem+'-evolution.csv.gz'),'wt') as f:
  w=csv.writer(f);w.writerow(['t_Te','E','mean_E','top0125_mean_E','top0125_mean_PV_variance','top0125_eddy_PV_variance','top0125_rms_gradient_ratio'])
  U=a['U'];E=a['E'];ME=a['Emean'];pv=a['pv_top0125'];top=.5*np.mean(U[:,a['zc']>=-.125]**2,axis=1)
  for j,t in enumerate(a['t']):w.writerow([t,E[j],ME[j],top[j],pv[j,2],pv[j,3],pv[j,4]])
 with gzip.open(out/(name.stem+'-profiles.csv.gz'),'wt') as f:
  w=csv.writer(f);w.writerow(['t_Te','z_face','Q','eddy_q_variance','momentum_flux'])
  for tt in [0,10,25,50,100]:
   for z,q,v,r in zip(a['zf'],a['Q'][tt*10],a['qvar'][tt*10],a['R'][tt*10]):w.writerow([tt,z,q,v,r])
 np.savetxt(out/(name.stem+'-radial-flux.csv'),a['radial_flux'],delimiter=',',header='t_Te,radius_cycles,advection,eddy_eddy,mean_related,stokes,viscosity,pressure,rhs',comments='')
print('Extra figures and data ready')
