from pathlib import Path
import json,csv,gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from PIL import Image
I=Path('outputs/deep-water-mechanism-pilot100-20261106/analysis');O=I/'gallery';O.mkdir(parents=True,exist_ok=True)
CASES=[p.stem for p in sorted(I.glob('*-s1.npz'))];D={c:np.load(I/(c+'.npz')) for c in CASES};M={c:json.loads((I/(c+'.json')).read_text()) for c in CASES}
ARMS=['linear','ql','full'];LABELS=['Fixed mean','Quasilinear','Fully nonlinear'];COL=['#70869a','#ce7b28','#176c9e'];u0=float(next(iter(D.values()))['u0'])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','pdf.fonttype':42})
def save(fig,name):
 fig.savefig(O/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(O/(name+'.pdf'),bbox_inches='tight')
 with Image.open(O/(name+'.png')) as im:im.save(O/(name+'.webp'),quality=90)
 plt.close(fig)
def get(profile,arm):return D[f'{profile}-{arm}-s1']
def layerjet(a,depth):
 mask=a['zc']>=-depth
 # Profile total energy is not retained at all t: jet fraction uses exact snapshot energy from all x modes below.
 return .5*np.mean(a['U'][:,mask]**2,axis=1)
# Dynamics and native resolution evidence.
fig,ax=plt.subplots(2,2,figsize=(10,7))
for i,arm in enumerate(ARMS):
 for profile,ls in [('deep','-'),('nowave','--')]:
  a=get(profile,arm);t=a['t'];label=LABELS[i]+(' / no waves' if profile=='nowave' else ' / deep water')
  ax[0,0].plot(t,a['Emean']/a['E'],ls,color=COL[i],label=label)
  ax[0,1].plot(t,a['E']/a['E'][0],ls,color=COL[i])
  fm=a['field_metrics'];ax[1,0].semilogy(fm[:,0],np.maximum(fm[:,3],1e-18),ls,color=COL[i])
  ax[1,1].plot(t,layerjet(a,.125)/a['E'][0],ls,color=COL[i])
for a in ax.ravel():a.set_xlabel('$t/T_e$')
ax[0,0].set_ylabel('Full-depth mean-energy fraction');ax[0,1].set_ylabel('$E/E(0)$');ax[1,0].set_ylabel('Grid-tail energy / total');ax[1,1].set_ylabel('Top 0.125 mean energy / $E(0)$')
ax[0,0].legend(fontsize=7,frameon=False,ncol=2);fig.tight_layout();save(fig,'evolution')
# Matched snapshots, velocity above spectra, same scale across all six.
tag='top025_hann';scale=max(D[c][tag+'_E_100'][...,D[c][tag+'_kx']!=0].max() for c in CASES)
for profile in ['deep','nowave']:
 fig,axs=plt.subplots(3,3,figsize=(11,9),gridspec_kw={'height_ratios':[.55,1,1]},layout='constrained')
 for j,arm in enumerate(ARMS):
  a=get(profile,arm);im=axs[0,j].pcolormesh(a['mapx'],a['mapz'],a['u_map_100']/u0,cmap='RdBu_r',vmin=-3,vmax=3,shading='nearest',rasterized=True);axs[0,j].set(xlabel=r'$x$',ylabel=r'$z$',ylim=(-.25,0),title=LABELS[j])
  for row,depthtag in [(1,'top025_hann'),(2,'top0125_hann')]:
   e=a[depthtag+'_E_100'].copy();e[:,a[depthtag+'_kx']==0]=np.nan
   sp=axs[row,j].pcolormesh(a[depthtag+'_kx'],a[depthtag+'_ell'],np.log10(np.maximum(e/scale,1e-6)),vmin=-4,vmax=0,cmap='Greys',shading='nearest',rasterized=True)
   axs[row,j].set(xlim=(-32,32),ylim=(-32,32),xlabel=r'$K/(2\pi)$',ylabel=r'$\ell/(2\pi)$',title=r'$D=0.25$' if row==1 else '$D=0.125$');axs[row,j].set_aspect('equal')
 fig.colorbar(im,ax=axs[0,:],shrink=.8,label=r'$u^L/u_0$');fig.colorbar(sp,ax=axs[1:,:],shrink=.65,label=r'$\log_{10}(E/E_*)$');save(fig,'snapshots-'+profile)
# Early to late matched full cases: compare identical average windows.
fig,axs=plt.subplots(2,2,figsize=(9,8),layout='constrained')
for row,(lo,hi) in enumerate([(10,25),(50,100)]):
 scale2=max(get(p,'full')[f'top025_hann_avg{lo}_{hi}'][:,get(p,'full')['top025_hann_kx']!=0].max() for p in ['deep','nowave'])
 for col,p in enumerate(['deep','nowave']):
  a=get(p,'full');e=a[f'top025_hann_avg{lo}_{hi}'].copy();e[:,a['top025_hann_kx']==0]=np.nan
  im=axs[row,col].pcolormesh(a['top025_hann_kx'],a['top025_hann_ell'],np.log10(np.maximum(e/scale2,1e-6)),vmin=-4,vmax=0,cmap='Greys',shading='nearest');axs[row,col].set(xlim=(-32,32),ylim=(-32,32),xlabel=r'$K/(2\pi)$',ylabel=r'$\ell/(2\pi)$',title=('Deep water' if p=='deep' else 'No waves')+f' / {lo}–{hi} $T_e$');axs[row,col].set_aspect('equal')
fig.colorbar(im,ax=axs.ravel(),label=r'$\log_{10}(E/E_*)$',shrink=.7);save(fig,'spectral-time-means')
# PV: show both Q and its change; identical large background cannot hide perturbations.
fig,ax=plt.subplots(2,3,figsize=(12,7),layout='constrained')
for j,arm in enumerate(ARMS):
 a=get('deep',arm);z=a['zf'];mask=z>=-.25
 for tt,color in [(0,'#999999'),(25,'#91b4c7'),(50,'#4483a5'),(100,'#083e60')]:
  it=int(tt*10);ax[0,j].plot(a['Q'][it,mask],z[mask],color=color,label=str(tt));ax[1,j].plot(a['Q'][it,mask]-a['Q'][0,mask],z[mask],color=color)
 ax[0,j].set(title=LABELS[j],xlabel=r'$Q$',ylabel=r'$z$');ax[1,j].set(xlabel=r'$Q-Q(0)$',ylabel=r'$z$');ax[1,j].ticklabel_format(axis='x',style='sci',scilimits=(-2,2))
ax[0,0].legend(title=r'$t/T_e$',frameon=False,fontsize=8);save(fig,'pv-profiles')
fig,ax=plt.subplots(2,3,figsize=(12,7),layout='constrained')
for j,tag2 in enumerate(['top0125','top025','interior']):
 for i,arm in enumerate(ARMS):
  a=get('deep',arm);v=a['pv_'+tag2];ax[0,j].plot(a['t'],v[:,2]/v[0,2],color=COL[i],label=LABELS[i]);ax[1,j].plot(a['t'],v[:,4],color=COL[i])
 ax[0,j].set(title={'top0125':'$-0.125<z<0$','top025':'$-0.25<z<0$','interior':'$-0.125<z<-0.0625$'}[tag2],xlabel=r'$t/T_e$',ylabel=r'Mean-PV variance / initial');ax[1,j].set(xlabel=r'$t/T_e$',ylabel=r'RMS $Q_z$ / initial')
 ax[0,j].axhline(1,color='k',lw=.6,ls=':');ax[1,j].axhline(1,color='k',lw=.6,ls=':')
ax[0,0].legend(frameon=False,fontsize=8);save(fig,'pv-tests')
# Momentum closure and signed flux. Shapes native (t,z,term).
fig,ax=plt.subplots(2,2,figsize=(10,8),layout='constrained')
for j,arm in enumerate(['ql','full']):
 a=get('deep',arm);z=a['zc'];zf=a['zf'];mi=a['mean_increment'][-1]
 assert mi.shape==(len(z),6)
 ax[0,j].plot(a['U'][-1]/u0,z,color='k',lw=2,label=r'$U(100)/u_0$');ax[0,j].plot(mi[:,0]/u0,z,color='#ce7b28',ls='--',label=r'Advection contribution');ax[0,j].plot(mi[:,2]/u0,z,color='#176c9e',ls=':',label=r'Viscosity contribution');ax[0,j].set(ylim=(-.5,0),xlabel=r'Velocity / $u_0$',ylabel=r'$z$',title=LABELS[ARMS.index(arm)])
 for p,ls in [('deep','-'),('nowave','--')]:
  b=get(p,arm);rr=np.trapezoid(b['R'][500:],b['t'][500:],axis=0)/50;ax[1,j].plot(rr/u0**2,zf,ls,label=r'Deep water' if p=='deep' else 'No waves')
 ax[1,j].axvline(0,color='k',lw=.6);ax[1,j].set(ylim=(-.5,0),xlabel=r'$\langle R\rangle_{50–100}/u_0^2$',ylabel=r'$z$')
ax[0,0].legend(fontsize=8,frameon=False);ax[1,0].legend(fontsize=8,frameon=False);save(fig,'momentum-budget')
fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
mx=max(abs(get(p,a)['stress_mean50_100'][-256:,:161]).max() for p in ['deep','nowave'] for a in ['ql','full'])/u0**2
for row,p in enumerate(['deep','nowave']):
 for col,arm in enumerate(['ql','full']):
  a=get(p,arm);im=axs[row,col].pcolormesh(a['K'][:161],a['zf'],a['stress_mean50_100'][:,:161]/u0**2,cmap='RdBu_r',norm=SymLogNorm(linthresh=mx/100,vmin=-mx,vmax=mx),shading='nearest',rasterized=True);axs[row,col].set(xlim=(0,32),ylim=(-.5,0),xlabel=r'$K/(2\pi)$',ylabel=r'$z$',title=('Deep water' if p=='deep' else 'No waves')+' / '+LABELS[ARMS.index(arm)])
fig.colorbar(im,ax=axs.ravel(),label=r'Signed stress / $u_0^2$',shrink=.75);save(fig,'momentum-cospectra')
# Full-domain inverse transfer; actual nonlinear contribution, not inferred from an image.
fig,ax=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for a,tt in zip(ax.ravel(),[10,25,50,100]):
 for p,color in [('deep','#176c9e'),('nowave','#ce7b28')]:
  b=get(p,'full');f=b['radial_flux'];f=f[f[:,0]==tt];scale=float(b['Te'])/b['E'][0]
  a.plot(f[:,1],f[:,2]*scale,color=color,label=r'Deep water' if p=='deep' else 'No waves');a.plot(f[:,1],f[:,3]*scale,color=color,ls='--',alpha=.8)
 a.axhline(0,color='k',lw=.6);a.set(title=f'{tt} $T_e$',xlabel=r'Radius / $2\pi$',ylabel=r'$\Pi T_e/E(0)$',xlim=(0,40))
ax[0,0].legend(frameon=False);save(fig,'nonlinear-flux')
# Window sensitivity: horizontal sector fraction deep/control, excludes mean.
fig,ax=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for row,(low,high) in enumerate([(4,16),(16,24)]):
 for col,tt in enumerate([25,100]):
  for arm,color in [('linear',COL[0]),('ql',COL[1]),('full',COL[2])]:
   for method,ls in [('hann','-'),('reflect','--')]:
    values=[]
    for tag3 in ['top05','top025','top0125']:
     rr=[]
     for p in ['deep','nowave']:
      r=next(r for r in M[f'{p}-{arm}-s1']['spectral_metrics'] if r['t']==tt and r['window']==tag3 and r['method']==method and r['low']==low);rr.append(r['horizontal_fraction'])
     values.append(rr[0]/rr[1] if rr[1]>1e-12 else np.nan)
    ax[row,col].plot([.5,.25,.125],values,ls,color=color,marker='o',label=LABELS[ARMS.index(arm)]+(' / Hann' if method=='hann' else ' / reflected'))
  ax[row,col].axhline(1,color='k',lw=.7);ax[row,col].set(title=f'{tt} $T_e$ / {low}–{high} cycles',xlabel=r'Window depth',ylabel=r'Horizontal fraction: deep / control',xticks=[.125,.25,.5]);ax[row,col].set_ylim(bottom=0)
ax[0,0].legend(fontsize=7,frameon=False,ncol=2);save(fig,'window-sensitivity')
# Compact auditable metrics and data files.
rows=[];pvr=[];mom=[];bound=[]
for c,a in D.items():
 m=M[c];last=a['field_metrics'][-1]
 row={'case':c,'E_final_over_initial':float(a['E'][-1]/a['E'][0]),'jet_fraction_final':float(a['Emean'][-1]/a['E'][-1]),'grid_tail_fraction_final':float(last[3]),'eddy_low16_fraction_final':float(last[4]),'max_divergence':float(np.max(a['divergence']))};rows.append(row)
 for tag4 in ['top0125','top025','interior','top05']:
  v=a['pv_'+tag4];b=a['pvbudget_'+tag4];pvr.append({'case':c,'layer':tag4,'mean_variance_ratio':float(v[-1,2]/v[0,2]) if v[0,2]>1e-25 else None,'total_variance_ratio':float(v[-1,1]/v[0,1]),'gradient_ratio':float(v[-1,4]),'flat_depth_fraction':float(v[-1,5]),'observed_half_variance_change':float((v[-1,1]-v[0,1])/2),**{term:float(b[i]) for i,term in enumerate(m['pv_budget_columns'])}})
 # Exact accumulated component contributions versus observed mean; physical stress integral is sampled, not exact RK.
 t=a['t']*float(a['Te']);dz=a['zc'][1]-a['zc'][0];pred=-np.diff(np.trapezoid(a['R'],t,axis=0))/dz
 actual=a['U'][-1]-a['U'][0];inc=a['mean_increment'][-1];shear=np.diff(a['U'],axis=1)/dz
 power=np.sum(a['R'][:,1:-1]*shear,axis=1)/len(a['zc']);integ=np.trapezoid(power,t)
 mom.append({'case':c,'stress_integral_vs_advection_relative':float(np.linalg.norm(pred-inc[:,0])/max(np.linalg.norm(inc[:,0]),u0*1e-12)),'mean_advection_norm':float(np.linalg.norm(inc[:,0])),'mean_wave_norm':float(np.linalg.norm(inc[:,1])),'mean_pressure_norm':float(np.linalg.norm(inc[:,3])),'mean_viscosity_norm':float(np.linalg.norm(inc[:,2])),'mean_constraint_norm':float(np.linalg.norm(inc[:,4])),'mean_update_residual_over_u0':float(np.max(abs(actual-inc[:,-1]))/u0),'integrated_stress_shear_work':float(integ),'mean_energy_change':float(a['Emean'][-1]-a['Emean'][0]),'net_momentum_over_u0':float(abs(actual.mean())/u0)})
for lo,hi in [(-.0625,0),(-.125,0),(-.25,0),(-.5,0),(-.125,-.0625)]:
 a=get('deep','full');zc=a['zc'];zf=a['zf'];dz=zc[1]-zc[0];mask=(zc>=lo)&(zc<=hi);zs=zc[mask];faces=(zs[:-1]+zs[1:])/2;B=np.r_[0,np.cumsum(.25*np.exp(8*faces)*dz)];X=np.c_[np.ones(len(zs)),zs];u=B-X@np.linalg.lstsq(X,B,rcond=None)[0];emin=np.sum(u*u)/(2*len(zc));bound.append({'lower':lo,'upper':hi,'minimum_mean_energy':float(emin),'ratio_to_initial_total_energy':float(emin/a['E'][0]),'ratio_to_final_total_energy':float(emin/a['E'][-1])})
summary={'cases':rows,'pv':pvr,'momentum':mom,'PV_homogenization_energy_lower_bounds':bound,'single_seed':1,'duration_Te':100,'analysis_source':'reduce.py and plot.py','simulation_jobs':['66829261','66837635'],'analysis_only':True}
(O/'assessment-data.json').write_text(json.dumps(summary,indent=2)+'\n')
for name,rr in [('case-metrics',rows),('pv-metrics',pvr),('momentum-metrics',mom),('homogenization-energy-bound',bound)]:
 with (O/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
for name,rr in [('spectral-metrics',[dict(case=c,**r) for c in M for r in M[c]['spectral_metrics']])]:
 with gzip.open(O/(name+'.csv.gz'),'wt') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
print(json.dumps(summary,indent=2))
