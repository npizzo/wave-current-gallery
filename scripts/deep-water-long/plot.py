from pathlib import Path
import csv,json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,ListedColormap
from PIL import Image
R=Path('outputs/deep-water-long-analysis');F=R/'figures';F.mkdir(exist_ok=True);a=np.load(R/'stage-spectra.npz');k=a['k'];u0=float(a['u0']);Te=float(a['Te']);stages=[100,300,600,900];cases=['deep','control'];titles=['Deep-water waves','No waves'];col=['#145c9e','#b15b35'];levels=[0,.003,.01,.03,.1,.3,1.00001];grey=ListedColormap(['#ffffff','#eeeeee','#cccccc','#999999','#666666','#242424']);norm=BoundaryNorm(levels,grey.N)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':10,'savefig.facecolor':'white','pdf.fonttype':42,'axes.titleweight':'normal'})
def save(fig,name,pdf=True):
 fig.savefig(F/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.12)
 if pdf:fig.savefig(F/(name+'.pdf'),dpi=180,bbox_inches='tight',pad_inches=.12)
 with Image.open(F/(name+'.png')) as im:im.save(F/(name+'.webp'),lossless=True)
 plt.close(fig)
def signed(E,ell):
 o=np.argsort(ell);ll=ell[o];ee=E[o].copy();ee[:,1:]/=2
 # E(-K,ell)=E(K,-ell); K=0 retained once.
 return np.r_[-k[:0:-1],k],ll,np.c_[ee[::-1, :0:-1],ee]
def display(c,t,tag,eddy=False,binx=True):
 e=a[f'{c}_{t}_{tag}_E'].copy()
 if eddy:e[:,0]=0
 xx,ll,e=signed(e,a[tag+'_ell'])
 if binx:
  width=float(np.diff(ll).mean());centers=np.arange(-40,40.01,width);W=np.maximum(np.minimum(xx[None,:]+.125,centers[:,None]+width/2)-np.maximum(xx[None,:]-.125,centers[:,None]-width/2),0)/.25
  e=e@W.T;xx=centers
 return xx,ll,e
for t in stages:
 for pixel in [False,True]:
  fig=plt.figure(figsize=(9.5,9.0));gs=fig.add_gridspec(3,2,height_ratios=[.85,3,3],left=.08,right=.85,bottom=.065,top=.96,hspace=.30,wspace=.25)
  for j,c in enumerate(cases):
   ax=fig.add_subplot(gs[0,j]);im=ax.pcolormesh(a['x_display'],a['z_display'],a[f'{c}_{t}_u']/u0,cmap='RdBu_r',vmin=-4,vmax=4,rasterized=True,shading='nearest');ax.axhline(-.25,c='.2',ls='--',lw=.6);ax.set(xlim=(0,4),ylim=(-.5,0),xlabel='$x$',ylabel='$z$',title=titles[j])
   if j==1:
    pos=ax.get_position();fig.colorbar(im,cax=fig.add_axes([.90,pos.y0,.015,pos.height]),label=r'$u^L/u_0$',ticks=[-4,0,4])
  for row,eddy in [(1,False),(2,True)]:
   dat=[display(c,t,'top0.25_hann',eddy,not pixel) for c in cases];scale=max(e.max() for _,_,e in dat)
   for j,(xx,ll,e) in enumerate(dat):
    ax=fig.add_subplot(gs[row,j]);ratio=e/scale
    if pixel:im=ax.pcolormesh(xx,ll,np.log10(np.maximum(ratio,1e-8)),vmin=-3.5,vmax=0,cmap='Greys',rasterized=True,shading='nearest')
    else:
     im=ax.contourf(xx,ll,ratio,levels=levels,cmap=grey,norm=norm)
     ax.contour(xx,ll,ratio,levels=[.01,.1,.3],colors='.3',linewidths=.35)
    ax.set(xlim=(-24,24),ylim=(-24,24),xlabel=r'$K/(2\pi)$',ylabel=r'$\ell/(2\pi)$',title=r'$K\ne0$' if eddy else 'Total');ax.set_aspect('equal')
    if j==1:
     pos=ax.get_position();fig.colorbar(im,cax=fig.add_axes([.90,pos.y0,.015,pos.height]),label=r'$\log_{10}(E/E_{\max})$' if pixel else '$E/E_{\max}$',ticks=[-3,-2,-1,0] if pixel else [.003,.01,.03,.1,.3,1])
  save(fig,f'velocity-spectra-{t}'+('-native' if pixel else ''),pdf=not pixel)
 # Estimator/depth sensitivity, K=0 removed, log display of actual samples.
 fig,axs=plt.subplots(3,4,figsize=(12,9),layout='constrained')
 for i,d in enumerate([.5,.25,.125]):
  for j,m in enumerate(['hann','reflect']):
   tag=f'top{d}_{m}';dat=[display(c,t,tag,True,False) for c in cases];scale=max(e.max() for _,_,e in dat)
   for cc,(xx,ll,e) in enumerate(dat):
    ax=axs[i,2*j+cc];im=ax.pcolormesh(xx,ll,np.log10(np.maximum(e/scale,1e-8)),vmin=-3.5,vmax=0,cmap='Greys',rasterized=True,shading='nearest');ax.set(xlim=(-24,24),ylim=(-24,24),xlabel=r'$K/(2\pi)$',ylabel=r'$\ell/(2\pi)$');ax.set_aspect('equal')
    if i==0:ax.set_title(titles[cc]+'\n'+('Hann' if m=='hann' else 'Reflected'))
    if j==0 and cc==0:ax.set_ylabel(f'$D={d:g}$\n'+r'$\ell/(2\pi)$')
 fig.colorbar(im,ax=axs,label=r'$\log_{10}(E/E_{\max})$',shrink=.6,pad=.015)
 save(fig,f'window-sensitivity-{t}')
# Tabulated sector diagnostics.
rows=list(csv.DictReader((R/'directional-budgets.csv').open()))
def get(c,t,tag,lo,hi,sector):return next(x for x in rows if x['case']==c and int(x['stage_Te'])==t and x['window']==tag and int(x['band_lo'])==lo and int(x['band_hi'])==hi and x['sector']==sector)
def ratio(t,tag,lo,hi):
 x,y=[get(c,t,tag,lo,hi,'horizontal') for c in cases]
 if x['angular_comparison_valid']!='True':return np.nan
 return (float(x['energy'])/float(x['annulus_eddy_energy']))/(float(y['energy'])/float(y['annulus_eddy_energy']))
fig,axs=plt.subplots(1,3,figsize=(10,3.5),layout='constrained',sharey=True)
for ax,d in zip(axs,[.5,.25,.125]):
 for lo,hi,color in [(4,8,'#9a7428'),(8,16,'#145c9e'),(16,24,'#8c3a78')]:
  for m,ls in [('hann','-'),('reflect','--')]:ax.plot(stages,[ratio(t,f'top{d}_{m}',lo,hi) for t in stages],ls,marker='o',ms=3,color=color,label=f'{lo}–{hi}' if m=='hann' else None)
 ax.axhline(1,c='.6',lw=.8);ax.set(xlabel='$t/T_e$',title=f'$D={d:g}$',ylim=(0,2),xticks=stages);ax.grid(alpha=.15)
axs[0].set_ylabel('Horizontal-sector fraction\nwave / control');axs[0].legend(title=r'$\kappa/(2\pi)$',frameon=False,fontsize=9)
save(fig,'angular-depletion')
# Native directional transfer by disjoint radial band: no local-window ambiguity.
fig,axs=plt.subplots(2,4,figsize=(12,6),layout='constrained',sharex=True)
for j,(lo,hi) in enumerate([(1,4),(4,8),(8,16),(16,24)]):
 for i,sector in enumerate(['horizontal','near_axis']):
  ax=axs[i,j]
  for c,color,label in zip(cases,col,titles):
   vals=[float(get(c,t,'full_native',lo,hi,sector)['T_centered'])*Te/float(a[c+'_E0']) for t in stages];ax.plot(stages,np.array(vals)*1e3,'o-',ms=3,c=color,label=label)
  ax.axhline(0,c='.5',lw=.7);ax.set(xticks=stages);ax.grid(alpha=.15)
  if i==0:ax.set_title(f'{lo}–{hi}')
  else:ax.set_xlabel('$t/T_e$')
  if j==0:ax.set_ylabel(('Off axis' if i==0 else 'Near jet axis')+'\n'+r'$10^3\,\mathcal{N}/(E_0/T_e)$')
axs[0,0].legend(frameon=False,fontsize=8)
save(fig,'directional-transfer')
# Signed full-depth net-input budget, 1<=kappa<24: measured dE/dt and components.
fig,axs=plt.subplots(2,3,figsize=(12,6),layout='constrained',sharex=True)
styles=[('T_centered','Nonlinear','#145c9e','-'),('T_wave','Wave','#8c3a78','-'),('T_numerical','Numerical','#be8429','--'),('observed_dE_dt','Measured change','#222','--')]
for i,c in enumerate(cases):
 for j,sec in enumerate(['horizontal','near_axis','K0']):
  ax=axs[i,j]
  for field,label,color,ls in styles:
   yy=[float(get(c,t,'full_native',1,24,sec)[field])*Te/float(a[c+'_E0'])*1e3 for t in stages];ax.plot(stages,yy,ls+'o',color=color,label=label,ms=3)
  ax.axhline(0,c='.6',lw=.7);ax.set(xticks=stages);ax.grid(alpha=.15)
  if i==0:ax.set_title(['Off axis','Near jet axis','$K=0$'][j])
  else:ax.set_xlabel('$t/T_e$')
  if j==0:ax.set_ylabel(titles[i]+'\n'+r'$10^3\,\mathrm{input}/(E_0/T_e)$')
axs[0,0].legend(frameon=False,fontsize=8)
save(fig,'directional-budget')
# Advecting-interaction split for deep-water: mean–eddy vs eddy–eddy.
fig,axs=plt.subplots(1,3,figsize=(10,3.3),layout='constrained',sharex=True)
for ax,sec,title in zip(axs,['horizontal','near_axis','K0'],['Off axis','Near jet axis','$K=0$']):
 for term,label,color in [('centered','Total','#222'),('eddy','Eddy–eddy','#145c9e'),('meaneddy','Mean–eddy','#b15b35')]:
  yy=[float(get('deep',t,'full_native',1,24,sec)['T_'+term])*Te/float(a['deep_E0'])*1e3 for t in stages];ax.plot(stages,yy,'o-',ms=3,c=color,label=label)
 ax.axhline(0,c='.6',lw=.7);ax.set(title=title,xlabel='$t/T_e$',xticks=stages);ax.grid(alpha=.15)
axs[0].set_ylabel(r'$10^3\,\mathcal{N}/(E_0/T_e)$');axs[0].legend(frameon=False,fontsize=8)
save(fig,'interaction-split')
# Evolution of native mean energy.
evo=np.genfromtxt(R/'evolution.csv',delimiter=',',names=True,dtype=None,encoding=None)
fig,axs=plt.subplots(1,3,figsize=(10,3.3),layout='constrained')
for c,color,label in zip(cases,col,titles):
 x=evo[evo['case']==c];tt=x['t_Te'];axs[0].plot(tt,x['native_energy']/x['native_energy'][0],c=color,label=label);axs[1].plot(tt,x['jet_fraction_full'],c=color);axs[2].plot(tt,x['jet_fraction_top025'],c=color)
for ax in axs:ax.set_xlabel('$t/T_e$');ax.grid(alpha=.15)
axs[0].set_ylabel('$E/E_0$');axs[0].legend(frameon=False,fontsize=8);axs[1].set_ylabel('Horizontal-mean energy fraction');axs[1].set_title('Full depth');axs[2].set_title('$-0.25<z<0$')
save(fig,'energy-and-jets')
print('Created',len(list(F.glob('*.webp'))),'figures')
