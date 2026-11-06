from pathlib import Path
import numpy as np,csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fft import dct,idct
from PIL import Image
R=Path('outputs/deep-water-long-analysis');F=R/'figures';a=np.load(R/'stage-spectra.npz');stages=[100,300,600,900];Te=float(a['Te']);u0=float(a['u0']);z=a['z'];rows=list(csv.DictReader((R/'directional-budgets.csv').open()))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','pdf.fonttype':42})
def save(fig,name):
 for ext in ['png','pdf']:fig.savefig(F/(name+'.'+ext),dpi=180,bbox_inches='tight',pad_inches=.12)
 with Image.open(F/(name+'.png')) as im:im.save(F/(name+'.webp'),lossless=True)
 plt.close(fig)
def get(c,t,sec):return next(x for x in rows if x['case']==c and int(x['stage_Te'])==t and x['window']=='top0.25_hann' and x['band_lo']=='1' and x['band_hi']=='24' and x['sector']==sec)
fig,axs=plt.subplots(2,3,figsize=(12,6),layout='constrained',sharex=True)
for i,c in enumerate(['deep','control']):
 for j,sec in enumerate(['horizontal','near_axis','K0']):
  ax=axs[i,j]
  for field,label,color,ls in [('T_centered','Nonlinear','#145c9e','-'),('T_wave','Wave','#8c3a78','-'),('T_pressure','Pressure','#38916c','-'),('observed_dE_dt','Measured change','#222','--')]:
   yy=[float(get(c,t,sec)[field])*Te/float(a[c+'_E0'])*1e3 for t in stages];ax.plot(stages,yy,ls+'o',ms=3,c=color,label=label)
  ax.axhline(0,c='.6',lw=.7);ax.set(xticks=stages);ax.grid(alpha=.15)
  if i==0:ax.set_title(['Off axis','Near jet axis','$K=0$'][j])
  else:ax.set_xlabel('$t/T_e$')
  if j==0:ax.set_ylabel(('Deep-water waves' if c=='deep' else 'No waves')+'\n'+r'$10^3\,\mathrm{input}/(E_0/T_e)$')
axs[0,0].legend(frameon=False,fontsize=8)
save(fig,'local-window-budget')
fig,axs=plt.subplots(1,4,figsize=(12,3.4),layout='constrained')
for ax,t in zip(axs,stages):
 for c,color,label in [('deep','#145c9e','Deep-water waves'),('control','#b15b35','No waves')]:
  ax.plot(a[f'{c}_{t}_full_native_radial_cutoffs'],-a[f'{c}_{t}_full_native_radial_input']*Te/float(a[c+'_E0'])*1e3,'o-',c=color,ms=3,label=label)
 ax.axhline(0,c='.6',lw=.7);ax.set(title=f'$t/T_e={t}$',xlabel=r'$\kappa_c/(2\pi)$');ax.grid(alpha=.15)
axs[0].set_ylabel(r'$10^3\,\Pi/(E_0/T_e)$');axs[0].legend(frameon=False,fontsize=8)
save(fig,'radial-transfer')
colors=['#5d77a3','#3a9b91','#d59d43','#9c4274'];n=len(z);qwave=-2*np.exp(8*z);ell=np.arange(n)*np.pi/2;qr=[]
fig,axs=plt.subplots(2,2,figsize=(8,7),layout='constrained',sharey=True)
for i,c in enumerate(['deep','control']):
 for t,color in zip(stages,colors):
  U=a[f'{c}_{t}_U'];axs[i,0].plot(U/u0,z,c=color,label=str(t))
  for sigma in [.005,.01,.02]:
   Uzz=idct(-ell**2*dct(U,type=2,norm='ortho')*np.exp(-.5*(sigma*ell)**2),type=2,norm='ortho');q=Uzz+(qwave if c=='deep' else 0)
   if sigma==.01:axs[i,1].plot(q,z,c=color)
   sel=(z>-.25)&(z<-.03);qr.append(dict(case=c,stage_Te=t,sigma=sigma,curvature_to_stokes_rms=float(np.sqrt(np.mean(Uzz[sel]**2)/np.mean(qwave[sel]**2)))))
 if c=='deep':axs[i,1].plot(qwave,z,'--',c='.4',label='Prescribed')
 for j in range(2):axs[i,j].set(ylim=(-.5,0));axs[i,j].grid(alpha=.15)
 axs[i,0].set_ylabel(('Deep-water waves' if c=='deep' else 'No waves')+'\n$z$')
axs[0,0].legend(title='$t/T_e$',frameon=False,fontsize=8);axs[0,1].legend(frameon=False,fontsize=8)
axs[1,0].set_xlabel('$U^L/u_0$');axs[0,1].set_xlabel(r'$Q_z=-2e^{8z}+U^L_{zz}$');axs[1,1].set_xlabel(r'$Q_z=U^L_{zz}$')
save(fig,'mean-current-gradient')
with (R/'mean-gradient-sensitivity.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(qr[0]));w.writeheader();w.writerows(qr)
print(json.dumps(qr,indent=2))
