from pathlib import Path
import numpy as np,json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'results';O.mkdir(exist_ok=True);curves={};data={}
for key in ['beta','nowave']:
 arrays=[]
 for seed in [1,2,3]:
  r=R/'data'/f'{key}-h2-lx4-s{seed}';assert (r/'completed.txt').exists();a=np.load(r/'analysis/spectra.npz');arrays.append(a['energy']);curves[r.name]=json.load(open(r/'analysis/summary.json'))
 data[key]=np.array(arrays);kx=a['kx']/(2*np.pi);kz=a['kz']/(2*np.pi);t=a['t_Te']
np.savez_compressed(O/'ensemble-spectra.npz',**data,kx_cycles=kx,kz_cycles=kz,t_Te=t)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(4,2,figsize=(7,11));ix=np.argsort(kx);zz=np.r_[-kz[:0:-1],kz]
for i,(lo,hi) in enumerate([(10,15),(15,20),(20,25),(25,30)]):
 Es=[data[key][:,(t>=lo)&(t<=hi)].mean((0,1)) for key in ['beta','nowave']];Es=[np.c_[E[:,0],E[:,1:]/2] for E in Es];emax=max(E.max() for E in Es)
 for j,E in enumerate(Es):
  e=np.c_[E[:, :0:-1],E];ax=axs[i,j]
  im=ax.pcolormesh(kx[ix],zz,np.log10(np.maximum(e[ix].T/emax,1e-8)),cmap='Greys',vmin=-3,vmax=0,shading='nearest',rasterized=True);ax.set_aspect('equal');ax.set(xlim=(-14,14),ylim=(-14,14),xlabel=r'$K/(2\pi)$',ylabel=r'$\ell/(2\pi)$');ax.set_title(('Constant gradient' if j==0 else 'No waves')+f', {lo}–{hi} $T_e$')
fig.subplots_adjust(left=.10,right=.86,bottom=.06,top=.95,hspace=.38,wspace=.3);fig.colorbar(im,cax=fig.add_axes([.9,.25,.02,.5]),label=r'$\log_{10}(E/E_{\max})$');fig.savefig(O/'spectra-time-windows.png',dpi=220,bbox_inches='tight',pad_inches=.12)
json.dump({'runs':curves,'wavenumber_units':'cycles per original unit length, not renormalized to new depth','processing':'native full-depth cosine/sine modes; no padding or taper; common colour normalization within each row; 3 seeds'},open(O/'summary.json','w'),indent=2)
print('Wrote',O,flush=True)
