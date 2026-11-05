"""Replot saved CL output only: horizontal Lagrangian velocity above spectra."""
from pathlib import Path
import argparse, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--wide-spectra',type=Path,required=True);p.add_argument('--deep-spectra',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
def field(campaign,case,t):
 d=args.data/campaign/'data'/case;m=dict(x.split('=',1) for x in (d/'meta.txt').read_text().splitlines());nx,nz=int(m['nx']),int(m['nz']);a=np.fromfile(d/'ymean'/f't{t:07.2f}.bin',np.float32);assert len(a)==nx*(2*nz+1)
 u=a[:nx*nz].reshape(nz,nx).T;z=np.loadtxt(d/'zc.txt');x=np.arange(nx)*float(m['lx'])/nx
 return x,z,u/float(m['u0'])
def draw(name,labels,fields,energies,kx,kz,depth,window=None):
 fig=plt.figure(figsize=(9,6.5 if depth==2 else 5.5));gs=fig.add_gridspec(2,2,height_ratios=[depth/4,1],left=.08,right=.86,bottom=.10,top=.94,hspace=.35,wspace=.25)
 emax=max(E.max() for E in energies)
 for j,(label,(x,z,u),E) in enumerate(zip(labels,fields,energies)):
  ax=fig.add_subplot(gs[0,j]);im=ax.pcolormesh(x,z,u.T,shading='nearest',cmap='RdBu_r',vmin=-3,vmax=3,rasterized=True);ax.set_aspect('equal');ax.set(xlim=(0,4),ylim=(-depth,0),xlabel='$x$',ylabel='$z$',title=label)
  if window:ax.axhline(-window,color='k',lw=.7,ls='--')
  ax=fig.add_subplot(gs[1,j]);es=ax.pcolormesh(kx,kz,np.log10(np.maximum(E.T/emax,1e-8)),shading='nearest',cmap='Greys',vmin=-3,vmax=0,rasterized=True);ax.set_aspect('equal');ax.set(xlim=(-14,14),ylim=(-14,14),xlabel=r'$K/(2\pi)$',ylabel=r'$\ell/(2\pi)$')
 top=fig.axes[0].get_position();bottom=fig.axes[1].get_position()
 fig.colorbar(im,cax=fig.add_axes([.895,top.y0,.016,top.height]),extend='both',label=r'$u^L/u_0$')
 fig.colorbar(es,cax=fig.add_axes([.895,bottom.y0,.016,bottom.height]),label=r'$\log_{10}(E/E_{\max})$')
 fig.savefig(args.out/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.12)
 fig.savefig(args.out/(name+'.pdf'),dpi=220,bbox_inches='tight',pad_inches=.12)
 Image.open(args.out/(name+'.png')).save(args.out/(name+'.webp'),lossless=True)
 plt.close(fig)
 return float(emax)
a=np.load(args.wide_spectra);kx=a['kx_cycles'];kz=a['kz_cycles'];t=a['t_Te'];order=np.argsort(kx);zz=np.r_[-kz[:0:-1],kz];checks={}
for lo,hi,tf in [(10,15,12),(15,20,17),(20,25,22),(25,30,27)]:
 Es=[a[key][:,(t>=lo)&(t<=hi)].mean((0,1)) for key in ['beta','nowave']]
 Es=[np.c_[E[:,0],E[:,1:]/2] for E in Es];Es=[np.c_[E[:,:0:-1],E][order] for E in Es]
 fs=[field('wide-beta-20261104',case,tf) for case in ['beta-h2-lx4-s1','nowave-h2-lx4-s1']]
 name=f'velocity-spectra-{lo}-{hi}';checks[name]={'velocity_time_Te':tf,'spectrum_window_Te':[lo,hi],'seeds':3,'Emax':draw(name,['Constant gradient','No waves'],fs,Es,kx[order],zz,2)}
d=np.load(args.deep_spectra);E=[d[key+'_0.5'].mean(0) for key in ['deep','none']]
f=[field('deep-ring',case,17) for case in ['ens-deep-k27-s1','ens-none-k27-s1']]
checks['deep-water-velocity-spectra']={'velocity_time_Te':17,'spectrum_window_Te':[15,20],'seeds':8,'Emax':draw('deep-water-velocity-spectra',['Deep-water waves','No waves'],f,E,d['kx_0.5'],d['kz_0.5'],1,.5)}
(args.out/'plot-details.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
