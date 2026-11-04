"""Option 2 diagnostics: jet fraction vs time, fields, and the 1D kx-spectrum collapse.
At each depth z: E(kx;z) = |FFT_x u|^2+|FFT_x w|^2 of the y-mean field, averaged over the last 20% of the run,
normalized by its sum over kx, plotted against kx/k_beta(z) with k_beta=(beta(z)/u~(z))^{1/2}/2pi.
Metric: share of the depth's energy at |kx| <= 0.5 k_beta(z) (kx=0 = jets, included), per depth."""
import numpy as np, matplotlib as mpl, os, json, glob
mpl.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D=f'{ROOT}/data'; OUT=f'{ROOT}/figs/exp2d'; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.size':9,'font.family':'serif','mathtext.fontset':'cm'})
RUNS=[('s2d-exp2-k27-lx4','$\\beta\\propto e^{2z}$'),('s2d-exp4-k27-lx4','$\\beta\\propto e^{4z}$'),('s2d-deep-k27-lx4','$\\beta\\propto e^{8z}$ (deep)'),
      ('s2d-ubeta-k27-lx4','uniform $\\beta$'),('s2d-none-k27-lx4','no waves ($e^{8z}$ scaling)')]
ZS=[-0.05,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8]
TEREF=113.09733552923254
def meta(r): return dict(l.strip().split('=',1) for l in open(f'{D}/{r}/meta.txt'))
def beta(m,z):
    s=m.get('shear','deep'); g=float(m.get('gam','8'))
    return 0.5+0*np.asarray(z) if s=='ubeta' else (0.5*np.exp(0.17*g)*np.exp(g*np.asarray(z)) if s=='exp' else 2*np.exp(8*np.asarray(z)))
def frames(r,frac=0.2):
    fs=sorted(glob.glob(f'{D}/{r}/ymean/t*.bin'),key=lambda f: float(os.path.basename(f)[1:-4])); return fs,fs[int((1-frac)*len(fs)):]
summary={}
# ---- jet fraction vs time (whole depth and upper half via profiles)
fig,axs=plt.subplots(1,2,figsize=(9,3.2),gridspec_kw=dict(wspace=0.25,left=0.07,right=0.98,bottom=0.15,top=0.88))
cols=['#1b7f5a','#2a6fb0','#0b3d6b','#7a3e9d','#d9822b']
for (r,lab),c in zip(RUNS,cols):
    m=meta(r); nz=int(m['nz']); z=np.loadtxt(f'{D}/{r}/zc.txt'); T=float(m['Te'])/TEREF
    ts=np.genfromtxt(f'{D}/{r}/timeseries.csv',delimiter=',',names=True); axs[0].plot(ts['t_Te']*T,ts['KEjet']/ts['KE'],color=c,lw=1.1,label=lab)
    p=np.loadtxt(f'{D}/{r}/profiles.csv',delimiter=',',skiprows=1); up=z>-0.5
    jf_up=0.5*(p[:,1:1+nz][:,up]**2).mean(1)/(p[:,1+nz:1+2*nz]+p[:,1+2*nz:])[:,up].mean(1)
    axs[1].plot(p[:,0]*T,jf_up,color=c,lw=1.1,label=lab)
    summary[r]=dict(jet_frac_final=float(ts['KEjet'][-1]/ts['KE'][-1]),jet_frac_upper_last20=float(jf_up[int(0.8*len(jf_up)):].mean()))
for a,t in zip(axs,['jet fraction, full depth','jet fraction, $z>-0.5$']):
    a.set_title(t); a.set_xlabel('$t$ / ($T_e$ of the 9-cycle ring)'); a.set_xlim(0,1000); a.set_ylim(0,1); a.grid(color='0.9',lw=0.5)
axs[0].legend(fontsize=7.5,frameon=False,loc='upper left'); fig.savefig(f'{OUT}/jet-fraction.png',dpi=140); plt.close(fig)
# ---- fields at the end of each wave run
fig,axs=plt.subplots(len(RUNS),1,figsize=(7.2,9.5),gridspec_kw=dict(hspace=0.35,left=0.08,right=0.9,bottom=0.05,top=0.96))
for a,(r,lab) in zip(axs,RUNS):
    m=meta(r); nx,nz,lx,u0=int(m['nx']),int(m['nz']),float(m['lx']),float(m['u0']); z=np.loadtxt(f'{D}/{r}/zc.txt')
    fs,_=frames(r); v=np.fromfile(fs[-1],dtype=np.float32); u=v[:nx*nz].reshape(nz,nx)
    im=a.pcolormesh(np.arange(nx)*lx/nx,z,u/u0,cmap='RdBu_r',vmin=-3,vmax=3,shading='auto',rasterized=True); a.set_aspect('equal'); a.set_title(lab+r', $u(x,z)/u_0$ at $t=1000\,T_e$ (9-cycle)',fontsize=8.5)
fig.colorbar(im,cax=fig.add_axes([0.92,0.35,0.012,0.3]),extend='both'); fig.savefig(f'{OUT}/fields-final.png',dpi=120); plt.close(fig)
# ---- 1D kx collapse
fig,axs=plt.subplots(2,len(RUNS),figsize=(3.0*len(RUNS),5.8),gridspec_kw=dict(hspace=0.4,wspace=0.3,left=0.05,right=0.98,bottom=0.09,top=0.9),sharey='row')
cmap=plt.cm.viridis
for c,(r,lab) in enumerate(RUNS):
    m=meta(r); nx,nz,lx=int(m['nx']),int(m['nz']),float(m['lx']); z=np.loadtxt(f'{D}/{r}/zc.txt'); _,use=frames(r)
    p=np.loadtxt(f'{D}/{r}/profiles.csv',delimiter=',',skiprows=1); p=p[int(0.8*len(p)):]; Ez=(p[:,1+nz:1+2*nz]+p[:,1+2*nz:]).mean(0)
    kx=np.fft.rfftfreq(nx,lx/nx); acc=np.zeros((nz,len(kx)))
    for f in use:
        v=np.fromfile(f,dtype=np.float32).astype(float); u=v[:nx*nz].reshape(nz,nx); w=v[nx*nz:].reshape(nz+1,nx); wc=(w[:-1]+w[1:])/2
        acc+=np.abs(np.fft.rfft(u,axis=1))**2+np.abs(np.fft.rfft(wc,axis=1))**2
    acc/=len(use); shares=[]
    for i,zc in enumerate(ZS):
        sel=np.abs(z-zc)<0.02; Ek=acc[sel].mean(0); Ek/=Ek.sum(); ut=np.sqrt(2*Ez[sel].mean()); kb=np.sqrt(float(beta(m,zc))/ut)/(2*np.pi)
        col=cmap(i/(len(ZS)-1)); xk=kx/kb
        axs[0,c].loglog(np.maximum(xk,1e-3),np.maximum(Ek,1e-12),color=col,lw=1,label=f'{zc:g}' if c==0 else None)
        axs[1,c].semilogx(np.maximum(xk,1e-3),np.cumsum(Ek),color=col,lw=1)
        shares.append(dict(z=zc,kb_cycles=round(float(kb),2),share_kx_le_half_kb=round(float(Ek[xk<=0.5].sum()),3),share_kx0=round(float(Ek[0]),3)))
    for a in axs[:,c]: a.axvline(0.5,color='k',ls=':',lw=0.8); a.set_xlim(1e-2,30)
    axs[0,c].set_title(lab,fontsize=9); axs[0,c].set_ylim(1e-6,1); axs[1,c].set_ylim(0,1); axs[1,c].set_xlabel('$k_x/k_\\beta(z)$')
    sh=np.array([s['share_kx_le_half_kb'] for s in shares[:5]])
    summary[r].update(kx_shares=shares,share_mean_upper=round(float(sh.mean()),3),share_spread_upper=round(float(sh.max()-sh.min()),3))
axs[0,0].set_ylabel('$E(k_x;z)/\\sum E$'); axs[1,0].set_ylabel('cumulative'); axs[0,0].legend(title='$z$',fontsize=6.5,title_fontsize=7,frameon=False,loc='lower left',ncol=2)
fig.suptitle('$k_x$ spectra of the $y$-mean field at fixed depths, last 20% of each run, scaled by the local Rhines wavenumber',fontsize=10)
fig.savefig(f'{OUT}/kx-collapse.png',dpi=140); plt.close(fig)
json.dump(summary,open(f'{OUT}/summary.json','w'),indent=1); print(json.dumps({k:{kk:v for kk,v in s.items() if kk!='kx_shares'} for k,s in summary.items()},indent=1))
