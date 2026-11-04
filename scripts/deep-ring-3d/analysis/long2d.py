"""Analysis of the long 2D ring runs (long2d.cases). Runs on Unity; writes PNGs + JSON to figs/long2d/."""
import numpy as np, matplotlib as mpl, os, json, glob, sys
mpl.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D=f'{ROOT}/data'; OUT=f'{ROOT}/figs/long2d'; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.size':9,'font.family':'serif','mathtext.fontset':'cm'})
RUNS=[('s2d-deep-rb1-s1-t1000','9-cycle ring, deep, $L_x=1$'),('s2d-deep-rb1-s1-lx4-t1000','9-cycle ring, deep, $L_x=4$'),
      ('s2d-deep-k27-lx4','27-cycle ring, deep, $L_x=4$'),('s2d-ubeta-k27-lx4','27-cycle ring, uniform $\\beta$, $L_x=4$'),('s2d-none-k27-lx4','27-cycle ring, no waves, $L_x=4$')]
TEREF=1/(0.5/(2*np.pi*9)**2*2*np.pi*9)          # Te of the 9-cycle rb=1 ring (=113.1)
def meta(r): return dict(l.strip().split('=',1) for l in open(f'{D}/{r}/meta.txt'))
def frames(r): return sorted(glob.glob(f'{D}/{r}/ymean/t*.bin'),key=lambda f: float(os.path.basename(f)[1:-4]))
def beta_of(m,z): return 2*np.exp(8*z) if m.get('shear','deep')=='deep' else 0.5+0*z
def load(r,f):
    m=meta(r); nx,nz=int(m['nx']),int(m['nz']); a=np.fromfile(f,dtype=np.float32).astype(float)
    return a[:nx*nz].reshape(nz,nx).T, a[nx*nz:].reshape(nz+1,nx).T
def spec(r,f,H):
    m=meta(r); nx,nz,lx=int(m['nx']),int(m['nz']),float(m.get('lx','1')); z=np.loadtxt(f'{D}/{r}/zc.txt'); u,w=load(r,f)
    uc=(u+np.roll(u,-1,0))/2; wc=(w[:,:-1]+w[:,1:])/2
    if H>=1: us,ws,L=uc,wc,2.
    else:
        sel=z>-H; zs=z[sel]; win=np.ones(sel.sum()); low=zs<-0.75*H; win[low]=np.sin(0.5*np.pi*(zs[low]+H)/(0.25*H))**2
        us,ws,L=uc[:,sel]*win,wc[:,sel]*win,2*H
    ue=np.concatenate([us,us[:,::-1]],1); we=np.concatenate([ws,-ws[:,::-1]],1)
    E=(np.abs(np.fft.fft2(ue))**2+np.abs(np.fft.fft2(we))**2)/nx**2*lx
    kx=np.fft.fftfreq(nx,lx/nx)[:,None]+0*E; kz=np.fft.fftfreq(E.shape[1],1/E.shape[1])[None,:]/L+0*E
    return E,kx,kz
summary={}
# ---- jet fraction vs physical time
fig,ax=plt.subplots(figsize=(6.4,3.4)); cols=['#2a6fb0','#2a6fb0','#1b7f5a','#7a3e9d','#d9822b']; ls=['--','-','-','-','-']
for (r,lab),c,l in zip(RUNS,cols,ls):
    m=meta(r); ts=np.genfromtxt(f'{D}/{r}/timeseries.csv',delimiter=',',names=True); T=float(m['Te'])/TEREF
    ax.plot(ts['t_Te']*T,ts['KEjet']/ts['KE'],color=c,ls=l,lw=1.1,label=lab)
    n=len(ts); summary[r]=dict(final_jet_frac=float(ts['KEjet'][-1]/ts['KE'][-1]),jet_frac_last20pct=float(np.mean((ts['KEjet']/ts['KE'])[int(0.8*n):])),
        KE_final_over_KE0=float(ts['KE'][-1]/ts['KE'][0]),umax_final_over_u0=float(ts['umax'][-1]/float(m['u0'])))
ax.set_xlabel('$t$ / ($T_e$ of the 9-cycle ring)'); ax.set_ylabel('jet fraction $\\frac{1}{2}\\langle U_L^2\\rangle$/KE'); ax.set_xlim(0,1000); ax.set_ylim(0,1); ax.grid(color='0.9',lw=0.5)
ax.legend(fontsize=7.5,frameon=False,loc='upper left'); fig.tight_layout(); fig.savefig(f'{OUT}/jet-fraction.png',dpi=150); plt.close(fig)
# ---- fields at four physical times
for r,lab in RUNS:
    m=meta(r); nx,nz,lx,u0=int(m['nx']),int(m['nz']),float(m.get('lx','1')),float(m['u0']); T=float(m['Te'])/TEREF
    z=np.loadtxt(f'{D}/{r}/zc.txt'); x=np.arange(nx)*lx/nx; p=np.loadtxt(f'{D}/{r}/profiles.csv',delimiter=',',skiprows=1); fs=frames(r)
    tf=np.array([float(os.path.basename(f)[1:-4]) for f in fs])*T
    fig=plt.figure(figsize=(7.2,6.4) if lx>1 else (7.2,3.2)); targets=[0,100,300,1000]
    if lx>1: gs=fig.add_gridspec(4,2,width_ratios=[4,0.55],wspace=0.06,hspace=0.18,left=0.08,right=0.9,bottom=0.07,top=0.94)
    else: gs=fig.add_gridspec(2,5,height_ratios=[1,0.6],hspace=0.35,wspace=0.12,left=0.07,right=0.9,bottom=0.12,top=0.88)
    for i,tt in enumerate(targets):
        j=int(np.argmin(abs(tf-tt))); u,_=load(r,fs[j]); k=int(np.argmin(abs(p[:,0]*T-tt)))
        ax=fig.add_subplot(gs[i,0] if lx>1 else gs[0,i]); im=ax.pcolormesh(x,z,(u/u0).T,cmap='RdBu_r',vmin=-3,vmax=3,shading='auto',rasterized=True); ax.set_aspect('equal')
        ax.text(0.01,0.05,f'$t={tf[j]:.0f}$',transform=ax.transAxes,fontsize=8,bbox=dict(fc='white',ec='none',alpha=0.8,pad=1))
        if lx>1:
            ax2=fig.add_subplot(gs[i,1]); ax2.plot(p[k,1:1+nz]/u0,z,'k',lw=1); ax2.axvline(0,color='0.7',lw=0.6); ax2.set_xlim(-3,3); ax2.set_yticklabels([])
            if i<3: ax.set_xticklabels([]); ax2.set_xticklabels([])
        else:
            ax2=fig.add_subplot(gs[1,i]); ax2.plot(z,p[k,1:1+nz]/u0,'k',lw=1); ax2.axhline(0,color='0.7',lw=0.6); ax2.set_ylim(-3,3)
    fig.suptitle(f'{lab}: $u(x,z)/u_0$ and $U_L/u_0$ ($t$ in 9-cycle $T_e$)',fontsize=9.5)
    fig.colorbar(im,cax=fig.add_axes([0.92,0.3,0.012,0.4]),extend='both'); fig.savefig(f'{OUT}/fields-{r}.png',dpi=130); plt.close(fig)
# ---- spectra over the last 20% of each run
SP=[('s2d-deep-rb1-s1-lx4-t1000',0.5),('s2d-deep-k27-lx4',0.5),('s2d-ubeta-k27-lx4',1.0),('s2d-none-k27-lx4',1.0)]
fig,axs=plt.subplots(2,4,figsize=(12,6.4),gridspec_kw=dict(hspace=0.3,wspace=0.28,left=0.05,right=0.93,bottom=0.08,top=0.9))
for c,(r,H) in enumerate(SP):
    m=meta(r); nz=int(m['nz']); z=np.loadtxt(f'{D}/{r}/zc.txt'); fs=frames(r); use=fs[int(0.8*len(fs)):]
    E=None
    for f in use:
        e,kx,kz=spec(r,f,H); E=e if E is None else E+e
    E/=len(use); K=np.hypot(kx,kz); th=np.arctan2(np.abs(kz),np.abs(kx))
    p=np.loadtxt(f'{D}/{r}/profiles.csv',delimiter=',',skiprows=1); p=p[int(0.8*len(p)):]; sel=z>-H
    ut=np.sqrt(2*(p[:,1+nz:1+2*nz]+p[:,1+2*nz:])[:,sel].mean()); beta=float(beta_of(m,z[sel]).mean()); kb=np.sqrt(beta/ut)/(2*np.pi)
    C=np.full_like(E,np.nan); Kb=np.round(K)
    for b in np.unique(Kb):
        if b==0: continue
        mk=Kb==b; mm=mk&(kx!=0)
        if mm.sum()>2: C[mk]=E[mk]/E[mm].mean()
    sh=np.fft.fftshift; kxs=sh(kx[:,0]); kzs=sh(kz[0]); tt=np.linspace(0,2*np.pi,721); kk=kb*np.sqrt(np.abs(np.cos(tt)))
    a0,a1=axs[0,c],axs[1,c]
    im=a0.pcolormesh(kxs,kzs,np.log10(sh(E)/E.max()+1e-30).T,cmap='Blues',vmin=-6,vmax=0,shading='nearest',rasterized=True)
    im2=a1.pcolormesh(kxs,kzs,np.log2(sh(C)).T,cmap='RdBu_r',vmin=-3,vmax=3,shading='nearest',rasterized=True)
    lim=max(20,2.5*kb) if 'none' not in r else 40
    for a,col in [(a0,'#c0392b'),(a1,'k')]:
        if 'none' not in r: a.plot(kk*np.cos(tt),kk*np.sin(tt),color=col,lw=1)
        a.set_xlim(-lim,lim); a.set_ylim(-lim,lim); a.set_aspect('equal'); a.set_xlabel('$k_x/2\\pi$')
    name=dict(RUNS)[r]; a0.set_title(name+('\n$z>-0.5$' if H<1 else '\nfull depth')+', last 20%',fontsize=8.5); a1.set_title(r'$\log_2[E/\langle E\rangle_{|k|}]$',fontsize=8.5)
    if c==0: a0.set_ylabel('$k_z/2\\pi$'); a1.set_ylabel('$k_z/2\\pi$')
    d={}
    for k1,k2 in [(1,3),(3,6),(6,10),(10,15)]:
        ann=(K>=k1)&(K<k2)&(kx!=0)&(np.abs(kz)>0); ins=ann&(K<kb*np.sqrt(np.cos(th))); outs=ann&~(K<kb*np.sqrt(np.cos(th)))
        d[f'{k1}-{k2}']=round(float(E[ins].mean()/E[outs].mean()),3) if ins.sum()>3 and outs.sum()>3 else None
    KX,KZ=kx,kz; m0=K>0
    summary[r].update(spectrum=dict(layer=H,beta=round(beta,3),utilde_over_u0=round(ut/float(m['u0']),3),rhines_cycles=round(kb,2),
        k_centroid=round(float((K*E).sum()/E.sum()),2),kx0_share=round(float(E[kx==0].sum()/E.sum()),3),
        anisotropy=round(float(((KZ**2-KX**2)*E)[m0].sum()/((KZ**2+KX**2)*E)[m0].sum()),3),inside_over_outside=d))
fig.colorbar(im,cax=fig.add_axes([0.945,0.55,0.01,0.33])).set_label('relative to max')
fig.colorbar(im2,cax=fig.add_axes([0.945,0.08,0.01,0.33]),extend='both').set_label('$\\log_2$ ratio to annulus mean')
fig.savefig(f'{OUT}/spectra.png',dpi=130); plt.close(fig)
json.dump(summary,open(f'{OUT}/summary.json','w'),indent=1); print(json.dumps(summary,indent=1))
