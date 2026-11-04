"""Seed-ensemble Vallis-style spectra. E(kx,kz) of the y-mean (u,w) (deep: z>-0.5, Hann-tapered bottom quarter;
uniform beta / no waves: full depth), averaged over 8 seeds at each output time (every 1 Te of the 27-cycle ring)."""
import numpy as np, matplotlib as mpl, os, json
mpl.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D=f'{ROOT}/data'; OUT=f'{ROOT}/figs/ens-vallis'; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.size':9,'font.family':'serif','mathtext.fontset':'cm'})
CASES=[('ubeta','uniform $\\beta$',1.0),('deep','deep, $\\beta_{\\rm eff}=2e^{8z}$ ($z>-0.5$)',0.5),('none','no waves',1.0)]
SEEDS=range(1,9); TIMES=list(range(0,101)); LIM=22
def meta(r): return dict(l.strip().split('=',1) for l in open(f'{D}/{r}/meta.txt'))
def spec(r,t,H,z):
    m=meta(r); nx,nz,lx=int(m['nx']),int(m['nz']),float(m['lx'])
    a=np.fromfile(f'{D}/{r}/ymean/t{t:07.2f}.bin',dtype=np.float32).astype(float); u=a[:nx*nz].reshape(nz,nx).T; w=a[nx*nz:].reshape(nz+1,nx).T
    uc=(u+np.roll(u,-1,0))/2; wc=(w[:,:-1]+w[:,1:])/2
    if H<1:
        s=z>-H; zs=z[s]; wn=np.ones(s.sum()); lo=zs<-0.75*H; wn[lo]=np.sin(0.5*np.pi*(zs[lo]+H)/(0.25*H))**2; uc,wc=uc[:,s]*wn,wc[:,s]*wn
    ue=np.concatenate([uc,uc[:,::-1]],1); we=np.concatenate([wc,-wc[:,::-1]],1)
    E=np.fft.fftshift(np.abs(np.fft.fft2(ue))**2+np.abs(np.fft.fft2(we))**2)
    kx=np.fft.fftshift(np.fft.fftfreq(nx,lx/nx)); kz=np.fft.fftshift(np.fft.fftfreq(ue.shape[1],1/ue.shape[1]))/(2*H)
    sx=np.abs(kx)<=LIM+2; sz=np.abs(kz)<=LIM+2
    return E[np.ix_(sx,sz)],kx[sx],kz[sz]
def smooth(E,n=1):
    for _ in range(n): E=0.25*np.roll(E,1,0)+0.5*E+0.25*np.roll(E,-1,0); E=0.25*np.roll(E,1,1)+0.5*E+0.25*np.roll(E,-1,1)
    return E
res={}; ENS={}
for key,lab,H in CASES:
    runs=[f'ens-{key}-k27-s{s}' for s in SEEDS if os.path.exists(f'{D}/ens-{key}-k27-s{s}/completed.txt')]
    m=meta(runs[0]); z=np.loadtxt(f'{D}/{runs[0]}/zc.txt')
    beta=0.5 if key=='ubeta' else (float((2*np.exp(8*z[z>-H])).mean()) if key=='deep' else None)
    KE=np.mean([np.genfromtxt(f'{D}/{r}/timeseries.csv',delimiter=',',names=True)['KE'][:101] for r in runs],0)
    ENS[key]={}; contrast=[]
    for t in TIMES:
        E=None
        for r in runs:
            e,kx,kz=spec(r,t,H,z); E=e if E is None else E+e
        E/=len(runs); ENS[key][t]=E
        kb=np.sqrt((beta if beta else 0.5*np.exp(8*-0.17))/np.sqrt(2*KE[t]))/(2*np.pi)
        KX,KZ=np.meshgrid(kx,kz,indexing='ij'); K=np.hypot(KX,KZ); th=np.arctan2(np.abs(KZ),np.abs(KX)); kap=kb*np.sqrt(np.cos(th))
        sec=(th<np.pi/4)&(np.abs(KZ)>0.6)&(np.abs(KX)>0.3)
        ins=sec&(K<kap)&(K>0.3*kb); out=sec&(K>=kap)&(K<2*kap)
        contrast.append(float(E[ins].mean()/E[out].mean()))
    res[key]=dict(runs=len(runs),kb_cycles_t0=round(float(np.sqrt((beta if beta else 0.5)/np.sqrt(2*KE[0]))/(2*np.pi)),2),
                  contrast=[round(c,4) for c in contrast])
    ENS[key]['kb']=lambda t,b=beta,KE=KE: np.sqrt((b if b else 0.5)/np.sqrt(2*KE[t]))/(2*np.pi)
    ENS[key]['grid']=(kx,kz)
def panel(ax,key,t,show_db=True):
    kx,kz=ENS[key]['grid']; E=smooth(ENS[key][t]); E=E/E.max()
    ax.contourf(kx,kz,E.T,levels=[10**-2,10**-1.5,10**-1,10**-0.5,1.01],colors=['#eeeeee','#c8c8c8','#8c8c8c','#3a3a3a'])
    ax.contour(kx,kz,E.T,levels=[10**-2,10**-1],colors='k',linewidths=[0.4,0.7])
    if show_db and key!='none':
        kb=ENS[key]['kb'](t); th=np.linspace(0,2*np.pi,721); kk=kb*np.sqrt(np.abs(np.cos(th))); ax.plot(kk*np.cos(th),kk*np.sin(th),color='#c0392b',lw=1,ls='--')
    ax.axhline(0,color='0.5',lw=0.4); ax.axvline(0,color='0.5',lw=0.4)
    ax.set_xlim(-LIM,LIM); ax.set_ylim(-LIM,LIM); ax.set_aspect('equal'); ax.set_xticks([-20,-10,0,10,20]); ax.set_yticks([-20,-10,0,10,20])
# full grid
TS=[0,10,15,20,25,30,40,60]
fig,axs=plt.subplots(3,len(TS),figsize=(2.1*len(TS),6.9),gridspec_kw=dict(hspace=0.15,wspace=0.08,left=0.06,right=0.99,bottom=0.06,top=0.93))
for i,(key,lab,H) in enumerate(CASES):
    for j,t in enumerate(TS):
        ax=axs[i,j]; panel(ax,key,t)
        if i==0: ax.set_title(f'$t={t}\\,T_e$')
        if j==0: ax.set_ylabel(lab+'\n$k_z/2\\pi$',fontsize=8)
        else: ax.set_yticklabels([])
        if i<2: ax.set_xticklabels([])
        else: ax.set_xlabel('$k_x/2\\pi$')
fig.suptitle('Energy in $(k_x,k_z)$, mean of 8 seeds; shades at $10^{-2},10^{-1.5},10^{-1},10^{-0.5}$ of the maximum; red dashed: Rhines dumbbell',fontsize=9.5)
fig.savefig(f'{OUT}/ens-grid.png',dpi=150); plt.close(fig)
# contrast vs time
fig,ax=plt.subplots(figsize=(5.2,3.0))
for (key,lab,H),c in zip(CASES,['#7a3e9d','#2a6fb0','#d9822b']):
    ax.semilogy(TIMES,res[key]['contrast'],color=c,lw=1.3,label=lab)
ax.axhline(1,color='0.5',lw=0.6); ax.set_xlabel('$t/T_e$ (27-cycle ring)'); ax.set_ylabel('energy inside / just outside dumbbell'); ax.legend(fontsize=7.5,frameon=False); ax.grid(color='0.9',lw=0.5)
fig.tight_layout(); fig.savefig(f'{OUT}/ens-contrast.png',dpi=150); plt.close(fig)
best={k:int(np.argmin(res[k]['contrast'][5:]))+5 for k in ('ubeta','deep')}
res['best_time']=best
# Vallis-style figure per case: spectra (8-seed mean) on top, u(x,z) of seed 1 below, at three times
def field(key,t,seed=1):
    r=f'ens-{key}-k27-s{seed}'; m=meta(r); nx,nz,lx,u0=int(m['nx']),int(m['nz']),float(m['lx']),float(m['u0'])
    a=np.fromfile(f'{D}/{r}/ymean/t{t:07.2f}.bin',dtype=np.float32); u=a[:nx*nz].reshape(nz,nx); z=np.loadtxt(f'{D}/{r}/zc.txt')
    return np.arange(nx)*lx/nx,z,u/u0
for key,lab,H in CASES:
    tb=best.get(key,best['ubeta']); ts=[0,tb,100]
    fig=plt.figure(figsize=(9.6,6.2)); gs=fig.add_gridspec(2,3,height_ratios=[1,0.55],hspace=0.18,wspace=0.08,left=0.07,right=0.92,bottom=0.08,top=0.9)
    for j,t in enumerate(ts):
        ax=fig.add_subplot(gs[0,j]); panel(ax,key,t); ax.set_title(f'$t={t}\\,T_e$'); ax.set_xlabel('$k_x/2\\pi$')
        if j==0: ax.set_ylabel('$k_z/2\\pi$')
        else: ax.set_yticklabels([])
        x,z,u=field(key,t); sel=x<=2.0
        ax2=fig.add_subplot(gs[1,j]); im=ax2.pcolormesh(x[sel],z,u[:,sel],cmap='RdBu_r',vmin=-3,vmax=3,shading='auto',rasterized=True)
        ax2.set_aspect('equal'); ax2.set_xlabel('$x$')
        if j==0: ax2.set_ylabel('$z$')
        else: ax2.set_yticklabels([])
        if H<1: ax2.axhline(-H,color='k',lw=0.6,ls=':')
    fig.colorbar(im,cax=fig.add_axes([0.935,0.08,0.012,0.22]),extend='both').set_label('$u/u_0$')
    fig.suptitle(f'{lab}. Top: energy in $(k_x,k_z)$, mean of 8 seeds'+(' (spectra over $z>-0.5$)' if H<1 else '')+'; red dashed: Rhines dumbbell. Bottom: $u(x,z)/u_0$, seed 1, $0<x<2$',fontsize=9)
    fig.savefig(f'{OUT}/vallis-{key}.png',dpi=170); plt.close(fig)
json.dump(res,open(f'{OUT}/summary.json','w'),indent=1); print(json.dumps({k:(v if k=='best_time' else {kk:vv for kk,vv in v.items() if kk!='contrast'}) for k,v in res.items()},indent=1))
for key in ('ubeta','deep','none'): print(key,'contrast at t=10,20,30,40,60,100:',[res[key]['contrast'][t] for t in (10,20,30,40,60,100)])
