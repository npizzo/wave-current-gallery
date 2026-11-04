"""Final Vallis-style figures (Vallis 2017, Figs 12.3-12.4 layout): spectra on top, u(x,z) below, at three times.
t=0 and t=t_best: spectra averaged over the 8 ensemble seeds (ens-*), field = seed 1.
Late column: the long seed-1 run of the same configuration (s2d-*-k27-lx4), spectrum averaged over 880-920 Te."""
import numpy as np, matplotlib as mpl, os, json
mpl.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D=f'{ROOT}/data'; OUT=f'{ROOT}/figs/ens-vallis'
plt.rcParams.update({'font.size':9,'font.family':'serif','mathtext.fontset':'cm'})
CASES=[('ubeta','uniform $\\beta$',1.0),('deep','deep water, $\\beta_{\\rm eff}=2e^{8z}$',0.5),('none','no waves',1.0)]
BEST=json.load(open(f'{OUT}/summary.json'))['best_time']; LIM=35; TLATE=900
def meta(r): return dict(l.strip().split('=',1) for l in open(f'{D}/{r}/meta.txt'))
def load(r,t):
    m=meta(r); nx,nz=int(m['nx']),int(m['nz']); a=np.fromfile(f'{D}/{r}/ymean/t{t:07.2f}.bin',dtype=np.float32).astype(float)
    return a[:nx*nz].reshape(nz,nx).T, a[nx*nz:].reshape(nz+1,nx).T
def spec(r,t,H):
    m=meta(r); nx,nz,lx=int(m['nx']),int(m['nz']),float(m['lx']); z=np.loadtxt(f'{D}/{r}/zc.txt'); u,w=load(r,t)
    uc=(u+np.roll(u,-1,0))/2; wc=(w[:,:-1]+w[:,1:])/2
    if H<1:
        s=z>-H; zs=z[s]; wn=np.ones(s.sum()); lo=zs<-0.75*H; wn[lo]=np.sin(0.5*np.pi*(zs[lo]+H)/(0.25*H))**2; uc,wc=uc[:,s]*wn,wc[:,s]*wn
    ue=np.concatenate([uc,uc[:,::-1]],1); we=np.concatenate([wc,-wc[:,::-1]],1); L=2*H
    if H<1:   # zero-pad in z (interpolates the spectrum to the same kz spacing as the full-depth runs; no new information)
        pad=np.zeros((ue.shape[0],int(round(ue.shape[1]*(1/H-1))))); ue=np.concatenate([ue,pad],1); we=np.concatenate([we,pad],1); L=2.0
    E=np.fft.fftshift(np.abs(np.fft.fft2(ue))**2+np.abs(np.fft.fft2(we))**2)
    kx=np.fft.fftshift(np.fft.fftfreq(nx,lx/nx)); kz=np.fft.fftshift(np.fft.fftfreq(ue.shape[1],1/ue.shape[1]))/L
    sx=np.abs(kx)<=LIM+2; sz=np.abs(kz)<=LIM+2; return E[np.ix_(sx,sz)],kx[sx],kz[sz]
def mean_spec(runs,times,H):
    E=None
    for r in runs:
        for t in times:
            e,kx,kz=spec(r,t,H); E=e if E is None else E+e
    return E/(len(runs)*len(times)),kx,kz
def smooth(E,n=1):
    for _ in range(n): E=0.25*np.roll(E,1,0)+0.5*E+0.25*np.roll(E,-1,0); E=0.25*np.roll(E,1,1)+0.5*E+0.25*np.roll(E,-1,1)
    return E
def kappaR(r,t,key,H,E,kx,kz):
    """Pizzo curve kappa(theta)=kappa_R|cos theta|^{3/5}, kappa_R=(|beta|/T^{1/3})^{3/5} (eq. cl_rhines_curve).
    The transfer rate is estimated dimensionally as T = u~^3 kappa_e, with u~=(2KE)^{1/2} and kappa_e the
    energy-weighted mean |k| of the plotted spectrum, so the curve meets the Rhines estimate when kappa_e=kappa_R."""
    if key=='none': return None
    ts=np.genfromtxt(f'{D}/{r}/timeseries.csv',delimiter=',',names=True); i=int(np.argmin(abs(ts['t_Te']-t))); z=np.loadtxt(f'{D}/{r}/zc.txt')
    b=0.5 if key=='ubeta' else float((2*np.exp(8*z[z>-H])).mean()); ut=np.sqrt(2*ts['KE'][i])
    K=2*np.pi*np.hypot(*np.meshgrid(kx,kz,indexing='ij')); ke=float((K*E).sum()/E.sum())
    return (b/(ut*ke**(1/3)))**0.6/(2*np.pi)
for key,lab,H in CASES:
    ens=[f'ens-{key}-k27-s{s}' for s in range(1,9)]; long=f's2d-{key}-k27-lx4'; tb=BEST.get(key,BEST['ubeta'])
    cols=[(ens,[0],ens[0],0,'$t=0$'),(ens,[tb],ens[0],tb,f'$t={tb}\\,T_e$'),([long],list(range(TLATE-20,TLATE+21,5)),long,TLATE,f'$t={TLATE}\\,T_e$')]
    fig=plt.figure(figsize=(9.6,6.6)); gs=fig.add_gridspec(2,3,height_ratios=[1,0.55],hspace=0.2,wspace=0.16,left=0.08,right=0.86,bottom=0.07,top=0.88)
    for j,(runs,times,fr,tf,ttl) in enumerate(cols):
        E,kx,kz=mean_spec(runs,times,H); kR=kappaR(fr,tf,key,H,E,kx,kz);  E=smooth(E); E/=E.max()
        ax=fig.add_subplot(gs[0,j])
        late=(j==2)
        if late:   # zoom to the energy-containing range (and the curve), continuous log grey scale
            KK=np.hypot(*np.meshgrid(kx,kz,indexing='ij')); o=np.argsort(KK.ravel()); c=np.cumsum(E.ravel()[o]); k80=KK.ravel()[o][np.searchsorted(c,0.80*c[-1])]
            lim=float(np.ceil(1.3*kR)) if kR else 8.0
            im3=ax.pcolormesh(kx,kz,np.log10(E+1e-12).T,cmap='Greys',vmin=-3,vmax=0,shading='nearest',rasterized=True)
            ax.contour(kx,kz,E.T,levels=[10**-2,10**-1],colors='k',linewidths=[0.4,0.8]); print(key,'late zoom',lim,'k80',k80,'kR',kR)
        else:
            lim=LIM
            ax.contourf(kx,kz,E.T,levels=[0.05,0.15,0.35,0.65,1.01],colors=['#e6e6e6','#b3b3b3','#6e6e6e','#262626'])
            ax.contour(kx,kz,E.T,levels=[0.05,0.35],colors='k',linewidths=[0.4,0.8])
        if kR is not None:
            th=np.linspace(0,2*np.pi,1441); kk=kR*np.abs(np.cos(th))**0.6; ax.plot(kk*np.cos(th),kk*np.sin(th),color='#c0392b',lw=1.1,ls='--')
        ax.axhline(0,color='0.5',lw=0.4); ax.axvline(0,color='0.5',lw=0.4); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect('equal')
        ax.set_title(ttl+(' (8 seeds)' if len(runs)>1 else ' (seed 1, 800–1000)\nzoomed; $\\log_{10}E/E_{\\max}$ from $-3$ to 0'),fontsize=9); ax.set_xlabel('$k_x/2\\pi$')
        if j==0: ax.set_ylabel('$k_z/2\\pi$')
        elif not late: ax.set_yticklabels([])
        else: ax.yaxis.tick_right()
        m=meta(fr); nx,nz,lx,u0=int(m['nx']),int(m['nz']),float(m['lx']),float(m['u0']); z=np.loadtxt(f'{D}/{fr}/zc.txt'); u,_=load(fr,tf)
        x=np.arange(nx)*lx/nx; sel=x<=2.0
        ax2=fig.add_subplot(gs[1,j]); im=ax2.pcolormesh(x[sel],z,(u[sel]/u0).T,cmap='RdBu_r',vmin=-3,vmax=3,shading='auto',rasterized=True)
        ax2.set_aspect('equal'); ax2.set_xlabel('$x$'); ax2.set_xticks([0,0.5,1,1.5,2]); ax2.set_xticklabels(['0','0.5','1','1.5','2'])
        if j==0: ax2.set_ylabel('$z$')
        else: ax2.set_yticklabels([])
        if H<1: ax2.axhline(-H,color='k',lw=0.6,ls=':')
    fig.colorbar(im,cax=fig.add_axes([0.875,0.07,0.012,0.22]),extend='both').set_label('$u/u_0$')
    fig.colorbar(im3,cax=fig.add_axes([0.915,0.47,0.01,0.33]),ticks=[-3,-2,-1,0]).set_label('$\\log_{10}E/E_{\\max}$ (top right)',fontsize=8)
    fig.suptitle(f'{lab}: energy in $(k_x,k_z)$'+(' over $z>-0.5$' if H<1 else '')+' (top; shades 0.05, 0.15, 0.35, 0.65 of max unless noted) and $u(x,z)/u_0$, seed 1, $0<x<2$ (bottom)\n'+('red dashed: $\\kappa=\\kappa_R|\\cos\\theta|^{3/5}$, $\\kappa_R=(|\\beta|/\\mathcal{T}^{1/3})^{3/5}$, $\\mathcal{T}=\\tilde u^3\\kappa_e$; ' if key!='none' else '')+'$T_e$ of the 27-cycle ring',fontsize=9)
    fig.savefig(f'{OUT}/vallis2-{key}.png',dpi=170); plt.close(fig)
print('ok')
