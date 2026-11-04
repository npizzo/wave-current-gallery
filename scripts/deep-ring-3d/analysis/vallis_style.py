"""Vallis (2017, Fig. 12.3)-style spectra: E(kx,kz) of the y-mean (u,w) at successive EARLY times, linear axes,
unnormalized except by each panel's maximum, a few log-spaced contour levels, light Gaussian smoothing (1 mode).
Rhines dumbbell kappa = k_beta |cos theta|^{1/2}, k_beta=(beta/u~)^{1/2}/2pi with u~=(2 KE)^{1/2} at that time."""
import numpy as np, matplotlib as mpl, os, glob
mpl.use('Agg'); import matplotlib.pyplot as plt
def smooth(E,n=2):   # n passes of the 1-2-1 binomial filter in each direction (no scipy in the Unity env)
    for _ in range(n):
        E=0.25*np.roll(E,1,0)+0.5*E+0.25*np.roll(E,-1,0); E=0.25*np.roll(E,1,1)+0.5*E+0.25*np.roll(E,-1,1)
    return E
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D=f'{ROOT}/data'; OUT=f'{ROOT}/figs/vallis-style'; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.size':8,'font.family':'serif','mathtext.fontset':'cm'})
RUNS=[('s2d-ubeta-k27-lx4','uniform $\\beta$',1.0),('s2d-exp2-k27-lx4','$\\beta\\propto e^{2z}$',1.0),('s2d-exp4-k27-lx4','$\\beta\\propto e^{4z}$',0.5),
      ('s2d-deep-k27-lx4','$\\beta\\propto e^{8z}$ (deep)',0.5),('s2d-none-k27-lx4','no waves',1.0)]
import sys
ZOOM=len(sys.argv)>1 and sys.argv[1]=='zoom'
TIMES=[10,20,30,45,60,90] if ZOOM else [0,10,25,50,100,300]; LIM=22 if ZOOM else 40
def meta(r): return dict(l.strip().split('=',1) for l in open(f'{D}/{r}/meta.txt'))
def beta(m,z):
    s=m.get('shear','deep'); g=float(m.get('gam','8'))
    return 0.5+0*z if s=='ubeta' else (0.5*np.exp(0.17*g)*np.exp(g*z) if s=='exp' else 2*np.exp(8*z))
fig,axs=plt.subplots(len(RUNS),len(TIMES),figsize=(1.9*len(TIMES),1.95*len(RUNS)),gridspec_kw=dict(hspace=0.12,wspace=0.06,left=0.06,right=0.99,bottom=0.04,top=0.95))
for i,(r,lab,H) in enumerate(RUNS):
    m=meta(r); nx,nz,lx=int(m['nx']),int(m['nz']),float(m['lx']); z=np.loadtxt(f'{D}/{r}/zc.txt'); ts=np.genfromtxt(f'{D}/{r}/timeseries.csv',delimiter=',',names=True)
    bl=float(beta(m,z[z>-H]).mean()) if m.get('waves','1')=='1' else None
    for j,t in enumerate(TIMES):
        f=f'{D}/{r}/ymean/t{t:07.2f}.bin'; ax=axs[i,j]
        a=np.fromfile(f,dtype=np.float32).astype(float); u=a[:nx*nz].reshape(nz,nx).T; w=a[nx*nz:].reshape(nz+1,nx).T
        uc=(u+np.roll(u,-1,0))/2; wc=(w[:,:-1]+w[:,1:])/2
        if H<1:
            s=z>-H; zs=z[s]; wn=np.ones(s.sum()); lo=zs<-0.75*H; wn[lo]=np.sin(0.5*np.pi*(zs[lo]+H)/(0.25*H))**2; uc,wc=uc[:,s]*wn,wc[:,s]*wn
        ue=np.concatenate([uc,uc[:,::-1]],1); we=np.concatenate([wc,-wc[:,::-1]],1); L=2*H
        E=np.fft.fftshift(np.abs(np.fft.fft2(ue))**2+np.abs(np.fft.fft2(we))**2)
        kx=np.fft.fftshift(np.fft.fftfreq(nx,lx/nx)); kz=np.fft.fftshift(np.fft.fftfreq(ue.shape[1],1/ue.shape[1]))/L
        sx=(np.abs(kx)<=45); sz=(np.abs(kz)<=45); E=E[np.ix_(sx,sz)]; KX,KZ=kx[sx],kz[sz]
        E=smooth(E); E/=E.max()
        ax.contourf(KX,KZ,E.T,levels=[10**-2.5,10**-2,10**-1.5,10**-1,10**-0.5,1.01],cmap='Greys',extend='neither')
        ax.contour(KX,KZ,E.T,levels=[10**-2,10**-1],colors='k',linewidths=0.4)
        if bl is not None:
            k=np.argmin(abs(ts['t_Te']-t)); ut=np.sqrt(2*ts['KE'][k]); kb=np.sqrt(bl/ut)/(2*np.pi)
            th=np.linspace(0,2*np.pi,721); kk=kb*np.sqrt(np.abs(np.cos(th))); ax.plot(kk*np.cos(th),kk*np.sin(th),color='#c0392b',lw=0.9)
        ax.axhline(0,color='0.6',lw=0.3); ax.axvline(0,color='0.6',lw=0.3)
        ax.set_xlim(-LIM,LIM); ax.set_ylim(-LIM,LIM); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        if i==0: ax.set_title(f'$t={t}\\,T_e$ (27-cycle)',fontsize=8)
        if j==0: ax.set_ylabel(lab+('\n$z>-0.5$' if H<1 else '\nfull depth'),fontsize=8)
fig.suptitle('Energy in $(k_x,k_z)$, $|k_x|,|k_z|\\leq$'+str(LIM)+' cycles; grey levels $10^{-2.5}\\ldots1$ of each panel max; red: Rhines dumbbell',fontsize=9)
fig.savefig(f'{OUT}/vallis-style'+('-zoom' if ZOOM else '')+'.png',dpi=150); print('ok')
