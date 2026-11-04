import numpy as np, matplotlib as mpl, sys, os
mpl.use('Agg'); import matplotlib.pyplot as plt
D=os.path.expanduser('~/drfig'); OUT=sys.argv[1]
plt.rcParams.update({'font.size':9,'axes.titlesize':9,'axes.labelsize':9,'font.family':'serif','mathtext.fontset':'cm'})
def meta(r): return dict(l.strip().split('=',1) for l in open(f'{D}/{r}/meta.txt'))
def ymean(r,t,nx=256,nz=256):
    a=np.fromfile(f'{D}/{r}/ymean/t{t:07.2f}.bin',dtype=np.float32).astype(float)
    return a[:nx*nz].reshape(nz,nx).T, a[nx*nz:].reshape(nz+1,nx).T
def profiles(r,nz=256):
    p=np.loadtxt(f'{D}/{r}/profiles.csv',delimiter=',',skiprows=1); return p[:,0],p[:,1:1+nz]
z=np.loadtxt(f'{D}/d3-deep-rb1-e1e-2-s1/zc.txt'); nx=nz=256; x=np.arange(nx)/nx
runs=[('d3-deep-rb1-e1e-2-s1','3D CL, deep Stokes drift'),('d3-none-rb1-e1e-2-s1','3D, no waves')]
times=[0,20,50,200]; u0=float(meta(runs[0][0])['u0'])
# ---------- Fig 3 ----------
vmax=3.0
fig=plt.figure(figsize=(7.2,3.9)); gs=fig.add_gridspec(2,6,width_ratios=[1,1,1,1,0.7,0.06],wspace=0.12,hspace=0.28,left=0.13,right=0.92,bottom=0.12,top=0.9)
cols=plt.cm.Greys(np.linspace(0.35,1,len(times)))
for row,(r,lab) in enumerate(runs):
    tp,UL=profiles(r)
    for c,t in enumerate(times):
        ax=fig.add_subplot(gs[row,c]); u,_=ymean(r,t)
        im=ax.pcolormesh(x,z,(u/u0).T,cmap='RdBu_r',vmin=-vmax,vmax=vmax,shading='auto',rasterized=True)
        ax.set_aspect('equal'); ax.set_xticks([0,0.5,1]); ax.set_xticklabels(['0','0.5','1']); ax.set_yticks([-1,-0.5,0]); ax.set_yticklabels(['$-1$','$-0.5$','0'])
        if row==0: ax.set_title(f'$t={t}\\,T_e$')
        if c==0: ax.set_ylabel('$z$'); ax.text(-0.62,0.5,lab.replace(', ',',\n'),transform=ax.transAxes,rotation=90,va='center',ha='center',fontsize=8.5)
        else: ax.set_yticklabels([])
        if row==1: ax.set_xlabel('$x$')
        else: ax.set_xticklabels([])
    ax=fig.add_subplot(gs[row,4])
    for c,t in enumerate(times):
        i=np.argmin(abs(tp-t)); ax.plot(UL[i]/u0,z,color=cols[c],lw=1.2,label=f'{t}')
    ax.axvline(0,color='0.7',lw=0.6); ax.set_ylim(-1,0); ax.set_xlim(-2.2,2.2); ax.set_yticklabels([])
    if row==0: ax.set_title('$U_L(z)/u_0$')
    if row==1: ax.legend(title='$t/T_e$',fontsize=7,title_fontsize=7,frameon=True,framealpha=0.95,edgecolor='none',loc='lower right',handlelength=1.0,borderpad=0.2)
    if row==1: ax.set_xlabel('$U_L/u_0$')
    else: ax.set_xticklabels([])
cax=fig.add_subplot(gs[:,5]); cb=fig.colorbar(im,cax=cax,extend='both'); cb.set_label(r'$\overline{u}^{\,y}/u_0$')
fig.savefig(f'{OUT}/fig3-deep-ring-3d.pdf',dpi=300); fig.savefig(f'{OUT}/fig3-deep-ring-3d.png',dpi=200); plt.close(fig)
# ---------- Fig 4 ----------
def fields(r,t):
    u,w=ymean(r,t); uc=(u+np.roll(u,-1,0))/2; wc=(w[:,:-1]+w[:,1:])/2
    zeta=np.gradient(wc,1/nx,axis=0)-np.gradient(uc,1/nz,axis=1)
    ue=np.concatenate([uc,uc[:,::-1]],1); we=np.concatenate([wc,-wc[:,::-1]],1)
    E=np.fft.fftshift(np.abs(np.fft.fft2(ue))**2+np.abs(np.fft.fft2(we))**2)
    kx=np.fft.fftshift(np.fft.fftfreq(nx,1/nx)); kz=np.fft.fftshift(np.fft.fftfreq(2*nz,1/(2*nz)))/2
    return zeta,E,kx,kz
tf=200
fig,axs=plt.subplots(2,2,figsize=(7.2,6.4),gridspec_kw=dict(hspace=0.3,wspace=0.3,left=0.08,right=0.86,bottom=0.07,top=0.94))
Z=[];S=[]
for r,_ in runs: Z.append(fields(r,tf))
zmax=np.percentile(np.abs(Z[0][0]),99.5)
Emax=max(f[1].max() for f in Z)
for c,((r,lab),(zeta,E,kx,kz)) in enumerate(zip(runs,Z)):
    m=meta(r); ts=np.genfromtxt(f'{D}/{r}/timeseries.csv',delimiter=',',names=True); i=np.argmin(abs(ts['t_Te']-tf))
    urms=np.sqrt(2*ts['KE'][i])
    ax=axs[0,c]; im=ax.pcolormesh(x,z,zeta.T,cmap='RdBu_r',vmin=-zmax,vmax=zmax,shading='auto',rasterized=True)
    ax.set_aspect('equal'); ax.set_title(f'{lab}: '+r'$\overline{\zeta}^{\,y}=\overline{w}_x-\overline{u}_z$'+f', $t={tf}T_e$',fontsize=8.5); ax.set_xlabel('$x$'); ax.set_ylabel('$z$')
    if c==1:
        cax=fig.add_axes([0.875,0.56,0.015,0.36]); cb=fig.colorbar(im,cax=cax,extend='both'); cb.set_label(r'$\overline{\zeta}^{\,y}$')
    ax=axs[1,c]; L=np.log10(E/Emax+1e-30)
    im2=ax.pcolormesh(kx,kz,L.T,cmap='Blues',vmin=-6,vmax=0,shading='auto',rasterized=True)
    ax.set_xlim(-30,30); ax.set_ylim(-30,30); ax.set_aspect('equal')
    ax.set_xlabel('$k_x/2\\pi$'); ax.set_ylabel('$k_z/2\\pi$'); ax.set_title(r'$\log_{10}\,E(k_x,k_z)$ of $(\overline{u}^{\,y},\overline{w}^{\,y})$',fontsize=8.5)
    if int(m['waves'])==1:
        th=np.linspace(0,2*np.pi,721); kb=np.sqrt(0.5/urms)/(2*np.pi)
        kap=kb*np.sqrt(np.abs(np.cos(th))); ax.plot(kap*np.cos(th),kap*np.sin(th),color='#c0392b',lw=1.3)
        ax.text(0.03,0.95,r'$\kappa(\theta)=(\beta_{\rm ref}/\tilde u)^{1/2}|\cos\theta|^{1/2}$',transform=ax.transAxes,color='#c0392b',fontsize=7.5,va='top')
    if c==1:
        cax=fig.add_axes([0.875,0.07,0.015,0.36]); cb=fig.colorbar(im2,cax=cax); cb.set_label('relative to max')
fig.savefig(f'{OUT}/fig4-deep-ring-3d.pdf',dpi=300); fig.savefig(f'{OUT}/fig4-deep-ring-3d.png',dpi=200); plt.close(fig)
print('urms/u0 final deep, none:',[np.sqrt(2*np.genfromtxt(f'{D}/{r}/timeseries.csv',delimiter=',',names=True)['KE'][-1])/u0 for r,_ in runs], 'kb cycles', np.sqrt(0.5/np.sqrt(2*np.genfromtxt(f'{D}/{runs[0][0]}/timeseries.csv',delimiter=',',names=True)['KE'][-1]))/(2*np.pi))
