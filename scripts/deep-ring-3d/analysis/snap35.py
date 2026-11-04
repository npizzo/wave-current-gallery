"""u/u0 at t=35 Te (27-cycle ring), seed 1, full Lx=4 box, deep vs uniform beta, with U_L(z); band diagnostics over 8 seeds."""
import numpy as np, matplotlib as mpl, os, sys
mpl.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D=f'{ROOT}/data'; OUT=f'{ROOT}/figs/ens-vallis'
plt.rcParams.update({'font.size':9,'font.family':'serif','mathtext.fontset':'cm'})
T=int(sys.argv[1]) if len(sys.argv)>1 else 35
CASES=[('deep','deep water, $\\beta_{\\rm eff}=2e^{8z}$'),('ubeta','uniform $\\beta$'),('none','no waves')]
fig=plt.figure(figsize=(11,7.2)); gs=fig.add_gridspec(3,2,width_ratios=[4,0.5],hspace=0.35,wspace=0.05,left=0.06,right=0.9,bottom=0.07,top=0.92)
for i,(key,lab) in enumerate(CASES):
    r=f'ens-{key}-k27-s1'; m=dict(l.strip().split('=',1) for l in open(f'{D}/{r}/meta.txt')); nx,nz,lx,u0=int(m['nx']),int(m['nz']),float(m['lx']),float(m['u0'])
    z=np.loadtxt(f'{D}/{r}/zc.txt'); a=np.fromfile(f'{D}/{r}/ymean/t{T:07.2f}.bin',dtype=np.float32).astype(float); u=a[:nx*nz].reshape(nz,nx)
    ax=fig.add_subplot(gs[i,0]); im=ax.pcolormesh(np.arange(nx)*lx/nx,z,u/u0,cmap='RdBu_r',vmin=-3,vmax=3,shading='auto',rasterized=True); ax.set_aspect('equal')
    ax.set_ylabel('$z$'); ax.set_title(f'{lab}: $u/u_0$ at $t={T}\\,T_e$ (seed 1)',fontsize=9)
    if i==2: ax.set_xlabel('$x$')
    fr=[];lo=[]
    for s in range(1,9):
        rr=f'ens-{key}-k27-s{s}'; b=np.fromfile(f'{D}/{rr}/ymean/t{T:07.2f}.bin',dtype=np.float32).astype(float); uu=b[:nx*nz].reshape(nz,nx)
        F=np.abs(np.fft.rfft(uu,axis=1))**2; F[:,1:]*=2; kx=np.fft.rfftfreq(nx,lx/nx)
        fr.append(F[:,0].sum()/F.sum()); lo.append(F[:,kx<=0.5].sum()/F.sum())
    ax2=fig.add_subplot(gs[i,1]); ax2.plot(u.mean(1)/u0,z,'k',lw=1); ax2.axvline(0,color='0.7',lw=0.6); ax2.set_xlim(-2,2); ax2.set_yticklabels([])
    ax2.set_title('$\\langle u\\rangle_x/u_0$',fontsize=8)
    print(key,f't={T}: u-energy at kx=0: {np.mean(fr):.3f} (seeds {min(fr):.3f}-{max(fr):.3f}); at |kx|<=0.5 cycles: {np.mean(lo):.3f} ({min(lo):.3f}-{max(lo):.3f})')
fig.colorbar(im,cax=fig.add_axes([0.93,0.3,0.012,0.4]),extend='both').set_label('$u/u_0$')
fig.savefig(f'{OUT}/u-t{T}.png',dpi=150); print('ok')
