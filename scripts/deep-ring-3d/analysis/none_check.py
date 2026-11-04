"""Where is the late no-wave energy? Shares in the gravest (kx,kz) modes; full-box field plot."""
import numpy as np, matplotlib as mpl, os
mpl.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D=f'{ROOT}/data'; OUT=f'{ROOT}/figs/ens-vallis'
for r in ['s2d-none-k27-lx4','s2d-ubeta-k27-lx4']:
    m=dict(l.strip().split('=',1) for l in open(f'{D}/{r}/meta.txt')); nx,nz,lx,u0=int(m['nx']),int(m['nz']),float(m['lx']),float(m['u0'])
    acc=None
    for t in range(800,1001,10):
        a=np.fromfile(f'{D}/{r}/ymean/t{t:07.2f}.bin',dtype=np.float32).astype(float); u=a[:nx*nz].reshape(nz,nx).T; w=a[nx*nz:].reshape(nz+1,nx).T
        uc=(u+np.roll(u,-1,0))/2; wc=(w[:,:-1]+w[:,1:])/2
        ue=np.concatenate([uc,uc[:,::-1]],1); we=np.concatenate([wc,-wc[:,::-1]],1)
        E=np.abs(np.fft.fft2(ue))**2+np.abs(np.fft.fft2(we))**2; acc=E if acc is None else acc+E
    E=acc/acc.sum(); kx=np.round(np.fft.fftfreq(nx,lx/nx),3); kz=np.round(np.fft.fftfreq(2*nz,1/(2*nz))/2,3)
    print(r,'energy share by |kx| (cycles per unit length), summed over kz:')
    for k in [0,0.25,0.5,0.75,1.0]: print('   |kx|=',k, round(float(E[np.abs(kx)==k].sum()),3))
    print('   |kx|<=1 total', round(float(E[np.abs(kx)<=1].sum()),3),'  share at kx=0 & |kz|=0.5 (full-depth shear, u~cos(pi z))',round(float(E[np.ix_(kx==0,np.abs(kz)==0.5)].sum()),3))
    print('   top 6 modes (kx,kz,share):',sorted([(float(kx[i]),float(kz[j]),round(float(E[i,j]),3)) for i in range(nx) for j in range(2*nz) if E[i,j]>0.01],key=lambda x:-x[2])[:6])
    fig,ax=plt.subplots(figsize=(10,2.8)); a=np.fromfile(f'{D}/{r}/ymean/t0900.00.bin',dtype=np.float32); u=a[:nx*nz].reshape(nz,nx)
    im=ax.pcolormesh(np.arange(nx)*lx/nx,np.loadtxt(f'{D}/{r}/zc.txt'),u/u0,cmap='RdBu_r',vmin=-3,vmax=3,shading='auto',rasterized=True); ax.set_aspect('equal'); fig.colorbar(im,ax=ax,shrink=0.8)
    ax.set_title(f'{r}: u/u0 at t=900 Te, full box'); fig.tight_layout(); fig.savefig(f'{OUT}/fullbox-{r}.png',dpi=130); plt.close(fig)
