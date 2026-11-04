"""Rhines-normalized composite spectra (option 1).
For each depth window centred at z_c: reflect (u even, w odd) about the free-slip surface, apply a Hann window of width
W = NW local Rhines wavelengths, FFT, divide by the annulus mean (kx=0 excluded), rescale (kx,kz) by the local
k_beta(z_c) = (beta(z_c)/u~(z_c))^{1/2}/2pi, and average all windows and times on a common grid in k/k_beta.
The unit dumbbell is |k|/k_beta = |cos(theta)|^{1/2}. Compared with the single global transform over z>-0.5."""
import numpy as np, matplotlib as mpl, os, json, glob
mpl.use('Agg'); import matplotlib.pyplot as plt
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D=f'{ROOT}/data'; OUT=f'{ROOT}/figs/'+os.environ.get('COMPOSITE_OUT','composite'); os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.size':9,'font.family':'serif','mathtext.fontset':'cm'})
NW=3.0; ZC=np.arange(0.0,-0.401,-0.05); G=np.linspace(-3,3,61); dG=G[1]-G[0]
RUNS=[('s2d-deep-k27-lx4','2D, 27-cycle ring, deep'),('s2d-deep-rb1-s1-lx4-t1000','2D, 9-cycle ring, deep'),
      ('d3-deep-rb1-e1e-2-s1','3D CL (y-mean), 9-cycle ring, deep'),('s2d-ubeta-k27-lx4','2D, 27-cycle ring, uniform $\\beta$'),
      ('s2d-none-k27-lx4','2D, 27-cycle ring, no waves\n(deep $\\beta$ used only for scaling)')]
if os.environ.get('COMPOSITE_SET')=='exp':
    RUNS=[('s2d-exp2-k27-lx4','2D, $\\beta\\propto e^{2z}$'),('s2d-exp4-k27-lx4','2D, $\\beta\\propto e^{4z}$'),('s2d-deep-k27-lx4','2D, $\\beta\\propto e^{8z}$ (deep)'),
          ('s2d-ubeta-k27-lx4','2D, uniform $\\beta$'),('s2d-none-k27-lx4','2D, no waves\n($e^{8z}$ $\\beta$ used only for scaling)')]
def meta(r): return dict(l.strip().split('=',1) for l in open(f'{D}/{r}/meta.txt'))
def beta(m,z):
    s=m.get('shear','deep'); g=float(m.get('gam','8'))
    return 0.5+0*z if s=='ubeta' else (0.5*np.exp(0.17*g)*np.exp(g*z) if s=='exp' else 2*np.exp(8*z))
def frames(r):
    fs=sorted(glob.glob(f'{D}/{r}/ymean/t*.bin'),key=lambda f: float(os.path.basename(f)[1:-4])); return fs[int(0.8*len(fs)):]
def compensate(E,K,kx,dk):
    Kb=np.round(K/dk).astype(np.int64).ravel(); nz_=(kx+0*E).ravel()!=0; e=E.ravel()
    num=np.bincount(Kb,weights=e*nz_); den=np.bincount(Kb,weights=nz_.astype(float))
    mean=np.where(den>2,num/np.maximum(den,1),np.nan); C=e/mean[Kb]; C[Kb==0]=np.nan
    return C.reshape(E.shape)
def accumulate(acc,cnt,C,KXn,KZn):
    ok=np.isfinite(C)&(np.abs(KXn)<3)&(np.abs(KZn)<3)
    h,_,_=np.histogram2d(KXn[ok],KZn[ok],bins=[np.append(G-dG/2,G[-1]+dG/2)]*2,weights=np.log2(C[ok])); acc+=h
    n,_,_=np.histogram2d(KXn[ok],KZn[ok],bins=[np.append(G-dG/2,G[-1]+dG/2)]*2); cnt+=n
def ratio(M):
    GX,GZ=np.meshgrid(G,G,indexing='ij'); K=np.hypot(GX,GZ); th=np.arctan2(np.abs(GZ),np.abs(GX)); kap=np.sqrt(np.cos(th))
    sec=(th<np.pi/4)&(np.abs(GZ)>0.15)&(np.abs(GX)>dG)&np.isfinite(M)
    ins=sec&(K<kap)&(K>0.2); out=sec&(K>=kap)&(K<2*kap)
    return float(2**(np.nanmean(M[ins])-np.nanmean(M[out]))) if ins.any() and out.any() else float('nan')
res={}; fig,axs=plt.subplots(2,len(RUNS),figsize=(3.0*len(RUNS),6.2),gridspec_kw=dict(hspace=0.5,wspace=0.3,left=0.04,right=0.94,bottom=0.08,top=0.86))
for c,(r,lab) in enumerate(RUNS):
    m=meta(r); nx,nz,lx=int(m['nx']),int(m['nz']),float(m.get('lx','1')); z=np.loadtxt(f'{D}/{r}/zc.txt'); dz=1/nz
    p=np.loadtxt(f'{D}/{r}/profiles.csv',delimiter=',',skiprows=1); p=p[int(0.8*len(p)):]; Ez=(p[:,1+nz:1+2*nz]+p[:,1+2*nz:]).mean(0)
    zz=np.concatenate([z,-z[::-1]])                         # reflected column, z in (-1,1)
    Cg_sum=None; ng=None
    acc=np.zeros((len(G),len(G))); cnt=np.zeros_like(acc); accg=np.zeros_like(acc); cntg=np.zeros_like(acc); info=[]
    # layer-mean scaling for the global transform
    sel=z>-0.5; kb_layer=np.sqrt(beta(m,z[sel]).mean()/np.sqrt(2*Ez[sel].mean()))/(2*np.pi)
    for f in frames(r):
        a=np.fromfile(f,dtype=np.float32).astype(float); u=a[:nx*nz].reshape(nz,nx).T; w=a[nx*nz:].reshape(nz+1,nx).T
        uc=(u+np.roll(u,-1,0))/2; wc=(w[:,:-1]+w[:,1:])/2
        ue=np.concatenate([uc,uc[:,::-1]],1); we=np.concatenate([wc,-wc[:,::-1]],1)
        kx=np.fft.fftfreq(nx,lx/nx)[:,None]
        for zc in ZC:
            ut=np.sqrt(2*np.interp(zc,z,Ez)); kb=np.sqrt(beta(m,zc)/ut)/(2*np.pi); W=min(NW/kb,0.8)
            win=np.where(np.abs(zz-zc)<W/2,np.cos(np.pi*(zz-zc)/W)**2,0.0)
            E=np.abs(np.fft.fft2(ue*win))**2+np.abs(np.fft.fft2(we*win))**2
            kz=np.fft.fftfreq(2*nz,dz)[None,:]; KX=kx+0*kz; KZ=kz+0*kx; K=np.hypot(KX,KZ)
            C=compensate(E,K,KX,max(0.5,1/W)); keep=K<3.2*kb; accumulate(acc,cnt,C[keep],KX[keep]/kb,KZ[keep]/kb)
            if f==frames(r)[-1]: info.append(dict(zc=round(float(zc),2),kb_cycles=round(float(kb),2),window=round(float(W),3)))
        # global transform over z>-0.5 (as before), rescaled by the layer-mean k_beta
        H=0.5; s=z>-H; zs=z[s]; wn=np.ones(s.sum()); lo=zs<-0.75*H; wn[lo]=np.sin(0.5*np.pi*(zs[lo]+H)/(0.25*H))**2
        us=uc[:,s]*wn; ws=wc[:,s]*wn; Ue=np.concatenate([us,us[:,::-1]],1); We=np.concatenate([ws,-ws[:,::-1]],1)
        E=np.abs(np.fft.fft2(Ue))**2+np.abs(np.fft.fft2(We))**2; kz=np.fft.fftfreq(Ue.shape[1],1/Ue.shape[1])[None,:]/(2*H)
        KX=kx+0*kz; KZ=kz+0*kx; Kg=np.hypot(KX,KZ); C=compensate(E,Kg,KX,1.0); keep=Kg<3.2*kb_layer; accumulate(accg,cntg,C[keep],KX[keep]/kb_layer,KZ[keep]/kb_layer)
        Cg_sum=np.log2(C) if Cg_sum is None else Cg_sum+np.log2(C); ng=1 if ng is None else ng+1; KXg,KZg=KX/kb_layer,KZ/kb_layer
    M=np.where(cnt>0,acc/np.maximum(cnt,1),np.nan); Mg=np.where(cntg>0,accg/np.maximum(cntg,1),np.nan)
    th=np.linspace(0,2*np.pi,721); kk=np.sqrt(np.abs(np.cos(th)))
    for row,(MM,tt) in enumerate([(Mg,'single transform over $z>-0.5$,\nscaled by layer-mean $k_\\beta$'),(M,'composite of local windows,\neach scaled by its own $k_\\beta(z)$')]):
        ax=axs[row,c]
        if row==0:
            sh=np.fft.fftshift; im=ax.pcolormesh(sh(KXg[:,0]),sh(KZg[0]),sh(Cg_sum/ng).T,cmap='RdBu_r',vmin=-3,vmax=3,shading='nearest',rasterized=True)
        else: im=ax.pcolormesh(G,G,MM.T,cmap='RdBu_r',vmin=-3,vmax=3,shading='nearest',rasterized=True)
        ax.plot(kk*np.cos(th),kk*np.sin(th),'k',lw=1); [ax.axvline(v,color='0.2',lw=0.7,ls=':') for v in (-0.5,0.5)]; ax.set_aspect('equal'); ax.set_xlim(-2.5,2.5); ax.set_ylim(-2.5,2.5)
        ax.set_xlabel('$k_x/k_\\beta$'); 
        if c==0: ax.set_ylabel('$k_z/k_\\beta$')
        ax.set_title((lab+'\n' if row==0 else '')+tt+f'\ninside/outside (|θ|<45°) = {ratio(MM):.2f}',fontsize=7.5)
    res[r]=dict(global_inside_over_outside=round(ratio(Mg),3),composite_inside_over_outside=round(ratio(M),3),kb_layer_cycles=round(float(kb_layer),2),windows=info)
fig.colorbar(im,cax=fig.add_axes([0.955,0.25,0.01,0.5]),extend='both').set_label('$\\log_2[E/\\langle E\\rangle_{|k|}]$, last 20% of run')
fig.savefig(f'{OUT}/composite.png',dpi=140); json.dump(res,open(f'{OUT}/composite.json','w'),indent=1); print(json.dumps(res,indent=1))
