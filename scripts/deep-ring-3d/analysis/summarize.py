import sys, json, numpy as np
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
def meta(d): return {k:v for k,v in (l.strip().split('=',1) for l in open(d/'meta.txt'))}
def aniso(u,w,nx,nz,lx=1.0):
    # eddy (x-varying) part of the y-mean field; even/odd z-extension so walls are handled
    u=u-u.mean(0,keepdims=True); w=w[:, :nz]
    ue=np.concatenate([u,u[:,::-1]],1); we=np.concatenate([w,-w[:,::-1]],1)
    E=np.abs(np.fft.fft2(ue))**2+np.abs(np.fft.fft2(we))**2
    kx=np.fft.fftfreq(nx,lx/nx)[:,None]; kz=np.fft.fftfreq(2*nz,1/(2*nz))[None,:]/2   # cycles per unit length
    return float(((kz**2-kx**2)*E).sum()/((kz**2+kx**2)*E).sum()), float((np.sqrt(kx**2+kz**2)*E).sum()/E.sum())
for d in sorted(p for p in root.iterdir() if (p/'timeseries.csv').exists()):
    m=meta(d); ts=np.genfromtxt(d/'timeseries.csv',delimiter=',',names=True); nx,nz=int(m['nx']),int(m['nz']); u0=float(m['u0'])
    pr=np.loadtxt(d/'profiles.csv',delimiter=',',skiprows=1,ndmin=2); UL=pr[:,1:1+nz]
    out=dict(name=d.name,complete=(d/'completed.txt').exists(),t_Te=float(ts['t_Te'][-1]),waves=int(m['waves']),rb=float(m['rb']),seed=int(m['seed']),ny=int(m['ny']),eps3d=float(m['eps3d']))
    for tag,i in [('final',-1)]+[('t%d'%t,int(np.argmin(abs(ts['t_Te']-t)))) for t in (20,50,100)]:
        KE=ts['KE'][i]; out[tag]=dict(t=float(ts['t_Te'][i]),KE_over_KE0=float(KE/ts['KE'][0]),jet_frac=float(ts['KEjet'][i]/KE),frac3d=float(ts['KE3D'][i]/KE),
            frac3d_upper=float(ts['KE3D_upper'][i]/(ts['KE3D_upper'][i]+ts['KE2D_upper'][i])),frac3d_lower=float(ts['KE3D_lower'][i]/(ts['KE3D_lower'][i]+ts['KE2D_lower'][i])),
            maxUL_over_u0=float(np.abs(UL[i]).max()/u0),UL_sign_changes=int(np.sum(np.diff(np.sign(UL[i][np.abs(UL[i])>0.1*u0]))!=0)))
    f=sorted((d/'ymean').glob('t*.bin'))
    if f:
        a=np.fromfile(f[-1],dtype=np.float32); u=a[:nx*nz].reshape(nz,nx).T; w=a[nx*nz:].reshape(nz+1,nx).T
        out['final']['anisotropy'],out['final']['centroid_cycles']=aniso(u.astype(float),w.astype(float),nx,nz,float(m.get('lx','1')))
    rows.append(out)
json.dump(rows,open(root.parent/'summary.json','w'),indent=1)
hdr='%-24s %5s %6s  %8s %7s %7s %7s %7s %6s %6s %5s'%('run','done','t/Te','jet_frac','3Dfrac','3D_up','3D_lo','max|UL|','aniso','cent','sign')
print(hdr)
for r in rows:
    f=r['final']; print('%-24s %5s %6.1f  %8.3f %7.3f %7.3f %7.3f %7.2f %6.2f %6.2f %5d'%(r['name'],'yes' if r['complete'] else 'no',r['t_Te'],f['jet_frac'],f['frac3d'],f['frac3d_upper'],f['frac3d_lower'],f['maxUL_over_u0'],f.get('anisotropy',np.nan),f.get('centroid_cycles',np.nan),f['UL_sign_changes']))
