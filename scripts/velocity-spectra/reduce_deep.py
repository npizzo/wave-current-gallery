import numpy as np, pathlib, json
import argparse
p=argparse.ArgumentParser();p.add_argument('campaign_root',type=pathlib.Path);p.add_argument('output',type=pathlib.Path);args=p.parse_args()
R=args.campaign_root; D=R/'data'; O=args.output; O.mkdir(parents=True,exist_ok=True)
def meta(r): return dict(l.strip().split('=',1) for l in (D/r/'meta.txt').read_text().splitlines())
def field(r,t):
 m=meta(r); nx,nz=int(m['nx']),int(m['nz']); a=np.fromfile(D/r/'ymean'/f't{t:07.2f}.bin',np.float32).astype(float); return a[:nx*nz].reshape(nz,nx).T,a[nx*nz:].reshape(nz+1,nx).T
out={}
for key in ['deep','ubeta','none']:
 for H in [.5,1.]:
  allE=[]; kes=[]
  for seed in range(1,9):
   r=f'ens-{key}-k27-s{seed}'; m=meta(r); z=np.loadtxt(D/r/'zc.txt'); nx,nz=int(m['nx']),int(m['nz']); lx=float(m['lx']); es=[]
   for t in range(15,21):
    u,w=field(r,t); u=(u+np.roll(u,-1,0))/2; w=(w[:,:-1]+w[:,1:])/2; sel=z>-H; zs=z[sel]; win=np.ones(sel.sum())
    if H<1: lo=zs<-.75*H; win[lo]=np.sin(.5*np.pi*(zs[lo]+H)/(.25*H))**2
    u=u[:,sel]*win; w=w[:,sel]*win
    ue=np.concatenate([u,u[:,::-1]],1); we=np.concatenate([w,-w[:,::-1]],1)
    E=.5*(abs(np.fft.fft2(ue))**2+abs(np.fft.fft2(we))**2)/ue.size**2/np.mean(win**2)
    E=np.fft.fftshift(E); kx=np.fft.fftshift(np.fft.fftfreq(nx,lx/nx)); kz=np.fft.fftshift(np.fft.fftfreq(ue.shape[1],1/nz)); ix=abs(kx)<=40; iz=abs(kz)<=40; es.append(E[np.ix_(ix,iz)])
   allE.append(np.mean(es,0))
  tag=f'{key}_{H}';out[tag]=np.array(allE);out[f'kx_{H}']=kx[ix];out[f'kz_{H}']=kz[iz]
  print(tag,flush=True)
for key in ['deep','none']:
 r=f'd3-{key}-rb1-e1e-2-s1'; m=meta(r);z=np.loadtxt(D/r/'zc.txt');out[f'{key}_z']=z;out[f'{key}_u0']=float(m['u0']);out[f'{key}_x']=np.arange(int(m['nx']))*float(m.get('lx',1.0))/int(m['nx'])
 for t in [0,20,50,200]:u,w=field(r,t);out[f'{key}_u_{t}']=u
out['provenance']=np.array('Eight seeds each, unpadded unsmoothed reflected spectra, 15-20Te arithmetic average; normalized to one-half mean-square collocated velocity with taper-square correction. Same window for every comparison.')
np.savez_compressed(O/'figure-data.npz',**out)
print('done',flush=True)
