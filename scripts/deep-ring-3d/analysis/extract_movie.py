"""Pack the slice records of a movie run into one compressed npz (u_xz, UL, t) for rendering."""
import numpy as np, sys, glob, os, re
run,out=sys.argv[1],sys.argv[2]
lay=open(f'{run}/slices/layout.txt').read().split('\n')[1]
kv=dict(x.split('=') for x in lay.split()); nx,ny,nz=int(kv['nx']),int(kv['ny']),int(kv['nz'])
files=sorted(glob.glob(f'{run}/slices/s*.bin')); t=np.array([float(re.search(r's([0-9.]+)\.bin',f).group(1)) for f in files])
uxz=np.empty((len(files),nx,nz),np.float16); UL=np.empty((len(files),nz),np.float32); uxy=np.empty((len(files),nx,ny),np.float16)
for i,f in enumerate(files):
    a=np.fromfile(f,dtype=np.float32); o=0
    uxz[i]=a[o:o+nx*nz].reshape(nz,nx).T/float(kv['u0']); o+=nx*nz
    uxy[i]=a[o:o+nx*ny].reshape(ny,nx).T/float(kv['u0']); o+=nx*ny
    UL[i]=a[o:o+nz]/float(kv['u0'])
np.savez_compressed(out,t=t,uxz=uxz,uxy=uxy,UL=UL,z=np.loadtxt(f'{run}/zc.txt'),u0=float(kv['u0']),zslice=float(kv['zslice']))
print(out,len(t),t[0],t[-1])
