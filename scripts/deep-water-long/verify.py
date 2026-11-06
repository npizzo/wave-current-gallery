"""Independent NumPy/SciPy check of Julia spectra against saved native velocities."""
from pathlib import Path
import numpy as np,h5py,hashlib,json
from scipy.fft import rfft,fft
import sys
root=Path(sys.argv[1]);nx=4096;nz=2048;dz=2/nz;errs=[];hashes=[]
for case in ['deep','nowave']:
 d=root/'data'/f'{case}-long-h2-lx4-s1';hashes.append(hashlib.sha256((d/'ymean/t0000.00.bin').read_bytes()).hexdigest())
 for t in [100,300,600,900]:
  raw=np.fromfile(d/'ymean'/f't{t:07.2f}.bin',np.float32).astype(float);u=raw[:nx*nz].reshape(nz,nx);w=raw[nx*nz:].reshape(nz+1,nx);uc=(u+np.roll(u,-1,axis=1))/2;wc=(w[:-1]+w[1:])/2
  hu=rfft(uc,axis=1,workers=2)[:,:161]/nx;hw=rfft(wc,axis=1,workers=2)[:,:161]/nx
  with h5py.File(d/'spatial'/f't{t:07.2f}.jld2') as f:
   for depth in [.5,.25,.125]:
    n=round(depth/dz)
    for method in ['hann','reflect']:
     q=(np.arange(n)+.5)/n;taper=np.sin(np.pi*q)**2 if method=='hann' else np.where(q<.25,np.sin(.5*np.pi*q/.25)**2,1)
     aa=hu[-n:]*taper[:,None];bb=hw[-n:]*taper[:,None]
     if method=='reflect':aa=np.concatenate([aa,aa[::-1]]);bb=np.concatenate([bb,-bb[::-1]])
     E=(abs(fft(aa,axis=0))**2+abs(fft(bb,axis=0))**2)/(2*len(aa)**2*np.mean(taper**2));E[:,1:]*=2;ell=np.fft.fftfreq(len(aa),dz);E=E[abs(ell)<=40]
     tag=f'top{depth}_{method}';ref=f[tag+'/E'][()];assert np.allclose(ell[abs(ell)<=40],f[tag+'/ell'][()]);err=np.linalg.norm(E-ref)/np.linalg.norm(ref);assert err<1e-6,(case,t,tag,err);errs.append(dict(case=case,t_Te=t,window=tag,relative_L2_error=float(err)))
assert hashes[0]==hashes[1]
result=dict(independent_spectral_checks=errs,max_relative_L2_error=max(x['relative_L2_error'] for x in errs),matched_initial_velocity_sha256=hashes[0],note='Independent SciPy FFT of Float32 archived velocities vs Float64 Julia online spectra; difference includes archive rounding. No time integration.')
(root/'analysis/results/independent-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='independent_spectral_checks'}))
