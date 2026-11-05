"""Native modal spectra (no taper/padding) and separated signed transfer."""
from pathlib import Path
import numpy as np,json,argparse,time
from scipy.fft import fft,dct,dst
import h5py

def modal(a,vertical):
 nx=a.shape[0];nz=a.shape[1] if vertical=='u' else a.shape[1]-1
 if vertical=='u':b=dct(a,type=2,axis=1,norm='ortho')
 else:
  b=np.zeros((nx,nz));b[:,1:]=dst(a[:,1:-1],type=1,axis=1,norm='ortho')
 return fft(b,axis=0,norm='ortho')/np.sqrt(nx*nz)
def analyze(run):
 started=time.monotonic();meta=dict(x.split('=',1) for x in (run/'meta.txt').read_text().splitlines());nx,nz=int(meta['nx']),int(meta['nz']);lx,h,Te=float(meta['lx']),float(meta['h']),float(meta['Te']);out=run/'analysis';out.mkdir(exist_ok=True)
 kx=2*np.pi*np.fft.fftfreq(nx,d=lx/nx);kz=np.arange(nz)*np.pi/h;radius=np.hypot(kx[:,None],kz[None,:]);dk=min(2*np.pi/lx,np.pi/h);shell=np.floor(radius/dk).astype(int);nb=shell.max()+1
 rows=[];closure=[];names=['full','avg','centered','wave','transverse','numerical']
 for path in sorted((run/'transfer').glob('*.jld2')):
  with h5py.File(path) as f:
   u0=f['u'][()].T;w0=f['w'][()].T;t=float(f['t'][()]);u=modal(u0,'u');w=modal(w0,'w');e=(abs(u)**2+abs(w)**2)/2;terms={}
   for key in names[:4]:terms[key]=(u.conj()*modal(f[key+'_u'][()].T,'u')+w.conj()*modal(f[key+'_w'][()].T,'w')).real
  terms['transverse']=terms['full']-terms['avg'];terms['numerical']=terms['avg']-terms['centered'];eb=np.bincount(shell.ravel(),e.ravel(),minlength=nb);flux={k:-np.cumsum(np.bincount(shell.ravel(),v.ravel(),minlength=nb)) for k,v in terms.items()}
  rows.extend(np.c_[np.full(nb,t/Te),dk*np.arange(1,nb+1),eb,*[flux[k] for k in names]])
  physical=(np.sum(u0**2)+np.sum(w0[:,1:-1]**2))/(2*nx*nz);closure.append([t/Te,e.sum(),physical,*[terms[k].sum() for k in ['centered','wave','numerical','transverse']]])
 np.savetxt(out/'flux.csv',rows,delimiter=',',header='t_Te,k_upper,E_shell,'+','.join('Pi_'+k for k in names),comments='');np.savetxt(out/'closure.csv',closure,delimiter=',',header='t_Te,E_modal,E_native,P_centered,P_wave,P_numerical,P_transverse',comments='')
 ix=abs(kx/(2*np.pi))<=50;iz=kz/(2*np.pi)<=50;spectra=[];times=[];errors=[]
 for path in sorted((run/'ymean').glob('t*.bin')):
  a=np.fromfile(path,dtype=np.float32).astype(float);u0=a[:nx*nz].reshape(nz,nx).T;w0=a[nx*nz:].reshape(nz+1,nx).T;u=modal(u0,'u');w=modal(w0,'w');e=(abs(u)**2+abs(w)**2)/2;physical=(np.sum(u0**2)+np.sum(w0[:,1:-1]**2))/(2*nx*nz);errors.append(abs(e.sum()-physical));spectra.append(e[np.ix_(ix,iz)].astype(np.float32));times.append(float(path.stem[1:]))
 np.savez_compressed(out/'spectra.npz',energy=np.array(spectra),kx=kx[ix],kz=kz[iz],t_Te=np.array(times))
 # Analytic initial modes for zero UL and constant s_z=-beta; no dense eigenproblem needed.
 modes=[]
 for m in range(1,17):
  K=2*np.pi*m/lx
  for n in range(1,17):modes.append([m,n,K,-int(meta['waves'])*float(meta['beta'])*K/(K*K+(n*np.pi/h)**2)])
 np.savetxt(out/'initial-mode-frequencies.csv',modes,delimiter=',',header='x_mode,z_mode,K,Omega',comments='')
 c=np.array(closure);summary=dict(name=run.name,spectral_samples=len(times),transfer_samples=len(c),native_parseval_max_error=float(abs(c[:,1]-c[:,2]).max()),spectrum_parseval_max_error=max(errors),cycles_x_spacing=1/lx,cycles_z_spacing=1/(2*h),analysis_wall_s=time.monotonic()-started,interpretation='Full-depth native modes. No padding, taper, coarsening or fitted exclusion boundary.')
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();analyze(a.run)
