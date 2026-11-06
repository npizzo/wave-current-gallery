"""Read-only reduction of six completed pilots. No state evolution or initial-condition generation."""
from pathlib import Path
import argparse,json,time,csv,hashlib
import numpy as np
import h5py
from scipy.fft import rfft,fft2,fftshift,fftfreq,dct,dst
BASE=Path('/scratch3/workspace/nicholas_pizzo_uri_edu-deep-ring-3d-20261104/mechanism-audit-20261105')
CASES=['deep-linear-s1','deep-ql-s1','deep-full-s1','nowave-linear-s1','nowave-ql-s1','nowave-full-s1']
LAYERS={'top0125':(-.125,0),'top025':(-.25,0),'top05':(-.5,0),'interior':(-.125,-.0625),'lower':(-2,-.5),'full':(-2,0)}
TERMS=['advection','wave','viscosity','pressure','constraint','total']
def comp(a):
 if a.dtype.names:return a['re']+1j*a['im']
 return a

def adv(u,w,dx,dz):
 uc=(u+np.roll(u,-1,1))/2;wx=(w+np.roll(w,1,1))/2
 uf=np.concatenate((u[:1],(u[:-1]+u[1:])/2,u[-1:]),axis=0)
 uu=uc*uc;wu=wx*uf;ww=((w[:-1]+w[1:])/2)**2
 nu=-(uu-np.roll(uu,1,1))/dx-np.diff(wu,axis=0)/dz
 nw=-(np.roll(wu,-1,1)-wu)/dx;nw[1:-1]-=np.diff(ww,axis=0)/dz;nw[[0,-1]]=0
 return nu,nw

def native_transform(u,w):
 hu=dct(rfft(u,axis=1,norm='ortho'),type=2,axis=0,norm='ortho')
 hw=np.zeros_like(hu);hw[1:]=dst(rfft(w[1:-1],axis=1,norm='ortho'),type=1,axis=0,norm='ortho')
 return hu,hw

def run(i,out):
 start=time.time();name=CASES[i];case=BASE/('ql-retry-20261106' if '-ql-' in name else 'production100')/'output'/name
 meta=dict(x.split('=',1) for x in (case/'meta.txt').read_text().splitlines() if '=' in x)
 completed=dict(x.split('=',1) for x in (case/'completed.txt').read_text().splitlines() if '=' in x)
 assert completed['status']=='PASS' and float(completed['stop_Te'])==100
 nx,nz=int(meta['nx']),int(meta['nz']);lx,H=float(meta['lx']),float(meta['h']);dx,dz=lx/nx,H/nz
 Te=float(meta['Te']);u0=float(meta['u0']);waves=meta['waves']=='true';arm=meta['arm'];zc=-H+(np.arange(nz)+.5)*dz;zf=np.linspace(-H,0,nz+1)
 interfaces=np.r_[-H,zc,0];weights={tag:np.maximum(0,np.minimum(interfaces[1:],hi)-np.maximum(interfaces[:-1],lo))/(hi-lo) for tag,(lo,hi) in LAYERS.items()}
 d={'zc':zc,'zf':zf,'u0':np.array(u0),'Te':np.array(Te)};hist={key:[] for key in ['t','U','Q','qvar','R','E','Emean','mean_increment','energy_increment']}
 layerhist={tag:[] for tag in LAYERS};pvbudget={tag:np.zeros(9) for tag in LAYERS};checks={};stress_sum=np.zeros((nz+1,nx//2+1));stress_count=0
 for j,path in enumerate(sorted((case/'diagnostics').glob('t*.jld2'))):
  with h5py.File(path) as f:
   tt=float(f['t_Te'][()]);uk=comp(f['u_K'][()]);U=uk[:,0].real
   p=f['initial_PV_state/profiles'] if j==0 else f['PV_interval_budget/final/profiles']
   Q=p['mean_q'][()].ravel();qv=2*p['half_eddy_variance'][()].ravel();q2=2*p['half_q2'][()].ravel()
   for key,val in [('t',tt),('U',U),('Q',Q),('qvar',qv),('R',f['stress/total'][()].ravel()),('E',f['energy_K'][()].sum()),('Emean',.5*np.mean(U*U)),('mean_increment',f['online_RK_mean_increments'][()]),('energy_increment',f['online_discrete_energy_increments'][()])]:hist[key].append(val)
   for tag,wg in weights.items():
    qm=wg@Q;var=wg@q2-qm*qm;meanvar=wg@(Q*Q)-qm*qm
    grad=np.gradient(Q,dz);bg=-2*np.exp(8*zf) if waves else np.zeros_like(zf)
    mask=(wg>0)&(zf<0)&(zf>-H)
    gradratio=np.sqrt(np.sum(wg[mask]*grad[mask]**2)/max(np.sum(wg[mask]*bg[mask]**2),1e-30)) if waves else np.nan
    flatten=np.sum(wg[mask]*(np.abs(grad[mask])<.2*np.abs(bg[mask])))/np.sum(wg[mask]) if waves else np.nan
    layerhist[tag].append([qm,var,meanvar,wg@qv,gradratio,flatten])
    if j:
     b=f['PV_interval_budget/increments'];old=f['PV_interval_budget/initial/profiles'];om=wg@old['mean_q'][()].ravel();mid=.5*(om+qm)
     for n,term in enumerate(TERMS):pvbudget[tag][n]+=wg@b['component_'+term+'_half_q2'][()].ravel()-mid*(wg@b['component_'+term+'_mean_q'][()].ravel())
     for n,prefix in enumerate(['transport','advection_wave_remainder','observed'],6):pvbudget[tag][n]+=wg@b[prefix+'_half_q2'][()].ravel()-mid*(wg@b[prefix+'_mean_q'][()].ravel())
   if tt>=50 and 'all_K' in f['stress']:
    wt=.5 if tt in [50,100] else 1;stress_sum+=wt*f['stress/all_K'][()];stress_count+=wt
   if j in [0,100,250,500,1000]:
    d[f'selected_work_{int(round(tt))}']=np.stack([f['work/'+k][()] for k in ['stokes','mean_related','eddy_eddy','viscosity','constraint','total']]);d['selected_K']=f['selected_K_cycles'][()]
   for k in ['stress_relative','component_reconstruction_relative','q_curl_relative','pressure_flux_relative','local_PV_identity_maximum_so_far']:
    checks[k]=max(checks.get(k,0),float(f['checks/'+k][()]))
  if j%250==0:print(name,'diagnostics',j,round(time.time()-start),flush=True)
 assert len(hist['t'])==1001
 for k,v in hist.items():d[k]=np.asarray(v)
 for k,v in layerhist.items():d['pv_'+k]=np.asarray(v);d['pvbudget_'+k]=pvbudget[k]
 d['stress_mean50_100']=stress_sum/stress_count
 # Exact online energies from lightweight table (the selected modes above are not the full energy).
 ts=np.genfromtxt(case/'timeseries.csv',names=True,delimiter=',');d['E']=ts['E'];d['Emean']=ts['Emean'];d['divergence']=ts['divergence_relative']
 close=np.genfromtxt(case/'online-closure.csv',names=True,delimiter=',');checks['energy_closure_relative']=float(np.max(abs(close['E_residual']))/ts['E'][0]);checks['mean_closure_over_u0']=float(np.max(abs(close['maximum_U_residual']))/u0)
 d['K']=np.arange(nx//2+1)/lx;d['ell_native']=np.arange(nz)/(2*H);K,L=np.meshgrid(d['K'],d['ell_native']);radius=np.hypot(K,L);mult=np.ones(nx//2+1)*2;mult[[0,-1]]=1
 erows=[];specrows=[];sums={};counts={};fluxrows=[];native_checks=[]
 selected_times=sorted(set([0,10,25,50,100]+list(range(50,101,5))+list(range(10,26,5))))
 for tt in selected_times:
  pth=case/'fields'/f't{tt:06.1f}.jld2'
  with h5py.File(pth) as f:
   u=f['u'][()];w=f['w'][()];U=u.mean(1);uc=(u+np.roll(u,-1,1))/2;wc=(w[:-1]+w[1:])/2
   hu,hw=native_transform(u,w);en=.5*(abs(hu)**2+abs(hw)**2)*mult/(nx*nz);E=.5*(np.sum(u*u)+np.sum(w[1:-1]**2))/(nx*nz)
   checks['parseval_relative']=max(checks.get('parseval_relative',0),abs(en.sum()-E)/E)
   tail=en[(K>.8*(nx/2/lx))|(L>.8*(nz/2/H))].sum()/E
   erows.append([tt,E,en[:,0].sum()/E,tail,en[(radius<16)&(K>0)].sum()/E])
   if tt in [0,10,25,50,100]:
    d[f'nativeE_{tt}']=en[:161,:161]
    q=np.zeros_like(w);q[1:-1]=np.diff(u,axis=0)/dz;q-=(w-np.roll(w,1,1))/dx
    if waves:q+=(-.25*np.exp(8*zf))[:,None]
    d[f'u_map_{tt}']=uc[-256:].reshape(128,2,512,4).mean((1,3));d[f'q_map_{tt}']=q[-257:-1].reshape(128,2,512,4).mean((1,3));d['mapz']=zc[-256:].reshape(128,2).mean(1);d['mapx']=(np.arange(512)+.5)*4*dx
    # Full-domain all-mode component work and cumulative radial flux: exact native basis.
    works={};native_tend=[]
    for term in ['n','s','d','gradp','constraint','rhs']:
     fu=f[term+'_u'][()];fw=f[term+'_w'][()];fu,fw=(-fu,-fw) if term=='gradp' else (fu,fw)
     tu,tw=native_transform(fu,fw);works[term]=np.real(hu.conj()*tu+hw.conj()*tw)*mult/(nx*nz);native_tend.append((fu,fw))
    eeu,eew=adv(u-U[:,None],w-w.mean(1)[:,None],dx,dz)
    if arm=='linear':eeu*=0;eew*=0
    elif arm=='ql':eeu=np.broadcast_to(eeu.mean(1)[:,None],eeu.shape);eew=np.broadcast_to(eew.mean(1)[:,None],eew.shape)
    eu,ew=native_transform(eeu,eew);works['ee']=np.real(hu.conj()*eu+hw.conj()*ew)*mult/(nx*nz);works['mean']=works['n']-works['ee']
    # Independent comparison of Python advection with saved solver component.
    au,aw=adv(u,w,dx,dz)
    if arm=='ql':aee,aew=adv(u-U[:,None],w-w.mean(1)[:,None],dx,dz);au-=aee-aee.mean(1)[:,None];aw-=aew-aew.mean(1)[:,None]
    if arm=='linear':au*=0;aw*=0
    err=np.sqrt(np.sum((au-native_tend[0][0])**2)+np.sum((aw-native_tend[0][1])**2))/max(np.sqrt(np.sum(native_tend[0][0]**2)+np.sum(native_tend[0][1]**2)),1e-30)
    native_checks.append({'t':tt,'advection_relative':float(err),'total_work':float(works['rhs'].sum()),'advection_net_work':float(works['n'].sum()),'pressure_net_work':float(works['gradp'].sum())})
    # Positive flux denotes net transfer out of modes inside radius; no local-cascade claim.
    for cutoff in np.arange(1,51):
     mask=radius<=cutoff;fluxrows.append([tt,cutoff]+[-works[k][mask].sum() for k in ['n','ee','mean','s','d','gradp','rhs']])
   for tag,D in [('top0125',.125),('top025',.25),('top05',.5)]:
    nn=int(round(D/dz));a=uc[-nn:];b=wc[-nn:];tap=np.hanning(nn);norm=np.mean(tap*tap)
    for method in ['hann','reflect']:
     if method=='hann':
      av=a*tap[:,None];bv=b*tap[:,None];size=nn
     else:
      # Explicit parity extension is a sensitivity control, not a physical wall.
      av=np.r_[a,a[::-1]];bv=np.r_[b,-b[::-1]];size=2*nn;norm2=1
     energy=fftshift(.5*(abs(fft2(av))**2+abs(fft2(bv))**2)/(nx*size)**2/(norm if method=='hann' else 1))
     kx=fftshift(fftfreq(nx,dx));ell=fftshift(fftfreq(size,dz));kk,ll=np.meshgrid(kx,ell);rr=np.hypot(kk,ll);theta=np.divide(abs(kk),rr,out=np.zeros_like(rr),where=rr>0)
     tagm=tag+'_'+method;ix=abs(kx)<=40;iz=abs(ell)<=40
     d[tagm+'_kx']=kx[ix];d[tagm+'_ell']=ell[iz]
     cropped=energy[np.ix_(iz,ix)]
     if tt in [0,10,25,50,100]:d[f'{tagm}_E_{tt}']=cropped
     for lo,hi in [(10,25),(50,100)]:
      if lo<=tt<=hi:
       key=f'{tagm}_avg{lo}_{hi}';wt=.5 if tt in [lo,hi] else 1
       sums[key]=sums.get(key,np.zeros_like(cropped))+wt*cropped;counts[key]=counts.get(key,0)+wt
     for low,high in [(4,16),(16,24)]:
      valid=(rr>=low)&(rr<high)&(kk!=0)
      if method=='reflect':valid&=(ll!=0)
      sector=valid&(theta>=np.sqrt(3)/2);total=energy[valid].sum()
      specrows.append({'t':tt,'window':tag,'method':method,'low':low,'high':high,'annulus_energy':float(total),'horizontal_fraction':float(energy[sector].sum()/max(total,1e-30)),'horizontal_bins':int(sector.sum()),'vertical_bin_spacing':float(ell[1]-ell[0]),'jet_fraction':float(energy[:,kx==0].sum()/energy.sum())})
   print(name,'field',tt,round(time.time()-start),flush=True)
 for key,val in sums.items():d[key]=val/counts[key]
 d['field_metrics']=np.asarray(erows);d['radial_flux']=np.asarray(fluxrows)
 assert checks['parseval_relative']<1e-12
 assert max(r['advection_relative'] for r in native_checks)<1e-9
 summary={'case':name,'meta':meta,'completed':completed,'checks':checks,'native_checks':native_checks,'spectral_metrics':specrows,'runtime_seconds':time.time()-start,'operation':'Read-only existing-output analysis; no simulation','pv_layer_columns':['mean_q','total_variance','mean_profile_variance','eddy_variance','rms_gradient_over_initial','flat_depth_fraction_20pct'],'pv_budget_columns':TERMS+['physical_transport_reference','advection_wave_remainder','observed'],'field_metric_columns':['t_Te','energy','full_depth_jet_fraction','upper20pct_grid_energy_fraction','eddy_energy_below16_over_E'],'radial_flux_columns':['t_Te','radius_cycles','advection','eddy_eddy','mean_related','stokes','viscosity','pressure','rhs'],'layers':LAYERS,'limitations':['Single seed, 100Te, no convergence study.','Local spectra use collocated velocities; native energies and transfer use exact C-grid norm.','Full-domain cumulative flux is not a near-surface cascade flux.','PV physical-transport remainder is not automatically mixing or diffusion.','Signed momentum stress does not independently identify propagating vortex waves.']}
 out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/(name+'.npz'),**d);(out/(name+'.json')).write_text(json.dumps(summary,indent=2)+'\n');print('DONE',name,time.time()-start,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('case',type=int);p.add_argument('out',type=Path);a=p.parse_args();run(a.case,a.out)
