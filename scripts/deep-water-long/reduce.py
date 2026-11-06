"""Read completed outputs; no integration. Fixed stage means and directional budgets."""
from pathlib import Path
import numpy as np,h5py,json,csv,argparse,time,hashlib
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
stages=[100,300,600,900];tags=['full_native']+[f'top{d}_{m}' for d in [.5,.25,.125] for m in ['hann','reflect']];terms=['centered','eddy','meaneddy','wave','numerical','pressure','numerical_pressure'];bands=[(1,4),(4,8),(8,16),(16,24),(4,16),(1,24)]
arr={};rows=[];tsrows=[];checks=[];started=time.time()
for case,name in [('deep','deep-long-h2-lx4-s1'),('control','nowave-long-h2-lx4-s1')]:
 d=args.root/'data'/name;meta=dict(x.split('=',1) for x in (d/'meta.txt').read_text().splitlines());nx,nz=int(meta['nx']),int(meta['nz']);lx,H,Te,u0=map(float,[meta['lx'],meta['h'],meta['Te'],meta['u0']]);k=np.arange(161)/lx;z=-H+(np.arange(nz)+.5)*H/nz;arr['k']=k;arr['z']=z;arr['Te']=np.array(Te);arr['u0']=np.array(u0)
 (args.out/(case+'-meta.json')).write_text(json.dumps(meta,indent=2)+'\n')
 for file in ['timeseries.csv','profiles.csv','completed.txt']:
  (args.out/(case+'-'+file)).write_bytes((d/file).read_bytes())
 cfg={}
 with h5py.File(d/'spatial/t0000.00.jld2') as f:
  E0=float(f['native_energy'][()]);arr[case+'_E0']=np.array(E0)
  for tag in tags:
   ell=f[tag+'/ell'][()];arr[tag+'_ell']=ell;K,L=np.meshgrid(k,ell);kap=np.hypot(K,L);cos=np.divide(K,kap,out=np.zeros_like(K),where=kap>0)
   masks={}
   for lo,hi in bands:
    base=(kap>=lo)&(kap<hi)
    # Reflected ell=0 is boundary/parity dependent; omit it from eddy angular comparisons.
    eddy=base&(K>0)
    if tag.endswith('reflect'):eddy&=(L!=0)
    masks[(lo,hi)]={'all':base,'eddy':eddy,'horizontal':eddy&(cos>=np.sqrt(3)/2),'vertical':eddy&(cos<=.5),'near_axis':base&(cos<=.1),'K0':base&(K==0)}
   cfg[tag]=(ell,masks)
 for tf in stages:
  for tag in tags:
   for term in ['E']+['T_'+x for x in terms]: arr[f'{case}_{tf}_{tag}_{term}']=np.zeros((len(cfg[tag][0]),len(k)))
 for tt in range(1001):
  active=[tf for tf in stages if tf-20<=tt<=tf+20]
  with h5py.File(d/'spatial'/f't{tt:07.2f}.jld2') as f:
   checks.append([case,tt,*[float(f[key][()]) for key in ['conservation_error','projection_error','projection_numerical_error','parseval_error']]])
   U=f['profiles/U'][()];ep=f['profiles/energy'][()]
   r={'case':case,'t_Te':tt,'native_energy':float(f['native_energy'][()]),'jet_fraction_full':float(.5*np.mean(U**2)/np.mean(ep))}
   for depth in [.5,.25,.125]:
    sel=z>-depth;r[f'jet_fraction_top{depth}']=float(.5*np.mean(U[sel]**2)/np.mean(ep[sel]))
   # Full native conservative directional inputs in every fixed annulus.
   tag='full_native';E=f[tag+'/E'][()];N=f[tag+'/T_centered'][()]
   for (lo,hi),masks in cfg[tag][1].items():
    for sector in ['all','horizontal','vertical','near_axis','K0']:
     mask=masks[sector];r[f'E_{lo}_{hi}_{sector}']=float(E[mask].sum());r[f'N_{lo}_{hi}_{sector}']=float(N[mask].sum())
   tsrows.append(r)
   for tf in active:
    weight=(.5 if tt in [tf-20,tf+20] else 1)/40
    for tag in tags:
     for term in ['E']+['T_'+x for x in terms]:arr[f'{case}_{tf}_{tag}_{term}']+=weight*f[tag+'/'+term][()]
    if tt in [tf-20,tf+20]:
     for tag in tags:arr[f'{case}_{tf}_{tag}_E_{"lo" if tt==tf-20 else "hi"}']=f[tag+'/E'][()]
   if tt in stages:
    arr[f'{case}_{tt}_U']=U;arr[f'{case}_{tt}_energy_profile']=ep
  if tt%100==0:print(case,tt,round(time.time()-started,1),flush=True)
 # Display fields are spatially averaged into 4x4 cell blocks AFTER spectra were computed at native resolution.
 for tf in stages:
  raw=np.fromfile(d/'ymean'/f't{tf:07.2f}.bin',np.float32);u=raw[:nx*nz].reshape(nz,nx);uc=.5*(u+np.roll(u,-1,axis=1));step=4
  arr[f'{case}_{tf}_u']=uc.reshape(nz//step,step,nx//step,step).mean((1,3));arr['x_display']=(np.arange(nx//step)+.5)*step*lx/nx;arr['z_display']=-H+(np.arange(nz//step)+.5)*step*H/nz
 for tf in stages:
  for tag,(ell,bandmasks) in cfg.items():
   E=arr[f'{case}_{tf}_{tag}_E'];DE=(arr[f'{case}_{tf}_{tag}_E_hi']-arr[f'{case}_{tf}_{tag}_E_lo'])/(40*Te)
   work={x:arr[f'{case}_{tf}_{tag}_T_{x}'] for x in terms};total=sum(work[x] for x in ['centered','wave','numerical','pressure','numerical_pressure'])
   for (lo,hi),masks in bandmasks.items():
    valid=bool(masks['horizontal'].any() and masks['vertical'].any())
    for sector,mask in masks.items():
     r={'case':case,'stage_Te':tf,'window':tag,'band_lo':lo,'band_hi':hi,'sector':sector,'sample_count':int(mask.sum()),'angular_comparison_valid':valid,'energy':float(E[mask].sum()),'annulus_eddy_energy':float(E[masks['eddy']].sum()),'observed_dE_dt':float(DE[mask].sum()),'predicted_dE_dt':float(total[mask].sum()),'budget_residual':float((DE-total)[mask].sum()),'E0':E0,'Te':Te}
     for term in terms:r['T_'+term]=float(work[term][mask].sum())
     rows.append(r)
 for tf in stages:
  for tag in tags:
   E=arr[f'{case}_{tf}_{tag}_E'];N=arr[f'{case}_{tf}_{tag}_T_centered'];K,L=np.meshgrid(k,cfg[tag][0]);kap=np.hypot(K,L)
   arr[f'{case}_{tf}_{tag}_radial_cutoffs']=np.array([3,6,9,12,18,24])
   arr[f'{case}_{tf}_{tag}_radial_input']=np.array([N[kap<c].sum() for c in [3,6,9,12,18,24]])
np.savez_compressed(args.out/'stage-spectra.npz',**arr)
for name,data in [('directional-budgets.csv',rows),('evolution.csv',tsrows)]:
 with (args.out/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
errors=np.array([r[2:] for r in checks]);summary={'cases':2,'seed':1,'stages_Te':stages,'averaging_windows':[[t-20,t+20] for t in stages],'samples_per_stage':41,'weighting':'Trapezoidal time average of integer-Te outputs; one matched seed, not an ensemble','diagnostic_files_checked':len(checks),'max_errors':dict(zip(['centered_conservation','pressure_projection','numerical_pressure_projection','retained_x_parseval'],errors.max(0).tolist())),'runtime_seconds':time.time()-started,'notes':['K>0 stored with multiplicity2; K=0 retained once. ell signed. All retained |K|,|ell| <=40 cycles per original unit length.','Full native T_centered is conservative advection evaluated on WENO-evolved fields. Local-window T includes transport. Eddy/meaneddy are advecting-interaction splits, not separate receiving modes.','Independent pressure-work/vertical-flux closure was validated before production; energy closure here additionally checks sampled finite-time budgets.','Velocity panels are 4x4 spatial block averages for display only. Spectra use original native resolution.']}
(args.out/'reduction-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
