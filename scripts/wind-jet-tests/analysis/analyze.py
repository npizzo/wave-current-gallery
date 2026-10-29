#!/usr/bin/env python3
"""Verified, cached analysis and standalone gallery for the regime campaign."""
from __future__ import annotations
import argparse,csv,hashlib,html,json,math,os,re,shutil,tempfile,datetime
from pathlib import Path
import numpy as np
from science import (finite,ratio,weighted_rms,weighted_dot,gaussian_matrix,filter_field,
    scalar_metrics,feedback_metrics,paired_fields,centered_time_derivative,x_mode_supply,
    transverse_growth,twin_collapse,qualified_twin_growth)

FILTERS=(('unfiltered',0.,0.),('g1',1.,.25),('g2',2.,.5),('g4',4.,1.))
CENTER_FIELDS=('Uc','Vc','Wc','Q','Qx','Qz','A','Rbar','Qdot_advection_full','Qdot_advection_mean','Qdot_wave','Qdot_viscous','Qdot_forcing','Qdot_total')
NATIVE_FIELDS=('U','W','U_adv_full','W_adv_full','U_adv_mean','W_adv_mean','U_wave','W_wave','U_viscous','W_viscous','U_forcing','W_forcing')
PUBLIC_CONFIG=('nx','ny','nz','lx','ly','h','k','S','S_reference','alpha','uoffset','eddy_rms','injection_wavelength','perturb_energy_ratio','seed','stop_time','max_dt','viscosity','nu','reynolds','re_lambda','maintain_background','advection','velocity_convention','base_ic_sha256','perturbation_ic_sha256','initial_condition_sha256','perturbation_shape_sha256','waves','initial_metrics')
MATCH_KEYS=('nx','ny','nz','lx','ly','h','k','S','S_reference','alpha','uoffset','eddy_rms','injection_wavelength','seed','max_dt','base_ic_sha256')

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not math.isfinite(v):return None
    return v

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('w') as f:json.dump(clean(value),f,indent=2,allow_nan=False);f.write('\n')
    temporary.replace(path)

def read_csv(path):
    with Path(path).open(newline='') as f:raw=list(csv.DictReader(f))
    rows=[]
    for row in raw:
        out={}
        for key,value in row.items():
            if value is None or not value.strip():out[key]=None;continue
            try:
                n=float(value);out[key]=n if math.isfinite(n) else None
            except ValueError:out[key]=value
        rows.append(out)
    return rows

def write_csv(path,rows):
    if not rows:return
    names=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        w.writerows({k:('' if v is None or isinstance(v,float) and not math.isfinite(v) else v) for k,v in r.items()} for r in rows)

def safe_path(root,name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts or p==Path('.'):raise ValueError('Unsafe relative output path')
    result=Path(root)/p
    if not result.resolve().is_relative_to(Path(root).resolve()):raise ValueError('Output path escapes run directory')
    return result

def identifier(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',value):raise ValueError('Invalid run identifier')
    return value

def campaign_status(root,entry,version):
    status=root/'data/status'/identifier(entry['id']);receipt=status/'completed.json'
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if r.get('id')!=entry['id'] or r.get('completed') is not True or r.get('source_version')!=version:raise ValueError('Completion receipt does not match campaign source')
        required={'configuration.json','completed.json','grid.json','regime-budget.csv','jet-budget.csv'}
        if not required<=set(r.get('files',{})):raise ValueError('Incomplete verified diagnostic inventory')
        return 'complete',r
    if (status/'failed.json').exists():return 'failed',None
    if (status/'scheduler.json').exists():
        state=json.loads((status/'scheduler.json').read_text()).get('state','').upper()
        if state in ('RUNNING','COMPLETING'):return 'running',None
        if state in ('PENDING','CONFIGURING','REQUEUED','SUSPENDED'):return 'queued',None
    if (status/'started.json').exists():return 'running',None
    return ('queued' if (root/'submission.json').exists() else 'waiting'),None

def verified_file(folder,name,receipt):
    path=safe_path(folder,name);expected=receipt['files'].get(name)
    if not expected or not path.is_file() or digest(path)!=expected:raise ValueError(f'Unverified completed diagnostic: {name}')
    return path

def load_map(folder,manifest_name,receipt,fields):
    manifest=json.loads(verified_file(folder,manifest_name,receipt).read_text());result={}
    for name in fields:
        entry=manifest.get('arrays',{}).get(name)
        if entry is None:raise ValueError(f'Missing map {name} in {manifest_name}')
        path=verified_file(folder,entry['file'],receipt)
        if entry.get('sha256')!=receipt['files'][entry['file']]:raise ValueError('Map digest differs from completion receipt')
        if entry.get('dtype')!='float64' or entry.get('order')!='F':raise ValueError('Expected Float64 Fortran-order maps')
        shape=entry['shape'];order=manifest.get('byte_order','little')
        dtype='<f8' if order=='little' else '>f8' if order=='big' else None
        if dtype is None:raise ValueError('Unknown byte order')
        data=np.fromfile(path,dtype=dtype)
        if data.size!=np.prod(shape):raise ValueError('Map length differs from declared shape')
        result[name]=finite(data.reshape(shape,order='F'),name)
    return manifest,result

def cache_key(receipt):
    return hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest()

def cache_valid(path,key):
    marker=path/'cache.json'
    if not marker.exists():return False
    saved=json.loads(marker.read_text())
    return saved.get('key')==key and all((path/name).is_file() and digest(path/name)==sha for name,sha in saved.get('files',{}).items()) and bool(saved.get('files'))

def complete_cache(temporary,target,key):
    files={str(p.relative_to(temporary)):digest(p) for p in temporary.rglob('*') if p.is_file()}
    write_json(temporary/'cache.json',{'key':key,'files':files})
    if target.exists():shutil.rmtree(target)
    temporary.replace(target)

def regions(z,dz,k):
    return [('full',np.ones(len(z),dtype=bool)),('stokes_layer',(z>=-1/(2*k))&(z<=0))]

def analyze_run(root,entry,receipt,cache):
    key=cache_key(receipt)
    if cache_valid(cache,key):return json.loads((cache/'summary.json').read_text())
    folder=safe_path(root,entry['path']);config=json.loads(verified_file(folder,'configuration.json',receipt).read_text());grid=json.loads(verified_file(folder,'grid.json',receipt).read_text())
    completed=json.loads(verified_file(folder,'completed.json',receipt).read_text())
    if completed.get('initial_condition_sha256')!=config.get('initial_condition_sha256'):raise ValueError('Initial-condition receipts differ')
    budget=read_csv(verified_file(folder,'regime-budget.csv',receipt));jet=read_csv(verified_file(folder,'jet-budget.csv',receipt))
    times=[r['time_s'] for r in budget]
    if len(times)<2 or np.any(np.diff(times)<=0) or not np.isclose(times[-1],completed.get('time',completed.get('time_s')),atol=1e-7,rtol=0):raise ValueError('Invalid completed budget time grid')
    manifests=sorted(n for n in receipt['files'] if re.fullmatch(r'mean_[0-9]+\.[0-9]+\.json',n))
    if len(manifests)<3:raise ValueError('At least three verified mean maps are required')
    mt=[];q=[];pair={name:[] for name in ('Uc','Wc','Q')};mapmeta=[]
    for name in manifests:
        meta,data=load_map(folder,name,receipt,('Uc','Wc','Q'));mt.append(float(meta.get('time',meta.get('time_s'))));mapmeta.append(meta)
        for field in pair:pair[field].append(data[field])
        q.append(data['Q'])
    if np.any(np.diff(mt)<=0) or not np.isclose(mt[-1],times[-1],atol=1e-7,rtol=0):raise ValueError('Mean-map times do not span completed run')
    z=finite(grid['z_centers']);zf=finite(grid['z_faces']);dz=np.diff(zf);x=finite(grid['x_centers']);dx=float(config['lx'])/len(x)
    trim=int(mapmeta[0].get('recommended_continuum_z_trim_cells',mapmeta[0].get('recommended_edge_exclusion',grid.get('recommended_continuum_z_trim_cells',2))));sel=slice(trim,len(z)-trim if trim else None)
    if trim<0 or len(z)-2*trim<4:raise ValueError('Invalid map boundary exclusion')
    zs,dzs=z[sel],dz[sel];delta=1/(2*float(config['k']));kernels={};filterinfo=[]
    for name,sx,sz in FILTERS:
        kx,wx=gaussian_matrix(x,np.full(len(x),dx),sx*delta);kz,wz=gaussian_matrix(zs,dzs,sz*delta)
        kernels[name]=(kx,kz);filterinfo.append({'id':name,'sigma_x_over_stokes_depth':sx,'sigma_z_over_stokes_depth':sz,'sigma_x_m':sx*delta,'sigma_z_m':sz*delta,'x_kernel':{'effective_sigma_min_m':float(wx.min()),'effective_sigma_max_m':float(wx.max()),'maximum_centroid_shift_m':float(np.max(np.abs(kx@x-x)))},'z_kernel':{'effective_sigma_min_m':float(wz.min()),'effective_sigma_max_m':float(wz.max()),'maximum_centroid_shift_m':float(np.max(np.abs(kz@zs-zs)))},'truncate_sigma':4})
    metrics=[];flux=[];qinitial=q[0][:,sel]
    tau=float(config['injection_wavelength'])/float(config['eddy_rms'])
    for i,name in enumerate(manifests):
        _,allfields=load_map(folder,name,receipt,(*CENTER_FIELDS,*NATIVE_FIELDS));center={n:allfields[n][:,sel] for n in CENTER_FIELDS}
        for stride in (1,2):
            qt=centered_time_derivative(mt,q,i,stride)
            if qt is None and stride==2:continue
            if qt is not None:qt=qt[:,sel]
            for filterid,_,_ in FILTERS:
                kx,kz=kernels[filterid];f={n:filter_field(a,kx,kz) for n,a in center.items()};fi=filter_field(qinitial,kx,kz)
                ft=filter_field(qt,kx,kz) if qt is not None else None
                for region,mask in regions(zs,dzs,float(config['k'])):
                    if not np.any(mask):continue
                    local={n:a[:,mask] for n,a in f.items()};localqt=ft[:,mask] if ft is not None else None
                    scalars=scalar_metrics(local|({'Qt':localqt} if localqt is not None else {}),dzs[mask],tau)
                    native=feedback_metrics(local,dzs[mask],fi[:,mask],localqt)
                    native_scalar={f'native_state_{key}':v for key,v in scalar_metrics(local|{'Qt':local['Qdot_total']},dzs[mask],tau).items()}
                    native.update(native_scalar)
                    metrics.append({'time_s':mt[i],'filter_id':filterid,'region':region,'temporal_stride':stride if qt is not None else 0,'z_min_m':float(zs[mask].min()),'z_max_m':float(zs[mask].max()),'volume_m3':float(config['lx'])*float(config['ly'])*dzs[mask].sum(),**scalars,**native})
        for term,un,wn in [('advection_full','U_adv_full','W_adv_full'),('advection_mean','U_adv_mean','W_adv_mean'),('wave','U_wave','W_wave'),('viscous','U_viscous','W_viscous'),('forcing','U_forcing','W_forcing'),('advection_3d',None,None)]:
            fu=allfields[un] if un else allfields['U_adv_full']-allfields['U_adv_mean'];fw=allfields[wn] if wn else allfields['W_adv_full']-allfields['W_adv_mean']
            s=x_mode_supply(allfields['U'],allfields['W'],fu,fw,dz,float(config['lx']))
            flux.extend({'time_s':mt[i],'term':term,'mode':int(m),'kx_rad_m':float(k),'mode_supply_m2_s3':float(v),'cumulative_supply_m2_s3':float(c),'all_modes_supply_m2_s3':s['sum_supply']} for m,k,v,c in zip(s['mode'],s['kx_rad_m'],s['supply'],s['cumulative_supply']))
    et=[r.get('Eperp',r.get('Etransverse_modes',r.get('Etotal',0)-r.get('Eky0',0))) for r in budget]
    ed=[r.get('Eeddy') for r in budget]
    if any(v is None for v in et+ed):raise ValueError('Missing transverse-mode or eddy energies')
    growth=transverse_growth(times,et,ed,float(config['perturb_energy_ratio']))
    signed_jet_supplies={}
    jt=np.array([r['time_s'] for r in jet])
    for term in ('adv_ky0','adv_3d','wave','closure','forcing','pressure','total'):
        values=[r.get(f'P_{term}_anomaly') for r in jet]
        if all(isinstance(v,(int,float)) and math.isfinite(v) for v in values):
            signed_jet_supplies[term]={'signed_integral_m2_s2':float(np.trapezoid(values,jt)),'absolute_integral_m2_s2':float(np.trapezoid(np.abs(values),jt))}
    native_qc={key:max((r[key] for r in budget if isinstance(r.get(key),(int,float))),default=None) for key in ('native_divergence_max','native_divergence_rms','native_pressure_curl_max','viscous_cfl')}
    valid_residual=[r for r in jet if isinstance(r.get('interval_residual_anomaly'),(int,float)) and (r.get('interval_dt') or 0)>0]
    closure={'cumulative_residual_m2_s2':jet[-1].get('cumulative_residual_anomaly'),'mean_absolute_interval_residual_m2_s3':sum(abs(r['interval_residual_anomaly'])*r['interval_dt'] for r in valid_residual)/sum(r['interval_dt'] for r in valid_residual) if valid_residual else None}
    for row,e in zip(budget,et):row['Eperp']=e;row['perp_eddy_fraction']=ratio(e,row['Eeddy'])
    # Public metadata are allow-listed; no local paths, environment or scheduler details.
    result={'id':entry['id'],'status':'complete','source_version':receipt['source_version'],'receipt_sha256':key,'config':{k:config[k] for k in PUBLIC_CONFIG if k in config},'final_time_s':times[-1],'maps':len(mt),'diagnostic_times':len(times),'boundary_exclusion_cells':trim,'retained_z_centers_m':[float(zs.min()),float(zs.max())],'filters':filterinfo,'transverse_growth':growth,'signed_jet_anomaly_supplies':signed_jet_supplies,'native_qc':native_qc,'jet_budget_residual':closure,'final_energies':{k:(jet[-1].get(k) if k=='Ejet_anomaly' else budget[-1].get(k)) for k in ('Etotal','Eky0','Eeddy','Eperp','perp_eddy_fraction','Ejet_anomaly')},'final_metrics':[r for r in metrics if r['time_s']==mt[-1] and r['filter_id']=='unfiltered'],'scope':'Finite-time individual realization. Native mean-feedback, scalar continuum balance and transverse energy are separate diagnostics; no single ratio assigns a regime.'}
    cache.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix=cache.name+'.tmp-',dir=cache.parent))
    try:
        write_json(tmp/'summary.json',result);write_csv(tmp/'metrics.csv',metrics);write_csv(tmp/'mode-supply.csv',flux);write_csv(tmp/'regime-budget.csv',budget);write_csv(tmp/'jet-budget.csv',jet)
        np.savez_compressed(tmp/'pair-fields.npz',time=np.array(mt),z=z,dz=dz,**{n:np.array(a) for n,a in pair.items()})
        complete_cache(tmp,cache,key)
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True);raise
    return result


def matched_pairs(entries,results,cache_root):
    byid={r['id']:r for r in results};groups={}
    for e in entries:
        # Root declares a common matched_group; a baseline_id is also supported.
        group=e.get('matched_group',e.get('cell',e.get('family',e.get('group'))))
        groups.setdefault((group,e.get('seed',e.get('env',{}).get('SEED'))),[]).append(e)
    pairs=[];twins=[]
    for group,groupentries in groups.items():
        available=[e for e in groupentries if byid[e['id']]['status']=='complete']
        baselines=[e for e in available if float(byid[e['id']]['config']['perturb_energy_ratio'])==0]
        if len(baselines)!=1:continue
        base=baselines[0];br=byid[base['id']];bpath=cache_root/base['id'];b=np.load(bpath/'pair-fields.npz')
        for entry in available:
            if entry==base:continue
            r=byid[entry['id']];config=r['config']
            for key in MATCH_KEYS:
                if key not in config or config[key]!=br['config'].get(key):raise ValueError(f'Unmatched {key}: {entry["id"]} / {base["id"]}')
            for key in ('nu','viscosity','maintain_background','advection'):
                if config.get(key)!=br['config'].get(key):raise ValueError(f'Unmatched physical/numerical parameter {key}')
            pcache=cache_root/'pairs'/entry['id'];pkey=hashlib.sha256((r['receipt_sha256']+br['receipt_sha256']).encode()).hexdigest()
            if cache_valid(pcache,pkey):data=read_csv(pcache/'comparison.csv')
            else:
                a=np.load(cache_root/entry['id']/'pair-fields.npz')
                if a['time'].shape!=b['time'].shape or not np.allclose(a['time'],b['time'],atol=1e-8,rtol=0) or not np.array_equal(a['z'],b['z']):raise ValueError('Matched maps differ in time or vertical grid')
                data=[];delta=1/(2*config['k']);x=(np.arange(config['nx'])+.5)*config['lx']/config['nx']
                trim=r['boundary_exclusion_cells']
                if trim!=br['boundary_exclusion_cells']:raise ValueError('Matched map exclusions differ')
                sl=slice(trim,len(a['z'])-trim if trim else None);pz=a['z'][sl];pdz=a['dz'][sl]
                for filterid,sx,sz in FILTERS:
                    kx,_=gaussian_matrix(x,np.full(len(x),config['lx']/len(x)),sx*delta);kz,_=gaussian_matrix(pz,pdz,sz*delta)
                    initial_all={n:filter_field(b[n][0][:,sl],kx,kz) for n in ('Uc','Wc','Q')}
                    for i,time in enumerate(a['time']):
                        af={n:filter_field(a[n][i][:,sl],kx,kz) for n in initial_all};bf={n:filter_field(b[n][i][:,sl],kx,kz) for n in initial_all}
                        for region,mask in regions(pz,pdz,config['k']):
                            if not mask.any():continue
                            actual={n:af[n][:,mask] for n in af};baseline={n:bf[n][:,mask] for n in bf};initial={n:initial_all[n][:,mask] for n in af}
                            data.append({'run_id':entry['id'],'baseline_id':base['id'],'time_s':float(time),'region':region,'filter_id':filterid,'z_min_m':float(pz[mask].min()),'z_max_m':float(pz[mask].max()),'volume_m3':config['lx']*config['ly']*float(pdz[mask].sum()),**paired_fields(actual,baseline,pdz[mask],initial,config['eddy_rms'])})
                pcache.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix=pcache.name+'.tmp-',dir=pcache.parent));write_csv(tmp/'comparison.csv',data);complete_cache(tmp,pcache,pkey)
            pairs.extend(data)
        small=sorted([e for e in available if 0<float(byid[e['id']]['config']['perturb_energy_ratio'])<.01],key=lambda e:byid[e['id']]['config']['perturb_energy_ratio'])
        if len(small)>=2:
            a,c=small[0],small[1];ab=read_csv(cache_root/a['id']/'regime-budget.csv');cb=read_csv(cache_root/c['id']/'regime-budget.csv')
            if byid[a['id']]['config'].get('perturbation_shape_sha256')!=byid[c['id']]['config'].get('perturbation_shape_sha256'):raise ValueError('Twin perturbation shapes differ')
            collapse=twin_collapse([r['time_s'] for r in ab],[r['Eperp'] for r in ab],byid[a['id']]['config']['perturb_energy_ratio'],[r['time_s'] for r in cb],[r['Eperp'] for r in cb],byid[c['id']]['config']['perturb_energy_ratio'])
            qualification=qualified_twin_growth([r['time_s'] for r in ab],[r['Eperp'] for r in ab],byid[a['id']]['config']['perturb_energy_ratio'],[r['Eeddy'] for r in ab],[r['Eperp'] for r in cb],byid[c['id']]['config']['perturb_energy_ratio'],[r['Eeddy'] for r in cb])
            qualification['smaller_run']=a['id'];qualification['larger_run']=c['id']
            byid[c['id']]['qualified_transverse_growth']=qualification
            byid[a['id']]['qualified_transverse_growth']=qualification
            for i,t in enumerate(collapse['time_s']):
                twins.append({'smaller_run':a['id'],'larger_run':c['id'],'time_s':float(t),'scaled_amplitude_smaller':float(collapse['scaled_amplitude_a'][i]),'scaled_amplitude_larger':float(collapse['scaled_amplitude_b'][i]),'relative_amplitude_difference':float(collapse['relative_difference'][i]),'both_below_one_percent_eddy_energy':ab[i]['perp_eddy_fraction']<.01 and cb[i]['perp_eddy_fraction']<.01})
    return pairs,twins


def plot_parameter_cells(results,pairs,twins,cache_root,output):
    import matplotlib.pyplot as plt
    complete=[r for r in results if r['status']=='complete' and r.get('experiment')=='primary' and r.get('seed')==1]
    if not complete:return None
    fig,axes=plt.subplots(1,3,figsize=(12.5,4),layout='constrained')
    specs=[('Initial transverse growth',r'Candidate amplitude rate $\gamma\,\tau_e$'),('Matched x–z mean dynamics','Final velocity error / initial eddy speed'),('Changed jet profile','Signed integrated 3D supply / initial eddy KE')]
    for j,(title,label) in enumerate(specs):
        records=[]
        for r in complete:
            c=r['config'];eta=c['perturb_energy_ratio'];value=None
            if j==0 and eta==1e-6 and r.get('qualified_transverse_growth',{}).get('available'):
                g=r['qualified_transverse_growth'];value=g['amplitude_growth_rate_s_inv']*c['injection_wavelength']/c['eddy_rms']
            elif j==1 and eta==1:
                p=[a for a in pairs if a['run_id']==r['id'] and a['region']=='stokes_layer' and a.get('filter_id','unfiltered')=='unfiltered']
                if p:value=p[-1]['inplane_error_to_initial_eddy']
            elif j==2 and eta==1:
                p=r.get('signed_jet_anomaly_supplies',{}).get('adv_3d',{}).get('signed_integral_m2_s2')
                if p is not None:value=p/(.5*c['eddy_rms']**2)
            if value is not None:records.append((1000*c['eddy_rms'],c['alpha'],value))
        if records:
            x,y,v=map(np.array,zip(*records));extent=max(np.max(np.abs(v)),1.)
            sc=axes[j].scatter(x,y,c=v,s=500,cmap='RdBu_r',vmin=-extent if j!=1 else 0,vmax=extent,edgecolor='white',linewidth=2)
            for xx,yy,vv in records:axes[j].annotate(f'{vv:.2g}',(xx,yy),ha='center',va='center',fontsize=10,color='#172b38',bbox=dict(fc='white',ec='none',alpha=.7,pad=1.5))
            cb=fig.colorbar(sc,ax=axes[j],orientation='horizontal',pad=.18);cb.set_label(label,fontsize=8)
        else:axes[j].text(.5,.5,'Required matched runs pending',transform=axes[j].transAxes,ha='center',fontsize=9,color='#627078')
        axes[j].set(title=title,xscale='log',xlim=(.8,18),ylim=(.4,1.1),xlabel='Initial eddy speed (mm s⁻¹)',ylabel='Background parameter α')
        axes[j].set_xticks([1.2,4,12],['1.2','4','12']);axes[j].set_yticks([.5,.75,1.]);axes[j].grid(alpha=.15)
    fig.suptitle('Primary parameter cells · finite-time measures, with no imposed regime threshold',fontsize=12)
    path='parameter-cells.png';fig.savefig(output/path,dpi=170);plt.close(fig);return path


def plot_scale_supply(results,cache_root,output):
    import matplotlib.pyplot as plt
    candidates=[r for r in results if r['status']=='complete' and r.get('experiment')=='primary' and r.get('seed')==1 and r['config']['perturb_energy_ratio']==1]
    if not candidates:return None
    r=sorted(candidates,key=lambda r:(abs(r['config']['alpha']-.75),abs(r['config']['eddy_rms']-.004)))[0]
    rows=read_csv(cache_root/r['id']/'mode-supply.csv');end=r['final_time_s'];fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    styles=[('advection_mean','Advection of y mean','#246382'),('advection_3d','3D-to-mean advection difference','#ba5143'),('wave','Wave term','#8076a5'),('viscous','Explicit viscosity','#666666'),('forcing','Background-maintenance force','#b5882e')]
    for term,label,color in styles:
        sub=[x for x in rows if x['term']==term and x['time_s']>=.5*end];times=sorted({x['time_s'] for x in sub});modes=sorted({int(x['mode']) for x in sub})
        if len(times)<2:continue
        for ax,key in zip(axes,('mode_supply_m2_s3','cumulative_supply_m2_s3')):
            average=[]
            for mode in modes:
                selected=sorted((x for x in sub if x['mode']==mode),key=lambda x:x['time_s'])
                average.append(float(np.trapezoid([x[key] for x in selected],[x['time_s'] for x in selected])/(times[-1]-times[0])))
            ax.plot(modes,average,label=label,color=color,lw=1.5)
    axes[0].set_ylabel('Signed supply by x mode (m² s⁻³)');axes[1].set_ylabel('Cumulative supply through x mode (m² s⁻³)')
    for ax in axes:ax.axhline(0,color='#999',lw=.7);ax.set_xlabel('Horizontal x Fourier mode, including mode 0');ax.grid(alpha=.15);ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
    axes[0].legend(fontsize=7);fig.suptitle(r['id']+' · average over the latter half of the run\nSupplies to y-mean in-plane kinetic energy; 3D exchange is not a conservative cascade flux',fontsize=10)
    path='horizontal-supplies.png';fig.savefig(output/path,dpi=170);plt.close(fig);return path


def plot_campaign(results,pairs,twins,cache_root,output):
    complete=[r for r in results if r['status']=='complete']
    if not complete:return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    complete=[r for r in results if r['status']=='complete'];paths=[]
    if not complete:return paths
    gridplot=plot_parameter_cells(results,pairs,twins,cache_root,output)
    if gridplot:paths.append(gridplot)
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    colors={0.:'#246382',1e-8:'#8c79b5',1e-6:'#d28c35',1.:'#ba5143'}
    for r in complete:
        c=r['config'];eta=float(c['perturb_energy_ratio']);color=colors.get(eta,'#777');rows=read_csv(cache_root/r['id']/'regime-budget.csv');t=np.array([a['time_s'] for a in rows]);turnover=c['injection_wavelength']/c['eddy_rms']
        if eta>0:
            e=np.array([a['Eperp'] for a in rows]);scale=e[0]
            if scale>0:axes[0].plot(t/turnover,np.sqrt(e/scale),color=color,alpha=.5,lw=1)
        axes[1].plot(t/turnover,[a['perp_eddy_fraction'] for a in rows],color=color,alpha=.5,lw=1)
        selected=[p for p in pairs if p['run_id']==r['id'] and p['region']=='stokes_layer' and p.get('filter_id','unfiltered')=='unfiltered']
        if selected:axes[2].plot([a['time_s']/turnover for a in selected],[a['inplane_error_to_initial_eddy'] for a in selected],color=color,alpha=.5,lw=1)
    axes[0].set(yscale='log',ylabel='Transverse amplitude / initial amplitude');axes[1].set(ylabel='Transverse-mode energy / eddy energy');axes[2].set(ylabel='Matched mean-field error / initial eddy speed')
    for ax in axes:ax.set_xlabel('Time / initial eddy turnover time');ax.grid(alpha=.18)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0],[0],color=colors[e],label=l) for e,l in [(0.,'2D'),(1e-8,'Tiny 3D, η=10⁻⁸'),(1e-6,'Small 3D, η=10⁻⁶'),(1.,'Finite 3D, η=1')]],loc='outside lower center',ncol=4,fontsize=8)
    fig.suptitle('Geometry, transverse growth and matched dynamics are distinct tests',fontsize=12)
    fig.savefig(output/'regime-overview.png',dpi=160);plt.close(fig);paths.append('regime-overview.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for r in complete:
        eta=float(r['config']['perturb_energy_ratio']);color=colors.get(eta,'#777');rows=read_csv(cache_root/r['id']/'metrics.csv')
        selected=[a for a in rows if a['region']=='stokes_layer' and a['filter_id']=='g2' and a['temporal_stride']==1]
        if not selected:continue
        axes[0].plot([a['time_s'] for a in selected],[a['native_3d_to_mean_terms'] for a in selected],color=color,lw=1,alpha=.5)
        axes[1].plot([a['time_s'] for a in selected],[a.get('rbar_to_scalar_evolution') for a in selected],color=color,lw=1,alpha=.5)
    axes[0].set_ylabel('Native 3D curl correction / retained term norms');axes[1].set_ylabel('Scalar R̄ / (‖Qₜ‖ + ‖A‖)')
    for ax in axes:ax.set_xlabel('Time (s)');ax.grid(alpha=.18)
    fig.suptitle('Upper Stokes layer · Gaussian σx=2δS, σz=δS/2\nTwo different equations: native mean feedback and scalar transport with in-plane eddies retained',fontsize=10)
    fig.savefig(output/'feedback-overview.png',dpi=160);plt.close(fig);paths.append('feedback-overview.png')
    supplyplot=plot_scale_supply(results,cache_root,output)
    if supplyplot:paths.append(supplyplot)
    return paths


def validate_packet(packet):
    if packet.get('schema_version')!=1 or packet.get('completed') is not True:raise ValueError('Packet reference is not a completed compatible report')
    if {f['id'] for f in packet['filters']}!={f[0] for f in FILTERS}:raise ValueError('Packet filters do not match declared common filter family')
    for row in packet['rows']:
        scale=row['Qt_rms']+row['A_rms'];r=ratio(row['Rbar_rms'],scale)
        if r is not None and not math.isclose(r,row['rbar_to_scalar_evolution'],rel_tol=1e-10,abs_tol=1e-12):raise ValueError('Packet scalar normalization mismatch')
        expected=ratio(row['scalar_unclosed_rms'],scale+row['Rbar_rms'])
        if expected is not None and not math.isclose(expected,row['unclosed_to_scalar_budget'],rel_tol=1e-10,abs_tol=1e-12):raise ValueError('Packet residual normalization mismatch')
        if any(not math.isfinite(row[k]) or row[k]<0 for k in ('Qt_rms','A_rms','Rbar_rms','scalar_unclosed_rms')):raise ValueError('Invalid packet norms')


def plot_packet(packet,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    peak_times=sorted({r['time_s'] for r in packet.get('peak_core_all_stencils',[])})
    if not peak_times:return None
    peak=peak_times[len(peak_times)//2]
    selected=[r for r in packet['rows'] if r['averaging']=='coherent' and r['member']=='packet_turbulence' and r['region']=='stokes_layer' and r['age_bin']=='core' and math.isclose(r['time_s'],peak,abs_tol=1e-8)]
    widths={f['id']:f['sigma_x_over_stokes_depth'] for f in packet['filters']}
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for space,color in [(1,'#246382'),(2,'#b65042')]:
        for temporal,style in [(1,'-'),(2,'--')]:
            rows=sorted((r for r in selected if r['stencil_step']==space and r['temporal_stride']==temporal),key=lambda r:widths[r['filter_id']])
            for ax,key in zip(axes,('rbar_to_scalar_evolution','unclosed_to_scalar_budget')):
                ax.plot([widths[r['filter_id']] for r in rows],[r[key] for r in rows],color=color,linestyle=style,marker='o',ms=4,label=f'Spatial step {space}, temporal stride {temporal}')
    axes[0].set_ylabel('Scalar R̄ / (‖Qₜ‖ + ‖A‖)');axes[1].set_ylabel('Scalar residual / budget norms')
    for ax in axes:ax.set_xlabel('Gaussian σx / Stokes depth');ax.set_xticks([0,1,2,4]);ax.set_ylim(bottom=0);ax.grid(alpha=.15)
    axes[0].legend(fontsize=8);fig.suptitle(f'Laboratory-reproduction packet: t={peak:.2f} s, local core, upper Stokes layer\nSeven signed realizations averaged before norms; σz/δS = 0, ¼, ½, 1',fontsize=10)
    name='packet-comparison.png';fig.savefig(output/name,dpi=170);plt.close(fig);return name


def render(summary,output):
    template=(Path(__file__).parent/'template.html').read_text();esc=html.escape
    counts=summary['counts'];rows=[]
    for r in summary['runs']:
        c=r.get('config',{});values=[esc(r['id']),esc(r['status']),f"{1000*c['eddy_rms']:.2g}" if c.get('eddy_rms') else '—',str(c.get('alpha','—')),f"{c['perturb_energy_ratio']:g}" if 'perturb_energy_ratio'in c else '—',str(r.get('final_time_s','—'))]
        if r['status']=='complete':values[0]=f'<a href="runs/{esc(r["id"])}/metrics.csv">{values[0]}</a>'
        rows.append('<tr data-status="'+esc(r['status'])+'">'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
    captions={
        'parameter-cells.png':('Finite-time parameter-cell measurements','Each point is one primary seed. Growth candidates require a contiguous window with both perturbations below 1% of eddy energy, scaled amplitude mismatch ≤5%, at least five outputs, and log-amplitude fit RMS ≤0.2. These qualify a numerical response, not a regime boundary; windows and rejected fits remain in the data. The final x–z field error and the signed 3D supply to changed-jet energy are distinct outcomes. Blank cells have no verified matched result yet.'),
        'regime-overview.png':('Transverse growth, energy partition and matched field error','Each line is one realization; colours mark initial 3D perturbation energy. Matched field errors retain all x–z eddies. Long-time trajectory separation alone does not establish failure of statistical mean dynamics. Finite perturbations add initial energy.'),
        'feedback-overview.png':('Native mean feedback and scalar-law correction','Signed fields are filtered before their norms. The left panel uses the native full-minus-y-mean tendency; the right uses the averaged scalar law, with in-plane eddy transport retained. These are different quantities. Absolute terms and temporal residuals are exported.'),
        'horizontal-supplies.png':('Signed horizontal-scale energy supplies','The selected finite-3D run is identified in the figure. Curves include x mode zero, so jet and mean-shear exchange remain visible. The integrated 3D contribution exchanges energy with transverse motions; it is not labelled a conservative inverse flux.')}
    figures=''.join('<figure><img src="'+p+'" alt="'+captions[p][0]+'"><figcaption>'+captions[p][1]+'</figcaption></figure>' for p in summary['figures'])
    if not figures:figures='<div class="empty"><h3>Results will appear as completed runs are verified.</h3><p>The campaign tests growth, mean-flow agreement and signed transfer separately. Pending simulations are not assigned a physical regime.</p></div>'
    packet=summary.get('packet_reference');packetbody='<p>The packet comparison is awaiting its verified filtered diagnostic export.</p>'
    if packet:
        packetbody='<p>The laboratory-reproduction packet is shown using the same scalar equation and Gaussian filter widths measured in Stokes depths. It is transient, has different boundaries and initial turbulence, and has no matched autonomous 2D run in this dataset.</p><p><a href="packet-reference.json">Packet diagnostics, windows and provenance</a></p>'
        if summary.get('packet_figure'):packetbody+='<figure><img src="'+summary['packet_figure']+'" alt="Filtered packet scalar source and diagnostic residual across derivative stencils"><figcaption>Filtering reduces the diagnostic residual in this packet while the scalar 3D source remains appreciable. Spatial-stencil sensitivity is visible. This does not supply a native counterfactual budget or a matched autonomous 2D experiment.</figcaption></figure>'
        p=packet.get('rows',[])
        peak_times=sorted({r['time_s'] for r in packet.get('peak_core_all_stencils',[])})
        peak=peak_times[len(peak_times)//2] if peak_times else None
        chosen=[r for r in p if r.get('averaging')=='coherent' and r.get('member')=='packet_turbulence' and r.get('region')=='stokes_layer' and r.get('age_bin')=='core' and r.get('stencil_step') in (1,2) and r.get('temporal_stride')==1 and (peak is None or math.isclose(r['time_s'],peak,abs_tol=1e-8))]
        if chosen:
            columns=['filter_id','stencil_step','rbar_to_scalar_evolution','unclosed_to_scalar_budget'];packetbody+='<div class="table-wrap"><table><thead><tr><th>Filter</th><th>Spatial derivative step</th><th>R̄ / evolution</th><th>Residual / budget</th></tr></thead><tbody>'
            for r in chosen[:24]:packetbody+='<tr>'+''.join('<td>'+esc(f'{r[k]:.3g}' if isinstance(r.get(k),(int,float)) else str(r.get(k,'—')))+'</td>' for k in columns)+'</tr>'
            packetbody+='</tbody></table></div>'
    text=template.replace('{{COUNTS}}',' · '.join(f'{counts.get(k,0)} {k}' for k in ('complete','running','queued','waiting','failed'))).replace('{{RUNS}}',''.join(rows)).replace('{{FIGURES}}',figures).replace('{{PACKET}}',packetbody).replace('{{UPDATED}}',esc(summary['generated_at'])).replace('{{VERSION}}',esc(summary['source_version'])).replace('{{TOTAL}}',str(len(summary['runs']))).replace('{{DOWNLOADS}}',' · '.join(link for count,link in [(summary.get('matched_field_rows',0),'<a href="matched-fields.csv">Matched field comparisons</a>'),(summary.get('twin_rows',0),'<a href="twin-amplitudes.csv">Twin-amplitude comparisons</a>'),(summary.get('sensitivity_rows',0),'<a href="sensitivity-summary.csv">Numerical and physical controls</a>'),(1,'<a href="summary.json">All results &amp; definitions</a>')] if count))
    (output/'index.html').write_text(text);shutil.copyfile(Path(__file__).parent/'style.css',output/'style.css')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--approved-runs',type=Path);a=parser.parse_args()
    root=a.campaign_root.resolve();output=a.output.resolve();manifest=json.loads((root/'campaign.json').read_text());source=json.loads((root/'source-manifest.json').read_text());entries=manifest['runs']
    ids=[identifier(e['id']) for e in entries]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate campaign IDs')
    approved=None if a.approved_runs is None else json.loads(a.approved_runs.read_text())
    if approved is not None:
        if not isinstance(approved,list) or len(approved)!=len(set(approved)) or any(x not in ids for x in approved):raise ValueError('Approved-run snapshot must contain unique campaign IDs')
        approved=set(approved)
    output.mkdir(parents=True,exist_ok=True);cache=root/'data/analysis-cache';results=[]
    (cache/'matplotlib').mkdir(parents=True,exist_ok=True);os.environ.setdefault('MPLCONFIGDIR',str(cache/'matplotlib'))
    for entry in entries:
        status,receipt=campaign_status(root,entry,source['version'])
        if status=='complete' and approved is not None and entry['id'] not in approved:status='waiting';receipt=None
        if status=='complete':
            result=analyze_run(root,entry,receipt,cache/entry['id']);dest=output/'runs'/entry['id'];dest.mkdir(parents=True,exist_ok=True)
            for name in ('summary.json','metrics.csv','mode-supply.csv','regime-budget.csv','jet-budget.csv'):shutil.copyfile(cache/entry['id']/name,dest/name)
        else:result={'id':entry['id'],'status':status}
        result['experiment']=entry.get('experiment');result['group']=entry.get('group');result['cell']=entry.get('cell');result['seed']=entry.get('seed');results.append(result)
    pairs,twins=matched_pairs(entries,results,cache)
    for r in results:
        if r['status']=='complete':write_json(output/'runs'/r['id']/'summary.json',r)
    write_csv(output/'matched-fields.csv',pairs);write_csv(output/'twin-amplitudes.csv',twins)
    sensitivity=[]
    for r in results:
        if r['status']!='complete':continue
        c=r['config'];m=[x for x in r['final_metrics'] if x['region']=='stokes_layer' and x.get('filter_id','unfiltered')=='unfiltered']
        selected=m[-1] if m else {}
        p=[x for x in pairs if x['run_id']==r['id'] and x['region']=='stokes_layer' and x.get('filter_id','unfiltered')=='unfiltered']
        sensitivity.append({'run_id':r['id'],'cell':r.get('cell'),'experiment':r.get('experiment'),'seed':r.get('seed'),'eddy_rms_m_s':c['eddy_rms'],'alpha':c['alpha'],'eta':c['perturb_energy_ratio'],'nu_m2_s':c.get('nu'),'nx':c['nx'],'ny':c['ny'],'nz':c['nz'],'ly_m':c['ly'],'max_dt_s':c['max_dt'],'final_time_s':r['final_time_s'],'Ejet_anomaly':r['final_energies'].get('Ejet_anomaly'),'Eperp':r['final_energies'].get('Eperp'),'native_3d_to_mean_terms':selected.get('native_3d_to_mean_terms'),'native_3d_correction_rms':selected.get('native_3d_correction_rms'),'paired_velocity_error_to_initial':p[-1]['inplane_error_to_initial_eddy'] if p else None,'paired_jet_profile_error_rms':p[-1]['jet_profile_error_rms'] if p else None,'max_divergence':r['native_qc'].get('native_divergence_max'),'max_pressure_curl':r['native_qc'].get('native_pressure_curl_max'),'jet_budget_cumulative_residual':r['jet_budget_residual'].get('cumulative_residual_m2_s2')})
    write_csv(output/'sensitivity-summary.csv',sensitivity)
    figures=plot_campaign(results,pairs,twins,cache,output);counts={s:sum(r['status']==s for r in results) for s in ('complete','running','queued','waiting','failed')}
    summary={'schema_version':1,'source_version':source['version'],'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'counts':counts,'runs':results,'figures':figures,'matched_field_rows':len(pairs),'twin_rows':len(twins),'sensitivity_rows':len(sensitivity),'methods':{'native_feedback':'Curl of fresh native advection of full flow, averaged in y, minus fresh native advection evaluated on its y-mean flow. Wave, explicit viscosity and maintained-background forcing remain in the retained mean equation.','scalar_law':'Qt+A=Rbar+viscous curl+forcing curl, A=mean_y(u qx+w qz). A retains in-plane eddy transport. This scalar Rbar is not the native counterfactual curl difference.','native_normalization':'Native 3D RMS divided by the sum of mean-advection RMS, wave RMS and RMS of viscosity plus compensating background force combined. The latter avoids diluting feedback with the two separately large laminar-maintenance terms. Raw component norms are also reported.',
        'filter':'Nonperiodic Gaussian quadrature truncated at four sigma and row-normalized, applied to already formed signed terms. Widths scaled by deltaS=1/(2k); actual boundary widths and retained depths reported.','temporal':'Centered finite differences of saved Q maps, compared with fresh native tendencies. Two temporal strides are reported; residual is not pure dissipation.','mode_supply':'Parseval x-mode supplies to native y-mean in-plane kinetic energy, including mean mode. Cumulative 3D supplies include exchange with transverse modes and are not called a conservative inverse-cascade flux.','matching':'Same base initial field, wave state, viscosity, domain, grid and seed. Full x-z averaged fields and changed x-mean jet profiles compared separately. Finite perturbations add energy; no equal-total-energy causal claim.','classification':'No single chi, transverse-energy fraction or RMS correction determines a regime. Results are finite-time individual realizations; twin-amplitude collapse, numerical controls, signed effects and paired dynamics must be assessed together.'}}
    reference=Path(__file__).parent/'reference/packet-regime-summary.json'
    if reference.exists():
        packet=json.loads(reference.read_text());validate_packet(packet);summary['packet_reference']=packet;shutil.copyfile(reference,output/'packet-reference.json');summary['packet_figure']=plot_packet(packet,output)
        reference_csv=reference.with_name('packet-regime-metrics.csv')
        if reference_csv.exists():shutil.copyfile(reference_csv,output/'packet-metrics.csv')
    write_json(output/'summary.json',summary);render(summary,output)
    print(json.dumps({'reported':True,'counts':counts,'source_version':source['version']}))

if __name__=='__main__':main()
