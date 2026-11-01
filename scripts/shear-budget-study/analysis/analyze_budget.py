#!/usr/bin/env python3
"""Verified direct-3D vorticity and jet budgets, separate from interventions."""
from __future__ import annotations
import argparse, json, html, hashlib, shutil, re, sys, gzip
from pathlib import Path
import numpy as np
import onset_reference as old
import shear_reference as shear
from budget_math import TERMS, FIELD_NAMES, cumulative, mode_budget, identity_error
from budget_comparisons import assess_interventions, assess_resolution

ALL_FIELDS=('Q','Qdot_advection_full','Qdot_advection_mean','Qdot_3d_native','Qdot_wave','Qdot_viscous','Qdot_forcing','Qdot_total')
TERM_LABELS={'A0':'Resolved advection','C':'Direct 3D feedback','W':'Wave','D':'Viscosity','F':'Forcing','P':'Pressure'}
TERM_COLORS={'A0':'#367e9a','C':'#c24a28','W':'#8b5c9f','D':'#647477','F':'#b28b22','P':'#b7b7b7'}


def write_compressed_csv(path,rows):
    """Keep full precision and cadence without duplicating large public CSVs."""
    old.write_csv(path,rows)
    with path.open('rb') as src,(Path(str(path)+'.gz')).open('wb') as dst:
        with gzip.GzipFile(fileobj=dst,mode='wb',filename='',mtime=0,compresslevel=6) as compressed:shutil.copyfileobj(src,compressed)
    path.unlink()


def regions(cfg):
    nx,nz=int(cfg['nx']),int(cfg['nz']);h=float(cfg['h']);z=-h+(np.arange(nz)+.5)*h/nz
    trim=max(2,int(np.ceil(2*nz/96)));inside=np.arange(trim,nz-trim);stokes=inside[z[inside]>=-3/(2*float(cfg['k']))]
    if len(stokes)<4:raise ValueError('Active Stokes region is unresolved')
    return {'full':np.arange(nz),'interior':inside,'stokes':stokes},z,trim


def read_map(run,name):
    manifest=run.json(name);result={}
    for field in ALL_FIELDS:
        e=manifest['arrays'][field]
        if e['dtype']!='float64' or e['order']!='F' or e['sha256']!=run.receipt['files'].get(e['file']):raise ValueError('Native map provenance mismatch')
        x=np.fromfile(run.file(e['file']),dtype='<f8')
        if x.size!=np.prod(e['shape']):raise ValueError('Truncated budget map')
        x=x.reshape(e['shape'],order='F')
        if x.ndim!=2 or not np.all(np.isfinite(x)):raise ValueError('Invalid budget map')
        result[field]=x
    return float(manifest['time']),result


def summarize_moment_rows(rows):
    t=np.array([r['time_s'] for r in rows]);out={}
    for region in ('full','interior','stokes'):
        region_out={}
        for state in ('Q','eddy','profile'):
            energy=np.array([r[f'{region}_Z_{state}'] for r in rows]);powers={term:np.array([r[f'{region}_P_{state}_{term}'] for r in rows]) for term in TERMS}
            total=sum(powers.values());work={term:cumulative(p,t) for term,p in powers.items()};delta=energy-energy[0]
            error=delta-cumulative(total,t);gross=cumulative(sum(abs(p) for p in powers.values()),t)
            scale=max(float(max(abs(delta))),float(max(gross)),1e-16)
            windows=[]
            for lo in np.arange(0,t[-1]-t[0]-199.999,200.):
                ids=np.flatnonzero((t>=t[0]+lo-1e-7)&(t<=t[0]+lo+200+1e-7))
                i,j=int(ids[0]),int(ids[-1]);change=float(energy[j]-energy[i]);signed={term:float(w[j]-w[i]) for term,w in work.items()}
                residual=change-sum(signed.values());gross_window=float(gross[j]-gross[i])
                windows.append(dict(start_elapsed_s=float(lo),stop_elapsed_s=float(lo+200),net_change_s_minus2=change,
                    integrated_signed_terms_s_minus2=signed,temporal_residual_s_minus2=residual,
                    relative_to_change_and_gross=abs(residual)/max(abs(change),gross_window,1e-16),
                    direct_3d_over_window_initial=float(signed['C']/energy[i]) if energy[i]>1e-20 else None))
            coarse_sensitivity={}
            for stride in (2,5,10):
                ids=np.unique(np.r_[np.arange(0,len(t),stride),len(t)-1]);coarse_sensitivity[str(stride)]=dict(
                    max_cadence_s=float(max(np.diff(t[ids]))),
                    signed_final_work_difference_s_minus2={term:float(cumulative(p[ids],t[ids])[-1]-work[term][-1]) for term,p in powers.items()})
            region_out[state]=dict(initial_enstrophy_s_minus2=float(energy[0]),final_enstrophy_s_minus2=float(energy[-1]),
                net_change_s_minus2=float(delta[-1]),integrated_signed_terms_s_minus2={term:float(w[-1]) for term,w in work.items()},
                direct_3d_over_initial=float(work['C'][-1]/energy[0]) if energy[0]>1e-20 else None,
                max_temporal_residual_s_minus2=float(max(abs(error))),relative_to_change_and_gross=float(max(abs(error))/scale),
                relative_to_change=float(max(abs(error))/max(float(max(abs(delta))),1e-16)),
                cadence_s=float(max(np.diff(t))),closure_qualified=bool(max(abs(error))/scale<=.05),windows=windows,
                temporal_sampling_sensitivity=coarse_sensitivity)
            for i,row in enumerate(rows):
                row[f'{region}_int_P_{state}_total']=float(sum(w[i] for w in work.values()))
                row[f'{region}_temporal_residual_{state}']=float(error[i])
                for term,w in work.items():row[f'{region}_int_P_{state}_{term}']=float(w[i])
        out[region]=region_out
    return out


def map_budgets(run,cfg):
    """Stream full maps; retain only selected images and scalar histories."""
    names=sorted(n for n in run.receipt['files'] if re.fullmatch(r'mean_[0-9]+\.[0-9]+\.json',n))
    if len(names)<3:raise ValueError('At least three native map times are required')
    selections,z,trim=regions(cfg);start=float(cfg['start_time']);rows=[];images=[];previous=None;q0=None;template={}
    accumulator=None;direct=None;gross=None;closure={r:dict(error_square=0.,change_square=0.,gross_square=0.,count=0,max_abs_error=0.) for r in selections}
    algebraic=0.;split=0.;relative_3d=0.;max_c=0.;max_total=0.
    for name in names:
        t,m=read_map(run,name);q=m['Q'];terms={k:m[v] for k,v in FIELD_NAMES.items()};total=m['Qdot_total'];dt=0 if previous is None else t-previous[0]
        if previous is not None and dt<=0:raise ValueError('Nonmonotone map times')
        check=identity_error(dict(**terms,total=total));algebraic=max(algebraic,check['relative'])
        split=max(split,float(np.max(abs(m['Qdot_advection_full']-terms['A0']-terms['C']))))
        max_c=max(max_c,float(np.max(abs(terms['C']))));max_total=max(max_total,float(np.max(abs(total))))
        if q0 is None:
            q0=q.copy();accumulator=np.zeros_like(q);direct=np.zeros_like(q);gross=np.zeros_like(q)
            for region,ids in selections.items():
                initial=q[:,ids];template[region]=initial-initial.mean(axis=0,keepdims=True)
        else:
            accumulator+=.5*dt*(previous[1]+total);direct+=.5*dt*(previous[2]+terms['C']);gross+=.5*dt*(previous[3]+sum(abs(v) for v in terms.values()))
        row=dict(time_s=t,elapsed_s=t-start)
        for region,ids in selections.items():
            qq=q[:,ids];qp=np.broadcast_to(qq.mean(axis=0,keepdims=True),qq.shape);qe=qq-qp
            for state,field in [('Q',qq),('eddy',qe),('profile',qp)]:
                row[f'{region}_Z_{state}']=float(.5*np.mean(field**2))
                for term,values in terms.items():row[f'{region}_P_{state}_{term}']=float(np.mean(field*values[:,ids]))
                row[f'{region}_P_{state}_total']=sum(row[f'{region}_P_{state}_{term}'] for term in TERMS)
            for term,values in terms.items():row[f'{region}_source_rms_{term}']=float(np.sqrt(np.mean(values[:,ids]**2)))
            norm=float(np.mean(template[region]**2));row[f'{region}_initial_eddy_projection']=float(np.mean(qe*template[region])/norm) if norm>1e-20 else None
            row[f'{region}_initial_eddy_projection_direct_3d_integral']=float(np.mean(direct[:,ids]*template[region])/norm) if norm>1e-20 else None
            error=(q-q0-accumulator)[:,ids];change=(q-q0)[:,ids];c=closure[region]
            c['error_square']+=float(np.sum(error**2));c['change_square']+=float(np.sum(change**2));c['gross_square']+=float(np.sum(gross[:,ids]**2));c['count']+=error.size;c['max_abs_error']=max(c['max_abs_error'],float(np.max(abs(error))))
            row[f'{region}_Q_integral_error_rms']=float(np.sqrt(np.mean(error**2)))
            row[f'{region}_Q_direct_integral_rms']=float(np.sqrt(np.mean(direct[:,ids]**2)))
        rows.append(row)
        if min(abs(t-start-x) for x in (0,200,400,600,1200))<1e-7 or name==names[-1]:
            images.append(dict(elapsed_s=t-start,Q=q.copy(),Q_eddy=q-q.mean(axis=0,keepdims=True),C=terms['C'].copy(),integrated_C=direct.copy(),Q_change=q-q0,closure_error=q-q0-accumulator))
        previous=(t,total.copy(),terms['C'].copy(),sum(abs(v) for v in terms.values()))
    if abs(rows[0]['time_s']-start)>1e-7 or abs(rows[-1]['time_s']-float(cfg['stop_time']))>1e-7:raise ValueError('Native budget maps miss branch endpoints')
    if algebraic>1e-11 or split>1e-10:raise ValueError('Native tendency split does not close algebraically')
    for c in closure.values():
        er=np.sqrt(c.pop('error_square')/c['count']);change=np.sqrt(c.pop('change_square')/c['count']);grossnorm=np.sqrt(c.pop('gross_square')/c['count'])
        c.update(rms_error=float(er),relative_to_change_and_gross=float(er/max(change,grossnorm,1e-12)),relative_to_change=float(er/max(change,1e-12)),cadence_s=float(max(np.diff([r['time_s'] for r in rows]))));c['closure_qualified']=c['relative_to_change_and_gross']<=.05
    if int(cfg['ny'])==1 and max_c>1e-10*max(max_total,1e-8):raise ValueError('Direct 3D feedback is nonzero in the exactly 2D run')
    return dict(rows=rows,moments=summarize_moment_rows(rows),field_closure=closure,
                algebraic_relative_error=algebraic,split_absolute_error=split,max_abs_C_s_minus2=max_c,
                direct_3d_zero_check_2d=(max_c==0 if int(cfg['ny'])==1 else None),images=images,z=z,trim_cells=trim)


def scalar_budget_rows(run):
    if 'vorticity-budget.csv' not in run.receipt['files']:return None
    rows=run.csv('vorticity-budget.csv')
    needed=[f'{region}_{suffix}' for region in ('full','interior','stokes') for suffix in
            ([f'Z_{state}' for state in ('Q','eddy','profile')]+[f'P_{state}_{term}' for state in ('Q','eddy','profile') for term in TERMS])]
    if any(k not in rows[0] for k in needed):raise ValueError('Signed native vorticity budget schema is incomplete')
    start=rows[0]['time_s']
    for row in rows:row['elapsed_s']=row['time_s']-start
    return rows


def check_scalar_maps(scalar,maps):
    """Independent sampled-map reconstruction agrees with native scalar export."""
    by_time={round(r['time_s'],7):r for r in scalar};errors=[]
    keys=[f'{region}_{suffix}' for region in ('full','interior','stokes') for suffix in
          ([f'Z_{state}' for state in ('Q','eddy','profile')]+[f'P_{state}_{term}' for state in ('Q','eddy','profile') for term in TERMS])]
    for key in keys:
        pairs=[]
        for row in maps:
            match=by_time.get(round(row['time_s'],7))
            if match is None:raise ValueError('Scalar budget lacks a saved map time')
            pairs.append((row[key],match[key]))
        pair=np.array(pairs);scale=max(float(np.max(abs(pair))),1e-16)
        errors.append(float(np.max(abs(pair[:,0]-pair[:,1])))/scale)
    maximum=max(errors)
    if maximum>1e-9:raise ValueError('Scalar moments disagree with independent native map reconstruction')
    return maximum


def jet_budgets(data,cfg):
    p=data['profiles'];z=data['z'];dz=data['dz'];h=float(cfg['h']);t=data['time'];tau=data['elapsed']
    a,norm=old.mode_coefficients(data['UE']-data['background'],h,z,dz)
    mapping={'A0':'adv_ky0','C':'adv_3d','W':'wave','D':'closure','F':'forcing','P':'pressure_mean'}
    tendencies={name:old.mode_coefficients(p[field],h,z,dz)[0] for name,field in mapping.items()}
    b=mode_budget(a,tendencies,t,norm);rows=[]
    for i,time in enumerate(t):
        row=dict(time_s=float(time),elapsed_s=float(tau[i]),jet_prominence_fraction=data['histories'][i]['jet_prominence_fraction'],
                 band_energy_m2_s2=float(sum(b['energy'][i,1:])),band_energy_change_m2_s2=float(sum(b['energy'][i,1:]-b['energy'][0,1:])),
                 band_budget_residual_m2_s2=float(sum(b['energy_error'][i,1:])))
        for term in mapping:
            row[f'band_power_{term}_m2_s3']=float(sum(b['powers'][term][i,1:]));row[f'band_work_{term}_m2_s2']=float(sum(b['work'][term][i,1:]))
            for m in range(8):row[f'mode{m+1}_delta_{term}_m_s']=float(b['cumulative_amplitude'][term][i,m])
        for m in range(8):row[f'mode{m+1}_amplitude_m_s']=float(a[i,m]);row[f'mode{m+1}_amplitude_error_m_s']=float(b['amplitude_error'][i,m])
        rows.append(row)
    band0=float(sum(b['energy'][0,1:]));windows=[]
    for stop in (200,400,600,800,1000,1200):
        ids=np.flatnonzero(abs(tau-stop)<1e-7)
        if not len(ids):continue
        i=int(ids[0]);windows.append(dict(elapsed_s=stop,net_band_energy_change_m2_s2=rows[i]['band_energy_change_m2_s2'],
            signed_cumulative_work_m2_s2={term:rows[i][f'band_work_{term}_m2_s2'] for term in mapping},
            direct_3d_over_initial_band_energy=rows[i]['band_work_C_m2_s2']/band0 if band0>1e-20 else None,
            retained_jet_prominence=rows[i]['jet_prominence_fraction']))
    band_delta=np.array([r['band_energy_change_m2_s2'] for r in rows]);band_error=np.array([r['band_budget_residual_m2_s2'] for r in rows])
    band_gross=cumulative(np.array([sum(abs(r[f'band_power_{term}_m2_s3']) for term in mapping) for r in rows]),t)
    band_scale=max(float(max(abs(band_delta))),float(max(band_gross)),1e-16)
    summary=dict(initial_band_energy_m2_s2=band0,band_modes=list(range(2,9)),max_mode_amplitude_relative_error=float(max(b['amplitude_relative_error'])),
        band_energy_relative_error=float(max(abs(band_error))/band_scale),band_energy_closure_qualified=bool(max(abs(band_error))/band_scale<=.05),
        max_mode_energy_relative_error=float(max(b['energy_relative_error'])),windows=windows,
        meaning='Signed work on modes 2–8 of actual Eulerian current minus the fixed maintained background. The initial generated jet is retained. Mode 1 (broad shear) is available in the amplitude data but excluded from this band. Direct 3D work is evaluated along the actual trajectory; subtracting it does not construct a realizable 2D flow.')
    return summary,rows


def load_case(root,entry,source,approved):
    reused=entry in json.loads((root/'campaign.json').read_text()).get('references',[])
    if reused:
        result,data,hashes=shear.reference_case(root,entry)
        runtime=json.loads((root/'runtime/references.json').read_text())['references'][entry['id']]
        pins=json.loads((root/'references-manifest.json').read_text())['references'][entry['id']]
        rroot=Path(runtime['campaign_root']);oldentry=next(e for e in json.loads((rroot/'campaign.json').read_text())['runs'] if e['id']==pins['case_id'])
        _,receipt=old.run_state(rroot,oldentry,pins['source_version'],None);run=old.Run(rroot,oldentry,receipt)
        result['mode']=entry.get('mode','2d' if int(result['configuration']['ny'])==1 else '3d')
    else:
        status,receipt=old.run_state(root,entry,source,approved)
        if status!='complete':return dict(id=entry['id'],status=status,reused=False,mode=entry.get('mode')),None,None
        result,data,hashes=old.assess(root,entry,receipt);result.update(mode=entry.get('mode','3d'),reused=False);run=old.Run(root,entry,receipt)
    run.checked.update(hashes)
    result['label']=entry.get('label',entry['id']);result['comparison_group']=entry.get('comparison_group')
    return result,data,run


def case_analysis(root,entry,result,data,run):
    cfg=result['configuration'];maps=map_budgets(run,cfg);scalar=scalar_budget_rows(run)
    if scalar is not None:
        if abs(scalar[0]['time_s']-data['time'][0])>1e-7 or abs(scalar[-1]['time_s']-data['time'][-1])>1e-7:raise ValueError('Signed budget timeline misses endpoints')
        agreement=check_scalar_maps(scalar,maps['rows'])
        moments=summarize_moment_rows(scalar);cadence='native one-second signed moments'
    else:scalar=maps['rows'];moments=maps['moments'];cadence='legacy map-cadence moments; temporal closure qualified separately';agreement=None
    jet,jetrows=jet_budgets(data,cfg)
    diagnostic_rows=run.csv('regime-budget.csv')
    pressure_curl=max(abs(r['native_pressure_curl_max']) for r in diagnostic_rows)
    if pressure_curl>1e-10:raise ValueError('Pressure curl is too large to omit from the vorticity budget')
    all_checked=dict(run.checked)
    if result.get('reused'):
        pins=json.loads((root/'references-manifest.json').read_text())['references'][entry['id']]['files']
        if any(pins.get(k)!=v for k,v in all_checked.items()):raise ValueError('A consumed reference budget input lacks a frozen pin')
    summary=dict(id=result['id'],status='complete',configuration=cfg,reused=result.get('reused',False),
                 native_identity='Q_t = A0 + C_3D + W + D + F; C_3D = curl_y[A(full velocity) − A(transverse mean)]',
                 moment_cadence_description=cadence,vorticity_moments=moments,field_temporal_closure=maps['field_closure'],
                 scalar_map_moment_max_relative_difference=agreement,
                 native_pressure_curl_max_s_minus2=pressure_curl,
                 algebraic_relative_error=maps['algebraic_relative_error'],advection_split_absolute_error=maps['split_absolute_error'],
                 max_abs_direct_3d_s_minus2=maps['max_abs_C_s_minus2'],direct_3d_zero_check_2d=maps['direct_3d_zero_check_2d'],
                 jet_budget=jet,trim_cells=maps['trim_cells'],
                 scope='Signed direct native budgets along the realized trajectory. These are distinct from the finite 3D-versus-2D intervention, which also changes subsequent resolved dynamics. A norm alone is not signed destructive work.')
    summary['regional_budget_note']='Interior and active-Stokes regions retain transport through their edges inside the native terms; their advection integrals need not vanish and viscous projections need not be purely negative. The primary interior trim has the same physical width across grids.'
    return summary,scalar,jetrows,maps,all_checked


def plots_for_case(folder,result,data,analysis,rows,jetrows,maps):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    t=np.array([r['elapsed_s'] for r in rows]);jt=np.array([r['elapsed_s'] for r in jetrows])
    fig,ax=plt.subplots(3,2,figsize=(12,11),constrained_layout=True)
    for state,a,title in [('Q',ax[0,0],'Complete mean-vorticity enstrophy'),('profile',ax[0,1],'Horizontal-mean shear enstrophy'),('eddy',ax[1,0],'x-varying mean-vorticity enstrophy')]:
        initial=max(analysis['vorticity_moments']['interior'][state]['initial_enstrophy_s_minus2'],1e-16)
        for term in TERMS:a.plot(t,[r[f'interior_int_P_{state}_{term}']/initial for r in rows],label=TERM_LABELS[term],color=TERM_COLORS[term])
        z0=rows[0][f'interior_Z_{state}'];a.plot(t,[(r[f'interior_Z_{state}']-z0)/initial for r in rows],'k--',label='Actual change')
        a.set(title=title,ylabel='Integrated production / initial enstrophy')
        ax[1,1].plot(t,[r[f'interior_temporal_residual_{state}']/initial for r in rows],label=state)
    for term in TERMS:
        ax[2,0].plot(jt,[1e6*r[f'band_work_{term}_m2_s2'] for r in jetrows],label=TERM_LABELS[term],color=TERM_COLORS[term])
    ax[1,1].axhline(0,color='.6',lw=.6);ax[1,1].legend(fontsize=8)
    ax[2,0].plot(jt,[1e6*r['band_energy_change_m2_s2'] for r in jetrows],'k--',label='Actual change')
    ax[2,1].plot(jt,[r['jet_prominence_fraction'] for r in jetrows],color='#263e47');ax[2,1].axhline(.5,color='.5',ls=':')
    ax[1,1].set(title='Sampled enstrophy-budget residuals',ylabel='Residual / respective initial enstrophy')
    ax[2,0].set(title='Signed work on current modes 2–8',ylabel='Integrated work (10⁻⁶ m²/s²)');ax[2,1].set(title='Original jet prominence',ylabel='Prominence / initial prominence')
    for a in ax.flat:a.set_xlabel('Time after switch (s)');a.grid(alpha=.15)
    ax[0,0].legend(fontsize=8);fig.suptitle(result['label']);fig.savefig(folder/'signed-budgets.png',dpi=145);plt.close(fig)
    chosen=[m for m in maps['images'] if m['elapsed_s'] in (0,200,600,1200)]
    if not chosen:chosen=maps['images']
    fig,axes=plt.subplots(4,len(chosen),figsize=(3.4*len(chosen),11.4),squeeze=False,constrained_layout=True)
    fields=('Q','Q_eddy','C','integrated_C');titles=('Complete transverse-mean vorticity (1/s)','x-varying mean vorticity (1/s)','Direct 3D tendency (1/s²)','Integral of direct 3D tendency (1/s)')
    xlim=(0,float(result['configuration']['lx']),-float(result['configuration']['h']),0)
    for row,(field,title) in enumerate(zip(fields,titles)):
        limit=max(float(np.max(abs(m[field]))) for m in chosen);limit=max(limit,1e-12)
        for j,m in enumerate(chosen):
            im=axes[row,j].imshow(m[field].T,origin='lower',extent=xlim,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
            axes[row,j].set_title(f'{m["elapsed_s"]:g} s',fontsize=10);axes[row,j].set_xlabel('x (m)');axes[row,j].set_ylabel('z (m)')
        fig.colorbar(im,ax=axes[row,:],label=title,shrink=.9)
    fig.suptitle(result['label']);fig.savefig(folder/'vorticity-maps.png',dpi=140);plt.close(fig)
    return ['signed-budgets.png','vorticity-maps.png']


def comparison_section(out,comparison,results,histories,jetrows):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    labels={r['id']:r.get('label',r['id']) for r in results};sections=[]
    for kind,records in comparison.items():
        for record in records:
            title=record['id'];filename='comparison-'+old.identifier(title)+'.png'
            if record['status']!='complete':
                sections.append('<section class="panel"><h3>'+html.escape(title)+'</h3><p>Awaiting '+html.escape(', '.join(record['missing_cases']))+'.</p></section>');continue
            fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
            if kind=='interventions':ids=list(record['cases'].values())
            else:ids=[record['coarse'],record['fine']]
            for n,identifier in enumerate(ids):
                h=histories[identifier];j=jetrows[identifier];t=[r['elapsed_s'] for r in h];jt=[r['elapsed_s'] for r in j]
                axes[0,0].plot(t,[r['jet_prominence_fraction'] for r in h],label=labels[identifier])
                axes[0,1].plot(t,[100*r['chi_3d'] for r in h])
                axes[1,0].plot(jt,[1e6*r['band_work_C_m2_s2'] for r in j])
                if kind!='interventions':axes[1,1].plot(jt,[1e6*r['band_energy_change_m2_s2'] for r in j])
            axes[0,0].set(title='Original jet prominence',ylabel='Prominence / initial prominence')
            axes[0,0].axhline(.5,color='.6',ls=':');axes[0,0].legend(fontsize=8)
            axes[0,1].set(title='Three-dimensional eddy energy',ylabel='3D fraction (%)');axes[0,1].axhline(10,color='.6',ls=':')
            axes[1,0].set(title='Direct 3D work on current modes 2–8',ylabel='Signed work (10⁻⁶ m²/s²)');axes[1,0].axhline(0,color='.6',lw=.6)
            if kind=='interventions':
                records_here=record['windows'];axes[1,1].plot([r['elapsed_s'] for r in records_here],[100*r['additional_3d_prominence_effect'] for r in records_here],'o-',color='#263e47')
                axes[1,1].set(title='Additional 3D effect of added wind',ylabel='Prominence difference (percentage points)');axes[1,1].axhline(0,color='.6',lw=.6)
                caption='The lower-right panel subtracts the wind-minus-control change in 2D from the wind-minus-control change in 3D. It measures a full nonlinear intervention. The signed C₃D work at left is a direct term along each realized trajectory; the two quantities have different meanings and units.'
            else:
                axes[1,1].set(title='Actual current-mode energy change',ylabel='Energy change (10⁻⁶ m²/s²)')
                caption='The coarse and refined integrations start from the same physical parent through documented native-grid refinement. Compare onset and early windows before interpreting late chaotic trajectory differences. This is a targeted numerical check, not proof of asymptotic convergence.'
            for axis in axes.flat:axis.set_xlabel('Time after switch (s)');axis.grid(alpha=.15)
            fig.suptitle(title);fig.savefig(out/filename,dpi=140);plt.close(fig)
            sections.append('<section class="panel"><h3>'+html.escape(title)+'</h3><figure><img src="'+filename+'" alt="Matched intervention or numerical sensitivity"><figcaption>'+caption+'</figcaption></figure></section>')
    return '<section class="panel"><h2>Matched interventions and numerical checks</h2><p>The 200, 400 and 600 s windows are retained alongside later outcomes. A single final three-dimensional energy fraction or a rematched late extremum does not identify when the original jet was eroded.</p></section>'+''.join(sections) if sections else ''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--approved-runs',type=Path);args=parser.parse_args()
    root=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);campaign=json.loads((root/'campaign.json').read_text());source=json.loads((root/'source-manifest.json').read_text());approved=set(json.loads(args.approved_runs.read_text())) if args.approved_runs else None
    entries=campaign.get('references',[])+campaign['runs'];results=[];budgets={};verified={};sections=[];histories={};jet_histories={}
    for entry in entries:
        result,data,run=load_case(root,entry,source['version'],approved);results.append(result)
        if result['status']!='complete':continue
        analysis,rows,jetrows,maps,hashes=case_analysis(root,entry,result,data,run);budgets[result['id']]=analysis;verified[result['id']]=hashes
        histories[result['id']]=data['histories'];jet_histories[result['id']]=jetrows
        folder=out/'runs'/old.identifier(result['id']);folder.mkdir(parents=True,exist_ok=True);old.write_json(folder/'summary.json',result);old.write_json(folder/'budget-assessment.json',analysis);write_compressed_csv(folder/'signed-vorticity-budget.csv',rows);write_compressed_csv(folder/'signed-jet-budget.csv',jetrows);write_compressed_csv(folder/'time-series.csv',data['histories'])
        figures=plots_for_case(folder,result,data,analysis,rows,jetrows,maps)
        closure=analysis['vorticity_moments']['interior']['eddy'];qual='qualified at the 5% screening tolerance' if closure['closure_qualified'] else 'NOT qualified at the 5% screening tolerance'
        sections.append('<section class="panel"><h2>'+html.escape(result['label'])+'</h2><p>Interior eddy-enstrophy temporal closure: '+format(100*closure['relative_to_change_and_gross'],'.3g')+'% of the larger of net change or gross integrated production; '+qual+'. Native moments use '+format(closure['cadence_s'],'.3g')+' s cadence. Current-band energy closure: '+format(100*analysis['jet_budget']['band_energy_relative_error'],'.3g')+'%.</p><figure><img src="runs/'+result['id']+'/signed-budgets.png" alt="Signed native vorticity and jet work budgets"><figcaption>Positive production increases the indicated quadratic measure; negative production decreases it. Resolved advection and direct 3D feedback may substantially cancel. These are trajectory budgets, not independent switched-off simulations.</figcaption></figure><figure><img src="runs/'+result['id']+'/vorticity-maps.png" alt="Mean vorticity and direct three-dimensional tendency maps"><figcaption>The integral is at fixed Eulerian positions and must be added to all other tendency integrals to recover Q. It is not the change that would occur in a separately evolved 2D flow.</figcaption></figure><p><a href="runs/'+result['id']+'/budget-assessment.json">Budget assessment</a> · <a href="runs/'+result['id']+'/signed-vorticity-budget.csv.gz">Signed vorticity data (CSV.gz)</a> · <a href="runs/'+result['id']+'/signed-jet-budget.csv.gz">Signed jet data (CSV.gz)</a></p></section>')
        del data,maps,run
    counts={s:sum(r['status']==s and not r.get('reused',False) for r in results) for s in ('complete','running','queued','waiting','failed')}
    comparisons=dict(interventions=assess_interventions(campaign.get('intervention_groups',[]),results,histories,jet_histories),
                     numerical_checks=assess_resolution(campaign.get('resolution_pairs',[]),results,histories,jet_histories,budgets))
    comparison_html=comparison_section(out,comparisons,results,histories,jet_histories)
    old.write_json(out/'comparisons.json',comparisons)
    summary=dict(title='Eulerian shear and direct 3D vorticity feedback',source_version=source['version'],new_run_counts=counts,reused_runs=sum(r.get('reused',False) for r in results),runs=results,budget_assessments=budgets,comparisons=comparisons,
                 assessment='STUDY_COMPLETE' if counts['complete']==len(campaign['runs']) else 'STUDY_IN_PROGRESS',qualification='Direct signed native budgets and separate resolution/intervention comparisons. A large source norm alone does not establish destructive feedback or a universal regime boundary.')
    old.write_json(out/'summary.json',summary);old.write_json(out/'provenance.json',dict(source_version=source['version'],verified_inputs=verified,native_snapshots_published=False));(out/'style.css').write_text(old.STYLE)
    tablerows=''.join('<tr><td>'+html.escape(r.get('label',r['id']))+'</td><td>'+r['status']+'</td><td>'+('existing result' if r.get('reused') else 'new simulation')+'</td></tr>' for r in results)
    body='<nav><a href="../../">Gallery</a><a href="../shear-instability/">Initial shear pilot</a></nav><h1>What does the 3D term do to the jet?</h1><p>The pilot linked Eulerian shear, three-dimensional growth and additional jet erosion. This study measures the direct signed contribution of the complete 3D feedback to the forced mean-vorticity and jet budgets, and adds targeted numerical checks.</p><div class="panel"><strong>Native forced vorticity budget:</strong> ∂t Q = A₀ + C₃D + W + D + F. A₀ is the native tendency from the transverse-mean velocity; C₃D is the difference between full-flow and mean-flow native advection curls. W, D and F retain wave, viscous and prescribed forcing terms. Pressure curl is checked independently. The complete C₃D includes transverse correlations, not only one stretching term. Added wind acts through the velocity boundary stress and is therefore included in D; F is the prescribed background-maintaining body force.</div><table><thead><tr><th>Case</th><th>Status</th><th>Origin</th></tr></thead><tbody>'+tablerows+'</tbody></table><section class="panel"><h2>Direct budget and intervention answer different questions</h2><p>Signed projections determine whether C₃D produces or removes mean-vorticity variance and jet-mode energy along the actual trajectory. The full 2D-versus-3D intervention also changes the later resolved eddies and therefore measures a broader response. We retain forcing and viscosity in both budgets, and report temporal closure before interpreting integrated terms. Complete transverse-mean enstrophy separates into horizontal-mean shear and x-varying contributions: Z(Q) = Z(⟨Q⟩ₓ) + Z(Q − ⟨Q⟩ₓ). The shear component and current modes address the jet; the x-varying component addresses its resolved eddies. Positive enstrophy production need not mean a stronger jet.</p><p>Legacy maps provide retrospective spatial budgets at their saved cadence. New one-second scalar moments improve temporal closure without interpolating the old data. The map integral and the scalar enstrophy integral have separate closure tests. Five percent is a numerical screening tolerance, not a physical transition criterion.</p></section>'+comparison_html+''.join(sections)+'<footer><a href="summary.json">Full assessment</a> · <a href="comparisons.json">Intervention and numerical comparisons</a> · <a href="PROTOCOL.txt">Frozen protocol</a> · <a href="campaign.json">Case matrix</a> · <a href="provenance.json">Provenance</a>. Results appear only after verified completion and persistent backup.</footer>'
    (out/'index.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Direct 3D feedback and jet budgets</title><link rel="stylesheet" href="style.css"></head><body><main>'+body+'</main></body></html>\n')
    for name in ('PROTOCOL.txt','campaign.json','source-manifest.json','references-manifest.json'):
        if (root/name).exists():shutil.copy2(root/name,out/name)
    size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if size>100*1024**2:raise ValueError('Budget gallery output exceeds100MiB')
    print(json.dumps(dict(new_run_counts=counts,reused_runs=summary['reused_runs'],public_bytes=size)))


if __name__=='__main__':main()
