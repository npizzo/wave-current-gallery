#!/usr/bin/env python3
"""Verified streaming analysis and gallery for scalar-vorticity moment tests."""
from __future__ import annotations
import argparse,csv,datetime,hashlib,html,json,math,os,re,shutil,tempfile
from pathlib import Path
import numpy as np
from analyze import (write_json,read_csv,digest,clean,safe_path,identifier,verified_file,
                     cache_key,cache_valid,complete_cache,campaign_status,PUBLIC_CONFIG)
from moment_science import (MOMENTS,SOURCES,COMPONENTS,regional_profile,interval_closure,
                            normalize_residual,integral_metrics,cumulative_trapezoid,eulerian_profile)

VERSION='scalar-moments-1'
PUBLIC_FILES=('summary.json','moments.csv','regional-budgets.csv','mean-profiles.csv',
              'moments.png','closure.png','current-profiles.png')
CONFIG_KEYS=(*PUBLIC_CONFIG,'two_d_three_c_rms','moment_interval','moment_diagnostic_version',
             'moment_q_convention','moment_budget_convention','two_d_three_c_rule')
COLORS={0.: '#202b38',1e-8:'#8ca5c5',1e-6:'#168b96',.01:'#8051a0',1.:'#d47136'}
PUBLIC_LIMIT=95*1024**2


def compact_csv(path,rows):
    rows=list(rows)
    if not rows:return
    names=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for r in rows:
            w.writerow({k:('' if v is None else format(float(v),'.10g') if isinstance(v,(float,np.floating)) else v) for k,v in r.items()})


def profile_groups(path):
    """Yield one time slice; never load the full time × depth table into RAM."""
    with Path(path).open(newline='') as f:
        reader=csv.DictReader(f);time=None;rows=[]
        if not reader.fieldnames or not {'time_s','z_m','dz_m'}<=set(reader.fieldnames):
            raise ValueError('Missing profile coordinates')
        for raw in reader:
            row={k:float(v) for k,v in raw.items()}
            if not all(math.isfinite(v) for v in row.values()):raise ValueError('Nonfinite profile data')
            t=row['time_s']
            if time is not None and t!=time:
                if t<=time:raise ValueError('Profile times must increase')
                yield time,{k:np.array([r[k] for r in rows]) for k in rows[0]}
                rows=[]
            time=t;rows.append(row)
        if rows:yield time,{k:np.array([r[k] for r in rows]) for k in rows[0]}


def check_profile_grid(p,z,dz):
    if p['z_m'].shape!=z.shape or not np.allclose(p['z_m'],z,atol=1e-11,rtol=1e-11) or not np.allclose(p['dz_m'],dz,atol=1e-11,rtol=1e-11):
        raise ValueError('Profile grid differs from verified grid metadata')


def selected_indices(times,count=5):
    t=np.asarray(times)
    return sorted(set(int(np.argmin(np.abs(t-target))) for target in np.linspace(t[0],t[-1],count)))


def series(rows,key):
    values=np.array([r[key] for r in rows],float)
    if not np.isfinite(values).all():raise ValueError('Nonfinite completed diagnostic '+key)
    return values


def style_axes(axes):
    for ax in np.ravel(axes):
        ax.grid(alpha=.16);ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=8)


def savefig(fig,path):
    import matplotlib.pyplot as plt
    fig.savefig(path,dpi=135,facecolor='white',bbox_inches='tight');plt.close(fig)


def plot_run(cache,rows,regional,profiles,config,run_id):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=series(rows,'time_s');full=regional['full'];upper=regional['stokes_layer']
    fig,ax=plt.subplots(2,3,figsize=(12,7),layout='constrained');style_axes(ax)
    for key,label,c in [('Zq','Total q','#243b53'),('Zmean','Transverse mean','#158c96'),('Zvariance','Transverse variance','#c56a30')]:
        ax[0,0].plot(t,series(rows,key),label=label,color=c)
    ax[0,0].set(title='Eulerian q-enstrophy partition',ylabel='Enstrophy (s⁻²)')
    ax[0,0].set_yscale('symlog',linthresh=max(float(series(rows,'Zq')[0])*1e-10,1e-20))
    ax[0,0].set_ylim(bottom=0)
    ax[0,0].set_yticks(sorted(set([0.,*ax[0,0].get_yticks()])))
    for key,label,c in [('Pa_T','3D production','#c56a30'),('Pa_background','Background q₀ conversion','#158c96')]:
        ax[0,1].plot(t,cumulative_trapezoid(t,series(rows,key)),label=label,color=c)
    ax[0,1].plot(t,series(rows,'Za')-rows[0]['Za'],color='#243b53',label='Observed anomaly change')
    ax[0,1].set(title='Anomaly-enstrophy sources',ylabel='Cumulative change (s⁻²)')
    for key,label,c in [('Pq_T','Signed 3D production','#c56a30'),('Pq_T_abs','Mean |local production|','#7e5598')]:
        ax[0,2].plot(t,series(rows,key),label=label,color=c)
    for key,label,c in [('Pq_stretch','q² vᵧ','#168b96'),('Pq_tilt_wy_vx','q wᵧ vₓ','#9bb6c7'),('Pq_tilt_minus_uy_vz','−q uᵧ v_z','#9b9b4f')]:
        ax[0,2].plot(t,series(rows,key),label=label,color=c,linewidth=.8,alpha=.8)
    ax[0,2].set(title='Signed versus absolute production',ylabel='Production (s⁻³)')
    for key,label,c in [('Pmean_exchange','To mean, from variance','#158c96'),('Pvariance_exchange','To variance, from mean','#8bbfc3'),('Pmean_T','3D production in mean','#c56a30'),('Pvariance_T','3D production in variance','#7e5598')]:
        ax[1,0].plot(t,cumulative_trapezoid(t,series(rows,key)),label=label,color=c)
    ax[1,0].set(title='Mean–variance exchange',ylabel='Cumulative change (s⁻²)')
    for reg,label,ls in [('full','Full depth','-'),('stokes_layer','Stokes layer','--')]:
        r=regional[reg]
        ax[1,1].plot(t,cumulative_trapezoid(t,series(r,'Pmean_anomaly_feedback_native')),ls,label=label)
    ax[1,1].set(title='Mean-anomaly feedback',ylabel='Cumulative supply (s⁻²)')
    for r,label,c in [(full,'Full depth','#243b53'),(upper,'Stokes layer','#158c96')]:
        ax[1,2].plot(t,series(r,'Zvariance')/np.maximum(series(r,'Za'),1e-30),label=label,color=c)
    ax[1,2].set(title='Transverse q-variance fraction',ylabel='Zvariance / Za')
    for a in ax.flat:a.set_xlabel('Time (s)');a.legend(fontsize=7,frameon=False)
    fig.suptitle('Scalar y-vorticity moments · '+run_id,fontsize=11);savefig(fig,cache/'moments.png')
    fig,ax=plt.subplots(2,2,figsize=(10.5,7),layout='constrained');style_axes(ax)
    for col,(reg,label) in enumerate([('full','Full depth'),('stokes_layer','Stokes layer')]):
        r=regional[reg]
        for key,ls in [('Zq','-'),('Za','--')]:
            c=interval_closure(t,series(r,key),series(r,'d'+key+'_native'),series(r,key+'_continuum_rate'))
            ax[0,col].plot(t,c['temporal_residual'],ls,label=key+' temporal',color='#168b96')
            ax[0,col].plot(t,c['spatial_residual'],ls,label=key+' spatial',color='#d47136')
        ax[0,col].set(title=label+': two residuals',ylabel='Cumulative residual (s⁻²)')
        for key,c in [('Zq','#243b53'),('Za','#7e5598')]:
            ax[1,col].plot(t,cumulative_trapezoid(t,series(r,key+'_boundary_supply')),color=c,label=key+' boundary transport')
        ax[1,col].set(title=label+': advective boundary transport',ylabel='Cumulative supply (s⁻²)')
    for a in ax.flat:a.set_xlabel('Time (s)');a.legend(fontsize=8,frameon=False)
    savefig(fig,cache/'closure.png')
    fig,ax=plt.subplots(1,2,figsize=(8,5),sharey=True,layout='constrained');style_axes(ax)
    ptime=profiles['time'];ue=profiles['UE'];z=profiles['z']
    for i in selected_indices(ptime,3):
        label=f'{ptime[i]:g} s';ax[0].plot(1000*ue[i],z,label=label);ax[1].plot(1000*(ue[i]-ue[0]),z,label=label)
    ax[0].set(xlabel='Eulerian horizontal mean (mm s⁻¹)',ylabel='z (m)',title='Total current')
    ax[1].set(xlabel='Change from own initial mean (mm s⁻¹)',title='Current change')
    for a in ax:a.legend(fontsize=8,frameon=False)
    savefig(fig,cache/'current-profiles.png')


def analyze_moment_run(root,entry,receipt,cache):
    key=hashlib.sha256((VERSION+cache_key(receipt)).encode()).hexdigest()
    if cache_valid(cache,key):return json.loads((cache/'summary.json').read_text())
    folder=safe_path(root,entry['path'])
    config=json.loads(verified_file(folder,'configuration.json',receipt).read_text())
    grid=json.loads(verified_file(folder,'grid.json',receipt).read_text())
    done=json.loads(verified_file(folder,'completed.json',receipt).read_text())
    if config.get('initial_condition_sha256')!=done.get('initial_condition_sha256'):raise ValueError('Initial-condition receipts differ')
    rows=read_csv(verified_file(folder,'moments.csv',receipt));t=series(rows,'time_s')
    if len(t)<3 or np.any(np.diff(t)<=0) or not np.isclose(t[-1],done.get('time',done.get('time_s')),atol=1e-7,rtol=0):
        raise ValueError('Moment records do not span the completed run')
    z=np.asarray(grid['z_centers']);dz=np.diff(grid['z_faces']);delta=1/(2*float(config['k']))
    upper=(z>=-delta)&(z<=0)
    if not np.any(upper):raise ValueError('No Stokes-layer cells')
    masks={'full':np.ones(len(z),bool),'stokes_layer':upper};regional={n:[] for n in masks}
    selected=selected_indices(t);selected_profiles={};count=0;global_errors={}
    for i,(time,p) in enumerate(profile_groups(verified_file(folder,'moment-profiles.csv',receipt))):
        if i>=len(t) or not np.isclose(time,t[i],atol=1e-8,rtol=0):raise ValueError('Moment profile schedule differs from global moments')
        check_profile_grid(p,z,dz)
        for name,mask in masks.items():regional[name].append({'time_s':time,**regional_profile(p,mask)})
        # Independently reduced volume means must agree with integrated xy profiles.
        full=regional['full'][-1]
        for name in MOMENTS+tuple('d'+x+'_native' for x in MOMENTS):
            error=abs(full[name]-rows[i][name]);scale=max(abs(full[name]),abs(rows[i][name]),1e-25)
            global_errors[name]=max(global_errors.get(name,0.),error/scale)
            if error>2e-9*scale+1e-23:raise ValueError('Global and profile reductions disagree: '+name)
        if i in selected:selected_profiles[time]=p
        count+=1
    if count!=len(t):raise ValueError('Incomplete moment profiles')
    # Stream horizontal mean profiles separately; the stored UE column is checked
    # against UL minus the actual wave drift, including no-wave controls.
    pt=[];ue=[];ul=[];last=-math.inf
    for time,p in profile_groups(verified_file(folder,'jet-profiles.csv',receipt)):
        check_profile_grid(p,z,dz)
        if time<=last:raise ValueError('Nonmonotone jet profile schedule')
        last=time;converted=eulerian_profile(p['UL'],z,config)
        if not np.allclose(p['UE'],converted,atol=1e-12,rtol=1e-10):raise ValueError('Stored Eulerian mean differs from UL-us')
        pt.append(time);ue.append(converted);ul.append(p['UL'])
    if not pt or not np.isclose(pt[0],t[0],atol=1e-8) or not np.isclose(pt[-1],t[-1],atol=1e-8):raise ValueError('Incomplete mean-current profile history')
    profiles=dict(time=np.asarray(pt),z=z,dz=dz,UE=np.asarray(ue),UL=np.asarray(ul))
    closure={};public_regional=[]
    for region,r in regional.items():
        closure[region]={}
        for name in MOMENTS:
            c=interval_closure(t,series(r,name),series(r,'d'+name+'_native'),series(r,name+'_continuum_rate'))
            closure[region][name]=normalize_residual(c,r[0][name])
            # The public table is explicitly a selected sample of the full 1 s analysis.
            for i in sorted(set(range(0,len(t),20))|{len(t)-1}):
                public_regional.append(dict(region=region,moment=name,time_s=t[i],value=r[i][name],
                    native_rate=r[i]['d'+name+'_native'],continuum_rate=r[i][name+'_continuum_rate'],
                    boundary_supply=r[i][name+'_boundary_supply'],delta=c['delta'][i],
                    native_integral=c['native_integral'][i],continuum_integral=c['continuum_integral'][i],
                    temporal_residual=c['temporal_residual'][i],spatial_residual=c['spatial_residual'][i],
                    differential_residual_rate=r[i].get(name+'_differential_residual'),
                    transport_product_residual_rate=r[i].get(name+'_transport_product_residual')))
    supplies={}
    for name in ('Pq_T','Pa_T','Pa_background','Pmean_exchange','Pvariance_exchange','Pmean_T','Pvariance_T','Pmean_feedback_native','Pmean_anomaly_feedback_native'):
        supplies[name]=integral_metrics(t,series(rows,name),series(rows,name+'_abs') if name+'_abs' in rows[0] else None)
    for prefix in ('Pq','Pa'):
        for component in COMPONENTS:
            name=prefix+'_'+component
            supplies[name]=integral_metrics(t,series(rows,name),series(rows,name+'_abs') if name+'_abs' in rows[0] else None)
        total_absolute=sum(series(rows,prefix+'_'+component+'_abs') for component in COMPONENTS)
        supplies[prefix+'_T_componentwise_absolute']=integral_metrics(t,series(rows,prefix+'_T'),total_absolute)
    partition={name:max(abs(r[name]) for r in regional['full']) for name in ('partition_residual','exchange_sum','production_partition_residual')}
    native_budget=read_csv(verified_file(folder,'regime-budget.csv',receipt))
    native_qc={name:max((abs(r[name]) for r in native_budget if isinstance(r.get(name),(int,float))),default=None) for name in ('native_divergence_max','native_divergence_rms','native_pressure_curl_max','viscous_cfl')}
    native_qc.update({name:max((abs(r[name]) for r in rows if isinstance(r.get(name),(int,float))),default=None) for name in ('native_partition_error','anomaly_partition_error','q_continuum_rhs_error_rms','T_unsimplified_error2')})
    result=dict(id=entry['id'],status='complete',source_version=receipt['source_version'],receipt_sha256=cache_key(receipt),
        config={k:config[k] for k in CONFIG_KEYS if k in config},final_time_s=float(t[-1]),samples=len(t),
        sampling_interval_min_s=float(np.diff(t).min()),sampling_interval_max_s=float(np.diff(t).max()),
        initial_moments={k:rows[0][k] for k in MOMENTS},final_moments={k:rows[-1][k] for k in MOMENTS},
        supplies=supplies,closure=closure,partition_checks=partition,global_profile_relative_errors=global_errors,native_qc=native_qc,
        max_abs_3d_production=float(np.max(np.abs(series(rows,'Pq_T_abs')))),
        max_T_rms=float(max(r.get('T_rms',0) for r in rows)),
        stokes_layer=dict(target_bottom_m=-delta,cell_face_bottom_m=float(np.asarray(grid['z_faces'])[np.flatnonzero(upper)[0]]),
                          cell_face_top_m=float(np.asarray(grid['z_faces'])[np.flatnonzero(upper)[-1]+1]),
                          cells=int(upper.sum()),selection='Whole cells whose centers lie in [-1/(2k), 0]. Boundary transport retained.'),
        q0=dict(definition='q0=-alpha*2k*S_reference*exp(2kz), with native discrete-center profile used for the anomaly.',
                reference_drift_is_not_actual_drift_in_no_wave_controls=True),
        scope='Finite-time individual realization. Tiny initial perturbations are not assumed to stay weak. q-enstrophy production is separate from vector enstrophy, background conversion and transport. Residuals are reported, not fitted away.')
    cache.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix=cache.name+'.tmp-',dir=cache.parent))
    try:
        write_json(tmp/'summary.json',result);compact_csv(tmp/'moments.csv',rows);compact_csv(tmp/'regional-budgets.csv',public_regional)
        public_profiles=[]
        for i in selected_indices(profiles['time']):
            time=profiles['time'][i];mp=selected_profiles.get(time)
            for j,zz in enumerate(z):
                public_profiles.append(dict(time_s=time,z_m=zz,dz_m=dz[j],UL=profiles['UL'][i,j],UE=profiles['UE'][i,j],
                    delta_UE=profiles['UE'][i,j]-profiles['UE'][0,j],
                    q0_native=(mp.get('q0_native_center')[j] if mp is not None and 'q0_native_center' in mp else None),
                    q0_analytic=-config['alpha']*2*config['k']*config['S_reference']*math.exp(2*config['k']*zz)))
        compact_csv(tmp/'mean-profiles.csv',public_profiles)
        np.savez_compressed(tmp/'profile-history.npz',**profiles)
        # Cached private arrays contain only horizontal-mean profiles and global
        # regional reductions, never native 3D snapshots.
        write_json(tmp/'regional-history.json',regional)
        plot_run(tmp,rows,regional,profiles,config,entry['id'])
        complete_cache(tmp,cache,key)
    except BaseException:shutil.rmtree(tmp,ignore_errors=True);raise
    return result


def run_label(result):
    c=result['config'];eta=float(c['perturb_energy_ratio'])
    if c.get('two_d_three_c_rms',0)>0:return '2D3C null'
    return '2D (η=0)' if eta==0 else f'Initial η={eta:g}'


def mean_comparisons(entries,results,cache,output):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    byid={r['id']:r for r in results};groups={};figures=[];comparison=[]
    for e in entries:
        if byid[e['id']]['status']=='complete' and e.get('group')!='2d3c':
            groups.setdefault((e.get('matched_group',e.get('cell')),e.get('seed')),[]).append(e)
    for (group,seed),ee in groups.items():
        available=sorted([byid[e['id']] for e in ee],key=lambda r:r['config']['perturb_energy_ratio'])
        bases=[r for r in available if r['config']['perturb_energy_ratio']==0]
        if len(bases)!=1 or len(available)<2:continue
        base=bases[0];b=np.load(cache/base['id']/'profile-history.npz')
        fields={r['id']:np.load(cache/r['id']/'profile-history.npz') for r in available}
        for r in available:
            c=r['config'];bc=base['config'];a=fields[r['id']]
            for key in ('nx','ny','nz','lx','ly','h','k','S','S_reference','alpha','uoffset','eddy_rms','injection_wavelength','seed','max_dt','nu','waves','maintain_background','advection','base_ic_sha256'):
                if c.get(key)!=bc.get(key):raise ValueError('Mismatched profile comparison '+key)
            if not np.array_equal(a['time'],b['time']) or not np.array_equal(a['z'],b['z']):raise ValueError('Mean-profile grids differ')
            for i,time in enumerate(a['time']):
                difference=(a['UE'][i]-a['UE'][0])-(b['UE'][i]-b['UE'][0])
                comparison.append(dict(run_id=r['id'],baseline_id=base['id'],time_s=time,
                    mean_current_change_error_m_s=float(np.sqrt(np.sum(difference**2*a['dz'])/sum(a['dz'])))))
        ti=selected_indices(b['time'],3);fig,ax=plt.subplots(2,len(ti),figsize=(11.5,7),sharey=True,layout='constrained');style_axes(ax)
        for r in available:
            a=fields[r['id']];eta=float(r['config']['perturb_energy_ratio']);color=COLORS.get(eta,'#777');ls='-' if eta in (0.,1.) else '--'
            for j,i in enumerate(ti):
                ax[0,j].plot(1000*a['UE'][i],a['z'],color=color,ls=ls,label=run_label(r))
                ax[1,j].plot(1000*(a['UE'][i]-a['UE'][0]),a['z'],color=color,ls=ls,label=run_label(r))
        for j,i in enumerate(ti):
            ax[0,j].set(title=f'{b["time"][i]:g} s',xlabel='Eulerian mean (mm s⁻¹)')
            ax[1,j].set(xlabel='Change from own initial mean (mm s⁻¹)')
        ax[0,0].set_ylabel('z (m)');ax[1,0].set_ylabel('z (m)');ax[0,-1].legend(fontsize=7,frameon=False)
        fig.suptitle(f'{group} · seed {seed}: matched mean-current profiles',fontsize=12)
        name=identifier(f'mean-{group}-s{seed}')+'.png';savefig(fig,output/name)
        figures.append(dict(file=name,title=f'{group}, seed {seed}',kind='mean',caption='Direct horizontal-mean Eulerian current (top) and change from each run’s initial profile (bottom). Colors identify initial perturbation energy; they do not assign the subsequent flow a 2D/3D regime. Matched runs have identical base fields. Finite perturbations add energy.'))
        for a in fields.values():a.close()
        b.close()
    compact_csv(output/'mean-comparisons.csv',comparison)
    return figures,comparison


def overview(results,cache,output):
    """A measured response, not an inferred regime boundary."""
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=[r for r in results if r['status']=='complete' and re.fullmatch(r'central-re(100|200|300)-(2d|tiny100|tiny|moderate|finite)-s1',r['id'])]
    if not chosen:return []
    fig,ax=plt.subplots(1,3,figsize=(12,4),layout='constrained');style_axes(ax)
    palette={100:'#168b96',200:'#8051a0',300:'#d47136'}
    for reynolds in (100,200,300):
        rr=sorted([r for r in chosen if f'-re{reynolds}-' in r['id']],key=lambda r:r['config']['perturb_energy_ratio'])
        if not rr:continue
        x=np.arange(len(rr));values=[[],[],[]]
        for r in rr:
            zi=max(r['initial_moments']['Za'],1e-30)
            values[0].append(r['final_moments']['Zvariance']/max(r['final_moments']['Za'],1e-30))
            values[1].append(r['supplies']['Pa_T']['signed_integral']/zi)
            values[2].append(r['supplies']['Pmean_anomaly_feedback_native']['signed_integral']/zi)
        categories=[{0:0,1e-8:1,1e-6:2,.01:3,1:4}[float(r['config']['perturb_energy_ratio'])] for r in rr]
        for j in range(3):ax[j].plot(categories,values[j],marker='o',color=palette[reynolds],label=f'Reλ={reynolds}')
    specs=[('Transverse q variance','Final Zvariance / Za'),('3D production of anomaly enstrophy','∫⟨aT⟩ dt / initial Za'),('Native 3D feedback on the mean','∫⟨(Q−q₀) Cnative⟩ dt / initial Za')]
    for a,(title,label) in zip(ax,specs):
        a.set(title=title,ylabel=label,xlabel='Initial perturbation energy η',xticks=range(5),xticklabels=['0','10⁻⁸','10⁻⁶','0.01','1']);a.legend(fontsize=8,frameon=False)
    savefig(fig,output/'moment-overview.png')
    return [dict(file='moment-overview.png',title='Geometry, production and mean feedback',kind='budget',caption='Completed central cases at 1200 s. The first panel measures q geometry; the other panels measure signed cumulative effects and retain their sign. Normalization uses each run’s own initial background-anomaly enstrophy. Finite perturbations add initial energy. These are distinct finite-time measurements, not regime labels; individual pages retain absolute production and closure residuals.')]


def numerical_controls(entries,results,cache,output):
    """Compare changed profiles across dt/grid controls without requiring same grid."""
    byid={r['id']:r for r in results};rows=[]
    for entry in entries:
        r=byid[entry['id']]
        if r['status']!='complete':continue
        if entry.get('comparison_kind')=='timestep':reference=entry.get('reference');kind='half timestep'
        elif entry.get('group')=='2d3c':
            reference='central-re100-2d-s1' if r['config'].get('waves',1) else 'nowaves-re100-2d-s1'
            kind='2D3C transverse-velocity null'
        elif entry['id'].startswith('central-fine-re300-'):
            reference=entry['id'].replace('central-fine-','central-');kind='finer grid and timestep'
        else:continue
        base=byid.get(reference)
        if not base or base['status']!='complete':continue
        for key in ('lx','ly','h','k','S','S_reference','alpha','uoffset','eddy_rms','injection_wavelength','seed','nu','waves','perturb_energy_ratio'):
            if r['config'].get(key)!=base['config'].get(key):raise ValueError('Numerical control changes physical parameter '+key)
        with np.load(cache/r['id']/'profile-history.npz') as fine,np.load(cache/reference/'profile-history.npz') as coarse:
            common=np.intersect1d(fine['time'],coarse['time'])
            for time in common:
                i=int(np.flatnonzero(fine['time']==time)[0]);j=int(np.flatnonzero(coarse['time']==time)[0])
                # Compare only center points inside both center grids; never extrapolate.
                mask=(coarse['z']>=fine['z'][0])&(coarse['z']<=fine['z'][-1]);z=coarse['z'][mask];dz=coarse['dz'][mask]
                fa=np.interp(z,fine['z'],fine['UE'][i]-fine['UE'][0]);ca=(coarse['UE'][j]-coarse['UE'][0])[mask]
                error=float(np.sqrt(np.sum(dz*(fa-ca)**2)/sum(dz)))
                amplitude=float(np.sqrt(np.sum(dz*ca**2)/sum(dz)))
                rows.append(dict(run_id=r['id'],reference_id=reference,comparison=kind,time_s=time,
                    changed_profile_error_m_s=error,reference_changed_profile_rms_m_s=amplitude,
                    relative_error=error/amplitude if amplitude>1e-15 else None,
                    comparison_z_min_m=z[0],comparison_z_max_m=z[-1],
                    interpolation='linear finer profile onto common coarse centers'))
    compact_csv(output/'numerical-controls.csv',rows)
    return rows


def render_moments(summary,output):
    esc=lambda x:html.escape(str(x),quote=True)
    figure_html=lambda ff:''.join(f'<figure><a href="{esc(f["file"])}"><img loading="lazy" src="{esc(f["file"])}" alt="{esc(f["title"])}"></a><figcaption><strong>{esc(f["title"])}</strong> {esc(f["caption"])}</figcaption></figure>' for f in ff)
    figures=figure_html([f for f in summary.get('figures',[]) if f.get('kind')!='budget'])
    budget=figure_html([f for f in summary.get('figures',[]) if f.get('kind')=='budget'])
    if not figures:figures='<div class="empty"><h3>Results will appear after verified completion</h3><p>No result or regime classification is inferred from a queued or running calculation. The first completed matched pair will populate the mean-current comparisons here.</p></div>'
    rows=[];cards=[]
    for r in summary['runs']:
        c=r.get('config',{});name=esc(r['id']);status=r['status'];complete=status=='complete'
        link=f'<a href="runs/{name}/index.html">{name}</a>' if complete else name
        rows.append(f'<tr data-status="{esc(status)}"><td>{link}</td><td>{esc(status)}</td><td>{esc(c.get("alpha","—"))}</td><td>{esc(c.get("perturb_energy_ratio","—"))}</td><td>{esc(r.get("final_time_s","—"))}</td></tr>')
        if complete:
            cards.append(f'<details><summary>{name} — moments, closure and mean current</summary><p><a href="runs/{name}/index.html">Open all diagnostics and downloads</a></p><figure><img loading="lazy" src="runs/{name}/moments.png" alt="Vorticity moments for {name}"></figure></details>')
    counts=' · '.join(f'{n} {esc(k)}' for k,n in summary['counts'].items() if n)
    downloads='<a href="summary.json">All results &amp; definitions</a>'
    if summary.get('mean_comparison_rows'):downloads+='<a href="mean-comparisons.csv">Matched mean-current errors</a>'
    if summary.get('numerical_control_rows'):downloads+='<a href="numerical-controls.csv">Grid and time-step comparisons</a>'
    replacements={'TOTAL':len(summary['runs']),'COUNTS':counts,'UPDATED':esc(summary['generated_at']),'VERSION':esc(summary['source_version']),'DOWNLOADS':downloads,
        'FIGURES':figures,'BUDGET_OVERVIEW':budget,'RUNS':''.join(rows),'RUN_FIGURES':''.join(cards) or '<p class="small">All calculations are awaiting verified output.</p>'}
    page=(Path(__file__).parent/'moments-template.html').read_text()
    for key,value in replacements.items():page=page.replace('{{'+key+'}}',str(value))
    if '{{' in page:raise ValueError('Unfilled gallery template')
    (output/'index.html').write_text(page);shutil.copyfile(Path(__file__).parent/'style.css',output/'style.css')


def render_run(result,dest):
    esc=lambda x:html.escape(str(x),quote=True);name=esc(result['id'])
    links=''.join(f'<a href="{p}">{p}</a> ' for p in ('summary.json','moments.csv','regional-budgets.csv','mean-profiles.csv'))
    plots=''.join(f'<figure><a href="{f}"><img src="{f}" alt="{esc(caption)}"></a><figcaption>{esc(caption)}</figcaption></figure>' for f,caption in [
        ('moments.png','True q-enstrophy production, background-gradient conversion, mean/variance exchange and signed native feedback are separate terms.'),
        ('closure.png','Temporal residual = observed moment change minus integrated fresh native tendency. Spatial residual = integrated native tendency minus continuum source and boundary-transport balance. Neither residual is called physical dissipation.'),
        ('current-profiles.png','Eulerian current is UL minus actual Stokes drift. Current changes subtract this run’s own initial horizontal mean.')])
    (dest/'index.html').write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{name} · Vorticity moments</title><link rel="stylesheet" href="../../style.css"></head><body><main class="wrap"><nav><a href="../../">← Vorticity moments campaign</a></nav><div class="hero"><p class="eyebrow">Verified completed calculation</p><h1>{name}</h1><p class="lead">{esc(run_label(result))} · {result["final_time_s"]:g} s</p></div><div class="downloads">{links}</div><p class="small">Global moments retain every diagnostic sample. Regional budgets are sampled every 20 records after integration on the full diagnostic grid. Published profiles select five times. Native 3D fields remain private.</p>{plots}</main></body></html>')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--approved-runs',type=Path);a=p.parse_args()
    root=a.campaign_root.resolve();output=a.output.resolve();manifest=json.loads((root/'campaign.json').read_text());source=json.loads((root/'source-manifest.json').read_text());entries=manifest['runs'];ids=[identifier(e['id']) for e in entries]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate campaign IDs')
    approved=None if a.approved_runs is None else json.loads(a.approved_runs.read_text())
    if approved is not None and (not isinstance(approved,list) or len(set(approved))!=len(approved) or any(x not in ids for x in approved)):raise ValueError('Invalid approved-run list')
    cache=root/'data/moment-analysis-cache';output.mkdir(parents=True,exist_ok=True);(cache/'matplotlib').mkdir(parents=True,exist_ok=True);os.environ.setdefault('MPLCONFIGDIR',str(cache/'matplotlib'))
    results=[]
    for entry in entries:
        status,receipt=campaign_status(root,entry,source['version'])
        if status=='complete' and approved is not None and entry['id'] not in approved:status='waiting'
        if status=='complete':
            result=analyze_moment_run(root,entry,receipt,cache/entry['id']);dest=output/'runs'/entry['id'];dest.mkdir(parents=True,exist_ok=True)
            for name in PUBLIC_FILES:shutil.copyfile(cache/entry['id']/name,dest/name)
            render_run(result,dest)
        else:result={'id':entry['id'],'status':status}
        for key in ('experiment','group','cell','seed','reference','comparison_kind'):result[key]=entry.get(key)
        results.append(result)
    figures,comparison=mean_comparisons(entries,results,cache,output)
    figures=overview(results,cache,output)+figures
    controls=numerical_controls(entries,results,cache,output)
    counts={s:sum(r['status']==s for r in results) for s in ('complete','running','queued','waiting','failed')}
    summary=dict(schema_version=1,analysis_version=VERSION,source_version=source['version'],generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),counts=counts,runs=results,figures=figures,
        mean_comparison_rows=len(comparison),numerical_control_rows=len(controls),methods={
        'q':'q = d_z uL - d_x w - d_z us, Eulerian transverse vorticity.',
        'T':'T = q d_y v + d_y w d_x v - d_y u d_z v. T=0 for all y-independent flows, including nonzero v(x,z).',
        'transport':'D_L q = T + Dq. The term -v q_y transports q; it is not an additional enstrophy production term.',
        'q0':'q0=-alpha d_z us_reference is retained. Its native centered discrete derivative defines a=q-q0.',
        'anomaly':'D_L a=T-w q0_prime+Dq; Pa_background=-<a w q0_prime> can change anomaly enstrophy in 2D.',
        'partition':'Zq=Zmean+Zvariance; exchange <F·gradQ> has opposite signs in their budgets. 3D production can change their sum.',
        'native_feedback':'Curl of full native advection minus advection evaluated on the y-mean field; weighted by Q or Q-q0. Distinct from a continuum Reynolds-stress approximation.',
        'closure':'Temporal closure compares observed moments with integrated freshly evaluated native qdot. Spatial closure compares the native rate with continuum production, viscosity, forcing, pressure and face transport. Signed residuals remain visible.',
        'boundary':'Full-depth and Stokes-layer budgets use complete cells and stored face transport; layer-base flux is retained.',
        'geometry':'Zvariance/Za removes the large prescribed background from the denominator; no threshold is used to assign a regime.',
        'comparison':'Horizontal-mean UE=UL-us and its change from own initial profile. Exact 2D, tiny, intermediate and finite initial disturbances use the same base field. Finite disturbances add energy.',
        'limits':'Finite-time cases with limited seed and numerical checks. A production or geometric measure alone neither proves a closure nor establishes an oceanic mixing barrier.'})
    write_json(output/'summary.json',summary);render_moments(summary,output)
    size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if size>PUBLIC_LIMIT:raise ValueError(f'New gallery assets exceed {PUBLIC_LIMIT/1024**2:g} MiB public size limit')
    print(json.dumps(dict(reported=True,counts=counts,source_version=source['version'],public_bytes=size)))


if __name__=='__main__':main()
