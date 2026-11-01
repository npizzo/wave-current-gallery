#!/usr/bin/env python3
"""Verified pilot: wind-driven Eulerian shear, roll susceptibility and jet erosion."""
from __future__ import annotations
import argparse, hashlib, html, json, shutil, sys
from pathlib import Path
import numpy as np
import onset_reference as old
from shear_stability import native_shear, sampled_spectrum

SAMPLE_TIMES=(0,40,80,120,160,200,320,440,560,680,800,920,1040,1160,1200)
MODES=tuple(range(1,32))
LABELS={'3d-u0-reference':'3D · no added wind (existing)', '3d-u002-reference':'3D · 2 mm/s (existing)',
        '3d-u00075':'3D · 0.75 mm/s','3d-u00125':'3D · 1.25 mm/s','2d-u0':'2D · no added wind','2d-u002':'2D · 2 mm/s'}
COLORS={'3d-u0-reference':'#263e47','3d-u002-reference':'#c05a20','3d-u00075':'#318c82','3d-u00125':'#8c5b9f','2d-u0':'#263e47','2d-u002':'#c05a20'}


def reference_case(root,entry):
    """Read immutable existing diagnostics, using both old and new hash pins."""
    pins=json.loads((root/'references-manifest.json').read_text())['references'][entry['id']]
    runtime=json.loads((root/'runtime/references.json').read_text())['references'][entry['id']]
    if runtime.get('files',pins['files'])!=pins['files']:raise ValueError('Runtime reference inventory differs from frozen pins')
    sys.path.insert(0,str(root/'scripts'))
    from references import verify_references
    verify_references(root)
    folder=Path(runtime['root']).resolve();campaign_root=Path(runtime['campaign_root']).resolve()
    if folder!=campaign_root/'data/runs'/pins['case_id']:raise ValueError('Reference run location differs')
    for name,sha in pins['files'].items():
        p=old.safe_path(folder,name)
        if old.digest(p)!=sha:raise ValueError('Reference diagnostic differs: '+name)
    prior_entry=next(x for x in json.loads((campaign_root/'campaign.json').read_text())['runs'] if x['id']==pins['case_id'])
    status,receipt=old.run_state(campaign_root,prior_entry,pins['source_version'],None)
    if status!='complete':raise ValueError('Reference is not verified complete')
    if any(receipt['files'].get(k)!=v for k,v in pins['files'].items()):raise ValueError('Old reference receipt differs from frozen inventory')
    result,data,hashes=old.assess(campaign_root,prior_entry,receipt)
    if any(pins['files'].get(k)!=v for k,v in hashes.items()):raise ValueError('Used reference input lacks the new frozen fingerprint')
    result.update(id=entry['id'],reused=True,mode='3d',reference_source_version=pins['source_version'])
    return result,data,hashes


def first_crossing(history,key,threshold):
    return next((float(r['elapsed_s']) for r in history if r[key]>threshold),None)


def inspect_profile(ue,z,cfg):
    h,k,S,alpha,Sref,ustar,nu=(float(cfg[x]) for x in ('h','k','S','alpha','S_reference','wind_ustar','nu'))
    q,beta,bounds=native_shear(ue,h,k,S,alpha,Sref,ustar,nu)
    ss=2*k*S*np.exp(2*k*z);product=q*ss
    mask=(z>=-3/(2*k));mask[:2]=False;mask[-2:]=False
    ids=np.flatnonzero(mask)
    if len(ids)<4:raise ValueError('Active Stokes layer unresolved')
    positive=mask&(product>0);dz=h/len(z)
    groups=np.split(np.flatnonzero(positive),np.flatnonzero(np.diff(np.flatnonzero(positive))>1)+1) if np.any(positive) else []
    largest=max((len(g) for g in groups),default=0)
    record=dict(unstable_active_thickness_m=float(sum(positive)*dz),active_layer_thickness_m=float(sum(mask)*dz),
                largest_positive_active_layer_cells=largest,positive_active_layer_resolved=bool(largest>=4),
                positive_active_fraction=float(np.mean(product[mask]>0)),signed_product_min_s_minus2=float(min(product[mask])),
                signed_product_max_s_minus2=float(max(product[mask])),local_inviscid_rate_upper_s_inverse=float(np.sqrt(max(0,max(product[mask])))),
                beta_active_rms_s_inverse_m_inverse=float(np.sqrt(np.mean(beta[mask]**2))),
                Q_active_rms_s_inverse=float(np.sqrt(np.mean(q[mask]**2))),imposed_top_Q_s_inverse=bounds[1])
    return record,q,beta,ss,bounds


def analyze_profiles(result,data):
    """Same-grid laminar base is independent of later turbulent outcomes.

    Actual-profile spectra are retrospective diagnoses. Laminar-profile
    spectra use only common initial mean, prescribed forces, viscosity and
    boundaries. Neither is advertised as a nonautonomous gain calculation.
    """
    cfg=result['configuration'];z=data['z'];elapsed=data['elapsed'];hist=data['histories'];records=[];maps=[]
    for target in SAMPLE_TIMES:
        i=int(np.argmin(abs(elapsed-target)))
        if abs(elapsed[i]-target)>1e-6:continue
        record=dict(elapsed_s=float(elapsed[i]),jet_prominence_fraction=hist[i]['jet_prominence_fraction'],
                    chi_3d=hist[i]['chi_3d'],roll_rms_m_s=hist[i]['roll_rms_m_s'])
        for kind,profiles in [('actual',data['UE']),('laminar',data['baseline'])]:
            local,q,beta,ss,bounds=inspect_profile(profiles[i],z,cfg)
            spectrum=sampled_spectrum(z,q,ss,float(cfg['ly']),float(cfg['h']),float(cfg['nu']),bounds,modes=MODES)
            record[kind]=dict(**local,**spectrum)
            if kind=='actual':
                for j,zz in enumerate(z):maps.append(dict(elapsed_s=float(elapsed[i]),z_m=float(zz),UE_m_s=float(profiles[i,j]),Q_s_inverse=float(q[j]),beta_s_inverse_m_inverse=float(beta[j]),signed_product_s_minus2=float(q[j]*ss[j])))
        records.append(record)
    t=np.array([r['elapsed_s'] for r in records]);growth={}
    for kind in ('actual','laminar'):
        # Integrate each fixed ky branch before maximizing, rather than
        # permitting the disturbance to jump arbitrarily between ky modes.
        rates=np.array([[m['growth_s_inverse'] for m in r[kind]['tested_modes']] for r in records])
        exposure=old.cumulative(rates,t)
        for i,r in enumerate(records):
            r[kind]['maximum_fixed_ky_net_exposure']=float(max(exposure[i]))
            r[kind]['exposure_winning_mode']=MODES[int(np.argmax(exposure[i]))]
            r[kind]['six_cell_band_net_exposure']=float(max(exposure[i,:10]))
            r[kind]['four_cell_band_net_exposure']=float(max(exposure[i,:16]))
        growth[kind]=dict(final_maximum_fixed_ky_net_exposure=float(max(exposure[-1])),
                          any_unconverged=any(not r[kind]['vertical_converged'] for r in records),
                          any_horizontal_edge=any(r[kind]['horizontal_edge'] for r in records),
                          any_horizontal_grid_unresolved=any(not r[kind]['horizontal_grid_resolved'] for r in records),
                          any_faster_outside_native_band=any(r[kind]['faster_outside_native_band'] for r in records),
                          any_thin_positive_active_layer=any(0<r[kind]['largest_positive_active_layer_cells']<4 for r in records))
    assessment=dict(case_id=result['id'],mode=result.get('mode','3d'),
        wind_ustar_m_s=cfg['wind_ustar'],initial_3d_fraction=hist[0]['chi_3d'],
        first_3d_energy_above_1_percent_s=first_crossing(hist,'chi_3d',.01),
        first_3d_energy_above_10_percent_s=first_crossing(hist,'chi_3d',.1),
        first_roll_rms_above_1_mm_s_s=first_crossing(hist,'roll_rms_m_s',.001),
        sustained_half_prominence_loss_intervals=result['initial_jet']['sustained_half_prominence_loss_intervals'],
        linear_growth=growth,records=records,
        scope='Full-depth viscous frozen-profile roll susceptibility. Net modal exposure is an adiabatic indicator, not a finite-time propagator gain, amplitude prediction or validated universal threshold. 2D integrations prohibit transverse rolls even when their counterfactual profiles are roll-unstable.')
    return assessment,maps


def cached_profile_analysis(root,result,data,hashes):
    identity={'source_files':{p.name:old.digest(p) for p in Path(__file__).parent.glob('*.py')},'inputs':hashes,'id':result['id']}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    p=root/'data/analysis-cache/shear'/f'{result["id"]}-{key}.json'
    if p.exists():
        cache=json.loads(p.read_text())
        if cache.get('identity')!=identity:raise ValueError('Analysis cache mismatch')
        return cache['assessment'],cache['maps']
    assessment,maps=analyze_profiles(result,data)
    old.write_json(p,dict(identity=identity,assessment=assessment,maps=maps));return assessment,maps


def compare_2d_3d(results,curves):
    complete={r['id']:r for r in results if r['status']=='complete'}
    pairs=[]
    for two,three in [('2d-u0','3d-u0-reference'),('2d-u002','3d-u002-reference')]:
        if two not in complete or three not in complete:continue
        a,b=curves[two],curves[three]
        if not np.allclose(a['UE'][0],b['UE'][0],rtol=0,atol=1e-11):raise ValueError('Projected pair changed initial mean')
        if not np.allclose(a['elapsed'],b['elapsed'],rtol=0,atol=1e-8):raise ValueError('Projected pair times differ')
        for field in ('Uc','Wc'):
            if not np.allclose(a['maps'][field][0],b['maps'][field][0],rtol=0,atol=1e-11):raise ValueError('Projected pair changed initial 2D fields')
        pairs.append(dict(two_dimensional=two,three_dimensional=three,initial_mean_and_2d_fields_match=True,
            two_d_half_loss=complete[two]['initial_jet']['sustained_half_prominence_loss_intervals'],
            three_d_half_loss=complete[three]['initial_jet']['sustained_half_prominence_loss_intervals'],
            final_profile_rms_difference_m_s=float(old.rms(a['UE'][-1]-b['UE'][-1],a['dz'])),
            interpretation='Removing transverse variation removes its nonlinear feedback and changes later in-plane dynamics; this intervention tests whether 3D freedom is necessary for the additional erosion in this pair.'))
    return pairs


def intervention_difference(curves):
    keys=('3d-u0-reference','3d-u002-reference','2d-u0','2d-u002')
    if any(k not in curves for k in keys):return dict(status='AWAITING_MATCHED_2D_CONTROLS',rows=[])
    rows=[]
    for time in (200,400,600,800,1000,1200):
        values=[]
        for key in keys:
            history=curves[key]['histories'];row=next((r for r in history if abs(r['elapsed_s']-time)<1e-7),None)
            if row is None:raise ValueError('Missing intervention comparison time')
            values.append(row['jet_prominence_fraction'])
        p30,p3w,p20,p2w=values
        rows.append(dict(elapsed_s=time,wind_effect_3d=p3w-p30,wind_effect_2d=p2w-p20,
                         difference_of_wind_effects=(p3w-p30)-(p2w-p20)))
    return dict(status='MATCHED_INTERVENTION_COMPLETE',rows=rows,
                interpretation='Negative difference means added wind reduces retained prominence more with 3D freedom than in the projected 2D system, relative to each system’s no-wind control. Nonlinear evolution and disappearance/rematching of extrema require the accompanying profile histories; this is not an isolated-term budget.')


def make_plots(out,results,curves,assessments):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    ready=[r for r in results if r['status']=='complete']
    if not ready:return []
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for r in ready:
        rid=r['id'];d=curves[rid];style='--' if r.get('mode')=='2d' else '-';c=COLORS.get(rid);lab=LABELS.get(rid,rid)
        a=assessments[rid];t=np.array([v['elapsed_s'] for v in a['records']])
        ax[0,0].plot(d['elapsed'],[x['jet_prominence_fraction'] for x in d['histories']],style,color=c,label=lab)
        ax[0,1].plot(d['elapsed'],100*np.array([x['chi_3d'] for x in d['histories']]),style,color=c,label=lab)
        ax[1,0].plot(t,[x['laminar']['six_cell_band_growth_s_inverse'] for x in a['records']],style,color=c,label=lab)
        ax[1,1].plot(t,[x['laminar']['six_cell_band_net_exposure'] for x in a['records']],style,color=c,label=lab)
    ax[0,0].axhline(.5,color='.5',ls=':');ax[0,0].set(title='Retention of the original jet',ylabel='Prominence / initial prominence')
    ax[0,1].set(title='Observed three-dimensional motion',ylabel='3D fraction of eddy energy (%)')
    ax[1,0].axhline(0,color='.5',ls=':');ax[1,0].set(title='Laminar roll growth: ≥6 cells per wavelength',ylabel='Largest tested growth rate in this band (1/s)')
    ax[1,1].axhline(0,color='.5',ls=':');ax[1,1].set(title='Frozen-profile exposure, not propagator gain',ylabel='Fixed-ky growth integral (≥6-cell band)')
    for a in ax.flat:a.set_xlabel('Time after switch (s)');a.grid(alpha=.15)
    ax[0,0].legend(fontsize=8);fig.savefig(out/'pilot-comparison.png',dpi=150);plt.close(fig)
    # Profiles show the control and strongest wind, with pending new cases
    # appearing automatically when verified. All panels use physical signs.
    n=len(ready);fig,ax=plt.subplots(n,3,figsize=(12,3.5*n),squeeze=False,constrained_layout=True)
    for j,r in enumerate(ready):
        d=curves[r['id']];cfg=r['configuration'];z=d['z']
        for target in (0,80,160,400,600):
            i=int(np.argmin(abs(d['elapsed']-target)));local,q,beta,ss,bounds=inspect_profile(d['UE'][i],z,cfg)
            label=f'{target} s';ax[j,0].plot(1e3*d['UE'][i],z,label=label);ax[j,1].plot(q*ss,z,label=label);ax[j,2].plot(beta,z,label=label)
        ax[j,0].set_title(LABELS.get(r['id'],r['id']),fontsize=10)
        ax[j,0].set_xlabel('Eulerian current (mm/s)');ax[j,1].set_xlabel('Q × Stokes shear (1/s²)');ax[j,2].set_xlabel('Effective beta (1/(m s))')
        for a in ax[j,:]:a.axvline(0,color='.5',lw=.7);a.set_ylabel('z (m)');a.grid(alpha=.15)
    ax[0,0].legend(fontsize=8);fig.savefig(out/'shear-and-curvature.png',dpi=140);plt.close(fig)
    return ['pilot-comparison.png','shear-and-curvature.png']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--approved-runs',type=Path);a=p.parse_args()
    root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    campaign=json.loads((root/'campaign.json').read_text());source=json.loads((root/'source-manifest.json').read_text());approved=set(json.loads(a.approved_runs.read_text())) if a.approved_runs else None
    results=[];curves={};hashes={};assessments={}
    for entry in campaign.get('references',[])+campaign['runs']:
        rid=old.identifier(entry['id']);reused=entry in campaign.get('references',[])
        if reused:
            result,data,checked=reference_case(root,entry)
        else:
            status,receipt=old.run_state(root,entry,source['version'],approved)
            if status!='complete':results.append(dict(id=rid,status=status,mode=entry.get('mode'),reused=False));continue
            result,data,checked=old.assess(root,entry,receipt);result.update(mode=entry.get('mode','3d'),reused=False)
        results.append(result);curves[rid]=data;hashes[rid]=checked
        assessment,maps=cached_profile_analysis(root,result,data,checked);assessments[rid]=assessment
        folder=out/'runs'/rid;folder.mkdir(parents=True,exist_ok=True)
        old.write_json(folder/'summary.json',result);old.write_json(folder/'shear-assessment.json',assessment)
        old.write_csv(folder/'time-series.csv',data['histories']);old.write_csv(folder/'shear-profiles.csv',maps)
        flat=[]
        for r in assessment['records']:
            row={k:r[k] for k in ('elapsed_s','jet_prominence_fraction','chi_3d','roll_rms_m_s')}
            for kind in ('actual','laminar'):
                row.update({kind+'_'+k:v for k,v in r[kind].items() if isinstance(v,(int,float,bool))})
            flat.append(row)
        old.write_csv(folder/'shear-timeseries.csv',flat)
    pairs=compare_2d_3d(results,curves);intervention=intervention_difference(curves);figures=make_plots(out,results,curves,assessments)
    counts={s:sum(r['status']==s and not r.get('reused',False) for r in results) for s in ('complete','running','queued','waiting','failed')}
    summary=dict(title='Eulerian shear and jet disruption: small pilot',source_version=source['version'],new_run_counts=counts,
        reused_runs=sum(r.get('reused',False) for r in results),runs=results,shear_assessments=assessments,projected_pairs=pairs,intervention=intervention,
        assessment='PILOT_COMPLETE' if counts['complete']==len(campaign['runs']) else 'PILOT_IN_PROGRESS',
        qualification='Frozen-profile susceptibility is a candidate predictor, not a validated growth forecast. No universal threshold is fitted to this single-parent pilot.')
    old.write_json(out/'summary.json',summary);old.write_json(out/'provenance.json',dict(source_version=source['version'],verified_inputs=hashes,native_arrays_published=False))
    (out/'style.css').write_text(old.STYLE)
    rows=''.join('<tr><td>'+html.escape(LABELS.get(r['id'],r['id']))+'</td><td>'+r['status']+'</td><td>'+('<a href="runs/'+r['id']+'/shear-assessment.json">Shear analysis</a> · <a href="runs/'+r['id']+'/time-series.csv">Time series</a>' if r['status']=='complete' else 'Awaiting verified completion')+'</td></tr>' for r in results)
    captions={'pilot-comparison.png':'Top: observed jet prominence and 3D energy. Bottom: full-depth viscous roll eigenvalues about the maintained laminar continuation, restricted to wavelengths with at least six cells on the parent transverse grid. The reference uses only the initial mean and prescribed forcing; the full and four-cell spectra are retained in the data. Six cells is a screening choice, not demonstrated nonlinear convergence. The exposure integrates each fixed spanwise wavenumber separately before taking the maximum. It assumes slowly adapting modes and is not a time-dependent propagator gain. Dashed lines are 2D interventions that prohibit transverse rolls; their spectra are counterfactual susceptibility if 3D freedom were restored.',
              'shear-and-curvature.png':'The signed Eulerian-shear × Stokes-shear product diagnoses local roll feedback; positive values favour inviscid CL2 growth. Effective beta is Eulerian curvature. A positive surface point is not a global instability threshold. Native boundary stresses enter the derivative; near-wall values are sensitive to the imposed boundary layer.'}
    sections=''.join('<figure><img src="'+f+'" alt="'+html.escape(captions[f])+'"><figcaption>'+html.escape(captions[f])+'</figcaption></figure>' for f in figures)
    if intervention['rows']:
        ir=''.join('<tr><td>'+str(r['elapsed_s'])+'</td><td>'+format(r['wind_effect_3d'],'.3f')+'</td><td>'+format(r['wind_effect_2d'],'.3f')+'</td><td>'+format(r['difference_of_wind_effects'],'.3f')+'</td></tr>' for r in intervention['rows'])
        sections+='<section class="panel"><h2>Does 3D freedom change the wind response?</h2><table><thead><tr><th>Elapsed s</th><th>Wind − control, 3D</th><th>Wind − control, 2D</th><th>Difference of effects</th></tr></thead><tbody>'+ir+'</tbody></table><p>Values are fractions of initial jet prominence. A negative last column indicates additional wind-driven erosion when 3D freedom is allowed, beyond the wind response in the projected 2D system. This intervention includes indirect changes to subsequent 2D eddies.</p></section>'
    else:sections+='<section class="panel"><h2>Does 3D freedom change the wind response?</h2><p>The difference of wind effects requires both new projected 2D controls. It will compare (wind − control) in 3D against (wind − control) in 2D. This assessment is pending; the existing pair alone cannot resolve it.</p></section>'
    qualifiers=' '.join(html.escape(rid)+': '+('vertical-rate refinement flag; ' if v['linear_growth']['laminar']['any_unconverged'] else '')+('fastest sampled mode reaches horizontal search edge; ' if v['linear_growth']['laminar']['any_horizontal_edge'] else '')+('some leading wavelengths have fewer than four transverse grid cells; ' if v['linear_growth']['laminar']['any_horizontal_grid_unresolved'] else '')+('some positive-shear surface layers have fewer than four vertical cells; ' if v['linear_growth']['laminar']['any_thin_positive_active_layer'] else '')+('faster modes occur beyond the native transverse band; ' if v['linear_growth']['laminar']['any_faster_outside_native_band'] else '') for rid,v in assessments.items())
    body='<nav><a href="../../">Gallery</a><a href="../wind-onset/">Original wind-onset pair</a></nav><h1>Does Eulerian shear predict jet disruption?</h1><p>This small pilot starts from the same spontaneously generated jet and tests two weaker wind stresses plus two runs constrained to remain independent of the transverse direction. The completed no-wind and strong-wind 3D runs are reused without rerunning them.</p><div class="panel"><strong>Physical question:</strong> wind changes Q = ∂z U<sub>E</sub> and β<sub>eff</sub> = ∂z Q. Signed Q ∂z u<sub>s</sub> controls the local roll feedback, while Eulerian curvature controls the reduced vortex-wave restoring effect. We test whether shear changes precede 3D growth and whether removing 3D freedom alters the additional jet erosion.</div><table><thead><tr><th>Case</th><th>Status</th><th>Outputs</th></tr></thead><tbody>'+rows+'</tbody></table>'+sections+'<section class="panel"><h2>How this pilot is assessed</h2><p>The control is natural continuation from the same saved field. The 2D interventions preserve its transverse mean and in-plane eddies while removing transverse variations. Their difference from 3D evolution tests the role of 3D freedom, including its indirect effect on later 2D motions. It does not identify one isolated term.</p><p>The linear model retains depth dependence, viscosity and homogeneous perturbation stress boundaries. It tests every spanwise mode 1–31 on the 64-cell reference grid, checks modes 32/48/64 beyond that band, and uses 48 vertical basis modes with 24/72 checks of the winning mode. Leading wavelengths with fewer than four transverse grid cells, or positive-shear layers thinner than four vertical cells, are flagged. These are resolution screens, not nonlinear convergence tests; the continuum growth operator also differs from the native centered-advection stencil. It neglects the x-varying eddy field and transient modal adaptation. A positive local shear product or an integrated frozen eigenvalue is therefore not by itself a validated jet-destruction criterion.</p><p>Resolution flags: '+qualifiers+'</p><p>Existing evidence is retrospective. The two intermediate stresses are new prospective tests. A broader sweep should wait until the rate diagnostics, 2D intervention and budget checks agree. No absence of jets with small 3D feedback is counted as failure of a necessary-condition argument.</p><p><a href="summary.json">Full analysis</a> · <a href="pre-run-predictions.json">Predictions recorded before submission</a> · <a href="pre-run-prediction-input.json">Prediction input</a> · <a href="PROTOCOL.txt">Frozen protocol</a> · <a href="campaign.json">Cases</a> · <a href="provenance.json">Input provenance</a></p></section><footer>Only verified completed and backed-up simulations enter the figures. Original simulation sources and results are unchanged.</footer>'
    (out/'index.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Eulerian shear and jet disruption</title><link rel="stylesheet" href="style.css"></head><body><main>'+body+'</main></body></html>\n')
    for name in ('PROTOCOL.txt','campaign.json','source-manifest.json','references-manifest.json'):
        if (root/name).exists():shutil.copy2(root/name,out/name)
    for name in ('pre-run-predictions.json','pre-run-prediction-input.json'):
        shutil.copy2(root/'reference'/name,out/name)
    size=sum(x.stat().st_size for x in out.rglob('*') if x.is_file())
    if size>25*1024**2:raise ValueError('Pilot public output exceeds 25 MiB')
    print(json.dumps(dict(new_run_counts=counts,reused_runs=summary['reused_runs'],public_bytes=size)))


if __name__=='__main__':main()
