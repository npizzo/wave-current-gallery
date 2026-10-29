#!/usr/bin/env python3
"""Recompute campaign interpretation metrics and produce standalone scientific figures.

Inputs are the verified published summary, full-cadence moment CSVs, compact
velocity-geometry CSVs, and derived mean-profile histories. Native snapshots are
neither needed nor exported. No original simulation source or results are changed.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, tempfile
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'vorticity-moment-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS={100:'#157f88',200:'#4667a6',300:'#cb6731'}
ETA_COLORS={1e-6:'#157f88',1.:'#cb6731'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def table(path):
    a=np.atleast_1d(np.genfromtxt(path,names=True,delimiter=','))
    assert len(a)>1 and np.all(np.diff(a['time_s'])>0),path
    return a


def timed_mean(t,y,start=600):
    use=t>=start
    assert np.sum(use)>1
    return float(np.trapezoid(y[use],t[use])/(t[use][-1]-t[use][0]))


def cumtrap(t,y):return np.r_[0.,np.cumsum(np.diff(t)*(y[1:]+y[:-1])/2)]


def base_id(r):
    rid=r['id']
    if r['group']=='2d3c':return ('central' if r['config']['waves'] else 'nowaves')+'-re100-2d-s1'
    if rid.startswith('central-halfdt-'):return 'central-re300-2d-s1'
    return r['reference']


def profile_error(a,b):
    assert np.array_equal(a['time'],b['time'])
    assert np.array_equal(a['z'],b['z'])
    assert np.all(a['dz']>0)
    da=a['UE']-a['UE'][0];db=b['UE']-b['UE'][0]
    weights=a['dz']/np.sum(a['dz'])
    err=np.sqrt(np.sum((da-db)**2*weights,axis=1))
    amp=np.sqrt(np.sum(db**2*weights,axis=1))
    return err,amp


def compute(data):
    summary=json.loads((data/'summary.json').read_text())
    assert summary['counts']=={'complete':36,'running':0,'queued':0,'waiting':0,'failed':0}
    profiles={r['id']:dict(np.load(data/'runs'/r['id']/'profile-history.npz')) for r in summary['runs']}
    results=[];histories={};inputs={'summary.json':digest(data/'summary.json')}
    for r in summary['runs']:
        rid=r['id'];c=r['config'];folder=data/'runs'/rid
        for name in ('moments.csv','regime-budget.csv','profile-history.npz'):
            inputs[f'runs/{rid}/{name}']=digest(folder/name)
        g=table(folder/'regime-budget.csv');m=table(folder/'moments.csv');p=profiles[rid];b=profiles[base_id(r)]
        tg=g['time_s'];tm=m['time_s'];assert tg[-1]==tm[-1]==1200
        chi_e=g['Eperp']/g['Eeddy'];chi_q=m['Zvariance']/m['Za']
        assert np.allclose(chi_e,g['chi_3d'],rtol=1e-10,atol=1e-14)
        assert np.all(chi_e>=-1e-14) and np.all(chi_e<=1+1e-10)
        assert np.all(chi_q>=-1e-14) and np.all(chi_q<=1+1e-8)
        error,amp=profile_error(p,b); eta=float(c['perturb_energy_ratio']);Z0=float(m['Za'][0])
        rec=dict(id=rid,matched_2d=base_id(r),experiment=r['experiment'],group=r['group'],seed=c['seed'],
            alpha=c['alpha'],waves=bool(c['waves']),nx=c['nx'],ny=c['ny'],nz=c['nz'],max_dt_s=c['max_dt'],
            Re_inj=c['eddy_rms']*c['injection_wavelength']/c['nu'],eta=eta,
            U_2D_initial_m_s=c['eddy_rms'],u_s_surface_m_s=c['S'],
            u_3D_initial_m_s=c['eddy_rms']*np.sqrt(eta),
            u_3D_initial_over_us=c['eddy_rms']*np.sqrt(eta)/c['S'] if c['S']>0 else None,
            total_initial_eddy_over_us=c['eddy_rms']*np.sqrt(1+eta)/c['S'] if c['S']>0 and c.get('two_d_three_c_rms',0)==0 else None,
            final_time_s=1200,geometry_sampling_s=float(np.max(np.diff(tg))),
            chi_E_final=float(chi_e[-1]),chi_E_max_sampled=float(max(chi_e)),chi_E_mean_600_1200=timed_mean(tg,chi_e),
            chi_q_final=float(chi_q[-1]),chi_q_max_sampled=float(max(chi_q)),chi_q_mean_600_1200=timed_mean(tm,chi_q),
            Eperp_initial_m2_s2=float(g['Eperp'][0]),Eperp_final_m2_s2=float(g['Eperp'][-1]),
            Eperp_final_over_initial=float(g['Eperp'][-1]/g['Eperp'][0]) if g['Eperp'][0]>0 else None,
            Eperp_peak_over_initial=float(max(g['Eperp'])/g['Eperp'][0]) if g['Eperp'][0]>0 else None,
            Eeddy_final_over_initial=float(g['Eeddy'][-1]/g['Eeddy'][0]),
            current_error_final_mm_s=1000*float(error[-1]),current_error_max_mm_s=1000*float(max(error)),
            current_error_mean_600_1200_mm_s=1000*timed_mean(p['time'],error),
            reference_current_change_final_rms_mm_s=1000*float(amp[-1]),
            current_error_final_over_reference_change=float(error[-1]/amp[-1]) if amp[-1]>1e-15 else None,
            current_error_mean_600_1200_over_mean_reference_change=timed_mean(p['time'],error)/timed_mean(p['time'],amp),
            native_mean_anomaly_feedback_integral=float(np.trapezoid(m['Pmean_anomaly_feedback_native'],tm)),
            native_mean_anomaly_feedback_integral_over_initial_Za=float(np.trapezoid(m['Pmean_anomaly_feedback_native'],tm)/Z0),
            q_production_integral=float(np.trapezoid(m['Pq_T'],tm)),
            absolute_local_q_production_integral=float(np.trapezoid(m['Pq_T_abs'],tm)),
            anomaly_production_integral_over_initial_Za=float(np.trapezoid(m['Pa_T'],tm)/Z0),
            background_conversion_integral_over_initial_Za=float(np.trapezoid(m['Pa_background'],tm)/Z0))
        # These residuals use the original published full-depth quadrature and its
        # stated scale (max of initial moment, excursion, and native integral).
        for quantity in ('Za','Zq','Zmean_anomaly'):
            closure=r['closure']['full'][quantity]
            for field in ('max_abs_temporal_residual_normalized','max_abs_spatial_residual_normalized'):
                rec[quantity+'_'+field]=closure[field]
        results.append(rec)
        histories[rid]=dict(geometry_time_s=tg,chi_E=chi_e,Eperp=g['Eperp'],Eeddy=g['Eeddy'],moment_time_s=tm,chi_q=chi_q,matched_2d=base_id(r),reference_change_rms_m_s=amp,
            profile_time_s=p['time'],current_error_over_final_reference_change=error/amp[-1],
            current_error_mm_s=error*1000,reference_final_change_mm_s=1000*amp[-1])
    return summary,results,histories,inputs


def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,
        'legend.fontsize':8.8,'xtick.labelsize':9,'ytick.labelsize':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'figure.facecolor':'white',
        'savefig.facecolor':'white','axes.linewidth':.75})


def decorate(a,letter,title):
    a.set_title(f'{letter}  {title}',loc='left',fontweight='bold',pad=10)
    a.grid(axis='y',which='major',color='#dce1e4',lw=.6)
    a.set_axisbelow(True)


def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=220,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)


def main_figure(rows,out):
    byid={r['id']:r for r in rows};fig,aa=plt.subplots(2,3,figsize=(14.2,9.2),layout='constrained')
    fields=('chi_E_final','Eperp_final_over_initial','current_error_final_over_reference_change');mult=(100,1,100)
    for re in (100,200,300):
        rr=[byid[f'central-re{re}-{g}-s1'] for g in ('tiny100','tiny','moderate','finite')]
        x=[r['u_3D_initial_over_us'] for r in rr]
        for a,key,factor in zip(aa[0],fields,mult):
            a.plot(x,[factor*r[key] for r in rr],'-o',color=COLORS[re],ms=4.7,lw=1.7,label=rf'$Re_{{\rm inj}}={re}$')
    for prefix,seed,marker,color in [('central-fine',1,'D','#263844'),('central',2,'s','#a14185')]:
        rr=[byid[f'{prefix}-re300-{g}-s{seed}'] for g in ('tiny','finite')]
        for a,key,factor in zip(aa[0],fields,mult):
            a.plot([r['u_3D_initial_over_us'] for r in rr],[factor*r[key] for r in rr],linestyle='none',
                marker=marker,ms=7,mfc='none',mec=color,mew=1.4,zorder=7)
    titles=('Final 3D kinetic-energy fraction','Absolute 3D energy: growth or decay','Departure of the mean current')
    ylabels=(r'$100\chi_E(1200)$  (%)',r'$E_{\perp}(1200)/E_{\perp}(0)$',r'$100\epsilon_U(1200)$  (%)')
    for i,a in enumerate(aa[0]):
        decorate(a,'abc'[i],titles[i]);a.set(xscale='log',yscale='log',xlabel=r"Initial $u'_{\mathrm{3D},0}/u_{s0}$",ylabel=ylabels[i])
        a.set_xticks([1e-5,1e-4,1e-2,1e-1]);a.set_xlim(6e-6,.2)
    aa[0,0].set_ylim(4e-7,100);aa[0,2].set_ylim(5e-6,180)
    aa[0,1].axhline(1,color='#67727b',ls=':',lw=1)
    aa[0,1].text(.96,.53,'growth > 1\ndecay < 1',transform=aa[0,1].transAxes,ha='right',va='center',color='#555',fontsize=8.5)
    for eta,g in ((1e-6,'tiny'),(1.,'finite')):
        rr=[byid[f'{prefix}-re300-{g}-s1'] for prefix in ('alpha05','central','alpha1')]
        for a,key in zip(aa[1,:2],('chi_E_final','current_error_final_over_reference_change')):
            a.plot([r['alpha'] for r in rr],[100*r[key] for r in rr],'-o',lw=1.7,ms=4.7,color=ETA_COLORS[eta],label=r'$\eta=10^{-6}$' if eta<1 else r'$\eta=1$')
    for i,a in enumerate(aa[1,:2]):
        decorate(a,'de'[i],('Background shear changes 3D motion','Background shear changes mean feedback')[i])
        a.set(xlabel=r'Background parameter $\alpha$  ($q_0=-\alpha u_s^{\prime}$)',yscale='log',xticks=[.5,.75,1],
              ylabel=(r'$100\chi_E(1200)$  (%)',r'$100\epsilon_U(1200)$  (%)')[i])
        a.legend(frameon=False,loc='best')
    for prefix,label,color in [('central','Waves on','#157f88'),('nowaves','Waves off','#7b556c')]:
        rr=[byid[f'{prefix}-re{re}-tiny-s1'] for re in (100,300)]
        aa[1,2].plot([r['Re_inj'] for r in rr],[r['Eperp_final_over_initial'] for r in rr],'-o',color=color,lw=1.7,ms=5,label=label)
    decorate(aa[1,2],'f','Effect of waves on 3D energy')
    aa[1,2].set(yscale='log',xlabel=r'Injection-scale $Re_{\rm inj}=U_{2D}\ell/\nu$',ylabel=r'$E_{\perp}(1200)/E_{\perp}(0)$',xticks=[100,300],xlim=(75,325))
    aa[1,2].axhline(1,color='#67727b',ls=':',lw=1);aa[1,2].legend(frameon=False,loc='upper left')
    aa[1,2].text(.98,.04,r'$\alpha=0.75,\ \eta=10^{-6}$',transform=aa[1,2].transAxes,ha='right',fontsize=9)
    handles=[Line2D([],[],color=COLORS[re],marker='o',label=rf'$Re_{{\rm inj}}={re}$') for re in (100,200,300)]
    handles += [Line2D([],[],color='#263844',marker='D',ls='none',mfc='none',label='Fine grid + half step (Re = 300)'),Line2D([],[],color='#a14185',marker='s',ls='none',mfc='none',label='Second seed (Re = 300)')]
    fig.legend(handles=handles,loc='outside upper center',ncol=5,frameon=False)
    fig.get_layout_engine().set(h_pad=.10,w_pad=.08,hspace=.08,wspace=.03)
    save(fig,out/'parameter-summary')


def history_figure(histories,out):
    ids=('central-re300-finite-s2','nowaves-re300-tiny-s1')
    titles=('Waves on: finite disturbance, second seed','Waves off: tiny initial disturbance')
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    for i,(rid,title,a) in enumerate(zip(ids,titles,axes)):
        h=histories[rid]
        a.plot(h['geometry_time_s'],100*h['chi_E'],color='#157f88',lw=2,label=r'3D energy fraction $100\chi_E(t)$')
        a.plot(h['profile_time_s'],100*h['current_error_over_final_reference_change'],color='#cb6731',lw=2,label='Mean-current error / final 2D change')
        decorate(a,'ab'[i],title);a.set(xlabel='Time (s)',ylabel='Percent',xlim=(0,1200))
        a.legend(frameon=False,loc='best',fontsize=8.5)
        a.text(*((.97,.20) if i==0 else (.63,.68)),f'At 1200 s: {100*h["chi_E"][-1]:.2g}% 3D energy\n{100*h["current_error_over_final_reference_change"][-1]:.3g}% current error',transform=a.transAxes,ha='right',va='center',fontsize=9,bbox=dict(facecolor='white',alpha=.94,edgecolor='none',pad=4))
    save(fig,out/'geometry-and-memory')


def compare_review(rows,review):
    if not review.is_file():return {'available':False}
    mapping={'chiE_f':'chi_E_final','chiE_max':'chi_E_max_sampled','chiE_late':'chi_E_mean_600_1200',
             'perp_growth':'Eperp_peak_over_initial','perp_final_ratio':'Eperp_final_over_initial',
             'chiQ_f':'chi_q_final','chiQ_max':'chi_q_max_sampled','chiQ_late':'chi_q_mean_600_1200',
             'error_final_mm_s':'current_error_final_mm_s','error_max_mm_s':'current_error_max_mm_s',
             'relative_final':'current_error_final_over_reference_change'}
    byid={r['id']:r for r in rows};n=0;maxerr=0.
    for old in json.loads(review.read_text()):
        for a,b in mapping.items():
            x=old.get(a);y=byid[old['id']].get(b)
            if x is None or y is None:continue
            rel=abs(x-y)/max(abs(x),abs(y),1e-12)
            assert rel<1e-7,(old['id'],a,x,y,rel)
            maxerr=max(maxerr,rel);n+=1
    return {'available':True,'compared_values':n,'max_scaled_difference':maxerr,'passed':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__);here=Path(__file__).resolve().parent
    parser.add_argument('--data',type=Path,default=here/'data');parser.add_argument('--output',type=Path,default=here/'site/interpretation');a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);summary,rows,histories,inputs=compute(a.data)
    check=compare_review(rows,here/'review/metrics.json')
    metadata=dict(schema_version=1,simulation_source_version=summary['source_version'],figure_script_sha256=digest(__file__),
        definitions={'chi_E':'Eperp / Eeddy; Eperp is kinetic energy of departure from y mean, Eeddy excludes horizontal mean.',
        'chi_q':'Zvariance / Za, where a=q-q0; a measure of transverse dependence of q, not all velocity components.',
        'mean_error':'epsilon_U = depth RMS of [Delta UE - Delta UE_2D] / depth RMS of Delta UE_2D at 1200 s. Each Delta subtracts its own initial profile.',
        'history_error':'The whole error(t) history is normalized by the matched 2D final 1200 s changed-profile RMS, not a vanishing early-time amplitude.',
        'Re_inj':'U2D_initial * injection_wavelength / nu; not Taylor-microscale Reynolds number.',
        'u3D':'Initial RMS transverse-dependent velocity, U2D_initial sqrt(eta). No-wave cases have no ratio to actual us=0.',
        'sampling':'All measured source CSV cadences are 1 s in this campaign; maxima are over saved times. Public geometry-history.csv selects every 20 s. mean-error-history.csv retains 1 s. Late averages cover 600–1200 s.',
        'residuals':'Residuals copied from published full-depth quadrature, normalized by max(initial moment, maximum excursion, maximum native integral).'},
        parameter_support={'u_star':'No imposed wind stress; not a varied control.','wave_on_us0_m_s':.036649447248308646,'U2D_initial_m_s':.004,'ell_m':.25,'k_m_inverse':6.1,'no_new_simulations':True},
        input_sha256=inputs,independent_review_comparison=check,runs=rows)
    (a.output/'metrics.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    with (a.output/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (a.output/'geometry-history.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['run_id','time_s','chi_E','Eperp_m2_s2','Eeddy_m2_s2','chi_q'])
        for rid,h in histories.items():
            for j,t in enumerate(h['geometry_time_s']):
                if not np.isclose(t % 20,0):continue
                i=int(np.flatnonzero(h['moment_time_s']==t)[0])
                w.writerow([rid,t,h['chi_E'][j],h['Eperp'][j],h['Eeddy'][j],h['chi_q'][i]])
    with (a.output/'mean-error-history.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['run_id','baseline_id','time_s','current_change_error_m_s','reference_changed_profile_rms_m_s','current_error_over_final_reference_change'])
        for rid,h in histories.items():
            for i,t in enumerate(h['profile_time_s']):
                w.writerow([rid,h['matched_2d'],t,h['current_error_mm_s'][i]/1000,h['reference_change_rms_m_s'][i],h['current_error_over_final_reference_change'][i]])
    provenance={k:metadata[k] for k in ('schema_version','simulation_source_version','figure_script_sha256','input_sha256','independent_review_comparison')}
    provenance['source_exports']={'geometry-history.csv':'Every20s selection of stored 1 s scalar geometry; no fields or native snapshots.', 'mean-error-history.csv':'Every1s depth-RMS comparison computed from derived mean profiles; no spatial arrays.'}
    (a.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    setup();main_figure(rows,a.output);history_figure(histories,a.output)
    (a.output/'figure-captions.txt').write_text('Parameter summary. All endpoint values are at 1200 s. Top panels use alpha=0.75, seed1; nonzero initial disturbance amplitude is plotted against the fixed surface Stokes speed. Exact 2D cases have zero 3D energy and zero paired-current error and are omitted from logarithmic axes. Reinj is varied through viscosity at fixed initial base eddy RMS, injection wavelength and wave profile. Fine-grid symbols also use half the maximum timestep; second-seed symbols are independent initial fields. Bottom panels separately change alpha, or remove waves while preserving the initial Eulerian background. Connecting lines guide the eye and do not define stability boundaries. No wind-stress axis was sampled.\n\nHistory comparison. Both cases have Reinj=300 and alpha=0.75. The current-error history uses the depth RMS mismatch between changes from each run’s own initial mean, divided by the matched 2D final changed-current RMS. That denominator is fixed throughout each panel. The two curves describe different quantities; neither a final kinetic fraction nor a single threshold diagnoses autonomous 2D mean dynamics.\n\nSimulation source '+summary['source_version']+'; full input hashes and numeric definitions are in metrics.json.\n')
    print(json.dumps({'runs':len(rows),'validation':check,'output':str(a.output)},indent=2))

if __name__=='__main__':main()
