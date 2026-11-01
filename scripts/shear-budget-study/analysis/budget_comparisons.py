"""Separate finite-dimensional interventions and numerical sensitivity checks."""
from __future__ import annotations
import numpy as np

TIMES=(200,400,600,800,1000,1200)
MODES=('3d_control','3d_wind','2d_control','2d_wind')


def at_time(rows,time):
    return next((r for r in rows if abs(r['elapsed_s']-time)<1e-7),None)


def crossing(rows,threshold):
    return next((float(r['elapsed_s']) for r in rows if r['chi_3d']>=threshold),None)


def difference_in_differences(values):
    return (values['3d_wind']-values['3d_control'])-(values['2d_wind']-values['2d_control'])


def assess_interventions(groups,results,histories,jet_rows):
    byid={r['id']:r for r in results};out=[]
    for group in groups:
        ids={mode:group[mode] for mode in MODES};missing=[v for v in ids.values() if v not in histories]
        if missing:out.append(dict(id=group['id'],status='pending',missing_cases=missing));continue
        cfg={key:byid[name]['configuration'] for key,name in ids.items()}
        for key in ('h','lx','nx','nz','k','S','alpha','nu','start_time','stop_time','seed'):
            if len({c.get(key) for c in cfg.values()})!=1:raise ValueError('Intervention cases differ in '+key)
        if any(int(cfg[k]['ny'])!=1 for k in ('2d_control','2d_wind')) or any(int(cfg[k]['ny'])<=1 for k in ('3d_control','3d_wind')):raise ValueError('Intervention mode mismatch')
        if cfg['3d_control']['wind_ustar']!=cfg['2d_control']['wind_ustar'] or cfg['3d_wind']['wind_ustar']!=cfg['2d_wind']['wind_ustar']:raise ValueError('Intervention stress mismatch')
        records=[]
        for time in TIMES:
            h={key:at_time(histories[name],time) for key,name in ids.items()};j={key:at_time(jet_rows[name],time) for key,name in ids.items()}
            if any(v is None for v in [*h.values(),*j.values()]):continue
            p={key:r['jet_prominence_fraction'] for key,r in h.items()};energy={key:r['band_energy_change_m2_s2'] for key,r in j.items()}
            work={term:difference_in_differences({key:r[f'band_work_{term}_m2_s2'] for key,r in j.items()}) for term in ('A0','C','W','D','F','P')}
            delta=difference_in_differences(energy)
            records.append(dict(elapsed_s=time,retained_prominence=p,wind_effect_3d=p['3d_wind']-p['3d_control'],wind_effect_2d=p['2d_wind']-p['2d_control'],
                additional_3d_prominence_effect=difference_in_differences(p),band_energy_change_difference_m2_s2=delta,
                signed_band_work_differences_m2_s2=work,band_energy_difference_closure_m2_s2=delta-sum(work.values())))
        out.append(dict(id=group['id'],status='complete',cases=ids,windows=records,
            interpretation='Negative prominence difference means added wind reduces retained jet prominence more when 3D motions are allowed. Signed work differences decompose four realized trajectories; they are not independent causal effects of turning off individual tendencies. Late extrema rematching is qualified by the original profile histories.'))
    return out


def assess_resolution(pairs,results,histories,jet_rows,budgets):
    byid={r['id']:r for r in results};out=[]
    for pair in pairs:
        a,b=pair['coarse'],pair['fine'];missing=[v for v in (a,b) if v not in histories]
        if missing:out.append(dict(id=pair['id'],status='pending',missing_cases=missing));continue
        cfg_a,cfg_b=byid[a]['configuration'],byid[b]['configuration']
        for key in ('h','lx','ly','k','S','alpha','nu','wind_ustar','seed','start_time','stop_time'):
            if cfg_a.get(key)!=cfg_b.get(key):raise ValueError('Numerical pair differs in physical parameter '+key)
        records=[]
        for time in TIMES:
            ha,hb=at_time(histories[a],time),at_time(histories[b],time);ja,jb=at_time(jet_rows[a],time),at_time(jet_rows[b],time)
            if any(v is None for v in (ha,hb,ja,jb)):continue
            record=dict(elapsed_s=time)
            for key in ('jet_prominence_fraction','chi_3d','roll_rms_m_s'):
                record[key]=dict(coarse=ha[key],fine=hb[key],fine_minus_coarse=hb[key]-ha[key])
            for term in ('A0','C','W','D','F'):
                key=f'band_work_{term}_m2_s2';record[key]=dict(coarse=ja[key],fine=jb[key],fine_minus_coarse=jb[key]-ja[key])
            records.append(record)
        out.append(dict(id=pair['id'],status='complete',kind=pair.get('kind','spatial'),coarse=a,fine=b,
            grid_coarse=[cfg_a[k] for k in ('nx','ny','nz')],grid_fine=[cfg_b[k] for k in ('nx','ny','nz')],
            dt_coarse_s=cfg_a['max_dt'],dt_fine_s=cfg_b['max_dt'],
            onset_1percent_s=dict(coarse=crossing(histories[a],.01),fine=crossing(histories[b],.01)),
            onset_10percent_s=dict(coarse=crossing(histories[a],.1),fine=crossing(histories[b],.1)),
            windows=records,
            coarse_eddy_vorticity_closure=budgets[a]['vorticity_moments']['interior']['eddy']['relative_to_change_and_gross'],
            fine_eddy_vorticity_closure=budgets[b]['vorticity_moments']['interior']['eddy']['relative_to_change_and_gross'],
            interpretation='Targeted numerical sensitivity, not an asymptotic convergence proof. Three-dimensional energy thresholds are descriptive onset markers. Refined-grid parent interpolation and subsequent nonlinear divergence remain part of this comparison.'))
    return out
