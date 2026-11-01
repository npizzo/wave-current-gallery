"""Signed native tendency budgets; all integrals follow the realized trajectory.

These diagnose a direct contribution, not the trajectory that would result
from removing it. The projected-2D integrations are the separate intervention.
"""
import numpy as np

TERMS=('A0','C','W','D','F')
FIELD_NAMES={'A0':'Qdot_advection_mean','C':'Qdot_3d_native','W':'Qdot_wave','D':'Qdot_viscous','F':'Qdot_forcing'}


def cumulative(values,time):
    values=np.asarray(values,float);time=np.asarray(time,float)
    if len(values)!=len(time) or len(time)<2 or np.any(np.diff(time)<=0):raise ValueError('Invalid budget timeline')
    dt=np.diff(time).reshape((-1,)+(1,)*(values.ndim-1))
    return np.concatenate((np.zeros_like(values[:1]),np.cumsum(.5*(values[1:]+values[:-1])*dt,axis=0)))


def rms(a):return float(np.sqrt(np.mean(np.asarray(a)**2)))


def identity_error(fields):
    total=sum(fields[name] for name in TERMS)
    absolute=float(np.max(abs(fields['total']-total)))
    scale=max(float(np.max(abs(fields['total']))),max(float(np.max(abs(fields[n]))) for n in TERMS),1e-16)
    return dict(absolute_s_minus2=absolute,relative=absolute/scale)


def field_closure(q,terms,time):
    """Integral identity, avoiding a fitted derivative of the sampled Q."""
    q=np.asarray(q);total=sum(terms.values());change=q-q[0];integral=cumulative(total,time);error=change-integral
    gross=cumulative(sum(abs(x) for x in terms.values()),time)
    scale=max(rms(change),rms(gross),1e-12)
    change_scale=max(rms(change),1e-12)
    return dict(relative_to_change_and_gross=rms(error)/scale,relative_to_change=rms(error)/change_scale,
                rms_error_s_inverse=rms(error),max_error_s_inverse=float(np.max(abs(error))),
                final_error_s_inverse=rms(error[-1]),max_sample_interval_s=float(max(np.diff(time))))


def moment_budget(q,terms,time,initial_eddy=None):
    """Area means on a uniform x-z region. Boundary transport stays in terms."""
    q=np.asarray(q,float)
    if q.ndim!=3:raise ValueError('Q must have time,x,z dimensions')
    profile=q.mean(axis=1,keepdims=True);eddy=q-profile
    zeros=np.zeros(q.shape[0]);energy={};power={};integrals={};closure={}
    for key,state in [('Q',q),('eddy',eddy),('profile',np.broadcast_to(profile,q.shape))]:
        energy[key]=.5*np.mean(state**2,axis=(1,2));power[key]={name:np.mean(state*values,axis=(1,2)) for name,values in terms.items()}
        power[key]['total']=sum(power[key].values());integrals[key]={name:cumulative(values,time) for name,values in power[key].items()}
        delta=energy[key]-energy[key][0];error=delta-integrals[key]['total']
        gross=cumulative(sum(abs(values) for name,values in power[key].items() if name!='total'),time)
        scale=max(float(np.max(abs(delta))),float(np.max(gross)),1e-16)
        closure[key]=dict(max_abs_error_s_minus2=float(max(abs(error))),relative_to_change_and_gross=float(max(abs(error))/scale),
                         final_error_s_minus2=float(error[-1]),relative_to_change=float(max(abs(error))/max(float(max(abs(delta))),1e-16)))
    template=eddy[0] if initial_eddy is None else np.asarray(initial_eddy)
    norm=float(np.mean(template**2));projection={}
    if norm>1e-20:
        projection['actual']=np.mean(eddy*template[None,:,:],axis=(1,2))/norm
        for name,values in terms.items():projection[name]=cumulative(np.mean(values*template[None,:,:],axis=(1,2))/norm,time)
    return dict(energy=energy,power=power,integrals=integrals,closure=closure,initial_template_projection=projection)


def mode_budget(amplitudes,tendencies,time,norm):
    """Cosine jet modes: exact projected amplitude and quadratic-work budget."""
    a=np.asarray(amplitudes);norm=np.asarray(norm);delta=a-a[0]
    cumulative_amplitude={name:cumulative(values,time) for name,values in tendencies.items()}
    amplitude_error=delta-sum(cumulative_amplitude.values())
    energy=.5*a*a*norm[None,:];powers={name:a*values*norm[None,:] for name,values in tendencies.items()}
    work={name:cumulative(values,time) for name,values in powers.items()};energy_error=energy-energy[0]-sum(work.values())
    gross_a=cumulative(sum(abs(values) for values in tendencies.values()),time)
    gross_e=cumulative(sum(abs(values) for values in powers.values()),time)
    return dict(amplitude=a,energy=energy,powers=powers,work=work,cumulative_amplitude=cumulative_amplitude,
                amplitude_error=amplitude_error,energy_error=energy_error,
                amplitude_relative_error=np.max(abs(amplitude_error),axis=0)/np.maximum(np.maximum(np.max(abs(delta),axis=0),np.max(gross_a,axis=0)),1e-12),
                energy_relative_error=np.max(abs(energy_error),axis=0)/np.maximum(np.maximum(np.max(abs(energy-energy[0]),axis=0),np.max(gross_e,axis=0)),1e-16))
