"""Pure, explicitly normalized diagnostics for the finite-time regime assays."""
from __future__ import annotations
import numpy as np


def finite(values, name='array'):
    a=np.asarray(values,dtype=float)
    if not np.all(np.isfinite(a)):raise ValueError(f'Nonfinite {name}')
    return a


def ratio(a,b):
    return float(a/b) if np.isfinite(a) and np.isfinite(b) and b>0 else None


def weighted_rms(a,dz):
    a=finite(a);w=finite(dz,'cell widths')
    if a.ndim!=2 or a.shape[1]!=len(w) or np.any(w<=0):raise ValueError('Expected x-z field and positive z widths')
    return float(np.sqrt(np.sum(a*a*w[None,:])/(a.shape[0]*w.sum())))


def weighted_dot(a,b,dz):
    a,b=finite(a),finite(b)
    if a.shape!=b.shape:raise ValueError('Field shapes differ')
    w=finite(dz)
    if a.ndim!=2 or a.shape[1]!=len(w) or np.any(w<=0):raise ValueError('Invalid quadrature')
    return float(np.sum(a*b*w[None,:])/(a.shape[0]*w.sum()))


def gaussian_matrix(coordinates,widths,sigma,truncate=4.):
    """Nonperiodic, row-normalized quadrature; exposes actual kernel width."""
    x,w=finite(coordinates),finite(widths)
    if x.ndim!=1 or x.shape!=w.shape or np.any(np.diff(x)<=0) or np.any(w<=0) or sigma<0:raise ValueError('Invalid filter grid')
    if sigma==0:return np.eye(len(x)),np.zeros(len(x))
    distance=x[None,:]-x[:,None]
    kernel=np.exp(-.5*(distance/sigma)**2)*w[None,:]
    kernel[np.abs(distance)>truncate*sigma]=0.
    kernel/=kernel.sum(axis=1)[:,None]
    centroid=np.sum(kernel*distance,axis=1)
    actual=np.sqrt(np.sum(kernel*(distance-centroid[:,None])**2,axis=1))
    return kernel,actual


def filter_field(a,kx,kz):
    a=finite(a)
    if a.shape!=(kx.shape[1],kz.shape[1]):raise ValueError('Filter dimensions differ')
    return kx@a@kz.T


def scalar_metrics(fields,dz,tau=None):
    """Norms after signed fields are formed/filtered, never norms of products."""
    names=('Q','A','Rbar','Qdot_viscous','Qdot_forcing','Qt','scalar_observed','scalar_unclosed')
    result={f'{n}_rms':weighted_rms(fields[n],dz) for n in names if n in fields}
    if 'Qt' in fields:
        observed=fields['Qt']+fields['A']
        rhs=fields['Rbar']+fields.get('Qdot_viscous',0.)+fields.get('Qdot_forcing',0.)
        result['scalar_observed_rms']=weighted_rms(observed,dz)
        result['scalar_unclosed_rms']=weighted_rms(observed-rhs,dz)
        evolution=result['Qt_rms']+result['A_rms']
        result['rbar_to_scalar_evolution']=ratio(result['Rbar_rms'],evolution)
        result['scalar_observed_to_evolution']=ratio(result['scalar_observed_rms'],evolution)
        result['unclosed_to_scalar_budget']=ratio(result['scalar_unclosed_rms'],evolution+weighted_rms(rhs,dz))
        if tau is not None:result['tau_scalar_observed_to_q']=ratio(tau*result['scalar_observed_rms'],result['Q_rms'])
    result['rbar_to_advection']=ratio(result['Rbar_rms'],result['A_rms'])
    return result


def feedback_metrics(fields,dz,q0=None,qt=None):
    """Counterfactual native mean equation. Distinct from scalar Rbar."""
    needed=('Qdot_advection_full','Qdot_advection_mean','Qdot_wave','Qdot_viscous','Qdot_forcing','Qdot_total')
    result={f'{name}_rms':weighted_rms(fields[name],dz) for name in needed}
    extra=fields['Qdot_advection_full']-fields['Qdot_advection_mean']
    meanrhs=fields['Qdot_advection_mean']+fields['Qdot_wave']+fields['Qdot_viscous']+fields['Qdot_forcing']
    result['native_3d_correction_rms']=weighted_rms(extra,dz)
    result['native_mean_rhs_rms']=weighted_rms(meanrhs,dz)
    result['maintained_diffusive_rms']=weighted_rms(fields['Qdot_viscous']+fields['Qdot_forcing'],dz)
    scale=result['Qdot_advection_mean_rms']+result['Qdot_wave_rms']+result['maintained_diffusive_rms']
    result['raw_component_norm_sum']=sum(result[f'{key}_rms'] for key in needed[1:5])
    result['native_3d_to_mean_terms']=ratio(result['native_3d_correction_rms'],scale)
    result['native_3d_to_mean_rhs']=ratio(result['native_3d_correction_rms'],result['native_mean_rhs_rms'])
    result['native_identity_residual_rms']=weighted_rms(fields['Qdot_total']-meanrhs-extra,dz)
    if qt is not None:
        result['Qt_fd_rms']=weighted_rms(qt,dz)
        result['native_temporal_residual_rms']=weighted_rms(qt-fields['Qdot_total'],dz)
        result['native_temporal_relative_residual']=ratio(result['native_temporal_residual_rms'],result['Qt_fd_rms']+result['Qdot_total_rms'])
    if q0 is not None:
        qa=fields['Q']-q0
        result['q_anomaly_rms']=weighted_rms(qa,dz)
        result['native_3d_q_anomaly_supply']=weighted_dot(qa,extra,dz)
        result['native_mean_q_anomaly_supply']=weighted_dot(qa,meanrhs,dz)
        result['native_3d_jet_q_anomaly_supply']=weighted_dot(qa.mean(axis=0,keepdims=True),extra.mean(axis=0,keepdims=True),dz)
        result['native_mean_jet_q_anomaly_supply']=weighted_dot(qa.mean(axis=0,keepdims=True),meanrhs.mean(axis=0,keepdims=True),dz)
    return result


def paired_fields(fields,baseline,dz,initial,eddy_rms):
    """Same-initial-state trajectory and profile errors; no phase alignment."""
    out={}
    for name in ('Uc','Wc','Q'):
        a,b,a0=fields[name],baseline[name],initial[name]
        error=weighted_rms(a-b,dz);ar=weighted_rms(a-a0,dz);br=weighted_rms(b-a0,dz)
        ac=a-np.average(a,weights=np.broadcast_to(dz,a.shape));bc=b-np.average(b,weights=np.broadcast_to(dz,b.shape))
        out[name+'_error_rms']=error
        out[name+'_error_to_changes']=ratio(error,ar+br)
        out[name+'_correlation']=ratio(weighted_dot(ac,bc,dz),weighted_rms(ac,dz)*weighted_rms(bc,dz))
        ae=a-a.mean(axis=0,keepdims=True);be=b-b.mean(axis=0,keepdims=True)
        out[name+'_xz_eddy_correlation']=ratio(weighted_dot(ae,be,dz),weighted_rms(ae,dz)*weighted_rms(be,dz))
    velocity_error=np.hypot(out['Uc_error_rms'],out['Wc_error_rms'])
    out['inplane_error_rms']=float(velocity_error)
    out['inplane_error_to_initial_eddy']=ratio(velocity_error,eddy_rms)
    a=fields['Uc'].mean(axis=0)-initial['Uc'].mean(axis=0)
    b=baseline['Uc'].mean(axis=0)-initial['Uc'].mean(axis=0)
    out['jet_profile_error_rms']=weighted_rms((a-b)[None,:],dz)
    out['jet_profile_error_to_changes']=ratio(out['jet_profile_error_rms'],weighted_rms(a[None,:],dz)+weighted_rms(b[None,:],dz))
    out['jet_anomaly_rms']=weighted_rms(a[None,:],dz)
    out['baseline_jet_anomaly_rms']=weighted_rms(b[None,:],dz)
    return out


def centered_time_derivative(times,values,index,stride=1):
    """Three-point nonuniform centered derivative, interior outputs only."""
    if index<stride or index+stride>=len(times):return None
    a,b,c=index-stride,index,index+stride
    h0,h1=times[b]-times[a],times[c]-times[b]
    if h0<=0 or h1<=0:raise ValueError('Times must increase')
    return (-h1/(h0*(h0+h1)))*values[a]+((h1-h0)/(h0*h1))*values[b]+(h0/(h1*(h0+h1)))*values[c]


def x_mode_supply(u,w,fu,fw,dz,lx):
    """Exact Parseval supplies into native y-mean KE, indexed by x mode.

    3D-to-mean exchange can make the total nonzero. Negative cumulative supply
    is therefore not automatically a conservative inverse-cascade flux.
    """
    u,w,fu,fw=map(finite,(u,w,fu,fw));dz=finite(dz)
    if u.shape!=fu.shape or w.shape!=fw.shape or u.shape[1]!=len(dz) or w.shape!=(u.shape[0],len(dz)+1):raise ValueError('Native U/W dimensions differ')
    nz=len(dz);weights_w=np.r_[dz[0]/2,(dz[:-1]+dz[1:])/2,dz[-1]/2]
    n=u.shape[0];uh,wh,fuh,fwh=(np.fft.rfft(a,axis=0)/n for a in (u,w,fu,fw))
    multiplicity=np.full(len(uh),2.);multiplicity[0]=1.
    if n%2==0:multiplicity[-1]=1.
    supply=multiplicity*(np.sum((uh.conj()*fuh).real*dz,axis=1)+np.sum((wh.conj()*fwh).real*weights_w,axis=1))/dz.sum()
    return {'mode':np.arange(len(supply)),'kx_rad_m':2*np.pi*np.arange(len(supply))/lx,'supply':supply,'cumulative_supply':np.cumsum(supply),'sum_supply':float(supply.sum())}


def transverse_growth(times,energy,eddy,initial_ratio,max_fraction=.01):
    t,e,b=map(finite,(times,energy,eddy))
    if t.shape!=e.shape or t.shape!=b.shape or np.any(np.diff(t)<=0):raise ValueError('Invalid transverse series')
    if not 0<initial_ratio<max_fraction:return {'available':False,'reason':'not_a_small_initial_perturbation'}
    keep=[]
    for i in range(len(t)):
        if b[i]<=0 or e[i]<0:raise ValueError('Invalid energies')
        if e[i]/b[i]>=max_fraction:break
        if e[i]>0:keep.append(i)
    if len(keep)<5:return {'available':False,'reason':'fewer_than_five_small_perturbation_outputs'}
    keep=np.array(keep);x=t[keep]-t[keep[0]];y=.5*np.log(e[keep]);design=np.c_[np.ones(len(x)),x]
    fit=np.linalg.lstsq(design,y,rcond=None)[0]
    return {'available':True,'amplitude_growth_rate_s_inv':float(fit[1]),'start_s':float(t[keep[0]]),'end_s':float(t[keep[-1]]),'samples':len(keep),'log_amplitude_fit_rms':float(np.sqrt(np.mean((y-design@fit)**2))),'max_energy_fraction':max_fraction,'scope':'Finite-time growth on the evolving nonlinear base, not an eigenvalue or asymptotic stability proof.'}


def twin_collapse(t1,e1,eta1,t2,e2,eta2):
    t1,t2,e1,e2=map(finite,(t1,t2,e1,e2))
    if t1.shape!=t2.shape or not np.allclose(t1,t2,rtol=0,atol=1e-8):raise ValueError('Twin times differ')
    if eta1<=0 or eta2<=0 or np.any(e1<0) or np.any(e2<0):raise ValueError('Invalid twin amplitude')
    a,b=np.sqrt(e1/eta1),np.sqrt(e2/eta2)
    return {'time_s':t1,'scaled_amplitude_a':a,'scaled_amplitude_b':b,'relative_difference':np.divide(abs(a-b),a+b,out=np.full_like(a,np.nan),where=(a+b)>0)}


def qualified_twin_growth(times,energy_small,eta_small,eddy_small,energy_large,eta_large,eddy_large,
                          collapse_tolerance=.05,max_energy_fraction=.01,max_log_fit_rms=.2):
    """Finite prefix where both amplitudes stay small and scale alike.

    The thresholds qualify a numerical linear-response fit. They do not define
    a physical regime boundary, an asymptotic exponent, or statistical evidence.
    """
    t,e1,b1,e2,b2=map(finite,(times,energy_small,eddy_small,energy_large,eddy_large))
    collapse=twin_collapse(t,e1,eta_small,t,e2,eta_large)
    config={'collapse_tolerance':collapse_tolerance,'maximum_energy_fraction':max_energy_fraction,'maximum_log_amplitude_fit_rms':max_log_fit_rms,'minimum_samples':5}
    indices=[];stop_reason='end_of_record'
    for i in range(len(t)):
        if b1[i]<=0 or b2[i]<=0 or e1[i]<=0 or e2[i]<=0:stop_reason='unresolved_or_nonpositive_energy';break
        if e1[i]/b1[i]>=max_energy_fraction or e2[i]/b2[i]>=max_energy_fraction:stop_reason='perturbation_no_longer_small';break
        if not np.isfinite(collapse['relative_difference'][i]) or collapse['relative_difference'][i]>collapse_tolerance:stop_reason='twin_amplitudes_do_not_collapse';break
        indices.append(i)
    out={'available':False,'qualification':config,'stop_reason':stop_reason,'qualified_samples':len(indices)}
    if len(indices)<5:return out|{'reason':'fewer_than_five_contiguous_qualified_samples'}
    n=len(indices);fit=transverse_growth(t[:n],e2[:n],b2[:n],eta_large,max_fraction=max_energy_fraction)
    out.update(fit);out['max_twin_relative_amplitude_error']=float(np.max(collapse['relative_difference'][:n]));out['qualified_samples']=n
    if fit.get('log_amplitude_fit_rms',float('inf'))>max_log_fit_rms:out.update(available=False,reason='poor_single_growth_rate_fit')
    return out
