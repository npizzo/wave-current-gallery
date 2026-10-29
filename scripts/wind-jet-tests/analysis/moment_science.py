"""Vorticity second moments with separate temporal and spatial budget tests.

Every quantity is a physical-volume average.  q is Eulerian y-vorticity;
advection is by the Lagrangian velocity.  No residual is fitted away.
"""
from __future__ import annotations
import numpy as np

MOMENTS = ('Zq', 'Zmean', 'Zvariance', 'Za', 'Zmean_anomaly')
FLUXES = dict(Zq='q', Zmean='mean', Zvariance='variance', Za='a', Zmean_anomaly='mean_anomaly')
SOURCES = dict(Zq=('Pq_T',), Zmean=('Pmean_exchange', 'Pmean_T'),
               Zvariance=('Pvariance_exchange', 'Pvariance_T'),
               Za=('Pa_T', 'Pa_background'),
               Zmean_anomaly=('Pmean_anomaly_exchange', 'Pmean_anomaly_T', 'Pmean_anomaly_background'))
COMPONENTS = ('stretch', 'tilt_wy_vx', 'tilt_minus_uy_vz')


def checked_times(time):
    t = np.asarray(time, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('A finite, strictly increasing time grid with at least two samples is required')
    return t


def cumulative_trapezoid(time, value):
    t = checked_times(time); y = np.asarray(value, dtype=float)
    if y.shape != t.shape or not np.isfinite(y).all():
        raise ValueError('Finite time-series values must match the time grid')
    return np.r_[0., np.cumsum(np.diff(t) * (y[:-1] + y[1:]) / 2)]


def integral_metrics(time, signed, absolute=None):
    """Absolute is mean(abs(local production)), not abs(mean(production))."""
    t = checked_times(time); s = np.asarray(signed, dtype=float)
    a = np.abs(s) if absolute is None else np.asarray(absolute, dtype=float)
    if a.shape != s.shape or not np.isfinite(a).all() or np.any(a < -1e-25):
        raise ValueError('Absolute production must be finite and nonnegative')
    scale = max(float(np.max(a)), float(np.max(np.abs(s))), 1e-30)
    if np.any(np.abs(s) > a + 1e-9*scale):
        raise ValueError('Absolute local production cannot be smaller than its signed mean')
    si = cumulative_trapezoid(t, s)[-1]; ai = cumulative_trapezoid(t, a)[-1]
    at = cumulative_trapezoid(t, np.abs(s))[-1]
    return dict(signed_integral=float(si), absolute_local_integral=float(ai) if absolute is not None else None,
                absolute_spatial_mean_integral=float(at),
                signed_over_absolute_local=float(si/ai) if absolute is not None and ai > 0 else None,
                cancellation_fraction=float(np.clip(1-abs(si)/ai, 0, 1)) if ai > 0 else None,
                cancellation_scope='space and time' if absolute is not None else 'time after spatial averaging')


def production_components(q, a, vy, wy, vx, uy, vz):
    """Pointwise scalar-vorticity production; y-independent 2D3C is a null."""
    q, a, vy, wy, vx, uy, vz = np.broadcast_arrays(q, a, vy, wy, vx, uy, vz)
    t = (q*vy, wy*vx, -uy*vz)
    result = {'T': sum(t)}
    for prefix, weight in (('Pq', q), ('Pa', a)):
        for name, term in zip(COMPONENTS, t): result[prefix+'_'+name] = weight*term
        result[prefix+'_T'] = weight*sum(t)
        result[prefix+'_T_abs'] = np.abs(result[prefix+'_T'])
    return result


def second_moments(q, q0, transverse_axis=1):
    """Pointwise y-mean partitions before x-z integration."""
    q = np.asarray(q, dtype=float); q0 = np.broadcast_to(q0, q.shape)
    mean = np.mean(q, axis=transverse_axis, keepdims=True)
    a = q-q0
    return dict(Zq=np.mean(q*q/2, axis=transverse_axis),
                Zmean=np.squeeze(mean*mean/2, axis=transverse_axis),
                Zvariance=np.mean((q-mean)**2/2, axis=transverse_axis),
                Za=np.mean(a*a/2, axis=transverse_axis),
                Zmean_anomaly=np.mean(a, axis=transverse_axis)**2/2)


def interval_closure(time, moment, native_rate, continuum_rate):
    """Interval and cumulative closure, without treating residual as dissipation."""
    t = checked_times(time); z, n, c = (np.asarray(a, dtype=float) for a in (moment, native_rate, continuum_rate))
    if any(a.shape != t.shape or not np.isfinite(a).all() for a in (z,n,c)):
        raise ValueError('Moment and rates must be finite and sampled at the same times')
    dt = np.diff(t); rate = np.diff(z)/dt
    ni = (n[:-1]+n[1:])/2; ci = (c[:-1]+c[1:])/2
    int_n = cumulative_trapezoid(t,n); int_c = cumulative_trapezoid(t,c)
    delta=z-z[0]; temporal=delta-int_n; spatial=int_n-int_c
    # Native-instantaneous minus continuum is a discrete product-rule/interpolation
    # residual; finite-interval native closure is a separate temporal test.
    return dict(time_s=t, delta=delta, native_integral=int_n, continuum_integral=int_c,
                temporal_residual=temporal, spatial_residual=spatial,
                total_residual=delta-int_c, instantaneous_spatial_residual=n-c,
                interval_time_s=(t[1:]+t[:-1])/2, interval_dt=dt,
                interval_state_rate=rate, interval_native_rate=ni,
                interval_continuum_rate=ci, interval_temporal_residual=rate-ni)


def normalize_residual(closure, initial_moment):
    """Report dimensional residuals first; denominator includes changes and rates."""
    temporal=float(closure['temporal_residual'][-1]);spatial=float(closure['spatial_residual'][-1])
    scale=max(abs(float(initial_moment)), float(np.max(np.abs(closure['delta']))),
              float(np.max(np.abs(closure['native_integral']))), 1e-30)
    return dict(final_temporal_residual=temporal, final_spatial_residual=spatial,
                max_abs_temporal_residual=float(np.max(np.abs(closure['temporal_residual']))),
                max_abs_spatial_residual=float(np.max(np.abs(closure['spatial_residual']))),
                normalization=scale,
                max_abs_temporal_residual_normalized=float(np.max(np.abs(closure['temporal_residual']))/scale),
                max_abs_spatial_residual_normalized=float(np.max(np.abs(closure['spatial_residual']))/scale))


def regional_profile(profile, mask):
    """Integrate stored cell means and account for flux through regional faces.

    Mask must be a contiguous set of complete cells. Flux uses exactly the
    stored first lower and last upper face; no assumption of zero basal flux.
    """
    dz=np.asarray(profile['dz_m'],float); mask=np.asarray(mask,bool)
    ix=np.flatnonzero(mask)
    if not len(ix) or mask.shape!=dz.shape or np.any(np.diff(ix)!=1) or np.any(dz<=0):
        raise ValueError('Regional integration requires contiguous positive-volume cells')
    depth=float(dz[mask].sum()); out={}
    for name,value in profile.items():
        if name in ('time_s','z_m','dz_m') or name.startswith('flux_'): continue
        v=np.asarray(value,float)
        if v.shape!=dz.shape or not np.isfinite(v).all():raise ValueError('Nonfinite or mismatched profile '+name)
        out[name]=float(np.sum(v[mask]*dz[mask])/depth)
    for name in MOMENTS:
        flux=FLUXES[name]
        lo=float(np.asarray(profile['flux_'+flux+'_lower'])[ix[0]])
        hi=float(np.asarray(profile['flux_'+flux+'_upper'])[ix[-1]])
        out[name+'_flux_divergence']=(hi-lo)/depth
        out[name+'_boundary_supply']=(lo-hi)/depth
        out[name+'_continuum_rate']=sum(out[k] for k in SOURCES[name])+sum(out[name+'_'+k] for k in ('viscous','forcing','pressure'))+(lo-hi)/depth
        out[name+'_spatial_residual']=out['d'+name+'_native']-out[name+'_continuum_rate']
    out['partition_residual']=out['Zq']-out['Zmean']-out['Zvariance']
    out['exchange_sum']=out['Pmean_exchange']+out['Pvariance_exchange']
    out['production_partition_residual']=out['Pq_T']-out['Pmean_T']-out['Pvariance_T']
    for family,prefix,extra in [('Zq','Pq',0.),('Za','Pa',out['Pa_background'])]:
        if family+'_inviscid' in out and prefix+'_advection' in out:
            out[family+'_differential_residual']=out[family+'_inviscid']-(out[prefix+'_advection']+out[prefix+'_T']+extra)
            out[family+'_transport_product_residual']=out[prefix+'_advection']+out[family+'_flux_divergence']
    out['q_variance_to_background_anomaly']=out['Zvariance']/out['Za'] if out['Za']>0 else None
    out['depth_m']=depth
    return out


def eulerian_profile(lagrangian, z, config):
    s=float(config['S'])
    return np.asarray(lagrangian)-s*np.exp(2*float(config['k'])*np.asarray(z))
