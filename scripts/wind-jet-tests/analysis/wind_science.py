"""Predeclared diagnostics for wind-forced closure and weak-band response tests.

No classification here equates a small 3-D correction with jet existence.
The native mean-vorticity correction is a measured omitted tendency, not a
rigorous bound on nonlinear forecast error.
"""
from __future__ import annotations
import numpy as np


DEFAULT_CRITERIA = {
    "weak_correction_threshold": .1,
    "forecast_error_threshold": .1,
    "sensitivity_thresholds": [.05, .1, .2],
    "current_activity_fraction_of_initial_eddy_rms": .01,
    "vorticity_activity_fraction_of_initial_eddy_over_injection_length": .001,
    "trim_cells": 2,
    "reference_vertical_cells": 96,
    "active_stokes_efolding_depths": 3.,
    "response_growth_factor": 1.2,
    "response_budget_relative_tolerance": .05,
    "linearity_relative_tolerance": .1,
    "refinement_metric_relative_tolerance": .1,
    "forecast_duration_s": 200.,
    "response_fit_windows_s": [[0., 50.], [50., 200.]],
}


def resolve_criteria(values=None):
    v = dict(values or {})
    aliases = {"closure_threshold": ("weak_correction_threshold", "forecast_error_threshold"),
               "mean_current_activity_floor_fraction_initial_eddy": ("current_activity_fraction_of_initial_eddy_rms",),
               "active_depth_stokes_e_folds": ("active_stokes_efolding_depths",),
               "probe_amplitude_linearity_tolerance": ("linearity_relative_tolerance",),
               "numerical_refinement_tolerance": ("refinement_metric_relative_tolerance",)}
    for old, new in aliases.items():
        if old in v:
            for key in new:
                if key in v and v[key] != v[old]: raise ValueError("Conflicting criterion aliases")
                v[key] = v[old]
    return DEFAULT_CRITERIA | v


def finite(x, name="data"):
    a = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"Nonfinite {name}")
    return a


def rms_xz(a, dz):
    a, dz = finite(a), finite(dz)
    if a.ndim != 2 or a.shape[-1] != len(dz) or np.any(dz <= 0):
        raise ValueError("Expected x-z array and positive depth weights")
    return float(np.sqrt(np.sum(a*a*dz[None, :]) / (a.shape[0]*dz.sum())))


def profile_rms(a, dz):
    return rms_xz(np.asarray(a)[None, :], dz)


def mask_regions(z, k, trim=2, efold=3., reference_cells=96):
    z = finite(z)
    if len(z) <= 2*trim or k <= 0:
        raise ValueError("Invalid depth range")
    # Production grids (96/192) exclude the identical physical boundary width.
    # Tiny manufactured validation grids still exclude at least two cells.
    effective_trim=max(trim,int(np.ceil(trim*len(z)/reference_cells)))
    interior=np.ones(len(z),dtype=bool)
    if effective_trim:
        interior[:effective_trim]=False;interior[-effective_trim:]=False
    return {"full": np.ones(len(z), dtype=bool), "interior": interior,
            "active_interior": interior & (z >= -efold/(2*k))}


def common_time_indices(parent_t, child_t, start, stop):
    a, b = finite(parent_t), finite(child_t)
    if np.any(np.diff(a) <= 0) or np.any(np.diff(b) <= 0):
        raise ValueError("Nonmonotonic times")
    ib = np.flatnonzero((b >= start-1e-7) & (b <= stop+1e-7))
    ia = np.searchsorted(a, b[ib])
    # Roundoff may place a matching time immediately before searchsorted.
    ia = np.array([min((i for i in (j-1, j) if 0 <= i < len(a)),
                       key=lambda i: abs(a[i]-t)) for j, t in zip(ia, b[ib])])
    if len(ia) < 3 or not np.allclose(a[ia], b[ib], atol=1e-7, rtol=0):
        raise ValueError("Forecast and parent diagnostic times do not match")
    if not np.isclose(b[ib[0]], start, atol=1e-7, rtol=0) or not np.isclose(b[ib[-1]], stop, atol=1e-7, rtol=0):
        raise ValueError("Forecast window is incomplete")
    return ia, ib


def closure_window(parent, forecast, config, criteria=None):
    """Arrays use time,x,z; Q includes mean shear, but normalization excludes it."""
    c = resolve_criteria(criteria)
    start, stop = float(forecast["time"][0]), float(forecast["time"][-1])
    if not np.isclose(stop-start, c["forecast_duration_s"], atol=1e-7):
        raise ValueError("Forecast duration differs from the preregistered window")
    ia, ib = common_time_indices(parent["time"], forecast["time"], start, stop)
    t = np.asarray(parent["time"])[ia]
    z, dz = finite(parent["z"]), finite(parent["dz"])
    if not np.array_equal(z, forecast["z"]):
        raise ValueError("Forecast must use the parent's x-z grid")
    regions = mask_regions(z, float(config["k"]), c["trim_cells"], c["active_stokes_efolding_depths"], c["reference_vertical_cells"])
    initial_eddy = float(config["eddy_rms"])
    qfloor = c["vorticity_activity_fraction_of_initial_eddy_over_injection_length"]*initial_eddy/float(config["injection_wavelength"])
    ufloor = c["current_activity_fraction_of_initial_eddy_rms"]*initial_eddy
    output, history = [], []
    for region, mask in regions.items():
        if mask.sum() < 4:
            output.append({"region": region, "assessment": "INCONCLUSIVE", "reason": "Insufficient depth cells"})
            continue
        widths = dz[mask]
        q = finite(parent["Q"])[ia][:, :, mask]
        qf = finite(forecast["Q"])[ib][:, :, mask]
        u = finite(parent["Uc"])[ia][:, :, mask]
        uf = finite(forecast["Uc"])[ib][:, :, mask]
        omitted = finite(parent["C"])[ia][:, :, mask]
        qref = rms_xz(q[0]-q[0].mean(axis=0, keepdims=True), widths)
        crms = np.array([rms_xz(a, widths) for a in omitted])
        ct, cnorm = t, crms
        if "native_norms" in parent:
            native = parent["native_norms"]
            suffix = "stokes" if region == "active_interior" else region
            nt = finite(native["time"])
            select = (nt >= start-1e-7) & (nt <= stop+1e-7)
            ct, cnorm = nt[select], finite(native["C_native_rms_"+suffix])[select]
            if len(ct) < 3 or not np.isclose(ct[0], start, atol=1e-7) or not np.isclose(ct[-1], stop, atol=1e-7):
                raise ValueError("Online correction norms do not span forecast")
            online_qref = float(finite(native["q_eddy_rms_"+suffix])[select][0])
            if not np.isclose(qref, online_qref, rtol=1e-7, atol=1e-12):
                raise ValueError("Online and saved-map reference vorticity disagree")
            if not np.allclose(np.interp(t, ct, cnorm), crms, rtol=1e-6, atol=1e-12):
                raise ValueError("Online and saved-map native correction norms disagree")
        qerror = np.array([rms_xz(a-b, widths) for a, b in zip(q, qf)])
        parent_change = u.mean(axis=1)-u[0].mean(axis=0)
        forecast_change = uf.mean(axis=1)-uf[0].mean(axis=0)
        uerror = np.array([profile_rms(a-b, widths) for a, b in zip(parent_change, forecast_change)])
        change = np.array([profile_rms(a, widths) for a in parent_change])
        uref = float(change.max())
        initial_q_error = float(qerror[0])
        initial_u_error = rms_xz(u[0]-uf[0], widths)
        initial_match = initial_q_error <= max(qref*1e-8, 1e-12) and initial_u_error <= max(initial_eddy*1e-8, 1e-12)
        active = qref >= qfloor and uref >= ufloor and initial_match
        integral = float(np.trapezoid(cnorm, ct))
        coarse_integral = float(np.trapezoid(crms, t))
        eps = integral/qref if qref > 0 else None
        eq = float(qerror.max()/qref) if qref > 0 else None
        eu = float(uerror.max()/uref) if uref > 0 else None
        sensitivity = {}
        for threshold in c["sensitivity_thresholds"]:
            if not active:
                outcome = "INCONCLUSIVE"
            elif eps > threshold:
                outcome = "OUTSIDE_SMALL_CORRECTION_REGIME"
            elif eq <= threshold and eu <= threshold:
                outcome = "PASS"
            else:
                outcome = "APPARENT_FAILURE_REQUIRES_REFINEMENT"
            sensitivity[str(threshold)] = outcome
        if not active:
            outcome, reason = "INCONCLUSIVE", "Insufficient activity or unmatched initial mean"
        elif eps > c["weak_correction_threshold"]:
            outcome, reason = "INCONCLUSIVE", "Outside the specified small-correction regime"
        elif eq <= c["forecast_error_threshold"] and eu <= c["forecast_error_threshold"]:
            outcome, reason = "PASS", "Reduced mean dynamics supported over this 200 s window"
        else:
            outcome, reason = "INCONCLUSIVE", "Apparent closure failure requires matching refinement checks"
        output.append(dict(region=region, start_s=start, stop_s=stop,
            z_min_m=float(z[mask].min()), z_max_m=float(z[mask].max()),
            correction_integral_s_inverse=integral, q_reference_s_inverse=qref,
            correction_sampling_s=float(np.max(np.diff(ct))), forecast_sampling_s=float(np.max(np.diff(t))),
            correction_coarse_sampling_relative_difference=abs(integral-coarse_integral)/max(integral,1e-30),
            current_reference_m_s=uref, q_activity_floor_s_inverse=qfloor,
            current_activity_floor_m_s=ufloor, initial_q_error_s_inverse=initial_q_error,
            initial_u_error_m_s=initial_u_error, initial_mean_matches=initial_match,
            active=active, epsilon_C=eps, max_q_forecast_error=eq,
            max_current_forecast_error=eu, assessment=outcome, reason=reason,
            sensitivity=sensitivity, apparent_failure=bool(active and eps <= c["weak_correction_threshold"]
                and (eq > c["forecast_error_threshold"] or eu > c["forecast_error_threshold"]))))
        native_accum = np.concatenate(([0.], np.cumsum(np.diff(ct)*(cnorm[:-1]+cnorm[1:])/2)))
        accum = np.interp(t, ct, native_accum)
        for j, time in enumerate(t):
            history.append(dict(region=region, start_s=start, time_s=float(time),
                C_rms_s_inverse2=float(crms[j]), correction_integral_s_inverse=float(accum[j]),
                epsilon_C=float(accum[j]/qref) if qref else None,
                q_error_s_inverse=float(qerror[j]), q_error_normalized=float(qerror[j]/qref) if qref else None,
                current_error_m_s=float(uerror[j]), current_error_normalized=float(uerror[j]/uref) if uref else None))
    required = [r for r in output if r["region"] in ("interior", "active_interior")]
    passed = len(required) == 2 and all(r["assessment"] == "PASS" for r in required)
    return dict(start_s=start, stop_s=stop, assessment="PASS" if passed else "INCONCLUSIVE",
                reason="Both prescribed depth regions support the reduced forecast" if passed else
                "See activity, correction-size and refinement qualifications for each region", regions=output), history


def modal_response(plus, minus, mode, amplitude, h, criteria=None):
    """Finite-time response of a weak imposed band, not a claimed eigenvalue."""
    c = resolve_criteria(criteria)
    t, z, dz = finite(plus["time"]), finite(plus["z"]), finite(plus["dz"])
    if not np.array_equal(t, minus["time"]) or not np.array_equal(z, minus["z"]):
        raise ValueError("Paired mode times/depths differ")
    if amplitude <= 0 or mode < 1:
        raise ValueError("Positive amplitude and mode required")
    phi = np.cos(mode*np.pi*(z+h)/h)
    norm = float(np.sum(phi*phi*dz))
    def projection(name):
        response = (finite(plus[name])-finite(minus[name]))/(2*amplitude)
        return np.sum(response*phi[None, :]*dz[None, :], axis=1)/norm
    a = projection("UE")
    terms = {name: projection(name) for name in ("adv_full", "adv_ky0", "adv_3d", "closure", "forcing", "wave", "pressure_mean")}
    total = sum(terms[name] for name in ("adv_full", "closure", "forcing", "wave", "pressure_mean"))
    integral = float(np.trapezoid(total, t))
    residual = float(a[-1]-a[0]-integral)
    scale = max(abs(float(a[-1]-a[0])), float(np.trapezoid(np.abs(total), t)), 1e-8)
    residual_fraction = abs(residual)/scale
    supplies = {name: float(np.trapezoid(a*values, t)) for name, values in terms.items()}
    gain = float(abs(a[-1]/a[0])) if abs(a[0]) > 1e-12 else None
    initial_match = bool(abs(a[0]-1.) < 1e-7)
    budget_ok = residual_fraction <= c["response_budget_relative_tolerance"]
    eddy_reinforces = supplies["adv_full"] > 0
    candidate = bool(gain is not None and gain >= c["response_growth_factor"] and eddy_reinforces and budget_ok and initial_match)
    outcome = "POSITIVE_RESPONSE_CANDIDATE" if candidate else "NO_POSITIVE_RESPONSE_DEMONSTRATED"
    if not initial_match or not budget_ok:
        outcome = "INCONCLUSIVE"
    # Fit only a sign-preserving positive response; an oscillatory response is
    # not silently treated as exponential instability.
    fit = (t-t[0] >= min(20., (t[-1]-t[0])/4)) & (a*a[0] > 0)
    rate = float(np.polyfit(t[fit]-t[0], np.log(np.abs(a[fit]/a[0])), 1)[0]) if fit.sum() >= 4 and np.all(a*a[0] > 0) else None
    fits = []
    for lo, hi in c["response_fit_windows_s"]:
        use = (t-t[0] >= lo-1e-7) & (t-t[0] <= hi+1e-7)
        slope = float(np.polyfit(t[use]-t[0], np.log(np.abs(a[use]/a[0])), 1)[0]) if use.sum() >= 4 and np.all(a[use]*a[0] > 0) else None
        alo,ahi=float(np.interp(lo,t-t[0],a)),float(np.interp(hi,t-t[0],a))
        fits.append(dict(start_elapsed_s=lo, stop_elapsed_s=hi, descriptive_log_slope_s_inverse=slope,
            absolute_amplitude_gain=abs(ahi/alo) if abs(alo)>1e-12 else None))
    full_response=(finite(plus['UE'])-finite(minus['UE']))/(2*amplitude)
    mode_coefficients={m:np.sum(full_response*np.cos(m*np.pi*(z+h)/h)[None,:]*dz[None,:],axis=1)/
        np.sum(np.cos(m*np.pi*(z+h)/h)**2*dz) for m in range(1,9)}
    response_norm2=np.sum(full_response**2*dz[None,:],axis=1)
    mixing=np.where(response_norm2>1e-24,np.maximum(0,1-a*a*norm/response_norm2),0.)
    rows = [dict(time_s=float(time), elapsed_s=float(time-t[0]), response_amplitude=float(a[i]),
                 mode_mixing_fraction=float(mixing[i]),
                 **{f'projection_mode_{m}':float(values[i]) for m,values in mode_coefficients.items()},
                 **{name+"_response_tendency_s_inverse": float(values[i]) for name, values in terms.items()}) for i, time in enumerate(t)]
    return dict(mode=int(mode), vertical_wavelength_m=float(2*h/mode), probe_amplitude_m_s=float(amplitude),
        start_s=float(t[0]), stop_s=float(t[-1]), initial_response=float(a[0]), final_response=float(a[-1]),
        final_absolute_gain=gain, maximum_absolute_gain=float(np.max(np.abs(a/a[0]))) if a[0] else None,
        descriptive_log_slope_s_inverse=rate, descriptive_fit_windows=fits, sign_changes=bool(np.any(a*a[0] <= 0)),
        final_mode_mixing_fraction=float(mixing[-1]),maximum_mode_mixing_fraction=float(mixing.max()),
        response_budget_residual=residual, response_budget_relative_residual=residual_fraction,
        budget_passed=budget_ok, initial_mode_matches=initial_match, integrated_modal_energy_supplies=supplies,
        eddy_reinforcement=eddy_reinforces, assessment=outcome,
        scope="Finite-time weak-band response; positive candidates require amplitude and seed checks; no negative result excludes finite-amplitude jets"), rows


def profile_properties(ue, initial, z, dz, tolerance=1e-8):
    """Describe profiles without calling every wind-driven shear profile a jet."""
    ue, initial, z, dz = map(finite, (ue, initial, z, dz))
    maxima, minima = [], []
    for j in range(1, len(z)-1):
        if ue[j]-ue[j-1] > tolerance and ue[j]-ue[j+1] > tolerance:
            maxima.append(dict(z_m=float(z[j]), U_E_m_s=float(ue[j])))
        if ue[j-1]-ue[j] > tolerance and ue[j+1]-ue[j] > tolerance:
            minima.append(dict(z_m=float(z[j]), U_E_m_s=float(ue[j])))
    return dict(current_change_rms_m_s=profile_rms(ue-initial, dz),
                depth_mean_change_m_s=float(np.sum((ue-initial)*dz)/dz.sum()),
                interior_maxima=maxima, interior_minima=minima,
                scope="Current-change amplitude is redistribution, not by itself a spontaneous-jet diagnostic")


def resolved_extrema(values,z,prominence_floor):
    """Interior extrema with two-sided prominence and half-prominence width."""
    values,z=finite(values),finite(z); found=[]
    for sign,kind in ((1.,'maximum'),(-1.,'minimum')):
        y=sign*values
        peaks=[j for j in range(1,len(z)-1) if y[j]>y[j-1] and y[j]>y[j+1]]
        troughs=[j for j in range(1,len(z)-1) if y[j]<y[j-1] and y[j]<y[j+1]]
        for j in peaks:
            left=max([0]+[l for l in troughs if l<j]);right=min([len(z)-1]+[l for l in troughs if l>j])
            prominence=float(y[j]-max(y[left],y[right]))
            if prominence<prominence_floor:continue
            level=y[j]-prominence/2
            lo=j
            while lo>left and y[lo]>level:lo-=1
            hi=j
            while hi<right and y[hi]>level:hi+=1
            zl=float(z[lo]+(level-y[lo])*(z[lo+1]-z[lo])/(y[lo+1]-y[lo])) if lo<j else float(z[j])
            zr=float(z[hi-1]+(level-y[hi-1])*(z[hi]-z[hi-1])/(y[hi]-y[hi-1])) if hi>j else float(z[j])
            found.append(dict(kind=kind,z_m=float(z[j]),U_E_m_s=float(values[j]),prominence_m_s=prominence,
                              half_prominence_width_m=max(0.,zr-zl)))
    return found


def current_organization(profiles,h,initial_eddy):
    """Descriptive persistence and modal amplitudes, not an automatic jet verdict."""
    t,z,dz=map(finite,(profiles['time'],profiles['z'],profiles['dz']));ue=finite(profiles['UE'])
    floor=.01*initial_eddy;start=max(t[0],t[-1]-200.);late=t>=start
    average=np.trapezoid(ue[late],t[late],axis=0)/(t[late][-1]-t[late][0])
    histories=[];by_time=[];metrics=[]
    for i,time in enumerate(t):
        extrema=resolved_extrema(ue[i],z,floor);by_time.append(extrema)
        if late[i]:histories.extend(dict(time_s=float(time),**e) for e in extrema)
        row=dict(time_s=float(time),current_change_rms_m_s=profile_rms(ue[i]-ue[0],dz),
                 interior_maxima=sum(e['kind']=='maximum' for e in extrema),interior_minima=sum(e['kind']=='minimum' for e in extrema))
        for m in range(1,9):
            phi=np.cos(m*np.pi*(z+h)/h);norm=np.sum(phi*phi*dz)
            row[f'current_mode_{m}_m_s']=float(np.sum(ue[i]*phi*dz)/norm)
            row[f'generated_mode_{m}_m_s']=float(np.sum((ue[i]-ue[0])*phi*dz)/norm)
        metrics.append(row)
    persistent=[]
    for e in resolved_extrema(average,z,floor):
        tolerance=max(e['half_prominence_width_m']/2,2*float(np.max(dz)))
        presence=[any(a['kind']==e['kind'] and abs(a['z_m']-e['z_m'])<=tolerance for a in by_time[i]) for i in np.flatnonzero(late)]
        fraction=float(np.mean(presence))
        persistent.append(e|dict(presence_fraction=fraction,persistent=fraction>=.8,matching_depth_tolerance_m=tolerance))
    return dict(late_window_s=[float(start),float(t[-1])],prominence_floor_m_s=floor,
        persistence_fraction_threshold=.8,late_mean_extrema=persistent,
        width_definition='Two-sided half-prominence width relative to nearest opposite extrema or boundaries',
        scope='Descriptive profile organization; imposed background and direct wind response can contribute, so persistence alone does not establish the beta mechanism'),metrics,histories
