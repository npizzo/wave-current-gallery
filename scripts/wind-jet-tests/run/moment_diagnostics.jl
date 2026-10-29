# Online Eulerian transverse-vorticity moments of the Lagrangian velocity.
# All volume fields and reductions remain on the model architecture. The only
# host transfers are O(Nz) profiles. Fixed velocity BCs imply zero q_t at z faces.
export init_moment_diagnostics, sample_moment_diagnostics!, close_moment_diagnostics!,
       MOMENT_FIELDS, MOMENT_SCHEMA, MOMENT_PROFILE_SCHEMA

@inline function background_qface(k,p)
    k==1 && return p.gradient_bottom-2p.k*p.S*exp(-2p.k*p.h)
    k==p.nz+1 && return p.gradient_top-2p.k*p.S
    z=-p.h+(k-1)*p.dz
    return -p.alpha*p.Sref*(exp(2p.k*(z+p.dz/2))-exp(2p.k*(z-p.dz/2)))/p.dz
end
@inline background_q(k,p)=(background_qface(k,p)+background_qface(k+1,p))/2
@inline function background_qz(k,p)
    k==1 && return (-3background_q(1,p)+4background_q(2,p)-background_q(3,p))/(2p.dz)
    k==p.nz && return (3background_q(k,p)-4background_q(k-1,p)+background_q(k-2,p))/(2p.dz)
    return (background_q(k+1,p)-background_q(k-1,p))/(2p.dz)
end
@inline function moment_T(i,j,k,u,v,w,q,p)
    s=q[i,j,k]*dc(i,j,k,v,p,2)
    x=dc(i,j,k,w,p,2)*dc(i,j,k,v,p,1)
    z=-dc(i,j,k,u,p,2)*dc(i,j,k,v,p,3)
    return s,x,z
end
@inline moment_covariance(i,j,k,g,u,q,U,Q)= (u[i,j,k]-U[i,1,k])*(q[i,j,k]-Q[i,1,k])
@inline function moment_native_inviscid(i,j,k,g,au,aw,su,sw,p)
    return tendency_curl_center(i,j,k,g,au,aw,p)+tendency_curl_center(i,j,k,g,su,sw,p)
end
@inline function moment_native_feedback(i,j,k,g,au,aw,a0,w0,p)
    return tendency_curl_center(i,j,k,g,au,aw,p)-tendency_curl_center(i,j,k,g,a0,w0,p)
end

const MOMENT_BASE_FIELDS=["Zq","Zmean","Zvariance","Za","Zmean_anomaly",
    "Pq_stretch","Pq_tilt_wy_vx","Pq_tilt_minus_uy_vz","Pq_T","Pq_T_abs",
    "Pq_stretch_abs","Pq_tilt_wy_vx_abs","Pq_tilt_minus_uy_vz_abs",
    "Pa_stretch","Pa_tilt_wy_vx","Pa_tilt_minus_uy_vz","Pa_T","Pa_T_abs",
    "Pa_stretch_abs","Pa_tilt_wy_vx_abs","Pa_tilt_minus_uy_vz_abs","Pa_background",
    "Pmean_exchange","Pvariance_exchange","Pmean_T","Pvariance_T","Pmean_anomaly_T",
    "Pmean_anomaly_background","Pmean_feedback_native","Pmean_anomaly_feedback_native",
    "T2","v2","qmean","q0_native_center","q0_analytic_center","q0_gradient_native",
    "q_continuum_rhs_error2","T_unsimplified_error2","Pq_advection","Pa_advection","Pmean_anomaly_exchange"]
const N_MOMENT_BASE_FIELDS=41
@assert length(MOMENT_BASE_FIELDS)==N_MOMENT_BASE_FIELDS
const MOMENT_FAMILIES=("Zq","Zmean","Zvariance","Za","Zmean_anomaly")
const MOMENT_RATE_TERMS=("inviscid","viscous","forcing","pressure","native")
const MOMENT_FIELDS=vcat(MOMENT_BASE_FIELDS,
    [term=="native" ? "d$(family)_native" : "$(family)_$(term)" for family in MOMENT_FAMILIES for term in MOMENT_RATE_TERMS])
const MOMENT_SCHEMA=vcat(["time_s","iteration"],MOMENT_FIELDS,
    ["T_rms","v_rms","q_continuum_rhs_error_rms","native_partition_error","anomaly_partition_error"])
const MOMENT_FLUX_NAMES=("q","mean","variance","a","mean_anomaly")
const MOMENT_PROFILE_SCHEMA=vcat(["time_s","z_m","dz_m"],MOMENT_FIELDS,
    ["flux_$(f)_$(side)" for f in MOMENT_FLUX_NAMES for side in ("lower","upper")])

# Integer mode shares one compiled kernel specialization across diagnostic fields.
# All native derivatives are fresh curls of the actual model operators, before
# averaging in y. These are semi-discrete chain-rule rates, not time differences.
@inline function moment_value(i,j,k,g,u,v,w,q,U,W,Q,Fx,Fz,qi,qv,qf,qp,qfb,p,mode)
    qq=q[i,j,k]; qm=Q[i,1,k]; q0=background_q(k,p); a=qq-q0; am=qm-q0; qt=qq-qm
    if mode<=5
        return mode==1 ? qq^2/2 : mode==2 ? qm^2/2 : mode==3 ? qt^2/2 : mode==4 ? a^2/2 : am^2/2
    end
    if mode>N_MOMENT_BASE_FIELDS
        n=mode-N_MOMENT_BASE_FIELDS-1; family=n÷5+1; term=n%5+1
        weight=family==1 ? qq : family==2 ? qm : family==3 ? qt : family==4 ? a : am
        rate=term==1 ? qi[i,j,k] : term==2 ? qv[i,j,k] : term==3 ? qf[i,j,k] : term==4 ? qp[i,j,k] :
            qi[i,j,k]+qv[i,j,k]+qf[i,j,k]+qp[i,j,k]
        return weight*rate
    end
    s,x,z=moment_T(i,j,k,u,v,w,q,p); tt=s+x+z
    mode==6 && return qq*s
    mode==7 && return qq*x
    mode==8 && return qq*z
    mode==9 && return qq*tt
    mode==10 && return abs(qq*tt)
    mode==11 && return abs(qq*s)
    mode==12 && return abs(qq*x)
    mode==13 && return abs(qq*z)
    mode==14 && return a*s
    mode==15 && return a*x
    mode==16 && return a*z
    mode==17 && return a*tt
    mode==18 && return abs(a*tt)
    mode==19 && return abs(a*s)
    mode==20 && return abs(a*x)
    mode==21 && return abs(a*z)
    mode==22 && return -a*w[i,j,k]*background_qz(k,p)
    exchange=Fx[i,1,k]*dc(i,1,k,Q,p,1)+Fz[i,1,k]*dc(i,1,k,Q,p,3)
    mode==23 && return exchange
    mode==24 && return -exchange
    mode==25 && return qm*tt
    mode==26 && return qt*tt
    mode==27 && return am*tt
    mode==28 && return -am*w[i,j,k]*background_qz(k,p)
    mode==29 && return qm*qfb[i,j,k]
    mode==30 && return am*qfb[i,j,k]
    mode==31 && return tt^2
    mode==32 && return v[i,j,k]^2
    mode==33 && return qq
    mode==34 && return q0
    mode==35 && return -p.alpha*2p.k*p.Sref*exp(2p.k*(-p.h+(k-.5)*p.dz))
    mode==36 && return background_qz(k,p)
    adv=u[i,j,k]*dc(i,j,k,q,p,1)+v[i,j,k]*dc(i,j,k,q,p,2)+w[i,j,k]*dc(i,j,k,q,p,3)
    mode==37 && return (qi[i,j,k]+adv-tt)^2
    unsimplified=(dc(i,j,k,w,p,2)-dc(i,j,k,v,p,3))*dc(i,j,k,v,p,1)+s+
        (dc(i,j,k,v,p,1)-dc(i,j,k,u,p,2))*dc(i,j,k,v,p,3)
    mode==38 && return (unsimplified-tt)^2
    mode==39 && return -qq*adv
    mode==40 && return -a*(adv-w[i,j,k]*background_qz(k,p))
    return exchange-Fz[i,1,k]*background_qz(k,p)
end

# Face transport, averaged in x,y. The mean flux includes covariance transport:
# w*(Q*q-Q²/2), so total = mean + variance exactly on every saved face.
# q interpolates the cell-centred diagnostic, not a separate face-vorticity norm.
@inline function moment_flux(i,j,k,g,w,q,Q,p,mode)
    (k==1 || k==p.nz+1) && return zero(eltype(q))
    qq=(q[i,j,k-1]+q[i,j,k])/2; qm=(Q[i,1,k-1]+Q[i,1,k])/2
    q0=(background_q(k-1,p)+background_q(k,p))/2
    a=qq-q0; am=qm-q0
    value=mode==1 ? qq^2/2 : mode==2 ? qm*qq-qm^2/2 : mode==3 ? (qq-qm)^2/2 :
          mode==4 ? a^2/2 : am*a-am^2/2
    return w[i,j,k]*value
end

mutable struct MomentDiagnosticsState
    regime::Any
    p::Any
    means::Any
    covariances::Any
    native::Any
    reduced::Any
    fluxes::Any
    io::IO
    profiles_io::IO
    previous_time::Float64
end

function init_moment_diagnostics(regime;resume=false)
    jets=regime.jets;m=jets.model;g=m.grid;c=jets.context
    p=merge(regime.p,(;alpha=c.alpha,Sref=c.S_reference))
    u,v,w,q=regime.centers
    means=(U=Field(Average(u;dims=2)),W=Field(Average(w;dims=2)),Q=Field(Average(q;dims=2)))
    U,W,Q=means
    covariances=(Fx=Field(Average(Field(KernelFunctionOperation{Center,Center,Center}(moment_covariance,g,u,q,U,Q));dims=2)),
                 Fz=Field(Average(Field(KernelFunctionOperation{Center,Center,Center}(moment_covariance,g,w,q,W,Q));dims=2)))
    n=jets.operators.native
    qi=Field(KernelFunctionOperation{Center,Center,Center}(moment_native_inviscid,g,n.U_adv_full,n.W_adv_full,n.U_wave,n.W_wave,p))
    # Viscosity acts on Lagrangian velocity. Its curl contribution is retained
    # exactly as an operator rate, including imposed boundary gradients; it is
    # not labeled -nu*|grad(q)|². Background-maintenance forcing stays separate.
    qv=Field(KernelFunctionOperation{Center,Center,Center}(tendency_curl_center,g,n.U_viscous,n.W_viscous,p))
    qf=Field(KernelFunctionOperation{Center,Center,Center}(tendency_curl_center,g,n.U_forcing,n.W_forcing,p))
    qfb=Field(KernelFunctionOperation{Center,Center,Center}(moment_native_feedback,g,n.U_adv_full,n.W_adv_full,n.U_adv_mean,n.W_adv_mean,p))
    qp=regime.pressure_curl
    native=(;qi,qv,qf,qp,qfb)
    # Read the already-refreshed native buffers in terminal reductions. Passing
    # computed Fields here would recursively recompute their entire dependency
    # graph for each of the 66 moments when compute!'s time argument is nothing.
    # .data preserves the native offset indices and halos on CPU and GPU. The
    # explicit refresh order in sample_moment_diagnostics! is required; no clock
    # caching is used, and no three-dimensional tendency is averaged prematurely.
    # Product fields remain lazy, so no full 3-D product arrays are retained.
    operands=map(f->f.data,(u,v,w,q,U,W,Q,covariances.Fx,covariances.Fz,qi,qv,qf,qp,qfb))
    reduced=[Field(Average(KernelFunctionOperation{Center,Center,Center}(moment_value,g,
        operands...,p,mode);dims=(1,2));compute=false) for mode in eachindex(MOMENT_FIELDS)]
    fluxes=[Field(Average(KernelFunctionOperation{Center,Center,Face}(moment_flux,g,
        m.velocities.w.data,q.data,Q.data,p,mode);dims=(1,2));compute=false) for mode in 1:5]
    paths=(joinpath(c.output,"moments.csv"),joinpath(c.output,"moment-profiles.csv"));previous=-Inf
    for (path,schema) in zip(paths,(MOMENT_SCHEMA,MOMENT_PROFILE_SCHEMA))
        if resume
            lines,rows=read_numeric_csv(path,schema)
            ids=findall(r->r["time_s"]<=m.clock.time+1e-10,rows)
            isempty(ids)&&error("No moment diagnostics preceding checkpoint")
            cp(path,path*".pre-restart-"*string(time_ns()))
            write(path,join(vcat(lines[1],lines[ids.+1]),"\n")*"\n")
            previous=maximum(r["time_s"] for r in rows[ids])
        else
            isfile(path)&&error("Refusing to overwrite moment diagnostics")
            write(path,join(schema,',')*"\n")
        end
    end
    MomentDiagnosticsState(regime,p,means,covariances,native,reduced,fluxes,open(paths[1],"a"),open(paths[2],"a"),previous)
end

function sample_moment_diagnostics!(sim,state)
    t=Float64(time(sim));t<=state.previous_time+1e-10 && return nothing
    r=state.regime;p=state.p
    update_regime_centers!(r)
    # sample_jet_diagnostics! has refreshed full and replicated-mean velocities
    # and their native operators at this exact time; recompute curls from them.
    for f in r.jets.operators.native;compute!(f);end
    for f in state.means;compute!(f);end
    for f in state.covariances;compute!(f);end
    for f in state.native;compute!(f);end
    profiles=Dict(name=>compute_profile(field) for (name,field) in zip(MOMENT_FIELDS,state.reduced))
    fluxes=Dict(name=>compute_profile(field) for (name,field) in zip(MOMENT_FLUX_NAMES,state.fluxes))
    row=Dict(name=>sum(values)/p.nz for (name,values) in profiles)
    merge!(row,Dict("time_s"=>t,"iteration"=>iteration(sim),"T_rms"=>sqrt(max(0,row["T2"])),
        "v_rms"=>sqrt(max(0,row["v2"])),"q_continuum_rhs_error_rms"=>sqrt(max(0,row["q_continuum_rhs_error2"])),
        "native_partition_error"=>row["dZq_native"]-row["dZmean_native"]-row["dZvariance_native"],
        "anomaly_partition_error"=>row["dZa_native"]-row["dZmean_anomaly_native"]-row["dZvariance_native"]))
    all(isfinite,values(row)) || error("Nonfinite vorticity moment")
    csvline(state.io,[row[name] for name in MOMENT_SCHEMA]);flush(state.io)
    for k in 1:p.nz
        values=vcat([t,-p.h+(k-.5)*p.dz,p.dz],[profiles[name][k] for name in MOMENT_FIELDS],
            [fluxes[name][k+side] for name in MOMENT_FLUX_NAMES for side in (0,1)])
        all(isfinite,values)||error("Nonfinite moment profile")
        csvline(state.profiles_io,values)
    end
    flush(state.profiles_io);state.previous_time=t
    row
end
close_moment_diagnostics!(state)=(close(state.io);close(state.profiles_io);nothing)
