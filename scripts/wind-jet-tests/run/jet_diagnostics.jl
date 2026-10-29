module JetDiagnostics

using Oceananigans, Statistics, Dates
using Oceananigans.Operators
using Oceananigans.Grids: topology
using Oceananigans.Advection: div_𝐯u, div_𝐯v, div_𝐯w
using Oceananigans.StokesDrifts: x_curl_Uˢ_cross_U, z_curl_Uˢ_cross_U
using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∂ⱼ_τ₃ⱼ
using Oceananigans.BoundaryConditions: fill_halo_regions!

export init_jet_diagnostics, sample_jet_diagnostics!, close_jet_diagnostics!, native_sample,
       finite_interval_budget, covariance_fluxes, JET_BUDGET_SCHEMA

# Every native operator is evaluated afresh on the current velocity fields.
# In particular no model.timestepper.Gⁿ field or cached RK-stage tendency is read.
@inline native_advection_u(i,j,k,g,adv,u,v,w) = -div_𝐯u(i,j,k,g,adv,(;u,v,w),u)
@inline native_advection_v(i,j,k,g,adv,u,v,w) = -div_𝐯v(i,j,k,g,adv,(;u,v,w),v)
@inline native_advection_w(i,j,k,g,adv,u,v,w) = -div_𝐯w(i,j,k,g,adv,(;u,v,w),w)
@inline native_wave_u(i,j,k,g,sd,u,v,w,clock) = x_curl_Uˢ_cross_U(i,j,k,g,sd,(;u,v,w),clock.time)
@inline native_wave_w(i,j,k,g,sd,u,v,w,clock) = z_curl_Uˢ_cross_U(i,j,k,g,sd,(;u,v,w),clock.time)
@inline native_viscous_u(i,j,k,g,c,cf,clock,u,v,w) = -∂ⱼ_τ₁ⱼ(i,j,k,g,c,cf,clock,(;u,v,w),nothing)
@inline native_viscous_v(i,j,k,g,c,cf,clock,u,v,w) = -∂ⱼ_τ₂ⱼ(i,j,k,g,c,cf,clock,(;u,v,w),nothing)
@inline native_viscous_w(i,j,k,g,c,cf,clock,u,v,w) = -∂ⱼ_τ₃ⱼ(i,j,k,g,c,cf,clock,(;u,v,w),nothing)
@inline native_forcing(i,j,k,g,f,clock,u,v,w) = f(i,j,k,g,clock,(;u,v,w))
@inline native_pressure_u(i,j,k,g,p) = -∂xᶠᶜᶜ(i,j,k,g,p)
@inline at_zface_u(i,j,k,g,u) = ℑxzᶜᵃᶠ(i,j,k,g,u)
@inline flux_uw(i,j,k,g,u,w) = at_zface_u(i,j,k,g,u)*w[i,j,k]
@inline flux_perp(i,j,k,g,u,w,u0,w0) =
    (at_zface_u(i,j,k,g,u)-at_zface_u(i,j,k,g,u0))*(w[i,j,k]-w0[i,j,k])
@inline square_native(i,j,k,g,a) = a[i,j,k]^2
@inline energy_inplane(i,j,k,g,u,w) =
    (ℑxᶜᵃᵃ(i,j,k,g,square_native,u)+ℑzᵃᵃᶜ(i,j,k,g,square_native,w))/2
@inline energy_transverse(i,j,k,g,v) = ℑyᵃᶜᵃ(i,j,k,g,square_native,v)/2
@inline square_difference(i,j,k,g,a,b) = (a[i,j,k]-b[i,j,k])^2
@inline energy_perp(i,j,k,g,u,v,w,u0,v0,w0) =
    (ℑxᶜᵃᵃ(i,j,k,g,square_difference,u,u0)+ℑyᵃᶜᵃ(i,j,k,g,square_difference,v,v0)+
     ℑzᵃᵃᶜ(i,j,k,g,square_difference,w,w0))/2

const SUFFIXES=("L","E","anomaly")
const PRODUCTION_TERMS=("adv_full","adv_ky0","adv_3d","wave","closure","forcing","pressure","total",
                       "continuum_flux","continuum_ky0_flux","continuum_3d_flux","native_minus_continuum")
const JET_BUDGET_SCHEMA=vcat(
    ["time_s","iteration","Ejet_L","Ejet_E","Ejet_anomaly","Etotal","Einplane","Etransverse",
     "Ehorizontal_mean","Eky0","Eeddy","Eky0_eddy","Eperp","ky0_eddy_fraction","Wmean_max",
     "pressure_mean_max","flux_decomposition_max","native_advection_decomposition_max"],
    ["P_$(term)_$suffix" for suffix in SUFFIXES for term in PRODUCTION_TERMS],
    ["interval_dt"],
    ["$(term)_$suffix" for suffix in SUFFIXES for term in
        ("dEjet_dt","interval_P_total","interval_residual","cumulative_P_total","cumulative_residual")])

mutable struct DiagnosticsState
    model::Any
    context::Any
    operators::Any
    zfaces::Vector{Float64}
    zcenters::Vector{Float64}
    dz::Vector{Float64}
    initial_UL::Union{Nothing,Vector{Float64}}
    previous::Any
    first_energies::Any
    cumulative::Dict{String,Float64}
    budget_io::IO
    profiles_io::IO
    flux_io::IO
end

csvline(io,values)=println(io,join((x isa AbstractFloat && !isfinite(x) ? "" : string(x) for x in values),','))
profile(a)=vec(Float64.(Array(interior(a))))
matrix(a)=dropdims(Float64.(Array(interior(a)));dims=2)
function compute_profile(a)
    compute!(a)
    return profile(a)
end

"""
Initialize diagnostics for periodic x, periodic/flat y, bounded uniform z;
fixed monochromatic Stokes drift u_s=S*exp(2kz), constant explicit
ScalarDiffusivity, and no rotation, buoyancy, background-field split, or moving
free surface. Explicit model forcing is included as its own budget term.

ctx must provide output,zf,k,S,Lx,Ly,H. The native advection contribution of
transverse structure is the DIFFERENCE of the same numerical operator applied
to full velocities and to their y averages replicated on the original grid.
It includes nonlinear WENO reconstruction and is not identified with a pure
continuum Reynolds-stress term. Direct Reynolds covariances are also retained.
"""
function init_jet_diagnostics(model,ctx;resume=false)
    Base.@nospecialize model ctx
    g=model.grid
    topology(g)[1]===Periodic && topology(g)[2] in (Periodic,Flat) && topology(g)[3]===Bounded ||
        error("Jet diagnostics require periodic x, periodic/flat y and bounded z")
    (model.closure===nothing || model.closure isa Oceananigans.TurbulenceClosures.ScalarDiffusivity) ||
        error("Only constant ScalarDiffusivity or no closure supported")
    model.coriolis===nothing || error("Coriolis tendency is not included")
    model.buoyancy===nothing || error("Buoyancy tendency is not included")
    model.free_surface===nothing || error("Moving free-surface budget not supported")
    all(a isa Oceananigans.Fields.ZeroField for a in model.background_fields.velocities) ||
        error("Background fields must be included in prognostic velocities")
    zf=Float64.(ctx.zf); dz=diff(zf); zc=(zf[1:end-1].+zf[2:end])./2
    all(>(0),dz) || error("Nonmonotone z faces")
    maximum(abs.(dz.-first(dz)))<=1e-10*first(dz) || error("Diagnostics currently require uniform z")
    isapprox(sum(dz),ctx.H;rtol=1e-10) || error("H and zfaces disagree")
    u,v,w=model.velocities
    u0=XFaceField(g;boundary_conditions=u.boundary_conditions)
    v0=YFaceField(g;boundary_conditions=v.boundary_conditions)
    w0=ZFaceField(g;boundary_conditions=w.boundary_conditions)
    ym=(U=Field(Average(u;dims=2)),V=Field(Average(v;dims=2)),W=Field(Average(w;dims=2)))
    au=Field(KernelFunctionOperation{Face,Center,Center}(native_advection_u,g,model.advection,u,v,w))
    a0=Field(KernelFunctionOperation{Face,Center,Center}(native_advection_u,g,model.advection,u0,v0,w0))
    av=Field(KernelFunctionOperation{Center,Face,Center}(native_advection_v,g,model.advection,u,v,w))
    aw=Field(KernelFunctionOperation{Center,Center,Face}(native_advection_w,g,model.advection,u,v,w))
    aw0=Field(KernelFunctionOperation{Center,Center,Face}(native_advection_w,g,model.advection,u0,v0,w0))
    sw=Field(KernelFunctionOperation{Face,Center,Center}(native_wave_u,g,model.stokes_drift,u,v,w,model.clock))
    swz=Field(KernelFunctionOperation{Center,Center,Face}(native_wave_w,g,model.stokes_drift,u,v,w,model.clock))
    vu=Field(KernelFunctionOperation{Face,Center,Center}(native_viscous_u,g,model.closure,model.closure_fields,model.clock,u,v,w))
    vv=Field(KernelFunctionOperation{Center,Face,Center}(native_viscous_v,g,model.closure,model.closure_fields,model.clock,u,v,w))
    vw=Field(KernelFunctionOperation{Center,Center,Face}(native_viscous_w,g,model.closure,model.closure_fields,model.clock,u,v,w))
    fu=Field(KernelFunctionOperation{Face,Center,Center}(native_forcing,g,model.forcing.u,model.clock,u,v,w))
    fw=Field(KernelFunctionOperation{Center,Center,Face}(native_forcing,g,model.forcing.w,model.clock,u,v,w))
    pp=Field(KernelFunctionOperation{Face,Center,Center}(native_pressure_u,g,model.pressures.pNHS))
    ff=Field(KernelFunctionOperation{Center,Center,Face}(flux_uw,g,u,w))
    f0=Field(KernelFunctionOperation{Center,Center,Face}(flux_uw,g,u0,w0))
    fp=Field(KernelFunctionOperation{Center,Center,Face}(flux_perp,g,u,w,u0,w0))
    uf=Field(KernelFunctionOperation{Center,Center,Face}(at_zface_u,g,u))
    ef=Field(KernelFunctionOperation{Center,Center,Center}(energy_inplane,g,u,w))
    ev=Field(KernelFunctionOperation{Center,Center,Center}(energy_transverse,g,v))
    e0=Field(KernelFunctionOperation{Center,Center,Center}(energy_inplane,g,u0,w0))
    v02=Field(KernelFunctionOperation{Center,Center,Center}(energy_transverse,g,v0))
    ep=Field(KernelFunctionOperation{Center,Center,Center}(energy_perp,g,u,v,w,u0,v0,w0))
    xy(a)=Field(Average(a;dims=(1,2)))
    reduced=(adv_full=xy(au),adv_ky0=xy(a0),wave=xy(sw),closure=xy(vu),forcing=xy(fu),pressure=xy(pp),
             raw_flux_full=xy(ff),raw_flux_ky0=xy(f0),flux_3d=xy(fp),Uface=xy(uf),
             inplane=xy(ef),transverse=xy(ev),ky0_inplane=xy(e0),ky0_transverse=xy(v02),perp=xy(ep))
    native=(U_adv_full=au,V_adv_full=av,W_adv_full=aw,U_adv_mean=a0,W_adv_mean=aw0,
            U_wave=sw,W_wave=swz,U_viscous=vu,V_viscous=vv,W_viscous=vw,U_forcing=fu,W_forcing=fw)
    mkpath(ctx.output)
    paths=[joinpath(ctx.output,n) for n in ("jet-budget.csv","jet-profiles.csv","jet-fluxes.csv")]
    profile_header=("time_s","z_m","dz_m","UL","UE","Wmean","flux_total","flux_ky0","flux_3d",
        "adv_full","adv_ky0","adv_3d","wave","closure","forcing","pressure_mean","native_total",
        "continuum_flux_rhs","continuum_ky0_flux_rhs","continuum_3d_flux_rhs","native_minus_continuum")
    flux_header=("time_s","z_face_m","flux_total","flux_ky0","flux_3d","mean_vertical_flux")
    if resume
        recovered=recover_diagnostics(paths,(JET_BUDGET_SCHEMA,profile_header,flux_header),Float64(model.clock.time),zc)
        ios=map(path->open(path,"a"),paths)
        return DiagnosticsState(model,ctx,(;ym,u0,v0,w0,reduced,native),zf,zc,dz,recovered.initial_UL,
            recovered.previous,recovered.first_energies,recovered.cumulative,ios...)
    end
    any(isfile,paths) && error("Jet diagnostics require fresh files or explicit resume=true")
    ios=map(path->open(path,"w"),paths)
    for (io,header) in zip(ios,(JET_BUDGET_SCHEMA,profile_header,flux_header))
        csvline(io,header)
    end
    return DiagnosticsState(model,ctx,(;ym,u0,v0,w0,reduced,native),zf,zc,dz,nothing,nothing,nothing,
        Dict(s=>0. for s in SUFFIXES),ios...)
end

function read_numeric_csv(path,expected)
    lines=readlines(path)
    !isempty(lines) || error("Empty diagnostic CSV $path")
    header=split(first(lines),',')
    header==collect(expected) || error("Diagnostic CSV schema changed: $path")
    records=Dict{String,Float64}[]
    for line in lines[2:end]
        cells=split(line,',')
        length(cells)==length(header) || error("Incomplete diagnostic CSV row in $path")
        push!(records,Dict(string(k)=>(isempty(v) ? NaN : parse(Float64,v)) for (k,v) in zip(header,cells)))
    end
    return lines,records
end

function recover_diagnostics(paths,headers,time,zc)
    all(isfile,paths) || error("Restart requires all three existing diagnostic CSVs")
    readsets=[read_numeric_csv(path,head) for (path,head) in zip(paths,headers)]
    retained=[findall(r->r["time_s"]<=time+1e-10*max(1,abs(time)),records) for (_,records) in readsets]
    all(ids->!isempty(ids),retained) || error("Restart precedes available diagnostic records")
    budget=readsets[1][2][retained[1]]
    all(diff([r["time_s"] for r in budget]).>0) || error("Nonmonotone diagnostic budget times")
    first_time=first(budget)["time_s"]
    initial=[r for r in readsets[2][2] if r["time_s"]==first_time]
    length(initial)==length(zc) || error("Missing initial profile for diagnostic restart")
    all(isapprox(r["z_m"],z;atol=1e-12,rtol=1e-12) for (r,z) in zip(initial,zc)) || error("Restart diagnostic grid changed")
    # Preserve an intact copy before dropping abandoned post-checkpoint rows.
    archive=joinpath(dirname(first(paths)),"diagnostics-pre-restart-"*Dates.format(now(),"yyyymmdd-HHMMSS")*"-"*string(time_ns()))
    mkpath(archive)
    for (path,(lines,_),ids) in zip(paths,readsets,retained)
        cp(path,joinpath(archive,basename(path)))
        open(path*".restart-tmp","w") do io
            println(io,first(lines))
            for id in ids; println(io,lines[id+1]); end
        end
        mv(path*".restart-tmp",path;force=true)
    end
    previous=last(budget)
    return (;initial_UL=[r["UL"] for r in initial],previous,
        first_energies=Dict(s=>first(budget)["Ejet_$s"] for s in SUFFIXES),
        cumulative=Dict(s=>previous["cumulative_P_total_$s"] for s in SUFFIXES))
end

"""Covariance fluxes; full/ky0 raw products and independently computed perp product."""
function covariance_fluxes(rawfull,rawky0,perp,Uface,Wface)
    all(size(a)==size(rawfull) for a in (rawky0,perp,Uface,Wface)) || error("Flux shape mismatch")
    meanflux=Uface.*Wface
    total=rawfull.-meanflux; ky0=rawky0.-meanflux
    return (;total,ky0,three_d=copy(perp),meanflux,
            decomposition_max=maximum(abs.(total.-ky0.-perp)))
end

"""
Evaluate fresh native operators and return host-resident small reductions only.
The stored projection-pressure field is not a current instantaneous pressure
solve. Its horizontal gradient mean is checked because periodic x makes that
mean zero for ANY stored pressure; this is the only pressure fact used here.
No full three-dimensional field is copied to the host.
"""
function native_sample(state)
    model=state.model; o=state.operators; g=model.grid
    # These calls fill current halos; they do not advance prognostic fields.
    fill_halo_regions!(model.velocities,model.clock,Oceananigans.fields(model))
    fill_halo_regions!(model.pressures.pNHS)
    for (dest,src) in zip((o.u0,o.v0,o.w0),o.ym)
        compute!(src)
        interior(dest).=interior(src)
    end
    fill_halo_regions!((u=o.u0,v=o.v0,w=o.w0),model.clock,Oceananigans.fields(model))
    ymeans=(U=matrix(o.ym.U),V=matrix(o.ym.V),W=matrix(o.ym.W))
    profiles=Dict{String,Vector{Float64}}(string(k)=>compute_profile(f) for (k,f) in pairs(o.reduced))
    UL=vec(mean(ymeans.U;dims=1)); Vmean=vec(mean(ymeans.V;dims=1))
    Wface=vec(mean(ymeans.W;dims=1)); Wmean=(Wface[1:end-1].+Wface[2:end])./2
    UE=UL.-Float64(state.context.S).*exp.(2Float64(state.context.k).*state.zcenters)
    flux=covariance_fluxes(profiles["raw_flux_full"],profiles["raw_flux_ky0"],profiles["flux_3d"],profiles["Uface"],Wface)
    ad=profiles["adv_full"].-profiles["adv_ky0"]
    cl=profiles["closure"]
    total=profiles["adv_full"].+profiles["wave"].+cl.+profiles["forcing"].+profiles["pressure"]
    rhs=-diff(flux.total)./state.dz
    rhs0=-diff(flux.ky0)./state.dz; rhs3=-diff(flux.three_d)./state.dz
    profiles["adv_3d"]=ad; profiles["closure"]=cl; profiles["total"]=total
    profiles["continuum_flux"]=rhs; profiles["continuum_ky0_flux"]=rhs0
    profiles["continuum_3d_flux"]=rhs3; profiles["native_minus_continuum"]=profiles["adv_full"].-rhs
    profiles["UL"]=UL; profiles["UE"]=UE; profiles["Vmean"]=Vmean; profiles["Wmean"]=Wmean
    profiles["Wface"]=Wface
    for a in values(profiles)
        all(isfinite,a) || error("Nonfinite diagnostic profile")
    end
    return (;time_s=Float64(model.clock.time),iteration=model.clock.iteration,ymeans,profiles,flux)
end

"""Finite-output-interval energy closure; trapezoidal quadrature is not exact RK integration."""
function finite_interval_budget(t,E,P,previous_t,previous_E,previous_P,cumulative,Einitial)
    dt=t-previous_t; dt>0 || throw(ArgumentError("Diagnostic times must increase"))
    endpoint_P=(P+previous_P)/2
    cumulative_next=cumulative+dt*endpoint_P
    derivative=(E-previous_E)/dt
    return (;dt,derivative,endpoint_P,residual=derivative-endpoint_P,
            cumulative=cumulative_next,cumulative_residual=E-Einitial-cumulative_next)
end

function sample_jet_diagnostics!(simulation,state)
    simulation.model===state.model || error("Diagnostic model mismatch")
    sample=native_sample(state); t=sample.time_s; p=sample.profiles; f=sample.flux
    if state.previous!==nothing && t<=state.previous["time_s"]
        t==state.previous["time_s"] && return (;sample...,scalars=copy(state.previous)) # No duplicate disk row.
        error("Diagnostic clock moved backwards")
    end
    avg(a)=sum(a.*state.dz)/sum(state.dz)
    state.initial_UL===nothing && (state.initial_UL=copy(p["UL"]))
    jets=Dict("L"=>p["UL"].-avg(p["UL"]),"E"=>p["UE"].-avg(p["UE"]),
              "anomaly"=>p["UL"].-state.initial_UL)
    scalar=Dict{String,Float64}("time_s"=>t,"iteration"=>sample.iteration)
    for suffix in SUFFIXES
        scalar["Ejet_$suffix"]=avg(jets[suffix].^2)/2
        for term in PRODUCTION_TERMS
            scalar["P_$(term)_$suffix"]=avg(jets[suffix].*p[term])
        end
    end
    scalar["Einplane"]=avg(p["inplane"]); scalar["Etransverse"]=avg(p["transverse"])
    scalar["Etotal"]=scalar["Einplane"]+scalar["Etransverse"]
    scalar["Eky0"]=avg(p["ky0_inplane"].+p["ky0_transverse"])
    # Interpolation of face squares matches the native-cell energy convention.
    mean_w2=(p["Wface"][1:end-1].^2 .+ p["Wface"][2:end].^2)./2
    scalar["Ehorizontal_mean"]=avg(p["UL"].^2 .+ p["Vmean"].^2 .+ mean_w2)/2
    scalar["Eeddy"]=scalar["Etotal"]-scalar["Ehorizontal_mean"]
    scalar["Eky0_eddy"]=scalar["Eky0"]-scalar["Ehorizontal_mean"]
    scalar["Eperp"]=avg(p["perp"])
    scalar["ky0_eddy_fraction"]=scalar["Eeddy"]>0 ? scalar["Eky0_eddy"]/scalar["Eeddy"] : NaN
    scalar["Wmean_max"]=maximum(abs,p["Wface"])
    scalar["pressure_mean_max"]=maximum(abs,p["pressure"])
    scalar["flux_decomposition_max"]=f.decomposition_max
    scalar["native_advection_decomposition_max"]=maximum(abs.(p["adv_full"].-p["adv_ky0"].-p["adv_3d"]))
    scalar["interval_dt"]=state.previous===nothing ? NaN : t-state.previous["time_s"]
    state.first_energies===nothing && (state.first_energies=Dict(s=>scalar["Ejet_$s"] for s in SUFFIXES))
    for suffix in SUFFIXES
        if state.previous===nothing
            scalar["dEjet_dt_$suffix"]=NaN; scalar["interval_P_total_$suffix"]=NaN
            scalar["interval_residual_$suffix"]=NaN
            scalar["cumulative_P_total_$suffix"]=0.; scalar["cumulative_residual_$suffix"]=0.
        else
            interval=finite_interval_budget(t,scalar["Ejet_$suffix"],scalar["P_total_$suffix"],
                state.previous["time_s"],state.previous["Ejet_$suffix"],state.previous["P_total_$suffix"],
                state.cumulative[suffix],state.first_energies[suffix])
            scalar["dEjet_dt_$suffix"]=interval.derivative
            scalar["interval_P_total_$suffix"]=interval.endpoint_P
            scalar["interval_residual_$suffix"]=interval.residual
            scalar["cumulative_P_total_$suffix"]=interval.cumulative
            scalar["cumulative_residual_$suffix"]=interval.cumulative_residual
            state.cumulative[suffix]=interval.cumulative
        end
    end
    csvline(state.budget_io,[scalar[k] for k in JET_BUDGET_SCHEMA])
    mid(a)=(a[1:end-1].+a[2:end])./2
    flux_profiles=(mid(f.total),mid(f.ky0),mid(f.three_d))
    for j in eachindex(state.zcenters)
        csvline(state.profiles_io,(t,state.zcenters[j],state.dz[j],p["UL"][j],p["UE"][j],p["Wmean"][j],
            (a[j] for a in flux_profiles)...,(p[n][j] for n in ("adv_full","adv_ky0","adv_3d","wave","closure","forcing","pressure","total", 
            "continuum_flux","continuum_ky0_flux","continuum_3d_flux","native_minus_continuum"))...))
    end
    for j in eachindex(state.zfaces)
        csvline(state.flux_io,(t,state.zfaces[j],f.total[j],f.ky0[j],f.three_d[j],f.meanflux[j]))
    end
    foreach(flush,(state.budget_io,state.profiles_io,state.flux_io))
    state.previous=scalar
    return (;sample...,scalars=scalar)
end

function close_jet_diagnostics!(state)
    foreach(close,(state.budget_io,state.profiles_io,state.flux_io))
    return nothing
end

include("regime_diagnostics.jl")
include("moment_diagnostics.jl")

end # module
