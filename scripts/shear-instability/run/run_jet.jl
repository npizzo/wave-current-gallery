# Shear pilot: full 3D and projected 2D3C continuations of a jet-bearing parent.
# Original prescribed background maintenance and gradients are retained.
# Velocities are Lagrangian.
# Viscosity and the imposed momentum flux act on that prognostic velocity;
# the uniform return force removes only depth-uniform acceleration.
using Oceananigans, JLD2, Statistics, Printf, SHA, Dates, TOML
if uppercase(get(ENV,"ARCH","GPU"))=="GPU"
    @eval using CUDA
end
include("initial_conditions.jl")
include("vortex_modes.jl")
include("regime_physics.jl")
include("wind_physics.jl")
include("jet_diagnostics.jl")
using .JetDiagnostics
JLD2.default_iotype()=IOStream

envfloat(k,d)=parse(Float64,get(ENV,k,string(d)))
envint(k,d)=parse(Int,get(ENV,k,string(d)))
jstr(s::AbstractString)="\""*replace(s,"\\"=>"\\\\","\""=>"\\\"","\n"=>"\\n","\r"=>"\\r","\t"=>"\\t")*"\""
jstr(x::Bool)=x ? "true" : "false"
jstr(x::Integer)=string(x)
jstr(x::AbstractFloat)=isfinite(x) ? string(x) : "null"
jstr(::Nothing)="null"
jstr(x::Union{AbstractVector,Tuple})="["*join(jstr.(x),",")*"]"
jstr(x::NamedTuple)="{"*join([jstr(string(k))*":"*jstr(v) for (k,v) in pairs(x)],",")*"}"
jstr(x::AbstractDict)="{"*join([jstr(string(k))*":"*jstr(x[k]) for k in sort(collect(keys(x));by=string)],",")*"}"
function write_json(path,data)
    open(path*".tmp","w") do io;println(io,jstr(data));end
    mv(path*".tmp",path;force=true)
end
filehash(path)=open(io->bytes2hex(sha256(io)),path)
arrayhash(a)=bytes2hex(sha256(reinterpret(UInt8,vec(a))))
fieldhash(arrays)=bytes2hex(sha256(join(arrayhash.(arrays),":")))
@inline jet_stokes_shear(z,t,p)=2p.k*p.S*exp(2p.k*z)

function archive_later_outputs(out,t,iter)
    archive=joinpath(out,"archived_after_"*Dates.format(now(),"yyyymmdd_HHMMSS")*"_"*string(getpid()))
    for name in readdir(out)
        m=match(r"^(?:t|mean_)([0-9]+\.[0-9]+)",name)
        ((m!==nothing && parse(Float64,m.captures[1])>t+5e-7) || name=="completed.json") || continue
        mkpath(archive);mv(joinpath(out,name),joinpath(archive,name))
    end
    if isfile(joinpath(out,"probe.csv"))
        path=joinpath(out,"probe.csv");lines=readlines(path)
        later=filter(s->parse(Float64,first(split(s,',')))>t+1e-9,lines[2:end])
        if !isempty(later)
            mkpath(archive);cp(path,joinpath(archive,"probe.csv"))
            keep=vcat(lines[1],filter(s->parse(Float64,first(split(s,',')))<=t+1e-9,lines[2:end]))
            write(path,join(keep,"\n")*"\n")
        end
    end
    ckdir=joinpath(out,"checkpoints")
    isdir(ckdir) || return
    for name in readdir(ckdir)
        m=match(r"^jet_iteration([0-9]+)\.jld2$",name)
        m!==nothing && parse(Int,m.captures[1])>iter || continue
        mkpath(joinpath(archive,"checkpoints"))
        mv(joinpath(ckdir,name),joinpath(archive,"checkpoints",name))
    end
end

function main()
    architecture=uppercase(get(ENV,"ARCH","GPU"));architecture in ("CPU","GPU") || error("Invalid ARCH")
    if architecture=="GPU"
        CUDA.functional() || error("GPU unavailable; request a GPU allocation")
        CUDA.allowscalar(false)
    end
    arch=architecture=="GPU" ? GPU() : CPU()
    nx,ny,nz=envint("NX",128),envint("NY",64),envint("NZ",96)
    lx,ly,h=envfloat("LX",2.),envfloat("LY",1.),envfloat("H",.5)
    k=envfloat("K0",6.1);Sref=envfloat("STOKES_SPEED",.17^2*sqrt(9.81/k))
    alpha=envfloat("ALPHA",.75);uoffset=envfloat("UOFFSET",0.);waves=envint("WAVES",1)
    S=waves*Sref;eddy=envfloat("EDDY_RMS",.004);wavelength=envfloat("INJECTION_WAVELENGTH",.25)
    eta=envfloat("PERTURB_ENERGY_RATIO",0.);seed=envint("SEED",1)
    two_d_three_c_rms=envfloat("TWO_D_THREE_C_RMS",0.)
    nu=envfloat("NU",eddy*wavelength/100)
    maintain=envint("MAINTAIN_BACKGROUND",1)==1
    maintain || error("Shear pilot must retain original background maintenance")
    ustar=envfloat("WIND_USTAR",0.)
    init_snapshot=get(ENV,"INIT_SNAPSHOT","");init_kind=get(ENV,"INIT_KIND","full")
    init_metadata=get(ENV,"INIT_METADATA","")
    init_approval=get(ENV,"INIT_APPROVAL","");init_source_id=get(ENV,"INIT_SOURCE_ID","")
    init_approval_sha256=get(ENV,"INIT_APPROVAL_SHA256","")
    jet_mode=envint("JET_MODE",0);jet_sign=envint("JET_SIGN",0);jet_amplitude=envfloat("JET_AMPLITUDE",0.)
    moments_enabled=envint("MOMENTS_ENABLED",0)==1
    advection_name=get(ENV,"ADVECTION","Centered2")
    advection_name in ("Centered2","Centered4","WENO5") || error("Unsupported ADVECTION")
    ic_kind=get(ENV,"IC_KIND","embedded")
    probe_mode=envint("PROBE_MODE",0);probe_amp=envfloat("PROBE_AMPLITUDE",0.);probe_sign=envint("PROBE_SIGN",0)
    stop=envfloat("STOP_TIME",120.);maxdt=envfloat("MAX_DT",.02)
    diagnostic_interval=envfloat("DIAGNOSTIC_INTERVAL",1.)
    moment_interval=envfloat("MOMENT_INTERVAL",1.)
    mean_interval=envfloat("REGIME_INTERVAL",envfloat("MEAN_INTERVAL",5.))
    checkpoint_wall=envfloat("CHECKPOINT_WALL",300.);keep_checkpoint=envint("KEEP_CHECKPOINT",1)==1
    out=abspath(get(ENV,"OUT_DIR",joinpath(@__DIR__,"..","data","run")));pickup=get(ENV,"PICKUP","")
    all(n->n>=8&&iseven(n),(nx,nz)) || error("NX/NZ must be even and >=8")
    (ny==1 || (ny>=8&&iseven(ny))) || error("NY must be1 (Flat) or even >=8")
    all(>(0),(lx,ly,h,k,Sref,wavelength,stop,maxdt,diagnostic_interval,moment_interval)) || error("Positive scales required")
    0<alpha<=1 || error("ALPHA must lie in (0,1]")
    all(>=(0),(eddy,eta,probe_amp,mean_interval,checkpoint_wall,nu,two_d_three_c_rms,ustar,jet_amplitude)) || error("Amplitudes/intervals must be nonnegative")
    mean_interval==0 || isapprox(mean_interval/diagnostic_interval,round(mean_interval/diagnostic_interval);atol=1e-10) ||
        error("REGIME_INTERVAL must be an integer multiple of DIAGNOSTIC_INTERVAL")
    waves in (0,1) && probe_sign in (-1,0,1) || error("Invalid WAVES or PROBE_SIGN")
    nu>0 || error("Wind campaign requires positive explicit viscosity")
    jet_sign in (-1,0,1) || error("Invalid JET_SIGN")
    0<=jet_mode<nz÷2 || error("JET_MODE outside resolved range")
    (jet_amplitude==0 || (jet_mode>0 && jet_sign!=0)) || error("Jet perturbation requires mode and sign")
    init_kind in ("full","mean") || error("INIT_KIND must be full or mean")
    if !isempty(init_approval)
        !isempty(init_snapshot) && isempty(pickup) || error("Approved parent import requires INIT_SNAPSHOT and no PICKUP")
        init_kind=="full" || (init_kind=="mean" && ny==1) || error("Approved mean branches require NY=1")
        jet_mode==0 && jet_sign==0 && jet_amplitude==0 && probe_mode==0 && probe_amp==0 && probe_sign==0 && two_d_three_c_rms==0 ||
            error("Approved shear branches permit no added initial velocity perturbation")
    else
        isempty(init_source_id) && isempty(init_approval_sha256) && isempty(init_metadata) || error("Approval identifiers supplied without INIT_APPROVAL")
    end
    isempty(pickup) || isempty(init_snapshot) || error("PICKUP and INIT_SNAPSHOT are mutually exclusive")
    probe_mode>=0 && probe_mode<nx÷2 || error("PROBE_MODE outside resolved range")
    (probe_amp==0 || (waves==1&&probe_mode>0)) || error("Nonzero probe requires waves and a resolved mode")
    isempty(pickup) && isdir(out) && !isempty(readdir(out)) && error("OUT_DIR is not empty; choose fresh directory or explicit PICKUP")
    if !isempty(pickup)
        isfile(pickup) || error("PICKUP missing")
        realpath(dirname(pickup))==realpath(joinpath(out,"checkpoints")) || error("Checkpoint must belong to this OUT_DIR")
    end
    mkpath(out)
    dx,dy,dz=lx/nx,ly/ny,h/nz;zf=collect(range(-h,0;length=nz+1));zc=(zf[1:end-1].+zf[2:end])./2
    mode=waves==1&&probe_mode>0 ? finite_depth_mode(;nx,nz,lx,h,k,S=Sref,alpha,uoffset,horizontal_mode=probe_mode) : nothing
    ic=initial_velocity_fields(;nx,ny,nz,lx,ly,h,k,S_reference=Sref,waves,alpha,uoffset,
        eddy_rms=eddy,injection_wavelength=wavelength,perturb_energy_ratio=isempty(init_snapshot) ? eta : 0.,seed,ic_kind=isempty(init_snapshot) ? ic_kind : "embedded")
    u0,v0,w0=ic.u,ic.v,ic.w
    # A y-independent transverse component is 2D3C: it contributes no vy
    # and does not alter the prescribed in-plane eddy normalization or eta.
    if two_d_three_c_rms>0
        transverse=[cos(2pi*(i-.5)/nx)*cos(pi*(kk-.5)/nz) for i in 1:nx,j in 1:ny,kk in 1:nz]
        transverse .*=two_d_three_c_rms/sqrt(mean(abs2,transverse))
        v0 .+=transverse
    end
    basehash=fieldhash(ic.base);perturbhash=fieldhash(ic.perturbation)
    perturbshapehash=ic.perturbation_shape===nothing ? nothing : fieldhash(ic.perturbation_shape)
    probe_scale=mode===nothing ? 0. : add_probe!(u0,w0,mode,lx,probe_amp,probe_sign)
    source_names=("run_jet.jl","initial_conditions.jl","vortex_modes.jl","jet_diagnostics.jl","regime_physics.jl","regime_diagnostics.jl","moment_diagnostics.jl","wind_physics.jl")
    sourcehashes=Dict(n=>filehash(joinpath(@__DIR__,n)) for n in source_names)
    snapshot_meta=nothing;parent_import=nothing;start_time=envfloat("START_TIME",0.)
    if !isempty(init_snapshot)
        loaded=load_wind_snapshot(init_snapshot,init_kind,(;nx,ny,nz,lx,ly,h,k,Sref,alpha,uoffset,waves,nu,ustar,maintain,advection_name),sourcehashes;
            metadata_path=init_metadata,approval_path=init_approval,source_id=init_source_id,approval_sha256=init_approval_sha256)
        u0,v0,w0=loaded.arrays
        snapshot_meta=loaded.metadata
        parent_import=loaded.approved
        start_time=Float64(snapshot_meta["time"])
        haskey(ENV,"START_TIME") && !isapprox(envfloat("START_TIME",start_time),start_time;atol=1e-10,rtol=0) && error("START_TIME must match snapshot time")
    end
    start_time>=0 && stop>start_time || error("STOP_TIME must exceed initial physical time")
    jet_shape=add_jet_perturbation!(u0,h,jet_mode,jet_amplitude,jet_sign)
    initialhash=fieldhash((u0,v0,w0))
    initialdiv=maximum(abs,native_divergence(u0,v0,w0,dx,dy,dz))
    initialmetrics=merge(ic.metrics,(;max_native_divergence=initialdiv,probe_streamfunction_amplitude=probe_scale,
        initial_total_rms=velocity_norms(u0,v0,w0).total,
        achieved_base_Re_lambda=nu>0 ? ic.metrics.base_total_rms*wavelength/nu : nothing,
        achieved_eddy_Re_lambda=nu>0 ? hypot(ic.metrics.base_total_rms,ic.metrics.perturb_total_rms)*wavelength/nu : nothing,
        achieved_native_base_Re_lambda=nu>0 ? ic.metrics.base_native_rms*wavelength/nu : nothing,
        achieved_native_eddy_Re_lambda=nu>0 ? hypot(ic.metrics.base_native_rms,ic.metrics.perturb_native_rms)*wavelength/nu : nothing,
        normalization_scope=isempty(init_snapshot) ? "generated initial condition" : "generator reference only; actual branch velocity is loaded from verified snapshot",
        achieved_U_over_S=ic.metrics.base_total_rms/Sref))
    baseline_gradient_bottom=(waves-alpha)*2k*Sref*exp(-2k*h)
    baseline_gradient_top=(waves-alpha)*2k*Sref
    gradient_bottom=baseline_gradient_bottom;gradient_top=baseline_gradient_top+ustar^2/nu
    wind_return_acceleration=-ustar^2/h
    diagnostic_trim_cells=wind_trim_cells(nz)
    config=(;physics_version=6,wind_ustar=ustar,wind_stress_kinematic=ustar^2,incremental_wind_stress=ustar^2,
        gradient_top,gradient_bottom,baseline_gradient_top,baseline_gradient_bottom,wind_return_acceleration,
        diagnostic_trim_reference_nz=96,diagnostic_trim_reference_cells=2,diagnostic_trim_cells,
        start_time,initial_snapshot_path=isempty(init_snapshot) ? nothing : abspath(init_snapshot),
        initial_snapshot_sha256=isempty(init_snapshot) ? nothing : filehash(init_snapshot),initial_snapshot_kind=init_kind,
        initial_snapshot_metadata_sha256=isempty(init_snapshot) ? nothing : filehash(isempty(init_metadata) ? replace(init_snapshot,r"\.json$"=>".toml") : init_metadata),
        approved_parent_import=parent_import,
        jet_mode,jet_sign,jet_amplitude,moments_enabled,nx,ny,nz,lx,ly,h,k,S,S_reference=Sref,alpha,uoffset,eddy_rms=eddy,nu,
        injection_wavelength=wavelength,perturb_energy_ratio=eta,seed,ic_kind,waves=waves==1,
        probe_mode,probe_amplitude=probe_amp,probe_sign,stop_time=stop,max_dt=maxdt,
        diagnostic_interval,moment_interval,mean_interval,two_d_three_c_rms,architecture,velocity_convention="Lagrangian",precision="Float64",
        base_ic_sha256=basehash,perturbation_ic_sha256=perturbhash,initial_condition_sha256=initialhash,
        perturbation_shape_sha256=perturbshapehash,
        source_hashes=sourcehashes,initial_metrics=initialmetrics,
        moment_diagnostic_version=1,
        moment_q_convention="cell-centred average of native Eulerian x-z curl; q0 is the same native curl of prescribed Eulerian background",
        moment_budget_convention="native semi-discrete chain-rule rates before transverse averaging; continuum production and interpolated advective face transport have a separately reported spatial discretization residual",
        two_d_three_c_rule="optional y-invariant v(x,z), normalized to requested centered RMS; eta and Re_lambda continue to refer to original in-plane base eddy",
        mean_flow_rule="UE=-alpha*us_reference+uoffset; UL=(waves-alpha)*us_reference+uoffset; q0=-alpha*us_reference_z",
        perturbation_rule="zero-y-mean native discrete curl; requested eta=centered-velocity total3DperturbationKE/base2DinplaneKE; achieved native face-square ratio also recorded",
        advection=advection_name,timestepper="RK3",closure="constant ScalarDiffusivity on Lagrangian velocity",advective_cfl=.4,
        diffusive_cfl=.2,maintain_background=maintain,
        background_force_rule="unchanged parent force: minus native vertical viscous stencil of prescribed UL=(waves-alpha)*Sref*exp(2kz)+uoffset, with ORIGINAL background gradient boundary fluxes",
        wind_force_rule="additional constant uniform -u_star^2/H; zero curl, balances only added top wind flux; original nonuniform maintenance remains",
        experiment_scope="shear pilot: full3D stress variations and native transverse-mean2D3C controls of the same parent; retain waves, viscosity, gradients and maintenance; vary only approved wind stress and projection",
        projection_rule=init_kind=="mean" ? "arithmetic transverse average of each native staggered velocity component; preserve y-independent v(x,z), all horizontal mean profiles and in-plane discrete solenoidality" : "full native parent velocity arrays",
        velocity_boundary="w=0; top UL_z=baseline_gradient_top+u_star^2/nu; bottom UL_z=baseline_gradient_bottom; zero v gradient. Wind onset is an abrupt added-gradient step; native interior velocities are unchanged",
        branch_time_convention="physical parent clock retained; elapsed branch time is time minus start_time; native boundary vorticity includes added wind stress from first diagnostic; original background q0 remains the prescribed reference",
        julia_version=string(VERSION),oceananigans_version=string(pkgversion(Oceananigans)))
    # Output cadence and stop time can be changed on an explicit continuation;
    # equations, initial state, dt cap, source and discretization cannot.
    signature=bytes2hex(sha256(jstr((;nx,ny,nz,lx,ly,h,k,Sref,alpha,uoffset,eddy,wavelength,eta,seed,ic_kind,
        waves,probe_mode,probe_amp,probe_sign,maxdt,nu,maintain,advection_name,two_d_three_c_rms,ustar,jet_mode,jet_sign,jet_amplitude,start_time,initialhash,sourcehashes,parent_import))))
    sigpath=joinpath(out,"restart_signature.txt")
    if isempty(pickup)
        write(sigpath,signature*"\n");write_json(joinpath(out,"configuration.json"),config)
    else
        isfile(sigpath)&&strip(read(sigpath,String))==signature || error("Restart signature mismatch")
        write_json(joinpath(out,"resume_configuration.json"),config)
    end
    usref=Sref.*exp.(2k.*zc)
    ue_background=-alpha.*usref.+uoffset
    q0_native_faces=vcat(baseline_gradient_bottom-2k*S*exp(-2k*h),diff(ue_background)./dz,baseline_gradient_top-2k*S)
    gridmetadata=(;z_faces=zf,z_centers=zc,x_faces=collect(0:nx-1).*dx,x_centers=(collect(0:nx-1).+.5).*dx,
        y_faces=collect(0:ny-1).*dy,y_centers=(collect(0:ny-1).+.5).*dy,
        us=waves.*usref,q0=-alpha*2k.*usref,q0_gradient=-alpha*(2k)^2 .*usref,
        q0_profile_convention="prescribed parent reference remains -alpha*us_reference_z, including original background-gradient boundary faces; instantaneous native Q uses the added wind gradient at the top separately",
        q0_native_faces,q0_native_gradient_centers=diff(q0_native_faces)./dz,
        imposed_q_boundary_bottom=gradient_bottom-2k*S*exp(-2k*h),imposed_q_boundary_top=gradient_top-2k*S,
        UL_background=(waves-alpha).*usref.+uoffset,UE_background=ue_background,
        dx,dy,dz,recommended_continuum_z_trim_cells=diagnostic_trim_cells,
        mean_map_conventions=(Q="native Eulerian y-vorticity curl averaged from x/z faces to cell centers",A="mean_y(Uc*Qx+Wc*Qz)",
            Rbar="mean_y(omegaEx*Vx+Q*Vy+omegaEz*Vz-Vc*Qy)",
            derivatives="centered second-order periodic x/y, second-order one-sided z at endpoint cells",
            Qdot="fresh native momentum-tendency curl; homogeneous tendency BC; pressure curl zero by discrete identity",
            scalar_residual="Qdot_total+A-Rbar-Qdot_viscous-Qdot_forcing; spatial truncation residual, not time-difference Qt"),
        byte_order=Base.ENDIAN_BOM==0x04030201 ? "little" : "big")
    isempty(pickup) && write_json(joinpath(out,"grid.json"),gridmetadata)
    mode!==nothing && isempty(pickup) && write_json(joinpath(out,"eigenmode.json"),mode)
    envint("PREPARE_ONLY",0)==1 && return
    grid=ny==1 ? RectilinearGrid(arch,Float64;size=(nx,nz),halo=(3,3),x=(0,lx),z=(-h,0),topology=(Periodic,Flat,Bounded)) :
        RectilinearGrid(arch,Float64;size=(nx,ny,nz),halo=(3,3,3),x=(0,lx),y=(0,ly),z=(-h,0),topology=(Periodic,Periodic,Bounded))
    sd=waves==1 ? UniformStokesDrift(∂z_uˢ=jet_stokes_shear,parameters=(;k,S)) : nothing
    ubcs=FieldBoundaryConditions(top=GradientBoundaryCondition(gradient_top),
        bottom=GradientBoundaryCondition(gradient_bottom))
    adv=advection_name=="Centered2" ? Centered(order=2) : advection_name=="Centered4" ? Centered(order=4) : WENO(order=5)
    force_parameters=merge(regime_forcing_parameters(;waves,alpha,S=Sref,k,h,nz,nu),(;wind_return_acceleration))
    forcing=(;u=Forcing(onset_total_force;discrete_form=true,parameters=force_parameters))
    model=NonhydrostaticModel(grid;advection=adv,timestepper=:RungeKutta3,
        stokes_drift=sd,closure=ScalarDiffusivity(ν=nu),forcing,boundary_conditions=(u=ubcs,))
    set!(model;u=u0,v=v0,w=w0)
    model.clock.time=start_time
    ic=nothing;u0=nothing;v0=nothing;w0=nothing;GC.gc()
    viscous_dt=nu>0 ? .2/(nu*(dx^-2+(ny==1 ? 0. : dy^-2)+dz^-2)) : Inf
    effective_maxdt=min(maxdt,viscous_dt)
    sim=Simulation(model;Δt=min(effective_maxdt,.001),stop_time=stop,verbose=false)
    conjure_time_step_wizard!(sim,IterationInterval(1);cfl=.4,diffusive_cfl=.2,max_Δt=effective_maxdt)
    if !isempty(pickup)
        set!(sim;checkpoint=pickup)
        stop>time(sim) || error("STOP_TIME must exceed checkpoint time")
        archive_later_outputs(out,time(sim),iteration(sim))
        sim.Δt=min(sim.Δt,effective_maxdt)
    end
    ctx=(;output=out,zf,k,S,S_reference=Sref,waves,alpha,nu,maintain,gradient_bottom,gradient_top,
        baseline_gradient_bottom,baseline_gradient_top,wind_ustar=ustar,Lx=lx,Ly=ly,H=h,dx,dy,probe=mode)
    diagnostic_state=init_jet_diagnostics(model,ctx;resume=!isempty(pickup))
    regime_state=init_regime_diagnostics(diagnostic_state;resume=!isempty(pickup))
    moment_state=moments_enabled ? init_moment_diagnostics(regime_state;resume=!isempty(pickup)) : nothing
    wind_state=init_wind_diagnostics(diagnostic_state,regime_state;resume=!isempty(pickup))
    launch=time_ns();laststep=Ref(time(sim));largestdt=Ref(0.);lastsample=Ref(-Inf);lastdiagnostics=Ref{Any}(nothing)
    function steps(s)
        largestdt[]=max(largestdt[],time(s)-laststep[]);laststep[]=time(s)
    end
    sim.callbacks[:steps]=Callback(steps,IterationInterval(1))
    function snapshot(s)
        stem=@sprintf("t%012.6f",time(s));path=joinpath(out,stem*".json")
        isfile(path) && return basename(path)
        arrays=Dict{String,Any}()
        for (name,field) in pairs(model.velocities)
            a=Array(interior(field));all(isfinite,a)||error("Nonfinite $name snapshot")
            file=stem*"_"*string(name)*".bin";target=joinpath(out,file)
            isfile(target)&&error("Uncommitted snapshot file exists; refusing overwrite: $file")
            open(target*".tmp","w") do io;write(io,a);end
            mv(target*".tmp",target)
            location=name==:u ? ["Face","Center","Center"] : name==:v ? ["Center","Face","Center"] : ["Center","Center","Face"]
            arrays[string(name)]=(;file,shape=collect(size(a)),dtype="float64",order="F",location,sha256=filehash(target))
        end
        manifest=(;time=time(s),iteration=iteration(s),arrays)
        write_json(path,manifest)
        write_wind_snapshot_metadata(path,manifest,(;nx,ny,nz,lx,ly,h,k,Sref,alpha,uoffset,waves,nu,ustar,maintain,advection_name),sourcehashes)
        println("SNAPSHOT ",stem);flush(stdout)
        return basename(path)
    end
    probe_path=joinpath(out,"probe.csv")
    if mode!==nothing && !isfile(probe_path)
        write(probe_path,"time_s,probe_real,probe_imag\n")
    end
    last_probe=if mode!==nothing && length(readlines(probe_path))>1
        parse(Float64,first(split(last(readlines(probe_path)),',')))
    else
        -Inf
    end
    probe_last=Ref(last_probe)
    function diagnose(s)
        t=time(s);t>lastsample[]+1e-10 || return
        sample=sample_jet_diagnostics!(s,diagnostic_state)
        sample_regime_diagnostics!(s,regime_state,sample)
        sample_wind_diagnostics!(s,wind_state,sample)
        if moments_enabled && (abs(t/moment_interval-round(t/moment_interval))<1e-8 || abs(t-stop)<1e-8)
            sample_moment_diagnostics!(s,moment_state)
        end
        lastsample[]=t;lastdiagnostics[]=sample.scalars
        if mode!==nothing && t>probe_last[]+1e-10
            a=project_probe(sample.ymeans,mode,lx,h)
            open(probe_path,"a") do io;println(io,join((t,real(a),imag(a)),','));end
            probe_last[]=t
        end
        if mean_interval>0 && (abs(t/mean_interval-round(t/mean_interval))<1e-8 || abs(t-stop)<1e-8)
            stem=@sprintf("mean_%012.6f",t);manifest=joinpath(out,stem*".json")
            if !isfile(manifest)
                arrays=Dict{String,Any}()
                for (name,a) in regime_maps(regime_state)
                    file=stem*"_"*string(name)*".bin"
                    open(joinpath(out,file),"w") do io;write(io,a);end
                    arrays[string(name)]=(;file,shape=collect(size(a)),dtype="float64",order="F",sha256=filehash(joinpath(out,file)))
                end
                write_json(manifest,(;time=t,iteration=iteration(s),recommended_continuum_z_trim_cells=diagnostic_trim_cells,arrays))
            end
        end
        any(ts->isapprox(t,ts;atol=1e-8,rtol=0),snapshot_times) && snapshot(s)
        write_json(joinpath(out,"status.json"),(;time=t,iteration=iteration(s),dt=s.Δt,wall_seconds=(time_ns()-launch)/1e9))
        @printf("PROGRESS iter=%d t=%.8g dt=%.6g wall_s=%.2f\n",iteration(s),t,s.Δt,(time_ns()-launch)/1e9);flush(stdout)
    end
    snapshot_times=isempty(get(ENV,"SNAPSHOT_TIMES","")) ? [start_time,stop] : parse.(Float64,split(ENV["SNAPSHOT_TIMES"],','))
    snapshot_times=sort(unique(filter(t->start_time<=t<=stop,vcat(snapshot_times,[start_time,stop]))))
    times=sort(unique(vcat(collect(start_time:diagnostic_interval:stop),moments_enabled ? collect(start_time:moment_interval:stop) : Float64[],snapshot_times,[stop])))
    sim.callbacks[:diagnostics]=Callback(diagnose,SpecifiedTimes(times))
    checkpointer=Checkpointer(model;schedule=WallTimeInterval(checkpoint_wall>0 ? checkpoint_wall : 1e12),
        dir=joinpath(out,"checkpoints"),prefix="jet",overwrite_existing=false,cleanup=true,verbose=true)
    sim.output_writers[:checkpoint]=checkpointer
    if !isempty(pickup)
        sim.callbacks[:diagnostics].schedule.previous_actuation=searchsortedlast(times,time(sim)+1e-10)
    end
    isempty(pickup) && snapshot(sim)
    diagnose(sim)
    println("START ARCH=$architecture size=$((nx,ny,nz)) eta=$eta alpha=$alpha t=$(time(sim)) stop=$stop");flush(stdout)
    run!(sim);final_snapshot=snapshot(sim);diagnose(sim);close_jet_diagnostics!(diagnostic_state);close_regime_diagnostics!(regime_state);
    moments_enabled && close_moment_diagnostics!(moment_state);close_wind_diagnostics!(wind_state)
    checkpoint_path=joinpath(out,"checkpoints","jet_iteration"*string(iteration(sim))*".jld2")
    if keep_checkpoint
        isfile(checkpoint_path)||Oceananigans.OutputWriters.write_output!(checkpointer,sim)
    end
    # Final native fields are finite and fully written with hashes before a
    # completion receipt permits removal of this run's disposable checkpoints.
    write_json(joinpath(out,"completed.json"),(;time=time(sim),iteration=iteration(sim),wall_seconds=(time_ns()-launch)/1e9,
        max_dt=largestdt[],initial_condition_sha256=initialhash,diagnostics=lastdiagnostics[],final_snapshot,
        checkpoint_retained=keep_checkpoint,checkpoint_path=keep_checkpoint ? checkpoint_path : nothing))
    if !keep_checkpoint && isdir(joinpath(out,"checkpoints"))
        for file in readdir(joinpath(out,"checkpoints"))
            occursin(r"^jet_iteration[0-9]+\.jld2$",file) && rm(joinpath(out,"checkpoints",file))
        end
    end
    println("COMPLETE t=$(time(sim)) iterations=$(iteration(sim))");flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
