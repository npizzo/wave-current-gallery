# REVIEW-ONLY100Te driver. No default execution and no approval are implied.
# Physical state evolution is supplied by the separately reviewed DeepMechanism.
include("DeepMechanism.jl")
include("PVLayerBudgets.jl")
include("SpectralDiagnostics.jl")
using .DeepMechanism, Statistics, Printf, SHA, JLD2

if ARGS == ["--production100"]
    @eval using CUDA
end

const NX=2048; const NZ=1024; const LX=4.0; const H=2.0
const SEED=1; const NU=1e-10; const STOP_TE=100.0
const SAMPLE_TE=0.1; const FIELD_TE=1.0
const COMPONENT_TIMES=(0.0,10.0,25.0,50.0,100.0)
const MAX_DT=1.0; const CFL=0.5; const CASE_LIMIT_SECONDS=3600.0
const CASES=((true,:linear),(true,:ql),(true,:full),(false,:linear),(false,:ql),(false,:full))
const OLD_BENCHMARK_DIGEST="09b8cbf0e892655e349555cdfdcc6c39cf5ed2d9708418b898bbfa9f1c2669c5"
const SOURCE_FILES=("DeepMechanism.jl","PVLayerBudgets.jl","SpectralDiagnostics.jl","production.jl","production.sbatch")

keyvalues(path)=Dict(split(line,"=";limit=2) for line in readlines(path) if occursin("=",line))
source_digest()=bytes2hex(sha256(join((name*"\n"*read(joinpath(@__DIR__,name),String) for name in SOURCE_FILES),"\n")))

function require_approval()
    a=keyvalues(joinpath(@__DIR__,"approval.txt"))
    a["approved_scope"]=="two_QL_100Te_retries_only" || error("Wrong approval scope")
    a["approved_by"]=="user" || error("Explicit user approval is required")
    a["source_digest"]==source_digest() || error("Approval source digest mismatch")
    for (key,value) in (("nx","2048"),("nz","1024"),("seed","1"),("stop_Te","100"),
                        ("case_count","2"),("maximum_gpu_hours","2"),("maximum_concurrent_cases","2"),
                        ("wall_hours_per_case","1"))
        a[key]==value || error("Approval mismatch: $key")
    end
    validation=keyvalues(joinpath(@__DIR__,"validation-approved.txt"))
    validation["status"]=="PASS" && validation["source_digest"]==source_digest() || error("Current CPU diagnostic validation is required")
    for key in ("core_static","spectral_static","pv_static")
        validation[key]=="PASS" || error("Missing static gate: $key")
    end
    b=keyvalues(joinpath(@__DIR__,"prerequisites","benchmark-completed.txt"))
    b["status"]=="PASS" && b["case_count"]=="6" && b["stop_Te_each"]=="1" || error("Six-case benchmark prerequisite missing")
    b["source_digest"]==OLD_BENCHMARK_DIGEST || error("Unexpected benchmark source")
    compact=keyvalues(joinpath(@__DIR__,"prerequisites","benchmark-postrun-summary.txt"))
    compact["status"]=="PASS_COMPACT" && compact["failed_checks"]=="0" && compact["missing_items"]=="0" || error("Benchmark postrun verification missing")
    canonical=keyvalues(joinpath(@__DIR__,"prerequisites","initial-condition.txt"))
    original=keyvalues(joinpath(@__DIR__,"prerequisites","benchmark-initial-condition.txt"))
    canonical["sha256"]==original["sha256"]==a["initial_sha256"] || error("Canonical initial-state approval mismatch")
    canonical["u0"]==original["u0"] && canonical["Te"]==original["Te"] || error("Initial physical scales differ from benchmark")
    return a
end

function write_tree!(f,prefix,value)
    if value isa AbstractDict || value isa NamedTuple
        for (k,v) in pairs(value)
            write_tree!(f,isempty(prefix) ? string(k) : prefix*"/"*string(k),v)
        end
    else
        f[prefix]=value
    end
end

function gpu_preflight(path)
    fixtures=joinpath(@__DIR__,"prerequisites","static-fixtures")
    io=open(joinpath(path,"gpu-core-static.csv"),"w")
    println(io,"case,term,relative_error")
    largest=0.0
    for line in readlines(joinpath(fixtures,"cases.csv"))[2:end]
        label,snx,snz,_,_=split(line,",")
        nx,nz=parse(Int,snx),parse(Int,snz)
        g=DeepMechanism.Channel(nx,nz;backend=:gpu)
        input=read_pair(joinpath(fixtures,"$label.input.bin"),nx,nz)
        v=(backend_array(input[1],:gpu),backend_array(input[2],:gpu))
        op=operators(v,g); fullscale=native_norm(op.full)
        for arm in (:full,:ql,:linear)
            computed=project(getproperty(op,arm),g)[1]
            expected=read_pair(joinpath(fixtures,"$label.$arm.bin"),nx,nz)
            err=native_norm(minus((Array(computed[1]),Array(computed[2])),expected))/max(native_norm(expected),fullscale,1e-30)
            @printf(io,"%s,%s,%.17g\n",label,arm,err)
            err<1e-10 || error("GPU core gate failed: $arm")
            largest=max(largest,err)
        end
        for (term,computed) in (("wave",stokes_force(v,g)),("wave_adjoint",adjoint_stokes_force(v,g)),("diffusion",diffusion(v,g)))
            expected=read_pair(joinpath(fixtures,"$label.$term.bin"),nx,nz)
            err=native_norm(minus((Array(computed[1]),Array(computed[2])),expected))/max(native_norm(expected),1e-30)
            @printf(io,"%s,%s,%.17g\n",label,term,err)
            err<1e-10 || error("GPU core gate failed: $term")
            largest=max(largest,err)
        end
    end
    close(io)
    # New diagnostic helpers are exercised on static manufactured GPU arrays
    # before any call to a fluid time step. They must throw if validation fails.
    pv=PVLayerBudgets.validate_gpu_diagnostics()
    spectral=SpectralDiagnostics.validate_gpu_diagnostics()
    jldopen(joinpath(path,"gpu-diagnostic-preflight.jld2"),"w") do f
        write_tree!(f,"pv",pv); write_tree!(f,"spectral",spectral)
        f["core_maximum_relative_error"]=largest
        f["PDE_time_integration"]=false
    end
    println("GPU_ALL_STATIC_GATES_PASS core_maximum_relative_error=",largest)
end

is_field_time(tTe)=abs(tTe/FIELD_TE-round(tTe/FIELD_TE))<1e-9
is_component_time(tTe)=any(abs(tTe-x)<1e-9 for x in COMPONENT_TIMES)

function local_pv_identity_error(budget)
    # Use total-q magnitude, never the potentially zero mean PV or a tiny
    # observed change. Transport-reference remainders are not error gates.
    halfq2=max(maximum(abs,budget.old.profiles.half_q2),
               maximum(abs,budget.new.profiles.half_q2),1e-30)
    qrms=sqrt(2halfq2)
    worst=0.0
    for key in PVLayerBudgets.PROFILE_KEYS
        scale=key==:mean_q ? qrms : halfq2
        for prefix in ("component_update_residual_","component_component_sum_residual_")
            value=maximum(abs,budget.increments[prefix*string(key)])/scale
            value<1e-9 || error("Local exact PV identity failed: $(prefix)$(key), error=$value")
            worst=max(worst,value)
        end
    end
    return worst
end

function write_native!(path,v,c,g,t,Te)
    tTe=t/Te
    is_field_time(tTe) || return
    checkpoint=is_component_time(tTe)
    target=joinpath(path,"fields",@sprintf("t%06.1f.jld2",tTe))
    jldopen(target,"w") do f
        f["t"]=t;f["t_Te"]=tTe;f["u"]=Array(v[1]);f["w"]=Array(v[2])
        f["payload"]=checkpoint ? "analysis_checkpoint_full_components" : "native_velocities_only"
        if checkpoint
            f["q"]=Array(native_pv(v,g;waves=c.waves))
            f["rhs_u"]=Array(c.fields.total[1]);f["rhs_w"]=Array(c.fields.total[2])
            f["actual_q_rhs"]=Array(native_pv(c.fields.total,g;waves=false,source=true))
            for name in (:n,:s,:d,:gradp,:constraint)
                pair=getproperty(c.fields,name)
                f["$(name)_u"]=Array(pair[1]);f["$(name)_w"]=Array(pair[2])
                f["$(name)_q_rhs"]=Array(native_pv(pair,g;waves=false,source=true))
            end
            f["pressure_component_sign"]="actual pressure tendency is minus gradp"
        end
        f["source_digest"]=source_digest()
    end
end

function sample!(path,streams,v,g,arm,waves,t,Te,iteration,dt,online,pvacc)
    c=components(v,g,arm;waves,nu=NU)
    E=native_inner(v,v)/2
    U=vec(Array(mean(v[1];dims=1)))
    R=vec(Array(mean(DeepMechanism.u_at_z_faces(v[1]).*DeepMechanism.w_at_x_faces(v[2]);dims=1)))
    divrel=maximum(abs,native_divergence(v,g))*min(g.dx,g.dz)/max(sqrt(2E),1e-30)
    isfinite(E) && all(isfinite,v[1]) && all(isfinite,v[2]) || error("Non-finite state")
    divrel<1e-9 || error("Divergence gate failed")
    work=Dict(name=>native_inner(v,getproperty(c,name)) for name in (:n,:s,:d,:gradp,:constraint,:total))
    @printf(streams.ts,"%.17g,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
        t/Te,iteration,dt,E,mean(abs2,U)/2,mean(U),divrel,work[:n],work[:s],work[:d],-work[:gradp],work[:constraint],work[:total])
    means=Dict(name=>vec(Array(mean(getproperty(c,name)[1];dims=1))) for name in (:n,:s,:d,:gradp,:constraint,:total))
    for k in 1:g.nz
        @printf(streams.momentum,"%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
            t/Te,-g.h+(k-0.5)*g.dz,U[k],R[k],R[k+1],means[:n][k],means[:s][k],means[:d][k],-means[:gradp][k],means[:constraint][k],means[:total][k])
    end
    q=native_pv(v,g;waves); qt=native_pv(c.total,g;waves=false,source=true)
    Q=mean(q;dims=1)
    Qhost=vec(Array(Q)); qvar=vec(Array(mean((q.-Q).^2;dims=1)))
    qflux=vec(Array(mean(DeepMechanism.w_at_x_faces(v[2]).*(q.-Q);dims=1)))
    qtr=vec(Array(mean(qt;dims=1)))
    for k in 1:g.nz+1
        @printf(streams.pv,"%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",t/Te,-g.h+(k-1)*g.dz,Qhost[k],qvar[k],qflux[k],qtr[k])
    end
    accumulated=Array(online.mean); observed=U.-online.initial_U
    meanres=observed.-vec(accumulated[end,:])
    Eobserved=E-online.initial_E; PVobserved=pv_variance(q)-online.initial_PV
    Eres=Eobserved-online.energy[end]; PVres=PVobserved-online.pv[end]
    maximum(abs,meanres)/online.u0<1e-9 || error("Online mean momentum closure failed")
    abs(Eres)/online.initial_E<1e-9 || error("Online discrete energy closure failed")
    abs(PVres)/max(online.initial_PV,1e-30)<1e-9 || error("Online global PVvariance closure failed")
    for k in 1:g.nz
        @printf(streams.online_mean,"%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
            t/Te,-g.h+(k-0.5)*g.dz,observed[k],accumulated[1,k],accumulated[2,k],accumulated[3,k],accumulated[4,k],accumulated[5,k],accumulated[6,k],meanres[k])
    end
    for (i,name) in enumerate(DeepMechanism.BUDGET_TERMS)
        @printf(streams.quadratic,"%.17g,%s,%.17g,%.17g\n",t/Te,name,online.energy[i],online.pv[i])
    end
    @printf(streams.closure,"%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
        t/Te,Eobserved,online.energy[end],Eres,PVobserved,online.pv[end],PVres,maximum(abs,meanres))
    foreach(flush,values(streams))

    # Full FFTs remain on the device; only selected coefficients/profiles cross
    # to the host. Dense diagnostics do not copy native full fields to the CPU.
    spectra=SpectralDiagnostics.diagnostic_sample(v,g,arm;waves,nu=NU,components=c,full_stress=is_field_time(t/Te))
    pvdata=isnothing(pvacc[]) ? PVLayerBudgets.pv_profile_state(v,g;waves,host=true) : PVLayerBudgets.snapshot(pvacc[];host=true)
    diagpath=joinpath(path,"diagnostics",@sprintf("t%06.1f.jld2",t/Te))
    jldopen(diagpath,"w") do f
        f["t"]=t;f["t_Te"]=t/Te;f["source_digest"]=source_digest()
        SpectralDiagnostics.write_diagnostics!(f,spectra)
        write_tree!(f,t==0 ? "initial_PV_state" : "PV_interval_budget",pvdata)
        f["online_RK_mean_increments"]=accumulated
        f["online_discrete_energy_increments"]=online.energy
        f["online_discrete_PV_variance_increments"]=online.pv
        f["online_terms"]=string.(DeepMechanism.BUDGET_TERMS)
        f["checks/local_PV_identity_maximum_so_far"]=online.pv_local_max[]
        f["checks/local_PV_identity_normalization"]="mean_q uses total qRMS; all quadratic densities use total half_q2; maxima over old/new depth profiles; no division by meanPV or observed change"
        f["PV_budget_time_convention"]="PV budgets are interval increments since the preceding dense output; global RK arrays are cumulative from t=0"
    end
    isnothing(pvacc[]) || PVLayerBudgets.reset!(pvacc[])
    write_native!(path,v,(;fields=c,waves),g,t,Te)
    return (;E,momentum=mean(U),total_work=work[:total],divrel,Eres,PVres,maxmeanres=maximum(abs,meanres))
end

function open_streams(path)
    definitions=(ts=("timeseries.csv","t_Te,iteration,dt,E,Emean,momentum,divergence_relative,advection_work,wave_work,viscosity_work,pressure_work,constraint_work,total_rhs_work"),
        momentum=("momentum-profiles.csv","t_Te,z,U,R_lower,R_upper,advection_u,wave_u,viscosity_u,pressure_u,constraint_u,total_rhs_u"),
        pv=("pv-profiles.csv","t_Te,z,Q,q_variance,q_flux,actual_Q_rhs"),
        online_mean=("online-RK-momentum.csv","t_Te,z,observed_delta_U,RK_advection,RK_wave,RK_viscosity,RK_pressure,RK_constraint,RK_total,residual"),
        quadratic=("online-discrete-quadratic-budgets.csv","t_Te,component,discrete_energy_increment,discrete_global_PV_variance_increment"),
        closure=("online-closure.csv","t_Te,observed_delta_E,allocated_delta_E,E_residual,observed_delta_PVvariance,allocated_delta_PVvariance,PVvariance_residual,maximum_U_residual"))
    return (;(key=>begin io=open(joinpath(path,value[1]),"w");println(io,value[2]);io end for (key,value) in pairs(definitions))...)
end

function run_case(task_id)
    require_approval()
    task_id in (1,4) || error("Only original QL task IDs1 and4 are authorized")
    CUDA.functional() || error("CUDA is not functional")
    CUDA.allowscalar(false)
    waves,arm=CASES[task_id+1]
    name="$(waves ? "deep" : "nowave")-$(arm)-s1"
    path=joinpath(@__DIR__,"output",name)
    ispath(path) && error("Refusing to overwrite existing case output")
    mkpath(path);mkpath(joinpath(path,"fields"));mkpath(joinpath(path,"diagnostics"))
    started=time_ns()
    gpu_preflight(path)
    initial_path=joinpath(@__DIR__,"prerequisites","initial-state.bin")
    initial_record=keyvalues(joinpath(@__DIR__,"prerequisites","initial-condition.txt"))
    initial_sha=bytes2hex(sha256(read(initial_path)))
    initial_sha==initial_record["sha256"] || error("Canonical initial-state file hash mismatch")
    initial=read_pair(initial_path,NX,NZ)
    meta=(u0=parse(Float64,initial_record["u0"]),Te=parse(Float64,initial_record["Te"]))
    isapprox(meta.u0,0.5/((1/9)*(2π*27)^2);rtol=1e-14) || error("Initial velocity scale mismatch")
    isapprox(meta.Te,1/(meta.u0*2π*27);rtol=1e-14) || error("Eddy-time scale mismatch")
    isapprox(native_rms(initial),meta.u0;rtol=1e-12) || error("Canonical initial rms mismatch")
    g=DeepMechanism.Channel(NX,NZ;lx=LX,h=H,backend=:gpu)
    v=(backend_array(initial[1],:gpu),backend_array(initial[2],:gpu))
    maximum(abs,native_divergence(v,g))*min(g.dx,g.dz)/meta.u0<1e-10 || error("Initial divergence failed")
    streams=open_streams(path)
    online=(;mean=backend_array(zeros(6,g.nz),:gpu),energy=zeros(6),pv=zeros(6),
             initial_U=vec(Array(mean(v[1];dims=1))),initial_E=native_inner(v,v)/2,
             initial_PV=pv_variance(native_pv(v,g;waves)),u0=meta.u0,pv_local_max=Ref(0.0))
    pvacc=Ref{Any}(nothing)
    callback=function(;old,new,increments,stages,qold,qnew,dt,waves,arm,g)
        PVLayerBudgets.pv_layer_budget(old,new,increments,g;
            dt,waves,stages,qold,qnew,model_arm=arm)
    end
    open(joinpath(path,"meta.txt"),"w") do io
        for (k,value) in pairs((;name,task_id,nx=NX,nz=NZ,lx=LX,h=H,seed=SEED,arm,waves,nu=NU,
            stop_Te=STOP_TE,sample_Te=SAMPLE_TE,field_Te=FIELD_TE,max_dt=MAX_DT,cfl=CFL,
            u0=meta.u0,Te=meta.Te,k0_cycles=27,initial_UL=0,initial_sha256=initial_sha,source_digest=source_digest()))
            println(io,k,"=",value)
        end
        println(io,"Stokes_shear=0.25exp(8z) in deep cases; zero in controls")
        println(io,"Stokes_stencil=archived production receiving-node shear; discrete wave work retained")
        println(io,"component_checkpoint_Te=0,10,25,50,100")
        println(io,"wall_limit_seconds=3600")
        println(io,"resolution_quality=One-Te benchmark does not establish long-run grid-scale resolution; native full fields are retained for2D grid-tail and convergence assessment before mechanism claims")
        println(io,"local_PV_identity_gate=update and component-sum residuals only; total-qRMS/half_q2 normalization; tolerance1e-9; physical-flux remainders are not error gates")
        println(io,"automatic_continuation=false")
    end
    t=0.0;iteration=0;dt=0.0
    previous=sample!(path,streams,v,g,arm,waves,t,meta.Te,iteration,dt,online,pvacc)
    initial_record=previous; quadrature_work=0.0
    for target in collect(SAMPLE_TE:SAMPLE_TE:STOP_TE).*meta.Te
        t0=t
        while t<target-1e-12*meta.Te
            (time_ns()-started)/1e9<CASE_LIMIT_SECONDS || error("One-hour case limit reached")
            advective_rate=maximum(abs,v[1])/g.dx+maximum(abs,v[2])/g.dz
            dt=min(MAX_DT,CFL/max(advective_rate,1e-30),target-t)
            isfinite(dt) && dt>0 || error("Invalid timestep")
            v,stepbudget=rk3_step_with_budget(v,dt,g,arm;waves,nu=NU,budget_callback=callback)
            stepbudget.mean_residual/meta.u0<1e-10 || error("RK mean-increment identity failed")
            abs(stepbudget.energy_residual)/online.initial_E<1e-10 || error("RK energy identity failed")
            abs(stepbudget.pv_residual)/max(online.initial_PV,1e-30)<1e-10 || error("RK PVvariance identity failed")
            online.mean .+=stepbudget.mean_increments
            online.energy .+=stepbudget.energy_increments
            online.pv .+=stepbudget.pv_increments
            online.pv_local_max[]=max(online.pv_local_max[],local_pv_identity_error(stepbudget.auxiliary))
            if isnothing(pvacc[])
                pvacc[]=PVLayerBudgets.init_accumulator(stepbudget.auxiliary)
            end
            PVLayerBudgets.accumulate!(pvacc[],stepbudget.auxiliary)
            t+=dt;iteration+=1
        end
        t=target
        now=sample!(path,streams,v,g,arm,waves,t,meta.Te,iteration,dt,online,pvacc)
        quadrature_work+=(t-t0)*(previous.total_work+now.total_work)/2
        abs(now.momentum-initial_record.momentum)/meta.u0<1e-8 || error("Integrated momentum drift")
        if is_field_time(t/meta.Te)
            @printf("PILOT %s t=%.1fTe steps=%d E/E0=%.8f meanres/u0=%.3e div=%.3e wall=%.2fs\n",name,t/meta.Te,iteration,now.E/initial_record.E,now.maxmeanres/meta.u0,now.divrel,(time_ns()-started)/1e9)
            flush(stdout)
        end
        previous=now
    end
    foreach(close,values(streams));synchronize_backend(:gpu)
    elapsed=(time_ns()-started)/1e9
    elapsed<=CASE_LIMIT_SECONDS || error("Completed case exceeded one-hour limit")
    open(joinpath(path,"completed.txt"),"w") do io
        println(io,"status=PASS");println(io,"stop_Te=",t/meta.Te);println(io,"iterations=",iteration)
        println(io,"wall_seconds=",elapsed);println(io,"source_digest=",source_digest())
        println(io,"online_E_residual=",previous.Eres);println(io,"online_PVvariance_residual=",previous.PVres)
        println(io,"online_mean_residual=",previous.maxmeanres)
        println(io,"maximum_local_PV_identity_relative_residual=",online.pv_local_max[])
        println(io,"snapshot_quadrature_energy_residual=",abs(previous.E-initial_record.E-quadrature_work)/initial_record.E)
        println(io,"automatic_continuation=false")
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    if ARGS==["--source-digest"]
        println(source_digest())
    elseif ARGS==["--production100"]
        get(ENV,"SLURM_ARRAY_TASK_COUNT","")=="2" || error("Expected exactly two QL array tasks")
        run_case(parse(Int,ENV["SLURM_ARRAY_TASK_ID"]))
    else
        error("No default execution; use --source-digest for review or --production100 only after approval")
    end
end
