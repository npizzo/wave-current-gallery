using CUDA, Oceananigans, JLD2, Statistics, LinearAlgebra, SHA, Dates
const ROOT=normpath(joinpath(@__DIR__,".."))
include(joinpath(ROOT,"upstream/experiments/anti_stokes/generate_turbulence.jl"))
include("packet_checks.jl")
using .PacketChecks
const OUT=Ref("")
const LAST=Ref{Any}(nothing)
const SNAPSHOTS=[0.,10.,30.,60.,120.,200.,400.,600.]

function binary(path,a)
    a=Array(a);open(path,"w") do io;write(io,a);end
    return (;file=basename(path),shape=collect(size(a)),dtype=eltype(a)==Float32 ? "float32" : "float64",order="F",sha256=bytes2hex(sha256(read(path))))
end
function native(sim)
    u,v,w=(Array(interior(f)) for f in sim.model.velocities)
    zf=Float64.(Array(znodes(sim.model.grid,Face())))
    m=PacketChecks.native_state_metrics(u,v,w,12.,.8,diff(zf);bounded_x=false)
    PacketChecks.check_state(m)
    return (;u,v,w,metrics=m)
end
function campaign_run!(sim,dir,case,member)
    out=OUT[];mkpath(out);empty!(sim.output_writers)
    g=sim.model.grid;nx,ny,nz=g.Nx,g.Ny,g.Nz
    zc=Float64.(Array(znodes(g,Center())));zf=Float64.(Array(znodes(g,Face())))
    u,v,w=sim.model.velocities
    uc=Field(@at (Center,Center,Center) u);vc=Field(@at (Center,Center,Center) v);wc=Field(@at (Center,Center,Center) w)
    fields=(;U=Field(Average(uc,dims=2)),V=Field(Average(vc,dims=2)),W=Field(Average(wc,dims=2)),
      UU=Field(Average(uc*uc,dims=2)),VV=Field(Average(vc*vc,dims=2)),WW=Field(Average(wc*wc,dims=2)),UW=Field(Average(uc*wc,dims=2)))
    # Exact transverse means of native face velocities enable a staggered continuity check.
    facefields=(;Un=Field(Average(u,dims=2)),Wn=Field(Average(w,dims=2)))
    write_json_atomic(joinpath(out,"grid.json"),(;nx,ny,nz,Lx=12.,Ly=.8,H=.4,zc,zf,k=Float64(case.k),S=Float64(case.Uˢ₀),member,dt=sim.Δt))
    times=sort(unique(vcat(collect(0.:1.:min(60.,sim.stop_time)),collect(0.:5.:sim.stop_time),[sim.stop_time])))
    last=Ref(-Inf);started=time_ns()
    function sample(s)
        t=Float64(time(s));t<=last[]+1e-9&&return;last[]=t
        compute!(uc);compute!(vc);compute!(wc)
        arrays=Dict{String,Any}();stem=@sprintf("mean_%010.3f",t)
        for (name,f) in pairs(merge(fields,facefields))
            a=Array(interior(compute!(f)));all(isfinite,a)||error("Nonfinite $name")
            arrays[string(name)]=binary(joinpath(out,stem*"_"*string(name)*".bin"),a)
        end
        maxs=Float64.([maximum(abs,f) for f in sim.model.velocities])
        cfl=s.Δt*(maxs[1]/(12/nx)+maxs[2]/(.8/ny)+maxs[3]/minimum(diff(zf)))
        cfl<.9||error("Conservative advective CFL exceeds .9: $cfl")
        write_json_atomic(joinpath(out,stem*".json"),(;time=t,iteration=iteration(s),arrays,cfl,wall_seconds=(time_ns()-started)/1e9))
        if any(isapprox(t,q;atol=1e-7) for q in SNAPSHOTS)||isapprox(t,s.stop_time;atol=1e-7)
            state=native(s);d=Dict{String,Any}()
            for name in (:u,:v,:w);d[string(name)]=binary(joinpath(out,@sprintf("native_%010.3f_%s.bin",t,name)),getproperty(state,name));end
            write_json_atomic(joinpath(out,@sprintf("native_%010.3f.json",t)),(;time=t,iteration=iteration(s),arrays=d,metrics=state.metrics))
        end
        status_tmp=joinpath(out,"status-next.json")
        write_json_atomic(status_tmp,(;time=t,iteration=iteration(s),cfl,wall_seconds=(time_ns()-started)/1e9))
        mv(status_tmp,joinpath(out,"status.json");force=true)
        @printf("SAMPLE t=%.3f CFL=%.4f wall=%.1fs\n",t,cfl,(time_ns()-started)/1e9);flush(stdout)
    end
    sim.callbacks[:campaign]=Callback(sample,SpecifiedTimes(times));sample(sim)
    run!(sim);sample(sim)
    LAST[]=native(sim)
    write_json_atomic(joinpath(out,"integrated.json"),(;time=Float64(time(sim)),iteration=iteration(sim),wall_seconds=(time_ns()-started)/1e9,metrics=LAST[].metrics))
end
include("member_instrumented.jl")
function integrate(out,icroot;member="steady_sheared_turbulence",alpha=2.,level="T0",arch=GPU(),stop=.2,dt=.01,tag="test")
    OUT[]=out
    run_member(;case_name="1.D",member,level,seed=1,arch,FT=Float32,x_topology="periodic",shear_amplitude=alpha,
        Δt=dt,stop_time=stop,output_interval=.1,snapshot_offsets=(),root=icroot,overwrite=false,tag_extra=tag,progress_interval=1000)
    return LAST[]
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS) in (5,6)||error("out icroot member alpha tag [dt]")
    dt=length(ARGS)==6 ? parse(Float64,ARGS[6]) : .01
    integrate(ARGS[1],ARGS[2];member=ARGS[3],alpha=parse(Float64,ARGS[4]),tag=ARGS[5],level="M2",stop=600.,dt)
end
