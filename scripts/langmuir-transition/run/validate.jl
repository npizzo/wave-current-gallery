include("driver.jl")
root=ARGS[1];mkpath(root)
VERSION==v"1.11.9"||error("Julia mismatch")
pkgversion(Oceananigans)==v"0.107.4"||error("Oceananigans mismatch")
CUDA.functional(true)||error("CUDA unavailable");CUDA.versioninfo()
small=joinpath(root,"small")
ic=generate_initial_condition(;case_name="1.D",level="T0",seed=1,arch=CPU(),FT=Float32,x_topology="periodic",amplitude="1.24,0.80,1.20",L_factor=1.15,precondition_time=.1,Δt=.02,root=small,overwrite=false)
errs=Dict();step_errors=Dict()
for member in ("steady_sheared_turbulence","sheared_control")
 a=integrate(joinpath(root,member*"-cpu"),small;member,arch=CPU(),tag=member*"cpu")
 b=integrate(joinpath(root,member*"-gpu"),small;member,arch=GPU(),tag=member*"gpu")
 err=maximum(norm(Float64.(getproperty(a,n)).-Float64.(getproperty(b,n)))/max(norm(Float64.(getproperty(a,n))),1e-12) for n in (:u,:v,:w))
 err<1e-4||error("CPU/GPU disagreement $err");errs[member]=err
 c=integrate(joinpath(root,member*"-halfdt"),small;member,arch=GPU(),dt=.005,tag=member*"halfdt")
 er=maximum(norm(Float64.(getproperty(c,n)).-Float64.(getproperty(b,n)))/max(norm(Float64.(getproperty(b,n))),1e-12) for n in (:u,:v,:w))
 er<.005||error("Small-grid timestep sensitivity $er");step_errors[member]=er
end
# Compare instrumented evolution with the original, unmodified upstream driver.
b=LAST[]
include(joinpath(ROOT,"upstream/experiments/anti_stokes/moving_packet_experiment.jl"))
sim,dir=run_member(;case_name="1.D",member="sheared_control",level="T0",seed=1,arch=GPU(),FT=Float32,
 x_topology="periodic",shear_amplitude=2.,Δt=.005,stop_time=.2,output_interval=.1,snapshot_offsets=(),root=small,overwrite=false,tag_extra="original")
a=native(sim)
hookerror=maximum(norm(Float64.(getproperty(a,n)).-Float64.(getproperty(b,n)))/max(norm(Float64.(getproperty(a,n))),1e-12) for n in (:u,:v,:w))
hookerror<1e-5||error("Instrumentation changed integration $hookerror")
include("member_instrumented.jl")
OUT[]=joinpath(root,"null")
sim,dir=run_member(;member="quiescent_control",level="T0",arch=GPU(),stop_time=.2,Δt=.01,snapshot_offsets=(),root=small,overwrite=false,tag_extra="null")
maximum(abs,LAST[].u)==0||error("Null flow moved")
# Generate once, then share this identical native IC across all four production runs.
GC.gc(true);CUDA.reclaim()
full=joinpath(ROOT,"data/ic")
ic=generate_initial_condition(;case_name="1.D",level="M2",seed=1,arch=GPU(),FT=Float32,x_topology="periodic",amplitude="1.24,0.80,1.20",L_factor=1.15,precondition_time=2.,Δt=.02,root=full,overwrite=false)
reports=Dict()
for member in ("steady_sheared_turbulence","sheared_control")
 GC.gc(true);CUDA.reclaim()
 a=integrate(joinpath(root,member*"-full"),full;member,level="M2",arch=GPU(),tag=member*"full",stop=.2)
 reports[member]=a.metrics
end
write_json_atomic(joinpath(root,"passed.json"),(;passed=true,cpu_gpu=errs,halfdt=step_errors,hookerror,full=reports,ic,ic_sha256=file_sha256(ic),gpu=CUDA.name(CUDA.device()),scope="Short backend, instrumentation, timestep, null and full-grid tests; not long-run convergence."))
println("VALIDATION PASSED")
