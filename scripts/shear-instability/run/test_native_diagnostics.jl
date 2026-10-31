using Test, Oceananigans, Statistics
include("jet_diagnostics.jl")
using .JetDiagnostics

const HTEST=.5
const KTEST=4.
const STEST=.04
const ATEST=.75
const LXTEST=2.
const LYTEST=1.
physical_function(f,ny)=ny==1 ? ((x,z)->f(x,0.,z)) : f

function test_model(;n=12,ny=8,advection=Centered(order=2),background=false)
    grid=ny==1 ? RectilinearGrid(CPU();size=(n,n),x=(0,LXTEST),z=(-HTEST,0),halo=(4,4),topology=(Periodic,Flat,Bounded)) :
                 RectilinearGrid(CPU();size=(n,ny,n),x=(0,LXTEST),y=(0,LYTEST),z=(-HTEST,0),halo=(4,4,4),topology=(Periodic,Periodic,Bounded))
    sd=UniformStokesDrift(∂z_uˢ=(z,t)->2KTEST*STEST*exp(2KTEST*z))
    ubc=FieldBoundaryConditions(bottom=GradientBoundaryCondition((1-ATEST)*2KTEST*STEST*exp(-2KTEST*HTEST)),
                               top=GradientBoundaryCondition((1-ATEST)*2KTEST*STEST))
    model=NonhydrostaticModel(grid;advection,timestepper=:RungeKutta3,stokes_drift=sd,
                             boundary_conditions=(u=ubc,),closure=nothing)
    base(z)=(1-ATEST)*STEST*exp(2KTEST*z)
    if background
        set!(model;u=physical_function((x,y,z)->base(z),ny),v=0,w=0)
    else
        # A projected smooth three-dimensional perturbation for native tests.
        set!(model;u=physical_function((x,y,z)->base(z)+.04sin(2pi*y/LYTEST)*cos(pi*(z+HTEST)/HTEST)+.03cos(2pi*x/LXTEST)*sin(pi*(z+HTEST)/HTEST),ny),
                   v=physical_function((x,y,z)->.02cos(2pi*x/LXTEST)*cos(pi*(z+HTEST)/HTEST),ny),
                   w=physical_function((x,y,z)->.04sin(2pi*x/LXTEST)*sin(pi*(z+HTEST)/HTEST)+.03sin(2pi*y/LYTEST)*sin(pi*(z+HTEST)/HTEST),ny))
    end
    return model
end

context(model,dir)=(;output=dir,zf=collect(range(-HTEST,0;length=model.grid.Nz+1)),
    k=KTEST,S=STEST,alpha=ATEST,Lx=LXTEST,Ly=LYTEST,H=HTEST)

@testset "Conservative centred operator and independently measured covariance" begin
    mktempdir() do dir
        model=test_model(); state=init_jet_diagnostics(model,context(model,dir))
        try
            a=native_sample(state); p=a.profiles; f=a.flux
            @test f.decomposition_max<5e-17
            @test maximum(abs.(p["adv_full"].-p["continuum_flux"]))<2e-14
            @test maximum(abs.(p["adv_3d"].-p["continuum_3d_flux"]))<2e-14
            @test abs(sum(p["adv_full"].*state.dz))<1e-15
            @test maximum(abs,p["pressure"])<2e-14
            @test size(a.ymeans.U)==(12,12)
            @test size(a.ymeans.W)==(12,13)
            @test length(f.total)==13
            @test maximum(abs,f.three_d)>1e-7
            @test all(isfinite,vcat(values(p)...))
            before=map(v->copy(Array(interior(v))),model.velocities)
            native_sample(state)
            @test all(a==Array(interior(b)) for (a,b) in zip(before,model.velocities))
        finally
            close_jet_diagnostics!(state)
        end
    end
end

@testset "Two-dimensional three-component flow has no native 3-D correction" begin
    for ny in (1,8), advection in (Centered(order=2),WENO(order=5))
      mktempdir() do dir
        model=test_model(;ny,advection)
        set!(model;u=physical_function((x,y,z)->.01exp(8z)+.03cos(pi*x)*sin(2pi*(z+HTEST)),ny),
                   v=physical_function((x,y,z)->.04sin(pi*x)*cos(2pi*(z+HTEST)),ny),
                   w=physical_function((x,y,z)->.03sin(pi*x)*sin(2pi*(z+HTEST)),ny))
        state=init_jet_diagnostics(model,context(model,dir))
        try
            a=native_sample(state)
            @test maximum(abs,a.profiles["adv_3d"])<1e-14
            @test maximum(abs,a.flux.three_d)<1e-28
            @test maximum(abs,a.ymeans.V)>1e-3
            @test a.flux.decomposition_max<1e-16
        finally
            close_jet_diagnostics!(state)
        end
      end
    end
end

@testset "Steady prescribed-shear equilibrium and diagnostic writer" begin
    mktempdir() do dir
        model=test_model(;advection=WENO(order=5),background=true)
        state=init_jet_diagnostics(model,context(model,dir))
        simulation=Simulation(model;Δt=.01,stop_time=.04,verbose=false)
        initial=sample_jet_diagnostics!(simulation,state)
        @test initial.scalars["Ejet_L"]>0
        @test initial.scalars["Ejet_E"]>initial.scalars["Ejet_L"]
        @test initial.scalars["Ejet_anomaly"]==0
        @test maximum(abs,initial.profiles["total"])<1e-14
        @test maximum(abs,initial.flux.total)==0
        simulation.callbacks[:jets]=Callback(s->sample_jet_diagnostics!(s,state),IterationInterval(1))
        run!(simulation)
        final=state.previous
        @test abs(final["Ejet_L"]-initial.scalars["Ejet_L"])<1e-14
        @test final["Ejet_anomaly"]<1e-24
        @test abs(final["cumulative_residual_L"])<1e-14
        @test final["pressure_mean_max"]<1e-14
        close_jet_diagnostics!(state)
        lines=readlines(joinpath(dir,"jet-budget.csv"))
        @test length(lines)==6 # Header + initial + four distinct steps.
        @test split(lines[1],',')==JET_BUDGET_SCHEMA
        @test all(length(split(line,','))==length(JET_BUDGET_SCHEMA) for line in lines)
        @test length(readlines(joinpath(dir,"jet-profiles.csv")))==1+5*12
        @test length(readlines(joinpath(dir,"jet-fluxes.csv")))==1+5*13
    end
end

function run_energy_check(dir,dt)
    model=test_model(;advection=WENO(order=5))
    state=init_jet_diagnostics(model,context(model,dir))
    simulation=Simulation(model;Δt=dt,stop_time=.16,verbose=false)
    sample_jet_diagnostics!(simulation,state)
    simulation.callbacks[:jets]=Callback(s->sample_jet_diagnostics!(s,state),IterationInterval(1))
    run!(simulation)
    result=copy(state.previous)
    close_jet_diagnostics!(state)
    return result
end

@testset "Native mean-energy budget improves under smaller time steps" begin
    mktempdir() do dir
        coarse=run_energy_check(joinpath(dir,"coarse"),.02)
        fine=run_energy_check(joinpath(dir,"fine"),.01)
        a,b=abs(coarse["cumulative_residual_L"]),abs(fine["cumulative_residual_L"])
        println("Native jet-energy convergence: coarse=$a fine=$b ratio=",a/b)
        @test b<.7a+2e-14
        @test fine["flux_decomposition_max"]<1e-16
        @test fine["native_advection_decomposition_max"]<1e-18
        @test fine["pressure_mean_max"]<1e-14
        @test isapprox(fine["P_total_L"],fine["P_adv_ky0_L"]+fine["P_adv_3d_L"]+fine["P_wave_L"]+fine["P_pressure_L"]+fine["P_closure_L"];atol=1e-18)
    end
end

@testset "Restart recovers initial profile and cumulative budget" begin
    mktempdir() do dir
        model=test_model(;background=true)
        state=init_jet_diagnostics(model,context(model,dir)); sim=Simulation(model;Δt=.01,stop_time=.03,verbose=false)
        sample_jet_diagnostics!(sim,state)
        model.clock.time=.01; sample_jet_diagnostics!(sim,state)
        at_checkpoint=copy(state.previous); initial=copy(state.initial_UL)
        model.clock.time=.02; sample_jet_diagnostics!(sim,state)
        close_jet_diagnostics!(state)
        model.clock.time=.01
        restored=init_jet_diagnostics(model,context(model,dir);resume=true)
        @test restored.initial_UL==initial
        @test isequal(restored.previous,at_checkpoint)
        @test restored.cumulative["L"]==at_checkpoint["cumulative_P_total_L"]
        @test length(readlines(joinpath(dir,"jet-budget.csv")))==3
        @test any(startswith("diagnostics-pre-restart-"),readdir(dir))
        # Re-sampling the checkpoint is deliberately idempotent.
        sample_jet_diagnostics!(sim,restored)
        @test length(readlines(joinpath(dir,"jet-budget.csv")))==3
        model.clock.time=.03; sample_jet_diagnostics!(sim,restored)
        @test restored.previous["interval_dt"]≈.02
        @test restored.previous["Ejet_anomaly"]==0
        close_jet_diagnostics!(restored)
        @test length(readlines(joinpath(dir,"jet-budget.csv")))==4
    end
end

@testset "Finite-output-interval budget is an explicitly approximate quadrature" begin
    function error_for(h)
        cumulative=0.; previous=0.
        for t in h:h:1
            b=finite_interval_budget(t,t^3,3t^2,previous,previous^3,3previous^2,cumulative,0.)
            cumulative=b.cumulative; previous=t
        end
        abs(1-cumulative)
    end
    @test error_for(.1)/error_for(.05)≈4 atol=1e-10
    @test_throws ArgumentError finite_interval_budget(0.,0.,0.,0.,0.,0.,0.,0.)
end
