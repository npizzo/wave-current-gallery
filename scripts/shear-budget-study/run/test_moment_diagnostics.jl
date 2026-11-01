using Test, Oceananigans, Statistics
include("initial_conditions.jl")
include("regime_physics.jl")
include("jet_diagnostics.jl")
using .JetDiagnostics

const RH=.5; const RLX=2.; const RLY=1.; const RK=6.1; const RS=.17^2*sqrt(9.81/RK)

function regime_test_model(;n=16,ny=8,alpha=.75,waves=1,nu=1e-5,maintain=true,arch=CPU())
    grid=ny==1 ? RectilinearGrid(arch;size=(n,n),x=(0,RLX),z=(-RH,0),halo=(3,3),topology=(Periodic,Flat,Bounded)) :
        RectilinearGrid(arch;size=(n,ny,n),x=(0,RLX),y=(0,RLY),z=(-RH,0),halo=(3,3,3),topology=(Periodic,Periodic,Bounded))
    p=regime_forcing_parameters(;waves,alpha,S=RS,k=RK,h=RH,nz=n,nu)
    forcing=maintain ? (;u=Forcing(maintained_background_forcing;discrete_form=true,parameters=p)) : NamedTuple()
    bcs=FieldBoundaryConditions(bottom=GradientBoundaryCondition((waves-alpha)*2RK*RS*exp(-2RK*RH)),
        top=GradientBoundaryCondition((waves-alpha)*2RK*RS))
    sd=waves==1 ? UniformStokesDrift(∂z_uˢ=(z,t)->2RK*RS*exp(2RK*z)) : nothing
    m=NonhydrostaticModel(grid;advection=Centered(order=2),closure=ScalarDiffusivity(ν=nu),
        timestepper=:RungeKutta3,forcing,stokes_drift=sd,boundary_conditions=(u=bcs,))
    return m
end

function regime_context(m,dir;alpha=.75,waves=1,nu=1e-5)
    (;output=dir,zf=collect(range(-RH,0;length=m.grid.Nz+1)),k=RK,S=waves*RS,
      S_reference=RS,alpha,waves,nu,Lx=RLX,Ly=RLY,H=RH)
end


function test_moment_case(arch;eta=.1,threec=false)
    mktempdir() do dir
        m=regime_test_model(;arch)
        ic=initial_velocity_fields(;nx=16,ny=8,nz=16,lx=RLX,ly=RLY,h=RH,k=RK,S_reference=RS,
            waves=1,alpha=.75,uoffset=0.,eddy_rms=.004,injection_wavelength=.5,perturb_energy_ratio=eta,seed=3)
        if threec
            ic.v .+= [0.004cos(2pi*(i-.5)/16)*cos(pi*(k-.5)/16) for i in 1:16,j in 1:8,k in 1:16]
        end
        set!(m;u=ic.u,v=ic.v,w=ic.w)
        j=init_jet_diagnostics(m,regime_context(m,dir))
        r=init_regime_diagnostics(j); d=init_moment_diagnostics(r)
        sim=Simulation(m;Δt=.001,stop_time=.002,verbose=false)
        a=sample_jet_diagnostics!(sim,j);sample_regime_diagnostics!(sim,r,a)
        row=sample_moment_diagnostics!(sim,d)
        @test row["Zq"]≈row["Zmean"]+row["Zvariance"] rtol=2e-13
        @test row["Za"]≈row["Zmean_anomaly"]+row["Zvariance"] rtol=2e-13
        @test row["Pq_T"]≈row["Pq_stretch"]+row["Pq_tilt_wy_vx"]+row["Pq_tilt_minus_uy_vz"] atol=1e-18
        @test row["Pq_T_abs"]+1e-20>=abs(row["Pq_T"])
        @test row["T_unsimplified_error2"]<1e-28
        for family in JetDiagnostics.MOMENT_FAMILIES
            @test row["d$(family)_native"]≈sum(row["$(family)_$term"] for term in ("inviscid","viscous","forcing","pressure")) atol=1e-17
        end
        @test abs(row["native_partition_error"])<1e-17
        @test abs(row["anomaly_partition_error"])<1e-17
        _,profiles=JetDiagnostics.read_numeric_csv(joinpath(dir,"moment-profiles.csv"),MOMENT_PROFILE_SCHEMA)
        for name in MOMENT_FIELDS
            @test row[name]≈sum(p[name]*p["dz_m"] for p in profiles)/RH atol=1e-17
        end
        for p in profiles, side in ("lower","upper")
            @test p["flux_q_$side"]≈p["flux_mean_$side"]+p["flux_variance_$side"] atol=1e-18
            @test p["flux_a_$side"]≈p["flux_mean_anomaly_$side"]+p["flux_variance_$side"] atol=1e-18
        end
        for f in JetDiagnostics.MOMENT_FLUX_NAMES
            @test first(profiles)["flux_$(f)_lower"]==0
            @test last(profiles)["flux_$(f)_upper"]==0
        end
        if eta==0
            @test row["T_rms"]<1e-18
            @test row["Zvariance"]<1e-30
            @test abs(row["Pmean_feedback_native"])<1e-19
            threec && @test row["v_rms"]>1e-3
        end
        # Independent semi-discrete chain-rule check: perturb native u,w using
        # freshly evaluated native tendencies, then recurl velocities. No RK
        # stages or pre-averaged native tendencies enter this verification.
        n=j.operators.native
        u,w=m.velocities.u,m.velocities.w
        ub=copy(interior(u));wb=copy(interior(w))
        du=copy(interior(n.U_adv_full));du .+=interior(n.U_wave).+interior(n.U_viscous).+interior(n.U_forcing)
        dw=copy(interior(n.W_adv_full));dw .+=interior(n.W_wave).+interior(n.W_viscous).+interior(n.W_forcing)
        # Tangency to the constrained phase space: normal velocity is fixed
        # at impermeable walls, hence its allowed time derivative is zero.
        # Bare unprojected momentum operators at the wall do not themselves
        # satisfy this condition; q_t uses the homogeneous tendency BC too.
        @views dw[:,:,1].=0
        @views dw[:,:,end].=0
        hstep=1e-5;energies=Float64[]
        for sign in (-1,1)
            interior(u).=ub.+sign*hstep.*du;interior(w).=wb.+sign*hstep.*dw
            compute!(r.centers.Q)
            push!(energies,Float64(sum(abs2,interior(r.centers.Q)))/(2length(interior(r.centers.Q))))
        end
        interior(u).=ub;interior(w).=wb
        @test (energies[2]-energies[1])/(2hstep)≈row["dZq_native"] rtol=3e-6 atol=3e-12
        # Terminal reductions read raw buffers. Verify that the next sample
        # refreshes those buffers after velocity changes, rather than retaining
        # the preceding sample or relying on a matching physical-time cache.
        interior(u).=1.01.*ub
        m.clock.time=.001
        JetDiagnostics.native_sample(j)
        changed=sample_moment_diagnostics!(sim,d)
        compute!(r.centers.Q)
        direct_Zq=Float64(sum(abs2,interior(r.centers.Q)))/(2length(interior(r.centers.Q)))
        @test changed["Zq"]≈direct_Zq rtol=2e-13
        @test !isapprox(changed["Zq"],row["Zq"];rtol=1e-6)
        interior(u).=ub;interior(w).=wb;m.clock.time=0.
        close_moment_diagnostics!(d);close_regime_diagnostics!(r);close_jet_diagnostics!(j)
        # Restart truncation is checked for both files and avoids duplicate t=0.
        j2=init_jet_diagnostics(m,regime_context(m,dir);resume=true)
        r2=init_regime_diagnostics(j2;resume=true);d2=init_moment_diagnostics(r2;resume=true)
        @test sample_moment_diagnostics!(sim,d2)===nothing
        close_moment_diagnostics!(d2);close_regime_diagnostics!(r2);close_jet_diagnostics!(j2)
        row
    end
end

@testset "Moment signs, native weights, cancellations, and chain rule" begin
    # Manufactured local derivatives away from periodic joins; q>0, vy<0
    # means stretching destroys this component's enstrophy.
    p=(;nx=6,ny=6,nz=6,dx=1.,dy=1.,dz=1.)
    u=[2.0j for i in 1:6,j in 1:6,k in 1:6]
    v=[3.0i-4j+5k for i in 1:6,j in 1:6,k in 1:6]
    w=[7.0j for i in 1:6,j in 1:6,k in 1:6];q=fill(2.,6,6,6)
    s,x,z=JetDiagnostics.moment_T(3,3,3,u,v,w,q,p)
    @test (s,x,z)==(-8.,21.,-10.)
    @test 2s<0 && 2x>0 && 2z<0
    global cpu_moment_result=test_moment_case(CPU())
    test_moment_case(CPU();eta=0.,threec=true)
end
if get(ENV,"TEST_GPU","0")=="1"
    using CUDA
    CUDA.functional()||error("Requested GPU moment validation requires a working GPU")
    CUDA.allowscalar(false)
    @testset "CPU GPU moment agreement" begin
        result=test_moment_case(GPU())
        for name in MOMENT_FIELDS
            @test result[name]≈cpu_moment_result[name] rtol=1e-9 atol=1e-16
        end
        test_moment_case(GPU();eta=0.,threec=true)
    end
end
println("VORTICITY_MOMENTS_VALIDATION_PASSED")
