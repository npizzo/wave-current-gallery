using Test, Oceananigans, Statistics
include("initial_conditions.jl")
include("regime_physics.jl")
include("jet_diagnostics.jl")
using .JetDiagnostics

const RH=.5; const RLX=2.; const RLY=1.; const RK=6.1; const RS=.17^2*sqrt(9.81/RK)

function regime_test_model(;n=16,ny=8,alpha=.75,waves=1,nu=1e-5,maintain=true)
    grid=ny==1 ? RectilinearGrid(CPU();size=(n,n),x=(0,RLX),z=(-RH,0),halo=(3,3),topology=(Periodic,Flat,Bounded)) :
        RectilinearGrid(CPU();size=(n,ny,n),x=(0,RLX),y=(0,RLY),z=(-RH,0),halo=(3,3,3),topology=(Periodic,Periodic,Bounded))
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

function regime_states(m,dir;kwargs...)
    j=init_jet_diagnostics(m,regime_context(m,dir;kwargs...))
    r=init_regime_diagnostics(j)
    sim=Simulation(m;Δt=.001,stop_time=.01,verbose=false)
    a=sample_jet_diagnostics!(sim,j)
    row=sample_regime_diagnostics!(sim,r,a)
    return j,r,sim,row
end
function close_states(j,r)
    close_regime_diagnostics!(r);close_jet_diagnostics!(j)
end

@testset "Explicit viscosity damps a resolved Fourier mode at its native rate" begin
    m=regime_test_model(;alpha=1.,nu=.01)
    set!(m;u=0,w=0,v=(x,y,z)->.002sin(pi*x))
    v0=copy(Array(interior(m.velocities.v)))
    sim=Simulation(m;Δt=.001,stop_time=.02,verbose=false)
    run!(sim)
    dx=RLX/m.grid.Nx
    rate=.01*(2sin(pi*dx/2)/dx)^2
    @test Array(interior(m.velocities.v))≈v0.*exp(-rate*.02) rtol=1e-11 atol=1e-15
    @test maximum(abs,Array(interior(m.velocities.u)))<1e-15
    @test maximum(abs,Array(interior(m.velocities.w)))<1e-15
end

@testset "Native maintenance exactly balances background viscosity" begin
    for (alpha,waves) in ((.5,1),(1.,1),(.75,0)), ny in (1,8)
      mktempdir() do dir
        m=regime_test_model(;alpha,waves,ny)
        fun(x,y,z)=(waves-alpha)*RS*exp(2RK*z)
        set!(m;u=ny==1 ? ((x,z)->fun(x,0.,z)) : fun,v=0,w=0)
        before=map(a->copy(Array(interior(a))),m.velocities)
        j,r,sim,row=regime_states(m,dir;alpha,waves)
        maps=regime_maps(r)
        @test maximum(abs,maps["U_viscous"].+maps["U_forcing"])<1e-17
        @test maximum(abs,maps["Qdot_total"])<2e-15
        @test maximum(abs,maps["A"])==0
        @test maximum(abs,maps["Rbar"])==0
        @test row["Eperp"]<1e-32
        @test row["native_divergence_max"]==0
        run!(sim)
        @test maximum(maximum(abs.(Array(interior(a)).-b)) for (a,b) in zip(m.velocities,before))<1e-14
        close_states(j,r)
      end
    end
end

@testset "Exact 2D3C is invariant and has no transverse correction" begin
    for ny in (1,8)
      mktempdir() do dir
        m=regime_test_model(;ny)
        ic=initial_velocity_fields(;nx=16,ny,nz=16,lx=RLX,ly=RLY,h=RH,k=RK,S_reference=RS,
            waves=1,alpha=.75,uoffset=0.,eddy_rms=.004,injection_wavelength=.5,perturb_energy_ratio=0.,seed=3)
        set!(m;u=ic.u,w=ic.w,v=ny==1 ? ((x,z)->.002sin(pi*x)*cos(2pi*z)) : ((x,y,z)->.002sin(pi*x)*cos(2pi*z)))
        set!(m.pressures.pNHS,ny==1 ? ((x,z)->.003sin(pi*x)*cos(2pi*z)) : ((x,y,z)->.003sin(pi*x)*cos(2pi*z)))
        j,r,sim,row=regime_states(m,dir)
        maps=regime_maps(r)
        @test row["native_divergence_max"]<1e-15
        @test row["Eperp"]<1e-32
        @test maximum(abs,maps["Qdot_3d_native"])<1e-16
        @test maximum(abs,maps["Rbar"])<1e-17
        @test maximum(abs,maps["R_tilt_x"])>1e-6
        @test maximum(abs,maps["R_tilt_x"].+maps["R_tilt_z"])<1e-17
        @test maximum(abs,maps["continuum_R_identity_error"])<1e-17
        @test row["native_pressure_curl_max"]<1e-14
        before=map(a->copy(Array(interior(a))),m.velocities)
        regime_maps(r)
        @test all(Array(interior(a))==b for (a,b) in zip(m.velocities,before))
        close_states(j,r)
      end
    end
end

function smooth_curl_ic(n,ny)
    dx,dy,dz=RLX/n,RLY/ny,RH/n
    xf=reshape(collect(0:n-1).*dx,n,1,1);xc=xf.+dx/2
    yf=reshape(collect(0:ny-1).*dy,1,ny,1);yc=yf.+dy/2
    zf=reshape(collect(0:n).*dz,1,1,n+1);zc=reshape((collect(0:n-1).+.5).*dz,1,1,n)
    ax=@. .0004*cos(pi*xc)*cos(2pi*yf)*sin(2pi*zf)
    ay=@. .0006*cos(pi*xf)*cos(2pi*yc)*sin(2pi*zf)
    az=@. .0002*sin(pi*xf)*sin(2pi*yf)*cos(2pi*zc)
    u,v,w=curl_potential(ax,ay,az,dx,dy,dz)
    u .+= reshape(.25RS.*exp.(2RK.*(-RH.+(collect(0:n-1).+.5).*dz)),1,1,n)
    return u,v,w
end

@testset "Continuum scalar identity converges to the native tendency" begin
    errors=Float64[]
    for n in (16,32,64)
      mktempdir() do dir
        m=regime_test_model(;n,ny=n)
        u,v,w=smooth_curl_ic(n,n);set!(m;u,v,w)
        j,r,sim,row=regime_states(m,dir)
        maps=regime_maps(r)
        # Hold the physical comparison region fixed under refinement.
        z=[-RH+(k-.5)*RH/n for k in 1:n];ids=findall(z->-.4<z<-.1,z)
        push!(errors,sqrt(mean(abs2,maps["continuum_scalar_residual"][:,ids])))
        @test row["native_divergence_max"]<1e-14
        @test maximum(abs,maps["continuum_R_identity_error"])<1e-17
        @test row["Eperp"]>0
        @test maximum(abs,maps["Qdot_3d_native"])>1e-6
        close_states(j,r)
      end
    end
    println("Continuum/native scalar identity errors: ",errors)
    @test errors[2]<.4errors[1]
    @test errors[3]<.4errors[2]
end

@testset "Direct transverse energy retains the twin-amplitude signal" begin
    energies=Float64[];final_energies=Float64[];shapes=Any[]
    for eta in (1e-6,1e-8)
      mktempdir() do dir
        m=regime_test_model()
        ic=initial_velocity_fields(;nx=16,ny=8,nz=16,lx=RLX,ly=RLY,h=RH,k=RK,S_reference=RS,
            waves=1,alpha=.75,uoffset=0.,eddy_rms=.004,injection_wavelength=.5,perturb_energy_ratio=eta,seed=3)
        push!(shapes,ic.perturbation_shape)
        set!(m;u=ic.u,v=ic.v,w=ic.w)
        j,r,sim,row=regime_states(m,dir)
        push!(energies,row["Eperp"])
        @test row["Eperp"]>1e-15
        @test row["native_divergence_max"]<1e-15
        @test row["viscous_cfl"]<.2
        run!(sim)
        sample=sample_jet_diagnostics!(sim,j)
        final_row=sample_regime_diagnostics!(sim,r,sample)
        push!(final_energies,final_row["Eperp"])
        close_states(j,r)
      end
    end
    @test shapes[1]==shapes[2]
    @test energies[1]/energies[2]≈100 rtol=1e-10
    # Short finite-amplitude evolution agrees in the two amplitudes. This is
    # an assay of the linear regime, not an exact tangent-linear integrator.
    @test final_energies[1]/final_energies[2]≈100 rtol=5e-4
end
