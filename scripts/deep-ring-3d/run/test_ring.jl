# CPU checks of the rebuilt initial condition and a short 2D/3D integration (run in a CPU Slurm job).
ENV["ARCH"]="CPU"
include("ring3d.jl")
using Test, FFTW
@testset "ring IC" begin
    nx=nz=256; u2,w2=vallis_ring(nx,nz;seed=1)
    # spectral centroid in "cycles/box" using |K| = sqrt((2πm)^2+(nπ)^2)/2π, from the corner streamfunction modes
    ic=initial_condition(;nx,ny=1,nz,ly=0.5,rb=1.0,seed=1,eps3d=0.0)
    @test isapprox(ic.u0,1.5635985129990398e-4;rtol=1e-12)          # REPORT/provenance q0 for rb=1
    @test isapprox(ic.Te,113.09733552923254;rtol=1e-12)            # REPORT T_e
    @test ic.max_divergence<1e-15 && ic.wall_w==0
    @test isapprox(ringrms(u2.*(ic.u0/ringrms(u2,w2)),w2.*(ic.u0/ringrms(u2,w2))),ic.u0;rtol=1e-12)
    @test maximum(abs,mean(ic.u;dims=(1,2)))<1e-12*ic.u0 + 1e-20   # no x-mean (jet) flow initially
    # energy-weighted wavenumber of u,w via 2D FFT in x and sine-ish in z (approximate): use the KE spectrum of uc in x
    uc=(ic.u[:,1,:].+circshift(ic.u[:,1,:],(-1,0)))./2
    Ex=vec(sum(abs2,fft(uc,1);dims=2))[2:nx÷2]; mx=collect(1:nx÷2-1)
    println("x-centroid of u energy (cycles; ring at 9 total): ",sum(mx.*Ex)/sum(Ex))
    ic3=initial_condition(;nx=64,ny=16,nz=64,ly=0.5,rb=1.0,seed=2,eps3d=0.01)
    @test isapprox(ic3.seed_energy_ratio,0.01;rtol=1e-8)
    @test ic3.max_divergence<1e-12*ic3.u0*64 && ic3.wall_w<1e-18 && ic3.ymean_seed<1e-12*ic3.u0
end
@testset "short integrations" begin
    for (ny,eps) in ((1,0.0),(8,0.1)), waves in (0,1)
        out=mktempdir(); run_case(;name="t",out,nx=32,ny,nz=32,ly=0.25,waves,rb=1.0,seed=1,eps3d=eps,stopTe=0.2,sampleTe=0.1,snapTe=(0.,),maxdt=1.0)
        L=readlines(joinpath(out,"timeseries.csv")); @test length(L)==4
        f=split(L[end],","); KE0=parse(Float64,split(L[2],",")[4]); KE=parse(Float64,f[4])
        println("ny=$ny waves=$waves eps=$eps KE/KE0=",KE/KE0," div=",f[end]," 3Dfrac=",parse(Float64,f[6])/KE)
        @test 0.97<KE/KE0<1.01 && parse(Float64,f[end])<1e-10*sqrt(KE)*32
    end
    # null: zero flow stays zero with waves (Stokes term acts only on velocity)
    out=mktempdir(); grid=RectilinearGrid(CPU();size=(16,8,16),halo=(5,5,5),x=(0,1),y=(0,.25),z=(-1,0),topology=(Periodic,Periodic,Bounded))
    m=NonhydrostaticModel(grid;advection=WENO(order=9),timestepper=:RungeKutta3,stokes_drift=UniformStokesDrift(∂z_uˢ=deep_shear))
    sim=Simulation(m;Δt=1.0,stop_iteration=20); run!(sim); @test maximum(abs,interior(m.velocities.u))==0
end
println("TESTS PASSED")
