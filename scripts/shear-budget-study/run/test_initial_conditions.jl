using Test
include("initial_conditions.jl")
include("vortex_modes.jl")

@testset "Discrete-curl IC and preserved q0" begin
    args=(;nx=32,ny=16,nz=32,lx=2.,ly=1.,h=.5,k=6.1,S_reference=.03665,waves=1,
        alpha=.75,uoffset=0.,eddy_rms=.004,injection_wavelength=.25,seed=1)
    b=initial_velocity_fields(;args...,perturb_energy_ratio=0.)
    @test b.metrics.base_inplane_rms ≈ .004 rtol=1e-13
    @test b.metrics.max_native_divergence <1e-15
    @test b.metrics.max_mean_profile_error <1e-15
    @test all(iszero,b.v)
    @test b.u[:,1,:]==b.u[:,end,:]
    @test all(iszero,b.w[:,:,1]) && all(iszero,b.w[:,:,end])
    for eta in (1e-4,1.)
        p=initial_velocity_fields(;args...,perturb_energy_ratio=eta)
        @test p.base==b.base
        @test p.metrics.achieved_perturb_energy_ratio ≈ eta rtol=1e-13
        @test p.metrics.max_native_divergence <5e-15
        @test p.metrics.max_mean_profile_error <1e-15
        @test p.metrics.max_transverse_mean_perturbation<1e-16
        @test maximum(abs,p.v)>0
        @test all(iszero,p.w[:,:,1]) && all(iszero,p.w[:,:,end])
    end
    p=initial_velocity_fields(;args...,perturb_energy_ratio=0.,ic_kind="three_d")
    @test p.metrics.perturb_inplane_rms ≈ .004 rtol=1e-13
    @test p.metrics.max_native_divergence<5e-15
    @test p.metrics.max_mean_profile_error<1e-15
end

@testset "Actual-mean Rayleigh mode and native probe projection" begin
    mode=finite_depth_mode(;nx=64,nz=96,lx=2.,h=.5,k=6.1,S=.03665,alpha=.75,uoffset=0.,horizontal_mode=2)
    @test mode.c<0
    @test mode.vertical_nodes==0
    @test mode.isolated_gap>.004
    @test mode.eigen_residual<1e-10
    @test mode.relative_refinement_difference<.002
    @test 210<mode.period<225
    u=zeros(64,8,96);v=copy(u);w=zeros(64,8,97)
    amplitude=1e-5
    scale=add_probe!(u,w,mode,2.,amplitude,1)
    @test velocity_norms(u,v,w).inplane ≈ amplitude rtol=1e-12
    @test maximum(abs,native_divergence(u,v,w,2/64,1/8,.5/96))<1e-16
    ymeans=(;U=dropdims(mean(u;dims=2);dims=2),W=dropdims(mean(w;dims=2);dims=2))
    a=project_probe(ymeans,mode,2.,.5)
    @test real(a) ≈ scale rtol=1e-12
    @test abs(imag(a))<1e-18
    add_probe!(u,w,mode,2.,amplitude,-1)
    @test maximum(abs,u)==0 && maximum(abs,w)==0
    # A uniform offset shifts c by exactly that offset; the intrinsic wave
    # and vertical eigenfunction remain unchanged to eigensolver roundoff.
    shifted=finite_depth_mode(;nx=64,nz=96,lx=2.,h=.5,k=6.1,S=.03665,alpha=.75,uoffset=.01,horizontal_mode=2)
    @test shifted.c-mode.c ≈ .01 atol=1e-11
    @test shifted.phi_faces ≈ mode.phi_faces atol=1e-8
end
