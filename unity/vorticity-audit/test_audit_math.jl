using Test
include("audit_math.jl")
using .PacketVorticityAuditMath

function native_fields(nx,ny,zf; Lx=2.0,Ly=2pi,
                       ufun=(x,y,z)->0.0,vfun=(x,y,z)->0.0,wfun=(x,y,z)->0.0)
    nz=length(zf)-1; dx=Lx/nx; dy=Ly/ny
    xf=collect(0:nx).*dx; xc=(xf[1:end-1].+xf[2:end])./2
    yf=collect(0:ny-1).*dy; yc=yf.+dy/2
    zc=(zf[1:end-1].+zf[2:end])./2
    u=[ufun(x,y,z) for x in xf,y in yc,z in zc]
    v=[vfun(x,y,z) for x in xc,y in yf,z in zc]
    w=[wfun(x,y,z) for x in xc,y in yc,z in zf]
    return (;u,v,w,dx,dy,zf,xc,yc,zc)
end

@testset "Native staggering and incompressibility" begin
    nx,ny,nz=12,32,14
    zf=[-1+((k/nz)^1.4) for k in 0:nz]
    a,b=.7,.3; dy=2pi/ny
    f=native_fields(nx,ny,zf;
        ufun=(x,y,z)->a*x,
        vfun=(x,y,z)->b*sin(y),
        wfun=(x,y,z)->-(a+b*(sin(y+dy/2)-sin(y-dy/2))/dy)*z)
    for step in (1,2)
        r=snapshot_reductions(f.u,f.v,f.w,f.dx,f.dy,f.zf;stencil_step=step)
        @test r.qc.max_native_divergence < 2e-14
        @test r.qc.scaled_native_divergence_max < 2e-15
        @test all(isapprox.(r.means["native_ux"],a;atol=2e-14))
        @test maximum(r.mean_squares["native_vy"]) > .04
        @test r.mean_squares["native_xz_divergence"] ≈ r.mean_squares["native_vy"] atol=2e-14
    end
end

@testset "2-D three-component flow: tilts cancel without requiring v=0" begin
    zf=collect(range(-1,0;length=17))
    f=native_fields(16,16,zf;
        ufun=(x,y,z)->1+z*z,
        vfun=(x,y,z)->x*z,
        wfun=(x,y,z)->.2x)
    for step in (1,2)
        r=snapshot_reductions(f.u,f.v,f.w,f.dx,f.dy,f.zf;stencil_step=step)
        @test maximum(r.maxabs["v"]) > 1
        @test maximum(r.maxabs["omegaEx_vx"]) > .5
        @test maximum(r.maxabs["omegaEz_vz"]) > .5
        @test maximum(r.maxabs["q_vy"]) == 0
        @test maximum(r.maxabs["v_qy"]) == 0
        @test maximum(r.maxabs["R3D"]) < 1e-12
        @test r.qc.R3D_identity_error < 1e-12
        @test all(isapprox.(r.means["q"],repeat(reshape(2 .* f.zc .- .2,1,:),16,1);atol=1e-12))
        @test all(isapprox.(r.means["qz"],2;atol=1e-10))
        @test maximum(r.mean_squares["adv_xz"]) > .1
    end
end

@testset "vy=0 does not eliminate three-dimensional vorticity terms" begin
    zf=collect(range(-1,0;length=18)); b=.8
    f=native_fields(14,64,zf;
        ufun=(x,y,z)->sin(y)*z,
        vfun=(x,y,z)->b*z)
    for step in (1,2)
        r=snapshot_reductions(f.u,f.v,f.w,f.dx,f.dy,f.zf;stencil_step=step)
        # q=sin(y); stretching=-b*z*cos(y); v*qy=+b*z*cos(y).
        # Periodic differences multiply cos(y) by sin(step*dy)/(step*dy).
        factor=sin(step*f.dy)/(step*f.dy)
        expected=repeat(reshape(2b*b .* f.zc.^2 .* factor^2,1,:),14,1)
        @test r.qc.max_native_divergence == 0
        @test maximum(r.maxabs["native_vy"]) == 0
        @test maximum(r.maxabs["v_y"]) == 0
        @test r.mean_squares["R3D"] ≈ expected atol=2e-12 rtol=2e-12
        @test r.mean_squares["R3D"] ≈ 4 .* r.mean_squares["stretching"] atol=2e-12
        @test r.mean_squares["q"] ≈ fill(.5,14,length(f.zc)) atol=2e-12
        @test maximum(r.maxabs["R3D"]) > 1
        @test r.qc.R3D_identity_error < 1e-12
        @test maximum(abs.(r.means["R3D"])) < 1e-12
        @test maximum(r.mean_squares["R3D"]) > 1
    end
end

@testset "Eulerian curl subtracts prescribed Stokes components" begin
    zf=collect(range(-1,0;length=16))
    f=native_fields(16,16,zf;ufun=(x,y,z)->z*z,wfun=(x,y,z)->x*x)
    # Supply centre values obtained by averaging the same native faces.
    us=dropdims((f.u[1:end-1,1:1,:].+f.u[2:end,1:1,:])./2;dims=2)
    ws=dropdims((f.w[:,1:1,1:end-1].+f.w[:,1:1,2:end])./2;dims=2)
    before=map(copy,(f.u,f.v,f.w,us,ws))
    for step in (1,2)
        r=snapshot_reductions(f.u,f.v,f.w,f.dx,f.dy,f.zf;us,ws,stencil_step=step)
        @test maximum(r.maxabs["q"]) < 1e-13
        @test maximum(r.maxabs["R3D"]) < 1e-13
        @test maximum(r.maxabs["adv_xz"]) < 1e-11
    end
    @test all(a==b for (a,b) in zip(before,(f.u,f.v,f.w,us,ws)))
end

@testset "Optional time-dependent nonuniform packet source and its sign" begin
    zf=collect(range(-1,0;length=14))
    f=native_fields(14,16,zf;ufun=(x,y,z)->1+z,wfun=(x,y,z)->.5x)
    dims=(14,length(f.zc))
    r=snapshot_reductions(f.u,f.v,f.w,f.dx,f.dy,f.zf;
        s=fill(5.,dims),s_t=fill(2.,dims),s_x=fill(3.,dims),s_z=fill(4.,dims))
    expected=[2+3*(1+z)+4*.5x for x in f.xc,z in f.zc]
    @test r.qc.source_supplied
    @test r.means["s_material"] ≈ expected atol=1e-12
    @test r.mean_squares["s_material"] ≈ expected.^2 atol=1e-12
    @test r.means["eta_rhs"] ≈ -expected atol=1e-12
    @test maximum(r.maxabs["s_vy"]) == 0
    @test maximum(r.maxabs["R3D"]) == 0
end

@testset "Stretched-z derivative convergence including q gradients" begin
    function errors(n,step)
        zf=[-1+(exp(1.5k/n)-1)/(exp(1.5)-1) for k in 0:n]
        f=native_fields(8,8,zf;ufun=(x,y,z)->exp(z))
        r=snapshot_reductions(f.u,f.v,f.w,f.dx,f.dy,f.zf;stencil_step=step)
        iz=(2step+1):(n-2step)
        analytic=exp.(f.zc[iz]); weights=diff(zf)[iz]
        e1=sqrt(sum(weights.*(r.means["q"][4,iz].-analytic).^2)/sum(weights))
        e2=sqrt(sum(weights.*(r.means["qz"][4,iz].-analytic).^2)/sum(weights))
        return e1,e2
    end
    for step in (1,2)
        coarse=errors(32,step); fine=errors(64,step)
        @test coarse[1]/fine[1] > 3
        @test coarse[2]/fine[2] > 3
        @test fine[1] < .002
        @test fine[2] < .005
    end
end

@testset "Invalid inputs fail explicitly" begin
    f=native_fields(8,8,collect(range(-1,0;length=9)))
    @test_throws ArgumentError snapshot_reductions(f.u,f.v,f.w,f.dx,f.dy,f.zf;stencil_step=3)
    @test_throws DimensionMismatch snapshot_reductions(f.u[1:end-1,:,:],f.v,f.w,f.dx,f.dy,f.zf)
    @test_throws ArgumentError snapshot_reductions(f.u,f.v,f.w,f.dx,f.dy,reverse(f.zf))
    @test_throws DimensionMismatch snapshot_reductions(f.u,f.v,f.w,f.dx,f.dy,f.zf;us=zeros(2,2))
    f.u[1]=NaN
    @test_throws ArgumentError snapshot_reductions(f.u,f.v,f.w,f.dx,f.dy,f.zf)
end
