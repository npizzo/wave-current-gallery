using Test
include("audit_math.jl")
include("packet_fields.jl")
using .PacketVorticityAuditMath

const TEST_PACKET=(k=9.3,Uˢ₀=.0497,σ₀=1.44,cᵍ=.514,h=.4,x₀=-5.76,periodic=0.)

function analytic_stokes(x,z,t,p)
    xi=x-p.x₀-p.cᵍ*t
    envelope=exp(-(xi/p.σ₀)^2)
    derivative=-2xi/p.σ₀^2*envelope
    return (p.Uˢ₀*envelope*exp(2p.k*z),
            -p.Uˢ₀/(2p.k)*derivative*(exp(2p.k*z)-exp(-2p.k*p.h)))
end

@testset "Analytic packet curl and x,z,t derivatives" begin
    p=TEST_PACKET; xf=collect(range(0,12;length=33)); zf=audit_zfaces(24,p.h)
    t=22.7; f=packet_fields(xf,zf,t,p)
    xc=(xf[1:end-1].+xf[2:end])./2; zc=(zf[1:end-1].+zf[2:end])./2
    h=1e-5
    curl=[-(analytic_stokes(x,z+h,t,p)[1]-analytic_stokes(x,z-h,t,p)[1])/(2h)+
           (analytic_stokes(x+h,z,t,p)[2]-analytic_stokes(x-h,z,t,p)[2])/(2h)
          for x in xc,z in zc]
    @test maximum(abs.(f.s.-curl))/maximum(abs.(f.s)) < 1e-7
    fd_x=(packet_fields(xf.+h,zf,t,p).s.-packet_fields(xf.-h,zf,t,p).s)./(2h)
    fd_z=(packet_fields(xf,zf.+h,t,p).s.-packet_fields(xf,zf.-h,t,p).s)./(2h)
    fd_t=(packet_fields(xf,zf,t+h,p).s.-packet_fields(xf,zf,t-h,p).s)./(2h)
    @test maximum(abs.(f.s_x.-fd_x))/maximum(abs.(f.s_x)) < 1e-7
    @test maximum(abs.(f.s_z.-fd_z))/maximum(abs.(f.s_z)) < 1e-7
    @test maximum(abs.(f.s_t.-fd_t))/maximum(abs.(f.s_t)) < 1e-7
    @test f.s_t ≈ -p.cᵍ .* f.s_x rtol=1e-14
    @test maximum(abs.(f.s_t)) > .1
    @test maximum(abs.(f.s_x)) > .1
end

@testset "Native-face Stokes interpolation and zero Eulerian curl" begin
    p=TEST_PACKET; nx,ny,nz=24,8,20
    xf=collect(range(0,12;length=nx+1)); zf=audit_zfaces(nz,p.h); t=21.3
    xc=(xf[1:end-1].+xf[2:end])./2; zc=(zf[1:end-1].+zf[2:end])./2
    u=[analytic_stokes(x,z,t,p)[1] for x in xf,j in 1:ny,z in zc]
    v=zeros(nx,ny,nz)
    w=[analytic_stokes(x,z,t,p)[2] for x in xc,j in 1:ny,z in zf]
    f=packet_fields(xf,zf,t,p)
    @test f.us ≈ dropdims((u[1:end-1,1:1,:].+u[2:end,1:1,:])./2;dims=2) rtol=1e-14
    @test f.ws ≈ dropdims((w[:,1:1,1:end-1].+w[:,1:1,2:end])./2;dims=2) rtol=1e-14
    geometric=[analytic_stokes(x,z,t,p)[1] for x in xc,z in zc]
    @test maximum(abs.(f.us.-geometric)) > 1e-5
    for step in (1,2)
        r=snapshot_reductions(u,v,w,12/nx,.8/ny,zf;f...,stencil_step=step)
        @test maximum(r.maxabs["q"]) < 1e-12
        @test maximum(r.maxabs["R3D"]) < 1e-12
        @test maximum(r.maxabs["adv_xz"]) < 1e-10
        @test r.qc.source_supplied
    end
end

@testset "Bounded Gaussian packet has no periodic images" begin
    p=TEST_PACKET; xf=collect(range(0,12;length=25)); zf=audit_zfaces(16,p.h)
    f=packet_fields(xf,zf,0.,p)
    @test maximum(abs.(f.us[end,:])) < 1e-50
    @test maximum(abs.(f.us[1,:])) > 1e-10
    @test_throws ErrorException packet_fields(xf,zf,0.,merge(p,(;periodic=1.)))
    disabled=packet_fields(xf,zf,22.,p;enabled=false)
    @test all(all(iszero,a) for a in values(disabled))
    @test first(zf) ≈ -.4
    @test last(zf) ≈ 0 atol=1e-15
    @test all(>(0),diff(zf))
end
