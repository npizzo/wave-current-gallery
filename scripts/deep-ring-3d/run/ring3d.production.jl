# Deep-wave Vallis-ring decay in the Craik–Leibovich equations, rebuilt in 2D and 3D.
# Target: the "deep" family of REPORT_vallis2d.md (NumericalWaveTanks.jl, branch glw/anti-stokes):
#   unit box, periodic x (and y), free-slip impermeable walls at z=-1,0 (surface z=0),
#   prognostic Lagrangian-mean velocity, no closure (WENO-9 implicit LES), RK3,
#   ∂z uˢ = ¼ e^{8z}  (β_eff = ∂zz uˢ = 2 e^{8z}), Lagrangian mean at rest (U_E = -uˢ, Q = s, sQ = s² > 0),
#   narrow Gaussian ring at 9 cycles, width one x-mode, kx=0 modes removed,
#   u'_rms,0 = β_ref/(rb K0²) with β_ref = 1/2.
# 3D members add a y-dependent, exactly solenoidal seed (regime-map transverse_eddy) of energy eps3d × ring energy.
using Oceananigans, Random, Statistics, Printf, JLD2
const ARCHNAME=uppercase(get(ENV,"ARCH","GPU"))
ARCHNAME=="GPU" && @eval using CUDA
include("regime_initial_conditions.jl")   # verbatim copy from the regime-map campaign (sha recorded)

envf(k,d)=parse(Float64,get(ENV,k,string(d))); envi(k,d)=parse(Int,get(ENV,k,string(d)))
@inline deep_shear(z,t)=0.25*exp(8z)

"y-invariant Vallis ring on the C-grid from a corner streamfunction ψ = Σ 2Re[A e^{i2πmx}] sin(nπ(z+1))."
function vallis_ring(nx,nz;k0cycles=9.0,width_modes=1.0,seed=1)
    dx,dz=1/nx,1/nz; xf=collect(0:nx-1).*dx; zf=collect(0:nz).*dz          # zf measured from the bottom wall
    K0=2π*k0cycles; σ=2π*width_modes; rng=Xoshiro(seed); psi=zeros(nx,nz+1)
    for m in 1:ceil(Int,(K0+6σ)/2π), n in 1:ceil(Int,(K0+6σ)/π)            # m ≥ 1: kx=0 (jet) modes removed
        kx,kz=2π*m,π*n; K=hypot(kx,kz); E=exp(-(K-K0)^2/(2σ^2))
        E<1e-12 && continue
        a=sqrt(E/K)/K*complex(randn(rng),randn(rng))/sqrt(2)              # energy per mode ∝ E(K)/K
        psi .+= 2 .* real.(a .* cis.(kx.*xf)) .* transpose(sin.(kz.*zf))
    end
    psi[:,1].=0; psi[:,end].=0
    u=diff(psi;dims=2)./dz; w=-(circshift(psi,(-1,0)).-psi)./dx
    return u,w   # (nx,nz), (nx,nz+1)
end

ringrms(u,w)=sqrt(mean(abs2,(u.+circshift(u,(-1,0)))./2)+mean(abs2,(w[:,1:end-1].+w[:,2:end])./2))

function initial_condition(;nx,ny,nz,ly,rb,seed,eps3d)
    K0=2π*9; u0=0.5/(rb*K0^2)
    u2,w2=vallis_ring(nx,nz;seed); s=u0/ringrms(u2,w2); u2.*=s; w2.*=s
    u=repeat(reshape(u2,nx,1,nz),1,ny,1); w=repeat(reshape(w2,nx,1,nz+1),1,ny,1); v=zeros(nx,ny,nz)
    seedE=0.0; ymean_seed=0.0
    if eps3d>0
        ny>=8 || error("eps3d>0 needs NY>=8")
        up,vp,wp=transverse_eddy(nx,ny,nz,1.0,ly,1.0,seed,1/9)             # zero y-mean, exactly solenoidal
        sc=sqrt(eps3d)*velocity_norms(u,v,w).total/velocity_norms(up,vp,wp).total
        seedE=(velocity_norms(sc.*up,sc.*vp,sc.*wp).total/velocity_norms(u,v,w).total)^2
        ymean_seed=sc*max(maximum(abs,mean(up;dims=2)),maximum(abs,mean(vp;dims=2)),maximum(abs,mean(wp;dims=2)))
        u.+=sc.*up; v.+=sc.*vp; w.+=sc.*wp
    end
    div=maximum(abs,native_divergence(u,v,w,1/nx,ly/ny,1/nz))
    return (;u,v,w,u0,K0,Te=1/(u0*K0),seed_energy_ratio=seedE,max_divergence=div,
            wall_w=max(maximum(abs,w[:,:,1]),maximum(abs,w[:,:,end])),ymean_seed)
end

layer(z)=z .> -0.3            # upper layer where β_eff/β_eff(0) ≥ e^{-2.4}≈0.09

function run_case(; name, out, nx=256, ny=1, nz=256, ly=0.5, waves=1, rb=1.0, seed=1, eps3d=0.0,
                  stopTe=200.0, sampleTe=1.0, snapTe=(0.,25.,50.,100.,150.,200.), maxdt=1.0, cfl=0.5)
    mkpath(out); isfile(joinpath(out,"completed.txt")) && error("Refusing to overwrite completed run $out")
    arch=ARCHNAME=="GPU" ? GPU() : CPU()
    ic=initial_condition(;nx,ny,nz,ly,rb,seed,eps3d)
    grid=ny==1 ? RectilinearGrid(arch,Float64;size=(nx,nz),halo=(5,5),x=(0,1),z=(-1,0),topology=(Periodic,Flat,Bounded)) :
                 RectilinearGrid(arch,Float64;size=(nx,ny,nz),halo=(5,5,5),x=(0,1),y=(0,ly),z=(-1,0),topology=(Periodic,Periodic,Bounded))
    sd=waves==1 ? UniformStokesDrift(∂z_uˢ=deep_shear) : nothing
    model=NonhydrostaticModel(grid;advection=WENO(order=9),timestepper=:RungeKutta3,stokes_drift=sd)
    set!(model;u=ic.u,v=ic.v,w=ic.w)
    Te=ic.Te; sim=Simulation(model;Δt=min(maxdt,0.25),stop_time=stopTe*Te)
    conjure_time_step_wizard!(sim,IterationInterval(10);cfl,max_Δt=maxdt)
    zc=collect(znodes(grid,Center())); up=layer(zc)
    meta=open(joinpath(out,"meta.txt"),"w")
    for (k,v) in pairs((;name,nx,ny,nz,ly,waves,rb,seed,eps3d,stopTe,maxdt,cfl,u0=ic.u0,Te,
            seed_energy_ratio=ic.seed_energy_ratio,ic_seed_ymean=ic.ymean_seed,ic_max_divergence=ic.max_divergence,ic_wall_w=ic.wall_w,arch=ARCHNAME))
        println(meta,k,"=",v)
    end; close(meta)
    open(joinpath(out,"zc.txt"),"w") do io; foreach(z->println(io,z),zc); end
    ts=open(joinpath(out,"timeseries.csv"),"w")
    println(ts,"t_Te,iteration,dt,KE,KE2D,KE3D,KEjet,KE3D_upper,KE2D_upper,KE3D_lower,KE2D_lower,umax,divmax"); flush(ts)
    prof=open(joinpath(out,"profiles.csv"),"w"); println(prof,"t_Te,"*join(["UL_$k" for k in 1:nz],",")*","*join(["E3D_$k" for k in 1:nz],",")*","*join(["E2D_$k" for k in 1:nz],",")); flush(prof)
    ymdir=joinpath(out,"ymean"); mkpath(ymdir); snaps=Set(round.(collect(snapTe);digits=6))
    lastsample=Ref(-Inf)
    function sample(s)
        tTe=time(s)/Te
        tTe<=lastsample[]+1e-9 && return; lastsample[]=tTe     # Oceananigans may also fire callbacks at initialization
        u=interior(model.velocities.u); v=interior(model.velocities.v); w=interior(model.velocities.w)
        uc=Array(u); vc=Array(v); wc=Array(w)          # host copies (≤ 70 MB at 256×128×256)
        wcc=(wc[:,:,1:end-1].+wc[:,:,2:end])./2
        all(isfinite,uc)&&all(isfinite,wc) || error("Non-finite velocity at t=$tTe Te")
        ub=mean(uc;dims=2); vb=mean(vc;dims=2); wb=mean(wcc;dims=2)
        e3=dropdims(mean((uc.-ub).^2 .+ (vc.-vb).^2 .+ (wcc.-wb).^2;dims=(1,2));dims=(1,2))./2   # per z
        e2=dropdims(mean(ub.^2 .+ vb.^2 .+ wb.^2;dims=(1,2));dims=(1,2))./2
        UL=dropdims(mean(ub;dims=(1,2));dims=(1,2))
        KE3D=mean(e3); KE2D=mean(e2); KEjet=mean(UL.^2)/2
        div=maximum(abs,native_divergence(uc,vc,wc,1/nx,ly/ny,1/nz))
        umax=maximum(abs,uc)
        @printf(ts,"%.6f,%d,%.6g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.6g,%.3g\n",tTe,iteration(s),s.Δt,KE3D+KE2D,KE2D,KE3D,KEjet,
                mean(e3[up]),mean(e2[up]),mean(e3[.!up]),mean(e2[.!up]),umax,div); flush(ts)
        println(prof,@sprintf("%.6f",tTe)*","*join((@sprintf("%.6e",x) for x in vcat(UL,e3,e2)),",")); flush(prof)
        open(joinpath(ymdir,@sprintf("t%07.2f.bin",tTe)),"w") do io; write(io,Float32.(dropdims(ub;dims=2))); write(io,Float32.(dropdims(mean(wc;dims=2);dims=2))); end
        if round(tTe;digits=6) in snaps || abs(tTe-stopTe)<1e-6
            jldsave(joinpath(out,@sprintf("snapshot_t%06.1f.jld2",tTe));u=Float32.(uc),v=Float32.(vc),w=Float32.(wc),t=time(s),Te)
        end
        @printf("t=%7.2f Te it=%d dt=%.3g KE/KE0 3D-frac=%.3e jet-frac=%.3f div=%.1e\n",tTe,iteration(s),s.Δt,KE3D/(KE3D+KE2D),KEjet/(KE3D+KE2D),div); flush(stdout)
    end
    sample(sim)
    sim.callbacks[:sample]=Callback(sample,SpecifiedTimes(collect(sampleTe:sampleTe:stopTe).*Te))
    started=time_ns(); run!(sim)
    close(ts); close(prof)
    open(joinpath(out,"completed.txt"),"w") do io
        println(io,"t_Te=",time(sim)/Te); println(io,"iterations=",iteration(sim)); println(io,"wall_s=",(time_ns()-started)/1e9)
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    a=Dict(split(x,"=";limit=2) for x in ARGS)
    g(k,d)=haskey(a,k) ? a[k] : d
    run_case(;name=a["name"],out=a["out"],nx=parse(Int,g("nx","256")),ny=parse(Int,g("ny","1")),nz=parse(Int,g("nz","256")),
        ly=parse(Float64,g("ly","0.5")),waves=parse(Int,g("waves","1")),rb=parse(Float64,g("rb","1")),seed=parse(Int,g("seed","1")),
        eps3d=parse(Float64,g("eps3d","0")),stopTe=parse(Float64,g("stopTe","200")),maxdt=parse(Float64,g("maxdt","1.0")))
end
