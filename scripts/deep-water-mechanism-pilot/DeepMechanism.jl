module DeepMechanism

using FFTW, Random, Statistics, LinearAlgebra

export native_inner, native_norm, native_divergence, project, advection,
       meanfield, plus, minus, times, operators, stokes_force, adjoint_stokes_force, diffusion,
       native_pv, components, rhs, rk3_step, rk3_step_with_budget, weighted_components,
       pv_inner, pv_variance, initial_ring, native_rms,
       backend_array, synchronize_backend, read_pair, write_pair, smooth_fixture

# This module has no top-level integration. Only the separately gated benchmark
# entry point may call rk3_step. Static verification calls operators only.
backend_array(a, backend::Symbol) = backend == :cpu ? Array(a) :
    (isdefined(Main, :CUDA) ? Main.CUDA.CuArray(a) : error("Load CUDA before selecting GPU"))
synchronize_backend(backend::Symbol) = backend == :gpu ? Main.CUDA.synchronize() : nothing

struct Channel{A}
    nx::Int
    nz::Int
    lx::Float64
    h::Float64
    dx::Float64
    dz::Float64
    inverse_laplacian::A
    shear_u::A
    shear_w::A
end

function Channel(nx, nz; lx=4.0, h=2.0, backend=:cpu)
    dx, dz = lx/nx, h/nz
    λx = -4 .* sin.(π .* (0:nx-1) ./ nx).^2 ./ dx^2
    λz = -4 .* sin.(π .* (0:2nz-1) ./ (2nz)).^2 ./ dz^2
    den = reshape(λx, :, 1) .+ reshape(λz, 1, :)
    invlap = zeros(Float64, nx, 2nz)
    for j in axes(invlap, 2), i in axes(invlap, 1)
        den[i,j] != 0 && (invlap[i,j] = inv(den[i,j]))
    end
    z = -h .+ ((0:nz-1) .+ 0.5).*dz
    shear = reshape(0.25 .* exp.(8 .* z), 1, :)
    shear_w = reshape(0.25 .*exp.(8 .*collect(range(-h,0;length=nz+1))),1,:)
    return Channel(nx, nz, lx, h, dx, dz,
                   backend_array(invlap, backend), backend_array(shear, backend),backend_array(shear_w,backend))
end

plus(fields...) = (sum(f[1] for f in fields), sum(f[2] for f in fields))
minus(a,b) = (a[1].-b[1], a[2].-b[2])
times(c,a) = (c.*a[1], c.*a[2])
function zero_like(a, n, m)
    b = similar(a, n, m)
    fill!(b, 0)
    return b
end
zerofield(v) = (zero_like(v[1], size(v[1])...), zero_like(v[2], size(v[2])...))
native_inner(a,b) = (sum(a[1].*b[1]) + sum(a[2][:,2:end-1].*b[2][:,2:end-1]))/length(a[1])
native_norm(v) = sqrt(max(native_inner(v,v), 0.0))
meanfield(v) = (repeat(mean(v[1]; dims=1), size(v[1],1), 1),
                repeat(mean(v[2]; dims=1), size(v[2],1), 1))

native_divergence(v,g) = (circshift(v[1],(-1,0)).-v[1])./g.dx .+ diff(v[2];dims=2)./g.dz

function pressure_gradient(p,g)
    gx = (p.-circshift(p,(1,0)))./g.dx
    z = zero_like(p, g.nx, 1)
    gz = hcat(z, diff(p;dims=2)./g.dz, z)
    return gx,gz
end

function project(v,g)
    d = native_divergence(v,g)
    # Even extension about the top/bottom cell faces is the exact Neumann
    # eigenbasis of the native second-order center Laplacian.
    extended = hcat(d, reverse(d;dims=2))
    pextended = real.(ifft(fft(extended,(1,2)).*g.inverse_laplacian,(1,2)))
    p = pextended[:,1:g.nz]
    gradp = pressure_gradient(p,g)
    return minus(v,gradp), p, gradp
end

u_at_z_faces(u) = hcat(u[:,1:1], (u[:,1:end-1].+u[:,2:end])./2, u[:,end:end])
w_at_x_faces(w) = (w.+circshift(w,(1,0)))./2

function advection(a,b,g)
    au,aw = a; bu,bw = b
    auc = (au.+circshift(au,(-1,0)))./2
    buc = (bu.+circshift(bu,(-1,0)))./2
    uu = auc.*buc
    wu = w_at_x_faces(aw).*u_at_z_faces(bu)
    nu = -(uu.-circshift(uu,(1,0)))./g.dx .- diff(wu;dims=2)./g.dz
    uw = u_at_z_faces(au).*w_at_x_faces(bw)
    ww = (aw[:,1:end-1].+aw[:,2:end]).*(bw[:,1:end-1].+bw[:,2:end])./4
    nw = -(circshift(uw,(-1,0)).-uw)./g.dx
    nw[:,2:end-1] .-= diff(ww;dims=2)./g.dz
    nw[:,1] .= 0; nw[:,end] .= 0
    return nu,nw
end

function diffusion(v,g)
    u,w = v
    uz = hcat(u[:,1:1],u,u[:,end:end])
    lu = (circshift(u,(1,0)).-2 .*u.+circshift(u,(-1,0)))./g.dx^2 .+
         diff(diff(uz;dims=2);dims=2)./g.dz^2
    lw = (circshift(w,(1,0)).-2 .*w.+circshift(w,(-1,0)))./g.dx^2
    lw[:,2:end-1] .+= diff(diff(w;dims=2);dims=2)./g.dz^2
    lw[:,1] .= 0; lw[:,end] .= 0
    return lu,lw
end

function adjoint_stokes_force(v,g)
    u,w = v
    # A interpolates native w to native u. The paired force (beta*A*w,
    # -A'*(beta*u)) is exactly skew-adjoint in the native kinetic-energy norm.
    # Its placement must be compared to Oceananigans; exact agreement is not
    # assumed merely because both approximate curl(u_s) cross u.
    wx = w_at_x_faces(w)
    su = g.shear_u.*(wx[:,1:end-1].+wx[:,2:end])./2
    t = g.shear_u.*u
    tx = t.+circshift(t,(-1,0))
    z = zero_like(u,g.nx,1)
    sw = hcat(z, -(tx[:,1:end-1].+tx[:,2:end])./4, z)
    return su,sw
end

function stokes_force(v,g)
    u,w = v
    # Exact archived UniformStokesDrift placement: interpolate velocity, then
    # multiply by analytic shear at the receiving component's vertical node.
    # This pair is not exactly skew-adjoint for depth-dependent shear. Its
    # actual discrete energy work is measured, never silently set to zero.
    wx = w_at_x_faces(w)
    su = g.shear_u.*(wx[:,1:end-1].+wx[:,2:end])./2
    ux = u.+circshift(u,(-1,0))
    z = zero_like(u,g.nx,1)
    sw = hcat(z,-g.shear_w[:,2:end-1].*(ux[:,1:end-1].+ux[:,2:end])./4,z)
    return su,sw
end

function operators(v,g)
    m = meanfield(v); e = minus(v,m)
    ee = advection(e,e,g)
    linear = plus(advection(m,e,g), advection(e,m,g))
    full = advection(v,v,g)
    ql = plus(advection(m,m,g),linear,meanfield(ee))
    return (;full,ql,linear,ee,m,e)
end

function nonlinear(v,g,arm)
    arm == :linear && return zerofield(v) # benchmark background is fixed U=0
    full = advection(v,v,g)
    arm == :full && return full
    arm == :ql || error("Unknown operator $arm")
    e = minus(v,meanfield(v)); ee = advection(e,e,g)
    return minus(full,minus(ee,meanfield(ee)))
end

function components(v,g,arm; waves=true, nu=1e-10)
    n = nonlinear(v,g,arm)
    s = waves ? stokes_force(v,g) : zerofield(v)
    d = times(nu,diffusion(v,g))
    raw = plus(n,s,d)
    total,p,gradp = project(raw,g)
    constraint = zerofield(v)
    if arm == :linear
        # The prescribed fixed background has no evolved K=0 component.
        constraint = times(-1,meanfield(total))
        total = plus(total,constraint)
    end
    return (; n,s,d,raw,total,p,gradp,constraint)
end

rhs(v,g,arm;waves=true,nu=1e-10) = components(v,g,arm;waves,nu).total

function rk3_step(v,dt,g,arm;waves=true,nu=1e-10)
    a = plus(v,times(dt,rhs(v,g,arm;waves,nu)))
    b = plus(times(3/4,v),times(1/4,plus(a,times(dt,rhs(a,g,arm;waves,nu)))))
    return plus(times(1/3,v),times(2/3,plus(b,times(dt,rhs(b,g,arm;waves,nu)))))
end

const BUDGET_TERMS = (:advection,:wave,:viscosity,:pressure,:constraint,:total)
const COMPONENT_FIELDS = (:n,:s,:d,:gradp,:constraint,:total)

function weighted_components(c0,c1,c2,dt)
    # SSPRK3 algebra: new-old=dt*(F0/6+F1/6+2F2/3). Keep the actual
    # pressure and any fixed-mean constraint at the same stage weights.
    result = Dict{Symbol,Any}()
    for (name,field,sign) in zip(BUDGET_TERMS,COMPONENT_FIELDS,(1,1,1,-1,1,1))
        result[name] = times(sign*dt,plus(times(1/6,getproperty(c0,field)),
                                         times(1/6,getproperty(c1,field)),
                                         times(2/3,getproperty(c2,field))))
    end
    return result
end

function pv_inner(a,b)
    # Periodic x faces and trapezoidal z-face quadrature, normalized by area.
    return (sum(a[:,2:end-1].*b[:,2:end-1]) +
            sum(a[:,1].*b[:,1])/2 + sum(a[:,end].*b[:,end])/2) /
           (size(a,1)*(size(a,2)-1))
end

function pv_mean(a)
    return (sum(a[:,2:end-1])+sum(a[:,1])/2+sum(a[:,end])/2) /
           (size(a,1)*(size(a,2)-1))
end
pv_variance(a) = pv_inner(a.-pv_mean(a),a.-pv_mean(a))/2

function rk3_step_with_budget(v,dt,g,arm;waves=true,nu=1e-10,budget_callback=nothing)
    c0 = components(v,g,arm;waves,nu)
    a = plus(v,times(dt,c0.total))
    c1 = components(a,g,arm;waves,nu)
    b = plus(times(3/4,v),times(1/4,plus(a,times(dt,c1.total))))
    c2 = components(b,g,arm;waves,nu)
    next = plus(times(1/3,v),times(2/3,plus(b,times(dt,c2.total))))
    increments = weighted_components(c0,c1,c2,dt)
    mean_increments = vcat((mean(increments[name][1];dims=1) for name in BUDGET_TERMS)...)
    midpoint = times(1/2,plus(v,next))
    energy_increments = [native_inner(midpoint,increments[name]) for name in BUDGET_TERMS]
    q0 = native_pv(v,g;waves)
    q1 = native_pv(next,g;waves)
    qmid = (q0.+q1)./2
    qmid .-= pv_mean(qmid)
    pv_increments = [pv_inner(qmid,native_pv(increments[name],g;waves=false,source=true))
                     for name in BUDGET_TERMS]
    # These are exact discrete quadratic-increment allocations, not claimed
    # continuous-time work integrals. Their sum must reconstruct the update.
    energy_change = (native_inner(next,next)-native_inner(v,v))/2
    pv_change = pv_variance(q1)-pv_variance(q0)
    mean_change = mean(next[1].-v[1];dims=1)
    mean_residual = maximum(abs,mean_change.-mean_increments[end:end,:])
    auxiliary = budget_callback === nothing ? nothing : budget_callback(;old=v,new=next,increments,stages=(v,a,b),qold=q0,qnew=q1,dt,waves,arm,g)
    return next,(;mean_increments,energy_increments,pv_increments,auxiliary,
                  energy_residual=energy_change-energy_increments[end],
                  pv_residual=pv_change-pv_increments[end],mean_residual)
end

function native_pv(v,g;waves=true,source=false)
    u,w = v
    z = zero_like(u,g.nx,1)
    uz = hcat(z,diff(u;dims=2)./g.dz,z)
    wx = (w.-circshift(w,(1,0)))./g.dx
    q = uz.-wx
    if waves && !source
        sf = reshape(-0.25 .*exp.(8 .*collect(range(-g.h,0;length=g.nz+1))),1,:)
        q .+= backend_array(sf, u isa Array ? :cpu : :gpu)
    end
    return q
end

native_rms(v) = sqrt(mean(abs2,(v[1].+circshift(v[1],(-1,0)))./2) +
                          mean(abs2,(v[2][:,1:end-1].+v[2][:,2:end])./2))

function initial_ring(nx,nz;lx=4.0,h=2.0,k0=27.0,width=1.0,seed=1,rb=1/9,legacy=false)
    # Coefficient formula, ordering and Xoshiro draws reproduce the archived
    # vallis_ring. FFT synthesis changes floating-point summation order only.
    K0,σ = 2π*k0,2π*width
    mmax,nmax = ceil(Int,(K0+6σ)*lx/2π),ceil(Int,(K0+6σ)*h/π)
    mmax < nx/2 || error("Ring exceeds horizontal synthesis Nyquist")
    nmax < nz || error("Ring exceeds vertical synthesis Nyquist")
    rng = Xoshiro(seed)
    aa = zeros(ComplexF64,mmax,nmax)
    for m in 1:mmax,n in 1:nmax
        K = hypot(2π*m/lx,π*n/h)
        E = exp(-(K-K0)^2/(2σ^2))
        E < 1e-12 && continue
        aa[m,n] = sqrt(E/K)/K * complex(randn(rng),randn(rng))/sqrt(2)
    end
    xf = collect(0:nx-1).*lx/nx
    zf = collect(0:nz).*h/nz
    if legacy
        psi = zeros(nx,nz+1)
        for m in 1:mmax,n in 1:nmax
            aa[m,n] == 0 && continue
            psi .+= 2 .*real.(aa[m,n].*cis.(2π*m/lx.*xf)).*transpose(sin.(π*n/h.*zf))
        end
    else
        vertical = aa*sin.(reshape(π.*(1:nmax)./h,:,1).*reshape(zf,1,:))
        coefficients = zeros(ComplexF64,nx,nz+1)
        for m in 1:mmax
            coefficients[m+1,:] .= nx.*vertical[m,:]
            coefficients[nx-m+1,:] .= nx.*conj.(vertical[m,:])
        end
        psi = real.(ifft(coefficients,1))
    end
    psi[:,1] .= 0; psi[:,end] .= 0
    # Archived builder uses the opposite streamfunction sign convention; these
    # are its actual velocity arrays, preserving the seed realization.
    u = diff(psi;dims=2)./(h/nz)
    w = -(circshift(psi,(-1,0)).-psi)./(lx/nx)
    u0 = 0.5/(rb*K0^2)
    scale = u0/native_rms((u,w))
    v = times(scale,(u,w))
    # Remove roundoff-only K=0 residual identically for every benchmark arm.
    v = minus(v,meanfield(v))
    v = times(u0/native_rms(v),v)
    return v,(;u0,K0,Te=1/(u0*K0))
end

function read_pair(path,nx,nz)
    data = reinterpret(Float64,read(path))
    @assert length(data) == nx*nz+nx*(nz+1)
    return reshape(copy(data[1:nx*nz]),nx,nz),reshape(copy(data[nx*nz+1:end]),nx,nz+1)
end

function write_pair(path,v)
    open(path,"w") do io
        write(io,Array(v[1])); write(io,Array(v[2]))
    end
end

function smooth_fixture(nx,nz;seed=1,mean_amplitude=0.0)
    g = Channel(nx,nz)
    rng = Xoshiro(seed); x = (0:nx-1).*g.dx; z = (0:nz).*g.dz
    psi = zeros(nx,nz+1)
    for m in 1:5,n in 1:5
        phase = 2π*rand(rng); amplitude = randn(rng)/(m*m+n*n)
        psi .+= amplitude.*cos.(2π*m/g.lx.*x .+phase).*transpose(sin.(π*n/g.h.*z))
    end
    psi[:,1] .= 0;psi[:,end] .= 0
    v = (-diff(psi;dims=2)./g.dz,(circshift(psi,(-1,0)).-psi)./g.dx)
    v = times(1/native_norm(v),v)
    zc = ((0:nz-1).+0.5).*g.dz
    v[1] .+= transpose(mean_amplitude.*(cos.(π.*zc./g.h).+cos.(3π.*zc./g.h)./3))
    return v
end

end # module
