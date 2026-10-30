# Included inside JetDiagnostics. All 3-D operations and transverse reductions
# run on the model architecture; only scalars and x-z matrices reach the host.
export init_regime_diagnostics, sample_regime_diagnostics!, regime_maps, close_regime_diagnostics!,
       REGIME_BUDGET_SCHEMA

@inline ctr_u(i,j,k,g,u)=ℑxᶜᵃᵃ(i,j,k,g,u)
@inline ctr_v(i,j,k,g,v)=ℑyᵃᶜᵃ(i,j,k,g,v)
@inline ctr_w(i,j,k,g,w)=ℑzᵃᵃᶜ(i,j,k,g,w)
@inline previous_index(i,n)=i==1 ? n : i-1
@inline next_index(i,n)=i==n ? 1 : i+1

# Centred differences in periodic x/y; second-order one-sided differences at
# the first/last z cell. Maps are retained everywhere; continuum comparisons
# should exclude two z cells by default, and report that coverage explicitly.
@inline function dc(i,j,k,a,p,axis)
    if axis==1
        return (a[next_index(i,p.nx),j,k]-a[previous_index(i,p.nx),j,k])/(2p.dx)
    elseif axis==2
        p.ny==1 && return zero(eltype(a))
        return (a[i,next_index(j,p.ny),k]-a[i,previous_index(j,p.ny),k])/(2p.dy)
    elseif k==1
        return (-3a[i,j,1]+4a[i,j,2]-a[i,j,3])/(2p.dz)
    elseif k==p.nz
        return (3a[i,j,k]-4a[i,j,k-1]+a[i,j,k-2])/(2p.dz)
    else
        return (a[i,j,k+1]-a[i,j,k-1])/(2p.dz)
    end
end
@inline derivative_center(i,j,k,g,a,p,axis)=dc(i,j,k,a,p,axis)

@inline function stokes_gradient_face(k,p)
    k==1 && return 2p.k*p.S*exp(-2p.k*p.h)
    k==p.nz+1 && return 2p.k*p.S
    z=-p.h+(k-1)*p.dz
    return p.S*(exp(2p.k*(z+p.dz/2))-exp(2p.k*(z-p.dz/2)))/p.dz
end
@inline function q_velocity_face(i,j,k,u,w,p)
    uz=k==1 ? p.gradient_bottom : k==p.nz+1 ? p.gradient_top : (u[i,j,k]-u[i,j,k-1])/p.dz
    return uz-(w[i,j,k]-w[previous_index(i,p.nx),j,k])/p.dx-stokes_gradient_face(k,p)
end
@inline function q_velocity_center(i,j,k,g,u,w,p)
    ip=next_index(i,p.nx)
    return (q_velocity_face(i,j,k,u,w,p)+q_velocity_face(ip,j,k,u,w,p)+
            q_velocity_face(i,j,k+1,u,w,p)+q_velocity_face(ip,j,k+1,u,w,p))/4
end

# The velocity gradient and impermeability conditions are time independent.
# Their tendency is zero, so the native q_t at either z face is zero. Applying
# this common homogeneous tendency BC to each component defines the split;
# interior component curls do not use boundary extrapolation.
@inline function tendency_curl_face(i,j,k,u,w,p)
    (k==1 || k==p.nz+1) && return zero(eltype(u))
    return (u[i,j,k]-u[i,j,k-1])/p.dz-(w[i,j,k]-w[previous_index(i,p.nx),j,k])/p.dx
end
@inline function tendency_curl_center(i,j,k,g,u,w,p)
    ip=next_index(i,p.nx)
    return (tendency_curl_face(i,j,k,u,w,p)+tendency_curl_face(ip,j,k,u,w,p)+
            tendency_curl_face(i,j,k+1,u,w,p)+tendency_curl_face(ip,j,k+1,u,w,p))/4
end
@inline native_divergence_kernel(i,j,k,g,u,v,w,p)=
    (u[next_index(i,p.nx),j,k]-u[i,j,k])/p.dx+
    (p.ny==1 ? zero(eltype(u)) : (v[i,next_index(j,p.ny),k]-v[i,j,k])/p.dy)+
    (w[i,j,k+1]-w[i,j,k])/p.dz
@inline pressure_w(i,j,k,g,p)=-∂zᶜᶜᶠ(i,j,k,g,p)

@inline function scalar_term(i,j,k,g,u,v,w,q,p,term)
    qx=dc(i,j,k,q,p,1);qy=dc(i,j,k,q,p,2);qz=dc(i,j,k,q,p,3)
    term==1 && return u[i,j,k]*qx+w[i,j,k]*qz
    term==3 && return (dc(i,j,k,w,p,2)-dc(i,j,k,v,p,3))*dc(i,j,k,v,p,1)
    term==4 && return q[i,j,k]*dc(i,j,k,v,p,2)
    term==5 && return (dc(i,j,k,v,p,1)-dc(i,j,k,u,p,2))*dc(i,j,k,v,p,3)
    term==6 && return -v[i,j,k]*qy
    return (dc(i,j,k,w,p,2)-dc(i,j,k,v,p,3))*dc(i,j,k,v,p,1)+
        q[i,j,k]*dc(i,j,k,v,p,2)+
        (dc(i,j,k,v,p,1)-dc(i,j,k,u,p,2))*dc(i,j,k,v,p,3)-v[i,j,k]*qy
end
@inline function gradient_square(i,j,k,g,u,v,w,u0,v0,w0,p,perp)
    result=zero(eltype(u))
    for axis in 1:3
        du=dc(i,j,k,u,p,axis);dv=dc(i,j,k,v,p,axis);dw=dc(i,j,k,w,p,axis)
        if perp
            du-=dc(i,j,k,u0,p,axis);dv-=dc(i,j,k,v0,p,axis);dw-=dc(i,j,k,w0,p,axis)
        end
        result+=du^2+dv^2+dw^2
    end
    return result
end

const REGIME_BUDGET_SCHEMA=["time_s","iteration","Etotal","Eeddy","Eky0","Eky0_eddy","Eperp",
    "chi_3d","eddy_rms","perp_rms","grad_velocity_rms","grad_3d_rms","qE_rms",
    "native_divergence_max","native_divergence_rms","native_pressure_curl_max","viscous_cfl"]

mutable struct RegimeDiagnosticsState
    jets::Any
    p::Any
    centers::Any
    basecenters::Any
    maps::Any
    native_means::Any
    divergence::Any
    pressure_curl::Any
    gradients::Any
    io::IO
    previous_time::Float64
end

function init_regime_diagnostics(jets;resume=false)
    Base.@nospecialize jets
    m=jets.model;g=m.grid;c=jets.context
    dx=c.Lx/g.Nx;dy=c.Ly/g.Ny;dz=c.H/g.Nz
    # These must match the driver's imposed gradients; old no-forcing diagnostic
    # tests retain the same alpha convention through the default S_reference.
    reference=hasproperty(c,:S_reference) ? c.S_reference : c.S
    waves=hasproperty(c,:waves) ? c.waves : 1
    nu=hasproperty(c,:nu) ? c.nu : 0.
    p=(;nx=g.Nx,ny=g.Ny,nz=g.Nz,dx,dy,dz,h=c.H,k=c.k,S=c.S,nu,
       gradient_bottom=hasproperty(c,:gradient_bottom) ? c.gradient_bottom : (waves-c.alpha)*2c.k*reference*exp(-2c.k*c.H),
       gradient_top=hasproperty(c,:gradient_top) ? c.gradient_top : (waves-c.alpha)*2c.k*reference,
       baseline_gradient_bottom=hasproperty(c,:baseline_gradient_bottom) ? c.baseline_gradient_bottom : (waves-c.alpha)*2c.k*reference*exp(-2c.k*c.H),
       baseline_gradient_top=hasproperty(c,:baseline_gradient_top) ? c.baseline_gradient_top : (waves-c.alpha)*2c.k*reference)
    u,v,w=m.velocities;o=jets.operators
    center(f,a)=Field(KernelFunctionOperation{Center,Center,Center}(f,g,a))
    centers=(Uc=center(ctr_u,u),Vc=center(ctr_v,v),Wc=center(ctr_w,w),
       Q=Field(KernelFunctionOperation{Center,Center,Center}(q_velocity_center,g,u,w,p)))
    basecenters=(U=center(ctr_u,o.u0),V=center(ctr_v,o.v0),W=center(ctr_w,o.w0))
    u1,v1,w1,q=centers
    ymean(a)=Field(Average(a;dims=2))
    maps=Dict{String,Any}(string(k)=>ymean(a) for (k,a) in pairs(centers))
    for (name,axis) in (("Qx",1),("Qz",3))
        maps[name]=ymean(Field(KernelFunctionOperation{Center,Center,Center}(derivative_center,g,q,p,axis)))
    end
    for (name,term) in (("A",1),("Rbar",2),("R_tilt_x",3),("R_stretch",4),("R_tilt_z",5),("R_adv_y",6))
        maps[name]=ymean(Field(KernelFunctionOperation{Center,Center,Center}(scalar_term,g,u1,v1,w1,q,p,term)))
    end
    native=o.native
    for (name,uk,wk) in (("Qdot_advection_full",:U_adv_full,:W_adv_full),
            ("Qdot_advection_mean",:U_adv_mean,:W_adv_mean),("Qdot_wave",:U_wave,:W_wave),
            ("Qdot_viscous",:U_viscous,:W_viscous),("Qdot_forcing",:U_forcing,:W_forcing))
        maps[name]=ymean(Field(KernelFunctionOperation{Center,Center,Center}(tendency_curl_center,g,
            getproperty(native,uk),getproperty(native,wk),p)))
    end
    native_means=Dict{String,Any}(string(k)=>ymean(a) for (k,a) in pairs(native) if !(k in (:V_adv_full,:V_viscous)))
    div=Field(KernelFunctionOperation{Center,Center,Center}(native_divergence_kernel,g,u,v,w,p))
    pu=Field(KernelFunctionOperation{Face,Center,Center}(native_pressure_u,g,m.pressures.pNHS))
    pw=Field(KernelFunctionOperation{Center,Center,Face}(pressure_w,g,m.pressures.pNHS))
    pc=Field(KernelFunctionOperation{Center,Center,Center}(tendency_curl_center,g,pu,pw,p))
    gradients=(all=Field(KernelFunctionOperation{Center,Center,Center}(gradient_square,g,u1,v1,w1,basecenters...,p,false)),
               perp=Field(KernelFunctionOperation{Center,Center,Center}(gradient_square,g,u1,v1,w1,basecenters...,p,true)))
    path=joinpath(c.output,"regime-budget.csv");previous=-Inf
    if resume
        lines,rows=read_numeric_csv(path,REGIME_BUDGET_SCHEMA)
        ids=findall(r->r["time_s"]<=m.clock.time+1e-10,rows)
        isempty(ids)&&error("No regime diagnostics preceding checkpoint")
        cp(path,path*".pre-restart-"*string(time_ns()))
        write(path,join(vcat(lines[1],lines[ids.+1]),"\n")*"\n")
        previous=rows[last(ids)]["time_s"]
    else
        isfile(path)&&error("Refusing to overwrite regime budget")
        write(path,join(REGIME_BUDGET_SCHEMA,',')*"\n")
    end
    RegimeDiagnosticsState(jets,p,centers,basecenters,maps,native_means,div,pc,gradients,open(path,"a"),previous)
end

function update_regime_centers!(state)
    for f in state.centers;compute!(f);end
    for f in state.basecenters;compute!(f);end
end

function sample_regime_diagnostics!(sim,state,sample)
    t=Float64(time(sim));t<=state.previous_time && return nothing
    update_regime_centers!(state)
    compute!(state.divergence);compute!(state.pressure_curl)
    for f in state.gradients;compute!(f);end
    p=state.p;s=sample.scalars
    rms(f)=sqrt(Float64(sum(abs2,interior(f)))/length(interior(f)))
    row=Dict{String,Float64}(k=>s[k] for k in ("Etotal","Eeddy","Eky0","Eky0_eddy","Eperp"))
    merge!(row,Dict("time_s"=>t,"iteration"=>iteration(sim),
        "chi_3d"=>s["Eeddy"]>0 ? s["Eperp"]/s["Eeddy"] : NaN,
        "eddy_rms"=>sqrt(max(0,2s["Eeddy"])),"perp_rms"=>sqrt(max(0,2s["Eperp"])),
        "grad_velocity_rms"=>sqrt(Float64(sum(interior(state.gradients.all)))/length(interior(state.gradients.all))),
        "grad_3d_rms"=>sqrt(Float64(sum(interior(state.gradients.perp)))/length(interior(state.gradients.perp))),
        "qE_rms"=>rms(state.centers.Q),"native_divergence_max"=>Float64(maximum(abs,interior(state.divergence))),
        "native_divergence_rms"=>rms(state.divergence),
        "native_pressure_curl_max"=>Float64(maximum(abs,interior(state.pressure_curl))),
        "viscous_cfl"=>sim.Δt*p.nu*(p.dx^-2+(p.ny==1 ? 0. : p.dy^-2)+p.dz^-2)))
    csvline(state.io,[row[k] for k in REGIME_BUDGET_SCHEMA]);flush(state.io);state.previous_time=t
    row
end

function regime_maps(state)
    update_regime_centers!(state)
    # Fresh native tendencies are not cached Runge--Kutta-stage derivatives.
    for f in state.jets.operators.native;compute!(f);end
    maps=Dict{String,Matrix{Float64}}()
    for (name,f) in state.maps
        compute!(f);maps[name]=matrix(f)
    end
    for (name,f) in state.native_means
        compute!(f);maps[name]=matrix(f)
    end
    for (name,f) in pairs(state.jets.operators.ym)
        compute!(f);maps[string(name)]=matrix(f)
    end
    maps["Qdot_total"]=maps["Qdot_advection_full"].+maps["Qdot_wave"].+maps["Qdot_viscous"].+maps["Qdot_forcing"]
    maps["Qdot_3d_native"]=maps["Qdot_advection_full"].-maps["Qdot_advection_mean"]
    maps["continuum_scalar_residual"]=maps["Qdot_total"].+maps["A"].-maps["Rbar"].-maps["Qdot_viscous"].-maps["Qdot_forcing"]
    maps["continuum_R_identity_error"]=maps["Rbar"].-maps["R_tilt_x"].-maps["R_stretch"].-maps["R_tilt_z"].-maps["R_adv_y"]
    all(a->all(isfinite,a),values(maps))||error("Nonfinite regime map")
    maps
end

close_regime_diagnostics!(state)=close(state.io)
