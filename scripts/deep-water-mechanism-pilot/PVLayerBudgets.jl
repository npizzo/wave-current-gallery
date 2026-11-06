module PVLayerBudgets

using Statistics

export pv_layer_budget, pv_profile_state, init_accumulator, accumulate!, snapshot,
       reset!, default_layers, PROFILE_KEYS, validate_diagnostics, validate_gpu_diagnostics

"All six densities are on q nodes: nx periodic x faces and nz+1 z faces."
const PROFILE_KEYS = (:mean_q, :half_q2, :half_mean_q2, :half_eddy_variance,
                      :half_global_variance, :half_mean_global_variance)
const DEFAULT_TERMS = (:advection, :wave, :viscosity, :pressure, :constraint)

profilemean(a) = mean(a; dims=1)
zeros_like(a, n, m) = fill!(similar(a, n, m), zero(eltype(a)))

function dual_weights(a, g)
    w = fill!(similar(a, 1, g.nz+1), g.dz)
    w[:,1:1] .= g.dz/2
    w[:,end:end] .= g.dz/2
    return w
end

weighted_mean(profile, g) = sum(profile .* dual_weights(profile,g))/g.h

"Pure native curl; neither state nor increment arguments are modified."
function diagnostic_q(v, g; waves=true)
    u,w = v
    size(u) == (g.nx,g.nz) || error("Unexpected u shape")
    size(w) == (g.nx,g.nz+1) || error("Unexpected w shape")
    zero_column = zeros_like(u,g.nx,1)
    uz = hcat(zero_column, diff(u;dims=2)./g.dz, zero_column)
    q = uz .- (w .- circshift(w,(1,0)))./g.dx
    return waves ? q .- g.shear_w : q
end

function geometry(g)
    zq = collect(range(-g.h,0;length=g.nz+1))
    interfaces = vcat(-g.h, -g.h .+ ((0:g.nz-1).+.5).*g.dz, 0.)
    return (; z_q=zq, z_interfaces=interfaces, dual_widths=diff(interfaces),
              h=g.h, dz=g.dz, nx=g.nx, nz=g.nz)
end

function profiles_from_q(q,g)
    Q = profilemean(q)
    μ = weighted_mean(Q,g)
    half_q2 = profilemean(q .* q)./2
    half_mean_q2 = Q .* Q ./ 2
    half_eddy_variance = half_q2 .- half_mean_q2
    half_global_variance = half_q2 .- μ .* Q .+ μ^2/2
    half_mean_global_variance = (Q .- μ).^2 ./ 2
    profiles = (;mean_q=Q,half_q2,half_mean_q2,half_eddy_variance,
                 half_global_variance,half_mean_global_variance)
    return (;profiles,global_mean=μ)
end

mapprofiles(f,args...) = NamedTuple{PROFILE_KEYS}(ntuple(j->f((a[j] for a in args)...),length(PROFILE_KEYS)))
addprofiles(a,b) = mapprofiles((x,y)->x.+y,a,b)
subprofiles(a,b) = mapprofiles((x,y)->x.-y,a,b)
zeroprofiles(a) = mapprofiles(x->zero.(x),a)

"Exact midpoint allocation of the quadratic-density change to one q increment."
function allocate(qmid,Qmid,μmid,dq,g)
    dQ = profilemean(dq)
    dμ = weighted_mean(dQ,g)
    half_q2 = profilemean(qmid .* dq)
    half_mean_q2 = Qmid .* dQ
    half_eddy_variance = half_q2 .- half_mean_q2
    reference = -(Qmid .- μmid) .* dμ
    half_global_variance = half_q2 .- μmid .* dQ .+ reference
    half_mean_global_variance = (Qmid .- μmid) .* dQ .+ reference
    values = (;mean_q=dQ,half_q2,half_mean_q2,half_eddy_variance,
               half_global_variance,half_mean_global_variance)
    return (;values,global_mean_increment=dμ,reference)
end

function reference_profiles(template, reference)
    z = zero.(template.mean_q)
    return (;mean_q=copy(z),half_q2=copy(z),half_mean_q2=copy(z),
              half_eddy_variance=copy(z),half_global_variance=copy(reference),
              half_mean_global_variance=copy(reference))
end

"Physical advective reference fluxes at dual-volume interfaces; not solver fluxes."
function interface_fluxes(v,q,g)
    w = v[2]
    wx = (w .+ circshift(w,(1,0)))./2
    wi = (wx[:,1:end-1].+wx[:,2:end])./2
    qi = (q[:,1:end-1].+q[:,2:end])./2
    z = zeros_like(q,1,1)
    return (; mean_q=hcat(z,profilemean(wi.*qi),z),
              half_q2=hcat(z,profilemean(wi.*qi.*qi)./2,z),
              mean_w=hcat(z,profilemean(wi),z))
end

dual_divergence(flux,g) = diff(flux;dims=2)./dual_weights(flux,g)
interface_mean(Q) = hcat(Q[:,1:1],(Q[:,1:end-1].+Q[:,2:end])./2,Q[:,end:end])

function default_layers(g)
    candidates = [("full",-g.h,0.),("top_stokes",-.125,0.),
                  ("next_stokes",-.25,-.125),("transition",-.5,-.25),
                  ("lower",-g.h,-.5),("top_quarter",-.25,0.),("top_half",-.5,0.),
                  ("candidate_mix",-.125,-.0625)]
    return [(;name,lower=max(-g.h,lo),upper=min(0.,hi))
            for (name,lo,hi) in candidates if min(0.,hi)>max(-g.h,lo)]
end

function hostcopy(x)
    x isa AbstractArray && return Array(x)
    x isa NamedTuple && return Dict(string(k)=>hostcopy(v) for (k,v) in pairs(x))
    x isa AbstractDict && return Dict(string(k)=>hostcopy(v) for (k,v) in pairs(x))
    x isa Tuple && return [hostcopy(v) for v in x]
    return x
end

"Piecewise-linear interface flux, consistent with partial dual-volume weights."
function value_at(z,flux,location)
    location <= z[1] && return flux[1]
    location >= z[end] && return flux[end]
    j = searchsortedlast(z,location)
    θ = (location-z[j])/(z[j+1]-z[j])
    return (1-θ)*flux[j]+θ*flux[j+1]
end

function layer_summaries(increments,initial,final,geom,layers)
    result=Dict{String,Any}()
    faces=geom.z_interfaces
    # Transfer each small profile once per output, never once per layer.
    hostincrements=Dict(name=>vec(Array(array)) for (name,array) in increments)
    hostinitial=Dict(k=>vec(Array(initial[k])) for k in PROFILE_KEYS)
    hostfinal=Dict(k=>vec(Array(final[k])) for k in PROFILE_KEYS)
    for layer in layers
        lo,hi=layer.lower,layer.upper
        weights=max.(0.,min.(faces[2:end],hi).-max.(faces[1:end-1],lo))
        depth=hi-lo
        integrals=Dict{String,Float64}();means=Dict{String,Float64}();fluxes=Dict{String,Any}()
        for (name,a) in hostincrements
            if length(a)==geom.nz+1
                val=sum(a.*weights)
                integrals[name]=val;means[name]=val/depth
            elseif length(a)==geom.nz+2
                bottom=value_at(faces,a,lo);top=value_at(faces,a,hi)
                fluxes[name]=Dict("lower"=>bottom,"upper"=>top,"net_inward"=>bottom-top)
            else
                error("Unknown profile length for $name")
            end
        end
        initial_means=Dict(string(k)=>sum(hostinitial[k].*weights)/depth for k in PROFILE_KEYS)
        final_means=Dict(string(k)=>sum(hostfinal[k].*weights)/depth for k in PROFILE_KEYS)
        result[layer.name]=Dict("lower"=>lo,"upper"=>hi,"depth"=>depth,
            "increment_integrals"=>integrals,"increment_means"=>means,
            "interface_values"=>fluxes,"initial_state_means"=>initial_means,"final_state_means"=>final_means)
    end
    return result
end

"Standalone t=0/snapshot diagnostics; no derivatives in time or model stepping."
function pv_profile_state(v,g;waves=true,host=false,layers=default_layers(g))
    q=diagnostic_q(v,g;waves)
    state=profiles_from_q(q,g)
    flux=interface_fluxes(v,q,g)
    geom=geometry(g)
    result=(;profiles=state.profiles,global_mean=state.global_mean,geometry=geom,
             instantaneous_fluxes=flux)
    if host
        answer=hostcopy(result)
        answer["layer_summaries"]=layer_summaries(Dict{String,Any}(),state.profiles,state.profiles,geom,layers)
        return answer
    end
    return result
end

"""
    pv_layer_budget(vold,vnew,increments,g; dt,waves,stages,qold,qnew,...)

All input arrays are read-only. `increments` contains already-computed,
stage-weighted velocity increments, including `:total`. `stages=(vold,a,b)`
selects SSPRK3 flux-reference weights (1/6,1/6,2/3); this function does not
evaluate a momentum operator or perform an RK/PDE step. Every array in the
returned `increments` dictionary is an accumulable increment or time-integrated
interface flux. Profiles are backend arrays; host transfer is deferred.
"""
function pv_layer_budget(vold,vnew,increments,g;dt,waves=true,stages=nothing,
                         qold=nothing,qnew=nothing,terms=DEFAULT_TERMS,
                         model_arm=:full,stage_weights=nothing)
    dt>0 || error("dt must be positive")
    haskey(increments,:total) || error("Expected existing :total velocity increment")
    all(haskey(increments,k) for k in terms) || error("Missing component increment")
    q0=qold===nothing ? diagnostic_q(vold,g;waves) : qold
    q1=qnew===nothing ? diagnostic_q(vnew,g;waves) : qnew
    old=profiles_from_q(q0,g);new=profiles_from_q(q1,g)
    qmid=(q0.+q1)./2
    Qmid=(old.profiles.mean_q.+new.profiles.mean_q)./2
    μmid=(old.global_mean+new.global_mean)/2
    observed=subprofiles(new.profiles,old.profiles)
    allocated=Dict{Symbol,Any}()
    dqsum=zero.(q0)
    for term in terms
        dq=diagnostic_q(increments[term],g;waves=false)
        allocated[term]=allocate(qmid,Qmid,μmid,dq,g)
        dqsum .+= dq
    end
    dqtotal=diagnostic_q(increments[:total],g;waves=false)
    allocated[:total]=allocate(qmid,Qmid,μmid,dqtotal,g)
    allocated[:component_sum_residual]=allocate(qmid,Qmid,μmid,dqtotal.-dqsum,g)
    allocated[:update_residual]=allocate(qmid,Qmid,μmid,q1.-q0.-dqtotal,g)
    actual_reference=-(Qmid.-μmid).*(new.global_mean-old.global_mean)
    reference=reference_profiles(observed,actual_reference)

    if stages===nothing
        flux_stages=(vold,vnew)
        weights=stage_weights===nothing ? (1/2,1/2) : stage_weights
        flux_quadrature="endpoint trapezoid; remainder includes flux time quadrature"
    else
        flux_stages=stages
        weights=stage_weights===nothing ? (1/6,1/6,2/3) : stage_weights
        flux_quadrature="supplied stage states with supplied/SSPRK3 weights"
    end
    length(flux_stages)==length(weights) || error("Stage and weight lengths differ")
    abs(sum(weights)-1)<1e-12 || error("Flux weights must sum to one")
    fluxint=nothing
    for (j,(v,weight)) in enumerate(zip(flux_stages,weights))
        q=(j==1 && v===vold) ? q0 : ((v===vnew) ? q1 : diagnostic_q(v,g;waves))
        f=interface_fluxes(v,q,g)
        if fluxint===nothing
            fluxint=(;mean_q=dt*weight.*f.mean_q,half_q2=dt*weight.*f.half_q2,mean_w=dt*weight.*f.mean_w)
        else
            fluxint.mean_q .+= dt*weight.*f.mean_q
            fluxint.half_q2 .+= dt*weight.*f.half_q2
            fluxint.mean_w .+= dt*weight.*f.mean_w
        end
    end
    tq=-dual_divergence(fluxint.mean_q,g)
    th=-dual_divergence(fluxint.half_q2,g)
    tm=Qmid.*tq
    te=th.-tm
    tg=th.-μmid.*tq
    tgm=(Qmid.-μmid).*tq
    transport=(;mean_q=tq,half_q2=th,half_mean_q2=tm,half_eddy_variance=te,
                 half_global_variance=tg,half_mean_global_variance=tgm)
    # This remainder is defined relative to the explicitly stored physical
    # flux reference. It includes explicit viscosity, constraints, deliberate
    # QL/linear omissions, temporal product quadrature and spatial discretization.
    remainder=subprofiles(subprofiles(observed,transport),reference)
    Qinterface=interface_mean(Qmid)
    meanflux=Qinterface.*fluxint.mean_q
    exchange=tm.+dual_divergence(meanflux,g)
    aw=addprofiles(allocated[:advection].values,allocated[:wave].values)
    aw_reference=reference_profiles(observed,allocated[:advection].reference.+allocated[:wave].reference)
    aw_remainder=subprofiles(subprofiles(aw,transport),aw_reference)

    saved=Dict{String,Any}()
    for (term,allocation) in allocated, key in PROFILE_KEYS
        saved["component_$(term)_$(key)"]=allocation.values[key]
    end
    for (prefix,values) in (("observed",observed),("transport",transport),
                            ("remainder",remainder),("reference",reference),
                            ("advection_wave_remainder",aw_remainder))
        for key in PROFILE_KEYS
            saved["$(prefix)_$(key)"]=values[key]
        end
    end
    saved["interface_mean_q"]=fluxint.mean_q
    saved["interface_half_q2"]=fluxint.half_q2
    saved["interface_half_mean_q2"]=meanflux
    saved["interface_half_eddy_variance"]=fluxint.half_q2.-meanflux
    saved["interface_half_global_variance"]=fluxint.half_q2.-μmid.*fluxint.mean_q
    saved["interface_half_mean_global_variance"]=(Qinterface.-μmid).*fluxint.mean_q
    saved["interface_mean_w"]=fluxint.mean_w
    saved["interface_fixed_reference_full_variance"]=fluxint.half_q2.-μmid.*fluxint.mean_q.+μmid^2/2 .*fluxint.mean_w
    saved["mean_eddy_exchange"]=exchange
    # Global variance flux uses a constant-flux gauge. The full physical flux
    # is also stored, so any numerical nonzero <w> is visible rather than hidden.
    metadata=Dict("model_arm"=>string(model_arm),"waves"=>waves,
        "density_quadrature"=>"q-node dual volumes: dz/2 at walls, dz in interior",
        "profile_convention"=>"half moments/variances; mean+eddy sums reproduce total",
        "flux_quadrature"=>flux_quadrature,
        "global_reference"=>"old/new global q mean; midpoint reference increment explicitly separated",
        "global_variance_flux_gauge"=>"half_q2 - mu_mid*mean_q; full fixed-reference flux additionally stores +mu_mid^2*mean_w/2",
        "remainder_meaning"=>"physical-flux reference remainder, not invented diffusion; includes viscosity, model omissions, space/time discretization and roundoff",
        "component_meaning"=>"exact midpoint quadratic allocations to existing stage-weighted velocity increments; not continuous work integrals")
    return (;increments=saved,old,new,geometry=geometry(g),dt,metadata)
end

mutable struct PVAccumulator
    increments::Dict{String,Any}
    initial::Any
    final::Any
    geometry::Any
    layers::Any
    metadata::Any
    elapsed::Float64
    nsteps::Int
end

copy_state(s) = (;profiles=mapprofiles(copy,s.profiles),global_mean=s.global_mean)

"Initialize zero interval storage; the supplied first budget is NOT accumulated."
function init_accumulator(b;layers=default_layers(b.geometry))
    values=Dict{String,Any}(k=>zero.(v) for (k,v) in b.increments)
    return PVAccumulator(values,copy_state(b.old),copy_state(b.old),b.geometry,
                         layers,copy(b.metadata),0.,0)
end

function accumulate!(acc::PVAccumulator,b)
    keys(acc.increments)==keys(b.increments) || error("PV budget key set changed")
    if acc.nsteps==0
        acc.initial=copy_state(b.old)
    end
    for (name,array) in b.increments
        acc.increments[name] .+= array
    end
    acc.final=copy_state(b.new)
    acc.elapsed+=b.dt;acc.nsteps+=1
    return acc
end

"Snapshot interval allocations and exact layer integrals; does not reset storage."
function snapshot(acc::PVAccumulator;host=true)
    values=Dict(k=>copy(v) for (k,v) in acc.increments)
    result=(;increments=values,initial=copy_state(acc.initial),final=copy_state(acc.final),
             geometry=acc.geometry,metadata=copy(acc.metadata),elapsed=acc.elapsed,nsteps=acc.nsteps)
    if host
        answer=hostcopy(result)
        answer["layer_summaries"]=layer_summaries(values,acc.initial.profiles,acc.final.profiles,acc.geometry,acc.layers)
        return answer
    end
    return result
end

function reset!(acc::PVAccumulator)
    for array in values(acc.increments);fill!(array,0);end
    acc.initial=copy_state(acc.final)
    acc.elapsed=0.;acc.nsteps=0
    return acc
end

# The following self-checks use manufactured arrays and fabricated increments
# only. They do not import DeepMechanism, evaluate RHS operators, call RK, or
# evolve a physical initial condition.
function manufactured_fixture()
    nx,nz=12,10;lx,h=2.,1.;dx,dz=lx/nx,h/nz
    x=reshape((0:nx-1).*dx,:,1)
    zc=reshape(-h.+((0:nz-1).+.5).*dz,1,:)
    zf=reshape(collect(range(-h,0;length=nz+1)),1,:)
    g=(;nx,nz,lx,h,dx,dz,shear_w=.25 .* exp.(8 .* zf))
    function stream(amplitude,m,n)
        ψ=amplitude.*cos.(2π*m.*x/lx).*sin.(π*n.*(zf.+h)/h)
        ψ[:,1].=0;ψ[:,end].=0
        return (-diff(ψ;dims=2)./dz,(circshift(ψ,(-1,0)).-ψ)./dx)
    end
    base=stream(.008,1,1)
    old=(base[1].+.015 .* cos.(π.*(zc.+h)/h),copy(base[2]))
    adv=stream(.00004,2,2)
    adv=(adv[1].+.0002 .* cos.(π.*(zc.+h)/h),adv[2])
    wave=stream(.00002,3,1)
    visc=(-.0003 .* old[1],-.0003 .* old[2])
    p=.00001 .* cos.(2π.*x/lx).*cos.(π.*(zc.+h)/h)
    pressure=((p.-circshift(p,(1,0)))./dx,
              hcat(zeros(nx,1),diff(p;dims=2)./dz,zeros(nx,1)))
    constraint=(fill(.00003,nx,nz),zeros(nx,nz+1))
    inc=Dict{Symbol,Any}(:advection=>adv,:wave=>wave,:viscosity=>visc,
                         :pressure=>pressure,:constraint=>constraint)
    total=(sum(inc[k][1] for k in DEFAULT_TERMS),sum(inc[k][2] for k in DEFAULT_TERMS))
    total=(total[1].+3e-15 .* cos.(π.*(zc.+h)/h),total[2])
    inc[:total]=total
    next=(old[1].+total[1].+7e-15 .* cos.(π.*(zc.+h)/h),old[2].+total[2])
    stagea=(old[1].+.37 .* total[1],old[2].+.37 .* total[2])
    stageb=(old[1].+.61 .* total[1],old[2].+.61 .* total[2])
    return (;old,next,inc,g,stages=(old,stagea,stageb),dt=.017)
end

function independent_q(v,g;waves=true)
    u,w=v;q=zeros(g.nx,g.nz+1)
    for j in 1:g.nz+1,i in 1:g.nx
        uz=1<j<g.nz+1 ? (u[i,j]-u[i,j-1])/g.dz : 0.
        prev=i==1 ? g.nx : i-1
        q[i,j]=uz-(w[i,j]-w[prev,j])/g.dx-(waves ? g.shear_w[1,j] : 0.)
    end
    return q
end

function independent_allocation(q0,q1,dq,g)
    qmid=(q0+q1)/2
    weights=vcat(.5,ones(g.nz-1),.5)./g.nz
    Qmid=vec(sum(qmid;dims=1))./g.nx
    dQ=vec(sum(dq;dims=1))./g.nx
    μmid=sum(Qmid.*weights);dμ=sum(dQ.*weights)
    ans=Dict(k=>zeros(1,g.nz+1) for k in PROFILE_KEYS)
    for j in 1:g.nz+1
        ans[:mean_q][1,j]=dQ[j]
        ans[:half_mean_q2][1,j]=Qmid[j]*dQ[j]
        ans[:half_mean_global_variance][1,j]=(Qmid[j]-μmid)*(dQ[j]-dμ)
        for i in 1:g.nx
            ans[:half_q2][1,j]+=qmid[i,j]*dq[i,j]/g.nx
            ans[:half_eddy_variance][1,j]+=(qmid[i,j]-Qmid[j])*(dq[i,j]-dQ[j])/g.nx
            ans[:half_global_variance][1,j]+=(qmid[i,j]-μmid)*(dq[i,j]-dμ)/g.nx
        end
    end
    return ans
end

function recursive_difference(a,b)
    if a isa AbstractDict
        Set(keys(a))==Set(keys(b)) || error("Diagnostic dictionaries differ")
        return maximum((recursive_difference(a[k],b[k]) for k in keys(a));init=0.)
    elseif a isa AbstractArray
        size(a)==size(b) || error("Diagnostic shapes differ")
        return maximum(abs,a.-b;init=0.)
    elseif a isa Number
        return abs(Float64(a)-Float64(b))
    else
        a==b || error("Diagnostic metadata differs")
        return 0.
    end
end

function run_manufactured(f; backend_label="cpu")
    q0=diagnostic_q(f.old,f.g);q1=diagnostic_q(f.next,f.g)
    before=[Array(a) for v in (f.old,f.next,f.stages...,values(f.inc)...) for a in v]
    qbefore=(Array(q0),Array(q1))
    b=pv_layer_budget(f.old,f.next,f.inc,f.g;dt=f.dt,waves=true,
                      stages=f.stages,qold=q0,qnew=q1)
    after=[Array(a) for v in (f.old,f.next,f.stages...,values(f.inc)...) for a in v]
    all(a==b for (a,b) in zip(before,after)) || error("Diagnostic mutated an input state or increment")
    qbefore==(Array(q0),Array(q1)) || error("Diagnostic mutated supplied PV arrays")
    abs(b.new.global_mean-b.old.global_mean)>1e-8 || error("Fixture failed to exercise moving global mean")
    layers=vcat(default_layers(b.geometry),[(name="cutcell",lower=-.413,upper=-.071)])
    acc=init_accumulator(b;layers)
    accumulate!(acc,b)
    snap=snapshot(acc;host=true)
    maxclosure=0.;maxtelescope=0.;maxsplit=0.
    for key in PROFILE_KEYS
        obs=Array(b.increments["observed_$(key)"])
        allocated=sum(Array(b.increments["component_$(term)_$(key)"])
                      for term in (DEFAULT_TERMS...,:component_sum_residual,:update_residual))
        closure=Array(b.increments["transport_$(key)"]).+
                Array(b.increments["remainder_$(key)"]).+
                Array(b.increments["reference_$(key)"])
        maxclosure=max(maxclosure,maximum(abs,obs.-allocated),maximum(abs,obs.-closure))
    end
    for prefix in ("observed","transport","remainder","reference",("component_$(k)" for k in keys(f.inc))...)
        a=Array(b.increments["$(prefix)_half_global_variance"])
        m=Array(b.increments["$(prefix)_half_mean_global_variance"])
        e=Array(b.increments["$(prefix)_half_eddy_variance"])
        maxsplit=max(maxsplit,maximum(abs,a.-m.-e))
    end
    for layer in values(snap["layer_summaries"]), key in ("mean_q","half_q2")
        tendency=layer["increment_integrals"]["transport_$key"]
        inward=layer["interface_values"]["interface_$key"]["net_inward"]
        maxtelescope=max(maxtelescope,abs(tendency-inward))
    end
    # A second fabricated update checks interval telescoping and accumulator
    # bookkeeping. This is not a model/RK step and calls no tendency function.
    next2=(f.next[1].+f.inc[:total][1],f.next[2].+f.inc[:total][2])
    b2=pv_layer_budget(f.next,next2,f.inc,f.g;dt=f.dt,waves=true,stages=(f.next,f.next,f.next))
    accumulate!(acc,b2)
    snap2=snapshot(acc;host=true)
    for key in PROFILE_KEYS
        difference=snap2["final"]["profiles"][string(key)].-snap2["initial"]["profiles"][string(key)]
        maxclosure=max(maxclosure,maximum(abs,difference.-snap2["increments"]["observed_$(key)"]))
    end
    snap2["nsteps"]==2 && snap2["elapsed"]==2*f.dt || error("Accumulator interval metadata failed")
    reset!(acc)
    reset=snapshot(acc;host=true)
    reset["nsteps"]==0 && reset["elapsed"]==0 || error("Accumulator reset failed")
    all(all(iszero,a) for a in values(reset["increments"])) || error("Accumulator retained increments")
    initial=pv_profile_state(f.old,f.g;waves=true,host=true,layers)
    maxclosure<5e-12 || error("Midpoint/profile closure failed: $maxclosure")
    maxtelescope<5e-12 || error("Layer interface telescoping failed: $maxtelescope")
    maxsplit<5e-12 || error("Mean/eddy split failed: $maxsplit")
    return (;budget=b,snapshot=snap2,initial,
             result=Dict("backend"=>backend_label,"status"=>"passed","no_PDE_or_RK_calls"=>true,
                         "max_midpoint_or_accumulator_error"=>maxclosure,"max_layer_telescoping_error"=>maxtelescope,
                         "max_mean_eddy_split_error"=>maxsplit,"input_nonmutation"=>true))
end

"CPU static diagnostic gate against separately looped midpoint allocations."
function validate_diagnostics()
    f=manufactured_fixture();tested=run_manufactured(f)
    q0=independent_q(f.old,f.g);q1=independent_q(f.next,f.g)
    maxerr=0.
    for term in (DEFAULT_TERMS...,:total)
        dq=independent_q(f.inc[term],f.g;waves=false)
        reference=independent_allocation(q0,q1,dq,f.g)
        for key in PROFILE_KEYS
            actual=tested.budget.increments["component_$(term)_$(key)"]
            maxerr=max(maxerr,maximum(abs,actual.-reference[key]))
        end
    end
    maxerr<5e-12 || error("Independent midpoint allocation failed: $maxerr")
    tested.result["max_independent_loop_allocation_error"]=maxerr
    return tested.result
end

"GPU/CPU static comparison with scalar GPU indexing disabled; no time evolution."
function validate_gpu_diagnostics()
    isdefined(Main,:CUDA) || error("Load CUDA in Main before the GPU diagnostic gate")
    cuda=getfield(Main,:CUDA)
    cuda.allowscalar(false)
    f=manufactured_fixture()
    cpu=run_manufactured(f)
    pair(v)=(cuda.CuArray(v[1]),cuda.CuArray(v[2]))
    gpuold=pair(f.old);gpunext=pair(f.next)
    gi=Dict(k=>pair(v) for (k,v) in f.inc)
    gg=merge(f.g,(shear_w=cuda.CuArray(f.g.shear_w),))
    gf=(;old=gpuold,next=gpunext,inc=gi,g=gg,stages=(gpuold,pair(f.stages[2]),pair(f.stages[3])),dt=f.dt)
    gpu=run_manufactured(gf;backend_label="gpu")
    err=max(recursive_difference(cpu.snapshot,gpu.snapshot),recursive_difference(cpu.initial,gpu.initial))
    err<5e-11 || error("CPU/GPU diagnostics differ: $err")
    answer=copy(gpu.result)
    answer["cpu_gpu_max_absolute_error"]=err
    answer["gpu_scalar_indexing_disabled"]=true
    return answer
end

end # module
