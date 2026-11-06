module SpectralDiagnostics

# Static diagnostics only. This module contains no time-stepper or scheduler.
using ..DeepMechanism
using FFTW, Statistics, LinearAlgebra

export diagnostic_sample, write_diagnostics!, diagnostic_storage_plan,
       validate_cpu_diagnostics, validate_gpu_diagnostics

const REQUESTED_K = vcat([0.0],collect(0.25:0.25:4.0),[6.,8.,12.,16.,20.,24.,27.,32.])
const PARTS = (:stokes,:mean_related,:eddy_eddy,:viscosity,:constraint,:total)
const PRESSURE_PARTS = (:stokes,:mean_related,:eddy_eddy,:viscosity)

function mode_selection(g)
    k = [x for x in REQUESTED_K if round(Int,x*g.lx)<=g.nx÷2]
    indices = round.(Int,k.*g.lx).+1
    maximum(abs,indices.-1 .-k.*g.lx)<1e-12 || error("Selected K not on native lattice")
    return k,indices
end

function selected_pair(v,indices,nx)
    # Transform on the state's backend; copy only selected modes to the host.
    return (Array(rfft(v[1],1)[indices,:])./nx,
            Array(rfft(v[2],1)[indices,:])./nx)
end

function project_selected(f,k,g)
    u,w=f;theta=2π.*k.*g.dx
    forward=reshape((exp.(im.*theta).-1)./g.dx,:,1)
    backward=reshape((1 .-exp.(-im.*theta))./g.dx,:,1)
    rhs=forward.*u.+diff(w;dims=2)./g.dz
    ph=complex.(FFTW.r2r(real(rhs),FFTW.REDFT10,2),FFTW.r2r(imag(rhs),FFTW.REDFT10,2))
    lam=reshape(-4 .*sin.(theta./2).^2 ./g.dx^2,:,1).+
        reshape(-4 .*sin.(π.*collect(0:g.nz-1)./(2g.nz)).^2 ./g.dz^2,1,:)
    for j in findall(iszero,k)
        ph[j,1]=0;lam[j,1]=1
    end
    ph ./=lam
    p=complex.(FFTW.r2r(real(ph),FFTW.REDFT01,2),FFTW.r2r(imag(ph),FFTW.REDFT01,2))./(2g.nz)
    pu=-backward.*p;pw=zeros(ComplexF64,length(k),g.nz+1)
    pw[:,2:end-1].=-diff(p;dims=2)./g.dz
    projected=(u.+pu,w.+pw)
    residual=forward.*projected[1].+diff(projected[2];dims=2)./g.dz
    # A divergence-free raw force (notably viscosity) has only roundoff rhs;
    # normalize by its native derivative scale, not that roundoff divergence.
    # The separate selected/full pressure-operator comparison remains required.
    force_derivative_scale=maximum(abs,u)/g.dx+maximum(abs,w)/g.dz
    projection_error=maximum(abs,residual)/max(force_derivative_scale,maximum(abs,rhs),1e-30)
    projection_error<2e-10 || error("Selected pressure projection failed: $projection_error")
    return projected,p,(pu,pw),projection_error
end

function modal_work(v,f,mult)
    u,w=v;fu,fw=f
    ww=real.(conj.(w).*fw)
    return mult.*(real.(conj.(u).*fu).+(ww[:,1:end-1].+ww[:,2:end])./2)
end

function modal_curl(v,k,g;waves=false)
    u,w=v;q=zeros(ComplexF64,length(k),g.nz+1)
    q[:,2:end-1].=diff(u;dims=2)./g.dz
    q .-=reshape((1 .-exp.(-2π*im.*k.*g.dx))./g.dx,:,1).*w
    if waves
        sf=-.25 .* exp.(8 .*collect(range(-g.h,0;length=g.nz+1)))
        for j in findall(iszero,k);q[j,:].+=sf;end
    end
    return q
end

function diagnostic_sample(v,g,arm;waves=true,nu=1e-10,components=nothing,full_stress=false)
    arm in (:linear,:ql,:full) || error("Unknown arm")
    c=isnothing(components) ? DeepMechanism.components(v,g,arm;waves,nu) : components
    k,indices=mode_selection(g);nk=length(k)
    hu=rfft(v[1],1)./g.nx;hw=rfft(v[2],1)./g.nx
    base=(Array(hu[indices,:]),Array(hw[indices,:]))
    q=DeepMechanism.native_pv(v,g;waves)
    qhat=Array(rfft(q,1)[indices,:])./g.nx
    mult=reshape([i==1 || i==g.nx÷2+1 ? 1.0 : 2.0 for i in indices],:,1)
    data=Dict{String,Any}("selected_K_cycles"=>k,"selected_K_indices_1based"=>indices,
        "u_K"=>base[1],"w_K"=>base[2],"q_K"=>qhat,
        "normalization"=>"Complex coefficients FFT_x/Nx; no factor2 on coefficients. Real work/cospectra use multiplicity2 exceptK0 andNyquist.",
        "native_locations"=>"u:x-face,z-center;w:x-center,z-face;q:x-face,z-face. Native grid-index Fourier phases are retained.",
        "arm"=>string(arm),"waves"=>waves,
        "energy_K"=>modal_work(base,base,mult)./2)

    # Compute the exact all-K native stress on device. Dense files keep only
    # selected K, its exact all-K sum and the omitted-mode remainder.
    backend=v[1] isa Array ? :cpu : :gpu
    factor=reshape((1 .+exp.(-2π*im.*collect(0:g.nx÷2)./g.nx))./2,:,1)
    factor=DeepMechanism.backend_array(factor,backend)
    allmult=reshape(vcat([1.],fill(2.,g.nx÷2-1),[1.]),:,1)
    allmult=DeepMechanism.backend_array(allmult,backend)
    uz=hcat(hu[:,1:1],(hu[:,1:end-1].+hu[:,2:end])./2,hu[:,end:end])
    allR=allmult.*real.(uz.*conj.(factor.*hw))
    selectedR=Array(allR[indices,:]);totalR=vec(Array(sum(allR;dims=1)))
    physicalR=vec(Array(mean(DeepMechanism.u_at_z_faces(v[1]).*DeepMechanism.w_at_x_faces(v[2]);dims=1)))
    velocity_scale=max(DeepMechanism.native_inner(v,v),1e-30)
    stress_error=maximum(abs,totalR.-physicalR)/velocity_scale
    stress_error<2e-11 || error("Native stress co-spectrum mismatch")
    data["stress/selected"]=selectedR;data["stress/total"]=totalR
    data["stress/omitted"]=totalR.-vec(sum(selectedR;dims=1))
    if full_stress
        data["stress/all_K"]=Array(allR)
        data["stress/all_K_cycles"]=collect(0:g.nx÷2)./g.lx
    end

    # N_mean_related=N_actual-N_EE_actual, so no state-dependent operator is
    # silently replaced. In this channel it is mean/eddy advection (plus any
    # roundoff-sized mean/self term). QL retains EE only atK0; linear retainsnone.
    nraw=selected_pair(c.n,indices,g.nx)
    if arm==:linear
        ee=(zeros(ComplexF64,nk,g.nz),zeros(ComplexF64,nk,g.nz+1))
    else
        e=DeepMechanism.minus(v,DeepMechanism.meanfield(v))
        ee=selected_pair(DeepMechanism.advection(e,e,g),indices,g.nx)
        if arm==:ql
            ee[1][k.!=0,:].=0;ee[2][k.!=0,:].=0
        end
    end
    meanrelated=(nraw[1].-ee[1],nraw[2].-ee[2])
    raw=Dict(:stokes=>selected_pair(c.s,indices,g.nx),:mean_related=>meanrelated,
             :eddy_eddy=>ee,:viscosity=>selected_pair(c.d,indices,g.nx))
    projected=Dict{Symbol,Tuple{Matrix{ComplexF64},Matrix{ComplexF64}}}()
    max_projection_error=0.;max_pressure_flux_error=0.
    for part in PRESSURE_PARTS
        pp,p,grad,err=project_selected(raw[part],k,g)
        projected[part]=pp;max_projection_error=max(max_projection_error,err)
        pf=hcat(p[:,1:1],(p[:,1:end-1].+p[:,2:end])./2,p[:,end:end])
        flux=mult.*real.(pf.*conj.(base[2]))
        pressurework=modal_work(base,grad,mult)
        residual=pressurework.+diff(flux;dims=2)./g.dz
        # In QL the EE pressure has only K=0, where w and its work vanish.
        # Use a velocity-times-pressure-force scale as well as measured work;
        # never divide solely by cancellation-level K=0 work. No mode is skipped.
        pressure_force_scale=max(maximum(abs,grad[1]),maximum(abs,grad[2]),maximum(abs,p)/g.dz)
        characteristic_work=sqrt(velocity_scale)*pressure_force_scale*maximum(mult)
        scale=max(maximum(abs,pressurework),maximum(abs,diff(flux;dims=2)./g.dz),maximum(abs,modal_work(base,raw[part],mult)),characteristic_work,1e-30)
        data["checks/pressure_transport_scale/$(part)"]=scale
        data["checks/pressure_transport_absolute_residual/$(part)"]=maximum(abs,residual)
        # Velocity divergence is at roundoff in the native model; this is an
        # independent pressure transport identity, not integrated-work closure.
        ferr=maximum(abs,residual)/scale
        ferr<2e-8 || error("Selected pressure transport identity failed: $ferr")
        max_pressure_flux_error=max(max_pressure_flux_error,ferr)
        data["pressure_flux/$(part)"]=flux
    end
    projected[:constraint]=selected_pair(c.constraint,indices,g.nx)
    projected[:total]=selected_pair(c.total,indices,g.nx)
    summed=(sum(projected[x][1] for x in PARTS[1:5]),sum(projected[x][2] for x in PARTS[1:5]))
    total_scale=max(sqrt(sum(abs2,projected[:total][1])+sum(abs2,projected[:total][2])),1e-30)
    reconstruction=sqrt(sum(abs2,summed[1].-projected[:total][1])+sum(abs2,summed[2].-projected[:total][2]))/total_scale
    reconstruction<2e-10 || error("Projected components do not reconstruct actual arm RHS: $reconstruction")
    for part in PARTS
        data["projected/$(part)/u_K"]=projected[part][1]
        data["projected/$(part)/w_K"]=projected[part][2]
        data["work/$(part)"]=modal_work(base,projected[part],mult)
    end
    qerror=norm(qhat.-modal_curl(base,k,g;waves))/max(norm(qhat),1e-30)
    qerror<2e-11 || error("Selected native q does not match curl")
    nonzero=findall(x->!iszero(x),k)
    ee_nonzero=norm(projected[:eddy_eddy][1][nonzero,:])+norm(projected[:eddy_eddy][2][nonzero,:])
    if arm==:ql;ee_nonzero/total_scale<2e-11 || error("QL contains nonzeroK EE");end
    if arm==:linear
        (norm(projected[:mean_related][1])+norm(projected[:mean_related][2])+norm(projected[:eddy_eddy][1])+norm(projected[:eddy_eddy][2]))/total_scale<2e-11 || error("Linear arm has nonlinear work")
        (norm(projected[:total][1][1,:])+norm(projected[:total][2][1,:]))/total_scale<2e-11 || error("Linear held-mean tendency is nonzero")
    end
    data["checks/stress_relative"]=stress_error
    data["checks/selected_pressure_divergence_relative"]=max_projection_error
    data["checks/selected_pressure_divergence_normalization"]="max(maxabs(raw_u)/dx+maxabs(raw_w)/dz,maxabs(raw_divergence),1e-30); tolerance 2e-10. Independent selected/full pressure-operator agreement is also validated."
    data["checks/component_reconstruction_relative"]=reconstruction
    data["checks/q_curl_relative"]=qerror
    data["checks/pressure_flux_relative"]=max_pressure_flux_error
    data["interpretation"]="Component work is nativeK/depth work, not a standalone local cascade flux. Pressure vertical transport is retained; advection transport remains and is recomputable from fullnativefields. Mean-related includes the actual remainder after the arm-specific EE truncation."
    return data
end

write_diagnostics!(f,data)=(foreach(kv->(f[kv.first]=kv.second),data);nothing)

function diagnostic_storage_plan(nx=2048,nz=1024;stop_Te=100.,sample_Te=.1)
    nk=length([k for k in REQUESTED_K if round(Int,4k)<=nx÷2])
    complex_bytes=16nk*(nz+2(nz+1)+6(2nz+1))
    real_bytes=8nk*(nz+6nz+4(nz+1)+(nz+1))+16(nz+1)
    array_metadata_bytes=16nk # selected cycle coordinates and Int64 indices
    dense_count=round(Int,stop_Te/sample_Te)+1;full_count=round(Int,stop_Te)+1
    full_stress_bytes=8(nx÷2+1)*(nz+1)+8(nx÷2+1)
    return Dict("mode_count"=>nk,"dense_array_bytes_per_output"=>complex_bytes+real_bytes+array_metadata_bytes,
                "dense_output_count"=>dense_count,"dense_array_bytes_per_case"=>(complex_bytes+real_bytes+array_metadata_bytes)*dense_count,
                "full_stress_extra_bytes_per_case"=>full_count*full_stress_bytes,
                "note"=>"Uncompressed numeric-array bytes only; add JLD2metadata, driverprofiles, nativefields andcheckpoints separately.")
end

function static_check(backend)
    largest=0.;checks=Dict{String,Any}[]
    for (nx,nz) in ((32,16),(512,64)),arm in (:full,:ql,:linear),waves in (false,true)
        gc=DeepMechanism.Channel(nx,nz)
        vc=DeepMechanism.smooth_fixture(nx,nz;seed=7,mean_amplitude=arm==:linear ? 0. : .3)
        cc=DeepMechanism.components(vc,gc,arm;waves,nu=1e-10)
        ref=diagnostic_sample(vc,gc,arm;waves,nu=1e-10,components=cc,full_stress=true)
        # Independent full-domain projection of each untruncated raw component.
        k,ind=mode_selection(gc);rawS=DeepMechanism.project(cc.s,gc)[1]
        rawD=DeepMechanism.project(cc.d,gc)[1]
        for (part,pair) in ((:stokes,rawS),(:viscosity,rawD))
            expected=selected_pair(pair,ind,nx)
            actual=(ref["projected/$(part)/u_K"],ref["projected/$(part)/w_K"])
            scale=max(norm(expected[1])+norm(expected[2]),norm(ref["projected/total/u_K"])+norm(ref["projected/total/w_K"]),1e-30)
            err=(norm(actual[1]-expected[1])+norm(actual[2]-expected[2]))/scale
            err<2e-10 || error("Selected/full projection disagreement")
            largest=max(largest,err)
        end
        backend_error=0.
        if backend==:gpu
            g=DeepMechanism.Channel(nx,nz;backend=:gpu)
            v=tuple((DeepMechanism.backend_array(x,:gpu) for x in vc)...)
            c=DeepMechanism.components(v,g,arm;waves,nu=1e-10)
            actual=diagnostic_sample(v,g,arm;waves,nu=1e-10,components=c,full_stress=true)
            # Common scales avoid dividing roundoff-only zero channels byzero.
            scale=max(norm(ref["u_K"])+norm(ref["w_K"]),norm(ref["projected/total/u_K"])+norm(ref["projected/total/w_K"]),1e-30)
            for key in keys(ref)
                if ref[key] isa AbstractArray && eltype(ref[key])<:Number
                    reference_norm=norm(ref[key])
                    denominator=reference_norm>1e-11*scale ? reference_norm : scale
                    e=norm(actual[key]-ref[key])/denominator
                    backend_error=max(backend_error,e)
                end
            end
            backend_error<2e-8 || error("CPU/GPU diagnostic mapping failed: $backend_error")
            largest=max(largest,backend_error)
        end
        push!(checks,Dict("nx"=>nx,"nz"=>nz,"arm"=>string(arm),"waves"=>waves,"backend_relative_error"=>backend_error,
             "reconstruction"=>ref["checks/component_reconstruction_relative"],"pressure_flux"=>ref["checks/pressure_flux_relative"]))
    end
    return Dict("status"=>"PASS","PDE_time_integration"=>false,"backend"=>string(backend),"maximum_error"=>largest,"cases"=>checks)
end

validate_cpu_diagnostics()=static_check(:cpu)
function validate_gpu_diagnostics()
    isdefined(Main,:CUDA) || error("Load CUDA before GPU diagnostic gate")
    Main.CUDA.allowscalar(false)
    return static_check(:gpu)
end

end # module
