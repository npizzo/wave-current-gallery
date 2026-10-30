# Wind-specific helpers. The imposed stress and viscosity are defined for the
# prognostic Lagrangian momentum. No instantaneous current profile is relaxed.
@inline function onset_total_force(i,j,k,grid,clock,fields,p)
    # Retain the original prescribed-background maintenance unchanged. Wind
    # adds only a depth-uniform momentum return, which has zero curl.
    maintained_background_forcing(i,j,k,grid,clock,fields,p)+p.wind_return_acceleration
end
wind_trim_cells(nz)=max(2,cld(2nz,96))

function add_jet_perturbation!(u,h,mode,amplitude,sign)
    mode==0 || amplitude==0 || sign==0 || begin
        nz=size(u,3)
        shape=cos.(mode*pi.*((collect(1:nz).-0.5)./nz))
        shape .-= mean(shape) # Discrete zero transport, including roundoff.
        u .+= reshape(sign*amplitude.*shape,1,1,nz)
        return shape
    end
    zeros(size(u,3))
end

function write_wind_snapshot_metadata(path,manifest,parameters,sourcehashes)
    # TOML is a standard-library, non-executable parser. The JSON manifest is
    # the public interface; this companion binds it to the hashes and source.
    metadata=Dict{String,Any}(
        "format"=>"wind-native-snapshot-v1", "json_sha256"=>filehash(path),
        "time"=>manifest.time,"iteration"=>manifest.iteration,
        "byte_order"=>(Base.ENDIAN_BOM==0x04030201 ? "little" : "big"),
        "parameters"=>Dict(string(k)=>v for (k,v) in pairs(parameters)),
        "source_hashes"=>sourcehashes,
        "arrays"=>Dict(name=>Dict(string(k)=>v for (k,v) in pairs(entry)) for (name,entry) in manifest.arrays))
    companion=replace(path,r"\.json$"=>".toml")
    open(companion*".tmp","w") do io;TOML.print(io,metadata;sorted=true);end
    mv(companion*".tmp",companion;force=false)
end

# This import is intentionally restricted to one previously validated driver.
# A caller must also supply a hash-pinned provenance file for the exact snapshot.
# The strict, same-source snapshot path below remains the default.
const ONSET_PARENT_SOURCE_VERSION="da81ab96d7e51fb0"
const ONSET_PARENT_SOURCE_HASHES=Dict(
    "initial_conditions.jl"=>"3a8f83da2f569bd311b9765baef9b8cd32ce70871da929bf7a38496882e526fc",
    "jet_diagnostics.jl"=>"136071415aee9b9b1d187e9f06aad7325dde09ad39ee8e92a9a18733364be05e",
    "regime_diagnostics.jl"=>"c3ea43fd36dab77051b2ba48a990d465f09ee0cbfce1ce5f0fe45e49541be3b7",
    "regime_physics.jl"=>"cd097ff947b3d4d5a36729f1f93325f13e2fd192063a0839a57f142ce831202f",
    "run_jet.jl"=>"f963abf0de7bf3175bec357c639c5d0e5523f74a9bbedc92292cbb38ee580722",
    "vortex_modes.jl"=>"020d0f191d8f39ca71ad5b8d0445a2c9f057be83aee292c520b89c382781ffa4")

function verify_onset_parent_approval(path,metadata_path,metadata,kind,parameters,approval_path,source_id,approval_sha256)
    kind=="full" || error("Approved onset imports require the full native parent field")
    !isempty(source_id) || error("INIT_SOURCE_ID is required with INIT_APPROVAL")
    occursin(r"^[0-9a-f]{64}$",approval_sha256) || error("INIT_APPROVAL_SHA256 must pin the provenance file")
    isfile(approval_path) && filehash(approval_path)==approval_sha256 || error("Parent approval fingerprint mismatch")
    approval=TOML.parsefile(approval_path)
    approval["format"]=="wind-onset-parent-approval-v1" || error("Invalid parent approval format")
    approval["campaign"]=="unity-wind-onset-tests" || error("Parent approval belongs to another campaign")
    approval["approved_parent_source_version"]==ONSET_PARENT_SOURCE_VERSION || error("Parent source version is not approved")
    record=approval["sources"][source_id]
    record["source_hashes"]==ONSET_PARENT_SOURCE_HASHES || error("Approval source hashes do not match the approved historical driver")
    record["source_hashes"]==metadata["source_hashes"] || error("Parent numerical source differs from approval")
    record["parameters"]==metadata["parameters"] || error("Parent parameters differ from approval")
    record["arrays"]==metadata["arrays"] || error("Parent array manifest differs from approval")
    record["snapshot_json_sha256"]==filehash(path) || error("Parent snapshot differs from approval")
    record["snapshot_metadata_sha256"]==filehash(metadata_path) || error("Parent metadata differs from approval")
    record["snapshot_time"]==metadata["time"] || error("Parent physical time differs from approval")
    metadata["translation"]=="legacy-native-json-plus-original-configuration-v1" || error("Parent sidecar translation is not approved")
    record["parent_configuration_sha256"]==metadata["original_configuration_sha256"] || error("Parent configuration differs from approval")
    occursin(r"^[0-9a-f]{64}$",record["parent_configuration_sha256"]) || error("Missing original configuration fingerprint")
    record["expected_maintain_background"]===true && metadata["parameters"]["maintain"]===true && parameters.maintain===true ||
        error("Original background maintenance must remain enabled")
    old_ustar=metadata["parameters"]["ustar"]
    record["expected_parent_ustar"]==old_ustar || error("Parent wind stress differs from approval")
    old_ustar==0 || error("Onset parent must have zero added wind stress")
    allowed=record["allowed_branch_ustar"]
    length(allowed)==2 && Set(allowed)==Set([0.0,0.002]) || error("Approval must permit exactly zero and 0.002 m/s added wind")
    parameters.ustar in allowed || error("Requested wind change is not approved")
    return (;source_id,approval_sha256,parent_source_version=ONSET_PARENT_SOURCE_VERSION,
        parent_configuration_sha256=record["parent_configuration_sha256"],maintain_background=true,
        parent_ustar=Float64(old_ustar),branch_ustar=Float64(parameters.ustar),
        transition="retain original background gradients and maintenance; add top wind stress and uniform wind return only; no interior velocity reset")
end

function load_wind_snapshot(path,kind,parameters,sourcehashes;
                            metadata_path="",approval_path="",source_id="",approval_sha256="")
    kind in ("full","mean") || error("Invalid snapshot projection")
    path=abspath(path);endswith(path,".json") || error("INIT_SNAPSHOT must name a JSON snapshot manifest")
    isfile(path) || error("Snapshot manifest does not exist")
    isempty(metadata_path) || !isempty(approval_path) || error("INIT_METADATA override requires approved parent import")
    metadata_path=isempty(metadata_path) ? replace(path,r"\.json$"=>".toml") : abspath(metadata_path)
    metadata=TOML.parsefile(metadata_path)
    metadata["format"]=="wind-native-snapshot-v1" || error("Snapshot format mismatch")
    metadata["byte_order"]==(Base.ENDIAN_BOM==0x04030201 ? "little" : "big") || error("Snapshot byte order mismatch")
    metadata["json_sha256"]==filehash(path) || error("Snapshot JSON fingerprint mismatch")
    approved=nothing
    if isempty(approval_path)
        isempty(source_id) && isempty(approval_sha256) || error("Approval identifiers supplied without INIT_APPROVAL")
        metadata["source_hashes"]==sourcehashes || error("Snapshot numerical source differs from this driver")
    else
        approved=verify_onset_parent_approval(path,metadata_path,metadata,kind,parameters,approval_path,source_id,approval_sha256)
    end
    old=metadata["parameters"]
    for (key,value) in pairs(parameters)
        key==:ny && kind=="mean" && continue
        key==:ustar && approved!==nothing && continue
        stored=old[string(key)]
        (value isa Real ? isapprox(stored,value;rtol=1e-13,atol=1e-15) : stored==value) || error("Snapshot parameter mismatch: $key")
    end
    kind=="mean" && parameters.ny!=1 && error("Transverse-mean forecasts require NY=1")
    nx,ny,nz=Int(old["nx"]),Int(old["ny"]),Int(old["nz"])
    arrays=map(("u","v","w")) do name
        entry=metadata["arrays"][name];file=entry["file"]
        basename(file)==file || error("Snapshot array path must be a basename")
        target=joinpath(dirname(path),file)
        isfile(target) && filehash(target)==entry["sha256"] || error("Snapshot array fingerprint mismatch: $name")
        shape=Tuple(Int.(entry["shape"]))
        expected=name=="w" ? (nx,ny,nz+1) : (nx,ny,nz)
        shape==expected || error("Native snapshot staggering mismatch: $name")
        entry["dtype"]=="float64" && entry["order"]=="F" || error("Snapshot precision/order mismatch")
        filesize(target)==8prod(shape) || error("Truncated or oversized snapshot array")
        a=Array{Float64}(undef,shape);read!(target,a)
        all(isfinite,a)||error("Nonfinite snapshot array")
        kind=="mean" ? Array(mean(a;dims=2)) : a
    end
    return (;arrays,metadata,approved)
end

@inline function wind_roll_energy(i,j,k,g,Vx,Wx,Vxy,Wxy)
    ((Vx[1,j,k]-Vxy[1,1,k])^2+(Wx[1,j,k]-Wxy[1,1,k])^2)/2
end
const WIND_BUDGET_SCHEMA=["time_s","iteration","wind_ustar","kinematic_surface_stress",
    "UL_depth_mean","UL_depth_mean_change","native_depth_mean_tendency",
    "roll_energy_xmean","C_native_rms_full","C_native_rms_interior","C_native_rms_stokes",
    "q_eddy_rms_full","q_eddy_rms_interior","q_eddy_rms_stokes",
    "baseline_surface_stress","baseline_bottom_stress","total_surface_stress","total_bottom_stress",
    "background_force_depth_mean","wind_return_acceleration"]
mutable struct WindDiagnosticsState
    jets::Any
    regime::Any
    means::Any
    roll::Any
    io::IO
    initial_momentum::Union{Nothing,Float64}
    previous_time::Float64
end
function init_wind_diagnostics(jets,regime;resume=false)
    g=jets.model.grid;v=regime.centers.Vc;w=regime.centers.Wc
    means=(Vx=Field(Average(v;dims=1)),Wx=Field(Average(w;dims=1)),
           Vxy=Field(Average(v;dims=(1,2))),Wxy=Field(Average(w;dims=(1,2))))
    roll=Field(Average(KernelFunctionOperation{Center,Center,Center}(wind_roll_energy,g,
        map(f->f.data,means)...);dims=(1,2));compute=false)
    path=joinpath(jets.context.output,"wind-budget.csv");previous=-Inf;initial=nothing
    if resume
        lines,rows=JetDiagnostics.read_numeric_csv(path,WIND_BUDGET_SCHEMA)
        ids=findall(r->r["time_s"]<=jets.model.clock.time+1e-10,rows)
        isempty(ids)&&error("No wind diagnostics preceding checkpoint")
        cp(path,path*".pre-restart-"*string(time_ns()))
        write(path,join(vcat(lines[1],lines[ids.+1]),"\n")*"\n")
        previous=rows[last(ids)]["time_s"];initial=rows[first(ids)]["UL_depth_mean"]-rows[first(ids)]["UL_depth_mean_change"]
    else
        isfile(path)&&error("Refusing to overwrite wind diagnostics")
        write(path,join(WIND_BUDGET_SCHEMA,',')*"\n")
    end
    WindDiagnosticsState(jets,regime,means,roll,open(path,"a"),initial,previous)
end
function sample_wind_diagnostics!(sim,state,sample)
    t=Float64(time(sim));t<=state.previous_time+1e-10 && return nothing
    r=state.regime;g=state.jets.model.grid
    for f in state.means;compute!(f);end
    compute!(state.roll)
    roll=sum(Float64.(Array(interior(state.roll))))/g.Nz
    maps=Dict{String,Matrix{Float64}}()
    for name in ("Q","Qdot_advection_full","Qdot_advection_mean")
        f=r.maps[name];compute!(f);maps[name]=JetDiagnostics.matrix(f)
    end
    correction=maps["Qdot_advection_full"].-maps["Qdot_advection_mean"]
    qeddy=maps["Q"].-mean(maps["Q"];dims=1)
    z=state.jets.zcenters
    trim=wind_trim_cells(g.Nz);interior_ids=collect(trim+1:length(z)-trim)
    selections=(collect(eachindex(z)),interior_ids,[i for i in interior_ids if z[i]>=-3/(2r.p.k)])
    rms(a,ids)=isempty(ids) ? NaN : sqrt(mean(abs2,@view a[:,ids]))
    momentum=sum(sample.profiles["UL"].*state.jets.dz)/state.jets.context.H
    state.initial_momentum===nothing && (state.initial_momentum=momentum)
    ustar=state.jets.context.wind_ustar
    baseline_top=state.jets.context.baseline_gradient_top
    baseline_bottom=state.jets.context.baseline_gradient_bottom
    values=vcat([t,iteration(sim),ustar,ustar^2,momentum,momentum-state.initial_momentum,
        sum(sample.profiles["total"].*state.jets.dz)/state.jets.context.H,roll],
        [rms(correction,ids) for ids in selections],[rms(qeddy,ids) for ids in selections],
        [r.p.nu*baseline_top,r.p.nu*baseline_bottom,r.p.nu*r.p.gradient_top,r.p.nu*r.p.gradient_bottom,
         -r.p.nu*(baseline_top-baseline_bottom)/state.jets.context.H,-ustar^2/state.jets.context.H])
    all(isfinite,values)||error("Nonfinite wind diagnostic")
    JetDiagnostics.csvline(state.io,values);flush(state.io);state.previous_time=t
    values
end
close_wind_diagnostics!(state)=close(state.io)
