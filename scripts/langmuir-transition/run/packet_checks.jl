"""Read-only verification of original packet-campaign output. No restart is implied."""
module PacketChecks

using JLD2
using Oceananigans
using SHA

export verify_run, verify_initial_condition, native_state_metrics, write_json_atomic

require(condition, message) = condition || error(message)
getmeta(meta, name) = meta isa NamedTuple ? getproperty(meta, Symbol(name)) : meta[name]
finite(x) = x isa Number ? isfinite(x) : all(isfinite, x)

function json_string(s)
    text = replace(string(s), "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n", "\r" => "\\r", "\t" => "\\t")
    return "\"" * text * "\""
end

function json_value(x)
    x === nothing && return "null"
    x isa Bool && return x ? "true" : "false"
    if x isa Number
        require(isfinite(x), "Cannot serialize nonfinite diagnostic")
        return string(x)
    elseif x isa AbstractString || x isa Symbol
        return json_string(x)
    elseif x isa NamedTuple
        return json_value(Dict(string(k) => v for (k, v) in pairs(x)))
    elseif x isa AbstractDict
        return "{" * join((json_string(k) * ":" * json_value(v) for (k, v) in sort!(collect(x); by=p -> string(first(p)))), ",") * "}"
    elseif x isa Tuple || x isa AbstractArray
        return "[" * join(json_value.(collect(x)), ",") * "]"
    end
    error("Unsupported JSON value $(typeof(x))")
end

function write_json_atomic(path, value)
    require(!isfile(path), "Refusing to overwrite report $path")
    mkpath(dirname(abspath(path)))
    temp = path * ".partial-$(getpid())"
    open(temp, "w") do io
        println(io, json_value(value))
    end
    mv(temp, path; force=false)
    return path
end

function native_state_metrics(u, v, w, Lx, Ly, dz; bounded_x=true)
    Nx, Ny, Nz = size(v)
    require(size(u) == (Nx + Int(bounded_x), Ny, Nz), "u is not a native x-face array")
    require(size(w) == (Nx, Ny, Nz + 1), "w is not a native z-face array")
    require(length(dz) == Nz && all(>(0), dz), "Invalid vertical cell widths")
    require(finite(u) && finite(v) && finite(w), "Nonfinite velocity")
    umax = max(maximum(abs, u), maximum(abs, v), maximum(abs, w))
    dx, dy = Float64(Lx) / Nx, Float64(Ly) / Ny
    maxdiv = 0.0
    # Evaluate the native finite-volume divergence without allocating another full volume.
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ip = bounded_x ? i + 1 : mod1(i + 1, Nx)
        jp = mod1(j + 1, Ny)
        d = (Float64(u[ip,j,k]) - Float64(u[i,j,k])) / dx +
            (Float64(v[i,jp,k]) - Float64(v[i,j,k])) / dy +
            (Float64(w[i,j,k+1]) - Float64(w[i,j,k])) / Float64(dz[k])
        maxdiv = max(maxdiv, abs(d))
    end
    wall_w = max(maximum(abs, @view(w[:,:,1])), maximum(abs, @view(w[:,:,end])))
    wall_u = bounded_x ? max(maximum(abs, @view(u[1,:,:])), maximum(abs, @view(u[end,:,:]))) : 0.0
    scaled = umax == 0 ? maxdiv : maxdiv * min(dx, dy, minimum(dz)) / umax
    return (; scaled_native_divergence_max=scaled, divergence_max=maxdiv,
              max_velocity=Float64(umax), max_wall_u=Float64(wall_u), max_wall_w=Float64(wall_w))
end

function vertical_widths(Nz, Lz)
    h(k) = (k - 1) / Nz
    z(k) = Lz * ((1 + (h(k) - 1) / 1.5) * (1 - exp(-8h(k))) / (1 - exp(-8)) - 1)
    # The original grid stores coordinates in Float32. Subtraction here follows those stored coordinates.
    return Float64.(diff(Float32[z(k) for k in 1:Nz+1]))
end

function check_state(metrics; tolerance=2e-4)
    require(metrics.scaled_native_divergence_max <= tolerance,
            "Native divergence too large: $(metrics.scaled_native_divergence_max)")
    require(metrics.max_wall_u <= 1e-6 && metrics.max_wall_w <= 1e-6, "Impermeable boundary condition failed")
end

function check_baseline(meta; initial_condition=false)
    require(getmeta(meta, "case") == "1.D" || (!initial_condition && getmeta(meta, "case").name == "1.D"), "Wrong experimental case")
    require(getmeta(meta, "level") == "M2", "Not an M2 run")
    require((getmeta(meta, "Nx"), getmeta(meta, "Ny"), getmeta(meta, "Nz")) == (1536,128,128), "Incorrect M2 dimensions")
    require(getmeta(meta, "x_topology") == "bounded", "Expected bounded tank")
    require(getmeta(meta, "FT") == "Float32", "Expected Float32")
    require(getmeta(meta, "numerics") == "weno", "Expected WENO5 implicit LES")
    require(isapprox(getmeta(meta, "Lx"),12) && isapprox(getmeta(meta,"Ly"),0.8) && isapprox(getmeta(meta,"Lz"),0.4; atol=1e-7), "Incorrect tank dimensions")
    require(getmeta(meta, "oceananigans") == "0.107.4", "Unexpected Oceananigans version")
    if initial_condition
        require(getmeta(meta,"seed") in (1,2,3,4,6,7,8), "Seed is not in the original seven-seed benchmark")
        require(all(isapprox.(collect(getmeta(meta,"amplitude")), [2.85,0.99,2.90]; atol=1e-10)), "Incorrect bounded-tank amplitude calibration")
        require(getmeta(meta,"L_factor") == 1.6, "Incorrect bounded-tank integral-scale calibration")
        require(getmeta(meta,"precondition_time") == 2.0, "Incorrect preconditioning duration")
        require(getmeta(meta,"Δt") == 0.02, "Incorrect preconditioning timestep")
        require(getmeta(meta,"projection_iterations") == 3 && getmeta(meta,"cutoff_cells") == 6 && getmeta(meta,"taper_depth") == 0.05, "Incorrect turbulence preparation")
    else
        require(getmeta(meta,"member") in ("packet_turbulence","turbulence_control","packet_null","quiescent_control"), "Unexpected campaign member")
        require(getmeta(meta,"Δt") in (0.02,0.01), "Unexpected production timestep")
        require(getmeta(meta,"output_interval") == 0.1, "Unexpected output interval")
        require(getmeta(meta,"remove_mean_transport") === true, "Original initialization changed")
        require(getmeta(meta,"σ_upstream") == 4 && getmeta(meta,"x_FOV") == 6.0, "Original packet trajectory changed")
        require(getmeta(meta,"shear_amplitude") == 0 && getmeta(meta,"noise_amplitude") == 0 && getmeta(meta,"wind_stress") == 0, "Extra forcing or shear in baseline")
        c = getmeta(meta,"case")
        require(isapprox(c.k,9.3;atol=1e-6) && isapprox(c.steepness,0.22;atol=1e-7) && isapprox(c.τ₀,2.8;atol=1e-6), "Incorrect wave parameters")
        p = getmeta(meta,"packet")
        require(p.periodic == 0, "Bounded packet unexpectedly includes periodic images")
        expected = Float64((p.x_end - p.x₀) / p.cᵍ)
        require(isapprox(getmeta(meta,"stop_time"),expected; atol=1e-6), "Run was shortened or extended")
        require(length(getmeta(meta,"snapshot_iterations")) == 6, "Original six snapshot times required")
        if getmeta(meta,"has_turbulence")
            check_baseline(getmeta(meta,"ic_metadata"); initial_condition=true)
            require(getmeta(meta,"seed") == getmeta(getmeta(meta,"ic_metadata"),"seed"), "Run/IC seed mismatch")
        end
    end
end

function verify_initial_condition(path; strict=true)
    meta, metrics = jldopen(path, "r"; iotype=IOStream) do f
        m = f["metadata"]
        strict && check_baseline(m; initial_condition=true)
        u, v, w = f["u"], f["v"], f["w"]
        require(size(v) == (m.Nx,m.Ny,m.Nz), "IC grid differs from metadata")
        r = native_state_metrics(u,v,w,m.Lx,m.Ly,vertical_widths(m.Nz,m.Lz); bounded_x=m.x_topology == "bounded")
        check_state(r)
        return m, r
    end
    return (; path=abspath(path), level=meta.level, seed=meta.seed, metrics)
end

function time_entries(f)
    its = sort(parse.(Int, collect(keys(f["timeseries/t"]))))
    require(!isempty(its), "Output has no time samples")
    ts = Float64[f["timeseries/t/$i"] for i in its]
    require(all(isfinite,ts) && all(>(0),diff(ts)), "Output times are not finite and strictly increasing")
    require(abs(first(ts)) < 1e-8, "Initial output is missing")
    return its, ts
end

function expected_clock_times(::Type{T}, stop, dt, iterations, field_type=Float32) where T <: AbstractFloat
    # Simulation converts its timestep and stop time to the field precision, which
    # can differ from the clock precision. Match that conversion and clock rounding.
    target = T(field_type(stop))
    step = Float64(field_type(dt))
    t = zero(T)
    times = Vector{Float64}(undef, iterations + 1)
    times[1] = 0.0
    for it in 1:iterations
        t = T(min(Float64(t) + step, Float64(target)))
        times[it+1] = Float64(t)
    end
    return times
end

function verify_run(dir; strict=true)
    files = ("metadata.jld2","run_summary.jld2","statistics.jld2","y_averages.jld2","fov_plane.jld2","snapshots.jld2")
    for file in files
        require(isfile(joinpath(dir,file)), "Missing $file in $dir")
    end
    meta = jldopen(f -> Dict(k => f[k] for k in keys(f)), joinpath(dir,"metadata.jld2"), "r"; iotype=IOStream)
    summary = jldopen(f -> Dict(k => f[k] for k in keys(f)), joinpath(dir,"run_summary.jld2"), "r"; iotype=IOStream)
    strict && check_baseline(meta)
    stop, dt = Float64(meta["stop_time"]), Float64(meta["Δt"])
    final = Float64(summary["final_time"])
    require(isfinite(final) && stop - 1e-7 <= final <= stop + dt + 1e-6, "Run did not complete its requested integration")
    require(summary["iterations"] > 0 && abs(summary["iterations"] * dt - final) < dt + 1e-5, "Inconsistent final iteration/time")
    require(isfinite(summary["wall_time"]) && summary["wall_time"] > 0, "Invalid recorded runtime")
    for key in ("rms_initial","rms_final"), value in values(summary[key])
        require(isfinite(value) && value >= 0, "Invalid $key")
    end
    counts = Dict{String,Int}()
    for name in ("statistics","y_averages","fov_plane")
        jldopen(joinpath(dir,name*".jld2"), "r"; iotype=IOStream) do f
            its, ts = time_entries(f)
            expected = collect(0:meta["n_out"]:summary["iterations"])
            require(its == expected, "$name has missing or unexpected output iterations")
            T = typeof(f["timeseries/t/0"])
            require(T <: AbstractFloat, "Unexpected clock type $T")
            FT = meta["FT"] == "Float32" ? Float32 : Float64
            expected_times = expected_clock_times(T,stop,dt,summary["iterations"],FT)[its .+ 1]
            clock_tolerance = max(1e-5, 4eps(T)*max(abs(final),1))
            require(maximum(abs.(ts .- expected_times)) <= clock_tolerance, "$name time grid is inconsistent")
            counts[name] = length(its)
            if name == "statistics"
                for key in ("u_mean","v_mean","u_rms","v_rms","w_rms","w_max"), it in its
                    require(finite(f["timeseries/$key/$it"]), "Nonfinite $key at iteration $it")
                end
            else
                fields = name == "y_averages" ? ("U","V","W","UU","VV","WW","UW") : ("u","v","w")
                for key in fields
                    require(finite(f["timeseries/$key/$(last(its))"]), "Nonfinite last $name $key")
                end
            end
        end
    end
    metrics, last_snapshot_time = jldopen(joinpath(dir,"snapshots.jld2"), "r"; iotype=IOStream) do f
        its, ts = time_entries(f)
        expected = sort(unique(vcat(0,meta["snapshot_iterations"])))
        require(its == expected, "Missing or unexpected 3D snapshots")
        T = typeof(f["timeseries/t/0"])
        require(T <: AbstractFloat, "Unexpected snapshot clock type $T")
        FT = meta["FT"] == "Float32" ? Float32 : Float64
        expected_times = expected_clock_times(T,stop,dt,summary["iterations"],FT)[its .+ 1]
        require(maximum(abs.(ts .- expected_times)) <= max(1e-5,4eps(T)*max(abs(final),1)), "Snapshot time grid is inconsistent")
        it = last(its)
        u,v,w = (f["timeseries/$name/$it"] for name in ("u","v","w"))
        require(size(v) == (meta["Nx"],meta["Ny"],meta["Nz"]), "Snapshot grid differs from metadata")
        r = native_state_metrics(u,v,w,meta["Lx"],meta["Ly"],vertical_widths(meta["Nz"],meta["Lz"]); bounded_x=meta["x_topology"] == "bounded")
        check_state(r)
        counts["snapshots"] = length(its)
        return r, last(ts)
    end
    if meta["has_turbulence"]
        ic = meta["initial_condition"]
        require(isfile(ic), "Original initial-condition file is missing: $ic")
        checksum = open(io -> bytes2hex(sha256(io)), ic)
        require(checksum == meta["initial_condition_sha256"], "Initial-condition checksum changed")
    end
    return (; directory=abspath(dir), member=meta["member"], level=meta["level"], seed=meta["seed"], final_time=final,
              iterations=summary["iterations"], counts, last_snapshot_time, metrics,
              scope="Completion, stored scalar history and latest saved native 3D state; not a resolution-convergence test.")
end

function main(args)
    parsed = Dict(split(a,'='; limit=2)[1] => split(a,'='; limit=2)[2] for a in args)
    result = if haskey(parsed,"directory")
        verify_run(parsed["directory"])
    elseif haskey(parsed,"initial_condition")
        verify_initial_condition(parsed["initial_condition"])
    else
        error("Use directory=<run> or initial_condition=<file>")
    end
    println(json_value(merge((; passed=true),result)))
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    PacketChecks.main(ARGS)
end
