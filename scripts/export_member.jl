# Read-only extraction from a FINISHED member. The Python wrapper validates its
# completion receipt, pinned source and input hashes before invoking this file.
using JLD2, Statistics, Printf
using Oceananigans

length(ARGS) == 3 || error("Usage: export_member.jl CAMPAIGN_ROOT MEMBER_DIR OUTPUT_DIR")
campaign, directory, output = abspath.(ARGS)
include(joinpath(campaign, "upstream/experiments/anti_stokes/moving_packet.jl"))
isdir(output) || mkpath(output)
readall(path) = jldopen(f -> Dict(k => f[k] for k in keys(f)), path, "r"; iotype=IOStream)
meta = readall(joinpath(directory, "metadata.jld2"))
summary = readall(joinpath(directory, "run_summary.jld2"))
isfinite(summary["final_time"]) || error("Nonfinite final time")
summary["final_time"] >= meta["stop_time"] - meta["Δt"] || error("Incomplete run")
meta["x_topology"] == "bounded" || error("Only bounded packet campaign supported")
meta["member"] in ("packet_turbulence", "turbulence_control", "packet_null", "quiescent_control") || error("Unexpected member")

function jsonvalue(x)
    x isa Bool && return x ? "true" : "false"
    x isa Number && return isfinite(x) ? string(x) : "null"
    return "\"" * replace(string(x), "\\"=>"\\\\", "\""=>"\\\"", "\n"=>"\\n") * "\""
end
public = Dict("member"=>meta["member"], "seed"=>meta["seed"], "Nx"=>meta["Nx"],
              "Ny"=>meta["Ny"], "Nz"=>meta["Nz"], "Lx_m"=>meta["Lx"], "Ly_m"=>meta["Ly"],
              "Lz_m"=>meta["Lz"], "dt_s"=>meta["Δt"], "final_time_s"=>summary["final_time"],
              "iterations"=>summary["iterations"], "peak_time_s"=>meta["t_peak"],
              "packet_width_s"=>meta["τ₀"], "has_packet"=>meta["has_packet"],
              "velocity_convention"=>"Eulerian y-mean; prescribed Stokes drift subtracted at native x faces")
open(joinpath(output,"metadata.json"),"w") do io
    println(io,"{",join([jsonvalue(k)*":"*jsonvalue(v) for (k,v) in sort(collect(public);by=first)],","),"}")
end

open(joinpath(output,"mean-profiles.csv"),"w") do profiles
open(joinpath(output,"mean-fields.csv"),"w") do fields
    println(profiles,"frame,time_s,z_m,mean_u_E_m_s")
    println(fields,"frame,time_s,x_m,z_m,u_E_m_s")
    jldopen(joinpath(directory,"y_averages.jld2"),"r";iotype=IOStream) do f
        its=sort(parse.(Int,collect(keys(f["timeseries/t"]))))
        ts=[Float64(f["timeseries/t/$it"]) for it in its]
        all(isfinite,ts) && issorted(ts) || error("Invalid output times")
        targets=[0.,Float64(meta["t_peak"]),Float64(meta["t_peak"]+3meta["τ₀"]),last(ts)]
        chosen=unique([argmin(abs.(ts.-target)) for target in targets])
        gridkey = haskey(f,"timeseries/U/serialized/grid_index") ?
                  "serialized/grid_$(f["timeseries/U/serialized/grid_index"])" : "serialized/grid"
        grid=f[gridkey]
        xf=Float64.(Array(xnodes(grid,Face())))
        z=Float64.(Array(znodes(grid,Center())))
        xc=(xf[1:end-1].+xf[2:end])./2
        length(xf)==meta["Nx"]+1 || error("Bounded face count mismatch")
        # Uniform-x arithmetic average on 1 <= x <= Lx-1 m, matching the
        # campaign's interior region; preserve every vertical grid level.
        ix=findall(x->1<=x<=meta["Lx"]-1,xc)
        isempty(ix) && error("No interior region")
        stride=max(1,cld(length(xc),384))
        display_ix=collect(1:stride:length(xc))
        last(display_ix)==length(xc) || push!(display_ix,length(xc))
        for (frame,n) in enumerate(chosen)
            t=ts[n]; it=its[n]
            U=Float64.(dropdims(f["timeseries/U/$it"];dims=2))
            size(U)==(length(xf),length(z)) && all(isfinite,U) || error("Invalid mean field")
            if meta["has_packet"]
                p=meta["packet"]
                for k in eachindex(z),i in eachindex(xf)
                    U[i,k]-=uˢ(xf[i],0,z[k],t,p)
                end
            end
            Uc=(U[1:end-1,:].+U[2:end,:])./2
            for k in eachindex(z)
                println(profiles,join((frame,t,z[k],mean(Uc[ix,k])),','))
            end
            for k in eachindex(z),i in display_ix
                println(fields,join((frame,t,xc[i],z[k],Uc[i,k]),','))
            end
        end
    end
end
end

# Preserve the native volume-weighted RMS (including the mean), not a turbulent
# RMS or Eulerian variance. In packet runs these are Lagrangian velocities.
open(joinpath(output,"velocity-statistics.csv"),"w") do io
    names=("u_mean","v_mean","u_rms","v_rms","w_rms","w_max")
    println(io,"time_s,u_L_mean_m_s,v_L_mean_m_s,u_L_rms_m_s,v_L_rms_m_s,w_L_rms_m_s,w_L_max_m_s")
    jldopen(joinpath(directory,"statistics.jld2"),"r";iotype=IOStream) do f
        its=sort(parse.(Int,collect(keys(f["timeseries/t"]))))
        for it in its
            values=[Float64(f["timeseries/t/$it"]);[Float64(f["timeseries/$name/$it"]) for name in names]]
            all(isfinite,values) || error("Nonfinite native statistics")
            println(io,join(values,','))
        end
    end
end
println("Finished read-only member extraction.")
