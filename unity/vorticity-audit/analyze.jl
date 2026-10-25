using JLD2, Printf, SHA, Dates
include("audit_math.jl")
include("packet_fields.jl")
using .PacketVorticityAuditMath

row(io,args...)=println(io,join(args,','))
readjld(path)=jldopen(f->Dict(k=>f[k] for k in keys(f)),path,"r";iotype=IOStream)
filehash(path)=open(io->bytes2hex(sha256(io)),path)

function write_regions(io, profiles, a, xf, t, p, seed, member, step, tau)
    x=(xf[1:end-1].+xf[2:end])./2; z=a.z; dz=a.dz
    nx,nz=length(x),length(z); dx=xf[2]-xf[1]
    dy_volume=0.8 # checked against metadata before entry
    ix=findall(xx->1<=xx<=11,x)
    ages=(t .- (x .- Float64(p.x₀))./Float64(p.cᵍ))./tau
    agebins=(("all",-Inf,Inf),("before",-Inf,-3.),("leading",-3.,-1.),
             ("core",-1.,1.),("trailing",1.,3.),("after",3.,Inf))
    regions=(("full",collect(1:nz)),("stokes_layer",findall(zz->zz>=-1/(2Float64(p.k)),z)),
             ("piv_window",findall(zz->-1.2<=Float64(p.k)*zz<=-.1,z)))
    for (agebin,lo,hi) in agebins
        xs=filter(i->lo<=ages[i]<hi,ix)
        isempty(xs) && continue
        for term in sort(collect(keys(a.means)))
            # Both derivative stencils use the same 4-cell vertical exclusion for
            # q terms, so changing the stencil does not change sampled volume.
            valid=collect(5:nz-4)
            for (region,rs) in regions
                js=intersect(rs,valid); isempty(js) && continue
                weight=length(xs)*sum(dz[js])
                avg(b)=sum(b[i,j]*dz[j] for j in js for i in xs)/weight
                maximum_abs=maximum(a.maxabs[term][i,j] for j in js for i in xs)
                row(io,seed,member,t,step,region,agebin,lo,hi,dx*dy_volume*weight,
                    length(xs),length(js),minimum(z[js]),maximum(z[js]),term,
                    avg(a.means[term]),avg(a.mean_squares[term]),maximum_abs)
            end
            # All terms use the same vertical interior; full-grid native
            # incompressibility remains available separately in qc.csv.
            for j in valid
                row(profiles,seed,member,t,step,agebin,lo,hi,z[j],dz[j],length(xs),term,
                    sum(a.means[term][xs,j])/length(xs),
                    sum(a.mean_squares[term][xs,j])/length(xs),maximum(a.maxabs[term][xs,j]))
            end
        end
    end
end

function main(; expected_resolution=(1536,128,128), expected_snapshots=7)
    length(ARGS)==3 || error("Usage: analyze.jl CAMPAIGN OUTPUT SEED")
    campaign,out=abspath.(ARGS[1:2]); seed=parse(Int,ARGS[3])
    seed in (1,2,3,4,6,7,8) || error("Unexpected seed")
    isdir(out) && !isempty(readdir(out)) && error("Fresh output directory required")
    mkpath(out)
    sources=Dict{String,Any}[]
    open(joinpath(out,"regions.csv"),"w") do io
    open(joinpath(out,"profiles.csv"),"w") do profiles
    open(joinpath(out,"qc.csv"),"w") do qc
        row(io,"seed","member","time_s","stencil_step","region","age_bin","age_low","age_high",
            "volume_m3","x_columns","z_levels","z_min_m","z_max_m","term","mean","mean_square","max_abs")
        row(profiles,"seed","member","time_s","stencil_step","age_bin","age_low","age_high","z_m","dz_m",
            "x_columns","term","mean","mean_square","max_abs")
        row(qc,"seed","member","time_s","stencil_step","native_divergence_max","native_divergence_scaled","R3D_identity_error")
        baseline_times=nothing
        for member in ("turbulence_control","packet_turbulence")
            dir=joinpath(campaign,"data/campaign/case_1D/M2",member,@sprintf("seed_%04d_dt0.020_boundedx",seed))
            meta_path=joinpath(dir,"metadata.jld2"); snapshot_path=joinpath(dir,"snapshots.jld2")
            isfile(joinpath(dir,"run_summary.jld2")) || error("Missing completed run: $dir")
            meta=readjld(meta_path)
            (meta["Nx"],meta["Ny"],meta["Nz"])==expected_resolution || error("Unexpected resolution")
            all(isapprox(a,b;rtol=1e-6) for (a,b) in zip((meta["Lx"],meta["Ly"],meta["Lz"]),(12,0.8,0.4))) || error("Unexpected domain")
            meta["x_topology"]=="bounded" || error("Expected bounded x")
            meta["FT"]=="Float32" || error("Expected the campaign's Float32 grid")
            meta["seed"]==seed && meta["member"]==member || error("Metadata mismatch")
            p=meta["packet"]; tau=Float64(meta["τ₀"])
            zf=Float64.(Float32.(audit_zfaces(meta["Nz"],meta["Lz"])))
            xf=collect(range(0.,12.;length=meta["Nx"]+1))
            dx,dy=12/meta["Nx"],0.8/meta["Ny"]
            before_stat=stat(snapshot_path)
            push!(sources,Dict("path"=>relpath(snapshot_path,campaign),"bytes"=>before_stat.size,
                              "mtime"=>before_stat.mtime,"metadata_sha256"=>filehash(meta_path)))
            jldopen(snapshot_path,"r";iotype=IOStream) do f
                its=sort(parse.(Int,collect(keys(f["timeseries/t"]))))
                times=[Float64(f["timeseries/t/$it"]) for it in its]
                length(times)==expected_snapshots || error("Missing scheduled 3-D snapshots")
                baseline_times===nothing ? (baseline_times=times) :
                    (maximum(abs.(times.-baseline_times))<1e-6 || error("Unmatched saved times"))
                for (it,t) in zip(its,times)
                    native=Tuple(f["timeseries/$n/$it"] for n in ("u","v","w"))
                    fields=packet_fields(xf,zf,t,p;enabled=member=="packet_turbulence")
                    for step in (1,2)
                        a=snapshot_reductions(native...,dx,dy,zf;fields...,stencil_step=step)
                        a.qc.scaled_native_divergence_max<=2e-4 || error("Native incompressibility check failed")
                        write_regions(io,profiles,a,xf,t,p,seed,member,step,tau)
                        row(qc,seed,member,t,step,a.qc.max_native_divergence,a.qc.scaled_native_divergence_max,a.qc.R3D_identity_error)
                        println("AUDIT seed=$seed member=$member t=$t stencil=$step")
                        flush(io); flush(profiles); flush(qc); flush(stdout)
                        a=nothing; GC.gc(false)
                    end
                    native=nothing; GC.gc(false)
                end
            end
            after_stat=stat(snapshot_path)
            (before_stat.size,before_stat.mtime)==(after_stat.size,after_stat.mtime) || error("Snapshot changed while reading")
        end
    end
    end
    end
    # JLD2 receipt is internal; Python creates the portable completion manifest.
    jldopen(joinpath(out,"audit-receipt.jld2"),"w";iotype=IOStream) do f
        f["complete"]=true; f["seed"]=seed; f["sources"]=sources
        f["created_utc"]=string(now(UTC))
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
