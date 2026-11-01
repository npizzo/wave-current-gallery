# Signed native transverse-mean vorticity budget at scalar cadence. No new
# approximation to the dynamics is introduced: all tendency maps come from
# the fresh model operators. A0+C+W+D+F is the retained complete native RHS.
const VORTICITY_TERMS=("A0","C","W","D","F","total")
const VORTICITY_REGIONS=("full","interior","stokes")
const VORTICITY_PARTS=("Q","profile","eddy")
const VORTICITY_BUDGET_SCHEMA=vcat(["time_s","iteration","interval_dt"],
    ["$(region)_Z_$(part)" for region in VORTICITY_REGIONS for part in VORTICITY_PARTS],
    ["$(region)_P_$(part)_$(term)" for region in VORTICITY_REGIONS for part in VORTICITY_PARTS for term in VORTICITY_TERMS],
    ["$(region)_source_rms_$(term)" for region in VORTICITY_REGIONS for term in VORTICITY_TERMS],
    ["$(region)_I_$(part)_$(term)" for region in VORTICITY_REGIONS for part in VORTICITY_PARTS for term in VORTICITY_TERMS],
    ["$(region)_balance_residual_$(part)" for region in VORTICITY_REGIONS for part in VORTICITY_PARTS],
    ["$(region)_split_identity_$(part)" for region in VORTICITY_REGIONS for part in VORTICITY_PARTS],
    ["$(region)_enstrophy_decomposition_error" for region in VORTICITY_REGIONS])

mutable struct VorticityBudgetState
    io::Any
    previous::Any
    initial::Any
    cumulative::Dict{String,Float64}
end

function init_vorticity_budget(path;resume=false,time=0.)
    previous=nothing;initial=nothing;cumulative=Dict{String,Float64}()
    if resume
        lines,rows=JetDiagnostics.read_numeric_csv(path,VORTICITY_BUDGET_SCHEMA)
        ids=findall(r->r["time_s"]<=time+1e-10,rows)
        isempty(ids)&&error("No vorticity budget preceding checkpoint")
        cp(path,path*".pre-restart-"*string(time_ns()))
        write(path,join(vcat(lines[1],lines[ids.+1]),"\n")*"\n")
        initial=first(rows);previous=rows[last(ids)]
        for region in VORTICITY_REGIONS,part in VORTICITY_PARTS,term in VORTICITY_TERMS
            key="$(region)_I_$(part)_$(term)";cumulative[key]=previous[key]
        end
    else
        isfile(path)&&error("Refusing to overwrite signed vorticity budget")
        write(path,join(VORTICITY_BUDGET_SCHEMA,',')*"\n")
    end
    VorticityBudgetState(open(path,"a"),previous,initial,cumulative)
end

function vorticity_budget_values(maps,selections,t,iteration_value)
    Q=maps["Q"];Qprofile=mean(Q;dims=1);qeddy=Q.-Qprofile
    terms=Dict("A0"=>maps["Qdot_advection_mean"],
        "C"=>maps["Qdot_advection_full"].-maps["Qdot_advection_mean"],
        "W"=>maps["Qdot_wave"],"D"=>maps["Qdot_viscous"],"F"=>maps["Qdot_forcing"])
    terms["total"]=terms["A0"].+terms["C"].+terms["W"].+terms["D"].+terms["F"]
    fields=Dict("Q"=>Q,"profile"=>Qprofile,"eddy"=>qeddy)
    row=Dict{String,Float64}("time_s"=>t,"iteration"=>iteration_value)
    for (region,ids) in zip(VORTICITY_REGIONS,selections)
        isempty(ids)&&error("Signed vorticity budget region is empty")
        for part in VORTICITY_PARTS
            a=fields[part][:,ids];row["$(region)_Z_$(part)"]=.5mean(abs2,a)
            for term in VORTICITY_TERMS
                row["$(region)_P_$(part)_$(term)"]=mean(a.*terms[term][:,ids])
            end
            row["$(region)_split_identity_$(part)"]=row["$(region)_P_$(part)_total"]-
                sum(row["$(region)_P_$(part)_$(term)"] for term in VORTICITY_TERMS[1:5])
        end
        for term in VORTICITY_TERMS
            row["$(region)_source_rms_$(term)"]=sqrt(mean(abs2,terms[term][:,ids]))
        end
        row["$(region)_enstrophy_decomposition_error"]=row["$(region)_Z_Q"]-
            row["$(region)_Z_profile"]-row["$(region)_Z_eddy"]
    end
    row
end

function accumulate_vorticity_budget!(state,row)
    state.previous!==nothing && row["time_s"]<=state.previous["time_s"]+1e-10 && return nothing
    dt=state.previous===nothing ? 0. : row["time_s"]-state.previous["time_s"]
    row["interval_dt"]=dt
    state.initial===nothing && (state.initial=copy(row))
    for region in VORTICITY_REGIONS,part in VORTICITY_PARTS
        for term in VORTICITY_TERMS
            key="$(region)_I_$(part)_$(term)";power="$(region)_P_$(part)_$(term)"
            state.cumulative[key]=get(state.cumulative,key,0.)+(dt==0 ? 0. : .5dt*(state.previous[power]+row[power]))
            row[key]=state.cumulative[key]
        end
        row["$(region)_balance_residual_$(part)"]=row["$(region)_Z_$(part)"]-
            state.initial["$(region)_Z_$(part)"]-row["$(region)_I_$(part)_total"]
    end
    all(isfinite,values(row))||error("Nonfinite signed vorticity diagnostic")
    JetDiagnostics.csvline(state.io,[row[key] for key in VORTICITY_BUDGET_SCHEMA]);flush(state.io)
    state.previous=copy(row);row
end

close_vorticity_budget!(state)=close(state.io)
