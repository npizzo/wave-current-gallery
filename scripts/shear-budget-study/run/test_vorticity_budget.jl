using Test, Statistics
# The numerical helper is independent of Oceananigans; these CSV primitives
# exercise its actual restart/integration state without importing a solver.
module JetDiagnostics
csvline(io,values)=println(io,join(values,','))
function read_numeric_csv(path,expected)
    lines=readlines(path);columns=split(first(lines),',')
    columns==expected || error("Budget schema mismatch")
    rows=[Dict(key=>parse(Float64,value) for (key,value) in zip(columns,split(line,','))) for line in lines[2:end]]
    lines,rows
end
end
include("vorticity_budget.jl")

@testset "Signed native-vorticity work and quadrature" begin
    nx,nz=12,16
    q0=[sin(2pi*(i-.5)/nx)*cos(pi*(k-.5)/nz)+.3k/nz for i in 1:nx,k in 1:nz]
    templates=Dict("Qdot_advection_mean"=>.12q0,"Qdot_advection_full"=>.2q0,
        "Qdot_wave"=>-.03q0,"Qdot_viscous"=>-.25q0,"Qdot_forcing"=>.04q0)
    total=templates["Qdot_advection_full"]+templates["Qdot_wave"]+templates["Qdot_viscous"]+templates["Qdot_forcing"]
    ids=(collect(1:nz),collect(3:nz-2),collect(nz-5:nz-2))
    mktempdir() do directory
        path=joinpath(directory,"budget.csv");state=init_vorticity_budget(path)
        for t in (0.,.2,.4)
            maps=merge(templates,Dict("Q"=>q0+t*total))
            row=accumulate_vorticity_budget!(state,vorticity_budget_values(maps,ids,t,round(Int,t*100)))
            @test all(isfinite,row[key] for key in VORTICITY_BUDGET_SCHEMA)
            for region in VORTICITY_REGIONS,part in VORTICITY_PARTS
                @test abs(row["$(region)_balance_residual_$(part)"])<1e-14
                @test abs(row["$(region)_split_identity_$(part)"])<1e-14
                @test row["$(region)_P_$(part)_D"]<0
                @test row["$(region)_P_$(part)_C"]>0
            end
            for region in VORTICITY_REGIONS
                @test abs(row["$(region)_enstrophy_decomposition_error"])<1e-14
            end
        end
        close_vorticity_budget!(state)
        resumed=init_vorticity_budget(path;resume=true,time=.2)
        @test resumed.previous["time_s"]==.2
        maps=merge(templates,Dict("Q"=>q0+.4total))
        row=accumulate_vorticity_budget!(resumed,vorticity_budget_values(maps,ids,.4,40))
        @test abs(row["full_balance_residual_Q"])<1e-14
        @test row["interval_dt"]≈.2
        close_vorticity_budget!(resumed)
    end
    cancel=Dict("Q"=>q0,"Qdot_advection_mean"=>zeros(nx,nz),"Qdot_advection_full"=>zeros(nx,nz),
        "Qdot_wave"=>zeros(nx,nz),"Qdot_viscous"=>-.1q0,"Qdot_forcing"=>.1q0)
    row=vorticity_budget_values(cancel,ids,0.,0)
    for region in VORTICITY_REGIONS,part in VORTICITY_PARTS
        @test row["$(region)_P_$(part)_total"]==0
        @test row["$(region)_P_$(part)_C"]==0
        @test row["$(region)_P_$(part)_F"]≈-row["$(region)_P_$(part)_D"]
    end
end
