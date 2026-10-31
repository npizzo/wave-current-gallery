# Local tiny CPU integration only. All artifacts remain in a temporary folder.
using Test
ENV["ARCH"]="CPU"
include("run_jet.jl")

function run_test_case(out,stop;pickup="")
    settings=("ARCH"=>"CPU","NX"=>"16","NY"=>"16","NZ"=>"16","LX"=>"2","LY"=>"1","H"=>".5",
        "ALPHA"=>".75","EDDY_RMS"=>".004","PERTURB_ENERGY_RATIO"=>"1","SEED"=>"2",
        "PROBE_MODE"=>"2","PROBE_AMPLITUDE"=>"1e-5","PROBE_SIGN"=>"1",
        "STOP_TIME"=>string(stop),"MAX_DT"=>".002","DIAGNOSTIC_INTERVAL"=>".01",
        "MOMENT_INTERVAL"=>".01","TWO_D_THREE_C_RMS"=>"0","MEAN_INTERVAL"=>".02","KEEP_CHECKPOINT"=>"1","CHECKPOINT_WALL"=>"0",
        "OUT_DIR"=>out,"PICKUP"=>pickup)
    withenv(settings...) do
        main()
    end
end

function final_native(out,component)
    path=joinpath(out,@sprintf("t%012.6f_%s.bin",.04,component))
    shape=component=="w" ? (16,16,17) : (16,16,16)
    data=Array{Float64}(undef,shape)
    read!(path,data)
    return data
end

@testset "Tiny 3D run, restart parity and no-overwrite guard" begin
  mktempdir() do root
    resumed=joinpath(root,"resumed");continuous=joinpath(root,"continuous")
    run_test_case(resumed,.02)
    checkpoint=only(filter(p->endswith(p,".jld2"),readdir(joinpath(resumed,"checkpoints");join=true)))
    @test isfile(joinpath(resumed,"completed.json"))
    run_test_case(resumed,.04;pickup=checkpoint)
    run_test_case(continuous,.04)
    for component in ("u","v","w")
        a,b=final_native(resumed,component),final_native(continuous,component)
        @test all(isfinite,a)
        @test sqrt(sum(abs2,a.-b)/max(sum(abs2,b),eps()))<1e-11
    end
    @test length(filter(p->endswith(p,".jld2"),readdir(joinpath(resumed,"checkpoints"))))==1
    for name in ("probe.csv","jet-budget.csv","moments.csv")
        lines=readlines(joinpath(resumed,name));cols=split(lines[1],',')
        timecol=findfirst(==("time_s"),cols)
        times=[parse(Float64,split(s,',')[timecol]) for s in lines[2:end]]
        @test times≈[0.,.01,.02,.03,.04] atol=1e-12
        @test length(unique(times))==length(times)
    end
    _,mr=JetDiagnostics.read_numeric_csv(joinpath(resumed,"moments.csv"),MOMENT_SCHEMA)
    _,mc=JetDiagnostics.read_numeric_csv(joinpath(continuous,"moments.csv"),MOMENT_SCHEMA)
    for name in MOMENT_FIELDS
        @test last(mr)[name]≈last(mc)[name] rtol=1e-9 atol=1e-16
    end
    @test isfile(joinpath(resumed,"mean_00000.040000.json"))
    before=filehash(joinpath(resumed,"configuration.json"))
    @test_throws ErrorException run_test_case(resumed,.04)
    @test filehash(joinpath(resumed,"configuration.json"))==before
    @test occursin(filehash(joinpath(@__DIR__,"run_jet.jl")),read(joinpath(continuous,"configuration.json"),String))
    println("DRIVER INTEGRATION: 16³, finite 3D energy, +/- branch-compatible probe, .02 checkpoint -> .04 parity")
  end
end
