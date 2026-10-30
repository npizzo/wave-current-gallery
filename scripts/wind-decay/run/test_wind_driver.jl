using Test
ENV["ARCH"]=get(ENV,"DRIVER_TEST_ARCH","CPU")
include("run_jet.jl")

function wind_test_case(out;stop=.04,ny=8,eta=1.,eddy=.004,ustar=.004,alpha=.75,
                        init="",kind="full",jet_mode=0,jet_sign=0,jet_amplitude=0.,
                        approval="",source_id="",approval_sha256="")
    settings=("ARCH"=>get(ENV,"DRIVER_TEST_ARCH","CPU"),"NX"=>"16","NY"=>string(ny),"NZ"=>"16",
        "LX"=>"2","LY"=>"1","H"=>".5","K0"=>"6.1","STOKES_SPEED"=>string(.17^2*sqrt(9.81/6.1)),
        "ALPHA"=>string(alpha),"WAVES"=>"1","UOFFSET"=>"0","EDDY_RMS"=>string(eddy),
        "INJECTION_WAVELENGTH"=>".5","PERTURB_ENERGY_RATIO"=>string(eta),"SEED"=>"2","IC_KIND"=>"embedded",
        "NU"=>"1e-5","WIND_USTAR"=>string(ustar),"MAINTAIN_BACKGROUND"=>"0","ADVECTION"=>"Centered2",
        "PROBE_MODE"=>"0","PROBE_AMPLITUDE"=>"0","PROBE_SIGN"=>"0","TWO_D_THREE_C_RMS"=>"0",
        "JET_MODE"=>string(jet_mode),"JET_SIGN"=>string(jet_sign),"JET_AMPLITUDE"=>string(jet_amplitude),
        "STOP_TIME"=>string(stop),"MAX_DT"=>".001","DIAGNOSTIC_INTERVAL"=>".01","REGIME_INTERVAL"=>".02",
        "MOMENT_INTERVAL"=>".01","MOMENTS_ENABLED"=>"0","SNAPSHOT_TIMES"=>"0,.02,.04",
        "KEEP_CHECKPOINT"=>"0","CHECKPOINT_WALL"=>"0","PICKUP"=>"","START_TIME"=>nothing,
        "INIT_SNAPSHOT"=>init,"INIT_KIND"=>kind,"OUT_DIR"=>out,
        "INIT_APPROVAL"=>approval,"INIT_SOURCE_ID"=>source_id,"INIT_APPROVAL_SHA256"=>approval_sha256)
    withenv(settings...) do;main();end
end
function test_manifest(out,t)
    joinpath(out,@sprintf("t%012.6f.json",t))
end
function test_array(out,t,name)
    manifest=test_manifest(out,t);m=TOML.parsefile(replace(manifest,".json"=>".toml"))
    entry=m["arrays"][name];a=Array{Float64}(undef,Tuple(entry["shape"]));read!(joinpath(out,entry["file"]),a);a
end
function test_wind_rows(out)
    last(JetDiagnostics.read_numeric_csv(joinpath(out,"wind-budget.csv"),WIND_BUDGET_SCHEMA))
end

# set!(model; velocities...) projects the loaded field to native zero divergence.
# Re-projecting an already solenoidal field can change it at floating-point
# roundoff. Raw snapshot loading is checked bitwise below; this separate check
# permits only 32 machine epsilons of the largest component value.
function projection_roundoff(label,a,b)
    delta=maximum(abs,a.-b);scale=maximum(abs,b)
    limit=32eps(Float64)*scale
    @printf("PROJECTION_ROUNDOFF %s max_abs=%.17g limit=%.17g eps_units=%.8g\n",label,delta,limit,scale>0 ? delta/(eps(Float64)*scale) : 0.)
    return delta<=limit
end

@testset "Wind momentum, exact 2D, branching and paired jet perturbations" begin
  mktempdir() do root
    laminar=joinpath(root,"laminar")
    wind_test_case(laminar;eta=0.,eddy=0.,alpha=1.)
    rows=test_wind_rows(laminar)
    @test maximum(abs(r["UL_depth_mean_change"]) for r in rows)<1e-13
    @test maximum(abs(r["native_depth_mean_tendency"]) for r in rows)<1e-13
    @test maximum(r["C_native_rms_full"] for r in rows)<1e-13
    @test maximum(r["roll_energy_xmean"] for r in rows)<1e-25
    profile=readlines(joinpath(laminar,"jet-profiles.csv"))
    columns=split(profile[1],',');ci=findfirst(==("closure"),columns);fi=findfirst(==("forcing"),columns)
    initial=[parse.(Float64,split(line,',')) for line in profile[2:17]]
    @test mean(row[ci] for row in initial)≈.004^2/.5 atol=1e-15
    @test mean(row[fi] for row in initial)≈-.004^2/.5 atol=1e-15
    @test maximum(test_array(laminar,.04,"u"))>0
    null=joinpath(root,"two_d")
    wind_test_case(null;eta=0.)
    @test maximum(r["C_native_rms_full"] for r in test_wind_rows(null))<1e-13
    @test maximum(abs(r["UL_depth_mean_change"]) for r in test_wind_rows(null))<1e-12

    parent=joinpath(root,"parent");wind_test_case(parent)
    initial=test_manifest(parent,.02)
    @test isfile(initial)&&isfile(replace(initial,".json"=>".toml"))
    meta=TOML.parsefile(replace(initial,".json"=>".toml"))
    p=(;nx=16,ny=8,nz=16,lx=2.,ly=1.,h=.5,k=6.1,Sref=.17^2*sqrt(9.81/6.1),alpha=.75,waves=1,nu=1e-5,ustar=.004,advection_name="Centered2")
    raw=load_wind_snapshot(initial,"full",p,meta["source_hashes"]).arrays
    for (name,a) in zip(("u","v","w"),raw)
        @test a==test_array(parent,.02,name)
    end
    restart=joinpath(root,"full_restart");wind_test_case(restart;init=initial)
    for name in ("u","v","w")
        @test projection_roundoff("reload_"*name,test_array(restart,.02,name),test_array(parent,.02,name))
        a,b=test_array(restart,.04,name),test_array(parent,.04,name)
        @test sqrt(sum(abs2,a.-b)/sum(abs2,b))<2e-9
    end
    reduced=joinpath(root,"mean_restart");wind_test_case(reduced;ny=1,eta=0.,init=initial,kind="mean")
    arrays=map(name->test_array(reduced,.02,name),("u","v","w"))
    for (name,a) in zip(("u","v","w"),arrays)
        @test a≈mean(test_array(parent,.02,name);dims=2) atol=1e-16 rtol=1e-14
    end
    @test maximum(abs,native_divergence(arrays...,2/16,1.,.5/16))<1e-12
    @test maximum(r["C_native_rms_full"] for r in test_wind_rows(reduced))<1e-13
    @test first(test_wind_rows(reduced))["time_s"]≈.02

    plus=joinpath(root,"plus");minus=joinpath(root,"minus")
    wind_test_case(plus;init=initial,jet_mode=2,jet_sign=1,jet_amplitude=2e-4)
    wind_test_case(minus;init=initial,jet_mode=2,jet_sign=-1,jet_amplitude=2e-4)
    up,um,ub=(test_array(path,.02,"u") for path in (plus,minus,parent))
    shape=cos.(2pi.*((collect(1:16).-0.5)./16));shape.-=mean(shape)
    @test (up.+um)./2≈ub atol=1e-17
    @test vec(mean((up.-um)./4e-4;dims=(1,2)))≈shape atol=1e-13
    @test abs(mean(up.-ub))<1e-17
    @test maximum(abs,native_divergence(up,test_array(plus,.02,"v"),test_array(plus,.02,"w"),2/16,1/8,.5/16))<1e-12
    for name in ("v","w")
        @test projection_roundoff("plus_"*name,test_array(plus,.02,name),test_array(parent,.02,name))
        @test projection_roundoff("minus_"*name,test_array(minus,.02,name),test_array(parent,.02,name))
    end
    @test all(isfinite,test_array(plus,.04,"u"))
    @test_throws ErrorException wind_test_case(parent)
    badsource=copy(meta["source_hashes"]);badsource["run_jet.jl"]="bad"
    @test_throws ErrorException load_wind_snapshot(initial,"full",p,badsource)
    @test_throws ErrorException load_wind_snapshot(initial,"mean",p,meta["source_hashes"])
    broken=joinpath(root,"broken");cp(parent,broken)
    entry=meta["arrays"]["u"];target=joinpath(broken,entry["file"])
    open(target,"r+") do io;write(io,UInt8(0));end
    # Guarantee a mismatch even in the unlikely event the original byte is 0.
    open(target,"a") do io;write(io,UInt8(1));end
    @test_throws ErrorException load_wind_snapshot(test_manifest(broken,.02),"full",p,meta["source_hashes"])
  end
end

println("WIND DRIVER TESTS PASSED on ",get(ENV,"DRIVER_TEST_ARCH","CPU"))
