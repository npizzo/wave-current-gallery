using Test
ENV["ARCH"]=get(ENV,"DRIVER_TEST_ARCH","CPU")
include("run_jet.jl")

function onset_test_case(out;stop=.04,ny=8,eta=1.,eddy=.004,ustar=0.,alpha=.75,
                        init="",kind="full",jet_mode=0,jet_sign=0,jet_amplitude=0.,
                        approval="",source_id="",approval_sha256="",metadata="",maintain=true)
    settings=("ARCH"=>get(ENV,"DRIVER_TEST_ARCH","CPU"),"NX"=>"16","NY"=>string(ny),"NZ"=>"16",
        "LX"=>"2","LY"=>"1","H"=>".5","K0"=>"6.1","STOKES_SPEED"=>string(.17^2*sqrt(9.81/6.1)),
        "ALPHA"=>string(alpha),"WAVES"=>"1","UOFFSET"=>"0","EDDY_RMS"=>string(eddy),
        "INJECTION_WAVELENGTH"=>".5","PERTURB_ENERGY_RATIO"=>string(eta),"SEED"=>"2","IC_KIND"=>"embedded",
        "NU"=>string(.004*.25/300),"WIND_USTAR"=>string(ustar),"MAINTAIN_BACKGROUND"=>(maintain ? "1" : "0"),"ADVECTION"=>"Centered2",
        "PROBE_MODE"=>"0","PROBE_AMPLITUDE"=>"0","PROBE_SIGN"=>"0","TWO_D_THREE_C_RMS"=>"0",
        "JET_MODE"=>string(jet_mode),"JET_SIGN"=>string(jet_sign),"JET_AMPLITUDE"=>string(jet_amplitude),
        "STOP_TIME"=>string(stop),"MAX_DT"=>".001","DIAGNOSTIC_INTERVAL"=>".01","REGIME_INTERVAL"=>".02",
        "MOMENT_INTERVAL"=>".01","MOMENTS_ENABLED"=>"0","SNAPSHOT_TIMES"=>"0,.02,.04",
        "KEEP_CHECKPOINT"=>"0","CHECKPOINT_WALL"=>"0","PICKUP"=>"","START_TIME"=>nothing,
        "INIT_SNAPSHOT"=>init,"INIT_KIND"=>kind,"OUT_DIR"=>out,
        "INIT_METADATA"=>metadata,"INIT_APPROVAL"=>approval,"INIT_SOURCE_ID"=>source_id,"INIT_APPROVAL_SHA256"=>approval_sha256)
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

function onset_test_profile(out,t)
    lines=readlines(joinpath(out,"jet-profiles.csv"));columns=split(lines[1],',')
    rows=[parse.(Float64,split(line,',')) for line in lines[2:end] if parse(Float64,first(split(line,',')))==t]
    Dict(name=>[row[j] for row in rows] for (j,name) in enumerate(columns))
end

function onset_test_map(out,t,name)
    manifest=read(joinpath(out,@sprintf("mean_%012.6f.json",t)),String)
    # Shapes are fixed by this test's 16x16 x-z grid; no untrusted JSON parser.
    occursin("\""*name*"\"",manifest) || error("Missing test map")
    a=Matrix{Float64}(undef,16,16)
    read!(joinpath(out,@sprintf("mean_%012.6f_%s.bin",t,name)),a);a
end

function onset_driver_tests(root)
    nu=.004*.25/300;Sref=.17^2*sqrt(9.81/6.1);h=.5;dz=h/16
    gtop=.25*2*6.1*Sref;gbottom=gtop*exp(-2*6.1*h)
    laminar=joinpath(root,"laminar-control");onset_test_case(laminar;eddy=0.,eta=0.)
    @test projection_roundoff("maintained_background",test_array(laminar,.04,"u"),test_array(laminar,0.,"u"))
    for row in test_wind_rows(laminar)
        @test abs(row["native_depth_mean_tendency"])<1e-13
        @test row["wind_ustar"]==0.
        @test row["kinematic_surface_stress"]==0.
        @test row["total_surface_stress"]≈nu*gtop atol=1e-16
        @test row["total_bottom_stress"]≈nu*gbottom atol=1e-16
    end
    lamwind=joinpath(root,"laminar-wind");onset_test_case(lamwind;eddy=0.,eta=0.,ustar=.002)
    initial_profile=onset_test_profile(lamwind,0.);expected=fill(-.002^2/h,16);expected[end]+=.002^2/dz
    @test initial_profile["native_total"]≈expected atol=1e-13
    @test mean(initial_profile["closure"])≈nu*(gtop-gbottom)/h+.002^2/h atol=1e-14
    @test mean(initial_profile["forcing"])≈-nu*(gtop-gbottom)/h-.002^2/h atol=1e-14
    @test maximum(abs(r["UL_depth_mean_change"]) for r in test_wind_rows(lamwind))<1e-13
    @test maximum(r["C_native_rms_full"] for r in test_wind_rows(lamwind))<1e-13
    @test_throws ErrorException onset_test_case(joinpath(root,"maintenance-removed");maintain=false)

    parent=joinpath(root,"tiny-parent");onset_test_case(parent)
    original=test_manifest(parent,.02);original_meta=TOML.parsefile(replace(original,r"\.json$"=>".toml"))
    p=(;nx=16,ny=8,nz=16,lx=2.,ly=1.,h,k=6.1,Sref,alpha=.75,uoffset=0.,waves=1,nu,ustar=0.,maintain=true,advection_name="Centered2")
    new_sources=original_meta["source_hashes"]

    # A copied, synthetic historical label exercises guards only. Independent
    # operations validation uses the actual approved historical native state.
    fixture=joinpath(root,"synthetic-native-parent");mkdir(fixture)
    initial=joinpath(fixture,basename(original));cp(original,initial)
    for entry in values(original_meta["arrays"])
        cp(joinpath(parent,entry["file"]),joinpath(fixture,entry["file"]))
    end
    config_sha=filehash(joinpath(parent,"configuration.json"))
    historical=deepcopy(original_meta);historical["source_hashes"]=copy(ONSET_PARENT_SOURCE_HASHES)
    historical["translation"]="legacy-native-json-plus-original-configuration-v1"
    historical["original_configuration_sha256"]=config_sha
    metadata=joinpath(root,"translated-parent.toml")
    open(metadata,"w") do io;TOML.print(io,historical;sorted=true);end
    @test !isfile(replace(initial,r"\.json$"=>".toml"))
    source_id="synthetic-jet-parent"
    record=Dict{String,Any}("snapshot_json_sha256"=>filehash(initial),"snapshot_metadata_sha256"=>filehash(metadata),
        "snapshot_time"=>.02,"parent_configuration_sha256"=>config_sha,"expected_parent_ustar"=>0.,
        "expected_maintain_background"=>true,"allowed_branch_ustar"=>[0.,.002],
        "source_hashes"=>historical["source_hashes"],"parameters"=>historical["parameters"],"arrays"=>historical["arrays"])
    approval=Dict("format"=>"wind-onset-parent-approval-v1","campaign"=>"unity-wind-onset-tests",
        "approved_parent_source_version"=>ONSET_PARENT_SOURCE_VERSION,"sources"=>Dict(source_id=>record))
    approval_path=joinpath(root,"parent-approval.toml");open(io->TOML.print(io,approval;sorted=true),approval_path,"w")
    approval_sha256=filehash(approval_path);options=(;metadata_path=metadata,approval_path,source_id,approval_sha256)
    @test_throws ErrorException load_wind_snapshot(initial,"full",p,new_sources;metadata_path=metadata)
    @test_throws ErrorException load_wind_snapshot(initial,"full",p,new_sources;metadata_path=metadata,approval_path,source_id)
    @test_throws ErrorException load_wind_snapshot(initial,"mean",merge(p,(;ny=1)),new_sources;options...)
    for changed in ((;maintain=false),(;nu=1e-5),(;ustar=.004),(;alpha=.5),(;waves=0),(;uoffset=.001))
        @test_throws ErrorException load_wind_snapshot(initial,"full",merge(p,changed),new_sources;options...)
    end
    for (label,mutate!) in (("source",a->a["sources"][source_id]["source_hashes"]["run_jet.jl"]="0"^64),
            ("version",a->a["approved_parent_source_version"]="unapproved"),
            ("configuration",a->a["sources"][source_id]["parent_configuration_sha256"]="0"^64),
            ("maintenance",a->a["sources"][source_id]["expected_maintain_background"]=false),
            ("wind",a->a["sources"][source_id]["allowed_branch_ustar"]=[0.,.004]))
        bad=deepcopy(approval);mutate!(bad);path=joinpath(root,"bad-"*label*".toml")
        open(io->TOML.print(io,bad;sorted=true),path,"w")
        @test_throws ErrorException load_wind_snapshot(initial,"full",p,new_sources;
            metadata_path=metadata,approval_path=path,source_id,approval_sha256=filehash(path))
    end
    before=filehash(initial);before_arrays=[filehash(joinpath(fixture,historical["arrays"][name]["file"])) for name in ("u","v","w")]
    for (name,ustar) in (("control",0.),("wind-on",.002))
        raw=load_wind_snapshot(initial,"full",merge(p,(;ustar)),new_sources;options...)
        @test raw.approved.parent_ustar==0. && raw.approved.branch_ustar==ustar
        for (field,a) in zip(("u","v","w"),raw.arrays);@test a==test_array(parent,.02,field);end
        out=joinpath(root,name)
        onset_test_case(out;ustar,init=initial,metadata,approval=approval_path,source_id,approval_sha256)
        arrays=map(field->test_array(out,.02,field),("u","v","w"))
        for (field,a) in zip(("u","v","w"),arrays);@test projection_roundoff(name*"_"*field,a,test_array(parent,.02,field));end
        @test maximum(abs,native_divergence(arrays...,2/16,1/8,h/16))<1e-12
        rows=test_wind_rows(out)
        @test maximum(abs(r["UL_depth_mean_change"]) for r in rows)<1e-12
        @test maximum(abs(r["native_depth_mean_tendency"]) for r in rows)<1e-12
        @test all(r["wind_ustar"]≈ustar for r in rows)
        @test all(r["total_surface_stress"]≈nu*gtop+ustar^2 for r in rows)
        @test all(r["wind_return_acceleration"]≈-ustar^2/h for r in rows)
    end
    @test filehash(initial)==before
    @test [filehash(joinpath(fixture,historical["arrays"][name]["file"])) for name in ("u","v","w")]==before_arrays
    for name in ("u","v","w")
        @test projection_roundoff("paired_"*name,test_array(joinpath(root,"wind-on"),.02,name),test_array(joinpath(root,"control"),.02,name))
        a=test_array(joinpath(root,"control"),.04,name);b=test_array(parent,.04,name)
        @test sqrt(sum(abs2,a.-b)/sum(abs2,b))<2e-9
    end
    qw=onset_test_map(joinpath(root,"wind-on"),.02,"Q");qc=onset_test_map(joinpath(root,"control"),.02,"Q")
    expected_q=zeros(16,16);expected_q[:,end].=.5*.002^2/nu
    @test qw.-qc≈expected_q atol=1e-12
    @test maximum(abs,test_array(joinpath(root,"wind-on"),.04,"u").-test_array(joinpath(root,"control"),.04,"u"))>1e-8
    @test_throws ErrorException onset_test_case(joinpath(root,"forbidden-probe");ustar=.002,init=initial,metadata,
        approval=approval_path,source_id,approval_sha256,jet_mode=2,jet_sign=1,jet_amplitude=2e-4)
    return (;passed=true,architecture=get(ENV,"DRIVER_TEST_ARCH","CPU"),synthetic_source_label=true,
        physical_start=.02,physical_stop=.04,cases=["tiny-parent","control","wind-on"],approval_sha256)
end

onset_test_receipt=Ref{Any}(nothing)
@testset "Preserved parent forcing and matched wind onset" begin
    if isempty(get(ENV,"ONSET_TEST_ROOT",""))
        onset_test_receipt[]=mktempdir(onset_driver_tests)
    else
        root=abspath(ENV["ONSET_TEST_ROOT"])
        ispath(root) && !isempty(readdir(root)) && error("ONSET_TEST_ROOT must be empty")
        mkpath(root);onset_test_receipt[]=onset_driver_tests(root)
    end
end
if !isempty(get(ENV,"ONSET_TEST_ROOT",""))
    write_json(joinpath(abspath(ENV["ONSET_TEST_ROOT"]),"onset-test-results.json"),onset_test_receipt[])
end
println("WIND ONSET DRIVER TESTS PASSED on ",get(ENV,"DRIVER_TEST_ARCH","CPU"))

