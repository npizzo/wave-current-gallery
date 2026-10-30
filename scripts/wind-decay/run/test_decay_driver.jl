# Includes the existing stress, closure, projection, restart and probe checks.
include("test_wind_driver.jl")

function write_decay_test_approval(path,parent_manifest,source_id)
    metadata=TOML.parsefile(replace(parent_manifest,r"\.json$"=>".toml"))
    record=Dict{String,Any}(
        "snapshot_json_sha256"=>filehash(parent_manifest),
        "snapshot_metadata_sha256"=>filehash(replace(parent_manifest,r"\.json$"=>".toml")),
        "expected_parent_ustar"=>metadata["parameters"]["ustar"],
        "allowed_branch_ustar"=>[0.0,metadata["parameters"]["ustar"]],
        "snapshot_time"=>metadata["time"],"source_hashes"=>metadata["source_hashes"],
        "parameters"=>metadata["parameters"],"arrays"=>metadata["arrays"])
    approval=Dict("format"=>"wind-decay-parent-approval-v1","campaign"=>"unity-wind-decay-tests",
        "approved_parent_source_version"=>DECAY_PARENT_SOURCE_VERSION,
        "sources"=>Dict(source_id=>record))
    open(path,"w") do io;TOML.print(io,approval;sorted=true);end
    return approval
end

function decay_driver_tests(root)
    parent=joinpath(root,"tiny-parent")
    wind_test_case(parent;ustar=.002)
    original=test_manifest(parent,.02)
    original_metadata=TOML.parsefile(replace(original,r"\.json$"=>".toml"))
    new_sources=original_metadata["source_hashes"]
    p=(;nx=16,ny=8,nz=16,lx=2.,ly=1.,h=.5,k=6.1,Sref=.17^2*sqrt(9.81/6.1),alpha=.75,waves=1,nu=1e-5,ustar=.002,advection_name="Centered2")

    # Synthetic historical-source fixture: the bytes are a current tiny run,
    # with a separately copied sidecar assigned the one allowed historical
    # fingerprint. This exercises import guards, not historical trajectory
    # agreement. The operations validator separately loads the real parent.
    fixture=joinpath(root,"synthetic-approved-parent");mkdir(fixture)
    initial=joinpath(fixture,basename(original));cp(original,initial)
    for entry in values(original_metadata["arrays"])
        cp(joinpath(parent,entry["file"]),joinpath(fixture,entry["file"]))
    end
    historical=deepcopy(original_metadata);historical["source_hashes"]=copy(DECAY_PARENT_SOURCE_HASHES)
    open(replace(initial,r"\.json$"=>".toml"),"w") do io;TOML.print(io,historical;sorted=true);end
    write(joinpath(fixture,"SYNTHETIC_TEST_ONLY.txt"),"Synthetic source-label fixture for import guard tests; not a historical simulation.\n")
    approval_path=joinpath(root,"test-parent-approval.toml");source_id="synthetic-tiny-parent"
    approval=write_decay_test_approval(approval_path,initial,source_id)
    approval_sha256=filehash(approval_path)
    options=(;approval_path,source_id,approval_sha256)

    @test_throws ErrorException load_wind_snapshot(initial,"full",p,new_sources)
    @test_throws ErrorException load_wind_snapshot(initial,"full",p,new_sources;approval_path,source_id)
    @test_throws ErrorException load_wind_snapshot(initial,"full",p,new_sources;approval_path,source_id,approval_sha256="0"^64)
    @test_throws ErrorException load_wind_snapshot(initial,"mean",merge(p,(;ny=1)),new_sources;options...)
    @test_throws ErrorException load_wind_snapshot(initial,"full",merge(p,(;nu=2e-5)),new_sources;options...)
    @test_throws ErrorException load_wind_snapshot(initial,"full",merge(p,(;ustar=.004)),new_sources;options...)
    @test_throws ErrorException load_wind_snapshot(original,"full",p,new_sources;options...)
    for (label,mutate!) in (
        ("source-version",a->a["approved_parent_source_version"]="unapproved"),
        ("source-hash",a->a["sources"][source_id]["source_hashes"]["run_jet.jl"]="0"^64),
        ("old-wind",a->a["sources"][source_id]["expected_parent_ustar"]=.004),
        ("new-wind",a->a["sources"][source_id]["allowed_branch_ustar"]=[0.,.002,.004]),
        ("time",a->a["sources"][source_id]["snapshot_time"]=1.0),
        ("array",a->a["sources"][source_id]["arrays"]["u"]["sha256"]="0"^64))
        changed=deepcopy(approval);mutate!(changed);bad=joinpath(root,"bad-"*label*".toml")
        open(bad,"w") do io;TOML.print(io,changed;sorted=true);end
        @test_throws ErrorException load_wind_snapshot(initial,"full",p,new_sources;
            approval_path=bad,source_id,approval_sha256=filehash(bad))
    end
    raw_on=load_wind_snapshot(initial,"full",p,new_sources;options...)
    raw_off=load_wind_snapshot(initial,"full",merge(p,(;ustar=0.)),new_sources;options...)
    @test raw_on.approved.parent_ustar==.002
    @test raw_on.approved.branch_ustar==.002
    @test raw_off.approved.branch_ustar==0.
    for (name,a,b) in zip(("u","v","w"),raw_on.arrays,raw_off.arrays)
        @test a==b==test_array(parent,.02,name)
    end

    for (name,ustar) in (("sustained",.002),("removed",0.))
        out=joinpath(root,name)
        wind_test_case(out;ustar,init=initial,approval=approval_path,source_id,approval_sha256)
        arrays=map(field->test_array(out,.02,field),("u","v","w"))
        for (field,a) in zip(("u","v","w"),arrays)
            @test projection_roundoff(name*"_"*field,a,test_array(parent,.02,field))
        end
        @test maximum(abs,native_divergence(arrays...,2/16,1/8,.5/16))<1e-12
        rows=test_wind_rows(out)
        @test maximum(abs(r["UL_depth_mean_change"]) for r in rows)<1e-12
        @test maximum(abs(r["native_depth_mean_tendency"]) for r in rows)<1e-12
        @test all(r["wind_ustar"]≈ustar for r in rows)
        @test all(isfinite,test_array(out,.04,"u"))
        profile=readlines(joinpath(out,"jet-profiles.csv"));columns=split(profile[1],',')
        ci=findfirst(==("closure"),columns);fi=findfirst(==("forcing"),columns)
        initial_profile=[parse.(Float64,split(line,',')) for line in profile[2:17]]
        @test mean(row[ci] for row in initial_profile)≈ustar^2/.5 atol=1e-13
        @test mean(row[fi] for row in initial_profile)≈-ustar^2/.5 atol=1e-15
    end
    for name in ("u","v","w")
        @test projection_roundoff("paired_"*name,test_array(joinpath(root,"removed"),.02,name),
            test_array(joinpath(root,"sustained"),.02,name))
        # A no-change cross-source branch reproduces the original tiny run.
        a=test_array(joinpath(root,"sustained"),.04,name);b=test_array(parent,.04,name)
        @test sqrt(sum(abs2,a.-b)/sum(abs2,b))<2e-9
    end
    @test maximum(abs,test_array(joinpath(root,"removed"),.04,"u").-test_array(joinpath(root,"sustained"),.04,"u"))>1e-8
    @test_throws ErrorException wind_test_case(joinpath(root,"forbidden-probe");ustar=0.,init=initial,
        approval=approval_path,source_id,approval_sha256,jet_mode=2,jet_sign=1,jet_amplitude=2e-4)
    return (;passed=true,architecture=get(ENV,"DRIVER_TEST_ARCH","CPU"),
        source_id,synthetic_source_label=true,physical_start=.02,physical_stop=.04,
        cases=["tiny-parent","sustained","removed"],parent_snapshot_sha256=filehash(initial),approval_sha256)
end

decay_test_receipt=Ref{Any}(nothing)
@testset "Approved parent import and identical-state wind removal" begin
    if isempty(get(ENV,"DECAY_TEST_ROOT",""))
        decay_test_receipt[]=mktempdir(decay_driver_tests)
    else
        test_root=abspath(ENV["DECAY_TEST_ROOT"])
        ispath(test_root) && !isempty(readdir(test_root)) && error("DECAY_TEST_ROOT must be empty")
        mkpath(test_root);decay_test_receipt[]=decay_driver_tests(test_root)
    end
end
if !isempty(get(ENV,"DECAY_TEST_ROOT",""))
    write_json(joinpath(abspath(ENV["DECAY_TEST_ROOT"]),"decay-test-results.json"),decay_test_receipt[])
end
println("WIND DECAY DRIVER TESTS PASSED on ",get(ENV,"DRIVER_TEST_ARCH","CPU"))
