using Test, Statistics, TOML, SHA
filehash(path)=open(io->bytes2hex(sha256(io)),path)
include("regime_physics.jl")
include("wind_physics.jl")
const TEST_ALLOWED_BRANCHES=Set(vcat([("full",ustar) for ustar in (0.,.00075,.001,.0011,.00125,.0015,.002)],
    [("mean",ustar) for ustar in (0.,.00125,.002)]))

@testset "Per-source frozen intervention matrix" begin
    record=Dict("allowed_branches"=>[Dict("kind"=>"full","ustar"=>.0011)])
    @test approved_shear_branches(record)==Set([("full",.0011)])
    for entries in ([],[Dict("kind"=>"plane","ustar"=>0.)],
            [Dict("kind"=>"full","ustar"=>-1.)],[Dict("kind"=>"full","ustar"=>Inf)],
            [Dict("kind"=>"full","ustar"=>NaN)],[Dict("kind"=>"full","ustar"=>true)],
            [Dict("kind"=>"full","ustar"=>"0.001")],
            [Dict("kind"=>"full","ustar"=>0.,"extra"=>1)],
            [Dict("kind"=>"full","ustar"=>0.),Dict("kind"=>"full","ustar"=>0.)])
        @test_throws ErrorException approved_shear_branches(Dict("allowed_branches"=>entries))
    end
end

@testset "Retained native background and incremental stress" begin
    n=96;h=.5;nu=.004*.25/300;k=6.1;S=.17^2*sqrt(9.81/k);alpha=.75;dz=h/n
    B=(1-alpha)*S.*exp.(2k.*(-h.+((1:n).-.5).*dz));g0=2k*(1-alpha)*S;gb=g0*exp(-2k*h)
    params=regime_forcing_parameters(waves=1,alpha=alpha,S=S,k=k,h=h,nz=n,nu=nu)
    for ust in (0.,.00075,.001,.0011,.00125,.0015,.002)
        force=[onset_total_force(1,1,j,nothing,nothing,nothing,merge(params,(;wind_return_acceleration=-ust^2/h))) for j=1:n]
        visc=nu.*vcat((B[2]-B[1]-gb*dz)/dz^2,diff(diff(B))./dz^2,(B[n-1]-B[n]+(g0+ust^2/nu)*dz)/dz^2)
        expected=fill(-ust^2/h,n);expected[end]+=ust^2/dz
        @test visc+force≈expected atol=1e-16
        @test abs(mean(visc+force))<1e-16
        @test mean(force)≈-nu*(g0-gb)/h-ust^2/h atol=1e-16
    end
end

@testset "Pinned stress/projection matrix and exact 2D3C import" begin
    mktempdir() do root
        native=joinpath(root,"native");mkdir(native);arrays=Dict{String,Any}()
        expected=([.001k+.0001cos(2pi*j/8) for i in 1:16,j in 1:8,k in 1:16],
                  [.0002cos(2pi*i/16) for i in 1:16,j in 1:8,k in 1:16],zeros(16,8,17))
        for (name,a) in zip(("u","v","w"),expected)
            file=name*".bin";write(joinpath(native,file),a)
            arrays[name]=(;file,shape=collect(size(a)),dtype="float64",order="F",sha256=filehash(joinpath(native,file)))
        end
        snap=joinpath(native,"t.json");write(snap,"{}\n")
        p=(;nx=16,ny=8,nz=16,lx=2.,ly=1.,h=.5,k=6.1,Sref=.036,alpha=.75,uoffset=0.,waves=1,nu=1e-5,ustar=0.,maintain=true,advection_name="Centered2")
        write_wind_snapshot_metadata(snap,(;time=600.,iteration=3,arrays),p,ONSET_PARENT_SOURCE_HASHES)
        oldmeta=replace(snap,".json"=>".toml");m=TOML.parsefile(oldmeta);rm(oldmeta)
        m["translation"]="legacy-native-json-plus-original-configuration-v1";m["original_configuration_sha256"]="a"^64
        md=joinpath(root,"translated.toml");open(io->TOML.print(io,m),md,"w")
        id="synthetic-test"
        record=Dict("snapshot_json_sha256"=>filehash(snap),"snapshot_metadata_sha256"=>filehash(md),"snapshot_time"=>600.,
            "parent_configuration_sha256"=>"a"^64,"expected_parent_ustar"=>0.,"metadata_translation"=>"legacy-native-json-plus-original-configuration-v1","expected_maintain_background"=>true,
            "allowed_branches"=>[Dict("kind"=>kind,"ustar"=>ustar) for (kind,ustar) in sort(collect(TEST_ALLOWED_BRANCHES))],
            "source_hashes"=>ONSET_PARENT_SOURCE_HASHES,"parameters"=>m["parameters"],"arrays"=>m["arrays"])
        approval=Dict("format"=>"shear-budget-parent-approval-v1","campaign"=>"unity-shear-budget-study", "approved_parent_source_version"=>ONSET_PARENT_SOURCE_VERSION,"sources"=>Dict(id=>record))
        ap=joinpath(root,"approval.toml");open(io->TOML.print(io,approval),ap,"w")
        options=(;metadata_path=md,approval_path=ap,source_id=id,approval_sha256=filehash(ap));current=Dict("current"=>"hash")
        input_hashes=[filehash(joinpath(native,arrays[name].file)) for name in ("u","v","w")]
        for (kind,ustar) in sort(collect(TEST_ALLOWED_BRANCHES))
            r=load_wind_snapshot(snap,kind,merge(p,(;ustar,ny=kind=="mean" ? 1 : 8)),current;options...)
            want=kind=="mean" ? map(a->Array(mean(a;dims=2)),expected) : expected
            @test r.arrays==want
            @test r.approved.projection==kind
            @test r.approved.maintain_background && r.approved.branch_ustar==ustar
            @test maximum(abs,r.arrays[2])>1e-4
            for (a,b) in zip(r.arrays,expected)
                @test mean(a;dims=(1,2))≈mean(b;dims=(1,2)) atol=1e-16
            end
        end
        @test_throws ErrorException load_wind_snapshot(snap,"full",p,current;metadata_path=md)
        for edit in ((;ustar=.004),(;maintain=false),(;nu=2e-5),(;waves=0),(;uoffset=.1),(;nx=32),(;nz=32),(;ly=2.))
            @test_throws ErrorException load_wind_snapshot(snap,"full",merge(p,edit),current;options...)
        end
        @test_throws ErrorException load_wind_snapshot(snap,"mean",merge(p,(;ny=1,ustar=.0011)),current;options...)
        @test_throws ErrorException load_wind_snapshot(snap,"mean",p,current;options...)
        @test_throws ErrorException load_wind_snapshot(snap,"full",merge(p,(;ny=1)),current;options...)
        @test [filehash(joinpath(native,arrays[name].file)) for name in ("u","v","w")]==input_hashes
        @test !isfile(oldmeta)

        # A second source record may allow a strict subset of the campaign;
        # another source's permissions never authorize this one.
        restricted=deepcopy(approval)
        restricted["sources"][id]["allowed_branches"]=[Dict("kind"=>"full","ustar"=>.0011)]
        restricted_path=joinpath(root,"restricted.toml");open(io->TOML.print(io,restricted),restricted_path,"w")
        restricted_options=(;metadata_path=md,approval_path=restricted_path,source_id=id,approval_sha256=filehash(restricted_path))
        @test load_wind_snapshot(snap,"full",merge(p,(;ustar=.0011)),current;restricted_options...).arrays==expected
        @test_throws ErrorException load_wind_snapshot(snap,"full",merge(p,(;ustar=.0015)),current;restricted_options...)
        @test_throws ErrorException load_wind_snapshot(snap,"full",merge(p,(;ustar=.0011)),current;
            metadata_path=md,approval_path=restricted_path,source_id=id,approval_sha256=filehash(ap))

        # Synthetic derived provenance checks the distinct import route. Real
        # prolongation/projection is checked by test_refine_native_snapshot.py
        # and the campaign's independently pinned full-grid input validation.
        fine=joinpath(root,"derived");mkdir(fine);fine_arrays=Dict{String,Any}()
        for (name,a) in zip(("u","v","w"),expected)
            b=repeat(a;inner=(2,1,1));file=name*".bin";write(joinpath(fine,file),b)
            fine_arrays[name]=(;file,shape=collect(size(b)),dtype="float64",order="F",sha256=filehash(joinpath(fine,file)))
        end
        fine_json=joinpath(fine,"fine.json");write(fine_json,"{}\n")
        fine_p=merge(p,(;nx=32));resampler=filehash(joinpath(@__DIR__,"refine_native_snapshot.py"))
        write_wind_snapshot_metadata(fine_json,(;time=600.,iteration=3,arrays=fine_arrays),fine_p,Dict("refine_native_snapshot.py"=>resampler))
        fine_meta_path=replace(fine_json,".json"=>".toml");fm=TOML.parsefile(fine_meta_path)
        fm["translation"]="native-parent-refinement-v1";fm["original_configuration_sha256"]="a"^64
        fm["derived_from"]=Dict("method"=>"periodic-spectral-even-odd-native-projection-v1","source_version"=>ONSET_PARENT_SOURCE_VERSION,
            "source_hashes"=>ONSET_PARENT_SOURCE_HASHES,"source_snapshot_sha256"=>filehash(snap),"source_metadata_sha256"=>filehash(md),
            "source_parameters"=>m["parameters"],"source_arrays"=>m["arrays"],"target_grid"=>[32,8,16],"resampler_sha256"=>resampler,
            "diagnostics"=>Dict("scaled_divergence_after"=>0.,"projection_fraction_eddy"=>0.))
        open(io->TOML.print(io,fm),fine_meta_path,"w")
        fr=copy(record);merge!(fr,Dict("snapshot_json_sha256"=>filehash(fine_json),"snapshot_metadata_sha256"=>filehash(fine_meta_path),
            "parameters"=>fm["parameters"],"arrays"=>fm["arrays"],"source_hashes"=>fm["source_hashes"],
            "metadata_translation"=>fm["translation"],"derived_from"=>fm["derived_from"]))
        fa=deepcopy(approval);fa["sources"]["derived-test"]=fr
        fap=joinpath(root,"fine-approval.toml");open(io->TOML.print(io,fa),fap,"w")
        fopts=(;metadata_path=fine_meta_path,approval_path=fap,source_id="derived-test",approval_sha256=filehash(fap))
        loaded=load_wind_snapshot(fine_json,"full",fine_p,current;fopts...)
        @test loaded.approved.initial_state_translation=="native-parent-refinement-v1"
        @test size(loaded.arrays[1])==(32,8,16)
        @test_throws ErrorException load_wind_snapshot(fine_json,"full",p,current;fopts...)
        # A byte-pinned but scientifically different derivation is rejected.
        bad=deepcopy(fm);bad["derived_from"]["resampler_sha256"]="0"^64
        bad_meta=joinpath(fine,"bad.toml");open(io->TOML.print(io,bad),bad_meta,"w")
        bad_approval=deepcopy(fa);bad_approval["sources"]["derived-test"]["derived_from"]=bad["derived_from"]
        bad_approval["sources"]["derived-test"]["snapshot_metadata_sha256"]=filehash(bad_meta)
        bad_path=joinpath(root,"bad-fine.toml");open(io->TOML.print(io,bad_approval),bad_path,"w")
        @test_throws ErrorException load_wind_snapshot(fine_json,"full",fine_p,current;
            metadata_path=bad_meta,approval_path=bad_path,source_id="derived-test",approval_sha256=filehash(bad_path))
    end
end
