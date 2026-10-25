#!/usr/bin/env python3
import csv, json, os, subprocess, sys
from pathlib import Path
from submit import check_campaign,digest,write,SEEDS

def main():
    state,campaign=map(Path,sys.argv[1:3]); seed=SEEDS[int(sys.argv[3])]
    record=json.loads((state/'submission.json').read_text())
    if check_campaign(campaign)!=record['campaign_fingerprints']: raise RuntimeError('Campaign changed')
    for name,sha in record['source_sha256'].items():
        if digest(state/'code'/name)!=sha: raise RuntimeError('Frozen analysis source changed')
    output=state/'results'/f'seed-{seed}'
    julia=os.environ['JULIA']
    subprocess.run([julia,'--startup-file=no',str(state/'code/test_audit_math.jl')],check=True)
    subprocess.run([julia,'--startup-file=no',str(state/'code/test_packet_fields.jl')],check=True)
    subprocess.run([julia,'--startup-file=no',str(state/'code/analyze.jl'),str(campaign),str(output),str(seed)],check=True)
    qc=list(csv.DictReader((output/'qc.csv').open()))
    if len(qc)!=28: raise RuntimeError('Not all member/time/stencil combinations completed')
    if check_campaign(campaign)!=record['campaign_fingerprints']: raise RuntimeError('Campaign changed during analysis')
    write(output/'completed.json',{'completed':True,'seed':seed,'source_version':record['version'],
        'files':{p.name:digest(p) for p in output.iterdir() if p.is_file()},
        'max_scaled_native_divergence':max(float(x['native_divergence_scaled']) for x in qc),
        'max_R3D_algebraic_identity_error':max(float(x['R3D_identity_error']) for x in qc)})
    print(f'Completed read-only snapshot audit for seed {seed}',flush=True)
if __name__=='__main__': main()
