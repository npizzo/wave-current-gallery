#!/usr/bin/env python3
"""Record exact array-task scheduler states without altering any scheduled job."""
import datetime
import json
from pathlib import Path
import subprocess
from guard import verify,write_json
BAD={'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
def normalize(state):return state.strip().split()[0].rstrip('+')
def rows(text):
    result={}
    for line in text.splitlines():
        fields=[v.strip() for v in line.split('|')]
        if len(fields)>=2:result[fields[0]]=fields
    return result

def reconcile(root):
    root=Path(root);source=verify(root)
    if not (root/'submission.json').exists():return
    submission=json.loads((root/'submission.json').read_text());array=str(submission['array'])
    cases=json.loads((root/'campaign.json').read_text())['runs']
    account=subprocess.run(['sacct','--array','-X','-n','-P','-j',array,'--format=JobID%64,State%32,ExitCode,End'],text=True,capture_output=True)
    queue=subprocess.run(['squeue','--array','--noheader','--jobs='+array,'--format=%i|%T|%r'],text=True,capture_output=True)
    history=rows(account.stdout) if account.returncode==0 else {}
    active=rows(queue.stdout) if queue.returncode==0 else {}
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    for i,case in enumerate(cases):
        jid=f'{array}_{i}';fields=active.get(jid) or history.get(jid)
        if not fields:continue
        state=normalize(fields[1]);status=root/'data/status'/case['id']
        record=dict(state=state,job_id=jid,checked_at=now,reason=fields[2] if jid in active and len(fields)>2 else '')
        write_json(status/'scheduler.json',record)
        if state in BAD and not (status/'completed.json').exists():
            failed=json.loads((status/'failed.json').read_text()) if (status/'failed.json').exists() else dict(
                id=case['id'],source_version=source['version'],failed=True,array_index=i,job_id=jid,
                phase='scheduler',message='Job ended before a verified completion receipt was written.')
            failed.update(slurm_state=state,accounting_checked_at=now)
            if jid in history:failed['exit_code']=history[jid][2]
            write_json(status/'failed.json',failed)
        if state=='COMPLETED' and not (status/'completed.json').exists():
            write_json(status/'failed.json',dict(id=case['id'],source_version=source['version'],failed=True,
                array_index=i,job_id=jid,phase='receipt',message='Scheduler completed but verified science/backup receipt is absent.'))
