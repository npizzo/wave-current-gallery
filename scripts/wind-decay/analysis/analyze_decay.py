#!/usr/bin/env python3
"""Read-only assessment of matched wind-removal and sustained-wind branches.

Only verified completed, backed-up diagnostics enter scientific output. Native
snapshots and mean-map binaries are used privately and never copied to the page.
"""
from __future__ import annotations
import argparse, csv, hashlib, html, json, math, re, shutil
import tomllib
from pathlib import Path
import numpy as np

PROFILE_FIELDS=('UL','UE','adv_full','adv_ky0','adv_3d','wave','closure','forcing','pressure_mean','native_total','flux_total','flux_ky0','flux_3d')
PUBLIC_CONFIG=('nx','ny','nz','lx','ly','h','k','S','S_reference','alpha','nu','eddy_rms','injection_wavelength','seed','max_dt','wind_ustar','start_time','stop_time','advection','closure','velocity_convention','waves','diagnostic_interval','mean_interval','initial_snapshot_sha256')
CRITERIA=dict(window_s=200.,weak_correction=.1,initial_eddy_rms=.004,current_activity_m_s=.00004,q_activity_s_inverse=.000016,trim_reference_nz=96,trim_reference_cells=2,active_stokes_efolds=3.,extremum_presence=.8)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x


def write_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')


def identifier(s):
    if not isinstance(s,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',s):raise ValueError('Invalid run ID')
    return s


def safe_path(root,name):
    root=Path(root).resolve();p=Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:raise ValueError('Unsafe relative path')
    dest=(root/p).resolve()
    if root not in dest.parents:raise ValueError('Path escapes root')
    return dest


def numeric_csv(path):
    with Path(path).open(newline='') as f:
        rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    if not rows or not all(math.isfinite(v) for r in rows for v in r.values()):raise ValueError('Empty or nonfinite diagnostics')
    return rows


def write_csv(path,rows):
    if not rows:return
    keys=list(rows[0])
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:(format(float(v),'.8g') if isinstance(v,(float,np.floating)) else v) for k,v in r.items()})


def integrate(y,t):
    y=np.asarray(y);t=np.asarray(t)
    return np.sum((y[1:]+y[:-1])*.5*np.diff(t).reshape((-1,)+(1,)*(y.ndim-1)),axis=0)


def cumulative(y,t):
    y=np.asarray(y);dt=np.diff(t).reshape((-1,)+(1,)*(y.ndim-1))
    return np.concatenate((np.zeros_like(y[:1]),np.cumsum((y[1:]+y[:-1])*.5*dt,axis=0)),axis=0)


def rms(a,dz):return np.sqrt(np.sum(np.asarray(a)**2*dz,axis=-1)/np.sum(dz))


def laminar_mean(initial_ul,elapsed,h,nu,ustar):
    """Exact semidiscrete 1D constant-viscosity response on the same uniform grid.

    Top stress ustar^2, bottom stress zero, and uniform -ustar^2/H return
    acceleration. Cell-centered Neumann finite-volume Laplacian is diagonal in
    a DCT-II basis; no fitted parameters and no turbulent tendency are used.
    """
    u=np.asarray(initial_ul,float);t=np.asarray(elapsed,float);n=len(u);dz=h/n
    if n<4 or nu<=0 or np.any(t<0):raise ValueError('Invalid laminar reference')
    m=np.arange(n);phi=np.cos(np.pi*(np.arange(n)+.5)[:,None]*m[None,:]/n)
    norm=np.sum(phi*phi,axis=0);a=u@phi/norm
    force=np.full(n,-ustar**2/h);force[-1]+=ustar**2/dz
    f=force@phi/norm;lam=-4*nu/dz**2*np.sin(np.pi*m/(2*n))**2
    decay=np.exp(t[:,None]*lam[None,:]);forcing=np.empty_like(decay)
    forcing[:,0]=t*f[0]
    forcing[:,1:]=np.expm1(t[:,None]*lam[None,1:])*f[None,1:]/lam[None,1:]
    return (decay*a[None,:]+forcing)@phi.T


def mode_coefficients(a,h,z,dz,max_mode=8):
    modes=np.arange(1,max_mode+1);phi=np.cos(np.pi*(z[:,None]+h)*modes[None,:]/h)
    norm=np.sum(phi**2*dz[:,None],axis=0)
    return (np.asarray(a)*dz)@phi/norm, norm/h


def extrema(a,z,floor):
    """Strict interior extrema with two-sided prominence and half width."""
    out=[]
    for sign,kind in ((1.,'maximum'),(-1.,'minimum')):
        y=sign*a
        peaks=[i for i in range(1,len(z)-1) if y[i]>y[i-1] and y[i]>y[i+1]]
        lows=[i for i in range(1,len(z)-1) if y[i]<y[i-1] and y[i]<y[i+1]]
        for i in peaks:
            left=max([0]+[j for j in lows if j<i]);right=min([len(z)-1]+[j for j in lows if j>i]);p=y[i]-max(y[left],y[right])
            if p<floor:continue
            level=y[i]-p/2;lo=i;hi=i
            while lo>left and y[lo]>level:lo-=1
            while hi<right and y[hi]>level:hi+=1
            zl=z[lo]+(level-y[lo])*(z[lo+1]-z[lo])/(y[lo+1]-y[lo])
            zr=z[hi-1]+(level-y[hi-1])*(z[hi]-z[hi-1])/(y[hi]-y[hi-1])
            out.append(dict(kind=kind,z_m=float(z[i]),current_m_s=float(a[i]),prominence_m_s=float(p),width_m=float(zr-zl)))
    return out


def persistent_extrema(values,time,z,floor,reference):
    late=time>=time[-1]-200.-1e-7;average=integrate(values[late],time[late])/(time[late][-1]-time[late][0])
    initial=extrema(reference,z,floor);found=extrema(average,z,floor)
    history=[extrema(a,z,floor) for a in values[late]]
    for e in found:
        e['width_resolved']=e['width_m']>=4*float(np.max(np.diff(z)))
        tolerance=max(e['width_m']/2,2*float(np.max(np.diff(z))))
        e['presence_fraction']=sum(any(q['kind']==e['kind'] and q['width_m']>=4*float(np.max(np.diff(z))) and abs(q['z_m']-e['z_m'])<=tolerance for q in row) for row in history)/len(history)
        e['persistent']=e['width_resolved'] and e['presence_fraction']>=.8
        e['matches_initial_extremum']=any(q['kind']==e['kind'] and abs(q['z_m']-e['z_m'])<=tolerance for q in initial)
    return found


class Run:
    def __init__(self,root,entry,receipt):
        self.root=safe_path(root,entry.get('path','data/runs/'+entry['id']));self.receipt=receipt;self.checked={}
    def file(self,name):
        p=safe_path(self.root,name);sha=self.receipt.get('files',{}).get(name)
        if not sha or not p.is_file() or digest(p)!=sha:raise ValueError('Missing or changed verified diagnostic: '+name)
        self.checked[name]=sha;return p
    def json(self,name):return json.loads(self.file(name).read_text())
    def csv(self,name):return numeric_csv(self.file(name))
    def profiles(self):
        rows=self.csv('jet-profiles.csv');time=np.unique([r['time_s'] for r in rows]);z=np.unique([r['z_m'] for r in rows])
        if len(time)<3 or len(z)<8 or len(rows)!=len(time)*len(z):raise ValueError('Incomplete profile grid')
        fields={name:np.zeros((len(time),len(z))) for name in PROFILE_FIELDS};seen=set();dz=np.zeros(len(z))
        for r in rows:
            i=int(np.searchsorted(time,r['time_s']));j=int(np.searchsorted(z,r['z_m']))
            if (i,j) in seen:raise ValueError('Duplicate profile cell')
            seen.add((i,j));dz[j]=r['dz_m']
            for name in PROFILE_FIELDS:fields[name][i,j]=r[name]
        fields.update(time=time,z=z,dz=dz);return fields
    def maps(self,fields=('Uc','Wc','Q')):
        names=sorted(n for n in self.receipt['files'] if re.fullmatch(r'mean_[0-9]+\.[0-9]+\.json',n))
        if len(names)<2:raise ValueError('Missing initial/final mean maps')
        result={name:[] for name in fields};time=[]
        # Initial/final and a small fixed set of intermediate fields stay private.
        for name in names:
            m=self.json(name);elapsed=m['time']-self.start
            if name not in (names[0],names[-1]) and min(abs(elapsed-x) for x in (200,600,1200,2000))>1e-6:continue
            time.append(m['time'])
            for field in fields:
                e=m['arrays'][field]
                if e['dtype']!='float64' or e['order']!='F' or e['sha256']!=self.receipt['files'].get(e['file']):raise ValueError('Map metadata mismatch')
                a=np.fromfile(self.file(e['file']),dtype='<f8')
                if a.size!=np.prod(e['shape']):raise ValueError('Truncated map')
                a=a.reshape(e['shape'],order='F')
                if a.ndim!=2 or not np.all(np.isfinite(a)):raise ValueError('Invalid mean map')
                result[field].append(a)
        result={k:np.array(v) for k,v in result.items()};result['time']=np.array(time);return result


def run_state(root,entry,source,approved):
    folder=root/'data/status'/identifier(entry['id']);p=folder/'completed.json'
    if p.exists() and (approved is None or entry['id'] in approved):
        r=json.loads(p.read_text())
        if r.get('id')!=entry['id'] or r.get('source_version')!=source:raise ValueError('Receipt source or case mismatch')
        if r.get('completed') is not True or r.get('persistent_backup_verified') is not True:raise ValueError('Completion is not backed up and verified')
        return 'complete',r
    if (folder/'failed.json').exists():return 'failed',None
    if (folder/'scheduler.json').exists():
        state=json.loads((folder/'scheduler.json').read_text()).get('state','').upper()
        if state in ('RUNNING','COMPLETING'):return 'running',None
        if state in ('PENDING','CONFIGURING','REQUEUED','SUSPENDED'):return 'queued',None
    if (folder/'started.json').exists():return 'running',None
    return ('queued' if (root/'submission.json').exists() else 'waiting'),None


def assess(root,entry,receipt):
    run=Run(root,entry,receipt);cfg=run.json('configuration.json');completed=run.json('completed.json');grid=run.json('grid.json');p=run.profiles()
    start=float(cfg['start_time']);stop=float(cfg['stop_time']);run.start=start
    for actual,expected in ((p['time'][0],start),(p['time'][-1],stop),(completed['time'],stop)):
        if not np.isclose(actual,expected,rtol=0,atol=1e-7):raise ValueError('Incomplete physical timeline')
    for name,expected in (('START_TIME',start),('STOP_TIME',stop),('WIND_USTAR',cfg['wind_ustar'])):
        if name in entry.get('env',{}) and not np.isclose(float(entry['env'][name]),expected,rtol=1e-12,atol=1e-10):raise ValueError('Case does not match requested '+name)
    if not cfg.get('initial_snapshot_sha256'):raise ValueError('Restart source fingerprint missing')
    approval_path=root/'source_provenance.toml'
    if approval_path.exists():
        approval=tomllib.loads(approval_path.read_text())['sources'][entry['source_id']]
        if cfg['initial_snapshot_sha256']!=approval['snapshot_json_sha256']:raise ValueError('Restart differs from frozen selected parent')
        if not np.isclose(start,approval['snapshot_time'],atol=1e-8):raise ValueError('Restart differs from selected physical time')
        for key in ('snapshot_json_sha256','snapshot_metadata_sha256'):
            if receipt.get(key)!=approval[key]:raise ValueError('Receipt parent provenance differs')
        if receipt.get('approval_sha256')!=digest(approval_path):raise ValueError('Receipt approval fingerprint differs')
    if cfg.get('velocity_convention')!='Lagrangian':raise ValueError('Unsupported velocity convention')
    h=float(cfg['h']);nu=float(cfg['nu']);ustar=float(cfg['wind_ustar']);t=p['time'];tau=t-start;z=p['z'];dz=p['dz']
    if not np.allclose(dz,h/len(z),rtol=1e-10,atol=1e-12):raise ValueError('Laminar reference requires uniform vertical grid')
    baseline_ul=laminar_mean(p['UL'][0],tau,h,nu,ustar);stokes=p['UL'][0]-p['UE'][0]
    if not np.allclose(p['UL']-p['UE'],stokes[None,:],rtol=0,atol=1e-11):raise ValueError('Stokes profile is not stationary')
    baseline=baseline_ul-stokes;residual=p['UE']-baseline
    a,norm=mode_coefficients(residual,h,z,dz);raw,_=mode_coefficients(p['UE'],h,z,dz)
    terms={name:mode_coefficients(p[name],h,z,dz)[0] for name in ('adv_full','adv_ky0','adv_3d','wave','closure','forcing','pressure_mean')}
    eig=-4*nu/(h/len(z))**2*np.sin(np.pi*np.arange(1,9)/(2*len(z)))**2
    sink=a*eig[None,:];residual_rate=terms['adv_full']+terms['wave']+terms['pressure_mean']+sink
    rateintegral=cumulative(residual_rate,t);mode_residual=a-a[0]-rateintegral
    mode_scales=np.maximum(np.max(abs(a-a[0]),axis=0),np.sum(abs(residual_rate[1:]+residual_rate[:-1])*.5*np.diff(t)[:,None],axis=0));mode_scales=np.maximum(mode_scales,1e-8)
    supplies={name:cumulative(a*val*norm[None,:],t) for name,val in terms.items() if name in ('adv_full','adv_ky0','adv_3d','wave','pressure_mean')}
    supplies['viscous_residual']=cumulative(a*sink*norm[None,:],t)
    b=run.csv('regime-budget.csv');w=run.csv('wind-budget.csv');bt=np.array([r['time_s'] for r in b]);wt=np.array([r['time_s'] for r in w])
    if not np.allclose(bt,wt,rtol=0,atol=1e-7) or not np.allclose(bt,t,rtol=0,atol=1e-7):raise ValueError('Diagnostic sample times differ')
    if any(not np.isclose(r['wind_ustar'],ustar,rtol=0,atol=1e-12) for r in w):raise ValueError('Wind stress changed within branch')
    energy={key:np.array([r[key] for r in b]) for key in ('Eeddy','Eky0_eddy','Eperp','chi_3d','eddy_rms')};energy['roll']=np.array([r['roll_energy_xmean'] for r in w])
    windows=[]
    for lo in np.arange(0,tau[-1]-199.999,200.):
        mask=(tau>=lo-1e-7)&(tau<=lo+200.+1e-7)
        if abs(tau[mask][0]-lo)>1e-6 or abs(tau[mask][-1]-lo-200)>1e-6:raise ValueError('Missing fixed-window endpoints')
        for region in ('full','interior','stokes'):
            C=np.array([r['C_native_rms_'+region] for r in w])[mask];qref=w[int(np.flatnonzero(mask)[0])]['q_eddy_rms_'+region]
            eps=float(integrate(C,t[mask])/qref) if qref>0 else None
            qactive=qref>=CRITERIA['q_activity_s_inverse'];motion=float(np.max(energy['eddy_rms'][mask]))>=CRITERIA['current_activity_m_s']
            windows.append(dict(start_elapsed_s=float(lo),stop_elapsed_s=float(lo+200),region=region,epsilon_C=eps,q_reference_s_inverse=qref,source_integral_s_inverse=float(integrate(C,t[mask])),active=qactive and motion,
                classification='WEAK_CORRECTION' if qactive and motion and eps is not None and eps<=.1 else 'LARGE_CORRECTION' if qactive and motion else 'QUIET_OR_UNRESOLVED_REFERENCE'))
    full_rate=p['native_total'];mean_budget_error=p['UE']-p['UE'][0]-cumulative(full_rate,t)
    mean_rate_reference=max(float(np.max(rms(p['UE']-p['UE'][0],dz))),float(np.max(rms(cumulative(abs(full_rate),t),dz))),1e-8)
    mean_budget_rel=float(np.max(rms(mean_budget_error,dz))/mean_rate_reference)
    extrema_initial=extrema(p['UE'][0],z,CRITERIA['current_activity_m_s']);late_extrema=persistent_extrema(p['UE'],t,z,CRITERIA['current_activity_m_s'],p['UE'][0])
    residual_extrema=persistent_extrema(residual,t,z,CRITERIA['current_activity_m_s'],residual[0])
    primary=tau<=600.+1e-7
    primary_extrema=persistent_extrema(p['UE'][primary],t[primary],z,CRITERIA['current_activity_m_s'],p['UE'][0])
    primary_residual_extrema=persistent_extrema(residual[primary],t[primary],z,CRITERIA['current_activity_m_s'],residual[0])
    maps=run.maps()
    if not np.isclose(maps['time'][0],start,atol=1e-7) or not np.isclose(maps['time'][-1],stop,atol=1e-7):raise ValueError('Map timeline missing start or end')
    histories=[]
    bandenergy=.5*np.sum(a[:,1:]**2*norm[None,1:],axis=1)
    beta=np.gradient(np.gradient(p['UE'],z,axis=1,edge_order=2),z,axis=1,edge_order=2)
    for i,time in enumerate(t):
        row=dict(time_s=float(time),elapsed_s=float(tau[i]),eddy_rms_m_s=energy['eddy_rms'][i],eddy_2d_rms_m_s=np.sqrt(2*energy['Eky0_eddy'][i]),eddy_3d_rms_m_s=np.sqrt(2*energy['Eperp'][i]),roll_rms_m_s=np.sqrt(2*energy['roll'][i]),chi_3d=energy['chi_3d'][i],roll_energy_fraction=energy['roll'][i]/energy['Eeddy'][i] if energy['Eeddy'][i]>0 else 0.,mean_residual_rms_m_s=rms(residual[i],dz),band_residual_energy_m2_s2=bandenergy[i],beta_rms_s_inverse_m_inverse=rms(beta[i,2:-2],dz[2:-2]))
        for region in ('full','interior','stokes'):
            row['C_rms_'+region]=w[i]['C_native_rms_'+region];row['q_eddy_rms_'+region]=w[i]['q_eddy_rms_'+region]
        for j in range(8):
            row['residual_mode_%d_m_s'%(j+1)]=a[i,j];row['actual_mode_%d_m_s'%(j+1)]=raw[i,j]
        row['band_advection_work_m2_s2']=float(np.sum(supplies['adv_full'][i,1:]));row['band_viscous_work_m2_s2']=float(np.sum(supplies['viscous_residual'][i,1:]));histories.append(row)
    profile_windows=[]
    for lo in np.arange(0,tau[-1]-199.999,200.):
        use=(tau>=lo-1e-7)&(tau<=lo+200+1e-7);ids=np.flatnonzero(use);i,j=ids[0],ids[-1]
        actual_extrema=persistent_extrema(p['UE'][use],t[use],z,CRITERIA['current_activity_m_s'],p['UE'][0])
        laminar_extrema=persistent_extrema(baseline[use],t[use],z,CRITERIA['current_activity_m_s'],p['UE'][0])
        for e in actual_extrema:
            tolerance=max(e['width_m']/2,2*float(np.max(dz)))
            e['matches_laminar_extremum']=any(q['kind']==e['kind'] and abs(q['z_m']-e['z_m'])<=tolerance for q in laminar_extrema)
        profile_windows.append(dict(start_elapsed_s=float(lo),stop_elapsed_s=float(lo+200),
            actual_extrema=actual_extrema,laminar_extrema=laminar_extrema,
            residual_extrema=persistent_extrema(residual[use],t[use],z,CRITERIA['current_activity_m_s'],residual[0]),
            residual_band_energy_change_m2_s2=float(bandenergy[j]-bandenergy[i]),
            residual_band_advection_work_m2_s2=float(np.sum(supplies['adv_full'][j,1:]-supplies['adv_full'][i,1:])),
            residual_band_viscous_work_m2_s2=float(np.sum(supplies['viscous_residual'][j,1:]-supplies['viscous_residual'][i,1:])),
            residual_max_rms_m_s=float(np.max(rms(residual[use],dz)))))
    late_extrema=profile_windows[-1]['actual_extrema']
    primary_extrema=next((w['actual_extrema'] for w in profile_windows if w['stop_elapsed_s']==600),primary_extrema)
    result=dict(id=entry['id'],arm=entry.get('arm','wind_off' if ustar==0 else 'wind_on'),status='complete',configuration={k:cfg[k] for k in PUBLIC_CONFIG if k in cfg},
        initial=dict(eddy_rms_m_s=energy['eddy_rms'][0],roll_rms_m_s=np.sqrt(2*energy['roll'][0]),roll_energy_fraction=energy['roll'][0]/energy['Eeddy'][0],chi_3d=energy['chi_3d'][0],eddy_2d_rms_m_s=np.sqrt(2*energy['Eky0_eddy'][0])),
        final=histories[-1],windows=windows,profile_windows=profile_windows,initial_extrema=extrema_initial,late_actual_extrema=late_extrema,late_residual_extrema=residual_extrema,
        primary_600s_actual_extrema=primary_extrema,primary_600s_residual_extrema=primary_residual_extrema,
        mean_momentum_budget_relative_error=mean_budget_rel,mode_budget_relative_errors=np.max(abs(mode_residual),axis=0)/mode_scales,
        residual_band_max_rms_m_s=float(np.sqrt(2*np.max(bandenergy))),residual_band_peak_elapsed_s=float(tau[np.argmax(bandenergy)]),
        qualification='Descriptive matched trajectory, not an ensemble instability test; weak C alone does not certify autonomous mean dynamics',
        numerical_scales=dict(mode1_viscous_efold_s=-1/eig[0],mode2_viscous_efold_s=-1/eig[1],mode4_viscous_efold_s=-1/eig[3]))
    curves=dict(time=t,elapsed=tau,z=z,dz=dz,UE=p['UE'],UL=p['UL'],baseline=baseline,residual=residual,profiles=p,histories=histories,maps=maps,bandenergy=bandenergy,supplies=supplies)
    return result,curves,run.checked


def compare_pair(results,curves):
    complete=[r for r in results if r['status']=='complete']
    if len(complete)<2:return dict(status='AWAITING_MATCHED_PAIR',reason='Both verified branches are required')
    if len(complete)!=2:raise ValueError('This protocol expects exactly two matched branches')
    off=next((r for r in complete if r['configuration']['wind_ustar']==0),None);on=next((r for r in complete if r['configuration']['wind_ustar']>0),None)
    if off is None or on is None:raise ValueError('Need wind-off and sustained-wind branches')
    a,b=off['configuration'],on['configuration']
    for key in ('initial_snapshot_sha256','nx','ny','nz','lx','ly','h','k','S','S_reference','alpha','nu','max_dt','start_time','stop_time','waves','advection','closure'):
        if a.get(key)!=b.get(key):raise ValueError('Pair physical parameters differ: '+key)
    x,y=curves[off['id']],curves[on['id']]
    if not np.array_equal(x['time'],y['time']):raise ValueError('Pair sample times differ')
    for key in ('UE','UL'):
        if not np.allclose(x[key][0],y[key][0],rtol=0,atol=1e-11):raise ValueError('Pair initial profiles differ')
    for key in ('Uc','Wc'):
        if not np.allclose(x['maps'][key][0],y['maps'][key][0],rtol=0,atol=1e-11):raise ValueError('Pair initial maps differ')
    # Native boundary vorticity uses the prescribed tangential gradient. The
    # velocity state is identical, but switching stress changes this one row.
    qjump=np.zeros_like(x['maps']['Q'][0]);qjump[:,-1]=.5*(a['wind_ustar']**2-b['wind_ustar']**2)/a['nu']
    if not np.allclose(x['maps']['Q'][0]-y['maps']['Q'][0],qjump,rtol=0,atol=1e-9):raise ValueError('Initial native Q differs beyond the expected boundary-gradient change')
    weak=[]
    for start in sorted(set(w['start_elapsed_s'] for w in off['windows'])):
        rows=[w for w in off['windows'] if w['start_elapsed_s']==start and w['region'] in ('interior','stokes')]
        if len(rows)==2 and all(r['classification']=='WEAK_CORRECTION' for r in rows):weak.append(start)
    new=[e for e in off['late_actual_extrema'] if e['persistent'] and not e['matches_initial_extremum'] and not e['matches_laminar_extremum']]
    def candidates(result):
        return [w['start_elapsed_s'] for w in result['profile_windows'] if any(e['persistent'] and not e['matches_initial_extremum'] and not e['matches_laminar_extremum'] for e in w['actual_extrema'])]
    newwindows=candidates(off)
    # Intentionally no automatic PASS: inherited extrema, finite-time response,
    # turbulence decay, and one realization make that stronger claim unjustified.
    return dict(status='DESCRIPTIVE_PAIR_COMPLETE',identical_restart_verified=True,wind_off_id=off['id'],wind_on_id=on['id'],weak_feedback_window_starts_s=weak,
        initial_top_cell_Q_off_minus_on_s_inverse=float(qjump[0,-1]),
        primary_weak_feedback_window_starts_s=[t for t in weak if t<600],
        wind_off_new_extrema_window_starts_s=newwindows,wind_on_new_extrema_window_starts_s=candidates(on),
        first_wind_off_new_extrema_window_start_s=min(newwindows) if newwindows else None,
        primary_new_extrema_window_starts_s=[t for t in newwindows if t<600],
        wind_off_new_late_extrema_candidates=new,wind_off_band_max_rms_m_s=off['residual_band_max_rms_m_s'],wind_on_band_max_rms_m_s=on['residual_band_max_rms_m_s'],
        assessment='INCONCLUSIVE',reason='Inspect temporal ordering, inherited structure, laminar-reference residual and advective work together. This single pair has no ensemble, resolution or dedicated wave-propagation qualification; no universal pass/fail is assigned.')


STYLE='body{font:17px/1.6 system-ui,sans-serif;color:#203238;background:#f5f8f7;margin:0}main{max-width:1120px;margin:auto;padding:35px 24px}h1{line-height:1.12;font-size:2.7rem}h2{margin-top:2rem}a{color:#146b70}.panel{background:white;padding:20px;border:1px solid #d7e2df;border-radius:12px;margin:20px 0}img{width:100%;height:auto}figure{margin:30px 0}figcaption,footer{font-size:.86rem;color:#526a6a}table{border-collapse:collapse;width:100%;font-size:.85rem}td,th{padding:8px;border-bottom:1px solid #d8e1df;text-align:left}nav{display:flex;gap:20px}.scroll{overflow:auto}'


def plot_all(out,results,curves):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    complete=sorted([r for r in results if r['status']=='complete'],key=lambda r:r['configuration']['wind_ustar']);n=len(complete)
    if not n:return []
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,n,figsize=(6*n,11),squeeze=False,constrained_layout=True)
    for col,r in enumerate(complete):
        d=curves[r['id']];z=d['z'];tau=d['elapsed'];label='Wind removed' if r['configuration']['wind_ustar']==0 else 'Wind sustained'
        for elapsed in (0,200,600,1200,3000):
            i=int(np.argmin(abs(tau-elapsed)));axes[0,col].plot(d['UE'][i]*1000,z,label='%g s'%tau[i]);axes[1,col].plot((d['UE'][i]-d['UE'][0])*1000,z,label='%g s'%tau[i]);axes[2,col].plot(d['residual'][i]*1000,z,label='%g s'%tau[i])
        axes[0,col].set(title=label+' — actual mean',xlabel='Eulerian mean current (mm/s)',ylabel='Depth (m)');axes[1,col].set(title='Change from the switch state',xlabel='Change in Eulerian mean current (mm/s)',ylabel='Depth (m)');axes[2,col].set(title='Mean minus laminar continuation',xlabel='Eddy-generated current residual (mm/s)',ylabel='Depth (m)')
        for ax in axes[:,col]:ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.savefig(out/'mean-profiles.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(2,n,figsize=(6*n,7),squeeze=False,constrained_layout=True)
    limit=max(float(np.max(abs(curves[r['id']]['UE']*1000))) for r in complete);rlimit=max(float(np.max(abs(curves[r['id']]['residual']*1000))) for r in complete);rlimit=max(rlimit,.01)
    for col,r in enumerate(complete):
        d=curves[r['id']];extent=[0,d['elapsed'][-1],d['z'][0],d['z'][-1]]
        im=axes[0,col].imshow(d['UE'].T*1000,origin='lower',extent=extent,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
        rm=axes[1,col].imshow(d['residual'].T*1000,origin='lower',extent=extent,aspect='auto',cmap='RdBu_r',vmin=-rlimit,vmax=rlimit)
        axes[0,col].set_title('Wind removed' if r['configuration']['wind_ustar']==0 else 'Wind sustained')
        for ax in axes[:,col]:ax.set(xlabel='Time after switch (s)',ylabel='Depth (m)')
    fig.colorbar(im,ax=axes[0,:],label='Actual mean UE (mm/s)');fig.colorbar(rm,ax=axes[1,:],label='Mean minus laminar reference (mm/s)');fig.savefig(out/'depth-time.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(3,n,figsize=(6*n,10),squeeze=False,constrained_layout=True)
    for col,r in enumerate(complete):
        h=curves[r['id']]['histories'];t=[q['elapsed_s'] for q in h]
        for key,label in (('eddy_rms_m_s','All eddies'),('eddy_2d_rms_m_s','y-independent eddies'),('roll_rms_m_s','x-independent rolls')):axes[0,col].plot(t,[q[key]*1000 for q in h],label=label)
        axes[0,col].set(title='Wind removed' if r['configuration']['wind_ustar']==0 else 'Wind sustained',ylabel='RMS speed (mm/s)')
        axes[1,col].plot(t,[100*q['chi_3d'] for q in h],label='3D eddy fraction');axes[1,col].plot(t,[100*q['roll_energy_fraction'] for q in h],label='Roll energy fraction');axes[1,col].set(ylabel='Fraction of eddy energy (%)')
        for region,label in (('interior','Interior'),('stokes','Active Stokes layer')):
            rows=[w for w in r['windows'] if w['region']==region and w['epsilon_C'] is not None and w['active']];axes[2,col].semilogy([w['start_elapsed_s']+100 for w in rows],[max(w['epsilon_C'],1e-12) for w in rows],'.-',label=label)
            quiet=[w for w in r['windows'] if w['region']==region and w['epsilon_C'] is not None and not w['active']]
            if quiet:axes[2,col].semilogy([w['start_elapsed_s']+100 for w in quiet],[max(w['epsilon_C'],1e-12) for w in quiet],'x',color='.6',label=label+' (quiet reference)')
        axes[2,col].axhline(.1,color='.5',ls='--');axes[2,col].set(ylabel='200 s accumulated correction / initial eddy Q')
        for ax in axes[:,col]:ax.set_xlabel('Time after switch (s)');ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.savefig(out/'motion-and-feedback.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(2,n,figsize=(6*n,7),squeeze=False,constrained_layout=True)
    for col,r in enumerate(complete):
        d=curves[r['id']];h=d['histories'];t=d['elapsed'];use=t<=600.+1e-7
        for key,label in (('eddy_rms_m_s','All eddies'),('eddy_2d_rms_m_s','y-independent eddies'),('roll_rms_m_s','x-independent rolls')):
            axes[0,col].plot(t[use],np.array([q[key] for q in h])[use]*1000,label=label)
        axes[0,col].set(title='Wind removed: first 600 s' if r['configuration']['wind_ustar']==0 else 'Wind sustained: first 600 s',ylabel='RMS speed (mm/s)')
        for m in (2,4,6):axes[1,col].plot(t[use],np.array([q['residual_mode_%d_m_s'%m] for q in h])[use]*1000,label='Mode %d'%m)
        axes[1,col].set(ylabel='Current residual mode (mm/s)')
        for ax in axes[:,col]:ax.set_xlabel('Time after switch (s)');ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.savefig(out/'primary-600s.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(2,n,figsize=(6*n,7),squeeze=False,constrained_layout=True)
    for col,r in enumerate(complete):
        d=curves[r['id']];h=d['histories'];t=d['elapsed']
        for m in (1,2,4,6):axes[0,col].plot(t,[row['residual_mode_%d_m_s'%m]*1000 for row in h],label='Mode %d%s'%(m,' (broad shear)' if m==1 else ''))
        axes[0,col].set(title='Wind removed' if r['configuration']['wind_ustar']==0 else 'Wind sustained',ylabel='Current residual mode (mm/s)')
        axes[1,col].plot(t,d['bandenergy']*1e6,label='Residual band energy m2–8')
        for key,label in (('adv_full','Integrated eddy-advection work'),('viscous_residual','Integrated viscous work')):axes[1,col].plot(t,np.sum(d['supplies'][key][:,1:],axis=1)*1e6,label=label)
        axes[1,col].set(ylabel='Energy / integrated work (mm²/s²)')
        for ax in axes[:,col]:ax.set_xlabel('Time after switch (s)');ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.savefig(out/'band-budget.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(n,3,figsize=(15,3.2*n),squeeze=False,constrained_layout=True)
    vmax=max(float(np.max(abs(d['maps']['Uc']-d['maps']['Uc'].mean(axis=1,keepdims=True)))) for d in curves.values())*1000;vmax=max(vmax,.01)
    for row,r in enumerate(complete):
        d=curves[r['id']];maps=d['maps'];label='Wind removed' if r['configuration']['wind_ustar']==0 else 'Wind sustained'
        mid=int(np.argmin(abs(maps['time']-maps['time'][0]-600)))
        for col,idx in enumerate((0,mid,-1)):
            a=maps['Uc'][idx];a=a-a.mean(axis=0,keepdims=True);im=axes[row,col].imshow(a.T*1000,origin='lower',extent=[0,r['configuration']['lx'],-r['configuration']['h'],0],aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax);axes[row,col].set(title=label+' — '+('switch state' if col==0 else '%g s after switch'%(maps['time'][idx]-maps['time'][0])),xlabel='x (m)',ylabel='z (m)')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='y-mean x-varying u (mm/s)');fig.savefig(out/'mean-eddies.png',dpi=145);plt.close(fig)
    files=['mean-profiles.png','depth-time.png','primary-600s.png','motion-and-feedback.png','band-budget.png','mean-eddies.png']
    if n==2:
        off,on=[curves[r['id']] for r in complete];fig,axes=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
        for ax,key,label in zip(axes,('UE','residual'),('Actual current: wind off minus wind on','Laminar-subtracted current: wind off minus wind on')):
            difference=(off[key]-on[key])*1000;limit=max(float(np.max(abs(difference))),.01)
            im=ax.imshow(difference.T,origin='lower',extent=[0,off['elapsed'][-1],off['z'][0],off['z'][-1]],aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit);ax.set(title=label,xlabel='Time after switch (s)',ylabel='Depth (m)');fig.colorbar(im,ax=ax,label='Difference (mm/s)')
        fig.savefig(out/'matched-difference.png',dpi=145);plt.close(fig);files.insert(2,'matched-difference.png')
    return files


def plot_parent_history(root,out):
    source=root/'reference/parent-history.csv'
    if not source.exists():return None
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    rows=numeric_csv(source);time=np.array([r['time_s'] for r in rows]);keep=time<=1000.+1e-7;time=time[keep]-1000;rows=[r for r,k in zip(rows,keep) if k]
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
    for key,label in (('eddy_rms_m_s','All eddies'),('roll_rms_m_s','x-independent rolls'),('ky0_eddy_rms_m_s','y-independent eddies')):axes[0].plot(time,[r[key]*1000 for r in rows],label=label)
    axes[0].set(ylabel='RMS speed (mm/s)',title='Parent develops roll-dominated motion')
    for key,label in (('chi_3d','3D eddy fraction'),('roll_energy_fraction','Roll energy fraction')):axes[1].plot(time,[100*r[key] for r in rows],label=label)
    axes[1].set(ylabel='Eddy-energy fraction (%)',title='Selected state at the switch')
    for ax in axes:ax.axvline(0,color='.4',ls='--');ax.set_xlabel('Time relative to switch (s)');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.savefig(out/'parent-history.png',dpi=145);plt.close(fig);return 'parent-history.png'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--approved-runs',type=Path);args=p.parse_args()
    root=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);campaign=json.loads((root/'campaign.json').read_text());source=json.loads((root/'source-manifest.json').read_text());approved=set(json.loads(args.approved_runs.read_text())) if args.approved_runs else None
    results=[];curves={};verified={}
    for entry in campaign['runs']:
        state,receipt=run_state(root,entry,source['version'],approved)
        if state!='complete':results.append(dict(id=entry['id'],arm=entry.get('arm'),status=state));continue
        result,data,hashes=assess(root,entry,receipt);results.append(result);curves[entry['id']]=data;verified[entry['id']]=hashes
        folder=out/'runs'/identifier(entry['id']);folder.mkdir(parents=True,exist_ok=True);write_json(folder/'summary.json',result)
        write_csv(folder/'time-series.csv',data['histories']);write_csv(folder/'correction-windows.csv',result['windows'])
        rows=[]
        for i,t in enumerate(data['elapsed']):
            if min(abs(t-x) for x in np.arange(0,t+40,40))>1e-7 and i!=len(data['elapsed'])-1:continue
            for j,z in enumerate(data['z']):rows.append(dict(elapsed_s=float(t),z_m=float(z),UE_m_s=float(data['UE'][i,j]),laminar_UE_m_s=float(data['baseline'][i,j]),eddy_residual_m_s=float(data['residual'][i,j]),adv_full_m_s2=float(data['profiles']['adv_full'][i,j]),adv_3d_m_s2=float(data['profiles']['adv_3d'][i,j]),flux_total_m2_s2=float(data['profiles']['flux_total'][i,j])))
        write_csv(folder/'profiles.csv',rows)
    comparison=compare_pair(results,curves);figures=plot_all(out,results,curves);parent_plot=plot_parent_history(root,out)
    if parent_plot:figures.insert(0,parent_plot)
    counts={s:sum(r['status']==s for r in results) for s in ('complete','running','queued','waiting','failed')}
    summary=dict(title='From forced rolls to decaying current organization',source_version=source['version'],counts=counts,criteria=CRITERIA,runs=results,comparison=comparison)
    write_json(out/'summary.json',summary);write_json(out/'provenance.json',dict(source_version=source['version'],analysis_sha256=digest(Path(__file__)),verified_inputs=verified,public_profile_interval_s=40,science_profile_intervals_s={r['id']:float(np.median(np.diff(curves[r['id']]['time']))) for r in results if r['status']=='complete'},native_snapshots_published=False));(out/'style.css').write_text(STYLE)
    captions={'parent-history.png':'Actual parent history before the selected switch. All branches inherit the same native velocity field; post-switch results are separate.', 'matched-difference.png':'The direct wind-off minus sustained-wind contrast. Actual difference includes changed stress and viscous response; subtracting each branch’s laminar continuation isolates the remaining mean-current contrast.', 'primary-600s.png':'Primary first 600 s interval: absolute turbulence and interior-band residual modes. Later evolution is also retained, but viscous decay of narrow bands can obscure an earlier transient.', 'mean-profiles.png':'Actual Eulerian mean profiles, their change from the switch, and departure from the same-grid laminar continuation. Elapsed times are measured from the common restart.','depth-time.png':'Shared color limits within each row. The laminar reference retains the initial current, constant viscosity, and the branch’s surface stress and uniform return force. It contains no turbulent advection.','motion-and-feedback.png':'Absolute motion accompanies energy fractions: a rising 2D fraction can reflect faster decay of 3D motion. Correction ratios use fixed nonoverlapping 200 s windows, normalized by their starting x-varying mean vorticity. Gray crosses flag quiet or unresolved references. Weak source alone is not a verified autonomous forecast.','band-budget.png':'The current residual excludes the laminar background. Mode 1 is broad shear; modes 2–8 carry interior structure. Advective work and viscous damping are integrated from the switch; sign and temporal ordering matter.','mean-eddies.png':'Switch, 600 s and final transverse-mean x-varying currents, on one color scale. Horizontally uniform current has been removed from these panels.'}
    budget_note='; '.join(r['id']+': full mean budget quadrature '+format(100*r['mean_momentum_budget_relative_error'],'.3g')+'%, maximum low-mode residual budget '+format(100*max(r['mode_budget_relative_errors']),'.3g')+'%' for r in results if r['status']=='complete')
    captions['band-budget.png']+=' Sampled-tendency closure: '+budget_note+'. Tendency integration uses the actual saved cadence (1 s in production); initial wall-adjustment quadrature is checked, and an unclosed budget is not evidence of physical production.'
    sections=''.join('<figure><img src="'+name+'" alt="'+html.escape(captions[name])+'"><figcaption>'+html.escape(captions[name])+'</figcaption></figure>' for name in figures)
    rows=''.join('<tr><td>'+html.escape(r['id'])+'</td><td>'+r['status']+'</td><td>'+('<a href="runs/'+r['id']+'/summary.json">Statistics</a> · <a href="runs/'+r['id']+'/time-series.csv">Time series</a> · <a href="runs/'+r['id']+'/profiles.csv">Profiles</a>' if r['status']=='complete' else 'Awaiting verified completion')+'</td></tr>' for r in results)
    conclusion='Both branches are required before interpreting the contrast.' if comparison['status']=='AWAITING_MATCHED_PAIR' else ('Both branches are complete and their shared restart is verified. '+str(len(comparison['weak_feedback_window_starts_s']))+' decay windows have small accumulated feedback in both prescribed depth regions. '+str(len(comparison['wind_off_new_extrema_window_starts_s']))+' elapsed windows contain persistent extrema that match neither initial nor laminar-reference extrema under the stated positional rule; '+str(len(comparison['primary_new_extrema_window_starts_s']))+' occur during the primary first 600 s. Late persistence is reported separately. These are unmatched candidates; positional matching cannot rule out drifting inherited features. They are not a mechanism verdict.')
    body='<nav><a href="../../">Gallery</a><a href="../wind-jet-tests/">Wind-maintained tests</a><a href="#assessment">Assessment</a></nav><h1>Can current bands emerge after the wind stops?</h1><p>Two three-dimensional branches restart from exactly the same wave-affected turbulent state. One removes the surface stress and its compensating uniform force; the other sustains them. The Stokes drift remains unchanged. This tests the proposed sequence of forced roll production followed by organization during decay.</p><div class="panel"><strong>Selection:</strong> the weak-initial-3D, u*=2 mm/s parent at 1000 s contains substantial along-wave rolls and finite x–z eddies. It was selected before running these branches. Inherited structure, viscous relaxation and late loss of turbulent activity are shown explicitly.</div><div class="scroll"><table><thead><tr><th>Branch</th><th>Status</th><th>Verified outputs</th></tr></thead><tbody>'+rows+'</tbody></table></div>'+sections+'<section class="panel" id="assessment"><h2>What can we conclude?</h2><p>'+conclusion+'</p><p>A new profile extremum alone is insufficient: inspect growth beyond the laminar continuation, the turbulent momentum work, available eddy energy, and whether 3D feedback weakens before organization. The matched control separates the consequence of wind removal from continued evolution. One realization cannot establish a universal regime boundary, and no binary claim is assigned automatically.</p><p><a href="summary.json">Full assessment</a> · <a href="PROTOCOL.txt">Frozen protocol</a> · <a href="campaign.json">Branch configuration</a> · <a href="provenance.json">Provenance</a></p></section><footer>Results are shown only after diagnostic verification and persistent backup. Native field files remain private.</footer>'
    (out/'index.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Wind removal and current organization</title><link rel="stylesheet" href="style.css"></head><body><main>'+body+'</main></body></html>\n')
    for name in ('PROTOCOL.txt','campaign.json','source-manifest.json'):
        if (root/name).exists():shutil.copy2(root/name,out/name)
    if (root/'source_provenance.toml').exists():write_json(out/'source-provenance.json',tomllib.loads((root/'source_provenance.toml').read_text()))
    for name in ('parent-history.csv','parent-selection.json'):
        src=root/'reference'/name
        if src.exists():
            (out/'reference').mkdir(exist_ok=True);shutil.copy2(src,out/'reference'/name)
    size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if size>20*1024**2:raise ValueError('Decay page exceeds 20 MiB')
    print(json.dumps(dict(counts=counts,public_bytes=size,comparison=comparison['status'])))


if __name__=='__main__':main()
