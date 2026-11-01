#!/usr/bin/env python3
"""Refine one hash-verified native staggered state without inventing fine eddies.

Horizontal fields are trigonometric interpolants at each component's native
face/center positions. In z, u minus its prescribed background and v use even
extensions; impermeable w uses an odd extension. A target-grid native discrete
Helmholtz projection restores solenoidality. The derived state is a resolution
test of this specified coarse-state interpolant, not an independently evolved
high-resolution realization or a legacy-identical restart.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tomllib
import numpy as np

METHOD='periodic-spectral-even-odd-native-projection-v1'
PARENT_VERSION='da81ab96d7e51fb0'
PARENT_SOURCES={
    'initial_conditions.jl':'3a8f83da2f569bd311b9765baef9b8cd32ce70871da929bf7a38496882e526fc',
    'jet_diagnostics.jl':'136071415aee9b9b1d187e9f06aad7325dde09ad39ee8e92a9a18733364be05e',
    'regime_diagnostics.jl':'c3ea43fd36dab77051b2ba48a990d465f09ee0cbfce1ce5f0fe45e49541be3b7',
    'regime_physics.jl':'cd097ff947b3d4d5a36729f1f93325f13e2fd192063a0839a57f142ce831202f',
    'run_jet.jl':'f963abf0de7bf3175bec357c639c5d0e5523f74a9bbedc92292cbb38ee580722',
    'vortex_modes.jl':'020d0f191d8f39ca71ad5b8d0445a2c9f057be83aee292c520b89c382781ffa4'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def periodic_interpolate(a,m,axis,offset):
    """Preserve the real trigonometric interpolant, splitting old Nyquist."""
    a=np.asarray(a,dtype=np.float64);n=a.shape[axis]
    if m<n or m%n:raise ValueError('Only integer refinements are supported')
    if m==n:return a.copy()
    f=np.fft.fft(a,axis=axis);shape=list(a.shape);shape[axis]=m
    g=np.zeros(shape,dtype=np.complex128)
    source=[slice(None)]*a.ndim;target=list(source)
    for index in range(n):
        source[axis]=index
        if n%2==0 and index==n//2:
            for mode in (n//2,-n//2):
                target[axis]=mode%m
                phase=np.exp(2j*np.pi*mode*(offset/m-offset/n))
                g[tuple(target)]+=f[tuple(source)]*.5*phase
        else:
            mode=index if index<=(n-1)//2 else index-n
            target[axis]=mode%m
            g[tuple(target)]=f[tuple(source)]*np.exp(2j*np.pi*mode*(offset/m-offset/n))
    b=np.fft.ifft(g,axis=axis)*(m/n)
    if np.max(np.abs(b.imag))>2e-12*max(np.max(np.abs(b.real)),1e-12):
        raise RuntimeError('Interpolation lost real conjugacy')
    return b.real


def vertical_interpolate(a,m,stagger):
    if stagger=='center':
        n=a.shape[2];extended=np.concatenate((a,a[:,:,::-1]),axis=2)
        return periodic_interpolate(extended,2*m,2,.5)[:,:,:m]
    if stagger!='face':raise ValueError('Unknown z staggering')
    n=a.shape[2]-1
    if max(np.max(abs(a[:,:,0])),np.max(abs(a[:,:,-1])))>1e-13:raise ValueError('Parent w is not impermeable')
    extended=np.concatenate((a[:,:,:-1],-a[:,:,1:][:,:,::-1]),axis=2)
    result=periodic_interpolate(extended,2*m,2,0.)[:,:,:m+1]
    result[:,:,0]=0.;result[:,:,-1]=0.
    return result


def divergence(arrays,lx,ly,h):
    u,v,w=arrays;nx,ny,nz=u.shape
    return (np.roll(u,-1,axis=0)-u)/(lx/nx)+(np.roll(v,-1,axis=1)-v)/(ly/ny)+np.diff(w,axis=2)/(h/nz)


def centered(arrays):
    u,v,w=arrays
    return ((u+np.roll(u,-1,axis=0))/2,(v+np.roll(v,-1,axis=1))/2,(w[:,:,:-1]+w[:,:,1:])/2)


def eddy_energy(arrays):
    return .5*sum(float(np.mean((a-a.mean(axis=(0,1),keepdims=True))**2)) for a in centered(arrays))


def native_projection(arrays,lx,ly,h):
    u,v,w=(a.copy() for a in arrays);nx,ny,nz=u.shape;dx,dy,dz=lx/nx,ly/ny,h/nz
    d=divergence((u,v,w),lx,ly,h)
    extension=np.concatenate((d,d[:,:,::-1]),axis=2)
    rhs=np.fft.fftn(extension)
    lam=(-4*np.sin(np.pi*np.fft.fftfreq(nx))[:,None,None]**2/dx**2
         -4*np.sin(np.pi*np.fft.fftfreq(ny))[None,:,None]**2/dy**2
         -4*np.sin(np.pi*np.fft.fftfreq(2*nz))[None,None,:]**2/dz**2)
    lam[0,0,0]=1.;rhs/=lam;rhs[0,0,0]=0.
    pressure=np.fft.ifftn(rhs).real[:,:,:nz]
    u-=(pressure-np.roll(pressure,1,axis=0))/dx
    v-=(pressure-np.roll(pressure,1,axis=1))/dy
    w[:,:,1:-1]-=np.diff(pressure,axis=2)/dz
    w[:,:,0]=0.;w[:,:,-1]=0.
    return u,v,w


def refine(arrays,p,target):
    nx,ny,nz=(int(x) for x in target);old=arrays[0].shape
    if any(n<o or n%o for n,o in zip(target,old)):raise ValueError('Target must be an integer refinement of each dimension')
    if not any(n>o for n,o in zip(target,old)):raise ValueError('Refinement must increase at least one dimension')
    h=float(p['h']);lx=float(p['lx']);ly=float(p['ly']);k=float(p['k'])
    b=(float(p['waves'])-float(p['alpha']))*float(p['Sref'])
    oldz=-h+(np.arange(old[2])+.5)*h/old[2];newz=-h+(np.arange(nz)+.5)*h/nz
    oldb=b*np.exp(2*k*oldz)+float(p['uoffset']);newb=b*np.exp(2*k*newz)+float(p['uoffset'])
    refined=[]
    for name,field in zip(('u','v','w'),arrays):
        a=field-oldb[None,None,:] if name=='u' else field.copy()
        a=periodic_interpolate(a,nx,0,0. if name=='u' else .5)
        a=periodic_interpolate(a,ny,1,0. if name=='v' else .5)
        a=vertical_interpolate(a,nz,'face' if name=='w' else 'center')
        if name=='u':a+=newb[None,None,:]
        refined.append(a)
    # Keep the parent's depth-integrated horizontal momentum exactly. The tiny
    # constant correction accounts for the changed sampling of exp(2kz).
    shifts={}
    for name,index in (('u',0),('v',1)):
        shift=float(np.mean(arrays[index])-np.mean(refined[index]));refined[index]+=shift;shifts[name]=shift
    before=tuple(refined);result=native_projection(before,lx,ly,h)
    rms=lambda a:float(np.sqrt(sum(np.mean(v*v) for v in a)))
    adjustment=rms(tuple(a-b for a,b in zip(result,before)))
    eddy_rms=float(np.sqrt(2*eddy_energy(before)));speed=rms(result)
    d_before=divergence(before,lx,ly,h);d_after=divergence(result,lx,ly,h)
    metrics=dict(scaled_divergence_before=float(np.max(abs(d_before))*min(lx/nx,ly/ny,h/nz)/max(speed,1e-30)),
        scaled_divergence_after=float(np.max(abs(d_after))*min(lx/nx,ly/ny,h/nz)/max(speed,1e-30)),
        projection_rms_m_s=adjustment,projection_fraction_total=adjustment/max(speed,1e-30),
        projection_fraction_eddy=adjustment/max(eddy_rms,1e-30),
        energy_parent=eddy_energy(arrays),energy_interpolated=eddy_energy(before),energy_projected=eddy_energy(result),
        bulk_momentum_shift_m_s=shifts,
        horizontal_mean_projection_max_m_s={name:float(np.max(abs(a.mean(axis=(0,1))-b.mean(axis=(0,1))))) for name,a,b in zip(('u','v','w'),result,before)},
        bulk_momentum_error_m_s={name:float(np.mean(result[i])-np.mean(arrays[i])) for i,name in enumerate(('u','v'))})
    if metrics['scaled_divergence_after']>1e-11:raise RuntimeError('Refined state is not natively solenoidal')
    if metrics['projection_fraction_eddy']>.10:raise RuntimeError('Native projection changes more than10% of the interpolated eddy RMS')
    return result,metrics


def toml_scalar(value):
    # NumPy 2 scalar repr is executable-looking text, e.g. np.float64(...),
    # rather than a TOML numeric literal. Serialize finite native values only.
    if isinstance(value,(bool,np.bool_)):return 'true' if bool(value) else 'false'
    if isinstance(value,(int,np.integer)):return repr(int(value))
    if isinstance(value,(float,np.floating)):
        number=float(value)
        if not math.isfinite(number):raise ValueError('Nonfinite value in native-state provenance')
        return repr(number)
    if isinstance(value,str):return json.dumps(value)
    if isinstance(value,(list,tuple)):return '['+', '.join(toml_scalar(v) for v in value)+']'
    # Approval records contain arrays of branch dictionaries. An inline table
    # keeps each branch self-contained and supports nested lists/dictionaries.
    if isinstance(value,dict):return '{ '+', '.join(json.dumps(k)+' = '+toml_scalar(v) for k,v in value.items())+' }'
    raise TypeError(type(value))


def write_toml(path,obj):
    lines=[]
    def section(value,keys=()):
        if keys:lines.append('['+'.'.join(json.dumps(k) for k in keys)+']')
        for k,v in value.items():
            if not isinstance(v,dict):lines.append(json.dumps(k)+' = '+toml_scalar(v))
        lines.append('')
        for k,v in value.items():
            if isinstance(v,dict):section(v,keys+(k,))
    section(obj);Path(path).write_text('\n'.join(lines))


def main():
    a=argparse.ArgumentParser();a.add_argument('--snapshot',type=Path,required=True);a.add_argument('--metadata',type=Path,required=True)
    a.add_argument('--parent-configuration',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--target',type=int,nargs=3,required=True);args=a.parse_args()
    if args.output.exists():raise RuntimeError('Refusing an existing refinement output')
    source=json.loads(args.snapshot.read_text());metadata=tomllib.loads(args.metadata.read_text())
    if metadata['translation']!='legacy-native-json-plus-original-configuration-v1':raise ValueError('Only the original approved native parent may be refined')
    if metadata['format']!='wind-native-snapshot-v1' or metadata['byte_order']!='little' or metadata['source_hashes']!=PARENT_SOURCES:
        raise ValueError('Historical source or native format is not approved')
    if digest(args.snapshot)!=metadata['json_sha256'] or digest(args.parent_configuration)!=metadata['original_configuration_sha256']:
        raise ValueError('Parent provenance mismatch')
    if source['arrays']!=metadata['arrays']:raise ValueError('Parent inventory mismatch')
    p=metadata['parameters'];arrays=[]
    for name in ('u','v','w'):
        entry=source['arrays'][name];file=entry['file']
        if Path(file).name!=file:raise ValueError('Array filename must be a basename')
        path=args.snapshot.parent/file
        if digest(path)!=entry['sha256'] or entry['dtype']!='float64' or entry['order']!='F':raise ValueError('Parent array fingerprint/format mismatch')
        arr=np.fromfile(path,dtype='<f8').reshape(entry['shape'],order='F')
        expected=(int(p['nx']),int(p['ny']),int(p['nz'])+(name=='w'))
        if arr.shape!=expected:raise ValueError('Native shape disagrees with parent parameters')
        if not np.isfinite(arr).all():raise ValueError('Nonfinite parent')
        arrays.append(arr)
    refined,metrics=refine(tuple(arrays),p,tuple(args.target));args.output.mkdir(parents=True)
    inventory={};stem=f"t{float(source['time']):012.6f}"
    for name,arr in zip(('u','v','w'),refined):
        file=stem+'_'+name+'.bin';target=args.output/file;arr.astype('<f8').ravel(order='F').tofile(target)
        inventory[name]=dict(file=file,shape=list(arr.shape),dtype='float64',order='F',location=source['arrays'][name]['location'],sha256=digest(target))
    manifest=dict(time=source['time'],iteration=source['iteration'],arrays=inventory)
    snapshot=args.output/(stem+'.json');snapshot.write_text(json.dumps(manifest,indent=2)+'\n')
    code=digest(__file__);new_parameters=dict(p);new_parameters.update(nx=args.target[0],ny=args.target[1],nz=args.target[2])
    derivation=dict(method=METHOD,source_version=PARENT_VERSION,source_hashes=metadata['source_hashes'],
        source_snapshot_sha256=digest(args.snapshot),source_metadata_sha256=digest(args.metadata),source_arrays=metadata['arrays'],
        source_parameters=p,target_grid=args.target,resampler_sha256=code,diagnostics=metrics)
    new=dict(format='wind-native-snapshot-v1',json_sha256=digest(snapshot),time=source['time'],iteration=source['iteration'],byte_order='little',
        parameters=new_parameters,source_hashes={'refine_native_snapshot.py':code},arrays=inventory,
        translation='native-parent-refinement-v1',original_configuration_sha256=metadata['original_configuration_sha256'],derived_from=derivation)
    sidecar=args.output/(stem+'.toml');write_toml(sidecar,new)
    if tomllib.loads(sidecar.read_text())!=new:raise RuntimeError('Refinement sidecar did not round-trip')
    shutil.copy2(args.parent_configuration,args.output/'parent-configuration.json')
    (args.output/'refinement-report.json').write_text(json.dumps(derivation,indent=2)+'\n')
    print(json.dumps(dict(snapshot=str(snapshot),metadata=str(sidecar),snapshot_sha256=digest(snapshot),metadata_sha256=digest(sidecar),diagnostics=metrics),indent=2))


if __name__=='__main__':main()
