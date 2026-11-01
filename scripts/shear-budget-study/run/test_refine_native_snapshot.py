import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import numpy as np
from refine_native_snapshot import periodic_interpolate,vertical_interpolate,native_projection,divergence,refine,centered,write_toml,toml_scalar,digest,PARENT_SOURCES,PARENT_VERSION


class RefinementTests(unittest.TestCase):
    def test_numpy_scalars_and_inline_approval_roundtrip(self):
        obj=dict(number=np.float64(.023),small=np.float32(.125),count=np.int64(42),enabled=np.bool_(True),
            allowed_branches=[dict(kind='full',ustar=np.float64(.0011)),dict(kind='mean',ustar=0.)],
            nested=[dict(values=[np.int32(3),dict(value=np.float64(.25))])])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'approval.toml';write_toml(path,obj)
            self.assertEqual(tomllib.loads(path.read_text()),obj)
            self.assertNotIn('np.',path.read_text())
        for bad in (float('nan'),float('inf'),float('-inf'),np.float64('nan'),np.float32('inf')):
            with self.assertRaises(ValueError):toml_scalar(bad)

    def test_full_native_sidecar_and_approval_roundtrip(self):
        # Exercise the actual CLI and every nested sidecar table, not merely
        # the interpolation helper. The historical label is synthetic here.
        nx,ny,nz=16,8,12;lx,ly,h=2.,1.,.5
        x=np.arange(nx)*lx/nx;zf=np.arange(nz+1)*h/nz
        psi=.0005*np.sin(2*np.pi*x/lx)[:,None,None]*np.sin(np.pi*zf/h)[None,None,:]
        psi=np.repeat(psi,ny,axis=1)
        u=-np.diff(psi,axis=2)/(h/nz);w=(np.roll(psi,-1,axis=0)-psi)/(lx/nx)
        v=np.repeat(.0002*np.cos(2*np.pi*(x+.5*lx/nx)/lx)[:,None,None]*np.cos(np.pi*(np.arange(nz)+.5)/nz)[None,None,:],ny,axis=1)
        p=dict(nx=nx,ny=ny,nz=nz,h=h,lx=lx,ly=ly,k=6.1,waves=1,alpha=.75,Sref=.036649447248308646,
            uoffset=0.,nu=1e-5,ustar=0.,maintain=True,advection_name='Centered2')
        z=-h+(np.arange(nz)+.5)*h/nz;u+=.25*p['Sref']*np.exp(2*p['k']*z)[None,None,:]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);inventory={}
            for name,a in zip(('u','v','w'),(u,v,w)):
                path=root/(name+'.bin');a.astype('<f8').ravel(order='F').tofile(path)
                inventory[name]=dict(file=path.name,shape=list(a.shape),dtype='float64',order='F',
                    location=['Face','Center','Center'] if name=='u' else ['Center','Face','Center'] if name=='v' else ['Center','Center','Face'],sha256=digest(path))
            source=root/'native.json';source.write_text(json.dumps(dict(time=600.,iteration=123,arrays=inventory)))
            config=root/'configuration.json';config.write_text(json.dumps(dict(parameters=p)))
            metadata=dict(format='wind-native-snapshot-v1',byte_order='little',translation='legacy-native-json-plus-original-configuration-v1',
                json_sha256=digest(source),original_configuration_sha256=digest(config),time=600.,iteration=123,
                parameters=p,source_hashes=PARENT_SOURCES,arrays=inventory)
            sidecar=root/'native.toml';write_toml(sidecar,metadata)
            output=root/'fine'
            command=[sys.executable,str(Path(__file__).with_name('refine_native_snapshot.py')),'--snapshot',str(source),
                '--metadata',str(sidecar),'--parent-configuration',str(config),'--output',str(output),'--target','32','16','24']
            result=subprocess.run(command,check=True,capture_output=True,text=True)
            receipt=json.loads(result.stdout);fine_path=Path(receipt['metadata']);fine=tomllib.loads(fine_path.read_text())
            self.assertEqual(fine['parameters']['nx'],32)
            self.assertEqual(fine['derived_from']['source_arrays'],inventory)
            self.assertEqual(fine['derived_from']['diagnostics'],receipt['diagnostics'])
            self.assertEqual(fine['json_sha256'],digest(receipt['snapshot']))
            self.assertEqual(json.loads((output/'refinement-report.json').read_text()),fine['derived_from'])
            self.assertEqual(digest(output/'parent-configuration.json'),digest(config))
            self.assertNotIn('np.float',fine_path.read_text())
            record=dict(snapshot_json_sha256=fine['json_sha256'],snapshot_metadata_sha256=digest(fine_path),snapshot_time=600.,
                parent_configuration_sha256=digest(config),metadata_translation=fine['translation'],source_hashes=fine['source_hashes'],
                parameters=fine['parameters'],arrays=fine['arrays'],derived_from=fine['derived_from'],expected_parent_ustar=0.,expected_maintain_background=True,
                allowed_branches=[dict(kind='full',ustar=np.float64(.00125)),dict(kind='mean',ustar=0.)])
            approval=dict(format='shear-budget-parent-approval-v1',campaign='unity-shear-budget-study',
                approved_parent_source_version=PARENT_VERSION,sources={'fine-parent':record})
            approved_path=root/'parent-approval.toml';write_toml(approved_path,approval)
            self.assertEqual(tomllib.loads(approved_path.read_text()),approval)
            # The CLI refuses a duplicate derivation, preserving the first.
            second=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(second.returncode,0)
            self.assertEqual(digest(fine_path),receipt['metadata_sha256'])

    def test_periodic_native_staggering(self):
        for offset in (0.,.5):
            for mode in (1,3,8):
                n=16;m=32;x=(np.arange(n)+offset)/n;xx=(np.arange(m)+offset)/m
                phase=0. if mode!=n//2 else -np.pi*offset
                a=np.cos(2*np.pi*mode*x+phase)[:,None,None]
                b=periodic_interpolate(a,m,0,offset)[:,0,0]
                np.testing.assert_allclose(b,np.cos(2*np.pi*mode*xx+phase),atol=2e-14)
    def test_vertical_staggering(self):
        n=12;m=24
        a=np.cos(3*np.pi*(np.arange(n)+.5)/n)[None,None,:]
        b=vertical_interpolate(a,m,'center')[0,0,:]
        np.testing.assert_allclose(b,np.cos(3*np.pi*(np.arange(m)+.5)/m),atol=2e-14)
        a=np.sin(3*np.pi*np.arange(n+1)/n)[None,None,:]
        b=vertical_interpolate(a,m,'face')[0,0,:]
        np.testing.assert_allclose(b,np.sin(3*np.pi*np.arange(m+1)/m),atol=2e-14)
    def test_native_projection(self):
        rng=np.random.default_rng(92)
        for ny in (1,8):
            a=(rng.normal(size=(16,ny,12)),rng.normal(size=(16,ny,12)),rng.normal(size=(16,ny,13)))
            a[2][:,:,0]=0.;a[2][:,:,-1]=0.
            b=native_projection(a,2.,1.,.5)
            self.assertLess(np.max(abs(divergence(b,2.,1.,.5))),3e-13)
            np.testing.assert_allclose(b[0].mean(axis=(0,1)),a[0].mean(axis=(0,1)),atol=1e-15)
            np.testing.assert_allclose(b[1].mean(axis=(0,1)),a[1].mean(axis=(0,1)),atol=1e-15)
            self.assertLess(sum(np.sum(x*x) for x in b),sum(np.sum(x*x) for x in a))
    def test_solenoidal_prolongation_and_mean(self):
        nx,ny,nz=16,8,12;lx,ly,h=2.,1.,.5
        x=np.arange(nx)*lx/nx;zf=np.arange(nz+1)*h/nz
        psi=.0005*np.sin(2*np.pi*x/lx)[:,None,None]*np.sin(np.pi*zf/h)[None,None,:]
        psi=np.repeat(psi,ny,axis=1)
        u=-np.diff(psi,axis=2)/(h/nz);w=(np.roll(psi,-1,axis=0)-psi)/(lx/nx)
        v=.0002*np.cos(2*np.pi*(x+.5*lx/nx)/lx)[:,None,None]*np.cos(np.pi*(np.arange(nz)+.5)/nz)[None,None,:]
        v=np.repeat(v,ny,axis=1)
        p=dict(h=h,lx=lx,ly=ly,k=6.1,waves=1.,alpha=.75,Sref=.036649447248308646,uoffset=0.)
        z=-h+(np.arange(nz)+.5)*h/nz;u+=.25*p['Sref']*np.exp(2*p['k']*z)[None,None,:]
        a=(u,v,w);before=[x.copy() for x in a]
        result,metrics=refine(a,p,(32,16,24))
        self.assertLess(metrics['scaled_divergence_after'],1e-13)
        self.assertLess(metrics['projection_fraction_eddy'],.01)
        self.assertGreater(np.max(abs(result[1])),1e-4)
        self.assertLess(max(abs(x) for x in metrics['bulk_momentum_error_m_s'].values()),1e-16)
        for x,y in zip(before,a):np.testing.assert_array_equal(x,y)


if __name__=='__main__':unittest.main()
