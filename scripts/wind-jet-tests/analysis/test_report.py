import hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
from analyze import (write_json,write_csv,digest,campaign_status,analyze_run,matched_pairs,
    safe_path,cache_valid,cache_key,CENTER_FIELDS,NATIVE_FIELDS,render)

class ReportTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def fixture(self,name,eta):
        entry={'id':name,'path':'data/runs/'+name,'cell':'cell','seed':1};folder=self.root/entry['path'];folder.mkdir(parents=True)
        config=dict(nx=8,ny=8,nz=8,lx=1.,ly=1.,h=.1,k=6.1,S=.03,S_reference=.03,alpha=.75,uoffset=0.,eddy_rms=.004,injection_wavelength=.25,perturb_energy_ratio=eta,seed=1,stop_time=10.,max_dt=.01,nu=1e-5,base_ic_sha256='base',perturbation_ic_sha256='perturb',initial_condition_sha256=name)
        write_json(folder/'configuration.json',config);write_json(folder/'completed.json',{'time':10.,'initial_condition_sha256':name})
        zf=np.linspace(-.1,0,9);z=(zf[:-1]+zf[1:])/2;x=(np.arange(8)+.5)/8
        write_json(folder/'grid.json',dict(z_faces=zf,z_centers=z,x_centers=x))
        budget=[{'time_s':t,'Etotal':1+eta,'Eky0':1.,'Eeddy':1.,'Eperp':eta,'Ejet_anomaly':0.} for t in np.arange(11.)]
        write_csv(folder/'regime-budget.csv',budget);write_csv(folder/'jet-budget.csv',budget)
        for t in (0.,5.,10.):
            arrays={};stem=f'mean_{t:012.6f}'
            for field in (*CENTER_FIELDS,*NATIVE_FIELDS):
                shape=(8,9) if field=='W' or field.startswith('W_') else (8,8)
                a=np.zeros(shape)
                if field in ('U','Uc'):a+=np.sin(2*np.pi*x)[:,None]*.001
                if field=='Q':a+=z[None,:]
                file=stem+'_'+field+'.bin';a.flatten(order='F').tofile(folder/file)
                arrays[field]={'file':file,'shape':list(shape),'dtype':'float64','order':'F','sha256':digest(folder/file)}
            write_json(folder/(stem+'.json'),{'time':t,'arrays':arrays,'recommended_edge_exclusion':2})
        receipt={'id':name,'completed':True,'source_version':'test-version','files':{p.name:digest(p) for p in folder.iterdir()}}
        status=self.root/'data/status'/name;write_json(status/'completed.json',receipt)
        return entry,receipt
    def test_receipt_required_and_safe_paths(self):
        e,r=self.fixture('tiny',1e-6);self.assertEqual(campaign_status(self.root,e,'test-version')[0],'complete')
        with self.assertRaises(ValueError):campaign_status(self.root,e,'wrong-version')
        (self.root/'data/status/tiny/completed.json').unlink()
        self.assertEqual(campaign_status(self.root,e,'test-version')[0],'waiting')
        with self.assertRaises(ValueError):safe_path(self.root,'../outside')
    def test_map_analysis_cache_and_matched_fields(self):
        entries=[];results=[];cache=self.root/'data/analysis-cache'
        for name,eta in [('base',0.),('tiny',1e-6),('tinier',1e-8)]:
            e,r=self.fixture(name,eta);entries.append(e)
            summary=analyze_run(self.root,e,r,cache/name);results.append(summary)
            self.assertEqual(summary['status'],'complete');self.assertTrue(cache_valid(cache/name,cache_key(r)))
            # Successful cache reuse does not touch raw maps again.
            binary=next((self.root/e['path']).glob('*.bin'));binary.unlink()
            self.assertEqual(analyze_run(self.root,e,r,cache/name),summary)
        pairs,twins=matched_pairs(entries,results,cache)
        self.assertTrue(pairs);self.assertTrue(twins)
        self.assertTrue(all(p['inplane_error_rms']==0 for p in pairs))
        self.assertTrue(all(abs(p['relative_amplitude_difference'])<1e-15 for p in twins))
    def test_modified_uncached_array_rejected(self):
        e,r=self.fixture('base',0.);binary=next((self.root/e['path']).glob('*_Uc.bin'));binary.write_bytes(b'changed')
        with self.assertRaises(ValueError):analyze_run(self.root,e,r,self.root/'cache')
    def test_pending_page_has_no_fake_data(self):
        output=self.root/'public';output.mkdir()
        summary={'counts':{'waiting':1},'runs':[{'id':'pending','status':'waiting'}],'figures':[],'source_version':'test','generated_at':'test'}
        render(summary,output);page=(output/'index.html').read_text()
        self.assertIn('Results will appear',page);self.assertNotIn('{{',page)
        self.assertNotIn('runs/pending/',page)

if __name__=='__main__':unittest.main()
