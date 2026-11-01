import sys,json,tomllib
from pathlib import Path
import tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from guard import inventory,version_for,digest,write_json
from inputs import approved_input
from test_onset_operations import input_fixture,toml_text


def refined_fixture(root):
    path,case,_=input_fixture(root)
    tool=root/'run/refine_native_snapshot.py';tool.parent.mkdir();tool.write_text('# pinned native transformation\n')
    companion=root/'translated/translated600.toml'
    metadata=tomllib.loads(companion.read_text())
    metadata['translation']='native-parent-refinement-v1'
    metadata['source_hashes']={'refine_native_snapshot.py':digest(tool)}
    metadata['derived_from']={'method':'periodic-spectral-even-odd-native-projection-v1','source_snapshot_sha256':'1'*64,'target_grid':[2,2,2]}
    companion.write_text(toml_text(metadata))
    approval_path=root/'source_provenance.toml';approval=tomllib.loads(approval_path.read_text());record=approval['sources']['parent']
    record.update(metadata_translation=metadata['translation'],derived_from=metadata['derived_from'],source_hashes=metadata['source_hashes'],snapshot_metadata_sha256=digest(companion))
    approval_path.write_text(toml_text(approval))
    files=inventory(root);source={'files':files,'version':version_for(files)};write_json(root/'source-manifest.json',source)
    return source


class RefinementOperationTests(unittest.TestCase):
    def test_refined_import_retains_pinned_transformation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=refined_fixture(root)
            approved_input(root,'parent',source)
            (root/'run/refine_native_snapshot.py').write_text('# changed numerical transform\n')
            with self.assertRaises(RuntimeError):approved_input(root,'parent',source)

    def test_refined_ancestry_cannot_be_swapped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=refined_fixture(root)
            approval_path=root/'source_provenance.toml';approval=tomllib.loads(approval_path.read_text())
            approval['sources']['parent']['derived_from']['source_snapshot_sha256']='2'*64
            approval_path.write_text(toml_text(approval))
            files=inventory(root);source={'files':files,'version':version_for(files)}
            with self.assertRaises(RuntimeError):approved_input(root,'parent',source)

if __name__=='__main__':unittest.main()
