"""Compressed CSVs retain the existing public-gallery content restrictions."""
import contextlib
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

checker_path = Path(__file__).with_name('check_gallery.py')
if not checker_path.is_file():
    checker_path = Path(__file__).resolve().parents[1] / 'scripts/check_gallery.py'
spec = importlib.util.spec_from_file_location('scoped_check_gallery', checker_path)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class CompressedCSVTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'data').mkdir()
        (self.root / 'data/campaign.json').write_text(json.dumps({
            'schema_version': 1, 'campaign': {'total_pairs': 7},
            'downloads': [{'href': 'budget.csv.gz'}]}), encoding='utf-8')

    def write_csv(self, content, name='budget.csv.gz'):
        (self.root / name).write_bytes(gzip.compress(content, mtime=0))

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return checker.check(self.root)

    def test_valid_compressed_csv_and_download_reference(self):
        self.write_csv('time,mean_vorticity\n0,0.005\n1,0.004\n'.encode('utf-8'))
        self.assertEqual(self.check()['schema_version'], 1)

    def test_credentials_rejected_inside_compressed_csv(self):
        for content in (b'value\ngithub_pat_example_test_token\n',
                        b'value\n-----BEGIN OPENSSH PRIVATE KEY-----\n',
                        b'value\nprivkey.key\n'):
            with self.subTest(content=content):
                self.write_csv(content)
                with self.assertRaisesRegex(ValueError, 'Private path or credential'):
                    self.check()

    def test_private_path_rejected_inside_compressed_csv(self):
        self.write_csv(b'name,path\nrun,/work/private_user/run\n')
        with self.assertRaisesRegex(ValueError, 'Private path or credential'):
            self.check()

    def test_non_utf8_compressed_csv_rejected(self):
        self.write_csv(b'name,value\nrow,\xff\n')
        with self.assertRaises(UnicodeDecodeError):
            self.check()

    def test_expanded_content_size_is_bounded(self):
        self.assertEqual(checker.MAX_FILE_BYTES, 90 * 1024**2)
        self.write_csv(b'a,' * 5000)
        # Exercise the same boundary without allocating a 90 MiB fixture.
        with patch.object(checker, 'MAX_FILE_BYTES', 4096):
            with self.assertRaisesRegex(ValueError, 'Decompressed gallery CSV exceeds'):
                self.check()

    def test_non_csv_gzip_remains_forbidden(self):
        self.write_csv(b'time,value\n0,1\n')
        for name in ('notes.txt.gz', 'archive.tar.gz', 'budget.gz', 'budget.csv.gz.gz'):
            with self.subTest(name=name):
                self.write_csv(b'time,value\n0,1\n', name)
                with self.assertRaisesRegex(ValueError, 'Non-public artifact'):
                    self.check()
                (self.root / name).unlink()

    def test_invalid_gzip_rejected(self):
        (self.root / 'budget.csv.gz').write_bytes(b'not gzip')
        with self.assertRaises(gzip.BadGzipFile):
            self.check()

    def test_plain_csv_still_scanned(self):
        self.write_csv(b'time,value\n0,1\n')
        (self.root / 'plain.csv').write_text('name,path\nrun,/home/private_user/run\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Private path or credential'):
            self.check()


if __name__ == '__main__':
    unittest.main()
