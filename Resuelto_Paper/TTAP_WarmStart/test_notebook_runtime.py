"""Bootstrap checks for both A/B notebooks; no solver runs or network needed."""
import ast
from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


def notebook_paths():
    folder = Path(__file__).resolve().parent
    roots = [folder.parent,
             folder.parent / 'ttap_execution_20261028' / 'notebooks']
    for root in roots:
        paths = [root / f'20261028_Experimento_{e}_WarmStart.ipynb' for e in 'AB']
        if all(path.is_file() for path in paths):
            return paths
    raise FileNotFoundError('The A/B notebooks must be next to the TTAP_WarmStart folder.')


def load_bootstrap(path):
    notebook = json.loads(path.read_text(encoding='utf-8'))
    source = ''.join(next(c for c in notebook['cells'] if c['id'] == 'runtime')['source'])
    tree = ast.parse(source)
    prefix = []
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'RUNTIME' for t in node.targets)):
            break
        prefix.append(node)
    namespace = {}
    exec(compile(ast.Module(body=prefix, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class NotebookRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = notebook_paths()
        cls.a, cls.b = [load_bootstrap(path) for path in cls.paths]

    def fixture(self):
        content = {'frozen_generator_A.py': b'x = 1\n', 'study_runner.py': b'y = 2\n'}
        expected = {name: hashlib.sha256(data).hexdigest() for name, data in content.items()}
        requests = []

        def opener(url, timeout):
            self.assertEqual(timeout, 90)
            requests.append(url)
            return BytesIO(content[url.rsplit('/', 1)[1]])

        return content, expected, requests, opener

    def test_both_notebooks_use_same_verified_manifest(self):
        self.assertEqual(self.a['EXPECTED_SHA256'], self.b['EXPECTED_SHA256'])
        self.assertEqual(self.a['RUNTIME_REVISION'], self.b['RUNTIME_REVISION'])
        folder = Path(__file__).resolve().parent
        for name, expected in self.a['EXPECTED_SHA256'].items():
            with self.subTest(name=name):
                content = (folder / name).read_bytes().replace(b'\r\n', b'\n')
                self.assertEqual(hashlib.sha256(content).hexdigest(), expected)

    def test_empty_folder_downloads_then_reuses_cache_offline(self):
        content, expected, requests, opener = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.a['_ttap_prepare_runtime']('abc123', expected, root=directory, opener=opener)
            self.assertEqual(len(requests), len(expected))
            for name, data in content.items():
                self.assertEqual((runtime / name).read_bytes(), data)
            same = self.a['_ttap_prepare_runtime']('abc123', expected, root=directory,
                                                  opener=lambda *a, **k: self.fail('Network called'))
            self.assertEqual(runtime, same)

    def test_windows_checkout_is_copied_without_changing_user_files(self):
        content, expected, requests, opener = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / 'TTAP_WarmStart'
            local.mkdir()
            for name, data in content.items():
                (local / name).write_bytes(data.replace(b'\n', b'\r\n'))
            runtime = self.a['_ttap_prepare_runtime']('abc123', expected, root=directory, opener=opener)
            self.assertFalse(requests)
            for name, data in content.items():
                self.assertEqual((runtime / name).read_bytes(), data)
                self.assertEqual((local / name).read_bytes(), data.replace(b'\n', b'\r\n'))

    def test_mismatched_local_file_is_preserved_and_pinned_source_downloaded(self):
        content, expected, requests, opener = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / 'TTAP_WarmStart'
            local.mkdir()
            original = content['frozen_generator_A.py'] + b'\n'
            (local / 'frozen_generator_A.py').write_bytes(original)
            runtime = self.a['_ttap_prepare_runtime']('abc123', expected, root=directory, opener=opener)
            self.assertEqual((local / 'frozen_generator_A.py').read_bytes(), original)
            self.assertEqual((runtime / 'frozen_generator_A.py').read_bytes(), content['frozen_generator_A.py'])
            self.assertEqual(len(requests), len(expected))

    def test_corrupt_cache_is_repaired(self):
        content, expected, requests, opener = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            prepare = self.a['_ttap_prepare_runtime']
            runtime = prepare('abc123', expected, root=directory, opener=opener)
            requests.clear()
            (runtime / 'frozen_generator_A.py').write_bytes(b'wrong\n')
            prepare('abc123', expected, root=directory, opener=opener)
            self.assertEqual(len(requests), 1)
            self.assertEqual((runtime / 'frozen_generator_A.py').read_bytes(), content['frozen_generator_A.py'])

    def test_wrong_download_is_rejected(self):
        content, expected, requests, opener = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'descarga no coincide'):
                self.a['_ttap_prepare_runtime']('abc123', expected, root=directory,
                                              opener=lambda *a, **k: BytesIO(b'wrong\n'))
            self.assertFalse(list(Path(directory).rglob('*.py')))

    def test_partial_download_can_resume(self):
        content, expected, requests, opener = self.fixture()
        calls = 0

        def interrupted(url, timeout):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError('connection interrupted')
            return opener(url, timeout)

        with tempfile.TemporaryDirectory() as directory:
            prepare = self.a['_ttap_prepare_runtime']
            with self.assertRaisesRegex(RuntimeError, 'No se pudo descargar'):
                prepare('abc123', expected, root=directory, opener=interrupted)
            requests.clear()
            runtime = prepare('abc123', expected, root=directory, opener=opener)
            self.assertEqual(len(requests), 1)
            self.assertEqual(len(list(runtime.glob('*.py'))), 2)


if __name__ == '__main__':
    unittest.main()
