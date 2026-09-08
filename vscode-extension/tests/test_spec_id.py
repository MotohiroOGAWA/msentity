"""Exercise SpecID assignment through the viewer's JSON protocol."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / 'vscode-extension/python/backend.py'


class SpecIdProtocolTest(unittest.TestCase):
    def test_assignment_overwrite_dataset_scope_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.tsv'
            other = Path(directory) / 'other.tsv'
            output = Path(directory) / 'export.tsv'
            source.write_text('Name\tPeak\n' + ''.join(f'row{i}\t100,10\n' for i in range(12)))
            other.write_text('Name\tPeak\nother\t200,20\n')
            requests = [
                {'type': 'add-dataset', 'path': str(other)},
                {'type': 'assign-spec-id', 'prefix': 'SP'},
                {'type': 'page', 'page': 1},
                {'type': 'assign-spec-id', 'prefix': 'BAD'},
                {'type': 'page'},
                {'type': 'assign-spec-id', 'prefix': '試料_', 'overwrite': True},
                {'type': 'page', 'filters': [{'column': 'Name', 'operator': 'text_eq', 'value': 'row11'}]},
                {'type': 'page', 'dataset_id': str(other)},
                {'type': 'export', 'path': str(output), 'file_type': 'tsv'},
                {'type': 'assign-spec-id', 'prefix': '' , 'overwrite': True},
                {'type': 'page'},
                {'type': 'assign-spec-id', 'prefix': 123},
                {'type': 'assign-spec-id', 'dataset_id': 'missing'},
            ]
            env = dict(os.environ, PYTHONPATH=str(ROOT))
            result = subprocess.run(
                [sys.executable, str(BACKEND), str(source), '--page-size', '5'],
                input=''.join(json.dumps(request) + '\n' for request in requests),
                text=True, capture_output=True, env=env, timeout=30, check=True,
            )
            messages = [json.loads(line.removeprefix('MSENTITY_JSON:'))
                        for line in result.stdout.splitlines() if line.startswith('MSENTITY_JSON:')]
            self.assertFalse([m for m in messages if m['type'] == 'error'], result.stdout)
            self.assertEqual(len([m for m in messages if m['type'] == 'spec-id-complete']), 3)
            self.assertEqual(len([m for m in messages if m['type'] == 'spec-id-error']), 3)
            pages = [m['value'] for m in messages if m['type'] == 'dataset-page']
            self.assertEqual(pages[1]['rows'][0]['SpecID'], 'SP06')
            self.assertEqual(pages[2]['rows'][0]['SpecID'], 'SP01')
            self.assertEqual(pages[3]['rows'][0]['SpecID'], '試料_12')
            self.assertNotIn('SpecID', pages[4]['columns'])
            self.assertEqual(pages[5]['rows'][0]['SpecID'], '01')
            self.assertIn('SpecID', output.read_text())
            self.assertIn('試料_12', output.read_text())
            self.assertNotIn('SpecID', source.read_text())


if __name__ == '__main__':
    unittest.main()
