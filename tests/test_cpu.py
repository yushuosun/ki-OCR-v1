"""Synthetic CPU integration controls; never claim model/GPU acceptance."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assets import load, sha, write
import inference


class PipelineControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / 'images'
        self.images.mkdir()
        # Image bytes are explicitly synthetic; pipeline controls do not decode.
        (self.images / 'page.one.png').write_bytes(b'CPU_SYNTHETIC_IMAGE_IDENTITY')
        self.out = self.root / 'out'
        self.out.mkdir()
        self.harness = self.out / 'harness'
        shutil.copytree(ROOT / 'core', self.harness, ignore=shutil.ignore_patterns('__pycache__'))
        write(self.harness / 'NATIVE_DETERMINISTIC_CODE.json',
              {'source': 'CPU_SYNTHETIC_CONVERSION_FIXTURE', 'files': {
                  'text_fix_dollar.py': 'def normalize_inline_math(text):\n    return text\n'}})
        row = {'id': 'page.one.png', 'input_index': 0,
               'image_sha256': sha(self.images / 'page.one.png'), 'size': [100, 100]}
        write(self.out / 'INPUTS.json', {'reference_access': False, 'bound_rows': [row]})
        self.contract = {'output': str(self.out), 'harness': str(self.harness),
                         'inputs_path': str(self.out / 'INPUTS.json'),
                         'inputs_sha256': sha(self.out / 'INPUTS.json'),
                         'bindings': {'image_root': str(self.images)},
                         'stage_pins': load(ROOT / 'DEPENDENCIES.json')['stage_pins']}
        self.cp = self.out / 'CONTRACT.json'
        write(self.cp, self.contract)
        for name in ('entry', 'current_r3', 'native', 'cascade', 'paddle_contract', 'tele_support'):
            sys.modules.pop(name, None)
        self.calls = []

    def tearDown(self):
        for name in ('entry', 'current_r3', 'native', 'cascade', 'paddle_contract', 'tele_support'):
            sys.modules.pop(name, None)
        sys.path[:] = [p for p in sys.path if p != str(self.harness)]
        self.temp.cleanup()

    def primary(self, markdown, payload=None):
        base = self.out / 'BATCHES/0000/B0'
        (base / 'markdown').mkdir(parents=True)
        (base / 'middle').mkdir()
        (base / 'markdown/page.one.md').write_text(markdown, encoding='utf-8', newline='')
        blocks = [] if payload is None else [{'lines': [{'spans': [
            {'type': 'text', 'content': payload, 'bbox': [10, 10, 90, 90]}]}]}]
        write(base / 'middle/page.one.json', {'pdf_info': [{'page_size': [100, 100], 'para_blocks': blocks}]})

    def donor(self, dimensions=(100, 100), label='text', content='backup'):
        directory = self.out / 'PADDLE/page.one'
        directory.mkdir(parents=True)
        write(directory / 'native.json', {'res': {'width': dimensions[0], 'height': dimensions[1],
              'parsing_res_list': [{'block_label': label, 'block_content': content,
                                   'block_bbox': [10, 10, 90, 90]}]}})
        (directory / 'prediction.md').write_text('whole page backup', encoding='utf-8')

    def event(self, raw, reason, contract_sha=None, page_id='page.one'):
        row = load(self.out / 'INPUTS.json')['bound_rows'][0]
        record = {'stage': 'NATIVE', 'call': 1, 'batch_index': 0, 'utc': '2026-10-06T00:00:00Z',
                  'raw_text': raw, 'finish_reason': reason, 'run_contract_sha256': contract_sha or sha(self.cp),
                  'binding': {'page_id': page_id, 'input_image_sha256': row['image_sha256'],
                              'kind': 'text', 'bbox': [10, 10, 90, 90]},
                  'source': {'script_sha256': self.contract['stage_pins']['run_tele_img.py']}}
        (self.out / 'PRIVATE_RAW_REQUESTS.jsonl').write_text(json.dumps(record) + '\n', encoding='utf-8')

    def run_control(self, callback):
        def driver(phase):
            self.calls.append(phase)
            callback(phase)
            return {'exit': 0, 'synthetic': True}
        result = inference.pipeline(self.contract, self.cp, driver=driver)
        self.assertEqual(result['status'], 'CPU_SYNTHETIC_ONLY')
        self.assertEqual(load(self.out / 'COST.json')['GPU_card_seconds'], 0)
        return result

    def test_healthy_output_is_byte_preserved_without_paddle(self):
        self.primary('prefix\nhealthy\nsuffix\n', 'healthy')
        result = self.run_control(lambda p: None)
        self.assertEqual(self.calls, ['tele'])
        self.assertEqual((self.out / 'markdown/page.one.md').read_bytes(), b'prefix\nhealthy\nsuffix\n')
        self.assertEqual(result['returned_pages'], 1)

    def test_directory_fallback_preserves_other_page_and_two_outputs(self):
        self.primary(' \n')
        second = self.images / 'page.two.png'
        second.write_bytes(b'CPU_SYNTHETIC_SECOND_IMAGE')
        data = load(self.out / 'INPUTS.json')
        data['bound_rows'].append({'id': second.name, 'input_index': 1,
                                  'image_sha256': sha(second), 'size': [100, 100]})
        write(self.out / 'INPUTS.json', data)
        self.contract['inputs_sha256'] = sha(self.out / 'INPUTS.json')
        write(self.cp, self.contract)
        base = self.out / 'BATCHES/0000/B0'
        (base / 'markdown/page.two.md').write_bytes(b'healthy second\n')
        write(base / 'middle/page.two.json', {'pdf_info': [{'page_size': [100, 100], 'para_blocks': []}]})
        result = self.run_control(lambda p: self.donor() if p == 'paddle' else None)
        self.assertEqual(result['pages'], 2)
        self.assertEqual(result['returned_pages'], 2)
        self.assertEqual(result['natural_fallback_pages'], 1)
        self.assertEqual({p.name for p in (self.out / 'markdown').iterdir()}, {'page.one.md', 'page.two.md'})
        self.assertEqual((self.out / 'markdown/page.one.md').read_text(), 'whole page backup')
        self.assertEqual((self.out / 'markdown/page.two.md').read_bytes(), b'healthy second\n')
        self.assertTrue(all(t['in_denominator'] for t in result['terminals']))

    def test_empty_page_triggers_whole_page_current_donor(self):
        self.primary(' \n')
        result = self.run_control(lambda p: self.donor() if p == 'paddle' else None)
        self.assertEqual(self.calls, ['tele', 'paddle'])
        self.assertEqual(result['natural_fallback_pages'], 1)
        self.assertEqual((self.out / 'markdown/page.one.md').read_text(), 'whole page backup')

    def probe_fixture(self, markdown):
        directory = self.out / 'PADDLE_PROBE/PADDLE/page.one'
        directory.mkdir(parents=True)
        write(directory / 'native.json', {'res': {'width': 100, 'height': 100,
              'parsing_res_list': [{'block_label': 'text', 'block_content': 'backup',
                                   'block_bbox': [10, 10, 90, 90]}]}})
        (directory / 'prediction.md').write_text(markdown, encoding='utf-8')

    def test_failed_probe_child_cannot_pass_with_partial_donor_files(self):
        self.primary('healthy', 'healthy')
        def driver(phase):
            if phase == 'paddle_probe':
                self.probe_fixture('backup')
                return {'exit': 2, 'synthetic': True}
            return {'exit': 0, 'synthetic': True}
        result = inference.pipeline(self.contract, self.cp, driver=driver, probe=True)
        self.assertEqual(result['paddle_probe']['status'], 'PROBE_FAILED')
        self.assertFalse(result['overall_success'])
        self.assertEqual((self.out / 'markdown/page.one.md').read_text(), 'healthy')

    def test_empty_probe_merge_is_failure_preserving_primary(self):
        self.primary('healthy', 'healthy')
        def driver(phase):
            if phase == 'paddle_probe':
                self.probe_fixture(' ')
            return {'exit': 0, 'synthetic': True}
        result = inference.pipeline(self.contract, self.cp, driver=driver, probe=True)
        self.assertEqual(result['paddle_probe']['status'], 'PROBE_FAILED')
        self.assertFalse(result['overall_success'])
        self.assertEqual((self.out / 'markdown/page.one.md').read_text(), 'healthy')

    def test_effective_probe_passes_without_changing_healthy_prediction(self):
        self.primary('healthy', 'healthy')
        def driver(phase):
            if phase == 'paddle_probe':
                self.probe_fixture('backup')
            return {'exit': 0, 'synthetic': True}
        result = inference.pipeline(self.contract, self.cp, driver=driver, probe=True)
        self.assertEqual(result['paddle_probe']['status'], 'CONTROLLED_DONOR_PROBE')
        self.assertTrue(result['paddle_probe']['synthetic_empty_primary_merge_pass'])
        self.assertTrue(result['overall_success'])
        self.assertEqual(result['natural_fallback_pages'], 0)
        self.assertEqual((self.out / 'markdown/page.one.md').read_text(), 'healthy')

    def test_failed_slot_uses_exact_anchor_preserving_other_text(self):
        self.primary('prefix\ntruncated\nsuffix', 'truncated')
        self.event('truncated', 'length')
        result = self.run_control(lambda p: self.donor() if p == 'paddle' else None)
        self.assertEqual(self.calls, ['tele', 'paddle'])
        self.assertEqual((self.out / 'markdown/page.one.md').read_text(), 'prefix\nbackup\nsuffix')
        self.assertEqual(result['tele_length_stops'], 1)

    def test_superseded_failure_does_not_trigger_paddle(self):
        self.primary('healthy', 'healthy')
        self.event('truncated', 'length')
        result = self.run_control(lambda p: None)
        self.assertEqual(self.calls, ['tele'])
        self.assertEqual(result['natural_fallback_pages'], 0)

    def test_stale_contract_raw_cannot_trigger_paddle(self):
        self.primary('truncated', 'truncated')
        self.event('truncated', 'length', contract_sha='0' * 64)
        result = self.run_control(lambda p: None)
        self.assertEqual(result['natural_fallback_pages'], 0)

    def test_other_page_raw_cannot_trigger_paddle(self):
        self.primary('truncated', 'truncated')
        self.event('truncated', 'length', page_id='another.page')
        result = self.run_control(lambda p: None)
        self.assertEqual(result['natural_fallback_pages'], 0)

    def test_bad_donor_geometry_preserves_primary_and_denominator(self):
        self.primary('truncated', 'truncated')
        self.event('truncated', 'length')
        result = self.run_control(lambda p: self.donor((90, 100)) if p == 'paddle' else None)
        self.assertEqual((self.out / 'markdown/page.one.md').read_text(), 'truncated')
        self.assertEqual(result['failed_retained'], 1)
        self.assertTrue(result['terminals'][0]['in_denominator'])

    def test_absent_donor_is_failure_not_silent_success(self):
        self.primary('truncated', 'truncated')
        self.event('truncated', 'length')
        result = self.run_control(lambda p: None)
        self.assertEqual(result['terminals'][0]['state'], 'FALLBACK_FAILED_RETAINED')
        self.assertEqual(result['failed_retained'], 1)

    def test_missing_primary_remains_empty_in_denominator(self):
        result = self.run_control(lambda p: None)
        self.assertEqual(result['pages'], 1)
        self.assertEqual(result['failed_retained'], 1)
        self.assertEqual((self.out / 'markdown/page.one.md').read_bytes(), b'')
        self.assertTrue(result['terminals'][0]['in_denominator'])

    def test_changed_input_fails_before_driver(self):
        (self.images / 'page.one.png').write_bytes(b'CHANGED')
        with self.assertRaisesRegex(ValueError, 'Frozen input image changed'):
            self.run_control(lambda p: None)
        self.assertEqual(self.calls, [])

    def test_native_postprocessor_bundle_change_is_rejected(self):
        native_bundle = self.harness / 'NATIVE_DETERMINISTIC_CODE.json'
        self.contract['native_bundle_sha256'] = sha(native_bundle)
        native_bundle.write_bytes(b'CHANGED_SYNTHETIC_POSTPROCESSOR')
        with self.assertRaisesRegex(ValueError, 'native postprocessor bundle changed'):
            inference.verify_code(self.contract)

    def test_answer_bearing_manifest_is_rejected(self):
        data = load(self.out / 'INPUTS.json')
        data['GT'] = 'answer'
        write(self.out / 'INPUTS.json', data)
        self.contract['inputs_sha256'] = sha(self.out / 'INPUTS.json')
        with self.assertRaisesRegex(ValueError, 'answer-bearing'):
            inference.checked_rows(self.contract)


class InputControls(unittest.TestCase):
    def test_same_paddle_version_with_wrong_CUDA_build_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'METADATA differs'):
            inference.check_paddle_wheel_requirements(['nvidia-cudnn-cu13==9.13.0.50'])

    def test_actual_pinned_CUDA_wheel_requirements_are_accepted(self):
        required = load(ROOT/'PADDLE_WHEEL_PIN.json')['CUDA_dependency_pins']
        self.assertEqual(inference.check_paddle_wheel_requirements(required), 15)

    def test_inherited_model_caches_are_rebound_to_this_run(self):
        from unittest.mock import patch
        import os
        from assets import clean_env
        with patch.dict(os.environ, {'HF_HOME':'SYNTHETIC_OLD_CACHE', 'PADDLE_PDX_CACHE_HOME':'SYNTHETIC_OLD_CACHE', 'VLLM_CACHE_ROOT':'SYNTHETIC_OLD_CACHE'}):
            env = clean_env(Path('synthetic_new_run'))
        self.assertEqual(env['HF_HOME'], str(Path('synthetic_new_run/hf')))
        self.assertEqual(env['PADDLE_PDX_CACHE_HOME'], str(Path('synthetic_new_run/paddlex')))
        self.assertEqual(env['VLLM_CACHE_ROOT'], str(Path('synthetic_new_run/vllm')))

    def test_empty_paddle_venv_cannot_pass_dependency_validation(self):
        with self.assertRaisesRegex(ValueError, 'dependency missing/changed'):
            inference.check_locked_versions({'pip':'23.0.1'}, {'paddlepaddle-gpu':'3.3.1'}, 'paddle')

    def test_changed_dependency_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'dependency missing/changed'):
            inference.check_locked_versions({'vllm':'0.12.0'}, {'vllm':'0.11.0'}, 'tele')

    def test_distribution_name_normalization_preserves_exact_version_check(self):
        self.assertEqual(inference.check_locked_versions({'PaddlePaddle_gpu':'3.3.1'}, {'paddlepaddle-gpu':'3.3.1'}, 'paddle'), 1)

    def test_single_page_and_sorted_directory(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('b.page.png', 'a.page.jpg'):
                Image.new('RGB', (24, 32), 'white').save(root / name)
            _, rows = inference.scan_inputs(root)
            self.assertEqual([r['id'] for r in rows], ['a.page.jpg', 'b.page.png'])
            _, single = inference.scan_inputs(root / 'b.page.png')
            self.assertEqual(single[0]['size'], [24, 32])

    def test_duplicate_markdown_stems_rejected(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('page.png', 'page.jpg'):
                Image.new('RGB', (24, 32), 'white').save(root / name)
            with self.assertRaisesRegex(ValueError, 'Duplicate Markdown stems'):
                inference.scan_inputs(root)

    def test_truncated_png_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'broken.png'
            path.write_bytes(b'\x89PNG\r\n\x1a\n')
            with self.assertRaises(Exception):
                inference.scan_inputs(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
