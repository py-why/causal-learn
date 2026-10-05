import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from causallearn.utils.cit import CIT


class TestCITCache(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache_path = str(Path(self.directory.name) / 'pvalues.json')
        x = np.linspace(-2, 2, 60)
        self.data = np.column_stack((x, x ** 2))

    def save_result(self, method, **parameters):
        test = CIT(self.data, method, cache_path=self.cache_path, **parameters)
        result = test(0, 1)
        # Persist the actual result without waiting for the normal autosave interval.
        test.SAVE_CACHE_CYCLE_SECONDS = 0
        test.save_to_local_cache()
        return result

    def test_rejects_different_method(self):
        original = self.save_result('fisherz')
        fresh = CIT(self.data, 'kci')(0, 1)
        # This fixture makes stale reuse change the independence decision.
        self.assertGreater(original, 0.05)
        self.assertLess(fresh, 0.05)
        with self.assertRaisesRegex(ValueError, 'method name mismatch'):
            CIT(self.data, 'kci', cache_path=self.cache_path)

    def test_rejects_different_parameters(self):
        original = self.save_result('kci', kernelX='Linear', kernelY='Linear')
        fresh = CIT(self.data, 'kci')(0, 1)
        self.assertGreater(original, 0.05)
        self.assertLess(fresh, 0.05)
        with self.assertRaisesRegex(ValueError, 'parameters mismatch'):
            CIT(self.data, 'kci', cache_path=self.cache_path)

    def test_reuses_matching_cache_without_recomputation(self):
        expected = self.save_result('fisherz')
        reloaded = CIT(self.data, 'fisherz', cache_path=self.cache_path)
        with patch('numpy.linalg.inv', side_effect=AssertionError('recomputed cached test')):
            self.assertEqual(reloaded(0, 1), expected)
            self.assertEqual(reloaded(1, 0), expected)

    def test_reuses_matching_kernel_parameters(self):
        expected = self.save_result('kci', kernelX='Linear', kernelY='Linear')
        reloaded = CIT(self.data, 'kci', cache_path=self.cache_path,
                       kernelY='Linear', kernelX='Linear')
        with patch.object(reloaded.kci_ui, 'compute_pvalue', side_effect=AssertionError('recomputed cached test')):
            self.assertEqual(reloaded(0, 1), expected)

    def test_rejected_load_preserves_checkpoint(self):
        self.save_result('fisherz')
        original = Path(self.cache_path).read_bytes()
        with self.assertRaisesRegex(ValueError, 'method name mismatch'):
            CIT(self.data, 'kci', cache_path=self.cache_path)
        self.assertEqual(Path(self.cache_path).read_bytes(), original)
        self.assertEqual(json.loads(original)['method_name'], 'fisherz')

    def test_new_cache_records_method_and_parameters(self):
        test = CIT(self.data, 'fisherz', cache_path=self.cache_path)
        self.assertEqual(test.pvalue_cache['method_name'], 'fisherz')
        self.assertIn('parameters_hash', test.pvalue_cache)
        self.assertEqual(test(0, 1), CIT(self.data, 'fisherz')(0, 1))


if __name__ == '__main__':
    unittest.main()
