import hashlib
import json
import os
import tempfile
import unittest

import numpy as np

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import CIT, CIT_Base


class TestCITDataFingerprint(unittest.TestCase):
    def setUp(self):
        self.data = np.tile(
            np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]), (250, 1)
        )

    def save_cache(self, data, path):
        test = CIT(data, 'fisherz', cache_path=path)
        result = test(0, 1)
        test.SAVE_CACHE_CYCLE_SECONDS = 0
        test.save_to_local_cache()
        return result

    def test_changed_undisplayed_values_reject_cached_results(self):
        changed = self.data.copy()
        changed[3:-3, 1] = changed[3:-3, 0]
        self.assertEqual(str(self.data), str(changed))
        self.assertLess(CIT(changed, 'fisherz')(0, 1), 0.05)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'cache.json')
            self.assertGreater(self.save_cache(self.data, path), 0.05)
            with self.assertRaisesRegex(ValueError, 'Data hash mismatch'):
                pc(changed, 0.05, 'fisherz', cache_path=path, show_progress=False)

    def test_rounded_values_have_different_fingerprints(self):
        data = np.array([[1., 2.], [3., 4.]])
        changed = data.copy()
        changed[0, 0] = np.nextafter(1., 2.)
        self.assertEqual(str(data), str(changed))
        self.assertNotEqual(CIT_Base(data).data_hash, CIT_Base(changed).data_hash)

    def test_print_options_do_not_change_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'cache.json')
            result = self.save_cache(self.data, path)
            with np.printoptions(threshold=np.inf, precision=2):
                self.assertEqual(CIT(self.data, 'fisherz', cache_path=path)(0, 1), result)

    def test_memory_layout_does_not_change_fingerprint(self):
        for data in (self.data, self.data[::-1], self.data[:, ::-1]):
            with self.subTest(strides=data.strides):
                self.assertEqual(CIT_Base(data).data_hash,
                                 CIT_Base(np.ascontiguousarray(data)).data_hash)
                self.assertEqual(CIT_Base(data).data_hash,
                                 CIT_Base(np.asfortranarray(data)).data_hash)

    def test_shape_and_dtype_are_part_of_fingerprint(self):
        data = np.arange(12, dtype=np.float64).reshape(6, 2)
        self.assertNotEqual(CIT_Base(data).data_hash,
                            CIT_Base(data.reshape(4, 3)).data_hash)
        self.assertNotEqual(CIT_Base(data).data_hash,
                            CIT_Base(data.view(np.int64)).data_hash)
        self.assertNotEqual(CIT_Base(data).data_hash,
                            CIT_Base(data.astype(np.float32)).data_hash)

    def test_unchanged_data_reuses_persisted_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'cache.json')
            result = self.save_cache(self.data, path)
            restored = CIT(self.data.copy(), 'fisherz', cache_path=path)
            self.assertIn('0;1', restored.pvalue_cache)
            self.assertEqual(restored(0, 1), result)

    def test_legacy_display_fingerprint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'cache.json')
            self.save_cache(self.data, path)
            with open(path) as stream:
                cache = json.load(stream)
            cache['data_hash'] = hashlib.md5(str(self.data).encode('utf-8')).hexdigest()
            with open(path, 'w') as stream:
                json.dump(cache, stream)
            with self.assertRaisesRegex(ValueError, 'Regenerate the cache'):
                CIT(self.data, 'fisherz', cache_path=path)


if __name__ == '__main__':
    unittest.main()
