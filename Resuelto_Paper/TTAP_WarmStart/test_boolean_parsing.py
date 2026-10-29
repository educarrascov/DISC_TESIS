"""Missing-flag regression checks for pandas object and nullable string arrays."""
import unittest

import numpy as np
import pandas as pd

from figure5_and_analysis import booleans


class BooleanParsingTests(unittest.TestCase):
    def test_nullable_strings_preserve_missing_flags(self):
        values = pd.Series(['True', ' false ', pd.NA, None], dtype='string')
        result = booleans(values)
        self.assertEqual(result.tolist(), [True, False, False, False])
        self.assertEqual(result.dtype, np.dtype(bool))

    def test_object_flags_and_missing_values(self):
        values = pd.Series([True, False, np.nan, None, pd.NA], dtype=object)
        self.assertEqual(booleans(values).tolist(), [True, False, False, False, False])

    def test_numeric_flags_after_csv_type_inference(self):
        self.assertEqual(booleans(pd.Series([1., 0., np.nan])).tolist(), [True, False, False])

    def test_bool_dtype(self):
        self.assertEqual(booleans(pd.Series([True, False])).tolist(), [True, False])

    def test_unknown_flags_still_raise(self):
        for invalid in ('yes', '2', 'unknown'):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'Unexpected boolean'):
                booleans(pd.Series([invalid, pd.NA], dtype='string'))


if __name__ == '__main__':
    unittest.main()
