import unittest
import os
from dptb.utils.orbital_parser import parse_orbital_file
from unittest.mock import patch, mock_open

class TestOrbitalParser(unittest.TestCase):
    def test_parse_valid_file(self):
        content = """
Element                     Si
Number of Sorbital-->       2
Number of Porbital-->       2
Number of Dorbital-->       1
"""
        with patch("builtins.open", mock_open(read_data=content)):
            with patch("os.path.exists", return_value=True):
                basis = parse_orbital_file("dummy.orb")
                self.assertEqual(basis, "2s2p1d")

    def test_parse_zero_count(self):
        content = """
Number of Sorbital-->       1
Number of Porbital-->       0
"""
        with patch("builtins.open", mock_open(read_data=content)):
            with patch("os.path.exists", return_value=True):
                basis = parse_orbital_file("dummy.orb")
                self.assertEqual(basis, "1s")

    def test_parse_invalid_file(self):
        content = "Invalid content"
        with patch("builtins.open", mock_open(read_data=content)):
            with patch("os.path.exists", return_value=True):
                with self.assertRaisesRegex(ValueError, "No valid orbital counts found"):
                    parse_orbital_file("dummy.orb")

    def test_file_not_found(self):
        with patch("os.path.exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                parse_orbital_file("nonexistent.orb")

if __name__ == '__main__':
    unittest.main()
