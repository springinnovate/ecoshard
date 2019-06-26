"""Ecoshard test suite."""
import os
import tempfile
import shutil
import unittest

import ecoshard


class EcoShardTests(unittest.TestCase):
    """Tests for the PyGeoprocesing 1.0 refactor."""

    def setUp(self):
        """Create a temporary workspace that's deleted later."""
        self.workspace_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up remaining files."""
        shutil.rmtree(self.workspace_dir)

    def test_hash_file(self):
        """Test ecoshard.hash_file."""
        working_dir = self.workspace_dir
        try:
            os.makedirs(working_dir)
        except OSError:
            pass
        base_path = os.path.join(working_dir, 'test_file.txt')
        target_token_path = '%s.COMPLETE' % base_path

        with open(base_path, 'w') as base_file:
            base_file.write('test')

        # make a different target dir so it gets copied
        target_dir = os.path.join(working_dir, 'output')
        ecoshard.hash_file(
            base_path, target_token_path=target_token_path,
            target_dir=target_dir, rename=False,
            hash_algorithm='md5', force=False)

        expected_file_path = os.path.join(
            target_dir, 'test_file_md5_098f6bcd4621d373cade4e832627b4f6.txt')
        self.assertTrue(os.path.exists(expected_file_path))
        self.assertTrue(os.path.exists(target_token_path))

    def test_hash_file_rename(self):
        """Test ecoshard.hash_file with a rename."""
        working_dir = self.workspace_dir
        try:
            os.makedirs(working_dir)
        except OSError:
            pass
        base_path = os.path.join(working_dir, 'test_file.txt')
        target_token_path = '%s.COMPLETE' % base_path

        with open(base_path, 'w') as base_file:
            base_file.write('test')

        ecoshard.hash_file(
            base_path, target_token_path=target_token_path,
            target_dir=None, rename=True,
            hash_algorithm='md5', force=False)

        expected_file_path = os.path.join(
            working_dir, 'test_file_md5_098f6bcd4621d373cade4e832627b4f6.txt')
        self.assertTrue(os.path.exists(expected_file_path))
        # indicates the file has been renamed
        self.assertFalse(os.path.exists(base_path))

    def test_force(self):
        """Test ecoshard.hash_file with a force rename."""
        working_dir = self.workspace_dir
        try:
            os.makedirs(working_dir)
        except OSError:
            pass
        base_path = os.path.join(working_dir, 'test_file_sha_fffffffffff.txt')
        target_token_path = '%s.COMPLETE' % base_path

        with open(base_path, 'w') as base_file:
            base_file.write('test')

        ecoshard.hash_file(
            base_path, target_token_path=target_token_path,
            target_dir=None, rename=True,
            hash_algorithm='md5', force=True)

        expected_file_path = os.path.join(
            working_dir, 'test_file_md5_098f6bcd4621d373cade4e832627b4f6.txt')
        self.assertTrue(os.path.exists(expected_file_path))
        # indicates the file has been renamed
        self.assertFalse(os.path.exists(base_path))
